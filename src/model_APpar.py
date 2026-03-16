import numpy as np
from scipy.interpolate import interp1d
#import bicker.emulator as BICKER # REMOVE THIS IF USING FOLPS
import baccoemu # REMOVE THIS IF NOT USING FOLPS

import os, sys
os.environ['FOLPS_BACKEND'] = 'numpy'  #'numpy' or 'jax'
sys.path.append('/cosma/home/dp322/dc-guan2/folps/folpsD/') # NEED TO MAKE THIS RIGHT!
import folps as FOLPS

from time import time
import re

'''
Important caveats:
* In the priors, always use the parameter ln10^{10}A_s. However, FOLPS samples over A_s, so there is an internal conversion!

NOTE:
* In FOLPSCalculator, remove the line
  > B110, B220, B022, B112 = None, None, None, None
  when computing these multipoles with folps.
'''

class FOLPSCalculator:

    def __init__(self, multipoles, mean_density, redshift,
                 model='EFT', damping=None, use_TNS_model=False,
                 AP=True, cosmo_fid=None, reparametrize=False):
        '''
            Damping: either "None" or " 'lor' "
            cosmo_fis: for AP;
        '''

        self.multipoles   = multipoles
        self.mean_density = mean_density
        self.zcen         = redshift
        self.expfactor    = 1.0/(1.0+redshift) # for baccoemu
        self.AP = AP
        self.cosmo_fid    = cosmo_fid or {'omega_b'  : 0.02237,
                                          'omega_cdm': 0.12,
                                          'omega_nu ': 0.00064420,
                                          'h'        : 0.6736,
                                          'ns'       : 0.9649,
                                          'As'       : 2.0830e-9}
        self.reparametrize = reparametrize

        ######################
        # Initialise linear power spectrum (bacco) emulator
        print('Initialising baccoemu for the linear power spectrum...')
        time_i = time()
        self._initialise_linear_pk_baccoemu()
        print('Total time:', (time()-time_i)/60)

        self.model = model
        self.damping = damping
        self.use_TNS_model = use_TNS_model

        if self.model == 'TNS':
            self.use_TNS_model=True
            if self.damping == None:
                self.damping = 'lor'
        if self.model == 'EFT':
            self.damping = None

        self._initialise_folps_matrices()
        self.folps_pk = FOLPS.RSDMultipolesPowerSpectrumCalculator(model=self.model)
        self.folps_bk = FOLPS.BispectrumCalculator(model=self.model)

    '''
        Helper functions
    '''

    ########################################################
    # BACCOEMU FOR LINEAR POWER SPECTRUM
    def _initialise_linear_pk_baccoemu(self):
        self.emulator = baccoemu.Matter_powerspectrum(verbose=False)
        self.kemul_pk = np.logspace(-4, np.log10(3), num=1000)

    def _get_linear_pk(self, pars):
        #data_path = '/Users/austerlitz/folps/folpsD/compare/pk_linear_simtocmass.txt'
        #self.k_arr, self.pk_arr = np.loadtxt(data_path, unpack=True)
        #return self.k_arr, self.pk_arr

        bacco_cosmo_pars = {
                    'omega_cold'    : (pars['omega_cdm'] + pars['omega_b']) / pars['h']**2,
                    'omega_baryon'  : pars['omega_b']/pars['h']**2,
                    'hubble'        : pars['h'],
                    'neutrino_mass' : pars.get('m_nu', 0.06),
                    'ns'            : pars.get('n_s', 0.9649),
                    'A_s'           : np.exp(pars['ln10^{10}A_s']) / 1e10,
                    'w0'            : -1.0,
                    'wa'            :  0.0,
                    'expfactor'     :  self.expfactor
                }
        bacco_cosmo_fid = {
                    'omega_cold'    : (self.cosmo_fid['omega_cdm'] + self.cosmo_fid['omega_b']) / self.cosmo_fid['h']**2,
                    'omega_baryon'  : self.cosmo_fid['omega_b']/self.cosmo_fid['h']**2,
                    'hubble'        : self.cosmo_fid['h'],
                    'neutrino_mass' : 0.06,
                    'ns'            : self.cosmo_fid['ns'],
                    'A_s'           : self.cosmo_fid['As'],
                    'w0'            : -1.0,
                    'wa'            :  0.0,
                    'expfactor'     :  self.expfactor
                }

        self.kemul_pk, self.pk_lin = self.emulator.get_linear_pk(k=self.kemul_pk, cold=True, **bacco_cosmo_pars)
        self.kemul_pk, self.pk_nw  = self.emulator.get_no_wiggles_pk(k=self.kemul_pk,cold=True,**bacco_cosmo_pars)
        #self.sigma8 = self.emulator.get_sigma8(cold=True, **bacco_cosmo_pars)
        self.sigma8ref = self.emulator.get_sigma8(cold=True, **bacco_cosmo_fid)

        self.output_dict = {'kemul_pk': self.kemul_pk,
                            'pk_lin': self.pk_lin,
                            'pk_nw': self.pk_nw,
                            'sigma8ref': self.sigma8ref }

        return self.output_dict

    ########################################################
    # FOLPS RELATED FUNCTIONS
    #
    # [1] Cosmology independent M matrices
    #
    def _initialise_folps_matrices(self):
        matrix = FOLPS.MatrixCalculator(
            A_full=True,
            use_TNS_model=self.use_TNS_model
        )
        self.mmatrices = matrix.get_mmatrices()
    #
    # [2] Everything folps requires for P(k) and B(k)
    #
    def _compute_folps_quantities(self, pars):

        # Build cosmology dictionary
        omega_b   = pars['omega_b']
        omega_cdm = pars['omega_cdm']
        h         = pars['h']
        m_nu      = pars.get('m_nu', 0.0)
        omega_nu  = 0.06/93.14 if pars.get('omega_nu', 0.0) == 0.0 else pars['omega_nu']

        # Compute Omega_m from sampled cosmology
        Omega_m = (omega_b+omega_cdm+omega_nu)/h**2
        f_nu    = omega_nu/(omega_cdm+omega_b+omega_nu)

        folps_cosmo = {
            **pars,
            'z': self.zcen,
            'Omega_m': Omega_m,
            'fnu': f_nu
        }

        # Alcock-Paczynski effect
        if self.AP:
            fid = self.cosmo_fid
            Omega_fid = ( fid['omega_b']+fid['omega_cdm']+fid.get('omega_nu', 0.0) )/fid['h']**2.0
            qpar, qperp = FOLPS.qpar_qperp( Omega_fid=Omega_fid,
                                            Omega_m=Omega_m,
                                            z_pk=self.zcen,
                                            cosmo=None
                                          )
        else:
            qpar, qperp = 1.0, 1.0

        # Get linear quantities
        bacco_quants = self._get_linear_pk(pars)
        k_lin  = bacco_quants['kemul_pk']
        pk_lin = bacco_quants['pk_lin']
        pk_nw  = bacco_quants['pk_nw']
        s8_fid = bacco_quants['sigma8ref']
        As_fid = self.cosmo_fid['As']
        lnAs   = pars.get('ln10^{10}A_s', np.log(1e10 * As_fid))
        As_new = np.exp(lnAs)/1e10
        sigma8 = s8_fid * np.sqrt(As_new / As_fid)
        k_pkl_pklnw = np.array([ k_lin,pk_lin,pk_nw ])

        nonlinear = FOLPS.NonLinearPowerSpectrumCalculator(
            mmatrices=self.mmatrices,
            kernels='fk',
            **folps_cosmo
        )

        # Loop tables
        table, table_nw = nonlinear.calculate_loop_table(
            k=k_lin,
            pklin=pk_lin,
            cosmo=None,
            **folps_cosmo
        )

        output_dict = { 'k': k_lin,
                        'table': table,
                        'table_nw': table_nw,
                        'k_pkl_pklnw': k_pkl_pklnw,
                        'folps_cosmo': folps_cosmo,
                        'qpar': qpar,
                        'qperp': qperp,
                        'sigma8': sigma8
                        }
        return output_dict
    #
    # [3] Bias parameters for the power spectrum
    #     In principle this could be removed, but it is being defined
    #     because the list is too long (it would make the reading of
    #     the power spectrum function cumbersome)
    #
    def _get_folps_Pk_bias_params(self, pars):
        '''
            This is only being defined because the list is too long
        '''
        #bias parameters
        # if bias_scheme='folps'   then b2=b2_mcdonald, bs=bs_mcdonald, b3=b3nl_mcdonald   (DEFAULT)
        # if bias_scheme='classpt' then b2=b2_assassi,  bs=bG2_assassi, b3=bGamma3_assassi

        #bias_scheme='classpt'
        bias_scheme='folps'

        b1 = pars['b1']
        b2 = pars['b2']
        bs = pars.get('bG2', 0.0)   # check mapping
        b3 = pars.get('bGamma3', 0.0)

        c0 = pars.get('c0', 0.0)
        c2 = pars.get('c2pp', 0.0)
        c4 = pars.get('c4pp', 0.0)

        ctilde = pars.get('ch', 0.0)
        alphashot0 = pars.get('a0', 0.0) / self.mean_density
        alphashot2 = pars.get('a2', 0.0) * (0.13*0.290521/ self.mean_density)
        PshotP = pars.get('PshotP', 1/self.mean_density)

        X_FoG = pars.get('X_FoG', 0.0)

        ppars = [
                    b1, b2, bs, b3,
                    c0, c2, c4,
                    ctilde,
                    alphashot0, alphashot2,
                    PshotP, X_FoG
                ]

        return bias_scheme, ppars

    def _apply_reparametrization(self, pars, folps_dict):
        """
            Add sigma8-A_AP reparametrization
        """

        s8    = folps_dict['sigma8']
        qpar  = folps_dict['qpar']
        qperp = folps_dict['qperp']
        A_AP  = 1.0 / (qpar * qperp**2)

        # Galaxy bias
        if 'b1_tilde' in pars:
            pars['b1'] = pars['b1_tilde'] / ( s8 * np.sqrt(A_AP) )
        if 'b2_tilde' in pars:
            pars['b2'] = pars['b2_tilde'] / ( s8**2 * np.sqrt(A_AP) )
        if 'bG2_tilde' in pars:
            pars['bG2'] = pars['bG2_tilde'] / ( s8**2 * np.sqrt(A_AP) )
        if 'bGamma3_tilde' in pars:
            pars['bGamma3'] = pars['bGamma3_tilde'] / ( s8**3 * np.sqrt(A_AP) )

        # Power spectrum counterterms
        if 'c0_tilde' in pars:
            pars['c0'] = pars['c0_tilde'] / (A_AP * s8**2)
        if 'c2pp_tilde' in pars:
            pars['c2pp'] = pars['c2pp_tilde'] / (A_AP * s8**2)
        if 'c4pp_tilde' in pars:
            pars['c4pp'] = pars['c4pp_tilde'] / (A_AP * s8**2)
        if 'a0_tilde' in pars:
            pars['a0'] = pars['a0_tilde'] / A_AP
        if 'a2_tilde' in pars:
            pars['a2'] = pars['a2_tilde'] / A_AP

        # Bispectrum
        if 'c1_tilde' in pars:
            pars['c1'] = pars['c1_tilde'] / (A_AP * s8**2)
        if 'c2_tilde' in pars:
            pars['c2'] = pars['c2_tilde'] / (A_AP * s8**2)
        if 'Pshot_tilde' in pars:
            pars['Pshot'] = pars['Pshot_tilde'] / A_AP
        if 'Bshot_tilde' in pars:
            pars['Bshot'] = pars['Bshot_tilde'] / A_AP

        return pars
    #
    # [4] Compute the 1-loop power spectrum multipoles
    #
    def pk_from_model(self, pars):

        folps = self._compute_folps_quantities(pars)
        if self.reparametrize:
            pars = self._apply_reparametrization(pars.copy(), folps)
        bias_scheme, NuisanceParams = self._get_folps_Pk_bias_params(pars)

        pkl0, pkl2, pkl4  = self.folps_pk.get_rsd_pkell(
                                            kobs=folps['k'],
                                            qpar=folps['qpar'], qper=folps['qperp'],
                                            pars=NuisanceParams,
                                            table=folps['table'], table_now=folps['table_nw'],
                                            bias_scheme=bias_scheme, damping=self.damping
                                       )

        # Build interpolation dictionary
        interp_dict = {
            '0': interp1d(folps['k'], pkl0, kind='cubic', fill_value='extrapolate'),
            '2': interp1d(folps['k'], pkl2, kind='cubic', fill_value='extrapolate'),
            '4': interp1d(folps['k'], pkl4, kind='cubic', fill_value='extrapolate'),
        }

        return interp_dict
        #self.k_pkl_pklnw = folps['k_pkl_pklnw']
        #return self.interp_function, self.k_pkl_pklnw
    #
    # [5] Compute the tree-level bispectrum multipoles
    #
    def bk_from_model(self, pars):

        folps = self._compute_folps_quantities(pars)
        if self.reparametrize:
            pars = self._apply_reparametrization(pars.copy(), folps)

        bpars = [
            pars['b1'],
            pars['b2'],
            pars.get('bG2', 0.0),
            pars.get('c1', 0.0),
            pars.get('c2', 0.0),
            pars.get('Bshot', 0.0) / self.mean_density,
            pars.get('Pshot', 0.0) / self.mean_density,
            pars.get('X_FoG_bk', 1.0)
        ]

        k1k2T = np.vstack([folps['k'],folps['k']]).T  # List of pairs of k. ( B = B(k1,k2) )
        f0 = FOLPS.f0_function(self.zcen,folps['folps_cosmo']['Omega_m'])

        #B000, B110, B220, B202, B022, B112 = self.folps_bk.Sugiyama_Bl1l2L(
        B000, B202 = self.folps_bk.Sugiyama_Bl1l2L(
                k1k2T,
                f0,
                bpars,
                qpar=folps['qpar'],
                qper=folps['qperp'],
                k_pkl_pklnw=folps['k_pkl_pklnw'],
                precision=[8,10,10],
                renormalize=True,
                damping=self.damping,
                interpolation_method='linear'
            )

        # REMOVE THIS LINE WHEN COMPUTING THESE MULTIPOLES WITH FOLPS
        B110, B220,B022, B112 = None, None, None, None

        B_map = {
                '000': B000,
                '110': B110,
                '220': B220,
                '202': B202,
                '022': B022,
                '112': B112
                }

        #interp_dict = { key: interp1d(folps['k'], value, kind='cubic', fill_value='extrapolate')
        #                    for key, value in B_map.items()
        #              }

        interp_dict = { key: interp1d(folps['k'], value, kind='cubic', fill_value='extrapolate')
                             for key, value in B_map.items()
                             if value is not None
                      }

        return interp_dict

    #### DONE WITH FOLPS

###########################################################
# BICKER EMULATOR (WILL CHANGE THE NAME TO BIKER)
group = [["c2_b2_f", "c2_b1_b2", "c2_b1_b1",
          "c2_b1_f", "c1_b1_b1_f",
          "c1_b2_f", "c1_b1_b2", "c1_b1_b1",
          "c2_b1_b1_f", "c1_b1_f"],
         ["c2_b1_f_f", "c1_f_f", "c1_f_f_f",
          "c2_f_f", "c2_f_f_f",
          "c1_b1_f_f"],
         ["c1_c1_f_f", "c2_c2_b1_f", "c2_c1_b1_f",
          "c2_c1_b1", "c2_c1_b2", "c2_c2_f_f",
          "c1_c1_f", "c2_c2_b1", "c2_c2_b2",
          "c2_c2_f", "c2_c1_f","c2_c1_f_f",
          "c1_c1_b1_f", "c1_c1_b1", "c1_c1_b2"],
         ["c1_c1_bG2", "c2_c2_bG2", "c2_c1_bG2"],
         ["c1_b1_bG2", "c1_bG2_f", "c2_bG2_f",
          "c2_b1_bG2"],
         ["b1_f_f", "b1_b1_f_f", "b1_b1_b2",
          "b2_f_f", "b1_b1_b1", "b1_b1_b1_f",
          "b1_b1_f", "b1_f_f_f", "f_f_f",
          "f_f_f_f", "b1_b2_f"],
         ["bG2_f_f", "b1_b1_bG2", "b1_bG2_f"]]

group_shot = [
                'Bshot_b1_b1', 'Bshot_b1_f', 'Bshot_b1_c1', 'Bshot_b1_c2',
                'Pshot_f_b1', 'Pshot_f_f', 'Pshot_f_c1', 'Pshot_f_c2', 'Pshot_Pshot'
            ]

kernel_name_Bk = [
                    'b1_b1_b1', 'b1_b1_b2','b1_b1_bG2','b1_b1_f','b1_b1_b1_f','b1_b1_f_f',
                    'b1_b2_f','b1_bG2_f','b1_f_f',
                    'b1_f_f_f','b2_f_f','bG2_f_f','f_f_f','f_f_f_f',
                    'c1_b1_b1','c1_b1_b2','c1_b1_bG2','c1_b1_f','c1_b1_b1_f','c1_b1_f_f','c1_b2_f',
                    'c1_bG2_f','c1_f_f','c1_f_f_f','c1_c1_b1','c1_c1_b2','c1_c1_bG2','c1_c1_f',
                    'c1_c1_b1_f','c1_c1_f_f','c2_b1_b1','c2_b1_b2','c2_b1_bG2','c2_b1_f','c2_b1_b1_f',
                    'c2_b1_f_f','c2_b2_f','c2_bG2_f','c2_f_f','c2_f_f_f','c2_c1_b1','c2_c1_b2',
                    'c2_c1_bG2','c2_c1_f','c2_c1_b1_f','c2_c1_f_f','c2_c2_b1','c2_c2_b2','c2_c2_bG2',
                    'c2_c2_f','c2_c2_b1_f','c2_c2_f_f',
                    'Bshot_b1_b1', 'Bshot_b1_f', 'Bshot_b1_c1', 'Bshot_b1_c2', 
                    'Pshot_f_b1', 'Pshot_f_f', 'Pshot_f_c1', 'Pshot_f_c2','Pshot_Pshot',
                    'fnlloc_b1_b1_b1','fnlloc_b1_b1_f','fnlloc_b1_f_f','fnlloc_f_f_f',
                    'fnlequi_b1_b1_b1','fnlequi_b1_b1_f','fnlequi_b1_f_f','fnlequi_f_f_f',
                    'fnlortho_b1_b1_b1','fnlortho_b1_b1_f','fnlortho_b1_f_f','fnlortho_f_f_f',
                    'fnlortho_LSS_b1_b1_b1','fnlortho_LSS_b1_b1_f','fnlortho_LSS_b1_f_f','fnlortho_LSS_f_f_f',]

params_sorted = ['omega_cdm','omega_b','h','ln10^{10}A_s','n_s','f','b1','b2','bG2','bGamma3', 'c0', 'c2pp', 'c4pp', 'c1', 'c2','ch','Pshot','a0','Bshot','fnlloc','fnlequi','fnlortho']

class BICKERCalculator:
    """
    Class to compute power spectrum and bispectrum models given EFT parameters.

    Attributes:
    -----------
    * cache_path : str
        Path to the cache where the emulator data is stored.
    * mean_density : float
        The mean density of the universe.
    * zcen : float
        The central redshift value.
    * fixed_params : list of strings
        Contains the parameters that were not considered in the training.
    * rescale_kernels : bool
        If the training was done with kernels scaled by As^n, then they should be rescaled back.
    * multipoles_pk : list
        List of multipoles for the power spectrum.
    * multipoles_bk : list
        List of multipoles for the bispectrum.
    * kemul_pk : array
        The k values for the power spectrum emulator.
    * kemul_bk : array
        The k values for the bispectrum emulator.
    * emulator_pk : dict
        Emulators for the power spectrum.
    * emulator_bk : dict
        Emulators for the bispectrum.
    
    Main functions:
    ---------------
    * __init__(multipoles, mean_density, redshift, cache_path, fixed_params, rescaled)
        Initialises the BICKERCalculator with the given parameters.
    
    * kernels_from_emulator(pars, ell)
        Fetches kernels from the bispectrum emulator for a given ell.
    
    * pk_from_model(pars, ell)
        Computes the power spectrum model for a given multipole.
    
    * bk_from_model(pars, l1l2L)
        Computes the bispectrum model for a given set of multipoles.
    
    * help()
        Provides documentation and usage instructions for the class.
    """
    
    def __init__(self, multipoles, mean_density, redshift, cache_path, fixed_params=['n_s'], rescale_kernels=True, ordering=1):
        """
        Parameters:
        - multipoles (list)
        - mean_density (float)
        - redshift (float)
        - cache_path (str)
        - fixed_params (list of str)
        - rescale_kernels (bool)
        """

        print('Initialising BICKERCalculator.\n')
        time_i = time()

        self.cache_path      = cache_path
        self.mean_density    = mean_density
        self.zcen            = redshift
        self.fixed_params    = fixed_params
        self.rescale_kernels = rescale_kernels
        self.ordering        = ordering

        #############################################################
        # Check if the redshift is the one used to train the emulator
        self._check_redshift()

        #############################
        # Set the multipole variables
        self.multipoles_pk = []
        self.multipoles_bk = []
        for i in multipoles:
            if len(i)==1:
                self.multipoles_pk.append(i)
            elif len(i)==3:
                self.multipoles_bk.append(i)
            else:
                print('Unrecognised multipole detected.')
        
        if self.multipoles_pk:
            self.kemul_pk = np.loadtxt(os.path.join(self.cache_path, 'powerspec/k_emul.txt'))
        if self.multipoles_bk:
            self.kemul_bk = np.loadtxt(os.path.join(self.cache_path, 'bispec/k_emul.txt'))

        #self._initialise_multipoles(multipoles) <----- this function is not working due to self.kemul

        ######################
        # Initialise emulators
        self._initialise_emulators()
        
        time_f = time()

        ######
        # Done
        print(f'Total time to initialise the calculator: {round(time_f-time_i,2)} seconds.') 
        print(f'You can now compute the {self.multipoles_pk} power spectrum and {self.multipoles_bk} bispectrum multipoles.')
        print(f'Use the function help() for further guidance.')
        
    '''
        Helper functions
    '''
    
    def _check_redshift(self):
        # Check if provided redshift is the same as the one used to train the emulator. 
        pattern = r'z(\d+\.\d+)'
        emulator_redshift = re.search(pattern, self.cache_path).group(1)
        if self.zcen != float(emulator_redshift):
            raise ValueError(f'Redshift {self.zcen} does not match the one used by the emulator: {emulator_redshift}.')
    

    def _initialise_emulators(self):
        # Initialise emulators for power spectrum and/or bispectrum
        if self.multipoles_pk:
            self.emulator_pk = {ell: BICKER.power(ell, self.kemul_pk, self.cache_path) for ell in self.multipoles_pk}
        
        if self.multipoles_bk:
            self._initialise_bk_emulators()
     
    def _initialise_bk_emulators(self):
        # Initialize bispectrum emulators from cache_path
        self.group_to_emul = self._get_groups_to_emulate()
        self.emulator_bk = {}

        for ell in self.multipoles_bk: 
            self.emulator_bk[ell] = {}
            for gp in self.group_to_emul:
                emulator_type = 'shot' if gp == 8 else gp
                self.emulator_bk[ell][gp] = BICKER.component_emulator(emulator_type, ell, self.kemul_bk, self.cache_path)
        
    def _get_groups_to_emulate(self):
        # Determine which groups to emulate based on kernel names
        group_to_emul = []
        for gp, kernels in enumerate(group):
            if any(kernel in kernel_name_Bk for kernel in kernels):
                group_to_emul.append(gp)
        if 'Bshot' in params_sorted:
            group_to_emul.append(8)
        return group_to_emul
    
    def _get_cosmo_params(self, pars):
        # Helper function to get cosmological parameters from the full EFT parameters
        # Used in the pk_from_model function.
        default_ns = 0.9649

        if self.fixed_params is None: 
            # n_s was included in the training of the emulator
            if 'n_s' not in pars:
                # but is not varied in the MCMC analysis
                if self.ordering == 0:
                    return [pars['omega_cdm'], pars['omega_b'], pars['h'], pars['ln10^{10}A_s'], default_ns]
                else:
                    return [pars['omega_b'], pars['h'], pars['omega_cdm'], pars['ln10^{10}A_s'], default_ns]
            else:
                # and it is varied in the MCMC analysis
                if self.ordering == 0:
                    return [pars['omega_cdm'], pars['omega_b'], pars['h'], pars['ln10^{10}A_s'], pars['n_s']]
                else:
                    return [pars['omega_b'], pars['h'], pars['omega_cdm'], pars['ln10^{10}A_s'], pars['n_s']]
        elif 'n_s' in self.fixed_params:
            # n_s was not included in the training of the emulator, therefore it cannot be varied
            if 'n_s' not in pars:
                if self.ordering == 0:
                    return [pars['omega_cdm'],pars['h'],pars['omega_b'], pars['ln10^{10}A_s']]
                else:
                    return [pars['omega_b'], pars['h'], pars['omega_cdm'], pars['ln10^{10}A_s']]
            else:
                raise ValueError(f"n_s was not included in the training of the emulator, therefore it cannot be varied. Fix this parameter to its fiducial value in the sampling procedure.")
                
    '''
        Main functions
    '''
    
    def kernels_from_emulator(self, pars, ell): 
        """
        Compute kernels from the emulator for given EFT parameters and multipole ell.

        Parameters:
        - pars (dict): EFT parameters.
        - ell (str): Multipole.

        Returns:
        - kbins (ndarray): Binned k values.
        - kernels (dict): Computed kernels.
        """
        cosmo_pars = self._get_cosmo_params(pars)
        self.kernels = {}

        # For the rescaled kernels
        if self.rescale_kernels:
            As = (np.exp(pars['ln10^{10}A_s'])*1e-10)
            As2 = As**2.0
        else:
            As = 1.0
            As2 = 1.0

        for gp in self.emulator_bk[ell].keys():
            predictions = self.emulator_bk[ell][gp].emu_predict(cosmo_pars, split=True)

            if gp == 8:
                for i, kern in enumerate(group_shot):
                    self.kernels[kern] = np.reshape(predictions[i], predictions[i].shape[1])
                    self.kernels[kern] *= As
            else:
                for i, kern in enumerate(group[gp]):
                    self.kernels[kern] = np.reshape(predictions[i], predictions[i].shape[1])
                    self.kernels[kern] *= As2

        return self.emulator_bk[ell][gp].kbins, self.kernels
            
    def pk_from_model(self, pars, ell):
        """
        Get power spectrum from emulator for given EFT parameters and multipole ell.

        Parameters:
        - pars (dict): EFT parameters.
        - ell (str): Multipole.

        Returns:
        - interp_function (callable): Interpolated power spectrum multipole.
        """
        
        if not len(ell) == 1:
            raise ValueError(f'{ell} is not a valid power spectrum multipole.')
        
        # Extract nuisance parameters
        b1, b2, bG2, bGamma3, ch = pars['b1'], pars['b2'], pars['bG2'], pars['bGamma3'], pars['ch']
        Pshot, a0 = pars['Pshot'], pars['a0']
        self.Pstoch = (1 + Pshot + a0 * self.emulator_pk[ell].kbins**2.0) / self.mean_density
        cosmo_pars = self._get_cosmo_params(pars)

        # Determine counterterms for each multipole and compute power spectrum from the emulator
        if ell == '0':
            c0 = pars.get('c0', 0.0)
            self.Pk_ell = self.emulator_pk[ell].emu_predict(cosmo_pars, [b1, b2, bG2, bGamma3, ch, c0])[0] + self.Pstoch
        elif ell == '2':
            c2pp = pars.get('c2pp', 0.0)
            self.Pk_ell = self.emulator_pk[ell].emu_predict(cosmo_pars, [b1, b2, bG2, bGamma3, ch, c2pp])[0]
        elif ell == '4':
            c4pp = pars.get('c4pp', 0.0)
            self.Pk_ell = self.emulator_pk[ell].emu_predict(cosmo_pars, [b1, b2, bG2, bGamma3, ch, c4pp])[0]
        else:
            raise ValueError('Unrecognised multipole.')

        self.interp_function = interp1d(self.kemul_pk, self.Pk_ell, kind='cubic', fill_value='extrapolate')
        
        return self.interp_function
            
    def bk_from_model(self, pars,l1l2L):
        """
        Compute the bispectrum from the emulator given the EFT parameters and multipoles l1, l2, L.

        Parameters:
        - pars (dict): Dictionary containing EFT parameters.
        - l1l2L (str): String representing the multipole combination for bispectrum calculation.

        Returns:
        - interp_function (callable): Interpolated bispectrum multipole.
        """
        
        if not len(l1l2L) == 3:
            raise ValueError(f'{l1l2L} is not a valid bispectrum multipole.')
            
        b1  = pars.get('b1',1.0)
        b2  = pars.get('b2',0.0)
        bG2 = pars.get('bG2',0.0)
        c1  = pars.get('c1',0.0)
        c2  = pars.get('c2',0.0)
        Pshot = pars.get('Pshot',0.0)
        Bshot = pars.get('Bshot',0.0)

        # Get kernels for bispectrum emulator
        self.kernels_k, self.kernels_Bk = self.kernels_from_emulator(pars, l1l2L)
        self.bk_model = np.zeros(len(self.kernels_k))

        # Iterate over the kernels to compute the bispectrum model
        for b, values in self.kernels_Bk.items():
            bias = 1
            if 'b1' in b:
                bias *= b1 ** b.count('b1')
            if 'b2' in b:
                bias *= b2
            if 'bG2' in b:
                bias *= bG2
            if 'c1' in b:
                bias *= c1 ** b.count('c1')
            if 'c2' in b:
                bias *= c2 ** b.count('c2')
            if 'Pshot' in b:
                bias *= (1 + Pshot) / self.mean_density
            if 'Bshot' in b:
                bias *= Bshot / self.mean_density
            if 'fnlequi' in b:
                bias *= fnlequi
            if 'fnlortho' in b:
                bias *= fnlortho

            # Get the bias-weighted kernel
            self.bk_model += bias * values

        #if Pshot != 0: 
        #    self.bk_model += ((1+Pshot)/self.mean_density)**2

        self.interp_function = interp1d(self.kernels_k, self.bk_model, kind='cubic', fill_value='extrapolate')

        return self.interp_function

    def help(self):
        """
        Help Function
        =============

        This function provides an overview of the BICKERCalculator class, its attributes, 
        and methods.

        Usage:
        ------
        calculator = BICKERCalculator(multipoles, mean_density, redshift, cache_path, fixed_params, rescale)

        Methods:
        --------
        1. kernels_from_emulator(pars, ell)
            - Fetches kernels from the emulator for a given ell.
            - Parameters: 
                pars (dict): A dictionary containing EFT parameters.
                ell (str): The multipole to compute (e.g., '000', '202').

        2. pk_from_model(pars, ell)
            - Computes the power spectrum model for a given multipole.
            - Parameters:
                pars (dict): A dictionary containing EFT parameters.
                ell (str): The multipole to compute (e.g., '0', '2', '4').

        3. bk_from_model(pars, l1l2L)
            - Computes the bispectrum model for a given set of multipoles.
            - Parameters:
                pars (dict): A dictionary containing EFT parameters.
                l1l2L (str): A string representing the bispectrum multipoles.

        Example:
        --------
        ```
        calculator = BICKERCalculator(multipoles=['0', '2','000'], mean_density=1e-3, redshift=0.8, cache_path='path/to/z0.8')
        Pk_0 = calculator.pk_from_model(pars, '2')
        Bk_202 = calculator.bk_from_model(pars, '202')
        ```
        """
        print(self.help.__doc__)

###########################################################

###########################################################
# MAIN FUNCTION TO BE CALLED BY THE PIPELINE
# NOTICE: IT IS BLIND TO THE MODEL CHOSEN
#         SO IT NEVER NEEDS TO BE CHANGED (IN PRINCIPLE)
class ModellingFunction:
    def __init__(self, priors, data, calculator, multipoles, window_matrix=None, k_theory_window=None):
        """
        Initialize the ModellingFunction class.

        Args:
            priors (dict): Dictionary of priors for the parameters.
            data (dict): Dictionary containing the loaded data.
            calculator: The emulator-based (`BICKERCalculator`) or FOLPS-based calculator object (`FOLPSCalculator`).
            multipoles (list): List of multipoles to compute the model for.
        """
        self.priors = priors
        self.data = data
        self.calculator = calculator
        self.multipoles = multipoles
        self.fixed_params = self._extract_fixed_params()
        self.window_matrix = window_matrix
        self.k_theory_window = k_theory_window

        # Separate multipoles into power spectrum (Pk) and bispectrum (Bk)
        self.multipoles_pk = [i for i in self.multipoles if len(i) == 1] or None
        self.multipoles_bk = [i for i in self.multipoles if len(i) == 3] or None

    def _extract_fixed_params(self):
        """
        Extract fixed parameters from the priors dictionary.

        Args:
            priors (dict): Dictionary of priors.

        Returns:
            fixed_params (dict): Dictionary of fixed parameters and their values.
        """
        fixed_params = {}
        for param, prior_info in self.priors.items():
            if prior_info['type'] == 'Fix' and param != 'n_s':
                fixed_params[param] = prior_info['lim']
        return fixed_params

    def get_parameters_dictionary(self, theta):
        """
        Convert sampler parameter vector theta (list of numbers) into a full parameter dictionary
        (free + fixed parameters).
        """
        parameters_to_vary = {}
        free_param_names   = [param for param in self.priors if self.priors[param]['type'] != 'Fix']
        for i, param in enumerate(free_param_names):
            parameters_to_vary[param] = theta[i]

        return {**parameters_to_vary, **self.fixed_params}

    def pk_convolved(self, full_params):
        '''
            This function is for testing only - it's not used in the pipeline!!!
        '''
        # Initialize an empty list to store the model predictions
        model_vector = []

        # Compute power spectrum predictions
        if self.multipoles_pk:
            pk_interp = self.calculator.pk_from_model(full_params)
            k_theory = self.k_theory_window if self.k_theory_window is not None else None
            for L in self.multipoles_pk:
                k_array = k_theory if k_theory is not None else self.data[L]['k']
                model_vector.append( pk_interp[L](k_array) )

        # Concatenate the model predictions into a single array
        theory_vector = np.concatenate(model_vector)

        # Window convolution
        if self.window_matrix is not None:
            theory_vector = self.window_matrix.dot(theory_vector)

        return theory_vector

    def compute_model_vector(self, theta):
        """
        Compute the model predictions for the power spectrum and bispectrum based on the input parameters.

        Args:
            theta (np.ndarray): Array of parameter values sampled by the Monte-Carlo method.

        Returns:
            np.ndarray: Concatenated model predictions for the specified multipoles.
                        (to be compared directly with the concatenated data vector in the Likelihood).
        """

        full_params = self.get_parameters_dictionary(theta)
        #print(f'theta={theta}')
        #print(f'full_params={full_params}')

        # Initialize an empty list to store the model predictions
        model_vector = []

        # Compute power spectrum predictions
        if self.multipoles_pk:
            pk_interp = self.calculator.pk_from_model(full_params)
            k_theory = self.k_theory_window if self.k_theory_window is not None else None
            for L in self.multipoles_pk:
                k_array = k_theory if k_theory is not None else self.data[L]['k']
                model_vector.append( pk_interp[L](k_array) )

        # Compute bispectrum predictions
        if self.multipoles_bk:
            bk_interp = self.calculator.bk_from_model(full_params)
            for l1l2L in self.multipoles_bk:
                k_from_data = self.data[l1l2L]['k']
                model_vector.append( bk_interp[l1l2L](k_from_data) )

        # Concatenate the model predictions into a single array
        theory_vector = np.concatenate(model_vector)

        # Window convolution
        if self.window_matrix is not None:
            theory_vector = self.window_matrix.dot(theory_vector)

        return theory_vector

###########################################################
