import numpy as np
from scipy.interpolate import interp1d
import baccoemu # REMOVE THIS IF NOT USING FOLPS

import os, sys
os.environ['FOLPS_BACKEND'] = 'numpy'
sys.path.append('/cosma/home/dp322/dc-guan2/folps/folpsD/')
import folps as FOLPS

from time import time
import re

'''
Important caveats:
* In the priors, always use the parameter ln10^{10}A_s. However, FOLPS samples over A_s, so there is an internal conversion!
'''

class FOLPSCalculator:

    def __init__(self, mean_density, redshift,
                 model='EFT', damping=None, use_TNS_model=False,
                 AP=True, cosmo_fid=None, reparametrize=False):
        '''
            Damping: either "None" or " 'lor' "
            cosmo_fis: for AP;
        '''

        self.mean_density = mean_density
        self.zcen         = redshift
        self.expfactor    = 1.0/(1.0+redshift) # for baccoemu
        self.AP           = AP
        self.cosmo_fid    = cosmo_fid or  {'omega_b' : 0.02237,
                                          'omega_cdm': 0.12,
                                          'omega_nu' : 0.00064420,
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

    def _sigma_from_pk(self, k_, pk_, h_=None, R_=8.0):
        """
            Sigma 8 computation from linear P(k)

            Args:
                k_ : np.array (1/Mpc or h/Mpc: must be consistent with Pk)
                pk_: np.array (Mpc^3 or (Mpc/h)^3: consistent with k)
                h_ : Hubble (optional. Must be passed if Mpc units, though)
                R  : smoothing scale in Mpc/h (default 8.0)
            ---------
            Returns: np.float
        """
        if h_ is not None:
            R_ = R_/h_ # now in Mpc if k is in 1/Mpc

        x = k_ * R_
        W = (3.0 / x**3) * (np.sin(x) - x * np.cos(x))
        W = np.where(x < 1e-6, 1.0, W)
        integrand = k_**3 * pk_ * W**2

        return np.sqrt(np.trapz(integrand, np.log(k_)) / (2.0 * np.pi**2))

    ########################################################
    # BACCOEMU FOR LINEAR POWER SPECTRUM
    def _initialise_linear_pk_baccoemu(self):
        self.emulator = baccoemu.Matter_powerspectrum(verbose=False)
        self.kemul_pk = np.logspace(-4, np.log10(3), num=1000)

    def _get_linear_pk(self, pars):
        # bacco calls Omega_x omega_x.
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
        
        tmpk_  = np.geomspace(1e-4,20,2000)
        tmppk_ = interp1d(np.log(self.kemul_pk),np.log(self.pk_lin),
                          bounds_error=False,fill_value='extrapolate',kind='cubic')(np.log(tmpk_))
        self.sigma8_at_z = self._sigma_from_pk(tmpk_,np.exp(tmppk_))

        self.output_dict = {'kemul_pk': self.kemul_pk,
                            'pk_lin': self.pk_lin,
                            'pk_nw': self.pk_nw,
                            'sigma8': self.sigma8_at_z}

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
            Omega_fid = ( fid['omega_b']+fid['omega_cdm']+fid['omega_nu'] )/fid['h']**2.0
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
        sigma8 = bacco_quants['sigma8']
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
    def _get_folps_Pk_bias_params(self, pars, f_):
        '''
            This is only being defined because the list is too long
        '''
        #bias_scheme='classpt'
        bias_scheme='folps'
        #bias_scheme = 'DESI'

        # Performing the change of from bK² btd to the folps basis directly
        b1 = pars['b1']
        b2 = pars['b2']
        bs = 2.0*pars.get('bG2', 0.0)
        b3 = 64/105 * (-5/4 * bs - pars.get('bGamma3', 0.0))

        c0 = b1**2 * pars.get('c0', 0.0) #pars.get('c0', 0.0)
        c2 = b1*f_ * (pars.get('c0', 0.0) + pars.get('c2pp', 0.0)) #pars.get('c2pp', 0.0)
        c4 = f_**2 * pars.get('c2pp', 0.0) + (b1 * f_) * pars.get('c4pp', 0.0) #pars.get('c4pp', 0.0)

        ctilde = pars.get('ch', 0.0)

        alphashot0 = 1e-4*pars.get('a0', 0.0) / self.mean_density
        alphashot2 = 1e-4*pars.get('a2', 0.0) * (0.15*38.36415260435053/ self.mean_density)
        PshotP     = 1e-4 * pars.get('PshotP', 1/self.mean_density)

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
            pars['bGamma3'] = pars['bGamma3_tilde'] / ( s8**4 * A_AP )

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
        f0 = FOLPS.f0_function(self.zcen,folps['folps_cosmo']['Omega_m'])

        if self.reparametrize:
            pars = self._apply_reparametrization(pars.copy(), folps)
        bias_scheme, NuisanceParams = self._get_folps_Pk_bias_params(pars,f0)

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

        B000, B110, B220, B202, B022, B112 = self.folps_bk.Sugiyama_Bl1l2L(
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

        B_map = {
                '000': B000,
                '110': B110,
                '220': B220,
                '202': B202,
                '022': B022,
                '112': B112
                }

        interp_dict = { key: interp1d(folps['k'], value, kind='cubic', fill_value='extrapolate')
                             for key, value in B_map.items()
                             if value is not None
                      }

        return interp_dict

    def bk_2d_from_model(self, pars, k_eval):
        """
            Return full 2D bispectrum grids evaluated on k_grid.
            This is ONLY used for window convolution.
        """

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

        f0 = FOLPS.f0_function(self.zcen, folps['folps_cosmo']['Omega_m'])

        # FULL multipoles
        Nk = len(k_eval)
        i, j = np.tril_indices(Nk)
        k1k2 = np.column_stack([k_eval[i], k_eval[j]])

        B000, B110, B220, B202, B022, B112 = self.folps_bk.Sugiyama_Bl1l2L(
            k1k2,
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

        def reconstruct_symmetric(Btri):
            B_tmp = np.zeros((Nk, Nk))
            B_tmp[i, j] = Btri
            B_tmp[j, i] = Btri
            return B_tmp

        def reconstruct_mixed(B202_tri, B022_tri):
            B202g = np.zeros((Nk, Nk))
            B022g = np.zeros((Nk, Nk))
            B202g[i, j] = B202_tri
            B022g[i, j] = B022_tri
            B202g[j, i] = B022_tri
            B022g[j, i] = B202_tri
            return B202g, B022g

        grids = {
            '000': reconstruct_symmetric(B000),
            '110': reconstruct_symmetric(B110),
            '220': reconstruct_symmetric(B220),
            '112': reconstruct_symmetric(B112),
        }

        B202g, B022g = reconstruct_mixed(B202, B022)
        grids['202'] = B202g
        grids['022'] = B022g

        return grids

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

        if isinstance(self.multipoles, dict):
            # These are the multipoles for convolution.
            # If the power spectrum is being computed, then self.multipoles_pk = ['0','2','4']
            # The power spectrum window matrix already takes care of the multipoles to be evaluated
            # in the analysis. For example: P_out = W P_{0,2,4}, P_out will have only the monopole if that's
            # the only multipole used for the power spectrum analysis (defined in the config.yml file)
            self.multipoles_pk = self.multipoles.get('Pk')
            # If the bispectrum is being computed, then this dictionary will be something
            # like {'000': ['000', '110', '220']}, where '000' is the multipole for the analysis,
            # and the remaining list are the required multipoles for the '000' convolution.
            self.multipoles_bk = self.multipoles.get('Bk')

        elif isinstance(self.multipoles, list):
            # These are the multipoles to be computed, no need for convolution
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

        # Initialize an empty list to store the model predictions
        pk_vector = []
        # Compute power spectrum predictions
        if self.multipoles_pk:
            pk_interp = self.calculator.pk_from_model(full_params)
            if self.k_theory_window is not None and self.k_theory_window.get('Pk') is not None:
                k_theory = self.k_theory_window['Pk']
            else:
                k_theory = None
            for L in self.multipoles_pk:
                k_array = k_theory if k_theory is not None else self.data[L]['k']
                pk_vector.append( pk_interp[L](k_array) )
            # Concatenate the model predictions into a single array
            pk_vector = np.concatenate(pk_vector)
            if self.window_matrix and self.window_matrix.get('Pk') is not None:
                pk_vector = self.window_matrix['Pk'].dot(pk_vector)

        bk_vector = []
        # Compute bispectrum predictions
        if self.multipoles_bk:
            #k_theory = self.k_theory_window['Bk'] if self.k_theory_window['Bk'] is not None else None
            if self.k_theory_window is not None and self.k_theory_window.get('Bk') is not None:
                k_theory = self.k_theory_window['Pk']
            else:
                k_theory = None

            if k_theory is not None and self.window_matrix and self.window_matrix.get('Bk') is not None:
                # Build (k1,k2) for 2d grid pairs
                bk_2d = self.calculator.bk_2d_from_model(full_params,k_theory)
                for L in self.multipoles_bk:
                    # Example:
                    # L = '000': convolution with ['000','110','220','202']
                    # L = '202': convolution with ['000','110','220','112','202']
                    combined_list = []
                    for l in self.multipoles_bk[L]:
                        combined_list.append(bk_2d[l].ravel())
                    combined = np.concatenate(combined_list)
                    Bconv = np.dot(self.window_matrix['Bk'][L], combined).reshape(len(k_theory), len(k_theory))
                    Bdiag = interp1d(k_theory,np.diag(Bconv),kind='cubic', fill_value='extrapolate')
                    # Now, the Bdiag was computed for the 64 bins of the window. So we will interpolate it and match
                    # to the k vector from the data we want to fit
                    k_array = self.data[L]['k']
                    bk_vector.append( Bdiag(k_array) )
            else:
                bk_interp = self.calculator.bk_from_model(full_params)
                for l1l2L in self.multipoles_bk:
                    k_array = self.data[l1l2L]['k']
                    bk_vector.append( bk_interp[l1l2L](k_array) )
            bk_vector = np.concatenate(bk_vector)

        theory_vector = []
        if len(pk_vector) > 0:
            theory_vector.append(pk_vector)
        if len(bk_vector) > 0:
            theory_vector.append(bk_vector)

        return np.concatenate(theory_vector).flatten()

###########################################################
