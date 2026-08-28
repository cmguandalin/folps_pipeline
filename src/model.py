import numpy as np
from scipy.interpolate import interp1d
from collections import OrderedDict

import os, sys
os.environ['FOLPS_BACKEND'] = 'numpy'  #'numpy' or 'jax'
sys.path.append('/cosma/home/dp322/dc-guan2/folps/folpsD/')
import folps as FOLPS

import warnings
from time import time
import re

# Key to make the neutrino-mass handling consistent across the file
# pklin_emulator_jit.py uses Mnu
NEUTRINO_MASS_KEYS = ('m_ncdm', 'm_nu', 'Mnu')
# Needed for folpsD, which takes the neutrino fraction fnu (computed from the physical neutrino density omega_nu)
NEUTRINO_MASS_TO_OMEGA_NU = 93.14


class FOLPSCalculator:

    def __init__(self, mean_density, redshift, tracer,
                 model='EFT', damping=None, use_TNS_model=False,
                 AP=True, cosmo_fid=None, reparametrize=False,
                 emulator='bacco', jaxmapse_plin_path=None, jaxmapse_pnw_path=None,
                 folps_nk=1000):
        '''
            Damping: either "None" or " 'lor' "
            cosmo_fis: for AP;
        '''

        self.mean_density = mean_density
        self.zcen         = redshift
        self.tracer       = tracer
        self.expfactor    = 1.0/(1.0+redshift) # for baccoemu
        self.AP           = AP
        self.cosmo_fid    = cosmo_fid or  {'omega_b' : 0.02237,
                                          'omega_cdm': 0.12,
                                          'omega_nu' : 0.00064420,
                                          'h'        : 0.6736,
                                          'ns'       : 0.9649,
                                          'As'       : 2.0830e-9,
                                          'w0'       : -1.0,
                                          'wa'       : 0.0}
        self.reparametrize = reparametrize
        self.linear_pk_emulator = emulator
        self.jaxmapse_plin_path = jaxmapse_plin_path
        self.jaxmapse_pnw_path  = jaxmapse_pnw_path
        self.folps_nk = int(folps_nk)
        self._folps_cache = OrderedDict()
        self._folps_cache_max_size = 8

        ######################
        # Initialise linear power spectrum emulator
        self._initialise_linear_pk_emulator()
        #print(f'Initialised {self.linear_pk_emulator} for {self.tracer} at z={self.zcen}.')

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

    def _get_neutrino_mass(self, pars, default=0.06):
        for key in NEUTRINO_MASS_KEYS:
            if key in pars:
                return pars[key]
        return default

    def _get_neutrino_density(self, pars, default_mass=0.06):
        for key in NEUTRINO_MASS_KEYS:
            if key in pars:
                return pars[key] / NEUTRINO_MASS_TO_OMEGA_NU
        if 'omega_nu' in pars:
            return pars['omega_nu']
        return default_mass / NEUTRINO_MASS_TO_OMEGA_NU

    def _sigma_from_pk(self, k_, pk_, h_=None, R_=8.0):
        '''
            Sigma 8 computation from linear P(k)

            Args:
                k_ : np.array (1/Mpc or h/Mpc: must be consistent with Pk)
                pk_: np.array (Mpc^3 or (Mpc/h)^3: consistent with k)
                h_ : Hubble (optional. Must be passed if Mpc units, though)
                R  : smoothing scale in Mpc/h (default 8.0)
            ---------
            Returns: np.float
        '''
        if h_ is not None:
            R_ = R_/h_ # now in Mpc if k is in 1/Mpc

        x = k_ * R_
        W = (3.0 / x**3) * (np.sin(x) - x * np.cos(x))
        W = np.where(x < 1e-6, 1.0, W)
        integrand = k_**3 * pk_ * W**2

        return np.sqrt(np.trapz(integrand, np.log(k_)) / (2.0 * np.pi**2))

    def _legendre(self, ell, mu):
        if ell == 0:
            return 1.0
        if ell == 2:
            return 0.5 * (3.0 * mu**2 - 1.0)
        if ell == 4:
            return (35.0 * mu**4 - 30.0 * mu**2 + 3.0) / 8.0
        raise ValueError(f'Unsupported P(k) multipole: {ell}')

    ########################################################
    # EMULATORS FOR LINEAR POWER SPECTRUM
    def _initialise_linear_pk_emulator(self):
        valid_emulators = {'bacco', 'jaxmapse'}
        if self.linear_pk_emulator not in valid_emulators:
            raise ValueError(
                f"Unknown linear power spectrum emulator '{self.linear_pk_emulator}'. "
                f"Choose one of {sorted(valid_emulators)}."
            )

        if self.linear_pk_emulator == 'bacco':
            self._initialise_linear_pk_baccoemu()
        elif self.linear_pk_emulator == 'jaxmapse':
            self._initialise_linear_pk_jaxmapse()

    def _initialise_linear_pk_baccoemu(self):
        import baccoemu
        print('Initialising baccoemu')
        self.emulator = baccoemu.Matter_powerspectrum(verbose=False)
        self.kemul_pk = np.logspace(-4, np.log10(3), num=self.folps_nk)

    def _initialise_linear_pk_jaxmapse(self):
        if self.jaxmapse_plin_path is None or self.jaxmapse_pnw_path is None:
            raise ValueError(
                "The jaxmapse emulator requires both 'jaxmapse_plin_path' and 'jaxmapse_pnw_path' "
                "in the config file."
            )

        from pklin_emulator_jit import PkEmulator
        import jaxmapse

        print('Initialising jaxmapse')

        self.emulator = PkEmulator(
            path_plin=self.jaxmapse_plin_path,
            path_pnw=self.jaxmapse_pnw_path
        )
        self._predict_plin_jaxmapse = self.emulator.get_jit_predict_plin()
        self._predict_pnw_jaxmapse  = self.emulator.get_jit_predict_pnw()
        self.kemul_pk  = np.asarray(self.emulator._plin['k'])
        self._jaxmapse = jaxmapse

    def _get_linear_pk(self, pars):
        if self.linear_pk_emulator == 'bacco':
            return self._get_linear_pk_bacco(pars)
        if self.linear_pk_emulator == 'jaxmapse':
            return self._get_linear_pk_jaxmapse(pars)
        raise ValueError(f"Unknown linear power spectrum emulator '{self.linear_pk_emulator}'.")

    def _get_linear_pk_bacco(self, pars):

        # OBS: bacco calls Omega_x omega_x.

        bacco_cosmo_pars = {
                    'omega_cold'    : (pars['omega_cdm'] + pars['omega_b']) / pars['h']**2,
                    'omega_baryon'  : pars['omega_b']/pars['h']**2,
                    'hubble'        : pars['h'],
                    'neutrino_mass' : self._get_neutrino_mass(pars),
                    'ns'            : pars.get('n_s', 0.9649),
                    'A_s'           : np.exp(pars['ln10^{10}A_s']) / 1e10,
                    'w0'            : pars.get('w0', -1.0),
                    'wa'            : pars.get('wa', 0.0),
                    'expfactor'     : self.expfactor
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
                            'sigma8': self.sigma8_at_z,
                            'f0': None,
                            'qpar': None,
                            'qperp': None}

        return self.output_dict

    def _get_linear_pk_jaxmapse(self, pars):
        logA = pars['ln10^{10}A_s']
        n_s  = pars.get('n_s', 0.9649)
        h    = pars['h']
        omega_b   = pars['omega_b']
        omega_cdm = pars['omega_cdm']
        m_ncdm = self._get_neutrino_mass(pars)
        w0 = pars.get('w0', -1.0)
        wa = pars.get('wa', 0.0)

        jax_cosmo = self._jaxmapse.w0waCDMCosmology(
                            ln10As=logA, ns=n_s, h=h,
                            omega_b=omega_b, omega_c=omega_cdm,
                            m_nu=m_ncdm, w0=w0, wa=wa,
                        )
        D_z = jax_cosmo.D_z(self.zcen)
        f0 = float(jax_cosmo.f_z(self.zcen))

        k_lin, pk_lin = self._predict_plin_jaxmapse(
                            z      = self.zcen,
                            ln10As = logA,
                            ns     = n_s,
                            H0     = h * 100.0,
                            ombh2  = omega_b,
                            omch2  = omega_cdm,
                            Mnu    = m_ncdm,
                            w0     = w0,
                            wa     = wa,
                            D      = D_z,
                        )
        _, pk_nw = self._predict_pnw_jaxmapse(
                        z      = self.zcen,
                        ln10As = logA,
                        ns     = n_s,
                        H0     = h * 100.0,
                        ombh2  = omega_b,
                        omch2  = omega_cdm,
                        Mnu    = m_ncdm,
                        w0     = w0,
                        wa     = wa,
                        D      = D_z,
                    )

        k_lin  = np.asarray(k_lin)
        pk_lin = np.asarray(pk_lin)
        pk_nw  = np.asarray(pk_nw)
        pk_lin = np.where(np.isfinite(pk_lin) & (pk_lin > 0.0), pk_lin, np.nan)
        pk_nw  = np.where(np.isfinite(pk_nw) & (pk_nw > 0.0), pk_nw, np.nan)

        if not np.isfinite(pk_lin).all() or not np.isfinite(pk_nw).all():
            raise ValueError( 'The jaxmapse linear P(k) emulator returned non-positive or non-finite values' )

        tmpk_  = np.geomspace(1e-4, 20, 2000)
        tmppk_ = interp1d( np.log(k_lin), np.log(pk_lin), bounds_error=False, fill_value='extrapolate', kind='cubic' )(np.log(tmpk_))
        sigma8_at_z = self._sigma_from_pk(tmpk_, np.exp(tmppk_))

        qpar, qperp = None, None
        if self.AP:
            fid = self.cosmo_fid
            fid_cosmo = self._jaxmapse.w0waCDMCosmology(
                            ln10As=np.log(1e10 * fid['As']),
                            ns=fid['ns'],
                            h=fid['h'],
                            omega_b=fid['omega_b'],
                            omega_c=fid['omega_cdm'],
                            m_nu=self._get_neutrino_mass(fid),
                            w0=fid.get('w0', -1.0),
                            wa=fid.get('wa', 0.0),
                        )
            qperp = float(h * jax_cosmo.r_z(self.zcen) / fid_cosmo.r_z(self.zcen) / fid['h'])
            #qperp = float(jax_cosmo.r_z(self.zcen) / fid_cosmo.r_z(self.zcen))
            qpar  = float(fid_cosmo.E_z(self.zcen) / jax_cosmo.E_z(self.zcen))

        self.output_dict = {'kemul_pk': k_lin,
                            'pk_lin': pk_lin,
                            'pk_nw': pk_nw,
                            'sigma8': sigma8_at_z,
                            'f0': f0,
                            'qpar': qpar,
                            'qperp': qperp}

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

    def set_folps_cache(self, cache):
        self._folps_cache = cache

    def _folps_cache_key(self, pars):
        '''
            Building a "label" used to decide whether a cached FOLPS result can be reused.
            Extracts FOLPS cosmology dependent ingredients to be used as a dictionary key.
            --------
            Returns:
                A tuple containing cached FOLPS results to be reused, including:
                    1 - emulator choice (e.g., bacco or jaxmapse)
                    2 - theory model    (e.g. EFT or TNS)
                    3 - whether TNS terms are used
                    4 - whether AP is enabled
                    5 - redshift
                    6 - sampled cosmological parameters
                    7 - fiducial cosmology (if AP)
        '''

        # Cosmological parameters that affect the linear power spectrum, growth, and AP factors;
        # This excludes nuisance/bias parameters as they are not required for recomputing the 
        # cosmology-dependent loop table (which will be recomputed for the analytical marginalisation)
        cosmo_keys = ( 'omega_b',
                       'omega_cdm',
                       'omega_nu',
                       'm_ncdm',
                       'm_nu',
                       'Mnu',
                       'h',
                       'n_s',
                       'ln10^{10}A_s',
                       'w0',
                       'wa'
                     )

        fid_items = ()
        # If AP corrections are enabled, the output depends on the fiducial cosmology.
        if self.AP and self.cosmo_fid is not None:
            fid_items = tuple(
                (key, float(self.cosmo_fid[key]))
                for key in sorted(self.cosmo_fid)
                if key in self.cosmo_fid
            )
        return ( self.linear_pk_emulator,
                 self.model,
                 self.use_TNS_model,
                 self.AP,
                 float(self.zcen),
                 tuple((key, float(pars[key])) for key in cosmo_keys if key in pars),
                 fid_items,
               )

    def _get_cached_folps_quantities(self, key):
        if self._folps_cache is None:
            return None
        try:
            cached = self._folps_cache.pop(key)
        except KeyError:
            return None
        self._folps_cache[key] = cached
        return cached

    def _store_cached_folps_quantities(self, key, value):
        if self._folps_cache is None:
            return
        self._folps_cache[key] = value
        while len(self._folps_cache) > self._folps_cache_max_size:
            self._folps_cache.popitem(last=False)
    #
    # [2] Everything folps requires for P(k) and B(k)
    #
    def _compute_folps_quantities(self, pars):
        cache_key = self._folps_cache_key(pars)
        cached = self._get_cached_folps_quantities(cache_key)
        if cached is not None:
            return cached

        # Build cosmology dictionary
        omega_b   = pars['omega_b']
        omega_cdm = pars['omega_cdm']
        h         = pars['h']
        m_nu      = self._get_neutrino_mass(pars)
        omega_nu  = self._get_neutrino_density(pars)

        # Compute Omega_m from sampled cosmology
        Omega_m = (omega_b+omega_cdm+omega_nu)/h**2
        f_nu    = omega_nu/(omega_cdm+omega_b+omega_nu)

        folps_cosmo = {
            **pars,
            'omega_nu': omega_nu,
            'm_ncdm': m_nu,
            'm_nu': m_nu,
            'Mnu': m_nu,
            'z': self.zcen,
            'Omega_m': Omega_m,
            'fnu': f_nu
        }

        # Get linear quantities
        aux_vars = self._get_linear_pk(pars)
        k_lin  = aux_vars['kemul_pk']
        pk_lin = aux_vars['pk_lin']
        pk_nw  = aux_vars['pk_nw']
        sigma8 = aux_vars['sigma8']
        k_pkl_pklnw = np.array([ k_lin,pk_lin,pk_nw ])

        # Alcock-Paczynski effect
        if self.AP and aux_vars.get('qpar') is not None:
            qpar = aux_vars['qpar']
            qperp = aux_vars['qperp']
        elif self.AP:
            fid = self.cosmo_fid
            Omega_fid = (fid['omega_b'] + fid['omega_cdm'] + self._get_neutrino_density(fid)) / fid['h']**2.0
            qpar, qperp = FOLPS.qpar_qperp( Omega_fid=Omega_fid,
                                            Omega_m=Omega_m,
                                            z_pk=self.zcen,
                                            cosmo=None
                                          )
        else:
            qpar, qperp = 1.0, 1.0

        f0 = aux_vars.get('f0')
        if f0 is None:
            f0 = FOLPS.f0_function(self.zcen, Omega_m)

        nonlinear = FOLPS.NonLinearPowerSpectrumCalculator(
            mmatrices=self.mmatrices,
            kernels='fk',
            **folps_cosmo
        )

        # Loop tables
        k_nw_loop, pk_nw_loop = FOLPS.extrapolate_pklin(k_lin, pk_nw)
        table, table_nw = nonlinear.calculate_loop_table(
            k     = k_lin,
            pklin = pk_lin,
            pknow = (k_nw_loop, pk_nw_loop),
            cosmo = None,
            f0    = f0,
            **folps_cosmo
        )

        output_dict = { 'k': k_lin,
                        'table': table,
                        'table_nw': table_nw,
                        'k_pkl_pklnw': k_pkl_pklnw,
                        'folps_cosmo': folps_cosmo,
                        'qpar': qpar,
                        'qperp': qperp,
                        'sigma8': sigma8,
                        'f0': f0
                        }
        self._store_cached_folps_quantities(cache_key, output_dict)
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
        #bias parameters
        # if bias_scheme='folps'   then b2=b2_mcdonald, bs=bs_mcdonald, b3=b3nl_mcdonald   (DEFAULT)
        # if bias_scheme='classpt' then b2=b2_assassi,  bs=bG2_assassi, b3=bGamma3_assassi

        #bias_scheme='classpt'
        bias_scheme='folps'

        b1 = pars['b1']
        b2 = pars['b2']
        bs = 2.0*pars.get('bG2', 0.0)
        b3 = 64/105 * (-5/4 * bs - pars.get('bGamma3', 0.0))

        c0 = b1**2 * pars.get('c0', 0.0)
        c2 = b1*f_ * (pars.get('c0', 0.0) + pars.get('c2pp', 0.0))
        c4 = f_**2 * pars.get('c2pp', 0.0) + (b1 * f_) * pars.get('c4pp', 0.0)

        ctilde = pars.get('ch', 0.0)

        alphashot0 = pars.get('a0', 0.0)
        alphashot2 = pars.get('a2', 0.0)
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
            pars['bGamma3'] = pars['bGamma3_tilde'] / ( s8**4 * A_AP )

        # Power spectrum counterterms
        if 'LRG' in self.tracer:
            sigv = 150*(10)**(1/3)*(1+0.8)**(1/2)/70.
            fsat = 0.15
        elif 'QSO' in self.tracer:
            sigv = 150*(10)**(0.7/3)*(2.4)**(1/2)/70.
            fsat = 0.03

        if 'c0_tilde' in pars:
            pars['c0']   = pars['c0_tilde'] / (A_AP * s8**2)
        if 'c2pp_tilde' in pars:
            pars['c2pp'] = pars['c2pp_tilde'] / (A_AP * s8**2)
        if 'c4pp_tilde' in pars:
            pars['c4pp'] = pars['c4pp_tilde'] / (A_AP * s8**2)
        if 'a0_tilde' in pars:
            pars['a0'] = pars['a0_tilde'] / A_AP
        if 'a2_tilde' in pars:
            pars['a2'] = fsat*sigv**2 * pars['a2_tilde'] / A_AP

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
        f0 = folps['f0']

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
    # [4.1] Compute the 1-loop power spectrum multipoles for analytical marginalisation
    #
    def pk_marginalised_from_model(self, pars, marginalised_params, multipoles, k_eval=None):
        """
            Return the non-marginalised P(k) multipoles and templates for the
            nuisance parameters that are analytically marginalised.
        """

        folps = self._compute_folps_quantities(pars)
        f0 = folps['f0']

        if self.reparametrize:
            pars = self._apply_reparametrization(pars.copy(), folps)
        bias_scheme, nuisance_params = self._get_folps_Pk_bias_params(pars, f0)

        ells = tuple(int(ell) for ell in multipoles)
        kobs = folps['k'] if k_eval is None else np.asarray(k_eval)
        nuisance_params = self.folps_pk.set_bias_scheme(nuisance_params, bias_scheme=bias_scheme)
        b1, b2, bs, b3, alpha0, alpha2, alpha4, ctilde, alphashot0, alphashot2, PshotP, X_FoG = nuisance_params

        # Get the constant (nuisance to be marginalised over set to zero) model
        '''
        nuisance_const = [
                            b1, b2, bs, b3,
                            0.0, 0.0, 0.0,
                            ctilde,
                            0.0, 0.0,
                            PshotP, X_FoG
                         ]
        '''
        marginalised_params = set(marginalised_params)

        marg_c0   = ('c0'   in marginalised_params) or ('c0_tilde'   in marginalised_params)
        marg_c2pp = ('c2pp' in marginalised_params) or ('c2pp_tilde' in marginalised_params)
        marg_c4pp = ('c4pp' in marginalised_params) or ('c4pp_tilde' in marginalised_params)
        marg_a0   = ('a0'   in marginalised_params) or ('a0_tilde'   in marginalised_params)
        marg_a2   = ('a2'   in marginalised_params) or ('a2_tilde'   in marginalised_params)

        nuisance_const = [
                            b1, b2, bs, b3,
                            0.0 if marg_c0 else alpha0,
                            0.0 if marg_c2pp else alpha2,
                            0.0 if marg_c4pp else alpha4,
                            ctilde,
                            0.0 if marg_a0 else alphashot0,
                            0.0 if marg_a2 else alphashot2,
                            PshotP, X_FoG
                         ]

        mu_nodes, mu_weights = np.polynomial.legendre.leggauss(6)
        jac = (folps['qpar'] * folps['qperp']**2)**(-1)
        pk_const = {ell: np.zeros_like(kobs, dtype=float) for ell in ells}
        pk_derivatives = {
            ell: {
                'alpha0': np.zeros_like(kobs, dtype=float),
                'alpha2': np.zeros_like(kobs, dtype=float),
                'alpha4': np.zeros_like(kobs, dtype=float),
                'alphashot0': np.zeros_like(kobs, dtype=float),
                'alphashot2': np.zeros_like(kobs, dtype=float),
            }
            for ell in ells
        }

        for mu, weight in zip(mu_nodes, mu_weights):
            kap = self.folps_pk.k_ap(kobs, mu, folps['qpar'], folps['qperp'])
            muap = self.folps_pk.mu_ap(mu, folps['qpar'], folps['qperp'])
            table_interp = self.folps_pk.interp_table(kap, folps['table'], FOLPS.A_full_status)
            table_now_interp = self.folps_pk.interp_table(kap, folps['table_nw'], FOLPS.A_full_status)

            f0_table = table_interp[-1]
            fk = table_interp[1] * f0_table
            pkl = table_interp[0]
            pkl_now = table_now_interp[0]
            sigma2, delta_sigma2 = table_now_interp[-3:-1]
            sigma2t = (
                (1 + f0_table * muap**2 * (2 + f0_table)) * sigma2
                + (f0_table * muap)**2 * (muap**2 - 1) * delta_sigma2
            )
            exp_term = np.exp(-kap**2 * sigma2t)
            exp_term_inv = 1.0 - exp_term

            pkmu_const = jac * (
                (b1 + fk * muap**2)**2
                * (pkl_now + exp_term * (pkl - pkl_now) * (1 + kap**2 * sigma2t))
                + exp_term * self.folps_pk.get_eft_pkmu(kap, muap, nuisance_const, table_interp, self.damping)
                + exp_term_inv * self.folps_pk.get_eft_pkmu(kap, muap, nuisance_const, table_now_interp, self.damping)
            )

            k2 = kap**2
            mu2 = muap**2
            d_alpha0 = jac * (exp_term * k2 * pkl + exp_term_inv * k2 * pkl_now)
            d_alpha2 = jac * (exp_term * k2 * mu2 * pkl + exp_term_inv * k2 * mu2 * pkl_now)
            d_alpha4 = jac * (exp_term * k2 * mu2**2 * pkl + exp_term_inv * k2 * mu2**2 * pkl_now)
            d_alphashot0 = jac * PshotP
            d_alphashot2 = jac * k2 * mu2 * PshotP

            for ell in ells:
                factor = 0.5 * (2 * ell + 1) * weight * self._legendre(ell, mu)
                pk_const[ell] += factor * pkmu_const
                pk_derivatives[ell]['alpha0'] += factor * d_alpha0
                pk_derivatives[ell]['alpha2'] += factor * d_alpha2
                pk_derivatives[ell]['alpha4'] += factor * d_alpha4
                pk_derivatives[ell]['alphashot0'] += factor * d_alphashot0
                pk_derivatives[ell]['alphashot2'] += factor * d_alphashot2

        const_interp = {
            str(ell): interp1d(kobs, pk_const[ell], kind='cubic', fill_value='extrapolate')
            for ell in ells
        }

        deriv_by_alpha = {}
        for ell in ells:
            deriv_by_alpha[str(ell)] = {
                'alpha0': pk_derivatives[ell]['alpha0'],
                'alpha2': pk_derivatives[ell]['alpha2'],
                'alpha4': pk_derivatives[ell]['alpha4'],
                'alphashot0': pk_derivatives[ell]['alphashot0'],
                'alphashot2': pk_derivatives[ell]['alphashot2'],
            }

        sigma8 = folps['sigma8']
        A_AP = 1.0 / (folps['qpar'] * folps['qperp']**2)
        b1 = pars['b1']

        if 'LRG' in self.tracer:
            sigv = 150 * (10)**(1/3) * (1 + 0.8)**(1/2) / 70.
            fsat = 0.15
        elif 'QSO' in self.tracer:
            sigv = 150 * (10)**(0.7/3) * (2.4)**(1/2) / 70.
            fsat = 0.03
        else:
            sigv = 0.0
            fsat = 0.0

        if self.reparametrize:
            dc0 = 1.0 / (A_AP * sigma8**2)
            dc2pp = 1.0 / (A_AP * sigma8**2)
            dc4pp = 1.0 / (A_AP * sigma8**2)
            da0 = 1.0 / A_AP
            da2 = fsat * sigv**2 / A_AP
        else:
            dc0 = dc2pp = dc4pp = da0 = da2 = 1.0

        template_interp = {}
        for param in marginalised_params:
            template_interp[param] = {}
            for ell in const_interp:
                d = deriv_by_alpha[ell]
                if param in ['c0', 'c0_tilde']:
                    template = dc0 * (b1**2 * d['alpha0'] + b1 * f0 * d['alpha2'])
                elif param in ['c2pp', 'c2pp_tilde']:
                    template = dc2pp * (b1 * f0 * d['alpha2'] + f0**2 * d['alpha4'])
                elif param in ['c4pp', 'c4pp_tilde']:
                    template = dc4pp * (b1 * f0 * d['alpha4'])
                elif param in ['a0', 'a0_tilde']:
                    template = da0 * d['alphashot0']
                elif param in ['a2', 'a2_tilde']:
                    template = da2 * d['alphashot2']
                else:
                    raise ValueError(f'Unsupported analytically marginalised parameter: {param}')

                template_interp[param][ell] = interp1d(
                    kobs,
                    template,
                    kind='cubic',
                    fill_value='extrapolate'
                )

        return const_interp, template_interp
    #
    # [5] Compute the tree-level bispectrum multipoles
    #
    def bk_from_model(self, pars):

        folps = self._compute_folps_quantities(pars)
        if self.reparametrize:
            pars = self._apply_reparametrization(pars.copy(), folps)

        kNL = 0.3
        bpars = [
            pars['b1'],
            pars['b2'],
            2.0*pars.get('bG2', 0.0),
            pars.get('c1', 0.0)/(kNL**2),
            pars.get('c2', 0.0)/(kNL**2),
            pars.get('Bshot', 0.0) / self.mean_density,
            pars.get('Pshot', 0.0) / self.mean_density,
            pars.get('X_FoG_bk', 0.0)
        ]

        k1k2T = np.vstack([folps['k'],folps['k']]).T  # List of pairs of k. ( B = B(k1,k2) )
        f0 = folps['f0']

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

        kNL = 0.3
        bpars = [
            pars['b1'],
            pars['b2'],
            2.0*pars.get('bG2', 0.0),
            pars.get('c1', 0.0)/(kNL**2),
            pars.get('c2', 0.0)/(kNL**2),
            pars.get('Bshot', 0.0) / self.mean_density,
            pars.get('Pshot', 0.0) / self.mean_density,
            pars.get('X_FoG_bk', 0.0)
        ]

        f0 = folps['f0']

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

    def _compute_bk_vector(self, full_params):
        bk_vector = []
        # Compute bispectrum predictions
        if self.multipoles_bk:
            #k_theory = self.k_theory_window['Bk'] if self.k_theory_window['Bk'] is not None else None
            if self.k_theory_window is not None and self.k_theory_window.get('Bk') is not None:
                k_theory = self.k_theory_window['Bk']
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
                    if combined.shape[0] != self.window_matrix['Bk'][L].shape[1]:
                        raise ValueError(
                            f'Bispectrum window shape mismatch for {L}: '
                            f"window expects {self.window_matrix['Bk'][L].shape[1]} "
                            f'model values, got {combined.shape[0]}.'
                        )
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

        return bk_vector

    def compute_model_vector(self, theta):
        '''
        Compute the model predictions for the power spectrum and bispectrum based on the input parameters.

        Args:
            theta (np.ndarray): Array of parameter values sampled by the Monte-Carlo method.

        Returns:
            np.ndarray: Concatenated model predictions for the specified multipoles.
                        (to be compared directly with the concatenated data vector in the Likelihood).
        '''

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

        ''' THE bk_vector = [] BLOCK WITH BISPECTRUM PREDICTIONS HAS BEEN REPLACED BY A HELPER FUNCTION '''
        bk_vector = self._compute_bk_vector(full_params)

        theory_vector = []
        if len(pk_vector) > 0:
            theory_vector.append(pk_vector)
        if len(bk_vector) > 0:
            theory_vector.append(bk_vector)

        return np.concatenate(theory_vector).flatten()

    def compute_model_vector_am(self, theta, marginalised_params):
        """
        Compute the constant model vector and per-parameter templates used for
        analytical marginalisation over linear P(k) nuisance parameters.
        """

        if self.k_theory_window is not None and self.k_theory_window.get('Pk') is not None:
            k_theory = self.k_theory_window['Pk']
        else:
            k_theory = None

        full_params   = self.get_parameters_dictionary(theta)
        theory_vector = []

        if not self.multipoles_pk:
            warnings.warn( 'Analytical marginalisation is only done for the power spectrum. '
                           f'Requested parameters {marginalised_params} will be ignored because '
                           'this run has no P(k) multipoles; proceeding with the bispectrum only fit.',
                           RuntimeWarning
                         )
            model_vector = self.compute_model_vector(theta)
            return model_vector, np.zeros((0, len(model_vector)))

        pk_const_interp, pk_template_interp = self.calculator.pk_marginalised_from_model(
                                                            full_params,
                                                            marginalised_params,
                                                            self.multipoles_pk,
                                                            k_eval=k_theory
                                                        )

        pk_const_vector  = []
        template_vectors = {param: [] for param in marginalised_params}
        for L in self.multipoles_pk:
            k_array = k_theory if k_theory is not None else self.data[L]['k']
            pk_const_vector.append(pk_const_interp[L](k_array))
            for param in marginalised_params:
                template_vectors[param].append(pk_template_interp[param][L](k_array))

        pk_const_vector = np.concatenate(pk_const_vector)
        template_matrix = np.array([ np.concatenate(template_vectors[param])
                                     for param in marginalised_params
                                  ])

        if self.window_matrix and self.window_matrix.get('Pk') is not None:
            pk_const_vector = self.window_matrix['Pk'].dot(pk_const_vector)
            template_matrix = np.array([ self.window_matrix['Pk'].dot(template)
                                         for template in template_matrix
                                      ])

        theory_vector.append(pk_const_vector)

        bk_vector = self._compute_bk_vector(full_params)
        if len(bk_vector) > 0:
            theory_vector.append(bk_vector)
            template_matrix = np.concatenate(
                                    [ template_matrix,
                                      np.zeros((len(marginalised_params), len(bk_vector)))
                                    ],
                                    axis=1
                                )

        return np.concatenate(theory_vector).flatten(), template_matrix

###########################################################
