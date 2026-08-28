from scipy.stats import norm, uniform
import pocomc as pc
import numpy as np
from collections import OrderedDict

# Key to make the neutrino-mass handling consistent across the file
# pklin_emulator_jit.py uses Mnu
NEUTRINO_MASS_KEYS = ('m_ncdm', 'm_nu', 'Mnu')
# Conversion factor
NEUTRINO_MASS_TO_OMEGA_NU = 93.14

def get_neutrino_mass_for_prior(pars, default=0.06):
    for key in NEUTRINO_MASS_KEYS:
        if key in pars:
            return pars[key]
    if 'omega_nu' in pars:
        return pars['omega_nu'] * NEUTRINO_MASS_TO_OMEGA_NU
    return default

class JointLikelihood:
    def __init__(self, runs, shared_parameters, debug_filename=None):
        '''
        Joint likelihood for several independent tracers/bins.

        Args:
            runs (dict): Dictionary like
                                {
                                    "LRG1": run_lrg1,
                                    "LRG2": run_lrg2,
                                    "LRG3": run_lrg3,
                                }
                where each run is returned by build_single_tracer_run(config) (defined in inference.py).

            shared_parameters (list): Parameters sampled only once and shared by all runs,
                e.g. ['omega_cdm', 'omega_b', 'h', 'n_s', 'ln10^{10}A_s'].
                This is defined in the config.yml file for the JOINT run.
        '''

        self.runs              = runs
        self.shared_parameters = set(shared_parameters)
        self.debug_filename    = debug_filename
        self.priors_dict       = self._build_joint_priors()
        self.joint_param_names = [ param for param, prior_info in self.priors_dict.items()
                                    if prior_info['type'] != 'Fix' ]

    def _rename_single_tracer_parameter(self, run_name, param):
        '''
            Convert a single-tracer parameter name that is not shared
            among other tracers/bins for the joint fit.

            Example:
                b1_tilde -> LRG1.b1_tilde
        '''
        return f'{run_name}.{param}'

    def _build_joint_priors(self):
        '''
            Build the prior dictionary for the joint run.
            Shared parameters keep their original names.
            Non-shared parameters are renamed by tracer/bin name.
        '''
        joint_priors = {}

        for run_name, run in self.runs.items():
            local_priors = run['priors']

            for param, prior_info in local_priors.items():
                if prior_info['type'] == 'Fix':
                    continue

                if param in self.shared_parameters:
                    if param not in joint_priors:
                        joint_priors[param] = prior_info
                    elif joint_priors[param] != prior_info:
                        raise ValueError(f'Shared parameter {param} has inconsistent priors. '
                                         f'Problem found while processing {run_name}.'
                                        )
                else:
                    joint_name = self._rename_single_tracer_parameter(run_name, param)
                    if joint_name in joint_priors:
                        raise ValueError(f'Duplicated joint parameter name: {joint_name}.')
                    joint_priors[joint_name] = prior_info

        return joint_priors


    def initialise_prior(self):
        prior_list = []

        for param in self.joint_param_names:
            prior_info = self.priors_dict[param]

            if prior_info['type'] in ['Uni', 'Uniform']:
                lower, upper = prior_info["lim"]
                prior_list.append(uniform(lower, upper - lower))

            elif prior_info['type'] in ['Gauss', 'Gaussian']:
                mean, std = prior_info["lim"]
                prior_list.append(norm(mean, std))

            else:
                raise ValueError(f"Unknown prior type: {prior_info['type']}")

        return pc.Prior(prior_list)

    def ln_prob(self, theta, data_=None, icov_=None):
        # Convert sampled theta array into named joint parameters.
        joint_params = {
            name: theta[i]
            for i, name in enumerate(self.joint_param_names)
        }

        total_lnprob = 0.0
        folps_cache = OrderedDict()

        for tracer_name, run in self.runs.items():
            calculator = run['model_function'].calculator
            if hasattr(calculator, 'set_folps_cache'):
                calculator.set_folps_cache(folps_cache)

            local_theta = self.make_local_theta(
                tracer_name,
                run,
                joint_params,
            )

            lnprob = run['likelihood'].ln_prob(
                local_theta,
                run['full_data'],
                run['inv_cov'],
            )

            if not np.isfinite(lnprob):
                return -np.inf

            total_lnprob += lnprob

        return total_lnprob

    def make_local_theta(self, tracer_name, run, joint_params):
        '''
        Build the theta vector expected by one existing single-tracer likelihood.

        Example:
            joint_params['omega_cdm']      -> local 'omega_cdm'
            joint_params['LRG1.b1_tilde']  -> local 'b1_tilde'
        '''
        local_theta = []

        for param, prior_info in run['priors'].items():
            if prior_info['type'] == 'Fix':
                continue

            if param in self.shared_parameters:
                local_theta.append(joint_params[param])
            else:
                joint_name = f'{tracer_name}.{param}'
                local_theta.append(joint_params[joint_name])

        return np.array(local_theta)

        #
        # TO BE IMPLEMENTED: DEBUGGING FILE
        # ...

class Likelihood:
    def __init__(self, priors_dict, model_function, emulator,
                    analytic_marginalisation=None, all_priors_dict=None,
                    debug_filename=None):
        """
            Initialise the Likelihood class.

            Args:
                priors_dict (dict): Dictionary of priors for the parameters.
                compute_model_function (callable): Function to compute the model predictions.
        """
        self.priors_dict = priors_dict
        self.model_function = model_function
        self.emulator = emulator
        self.debug_filename = debug_filename
        self.all_priors_dict = all_priors_dict or priors_dict
        # Analytical marginalisation related
        # Keep only marginalised parameter names that exist in the full prior dictionary.
        self.analytic_marginalisation = []
        if analytic_marginalisation is not None:
            for param in analytic_marginalisation:
                # Checking if an AM parameter is not present in the tracer's prior list.
                if param in self.all_priors_dict:
                    self.analytic_marginalisation.append(param)
        self.am_means, self.am_sigmas = self._initialise_am_priors()
        self._failure_counter = 0

        '''
        self.debug_filename = debug_filename
        if self.debug_filename is not None:
            self._debug_counter = 0
            self.debug_every = 100
            # Create/overwrite file at start of run
            with open(self.debug_filename, "w") as f:
                f.write("# theta chi2\n")
        '''

    def _initialise_am_priors(self):
        means = []
        sigmas = []
        for param in self.analytic_marginalisation:
            prior_info = self.all_priors_dict[param]
            if prior_info['type'] not in ['Gauss', 'Gaussian']:
                raise ValueError(
                    f'Analytically marginalised parameter {param} must have a Gaussian prior.'
                )
            means.append(prior_info['lim'][0])
            sigmas.append(prior_info['lim'][1])
        return np.array(means), np.array(sigmas)

    def initialise_prior(self):
        """
        Initialise the prior distributions based on the priors dictionary.

        Returns:
            pc.Prior: A Prior object from the pocomc library.
        """
        prior_list = []
        for param, prior_info in self.priors_dict.items():
            if prior_info['type'] == 'Fix':
                # Skip fixed parameters
                continue
            if prior_info['type'] in ['Uni', 'Uniform']:
                # Uniform distribution
                lower, upper = prior_info['lim']
                prior_list.append(uniform(lower, upper - lower))
            elif prior_info['type'] in ['Gauss', 'Gaussian']:
                # Gaussian distribution
                mean, std = prior_info['lim'][0], prior_info['lim'][1]
                prior_list.append(norm(mean, std))
            else:
                raise ValueError(f"Unknown prior type: {prior_info['type']}")
        return pc.Prior(prior_list)

    def ln_prob(self, theta, data_, icov_):
        """
        Compute the log-probability for the given parameters.

        Args:
            theta (np.ndarray): Array of parameter values.
            data_ (np.ndarray): Observed data vector.
            icov_ (np.ndarray): Inverse covariance matrix.

        Returns:
            float: Log-probability.
        """

        # Convert theta (list) to dictionary
        pars = self.model_function.get_parameters_dictionary(theta)

        if self.emulator:
            if self.emulator == 'bacco':
                # BACCO HARD PRIOR
                Omega_b = pars.get('omega_b',0.02237) / pars.get('h',0.6736)**2
                if (Omega_b < 0.03) or (Omega_b > 0.07):
                    return -np.inf

                Omega_cold = ( pars.get('omega_b',0.02237) + pars.get('omega_cdm',0.120) ) / pars.get('h',0.6736)**2
                if (Omega_cold < 0.15) or (Omega_cold > 0.6):
                    return -np.inf
                if ( pars.get('h',0.6736) < 0.5 ) or ( pars.get('h',0.6736) > 0.9 ):
                    return -np.inf

                w0 = pars.get('w0', -1.0)
                wa = pars.get('wa', 0.0)
                if ( w0 < -1.3 or w0 > -0.7 or wa < -0.5 or wa > 0.5 or w0 + wa > 0.0 ):
                    return -np.inf
            elif self.emulator == 'jaxmapse':
                # Early matter domination - also in bacco
                if pars.get('w0',-1.0) + pars.get('wa',0.0) > 0.0:
                    return -np.inf
            else:
                raise ValueError(f'Unknown linear power spectrum emulator: {self.emulator}')

        if get_neutrino_mass_for_prior(pars) <= 0.0:
            return -np.inf

        '''
        try:
            m = self.model_function.compute_model_vector(theta)
        except (ValueError, FloatingPointError):
            return -np.inf

        diff = m - data_
        chi2_try = np.dot(diff.T, np.dot(icov_, diff))

        if self.debug_filename is not None:
            self._debug_counter += 1
            if self._debug_counter % self.debug_every == 0:
                with open(self.debug_filename, "a") as f:
                    f.write(
                        " ".join(map(str, theta)) + f" {chi2_try}\n"
                    )
        '''

        if self.analytic_marginalisation:
            try:
                m, templates = self.model_function.compute_model_vector_am(
                    theta,
                    self.analytic_marginalisation
                )
            except Exception as exc:
                self._failure_counter += 1
                if self._failure_counter <= 3:
                    print(f'Analytical marginalisation failed: {type(exc).__name__}: {exc}')
                return -np.inf
            if not (np.all(np.isfinite(m)) and np.all(np.isfinite(templates))):
                return -np.inf

            diff = m - data_
            if templates.shape[0] == 0:
                chi2_try = np.dot(diff.T, np.dot(icov_, diff))
                return -0.5 * chi2_try

            F0   = (
                    np.dot(diff.T, np.dot(icov_, diff))
                    + np.sum((self.am_means / self.am_sigmas) ** 2)
                   )
            F1i  = (
                    -np.einsum("ij,jk,k->i", templates, icov_, diff)
                    + self.am_means / self.am_sigmas**2
                   )
            F2ij = (
                    np.einsum("ik,kp,jp->ij", templates, icov_, templates)
                    + np.diag(1.0 / self.am_sigmas**2)
                   )

            if not (  np.isfinite(F0)
                      and np.all(np.isfinite(F1i))
                      and np.all(np.isfinite(F2ij)) ):
                return -np.inf

            sign, logdet = np.linalg.slogdet(F2ij)
            if sign <= 0 or not np.isfinite(logdet):
                return -np.inf
            try:
                self.marg_pars_means_raw = np.linalg.solve(F2ij, F1i)
            except np.linalg.LinAlgError:
                return -np.inf

            chi2_try = F0 - np.dot(F1i.T, self.marg_pars_means_raw) + logdet
            if not np.isfinite(chi2_try):
                return -np.inf

            self.marg_pars_means_dict = { param: self.marg_pars_means_raw[i]
                                          for i, param in enumerate(self.analytic_marginalisation)
                                        }
        else:
            m = self.model_function.compute_model_vector(theta)
            diff = m - data_
            chi2_try = np.dot(diff.T, np.dot(icov_, diff))

        if self.debug_filename is not None:
            self._debug_counter += 1
            if self._debug_counter % self.debug_every == 0:
                with open(self.debug_filename, "a") as f:
                    f.write(
                        " ".join(map(str, theta)) + f" {chi2_try}\n"
                    )

        return -0.5 * chi2_try
