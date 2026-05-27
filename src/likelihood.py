from scipy.stats import norm, uniform
import pocomc as pc
import numpy as np

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
                        raise ValueError(f'Shared parameter {param} has inconsistent priors.'
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

        for tracer_name, run in self.runs.items():
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
    def __init__(self, priors_dict, model_function, debug_filename=None):
        """
            Initialise the Likelihood class.

            Args:
                priors_dict (dict): Dictionary of priors for the parameters.
                compute_model_function (callable): Function to compute the model predictions.
        """
        self.priors_dict = priors_dict
        self.model_function = model_function

        self.debug_filename = debug_filename
        if self.debug_filename is not None:
            self._debug_counter = 0
            self.debug_every = 100
            # Create/overwrite file at start of run
            with open(self.debug_filename, "w") as f:
                f.write("# theta chi2\n")

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

        # BACCO HARD PRIOR
        # the following step can be removed if the priors' range
        # are the same as the emulators (added because of bacco:
        # for some parameters, e.g. Omega_b, = omega_b/h^2 falls
        # outside [0.03,0.07])
        Omega_b = pars.get('omega_b',0.02237) / pars.get('h',0.6736)**2
        if (Omega_b < 0.03) or (Omega_b > 0.07):
            return -np.inf
        Omega_cold = ( pars.get('omega_b',0.02237) + pars.get('omega_cdm',0.120) ) / pars.get('h',0.6736)**2
        if (Omega_cold < 0.15) or (Omega_cold > 0.6):
            return -np.inf
        if ( pars.get('h',0.6736) < 0.5 ) or ( pars.get('h',0.6736) > 0.9 ):
            return -np.inf

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
