from scipy.stats import norm, uniform
import pocomc as pc
import numpy as np

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
