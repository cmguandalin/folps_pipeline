import numpy as np
import argparse
import os,sys
import glob, re
import yaml
from time import time
import pocomc as pc
import data_loader as dload
import covariance_loader as cload
import likelihood as clike
import model_APpar as model
#import model_window as model
#import model as model
from datetime import datetime
#import multiprocessing as mp
import multiprocess as mp

ctx = mp.get_context('fork')

#os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

sys.stderr = sys.stdout

# Global variables (avoid passing them as arguments) - this should make the sampler faster in slurm
global global_full_data, global_inv_cov, global_likelihood
global_full_data = None
global_inv_cov = None
global_likelihood = None

def likelihood_wrapper(theta):
    return global_likelihood.ln_prob(theta, global_full_data, global_inv_cov)

if __name__ == '__main__':

    time_i = time()

    ##############################
    # LOADING CONFIGURATION FILE #
    ##############################
    parser = argparse.ArgumentParser(description='Configuration file to load')
    parser.add_argument('-config', '-c', '-C', type=str, help='config file', required=True,dest='config')
    parser.add_argument('-ncpus', type=int, help='Number of CPUs in a PC to use.', required=False, default=1)
    cmdline = parser.parse_args()

    print(f'Using {cmdline.config}')

    with open(cmdline.config, 'r') as file:
        config = yaml.safe_load(file)

    # Get the path for data and covariance files
    data_path = config['data_path']
    # Get the files names
    data_files = config['data_files']
    # Get the covariance path
    cov_path = config.get('cov_path')
    if not cov_path:
        # If only the file name is given, it's assumed the data path is the same as the covariance path
        cov_file = config['cov_file']
        cov_path = data_path+cov_file
    # Number of mocks for hartlap correction
    number_of_mocks = config['number_of_mocks']
    # Rescaling factor for the covariance
    rescale = config['rescale']
    # Minimum and maximum wavenumbers to consider
    k_edges = config['k_edges']
    # The priors
    priors = config['prior']
    reparametrize = config['reparametrize']
    # The files
    path_to_save = config['path_to_save']
    file_name = config['file_name']

    # For the emulator
    multipoles   = list(data_files.keys())
    mean_density = config['mean_density']
    redshift     = config['redshift']
    backend      = config['backend']

    if backend == 'emulator':
        cache_path = config['cache_path']
    else:
        theory_model  = config.get('theory_model','EFT')
        damping       = config.get('damping', None)
        use_TNS_model = config.get('TNS', False)
        AP            = config.get('AP', True)

    #######################
    # CLEANING PARAMETERS #
    #######################

    # # Iterate over a copy of the dictionary to avoid modifying it while iterating
    # for param, prior_info in list(priors.items()):
    #    if prior_info['type'] == 'Fix':
    #        del priors[param]

    parameters_to_be_varied = priors.copy()
    for param, prior_info in list(priors.items()):
        if prior_info['type'] == 'Fix':
            del parameters_to_be_varied[param]

    #############
    # LOAD DATA #
    #############
    loader = dload.DataLoader(data_path,data_files,multipoles)
    loader.load_data(k_edges)
    data = loader.get_data()
    full_k, full_data = loader.get_concatenated_data()

    ###################
    # LOAD COVARIANCE #
    ###################
    cov_loader = cload.CovarianceLoader(cov_path, multipoles, k_edges, rescale)
    cov_loader.process()
    covariance = cov_loader.get_covariance()

    # Apply Hartlap correction factor and invert covariance
    hartlap = (number_of_mocks - len(full_data) - 2) / (number_of_mocks - 1)
    inv_cov = hartlap * np.linalg.inv(covariance)

    ########################
    # LOAD WINDOW FUNCTION #
    ########################

    if 'window_file' in config:
        # Load window matrix once
        data = np.load(config['window_file'],allow_pickle=True).item()

        value    = data['value']
        xin      = data['xin']
        xout     = data['xout']
        ells_in  = np.array([proj['ell'] for proj in data['projsin']])
        ells_out = np.array([proj['ell'] for proj in data['projsout']])

        xin_flat  = np.array(xin).flatten()
        xout_flat = np.array(xout).flatten()

        # IN  k-range selection
        k_min_in = 0.001
        k_max_in = 0.35
        print(f'Fixing window convolution (hard-coded) limits to: k_min^IN={k_min_in}, k_max^IN={k_max_in}')
        mask_in  = (xin_flat >= k_min_in) & (xin_flat <= k_max_in)
        xin_flat = xin_flat[mask_in]
        value    = value[mask_in,:]

        # OUT k-range selection
        kmin_data = min([v[0] for v in k_edges.values()])
        kmax_data = max([v[1] for v in k_edges.values()])
        print(f'Selecting k_obs limits: k_min={kmin_data}, k_max={kmax_data}')
        mask_out  = (xout_flat >= kmin_data) & (xout_flat <= kmax_data)
        xout_flat = xout_flat[mask_out]
        value     = value[:, mask_out]

        # OUT multipole selection
        print(f'Selecting multipoles for analysis: {multipoles}')
        Nout_orig = 3
        Nin_total = value.shape[0]

        mask_ell_out   = np.isin(ells_out, list(map(int, multipoles)))
        value_reshaped = value.reshape(value.shape[0], Nout_orig, int(value.shape[1]/Nout_orig))
        value_reshaped_selected = value_reshaped[:, mask_ell_out, :]
        value = value_reshaped_selected.reshape(Nin_total, -1)

        k_theory_window = np.array(xin_flat)
        k_obs_window    = np.array(xout_flat)
        wvalue          = value.T

        multipoles_for_convolution = [str(ell_) for ell_ in ells_in]

    else:
        wvalue = None

    ################
    # MODEL VECTOR #
    ################
    # Initialise emulators
    if wvalue is not None:
        print('Convolving with window')
        # Change the redshift to the effective one from the window
        redshift = data['attrs']['zeff']
        print(f'Updating redshift to the window effective z={redshift}.')

        if backend == 'folps':
            calculator = model.FOLPSCalculator(
                multipoles_for_convolution,
                mean_density,
                redshift,
                model=theory_model,
                damping=damping,
                use_TNS_model=use_TNS_model,
                AP=AP,
                reparametrize=reparametrize
            )
        else:
            # Use the emulator
            calculator = model.BICKERCalculator(
                multipoles_for_convolution,
                mean_density,
                redshift,
                cache_path,
                fixed_params=None,
                rescale_kernels=True,
                ordering=1
            )

        model_function = model.ModellingFunction(
                                priors,
                                data,
                                calculator,
                                multipoles_for_convolution,
                                window_matrix=wvalue,
                                k_theory_window=k_theory_window
                            )
    else:
        if backend == 'folps':
            calculator = model.FOLPSCalculator(
                multipoles,
                mean_density,
                redshift,
                model=theory_model,
                damping=damping,
                use_TNS_model=use_TNS_model,
                AP=AP,
                reparametrize=reparametrize
            )
        else:
            # Use the emulator
            calculator = model.BICKERCalculator(
                multipoles,
                mean_density,
                redshift,
                cache_path,
                fixed_params=None,
                rescale_kernels=True,
                ordering=1
            )

        model_function = model.ModellingFunction(priors, data, calculator, multipoles)

    ##############
    # LIKELIHOOD #
    ##############
    likelihood = clike.Likelihood(priors, model_function)#, debug_filename="/Users/austerlitz/folps/pipeline/test_debug_2.txt")
    prior = likelihood.initialise_prior()

    # Assign to global variables
    global_full_data  = full_data
    global_inv_cov    = inv_cov
    global_likelihood = likelihood

    ##################
    # START SAMPLING #
    ##################

    # number of effective particles
    neff = 4000
    # number of effectively independent samples
    ntot = 20000

    if cmdline.ncpus is not None:
        ncpus = int(cmdline.ncpus)
    else:
        ncpus = 1

    print(f'Starting sampling at {datetime.now()} with {ncpus} CPUs. \n')

    # Check for checkpoints

    final_state = os.path.join(path_to_save, f"{file_name}_final.state")

    if os.path.exists(final_state):
        resume_file = final_state
    else:
        state_files = sorted(glob.glob(os.path.join(path_to_save, f"{file_name}_*.state")))

    def extract_iteration(filename):
        match = re.search(r'_(\d+)\.state$', filename)
        return int(match.group(1)) if match else -1

    state_files = sorted(state_files, key=extract_iteration)

    resume_file = None
    if state_files:
        resume_file = state_files[-1]
        print(f"Resuming from {resume_file}")

    if ncpus > 1:
        with ctx.Pool(ncpus) as pool:
            sampler = pc.Sampler(
                prior=prior,
                likelihood=likelihood_wrapper,
                n_effective=neff,
                pool=pool,
                output_dir=path_to_save,
                output_label=file_name
            )
            sampler.run(n_total=ntot, progress=True, save_every=25, resume_state_path=resume_file)

    else:
        sampler = pc.Sampler(
            prior=prior,
            likelihood=likelihood_wrapper,
            n_effective=neff,
            output_dir=path_to_save,
            output_label=file_name
        )
        sampler.run(n_total=ntot, progress=True, save_every=25, resume_state_path=resume_file)

    samples, weights, logl, logp = sampler.posterior()

    print(f"Sampling ended at: {datetime.now()}")

    # Save results
    os.makedirs(path_to_save, exist_ok=True)

    print(f"Results saved to {os.path.join(path_to_save, file_name + '.npy')}")

    results = {}
    results['priors'] = priors
    results['samples'] = samples
    results['weights'] = weights
    results['logl'] = logl
    results['logp'] = logp

    np.save(os.path.join(path_to_save, file_name + '.npy'), results)

    time_f = time()

    print('Sampling efficiency:', sampler.results["efficiency"])
    print('Time to estimate (in minutes):', np.round((time_f-time_i)/60,2))
