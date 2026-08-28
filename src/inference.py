import os,sys

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ.setdefault(
    "XLA_FLAGS",
    "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
)

import numpy as np
import argparse
import glob
import re
import yaml
import h5py
import time
import pocomc as pc
import data_loader as dload
import covariance_loader as cload
import likelihood_neutrinos as clike
import model_neutrinos as model
from datetime import datetime
import multiprocess as mp

ctx = mp.get_context('spawn')

sys.stderr = sys.stdout

# Global variables (avoid passing them as arguments) - this should make the sampler faster in slurm
global global_full_data, global_inv_cov, global_likelihood
global_full_data  = None
global_inv_cov    = None
global_likelihood = None

############################################################################
# ~ FOR OPTIONAL DEBUGGING ~
# It verifies that the sampler is reaching the likelihood function,
# without flooding the logs. Since likelihood is called thousands of times,
# it will let us know that the sampler reached the first 5 calls, tracked by
# the counter "global_likelihood_calls"; it appears in the likelihood_wrapper
# below. To use this, submit the job as
# $ DEBUG_LIKELIHOOD_CALLS=1 python src/inference.py ...
global_likelihood_calls = 0
debug_likelihood_calls = bool(os.environ.get('DEBUG_LIKELIHOOD_CALLS'))
# END
############################################################################

def likelihood_wrapper(theta):
    global global_likelihood_calls
    global_likelihood_calls += 1
    if debug_likelihood_calls and global_likelihood_calls <= 5:
        print(f'likelihood_wrapper call {global_likelihood_calls}')
    return global_likelihood.ln_prob(theta, global_full_data, global_inv_cov)

def initialise_worker(likelihood, full_data, inv_cov):
    global global_full_data, global_inv_cov, global_likelihood
    global_full_data  = full_data
    global_inv_cov    = inv_cov
    global_likelihood = likelihood

# Using the following function to bypass the emulator given in individual config files 
# if running a joint analysis. That is, if an emulator is provided in the joint yml file,
# it overwrites the emulator passed in the individual files, if it exisits.  
def overwrite_individual_emulators(single_config, joint_config):
    for key in ['emulator', 'jaxmapse_plin_path', 'jaxmapse_pnw_path', 'folps_nk']:
        if key in joint_config:
            single_config[key] = joint_config[key]
    return single_config

# Useful for debugging processes ran with spawn
import builtins
_REAL_PRINT = builtins.print
def quiet_worker_output():
    devnull = open(os.devnull, "w")
    sys.stdout = devnull
    sys.stderr = devnull

def initialise_worker_from_config( run_builder, config_, analytic_marginalisation_,
                                   verbose=False):
    if not verbose:
        quiet_worker_output()

    global global_full_data, global_inv_cov, global_likelihood

    if config_.get('joint'):
        runs = {}

        for name, path in config_['joint']['configs'].items():
            single_config = load_yaml(path)
            single_config = overwrite_individual_emulators(single_config, config_)
            runs[name]    = run_builder(single_config, analytic_marginalisation_)

        global_likelihood = clike.JointLikelihood( runs = runs,
                            shared_parameters = config_['shared_parameters'],
                            )

        global_full_data = None
        global_inv_cov   = None

    else:
        run = run_builder(config_)
        global_likelihood = run['likelihood']
        global_full_data  = run['full_data']
        global_inv_cov    = run['inv_cov']

def load_yaml(path_):
    with open(path_, 'r') as file_:
        config_ = yaml.safe_load(file_)
    return config_

############################################################################
# ~ FOR OPTIONAL DEBUGGING ~
# It prints the time to compute one likelihood call, for a random value
# of the parameters being varied in the priors.
# It's not used anywhere in the code, only around the line below:
# print('Checking one reference likelihood call before starting pocoMC...')
def get_reference_theta(priors):
    theta = []
    for param, prior_info in priors.items():
        if prior_info['type'] == 'Fix':
            continue

        if param in ['w0', 'w0_fld']:
            theta.append(-1.0)
            continue

        if param in ['wa', 'wa_fld']:
            theta.append(0.0)
            continue

        if prior_info['type'] in ['Uni', 'Uniform']:
            lower, upper = prior_info['lim']
            theta.append(0.5 * (lower + upper))
        elif prior_info['type'] in ['Gauss', 'Gaussian']:
            theta.append(prior_info['lim'][0])
        else:
            raise ValueError(f"Unknown prior type: {prior_info['type']}")
    return np.array(theta)

def print_reference_q_sigma8(run, theta_local, label=''):
    model_function = run['model_function']
    calculator = model_function.calculator
    pars = model_function.get_parameters_dictionary(theta_local)

    z_original = calculator.zcen
    expfactor_original = getattr(calculator, 'expfactor', None)

    try:
        for z in [0.0, z_original]:
            calculator.zcen = float(z)
            if hasattr(calculator, 'expfactor'):
                calculator.expfactor = 1.0 / (1.0 + float(z))

            aux = calculator._get_linear_pk(pars)
            sigma8 = aux['sigma8']

            if z == 0.0:
                qpar, qperp = 1.0, 1.0
            elif aux.get('qpar') is not None:
                qpar, qperp = aux['qpar'], aux['qperp']
            else:
                folps = calculator._compute_folps_quantities(pars)
                qpar, qperp = folps['qpar'], folps['qperp']
                sigma8 = folps['sigma8']

            prefix = f'[{label}] ' if label else ''
            print(
                f'{prefix}theta_ref diagnostics at z = {z:.6g}: '
                f'qpar = {qpar:.10g}, qperp = {qperp:.10g}, sigma8 = {sigma8:.10g}'
            )

    finally:
        calculator.zcen = z_original
        if expfactor_original is not None:
            calculator.expfactor = expfactor_original
#
# END
############################################################################

def uses_jaxmapse_emulator(config_):
    if config_.get('joint'):
        return any(
            load_yaml(path).get('emulator') == 'jaxmapse'
            for path in config_['joint']['configs'].values()
        )
    return config_.get('emulator') == 'jaxmapse'

if __name__ == '__main__':

    time_i = time.time()

    def build_single_tracer_run(config, analytic_marginalisation=None):

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
        # Rescaling factor for the covariance (e.g., rescale = 0.25 => Cov = Cov*0.25)
        rescale = config['rescale']
        # Minimum and maximum wavenumbers to consider
        k_edges = config['k_edges']
        # Multipoles for the analysis
        multipoles = list(data_files.keys())
        # Tracer analysed
        tracer = config['tracer']

        # The priors
        priors = config['prior']
        # Parameters to be analytically marginalised over
        analytic_marginalisation = config.get(
            'analytic_marginalisation',
            analytic_marginalisation or []
        )
        # Remove parameters that are in the analytical marginalisation, but not in the priors dictionary.
        # This is useful for joint runs, where the joint config might have an extra parameter that could
        # be fixed for a given tracer.
        analytic_marginalisation = [
            param for param in analytic_marginalisation
            if param in priors
        ]
        # The actual prior's dictionary passed to the sampler. 
        # It contains everything but the analytically marginalised parameters.
        # So it enters the model_function instead of the priors, as in the versions before AM
        sampled_priors = {
            param: prior_info
            for param, prior_info in priors.items()
            if param not in analytic_marginalisation
        }

        reparametrize = config['reparametrize']
        # Saving options
        path_to_save = config['path_to_save']
        file_name    = config['file_name']
        # Sample related things
        mean_density = config['mean_density']
        redshift     = config['redshift']

        # Removed the emulator option. Only folps now.
        backend = config['backend']
        if backend == 'bicker':
            cache_path = config['cache_path']
        else:
            theory_model  = config.get('theory_model','EFT')
            damping       = config.get('damping', None)
            use_TNS_model = config.get('TNS', False)
            AP            = config.get('AP', True)
            emulator      = config.get('emulator', 'bacco')
            jaxmapse_plin_path = config.get('jaxmapse_plin_path')
            jaxmapse_pnw_path  = config.get('jaxmapse_pnw_path')
            folps_nk      = config.get('folps_nk', 1000)

        #######################
        # CLEANING PARAMETERS #
        #######################
        # Iterate over a copy of the dictionary to avoid modifying it while iterating
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
        #print(f'hartlap={hartlap}')
        inv_cov = hartlap * np.linalg.inv(covariance)

        #########################
        # LOAD WINDOW FUNCTIONS #
        #########################
        #
        # POWER SPECTRUM WINDOW MATRIX
        if 'window_file_P' in config:
            wcmat_P = np.load(config['window_file_P'],allow_pickle=True).item()

            value    = wcmat_P['value']
            xin      = wcmat_P['xin']
            xout     = wcmat_P['xout']
            ells_in  = np.array([proj['ell'] for proj in wcmat_P['projsin']])
            ells_out = np.array([proj['ell'] for proj in wcmat_P['projsout']])

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
            multipoles_for_convolution = {}
            multipoles_for_convolution['Pk'] = [str(ell_) for ell_ in ells_in]

            print(f'Selecting multipoles for analysis: {multipoles}')
            Nout_orig = 3
            Nin_total = value.shape[0]

            mask_ell_out   = np.isin(ells_out, list(map(int, multipoles)))
            value_reshaped = value.reshape(value.shape[0], Nout_orig, int(value.shape[1]/Nout_orig))
            value_reshaped_selected = value_reshaped[:, mask_ell_out, :]
            value = value_reshaped_selected.reshape(Nin_total, -1)

            k_theory_window = {}
            k_theory_window['Pk'] = xin_flat.reshape(np.array(xin).shape[0], int(len(xin_flat)/Nout_orig))[0]
            window_matrix       = {}
            window_matrix['Pk'] = value.T
        else:
            window_matrix = None
        #
        # BISPECTRUM WINDOW MATRIX
        #
        if 'window_file_B' in config:
            # GCcombined window multipoles for convolution
            window_file_B = config['window_file_B']

            if 'window_matrix' not in locals():
                window_matrix = {}
            if 'k_theory_window' not in locals():
                k_theory_window = {}
            if 'multipoles_for_convolution' not in locals():
                multipoles_for_convolution = {}

            window_matrix['Bk'] = {}
            multipoles_for_convolution['Bk'] = {}
            for L in multipoles:
                if len(L)==3:
                    with h5py.File(window_file_B[L], 'r') as f:
                        window_matrix['Bk'][L] = f['wcmat'][:]
                        k_theory_window['Bk'] = f['k_input'][:]
                        l1l2L_input = f['l1l2L_input'][:]
                        multipoles_for_convolution['Bk'][L] = [l1l2L.decode('utf-8') for l1l2L in l1l2L_input]
            f.close()

        ################
        # MODEL VECTOR #
        ################
        #
        # INITIALISE CALCULATORS
        #
        if window_matrix is not None:
            print('Model will be convolved with window.')
            # Change the redshift to the effective one from the window
            redshift = wcmat_P['attrs']['zeff']
            print(f'Updating redshift to z_eff = {redshift}.\n')

            if backend == 'folps':
                calculator = model.FOLPSCalculator(
                    mean_density,
                    redshift,
                    tracer,
                    model=theory_model,
                    damping=damping,
                    use_TNS_model=use_TNS_model,
                    AP=AP,
                    reparametrize=reparametrize,
                    emulator=emulator,
                    jaxmapse_plin_path=jaxmapse_plin_path,
                    jaxmapse_pnw_path=jaxmapse_pnw_path,
                    folps_nk=folps_nk
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
                                    sampled_priors,
                                    data,
                                    calculator,
                                    multipoles_for_convolution,
                                    window_matrix=window_matrix,
                                    k_theory_window=k_theory_window
                                )
        else:
            if backend == 'folps':
                calculator = model.FOLPSCalculator(
                    mean_density,
                    redshift,
                    tracer,
                    model=theory_model,
                    damping=damping,
                    use_TNS_model=use_TNS_model,
                    AP=AP,
                    reparametrize=reparametrize,
                    emulator=emulator,
                    jaxmapse_plin_path=jaxmapse_plin_path,
                    jaxmapse_pnw_path=jaxmapse_pnw_path,
                    folps_nk=folps_nk
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

            model_function = model.ModellingFunction(sampled_priors, data, calculator, multipoles)

        ##############
        # LIKELIHOOD #
        ##############
        likelihood_single = clike.Likelihood( sampled_priors, model_function, emulator,
                                              analytic_marginalisation=analytic_marginalisation,
                                              all_priors_dict=priors
                                            ) #, debug_filename="/Users/austerlitz/folps/pipeline/test_debug_2.txt")
        prior = likelihood_single.initialise_prior()

        return {
                'priors': sampled_priors,
                'all_priors': priors,
                'analytic_marginalisation': analytic_marginalisation,
                'data': data,
                'full_data': full_data,
                'covariance': covariance,
                'inv_cov': inv_cov,
                'model_function': model_function,
                'likelihood': likelihood_single,
                'multipoles': multipoles,
                'tracer': tracer,
                'redshift': redshift,
                }

    ##############################
    # LOADING CONFIGURATION FILE #
    ##############################
    parser = argparse.ArgumentParser(description='Configuration file to load')
    parser.add_argument('-config', '-c', '-C', type=str, help='config file', required=True,dest='config')
    parser.add_argument('-ncpus', type=int, help='Number of CPUs in a PC to use.', required=False, default=1)
    parser.add_argument('-neff', type=int, help='Number of effective particles.', required=False, default=4000)
    parser.add_argument('-ntot', type=int, help='Number of total/effectively independent samples.', required=False, default=20000)
    parser.add_argument('-nstates', type=int, help='Save every N states.', required=False, default=20)

    cmdline = parser.parse_args()

    print(f'Using {cmdline.config} \n')

    with open(cmdline.config, 'r') as file:
        config = yaml.safe_load(file)

    path_to_save = config['path_to_save']
    file_name    = config['file_name']

    # Added: overwrite emulators of individual runs 
    #        if an emulator is provided in the joint config file.
    if config.get('joint'):
        runs = {}
        analytic_marginalisation = config.get('analytic_marginalisation', [])
        for name, path in config['joint']['configs'].items():
            print('\n')
            print(f'Tracer: {name}')
            single_config = load_yaml(path)
            single_config = overwrite_individual_emulators(single_config, config)
            runs[name]    = build_single_tracer_run(single_config,
                                analytic_marginalisation=analytic_marginalisation)

        likelihood = clike.JointLikelihood( runs=runs,
                                            shared_parameters=config['shared_parameters'],
                                          )

        # pocoMC priors: exclusive for the likelihood analysis
        prior = likelihood.initialise_prior()
        # priors dictionary to keep track of parameters in the .npy output
        priors = likelihood.priors_dict

        global_full_data  = None # These are set to None because they are taken from the individual runs
        global_inv_cov    = None # These are set to None because they are taken from the individual runs
        global_likelihood = likelihood

    else:
        # SINGLE TRACER RUN
        analytic_marginalisation = config.get('analytic_marginalisation', [])
        run = build_single_tracer_run(config,analytic_marginalisation=analytic_marginalisation)

        likelihood = run['likelihood']
        # pocoMC priors: exclusive for the likelihood analysis
        prior = likelihood.initialise_prior()
        # priors dictionary to keep track of parameters in the .npy output
        priors = run['priors']

        global_full_data  = run['full_data']
        global_inv_cov    = run['inv_cov']
        global_likelihood = likelihood


    ##################
    # START SAMPLING #
    ##################
    print('\n')
    print('Priors:\n', priors)
    print('\n')

    ####################################################
    # ADDED FOR DEBUGGING
    print('Checking one reference likelihood call before starting pocoMC...')
    theta_ref = get_reference_theta(priors)
    '''
    if isinstance(likelihood, clike.JointLikelihood):
        joint_params = {
            name: theta_ref[i]
            for i, name in enumerate(likelihood.joint_param_names)
        }

        for tracer_name, run in likelihood.runs.items():
            theta_local = likelihood.make_local_theta(
                tracer_name,
                run,
                joint_params
            )
            print_reference_q_sigma8(run, theta_local, label=tracer_name)
    else:
        print_reference_q_sigma8(run, theta_ref)
    '''
    time_i = time.time()
    lnprob_ref = likelihood_wrapper(theta_ref)
    time_f = time.time()
    print(f'Reference lnprob = {lnprob_ref} (computed in {np.round(time_f-time_i, 2)} s)')
    if not np.isfinite(lnprob_ref):
        print('WARNING: reference likelihood is not finite. If pocoMC remains at iteration 0, inspect the analytical marginalisation failure printed above.')


    print('\nTiming repeated reference likelihood calls...')
    for i in range(5):
        t0 = time.perf_counter()
        lp = global_likelihood.ln_prob(theta_ref, global_full_data, global_inv_cov)
        t1 = time.perf_counter()
        print(f'  call {i}: lnprob = {lp:.8g}, time = {t1 - t0:.4f} s')
    print()
    # END
    ####################################################

    # number of effective partciles
    neff = int(cmdline.neff)
    # number of effectively independent samples
    ntot = int(cmdline.ntot)
    # save every nstates
    nstates = int(cmdline.nstates)

    if cmdline.ncpus is not None:
        ncpus = int(cmdline.ncpus)
    else:
        ncpus = 1

    print(f'Starting sampling at {datetime.now()} with {ncpus} CPUs, neff = {neff} and ntot = {ntot}. \n')

    #########################
    # Looking for checkpoints
    #def extract_iteration(filename):
    #    match = re.search(r'_(\d+)\.state$', os.path.basename(filename))
    #    return int(match.group(1)) if match else -1
    def extract_iteration(filename):
        basename = os.path.basename(filename)
        pattern = rf'^{re.escape(file_name)}_(\d+)\.state$'
        match = re.match(pattern, basename)
        return int(match.group(1)) if match else -1

    resume_file = None

    final_state = os.path.join(path_to_save, f'{file_name}_final.state')
    if os.path.exists(final_state):
        resume_file = final_state
    else:
        state_files = glob.glob(os.path.join(path_to_save, f'{file_name}_*.state'))
        state_files = [ f for f in state_files
                        if extract_iteration(f) >= 0
                      ]

        if state_files:
            resume_file = max(state_files, key=extract_iteration)

    if resume_file is not None:
        print(f'Resuming run from existing state: {resume_file}')
    else:
        print('No checkpoint found; starting fresh.')
    #
    ########################

    if ncpus > 1:
        with ctx.Pool( ncpus,
                       initializer = initialise_worker_from_config,
                       initargs    = (build_single_tracer_run, config, analytic_marginalisation)
                     ) as pool:
        #with mp.Pool(ncpus) as pool:
            sampler = pc.Sampler(
                                    prior        = prior,
                                    likelihood   = likelihood_wrapper,
                                    n_effective  = neff,
                                    pool         = pool,
                                    output_dir   = path_to_save,
                                    output_label = file_name
                                )

            if resume_file is not None:
                print(f'Loading pocoMC state from {resume_file}')
                sampler.load_state(resume_file)

            sampler.run(n_total=ntot, progress=True, save_every=nstates)

    else:
        sampler = pc.Sampler(
                                prior        = prior,
                                likelihood   = likelihood_wrapper,
                                n_effective  = neff,
                                output_dir   = path_to_save,
                                output_label = file_name
                            )
        if resume_file is not None:
            print(f'Loading pocoMC state from {resume_file}')
            sampler.load_state(resume_file)

        sampler.run(n_total=ntot, progress=True, save_every=nstates)

    samples, weights, logl, logp = sampler.posterior()

    print(f'Sampling ended at: {datetime.now()}')

    # Save results
    os.makedirs(path_to_save, exist_ok=True)

    print(f"Results saved to {os.path.join(path_to_save, file_name + '.npy')}")

    results = {}
    results['priors']  = priors
    results['samples'] = samples
    results['weights'] = weights
    results['logl']    = logl
    results['logp']    = logp

    np.save(os.path.join(path_to_save, file_name + '.npy'), results)

    time_f = time.time()

    print('Sampling efficiency:', sampler.results["efficiency"])
    print('Time to estimate (in minutes):', np.round((time_f-time_i)/60,2))
