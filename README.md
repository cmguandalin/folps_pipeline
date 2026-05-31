# Installing the FOLPS Pipeline

This guide explains how to set up the environment and install the dependencies required to run the **FOLPS pipeline**.

---

## 1. Create a Conda Environment

(Recommended) Run 

```bash
bash create_folps_env.sh folps
```

to create a conda environment with the versions I'm currently using. Some versions are specific due to a jax incompatibility when dealing with the DESI-generated power spectrum window matrices. Check that script for the current versions.

Activate your environment:

```bash
conda activate folps
```

## 2. Download the BACCO Emulator Cache

BACCO emulator downloads additional files on the first run.

Some HPC systems do not allow for connection during execution time. 
In that case, you must provide the emulator cache manually.

### 2.1. Locate the BACCO path

Run
```bash
pip show baccoemu
```

This will show the installation location of the package (e.g., `/my/path/to/conda-envs/folps/lib/python3.10/site-packages/baccoemu`).

### 2.2. Add the cache files

Unzip the provided cache file (`bacco_cache.zip`) inside the `baccoemu` directory.

## 3. Configure FOLPS Backend
Before running the pipeline, edit
```bash
src/model.py
```

Replace `/path/to/folps/folpsD/` with the correct path to your folpsD folder:
```python
import os, sys
os.environ['FOLPS_BACKEND'] = 'numpy'

sys.path.append('/path/to/folps/folpsD/')
import folps as FOLPS
```

## 4. Update the paths in the configuration files
Before running the pipeline, make sure all required paths are correctly set in the `.yml` files.

### 4.1. Guidance for the current set up (temporary)
Choose one of the available files inside `config/` as an `example.yml`

For the AP reparametrization check, for example, `config/p0p2b0_LRG2_nowindow_reparam.yml`
Notice that, if you don't rename the paramameters `PAR` to be reparametrized by `PAR_tilde` (i.e., adding the `_tilde` after the parameter name), the reparametrization will not be applied to the parameter. You must also set `reparametrize: true` in the `.yml` file.

## 5. Usage
### 5.1. Submit on HPC
```bash
sbatch scripts/run_fit-poco.sh config/example.yml
```
Track the process with `tail -f logs/JOB_ID.out`

### 5.2. Local usage
```bash
nohup python -u src/inference.py -config config/example.yml > nohup.out 2>&1 &
```

Track the process with `tail -f nohup.out``
