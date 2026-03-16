# Installing the FOLPS Pipeline

This guide explains how to set up the environment and install the dependencies required to run the **FOLPS pipeline**.

---

## 1. Create a Conda Environment

(Recommended) Create a new Conda environment:

```bash
mkdir -p /path/to/conda-envs
conda create --prefix /path/to/conda-envs/folps python=3.14 -y
```

Activate your environment:

```bash
conda activate /path/to/conda-envs/folps
```

## 2. Install dependencies
```bash
pip install pocomc
pip install baccoemu
pip install pyyaml
pip install multiprocess
```

## 3. Download the BACCO Emulator Cache

BACCO emulator downloads additional files on the first run.

Some HPC systems do not allow for connection during execution time. 
In that case, you must provide the emulator cache manually.

### 3.1. Locate the BACCO path

Run
```bash
pip show baccoemu
```

This will show the installation location of the package (e.g., `/my/path/to/conda-envs/folps/lib/python3.14/site-packages/baccoemu`).

### 3.2. Add the cache files

Unzip the provided cache file (`bacco_cache.zip`) inside the `baccoemu` directory.

## 4. Configure FOLPS Backend
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

## 5. Update the paths in the configuration files
Before running the pipeline, make sure all required paths are correctly set in the `.yml` files.

### 5.1. Guidance for the current set up (temporary)
Choose one of the available files inside `config/` as an `example.yml`

For the AP reparametrization check, for example, `config/p0p2b0_LRG2_nowindow_reparam.yml`
Notice that, if you don't rename the paramameters `PAR` to be reparametrized by `PAR_tilde` (i.e., adding the `_tilde` after the parameter name), the reparametrization will not be applied to the parameter. You must also set `reparametrize: true` in the `.yml` file.

## 6. Usage
### 6.1. Submit on HPC
```bash
sbatch scripts/run_fit-poco.sh config/example.yml
```
Track the process with `tail -f logs/JOB_ID.out`

### 6.2. Local usage
```bash
nohup python -u src/inference.py -config config/example.yml > nohup.out 2>&1 &
```

Track the process with `tail -f nohup.out`
