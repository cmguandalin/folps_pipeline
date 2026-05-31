#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-folps_oldspec}"

echo "Creating conda environment: ${ENV_NAME}"

conda create -y -n "${ENV_NAME}" python=3.10 pip

# Activate conda env inside a non-interactive script
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

python -m pip install --upgrade pip

# Match versions running on cosma8 without any issues
python -m pip install \
  "numpy==1.26.3" \
  "scipy==1.15.3" \
  "jax==0.4.23" \
  "jaxlib==0.4.23" \
  "multiprocess==0.70.19" \
  "pocomc==1.2.6" \
  "h5py" \
  "pyyaml"

echo
echo "Installed versions:"
python -c "import numpy, scipy, jax, jaxlib, multiprocess, pocomc; print('numpy', numpy.__version__); print('scipy', scipy.__version__); print('jax', jax.__version__); print('jaxlib', jaxlib.__version__); print('multiprocess', multiprocess.__version__); print('pocomc', pocomc.__version__)"

echo
echo "Environment ready:"
echo "conda activate ${ENV_NAME}"
