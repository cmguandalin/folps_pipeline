#!/bin/bash
#SBATCH --job-name=LRG2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=72:00:00           # Max wall time
#SBATCH --output=/cosma/home/dp322/dc-guan2/folps/pipeline/logs/%j_%x.out
#SBATCH --error=/cosma/home/dp322/dc-guan2/folps/pipeline/logs/%j_%x.err
#SBATCH --partition=cosma8-serial
#SBATCH --account=dp322
#SBATCH --mail-type=ALL
#SBATCH --mail-user=caroline.guandalin@roe.ac.uk

# Critical: set threading variables (should help to use as much as possible of the CPUs available)
#export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
#export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
#export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

# Check if a config file was provided
if [ -z "$1" ]; then
  echo "Error: No config file provided."
  echo "Usage: sbatch script_name.sh <config_file>"
  exit 1
fi

CONFIG_FILE=$1

module load python/conda3-2023.09
eval "$(/cosma/local/anaconda3/202309/bin/conda shell.bash hook)"
conda activate /cosma/apps/dp322/dc-guan2/conda-envs/folps

export GLOBAL_DIR="/cosma/home/dp322/dc-guan2/folps/pipeline/"
#python -u $GLOBAL_DIR/src/inference.py -config $GLOBAL_DIR/$CONFIG_FILE
# Use the following if there's no intention to run it in your pc
python -u $GLOBAL_DIR/src/inference.py -config $GLOBAL_DIR/$CONFIG_FILE -ncpus $SLURM_CPUS_PER_TASK
