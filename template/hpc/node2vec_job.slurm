#!/bin/bash
## Job Config
#SBATCH --job-name=cardiokb_node2vec
#SBATCH --mail-user=Asma.Nawaz@cshs.org
#SBATCH --mail-type=ALL

## Task and hardware config — CPU-heavy, no GPU needed
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

## Time Config — SparseOTF walks are slower but memory-safe
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out

# Name of conda env
ENV_NAME=cardiokb_ml

# End of Parameters
echo "My job ran on:"
echo $SLURM_NODELIST

# Loading modules
module purge
module load slurm
module load gcc/11.2.0
module load miniconda3-4.10.3-gcc-11.2.0-33aghju

echo "Loaded Modules"
module list

# Getting path to miniconda3
CONDA_BASE=$(conda info --base)

# Enable conda at path
source "$CONDA_BASE/etc/profile.d/conda.sh"

# Setting up python env
if ! conda info --envs | grep -q "$ENV_NAME"; then
    echo "Environment $ENV_NAME does not exist. Creating..."
    conda create -y -n "$ENV_NAME" python=3.11
fi

conda activate "$ENV_NAME"

# Sanity checks
echo "Active conda environment:"
which python
python --version

# Install dependencies
pip install -q pecanpy gensim numpy python-dotenv

# Running Node2Vec training
echo "Starting Node2Vec training..."
cd $SLURM_SUBMIT_DIR

srun python -u train_node2vec.py
