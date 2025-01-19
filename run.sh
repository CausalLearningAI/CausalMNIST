#!/bin/bash
#
#---------------------------------------------
# SLURM job script for single CPU/GPU
#---------------------------------------------
#
#
#SBATCH --job-name=PPCI_test        # Job name
#SBATCH --output=logs/PPCI_%j.log   # Output file
#SBATCH --time=04:00:00             # Maximum walltime
#SBATCH --ntasks=1                  # Number of tasks
#SBATCH --mem=10G                   # Memory allocation
#SBATCH --partition=gpu100          # Partition to use (e.g., gpu)
#SBATCH --gres=gpu:1                # Number of GPUs requested (adjust as needed)
#
# Load your environment or module
##source ~/.bashrc                  # Ensure you load your bash profile
module load conda
conda activate crl                  # Activate your conda environment


# Run experiments, general variables
epochs=30
seeds=20

# Generalization
pU=0.02

# # seeing a little of Y, and Z bad
# pW=0.3
# srun python ./src/run_gen.py --pW ${pW} --pU ${pU} --exp OS --epochs ${epochs} --seeds ${seeds}
# srun python ./src/run_gen.py --pW ${pW} --pU ${pU} --exp RCT --epochs ${epochs} --seeds ${seeds}

# seeing Z well and some Y 
pW=0.5
# srun python ./src/run_gen.py --pW ${pW} --pU ${pU} --exp OS --epochs ${epochs} --seeds ${seeds}
srun python ./src/run_gen.py --pW ${pW} --pU ${pU} --exp RCT --epochs ${epochs} --seeds ${seeds}

# # seeing the most of Y, but Z bad
# pW=0.8
# srun python ./src/run_gen.py --pW ${pW} --pU ${pU} --exp OS --epochs ${epochs} --seeds ${seeds}
# srun python ./src/run_gen.py --pW ${pW} --pU ${pU} --exp RCT --epochs ${epochs} --seeds ${seeds}

# Efficiency
# ...