#!/bin/bash
#SBATCH --job-name=train_RBM          # Job name
#SBATCH --output=qtrain_RBM_%j.out    # Standard output log (%j = job ID)
#SBATCH --error=qtrain_RBM_%j.err     # Error log
#SBATCH --partition=test           # Partition name
#SBATCH --nodes=1                      # Request 1 node
#SBATCH --ntasks=1                     # 1 task 
#SBATCH --cpus-per-task=4             # Number of cores to allocate on the node
#SBATCH --time=02:00:00                # Time limit (adjust as needed)
#SBATCH --mem=4G                       # Memory per node (adjust as needed)

set -e 
# load modules
module load python

mamba activate olfaction

mamba deactivate 

mamba activate olfaction

which python 

python -u train_RBM.py
