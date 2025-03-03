#!/bin/bash
#SBATCH --job-name=train_RBM          # Job name
#SBATCH --output=qtrain_RBM_%j.out    # Standard output log (%j = job ID)
#SBATCH --error=qtrain_RBM_%j.err     # Error log
#SBATCH --partition=sapphire           # Partition name
#SBATCH --nodes=1                      # Request 1 node
#SBATCH --ntasks=1                     # 1 task 
#SBATCH --cpus-per-task=4             # Number of cores to allocate on the node
#SBATCH --time=08:00:00                # Time limit (adjust as needed)
#SBATCH --mem=4G                       # Memory per node (adjust as needed)

# this script requires one command line argument: the path to a json file with all the parameters for training the rbm. 
set -e 
# load modules -- often the environment only kicks in when you activate it twice. 
module load python

mamba activate olfaction

mamba deactivate 

mamba activate olfaction

python -u train_RBM.py $SLURM_JOB_ID $1 
