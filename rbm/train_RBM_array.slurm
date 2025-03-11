#!/bin/bash
#SBATCH --job-name=train_RBM          # Job name
#SBATCH --output=qtrain_RBM_%A_%a.out      # Standard output log (%A = array job ID, %a = array task ID)
#SBATCH --error=qtrain_RBM_%A_%a.err       # Error log
#SBATCH --partition=test           # Partition name
#SBATCH --nodes=1                      # Request 1 node
#SBATCH --ntasks=1                     # 1 task 
#SBATCH --cpus-per-task=1           # Number of cores to allocate on the node
#SBATCH --time=00:10:00                # Time limit (adjust as needed)
#SBATCH --mem=2G                       # Memory per node (adjust as needed)
#SBATCH --array=0-1                  # Create an array of 49 tasks (indices 0 through 12)


# this script requires one command line argument: the path to a json file with all the parameters for training the rbm. 
set -e 
# load modules -- often the environment only kicks in when you activate it twice. 
module load python

mamba activate olfaction 

mamba activate base

mamba activate olfaction # oddly, you often have to activate the environment twice on the cluster. 

which python
date 
python -u train_RBM.py ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} hyperparameters/hyperparameters_${SLURM_ARRAY_TASK_ID}.json
date 