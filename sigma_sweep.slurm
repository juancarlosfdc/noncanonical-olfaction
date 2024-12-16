#!/bin/bash
#SBATCH --job-name=sigma_sweep          # Job name
#SBATCH --output=sigma_sweep_%j.out    # Standard output log (%j = job ID)
#SBATCH --error=sigma_sweep_%j.err     # Error log
#SBATCH --partition=sapphire           # Partition name
#SBATCH --nodes=1                      # Request 1 node
#SBATCH --ntasks=1                     # 1 task (GNU Parallel handles parallelization)
#SBATCH --cpus-per-task=16             # Number of cores to allocate on the node
#SBATCH --time=01:00:00                # Time limit (adjust as needed)
#SBATCH --mem=8G                       # Memory per node (adjust as needed)


set -e 
# load modules
module load python

mamba activate olfaction

mamba deactivate 

mamba activate olfaction

which python 

# Check if the input file (sigmas.txt) is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: sbatch sigma_sweep.slurm <sigmas.txt>"
    exit 1
fi

# Get the input file
SIGMAS_FILE=$1

# Check if the file exists
if [ ! -f "$SIGMAS_FILE" ]; then
    echo "Error: File $SIGMAS_FILE not found!"
    exit 1
fi

# Load required modules (if needed)
module load ncf/1.0.0-fasrc01 
module load parallel/20230422-rocky8_x64-ncf  # Load GNU Parallel

# Export the number of available cores for GNU Parallel
export PARALLEL="--jobs $SLURM_CPUS_PER_TASK"

# Define the command to run for each sigma
run_simulation() {
    sigma=$1
    echo "Running simulation for sigma=$sigma on $(hostname)"
    python single_sigma.py --sigma $sigma --results_file rho_vs_sigma.txt
}

# Export the function for GNU Parallel
export -f run_simulation

# Run GNU Parallel with sigma values from the file
parallel run_simulation ::: $(cat "$SIGMAS_FILE")
