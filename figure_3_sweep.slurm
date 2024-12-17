#!/bin/bash
#SBATCH --job-name=NMn_sigma_sweep       # Job name
#SBATCH --output=NMn_sigma_sweep_%j.out # Standard output log (%j = job ID)
#SBATCH --error=NMn_sigma_sweep_%j.err  # Error log
#SBATCH --partition=sapphire            # Partition name
#SBATCH --nodes=1                       # Request 1 node
#SBATCH --ntasks=1                      # 1 task (GNU Parallel handles parallelization)
#SBATCH --cpus-per-task=10              # Number of cores to allocate on the node
#SBATCH --time=02:00:00                 # Time limit (adjust as needed)
#SBATCH --mem=8G                        # Memory per node (adjust as needed)

set -e
# Load required modules
module load python
module load ncf/1.0.0-fasrc01
module load parallel/20230422-rocky8_x64-ncf

# Activate Conda environment
mamba activate olfaction
mamba activate base 
mamba activate olfaction
# oddly you sometimes need to do this twice to point to activate the environment

# Debug: Check Python path
which python

# Check for four input files
if [ "$#" -ne 4 ]; then
    echo "Usage: sbatch figure_3_sweep.slurm <N_values.txt> <M_values.txt> <n_values.txt> <sigma_values.txt>"
    exit 1
fi

# Input files
N_FILE=$1
M_FILE=$2
n_FILE=$3
SIGMA_FILE=$4

# Check that files exist
for FILE in "$N_FILE" "$M_FILE" "$n_FILE" "$SIGMA_FILE"; do
    if [ ! -f "$FILE" ]; then
        echo "Error: File $FILE not found!"
        exit 1
    fi
done

# Load number of cores for GNU Parallel
export PARALLEL="--jobs $SLURM_CPUS_PER_TASK"

# Define the simulation commands
run_simulation_N() {
    N=$1
    echo "Running simulation for N=$N on $(hostname)"
    python run_optimization.py --N $N --results_file N_sweep_results.txt --output_dir N_sweep
}

run_simulation_M() {
    M=$1
    echo "Running simulation for M=$M on $(hostname)"
    python run_optimization.py --M $M  --results_file M_sweep_results.txt --output_dir M_sweep
}

run_simulation_n() {
    n=$1
    echo "Running simulation for n=$n on $(hostname)"
    python run_optimization.py --n $n --results_file n_sweep_results.txt --output_dir n_sweep
}

run_simulation_sigma() {
    sigma=$1
    echo "exported function correctly" 
    echo "Running simulation for sigma=$sigma on $(hostname)"
    python run_optimization.py --sigma $sigma --results_file sigma_sweep_results.txt --output_dir sigma_sweep
}

# Export the functions for GNU Parallel
export -f run_simulation_N
export -f run_simulation_M
export -f run_simulation_n
export -f run_simulation_sigma

# Run sweeps for N, M, n, and sigma using GNU Parallel
echo "Starting N sweep..."
parallel run_simulation_N ::: $(cat "$N_FILE")

echo "Starting M sweep..."
parallel run_simulation_M ::: $(cat "$M_FILE")

echo "Starting n sweep..."
parallel run_simulation_n ::: $(cat "$n_FILE")

echo "Starting sigma sweep..."
parallel -j 7 run_simulation_sigma ::: $(cat "$SIGMA_FILE")
