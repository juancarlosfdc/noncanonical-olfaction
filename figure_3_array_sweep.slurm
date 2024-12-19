#!/bin/bash
#SBATCH --job-name=multi_sweep          # Job name
#SBATCH --output=multi_sweep_%A_%a.out # Standard output log (%A = job ID, %a = array task ID)
#SBATCH --error=multi_sweep_%A_%a.err  # Error log
#SBATCH --partition=test           # Partition name
#SBATCH --nodes=1                      # Request 1 node
#SBATCH --ntasks=1                     # 1 task per job (1 parameter value per job)
#SBATCH --cpus-per-task=4              # 1 CPU per task
#SBATCH --time=00:30:00                # Time limit (adjust as needed)
#SBATCH --mem=4G                       # Memory per job

# Load Python environment
module load python
mamba activate olfaction
mamba activate base
mamba activate olfaction
which python 

# Parameter sweep definitions
PARAMETER=$1
OUTPUT_DIR=$2
PARAMETER_FILE="parameters_${PARAMETER}.txt"

# Ensure parameter file exists
if [ ! -f "$PARAMETER_FILE" ]; then
    echo "Error: Parameter file $PARAMETER_FILE not found!"
    exit 1
fi

# Extract parameter value for this task
PARAM_VALUE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$PARAMETER_FILE")

# Define other constant parameters
N=50
M=13
n=3
sigma=2.0
gamma=50

# Adjust constant parameters for specific sweeps
if [ "$PARAMETER" == "N" ]; then
    N=$PARAM_VALUE
elif [ "$PARAMETER" == "M" ]; then
    M=$PARAM_VALUE
elif [ "$PARAMETER" == "n" ]; then
    n=$PARAM_VALUE
elif [ "$PARAMETER" == "sigma" ]; then
    sigma=$PARAM_VALUE
elif [ "$PARAMETER" == "gamma" ]; then
    gamma=$PARAM_VALUE
fi

# Debug information
echo "Running sweep for $PARAMETER=$PARAM_VALUE (N=$N, M=$M, n=$n, sigma=$sigma, gamma=$gamma) on $(hostname)"

# Run the simulation
python run_optimization.py --N $N --M $M --n $n --sigma $sigma --gamma $gamma --results_file sweep_${PARAMETER}_results.txt --output_dir $OUTPUT_DIR