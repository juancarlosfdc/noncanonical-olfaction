#!/bin/bash
#SBATCH -c 1
#SBATCH -t 01:00:00
#SBATCH -p serial_requeue
#SBATCH --mem=1000
#SBATCH -o sigma_sweep%j.out 
#SBATCH -e sigma_sweep%j.err 

set -e 
# load modules
module load python

mamba activate olfaction

mamba deactivate 

mamba activate olfaction

which python 

python -u sigma_sweep.py 
