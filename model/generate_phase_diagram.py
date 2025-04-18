import numpy as np
import json
import glob
import sys 
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate contour plots with optional log scaling')
    
    # Required arguments
    parser.add_argument('--output_directory', type=str, help='output directory')
    parser.add_argument('--param1_key', type=str, help='Key for first parameter')
    parser.add_argument('--param2_key', type=str, help='Key for second parameter')
    
    # Optional argument with default False
    parser.add_argument('--log_param1', action='store_true',
                       help='Use log scale for param1 (default: False)')
    
    return parser.parse_args()

# Usage example
args = parse_arguments()

# Accessing the values
OD = args.output_directory
param1_key = args.param1_key
param2_key = args.param2_key
log_param1_scale = args.log_param1  # Will be False if not specified

E_files = sorted(glob.glob(f"{OD}/E_final_*.npy"), key=lambda x: int(x.split('_')[-1].split('.')[0]))
config_files = sorted(glob.glob(f"{OD}/config_*.json"), key=lambda x: int(x.split('_')[-1].split('.')[0]))

data = {
    param1_key: [],  # List of values for parameter 1 (from config)
    param2_key: [],  # List of values for parameter 2 (from config)
    "non-canonical score": [],  # List of computed metrics (from E_final)
}

def compute_noncanonical_score(E): 
    return - np.mean(E * np.log(E))

for E_file, config_file in zip(E_files, config_files):
    # Load config and extract parameters
    with open(config_file, 'r') as f:
        config = json.load(f)
    param1 = config["hyperparams"][param1_key]  # Replace with your keys
    param2 = config["hyperparams"][param2_key]
    
    # Load E_final and compute metric (e.g., mean energy)
    E_final = np.load(E_file)
    metric = compute_noncanonical_score(E_final)
    
    # Append to dictionary
    data[param1_key].append(int(param1)) 
    data[param2_key].append(float(param2)) 
    data["non-canonical score"].append(float(metric)) 

param1 = np.array(data[param1_key])
param2 = np.array(data[param2_key])
metric = np.array(data["non-canonical score"])


fig, ax = plt.subplots() 

if np.min(metric) < 1e-5:
    metric = np.log10(metric) 
    cbar_label = r'$\log_{10}(score)$'  # Label for log scale
else:
    cbar_label = 'score'  # Label for linear scale

if log_param1_scale: 
    contour = ax.tricontourf(np.log10(param1), param2, metric, levels=20, cmap='viridis')
    ax.scatter(np.log10(param1), param2, c='black') 
else: 
    contour = ax.tricontourf(param1, param2, metric, levels=20, cmap='viridis')
    ax.scatter(param1, param2, c='black') 



# Add colorbar with the correct label
cbar = plt.colorbar(contour, ax=ax)
cbar.set_label(cbar_label)


if log_param1_scale:
    xticks = np.log10(param1)
    xtick_labels = [fr'$10^{{{x:.1f}}}$' for x in xticks]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels)
    ax.xaxis.set_minor_locator(LogLocator(subs='all', numticks=10))  
    ax.grid(True, which='both', linestyle=':', alpha=0.5) 

ax.set_xlabel(param1_key) 
ax.set_ylabel(param2_key)
# Save
with open(f"{OD}/phase_diagram_data.json", 'w') as f:
    json.dump(data, f, indent=4) 

fig.savefig(f"{OD}/phase_diagram.png", bbox_inches='tight')
