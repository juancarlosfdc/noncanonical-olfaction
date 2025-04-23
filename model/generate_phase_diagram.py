import numpy as np
import json
import glob
import sys 
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
import argparse
plt.rcParams['font.size'] = 18

def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate contour plots with optional log scaling')
    
    # Required arguments
    parser.add_argument('--output_directory', type=str, help='output directory')
    parser.add_argument('--param1_key', type=str, help='Key for first parameter')
    parser.add_argument('--param2_key', type=str, help='Key for second parameter')
    
    # Optional argument with default False
    parser.add_argument('--log_param1', action='store_true',
                       help='Use log scale for param1 (default: False)')
    parser.add_argument('--overlay_values', action='store_true',
                       help='Write numbers on heatmap (default: False)')
    
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
    return - np.mean(E * np.log(E)) / np.log(E.shape[0])

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

# Get sorted unique values
unique_param1 = np.sort(np.unique(param1))
unique_param2 = np.sort(np.unique(param2))

# Number of bins
n_x = len(unique_param1)
n_y = len(unique_param2)

# Calculate bin edges
if log_param1_scale:
    log_param1 = np.log10(unique_param1)
    param1_edges = np.concatenate([
        [log_param1[0] - (log_param1[1]-log_param1[0])/2],  # Left edge
        (log_param1[1:] + log_param1[:-1])/2,               # Midpoints
        [log_param1[-1] + (log_param1[-1]-log_param1[-2])/2] # Right edge
    ])
else:
    param1_edges = np.concatenate([
        [unique_param1[0] - (unique_param1[1]-unique_param1[0])/2],
        (unique_param1[1:] + unique_param1[:-1])/2,
        [unique_param1[-1] + (unique_param1[-1]-unique_param1[-2])/2]
    ])

param2_edges = np.concatenate([
    [unique_param2[0] - (unique_param2[1]-unique_param2[0])/2],
    (unique_param2[1:] + unique_param2[:-1])/2,
    [unique_param2[-1] + (unique_param2[-1]-unique_param2[-2])/2]
])

# Create plot
fig, ax = plt.subplots(figsize=(10, 8))

    # mosaic = [["E_init", "E_final", "MI"], ["hist_init", "hist_final", "params"]]
    # fig, axs = plt.subplot_mosaic(
    #     mosaic, 
    #     figsize=(18, 10),
    #     gridspec_kw={"hspace": 0.5}
    # )

# Plot using pcolormesh with exact edges
plot_data = metric.reshape(n_x, n_y).T  # Transpose for correct orientation

if log_param1_scale:
    mesh = ax.pcolormesh(param1_edges, param2_edges, plot_data,
                        shading='flat', cmap='viridis')
else:
    mesh = ax.pcolormesh(param1_edges, param2_edges, plot_data,
                        shading='flat', cmap='viridis')

# Set ticks at box centers
if log_param1_scale:
    ax.set_xticks(log_param1)
    ax.set_xticklabels([fr'$10^{{{x:.1f}}}$' for x in log_param1])
else:
    ax.set_xticks(unique_param1)
    ax.set_xticklabels([f'{x:.1f}' for x in unique_param1])

ax.set_yticks(unique_param2)
ax.set_yticklabels([f'{y:.1f}' for y in unique_param2])

# Critical fix: Set grid lines at the EDGES (not centers)
ax.set_xticks(param1_edges, minor=True)
ax.set_yticks(param2_edges, minor=True)
ax.grid(which='minor', color='white', linestyle='-', linewidth=1)
ax.tick_params(which='minor', length=0)  # Hide minor ticks
# Increase tick label size for both axes
ax.tick_params(axis='both', which='major', labelsize=18)  # Adjust 14 to your preferred size

if args.overlay_values: 
    for i in range(plot_data.shape[0]):  # Loop over y-axis (rows)
        for j in range(plot_data.shape[1]):  # Loop over x-axis (columns)
            # Get the center of the current cell
            x_center = (param1_edges[j] + param1_edges[j+1]) / 2
            y_center = (param2_edges[i] + param2_edges[i+1]) / 2
            
            # Add the metric value as text
            ax.text(
                x_center, y_center,
                f"{plot_data[i, j]:.2f}",  # Format to 2 decimal places
                ha='center', va='center',  # Center the text
                color='white' if plot_data[i, j] > 0.5 * plot_data.max() else 'black'
        )

fig.colorbar(mesh, label=cbar_label) 

# Adjust layout
# plt.setp(ax.get_xticklabels())
plt.tight_layout()
plt.show()

ax.set_xlabel(param1_key) 
ax.set_ylabel(param2_key)
# Save
with open(f"{OD}/phase_diagram_data.json", 'w') as f:
    json.dump(data, f, indent=4) 

fig.savefig(f"{OD}/phase_diagram.png", bbox_inches='tight')
