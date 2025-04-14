import json
import numpy as np
# Load the config
with open("configs/gamma_sweep/config_1.json", "r") as f:
    config = json.load(f)

# Edit some hyperparameters
# config["hyperparams"]["L"] = 100
# config["hyperparams"]["M"] = 10

a_s = np.logspace(-2, 2, 10)
for i, a in enumerate(a_s): 
    config["hyperparams"]["odor_model"] = 'data-driven log normal'
    config["hyperparams"]["N"] = 1000
    config["hyperparams"]["L"] = 1300
    config["hyperparams"]["M"] = 60
    config["hyperparams"]["canonical_init"] = False
    config["hyperparams"]["W_shape"] = a
    config["training"]["scans"] = 1
    config["training"]["epochs_per_scan"] = 100000
    config["logging"]["output_dir"] = "./results/a_sweep/"
    config["logging"]["config_id"] = str(i) 

    # Save to a new config file
    with open(f"configs/a_sweep/config_{i}.json", "w") as f:
        json.dump(config, f, indent=2)

print("Saved updated config(s)") 
