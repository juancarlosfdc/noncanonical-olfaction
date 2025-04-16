import json
import numpy as np
# Load the config
with open("configs/gamma_sweep/config_1.json", "r") as f:
    config = json.load(f)

# Edit some hyperparameters
# config["hyperparams"]["L"] = 100
# config["hyperparams"]["M"] = 10

Ls = [int(L) for L in np.logspace(1, 3.5, 7)]
W_shapes = np.logspace(-3, 1, 7)
i = 0
for L in Ls: 
    for W_shape in W_shapes: 
        config["hyperparams"]["odor_model"] = 'data-driven log normal'
        config["hyperparams"]["N"] = 1000
        config["hyperparams"]["L"] = L
        config["hyperparams"]["M"] = 60
        config["hyperparams"]["canonical_init"] = False
        config["hyperparams"]["W_shape"] = W_shape
        config["hyperparams"]["W_scale"] = 0.36 
        config["hyperparams"]["sigma_0"] = 0.1
        config["training"]["scans"] = 10
        config["training"]["epochs_per_scan"] = 100000
        config["training"]["gamma_p"] = 1.0 
        config["logging"]["output_dir"] = "./results/W_shape_L_sweep/"
        config["logging"]["config_id"] = str(i) 

        # Save to a new config file
        with open(f"configs/W_shape_L_sweep/config_{i}.json", "w") as f:
            json.dump(config, f, indent=2)
        i += 1

print("Saved updated config(s)") 
