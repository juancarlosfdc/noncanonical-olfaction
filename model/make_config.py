import json

# Load the config
with open("example_config.json", "r") as f:
    config = json.load(f)

# Edit some hyperparameters
# config["hyperparams"]["L"] = 100
# config["hyperparams"]["M"] = 10

gamma_ps = [.1, 1, 10]
gamma_ts = [.001, .01, .1]
for i, gammas in enumerate(zip(gamma_ps, gamma_ts)): 
    config["hyperparams"]["odor_model"] = 'data-driven log normal'
    config["hyperparams"]["N"] = 1000
    config["hyperparams"]["L"] = 1300
    config["hyperparams"]["M"] = 60
    config["hyperparams"]["canonical_init"] = False
    config["training"]["scans"] = 1
    config["training"]["epochs_per_scan"] = 10000
    config["training"]["gamma_p"] = gammas[0]
    config["training"]["gamma_T"] = gammas[1]
    config["logging"]["output_dir"] = "./results/"
    config["logging"]["config_id"] = str(i) 

    # Save to a new config file
    with open(f"configs/config_{i}.json", "w") as f:
        json.dump(config, f, indent=2)

print("Saved updated config(s)") 
