import json

# Load the config
with open("example_config.json", "r") as f:
    config = json.load(f)

# Edit some hyperparameters
config["model"]["L"] = 100
config["model"]["M"] = 10

# Save to a new config file
with open("new_config.json", "w") as f:
    json.dump(config, f, indent=2)

print("Updated config saved to new_config.json")
