import argparse
import json
import jax
import numpy as np
from model import (
    FullConfig,
    HyperParams,
    TrainingConfig,
    LoggingConfig,
    initialize_p,
    initialize_training_state,
    make_constant_gammas,
    closure_draw_cs_data_driven,
    train_natural_gradient_scan_over_epochs,
    linear_filter_plus_glomerular_layer
)
import model 
from plotting import plot_expression 
import shutil
import matplotlib.pyplot as plt 


print(jax.default_backend())
jax.config.update("jax_default_matmul_precision", "high") 

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=False)
args = parser.parse_args()


def load_config(path: str) -> FullConfig:
    with open(path, "r") as f:
        cfg_dict = json.load(f)
    return FullConfig(
        hyperparams=HyperParams(**cfg_dict["hyperparams"]),
        training=TrainingConfig(**cfg_dict["training"]),
        logging=LoggingConfig(**cfg_dict["logging"]),
        seed=cfg_dict["seed"], 
        data_path=cfg_dict["data_path"]
    )


config = load_config(args.config)
key = jax.random.key(config.seed)
key, *subkeys = jax.random.split(key, 10)
hp = config.hyperparams

# this is a patch because we don't want to load all 10 data-driven odor models, each of which is 10^3 x 10^6. Besides this, it's nice to have all this logic in model.py
if hp.odor_model == "block covariance binary": 
    model.ODOR_MODEL_REGISTRY = {
        "block covariance binary": closure_draw_cs_data_driven(config.data_path)[0]
    }
elif hp.odor_model == "block covariance log normal": 
    model.ODOR_MODEL_REGISTRY = {
        "block covariance log normal": closure_draw_cs_data_driven(config.data_path)[1]
    }


hp, p_init = initialize_p(subkeys[0], hp=hp)

init_state = initialize_training_state(subkeys[1], hp, p_init, config.training)

t = config.training
gammas = make_constant_gammas(
    t.scans, t.epochs_per_scan, gamma_T=t.gamma_T, gamma_p=t.gamma_p, gamma_g=t.gamma_g
)

### remove after debugging G optimization

draw_cs = model.ODOR_MODEL_REGISTRY[hp.odor_model]
activity_function = model.ACTIVITY_FUNCTION_REGISTRY[hp.activity_model]
cs = draw_cs(subkeys[-1], hp)
g_init = linear_filter_plus_glomerular_layer(hp, p_init, cs, subkeys[-2])

state, metrics = train_natural_gradient_scan_over_epochs(
    init_state, hp, gammas, t.scans, t.epochs_per_scan
)

g_final = linear_filter_plus_glomerular_layer(hp, state.p, cs, subkeys[-3])


idx = np.random.choice(g_init.shape[1], 120, replace=False)

# Plot side-by-side heatmaps
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].imshow(g_init[:, idx], aspect='auto')
axes[0].set_title('g_init')
axes[1].imshow(g_final[:, idx], aspect='auto')
axes[1].set_title('g_final')
plt.tight_layout()
plt.show()
fig.savefig(f"{config.logging.output_dir}/g_activity_{config.logging.config_id}.png", bbox_inches="tight", dpi=300)


# fig, ax, *_ = plot_activity(init_state.p, state.p, hp, metrics["mi"], subkeys[2])
fig, axs = plot_expression(init_state.p.E, state.p.E, metrics["mi"], hp, t, key=subkeys[2])

figtitle = f"{hp.activity_model} activity, {hp.odor_model} odor model"

if "block covariance" in hp.odor_model: 
    k = config.data_path.split("_")[-1].split(".")[0]
    figtitle += f" (blocks = {k})"

fig.suptitle(figtitle)

fig.savefig(f"{config.logging.output_dir}/expression_{config.logging.config_id}.png", bbox_inches="tight", dpi=300)

fig, axs = plot_expression(init_state.p.G, state.p.G, metrics["mi"], hp, t, key=subkeys[2])
fig.savefig(f"{config.logging.output_dir}/G_{config.logging.config_id}.png", bbox_inches="tight", dpi=300)

jax.numpy.save(f"{config.logging.output_dir}/E_final_{config.logging.config_id}", state.p.E)
jax.numpy.save(f"{config.logging.output_dir}/W_{config.logging.config_id}", state.p.W)
jax.numpy.save(f"{config.logging.output_dir}/G_{config.logging.config_id}", state.p.G)

# with open(f"{config.logging.output_dir}/config_{config.logging.config_id}.json", "w") as c: 
#     json.dump(config, c)

shutil.copy2(args.config, f"{config.logging.output_dir}/config_{config.logging.config_id}.json")