import argparse
import json
import jax
from model import FullConfig, HyperParams, TrainingConfig, LoggingConfig, initialize_p, initialize_training_state, make_constant_gammas, train_natural_gradient_scan_over_epochs
from plotting import plot_activity


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=False)
args = parser.parse_args()

def load_config(path: str) -> FullConfig:
    with open(path, 'r') as f:
        cfg_dict = json.load(f)
    return FullConfig(
        hyperparams=HyperParams(**cfg_dict['hyperparams']),
        training=TrainingConfig(**cfg_dict['training']),
        logging=LoggingConfig(**cfg_dict['logging']),
    )

config = load_config(args.config)

key = jax.random.key(config.seed) 

key, *subkeys = jax.random.split(key, 4)

hp = config.hyperparams

hp, p_init = initialize_p(subkeys[0], hp=hp)

init_state = initialize_training_state(subkeys[1], hp, p_init)

t = config.training
gammas = make_constant_gammas(t.scans, t.epochs_per_scan, gamma_T=t.gamma_T, gamma_p=t.gamma_p) 

state, metrics = train_natural_gradient_scan_over_epochs(init_state, hp, gammas, t.scans, t.epochs_per_scan)

fig, ax, *_ = plot_activity(init_state.p, state.p, hp, metrics['mi'], subkeys[2]) 
fig.suptitle(f'{hp.activity_model} activity, {hp.odor_model} odor model', y=1.05)
fig.savefig('tmp.png', bbox_inches='tight')