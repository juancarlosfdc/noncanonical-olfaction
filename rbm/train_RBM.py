from generative_RBM_jax import GenerativeRBM
import jax.numpy as jnp
import sys
import os 
import json 
import shutil 
import jax
import glob 

print(jax.default_backend())
jax.config.update("jax_default_matmul_precision", "high")  

# read arguments and copy training arguments into output directory 
job_id = sys.argv[1] 
arg_dict_path = sys.argv[2]
sweep_dir = sys.argv[3] 

with open(arg_dict_path, 'r') as adp: 
    hyperparameters = json.load(adp)


reg = hyperparameters['l2_reg']
lr = hyperparameters['learning_rate']
lambda_exp = jnp.log10(reg) 
lr_exp = jnp.log10(lr) 

output_dir_name = f"{sweep_dir}/lambda_{lambda_exp:.2f}_gamma_{lr_exp:.2f}"
os.makedirs(output_dir_name, exist_ok=True) 


parameter_path = os.path.join(output_dir_name, 'parameters.json') 

shutil.copy2(arg_dict_path, parameter_path) 

# load rbm class and run the code 
hyperparameters['key'] = jax.random.PRNGKey(hyperparameters['random_seed'])
rbm = GenerativeRBM(key=hyperparameters['key'], 
                    n_hidden=hyperparameters['n_hidden'], 
                    batch_size=hyperparameters['batch_size'], 
                    digits=[5])

hyperparameters.pop('n_hidden') 
hyperparameters.pop('random_seed')

# clear samples directory: 
sample_files = glob.glob(f'{output_dir_name}/samples/*')
[os.remove(sf) for sf in sample_files]
fig, axs, samples, err = rbm.plot_deviations_over_time(output_dir_name, hyperparameters)

# create output paths and write outputs 
deviation_numpy_path = os.path.join(output_dir_name, 'moment_deviations') 
deviation_png_path = os.path.join(output_dir_name, 'moment_deviations.png')

jnp.save(deviation_numpy_path, err) 

reg = hyperparameters['l2_reg']
exp = jnp.log10(reg) 

fig.suptitle(rf'$10^{{{exp:.2f}}}$', size=20) 

fig.savefig(deviation_png_path) 

# make samples output paths
samples_numpy_path = os.path.join(output_dir_name, 'samples') 
samples_png_path = os.path.join(output_dir_name, 'samples.png') 

key, subkey = jax.random.split(hyperparameters['key'])
fig, ax = rbm.plot_samples(subkey, samples) 

fig.suptitle(rf'$10^{{{exp:.2f}}}$', size=40) 

jnp.save(samples_numpy_path, samples) 
fig.savefig(samples_png_path)  
