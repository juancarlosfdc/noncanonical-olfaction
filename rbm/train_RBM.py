from generative_RBM import GenerativeRBM
import jax.numpy as jnp
import sys
import os 
import json 
import shutil 

# read arguments and copy training arguments into output directory 
job_id = sys.argv[1] 
arg_dict_path = sys.argv[2]

with open(arg_dict_path, 'r') as adp: 
    train_args = json.load(adp)


reg = train_args['l2_reg']
lr = train_args['learning_rate']
lambda_exp = jnp.log10(reg) 
lr_exp = jnp.log10(lr) 

output_dir_name = f"gamma_lambda_sweep/lambda_${lambda_exp:.2f}_gamma_${lr_exp:.2f}$"
os.mkdir(output_dir_name)


parameter_path = os.path.join(output_dir_name, 'parameters.json') 

shutil.copy2(arg_dict_path, parameter_path) 

# load rbm class and run the code 
rbm = GenerativeRBM(n_hidden=128)
fig, axs, samples, err = rbm.plot_deviations_over_time(train_args)

# create output paths and write outputs 
deviation_numpy_path = os.path.join(output_dir_name, 'moment_deviations') 
deviation_png_path = os.path.join(output_dir_name, 'moment_deviations.png')

jnp.save(deviation_numpy_path, err) 

reg = train_args['l2_reg']
exp = jnp.log10(reg) 

fig.suptitle(rf'$10^{{{exp:.2f}}}$', size=40) 

fig.savefig(deviation_png_path) 

# make samples output paths
samples_numpy_path = os.path.join(output_dir_name, 'samples') 
samples_png_path = os.path.join(output_dir_name, 'samples.png') 

fig, ax = rbm.plot_samples(samples) 

fig.suptitle(rf'$10^{{{exp:.2f}}}$', size=40) 

jnp.save(samples_numpy_path, samples) 
fig.savefig(samples_png_path)  
