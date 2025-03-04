from generative_RBM import GenerativeRBM
import jax.numpy as jnp
import sys
import os 
import json 
import shutil 

# read arguments and copy training arguments into output directory 
job_id = sys.argv[1] 
arg_dict_path = sys.argv[2] 

os.mkdir(job_id)

with open(arg_dict_path, 'r') as adp: 
    train_args = json.load(adp)

parameter_path = os.path.join(job_id, 'parameters.json') 

shutil.copy2(arg_dict_path, parameter_path) 

# load rbm class and run the code 
rbm = GenerativeRBM(n_hidden=128)
fig, axs, samples, err = rbm.plot_deviations_over_time(train_args)

# create output paths and write outputs 
deviation_numpy_path = os.path.join(job_id, 'moment_deviations') 
deviation_png_path = os.path.join(job_id, 'moment_deviations.png')

jnp.save(deviation_numpy_path, err) 
fig.savefig(deviation_png_path) 

samples_numpy_path = os.path.join(job_id, 'samples') 

samples_png_path = os.path.join(job_id, 'samples.png') 


fig, ax = rbm.plot_samples(samples) 

reg = train_args['l2_reg']
exp = jnp.log10(reg) 

fig.suptitle(rf'$10^{{{exp:.2f}}}$', size=40) 

jnp.save(samples_numpy_path, samples) 
fig.savefig(samples_png_path)  
