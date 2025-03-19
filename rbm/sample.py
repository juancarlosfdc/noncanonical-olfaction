from generative_RBM_jax import GenerativeRBM
import jax.numpy as jnp
import sys
import os 
import json 
import shutil 
import jax
import glob 

print('using cpu or gpu?\n', jax.default_backend())
jax.config.update("jax_default_matmul_precision", "high")  

# read arguments and copy training arguments into output directory 
job_id = sys.argv[1] 
weights_dir = sys.argv[2]
output_dir_name = sys.argv[3] 

weights_path = os.path.join(weights_dir, 'W.npy')
v_bias_path = os.path.join(weights_dir, 'v_bias.npy')
h_bias_path = os.path.join(weights_dir, 'h_bias.npy')
W = jnp.load(weights_path)
v_bias = jnp.load(v_bias_path)
h_bias = jnp.load(h_bias_path)

# load rbm class and generate samples 
task_id = int(job_id.split('_')[-1]) 
key = jax.random.PRNGKey(task_id)
rbm = GenerativeRBM(key=key, 
                    n_hidden=len(h_bias), 
                    batch_size=10, 
                    data_path='../Matrix2_filtered_and_downsampled_18March2025.csv')

key, subkey = jax.random.split(key) 
samples, _ = rbm.generate(subkey, n_samples=10000, W=W, v_bias=v_bias, h_bias=h_bias, gibbs_steps=10000)

print(rbm.compute_rmse(samples)) 
key, subkey = jax.random.split(key) 
print(rbm.compute_background_rmse(subkey))

output_path = os.path.join(output_dir_name, f'samples_{task_id}')

jnp.save(output_path, samples) 