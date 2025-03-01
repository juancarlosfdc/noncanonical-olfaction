from generative_RBM import GenerativeRBM
import jax.numpy as jnp


rbm = GenerativeRBM(n_hidden=128)

train_args = {'X_train': rbm.X_train, 
              'epochs': 1000,
              'batch_size': 64, 
              'learning_rate': 0.01, 
              'k': 1, 
              'l2_reg': 0.0, 
              'sample_number': 1000}

fig, axs, samples, err = rbm.plot_deviations_over_time(train_args)

fig.savefig('devs.png') 

jnp.save('samples', samples) 

fig, ax = rbm.plot_samples(samples) 
fig.savefig('samples.png') 

