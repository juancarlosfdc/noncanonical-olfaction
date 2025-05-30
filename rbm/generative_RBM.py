import jax
import jax.numpy as jnp
from jax import random
from jax import jit 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import pandas as pd

class GenerativeRBM:
    def __init__(self, n_hidden, data_path=None, key=None, W_scale=0.01, digits=None):
        """
        Initializes the RBM with the given number of hidden units.
        Data is expected to be arranged such that each column is a sample
        and each row is a variable.
        If no random key is provided, a default one is created.
        """
        if key is None:
            key = random.PRNGKey(0)
        self.key = key
        self.data = self.load_data(data_path, digits)
        self.set_empirical_means()
        self.n_visible = self.data.shape[0]  # each row is a variable
        self.n_hidden = n_hidden
        self.key, subkey = random.split(self.key)
        # Initialize weights and biases.
        # W has shape (n_hidden, n_visible)
        self.W = random.normal(subkey, shape=(n_hidden, self.n_visible)) * W_scale
        self.h_bias = jnp.zeros((n_hidden,))
        self.v_bias = self.initialize_v_bias() 

    def load_data(self, data_path, digits):
        if data_path is None:
            mnist = fetch_openml('mnist_784', version=1)
            data = mnist.data
            target = mnist.target
            if digits is not None: 
                data = mnist.data[mnist.target.isin([str(d) for d in digits])] 
            # Original shape is (n_samples, n_features). Transpose to have shape (n_features, n_samples).
            X = np.array(data.T, dtype=np.float32) / 255.0  # Normalize to [0,1]
            print("Data shape:", X.shape)
            X = (X > 0.5).astype(np.float32)  # Binarize the data
        else:
            X = pd.read_csv(data_path, index_col=0).values # this is assumed to be in (n_features, n_samples) format, with an index column. 
            # X = np.load(data_path) # this is assumed to be in (n_features, n_samples) format
        # Split into training and test sets.
        # train_test_split expects samples as rows, so we transpose before and after splitting.
        X_train, X_test = train_test_split(X.T, test_size=0.2, random_state=int(self.key[0]))
        # Transpose back so that each column is a sample.
        X_train = jnp.array(X_train.T)
        X_test = jnp.array(X_test.T)
        self.X_train = X_train
        self.X_test = X_test
        return X
    
    def initialize_v_bias(self): 
        v_probs = jnp.sum(self.data, axis=1) / self.data.shape[1] 
        nonzero = v_probs[v_probs > 0]
        v_probs = jnp.where(v_probs == 0, jnp.min(nonzero), v_probs)
        v_bias_init = jnp.log(v_probs / (1 - v_probs))
        return v_bias_init 

    def sigmoid(self, x):
        return 1.0 / (1.0 + jnp.exp(-x))

    def sample_h(self, v, key):
        """Sample hidden units with explicit key handling."""
        key, subkey = random.split(key)
        p_h = self.sigmoid(jnp.dot(self.W, v) + self.h_bias[:, None])
        h = random.bernoulli(subkey, p_h).astype(jnp.float32)
        return h, p_h, key  # Return updated key

    def sample_v(self, h, key):
        """Sample visible units with explicit key handling."""
        key, subkey = random.split(key)
        p_v = self.sigmoid(jnp.dot(self.W.T, h) + self.v_bias[:, None])
        v = random.bernoulli(subkey, p_v).astype(jnp.float32)
        return v, p_v, key   # Return updated key

    def contrastive_divergence(self, v0, k=1, key=None):
        """Performs k Gibbs steps with explicit key handling."""
        if key is None:
            key = self.key
        vk = v0
        for _ in range(k):
            # Let sample_h/sample_v handle key splitting internally
            h, _, key = self.sample_h(vk, key)
            vk, _, key = self.sample_v(h, key)
        return vk, key

    def train_batch(self, v0, learning_rate=0.01, k=1, l2_reg=0.0, key=None):
        """
        Updates the parameters of the RBM using Contrastive Divergence.
        
        Parameters:
          v0 (array): Batch of visible units with shape (n_visible, batch_size).
          learning_rate (float): Learning rate for the update.
          k (int): Number of Gibbs sampling steps.
          l2_reg (float): L2 regularization coefficient.
        """
        if key is None: 
            key = self.key
        key, subkey = random.split(key) 
        vk, key = self.contrastive_divergence(v0, k, subkey)
        h0, _, key = self.sample_h(v0, key)
        hk, _, key = self.sample_h(vk, key)
        batch_size = v0.shape[1]
        # Compute gradients (note: outer products are computed by dotting with the transpose).
        delta_W = jnp.dot(h0, v0.T) - jnp.dot(hk, vk.T)
        delta_v_bias = jnp.sum(v0 - vk, axis=1)  # Sum over samples (columns)
        delta_h_bias = jnp.sum(h0 - hk, axis=1)  # Sum over samples (columns)
        # Update parameters with optional L2 regularization.
        self.W = self.W + learning_rate * (delta_W / batch_size - l2_reg * self.W)
        self.v_bias = self.v_bias + learning_rate * delta_v_bias / batch_size
        self.h_bias = self.h_bias + learning_rate * delta_h_bias / batch_size
        self.key = key
    
    def reconstruct(self, v, k=1, key=None):
        """
        Reconstructs the input visible units by running Gibbs sampling.
        v is expected to have shape (n_visible, batch_size).
        """
        if key is None: 
            key = self.key
        return self.contrastive_divergence(v, k, key) # this returns a v_recon and a key
    
    def generate(self, n_samples, gibbs_steps=100, key=None):
        """Generate samples with explicit key management."""
        if key is None:
            key = self.key  # Use class key if none provided
        key, subkey = random.split(key)
        # Initialize samples with fresh key
        samples = random.bernoulli(subkey, p=jnp.ones((self.n_visible, n_samples)) * 0.5).astype(jnp.float32)
        
        # Gibbs sampling with explicit key passing
        for _ in range(gibbs_steps):
            h, _, key = self.sample_h(samples, key) # these functions split the key inside and return 
            samples, _, key = self.sample_v(h, key)
        
        return samples, key  # Return samples and updated key

    def fit(self, epochs=10, batch_size=64, learning_rate=0.01, k=1, l2_reg=0.0, sample_number=1000, key=None):
        """
        Trains the RBM on the training data.
        
        Parameters:
          X_train (array): Training data as a JAX array with shape (n_visible, n_train),
                           where each column is a sample.
          epochs (int): Number of epochs.
          batch_size (int): Number of samples per batch.
          learning_rate (float): Learning rate.
          k (int): Number of Gibbs sampling steps per batch.
          l2_reg (float): L2 regularization coefficient.
          
        Returns:
          losses: A list of reconstruction losses per epoch.
          sample_list: An array containing generated samples at intervals.
        """

        if key is None: 
            key = self.key 

        key, subkey = jax.random.split(key) 
        num_samples = self.X_train.shape[1]
        losses = []
        sample_list = []
        # Shuffle training data along the sample axis (columns).
        permutation = jax.random.permutation(subkey, num_samples)
        X_train = self.X_train[:, permutation]
        for epoch in range(epochs):
            key, subkey = random.split(key) 
            epoch_loss = 0.0
            for i in range(0, num_samples, batch_size):
                batch = X_train[:, i:i+batch_size]
                if batch.shape[1] != batch_size: 
                    continue 
                key = self.train_batch_pcd(batch, learning_rate=learning_rate, k=k, l2_reg=l2_reg, key=subkey) # use it and return the key to advance state
                v_recon, key = self.reconstruct(batch, k, key)  
                epoch_loss += jnp.sum((batch - v_recon) ** 2)
            losses.append(epoch_loss)
            if (epoch * 10) % epochs == 0:
                print(f"Epoch {epoch}/{epochs}, Reconstruction Loss: {epoch_loss:.4f}")
                samples, key = self.generate(sample_number, gibbs_steps=1000, key=key) # self.generate() splits keys internally 
                sample_list.append(samples)
        return losses, jnp.array(sample_list), key
    
    def train_batch_pcd(self, v0, learning_rate=0.01, k=1, l2_reg=0.0, key=None):
        """
        Updates the RBM parameters using Persistent Contrastive Divergence (PCD).
        Instead of initializing the chain from the data v0 for the negative phase,
        a persistent chain is maintained and updated.
        
        Parameters:
          v0 (array): Batch of visible units with shape (n_visible, batch_size).
          learning_rate (float): Learning rate.
          k (int): Number of Gibbs steps to update the persistent chain.
          l2_reg (float): L2 regularization coefficient.
        """
        if key is None:
            key = self.key
        # Positive phase
        h0, _, key = self.sample_h(v0, key) # sample_h performs a split 
        # Negative phase
        vk, key = self.persistent_contrastive_divergence(k, batch_size=v0.shape[1], key=key)
        hk, _, key = self.sample_h(vk, key)
        
        batch_size = v0.shape[1]
        # Compute gradients using the difference between positive and negative phases.
        delta_W = jnp.dot(h0, v0.T) - jnp.dot(hk, vk.T)
        delta_v_bias = jnp.sum(v0 - vk, axis=1)
        delta_h_bias = jnp.sum(h0 - hk, axis=1)
        
        # Update parameters with learning rate and optional L2 regularization.
        self.W = self.W + learning_rate * (delta_W / batch_size - l2_reg * self.W)
        self.v_bias = self.v_bias + learning_rate * delta_v_bias / batch_size
        self.h_bias = self.h_bias + learning_rate * delta_h_bias / batch_size
        self.key = key  # Update class key
        return key

    def persistent_contrastive_divergence(self, k=1, batch_size=None, key=None):
        """Persistent chain with explicit key handling."""
        if key is None:
            key = self.key
        # Initialize persistent chain if needed
        if not hasattr(self, 'persistent_chain'):
            key, subkey = random.split(key)
            self.persistent_chain = random.bernoulli(
                subkey, 
                p=jnp.full((self.n_visible, batch_size), 0.5)
            ).astype(jnp.float32)
        
        # Update chain with fresh keys
        chain = self.persistent_chain
        for _ in range(k):
            h, _, key = self.sample_h(chain, key) # these methods split keys, so no need to split out here. 
            chain, _, key = self.sample_v(h, key)
        
        self.persistent_chain = chain
        return chain, key

    @staticmethod
    def compute_averages(samples):
        """
        Computes the mean (per variable) and covariance from the samples.
        Samples are expected to have shape (n_visible, n_samples).
        """
        return jnp.mean(samples, axis=1), jnp.cov(samples)

    def set_empirical_means(self):
        self.mean, self.cov = self.compute_averages(self.data)

    def compute_squared_deviations(self, samples):
        sample_mean, cov = self.compute_averages(samples)
        return (sample_mean - self.mean)**2, (cov - self.cov)**2

    def split_data_in_half(self, key):
        cols = self.data.shape[1]
        perm = jax.random.permutation(key, cols)
        fh_indices = perm[:cols // 2]
        sh_indices = perm[-cols // 2:]
        first_half_data = self.data[:, fh_indices]
        second_half_data = self.data[:, sh_indices]
        return first_half_data, second_half_data

    def background_model_squared_deviations(self, key):
        first_half_data, second_half_data = self.split_data_in_half(key)
        fhm, fhc = self.compute_averages(first_half_data)
        shm, shc = self.compute_averages(second_half_data)
        return (fhm - shm)**2, (fhc - shc)**2

    def compute_background_rmse(self, key):
        mean_diffs, cov_diffs = self.background_model_squared_deviations(key)
        return jnp.sqrt(jnp.mean(mean_diffs)), jnp.sqrt(jnp.mean(cov_diffs.flatten()))

    def compute_rmse(self, samples):
        mean_devs, cov_devs = self.compute_squared_deviations(samples)
        return jnp.sqrt(jnp.mean(mean_devs)), jnp.sqrt(jnp.mean(cov_devs.flatten()))

    def plot_deviations_over_time(self, train_args):
        losses, samples, key = self.fit(**train_args)
        fig, axs = plt.subplots(2, 1, height_ratios=[4, 1])
        errors = jnp.array(jax.vmap(self.compute_rmse)(samples)) 
        background_mean, background_cov = self.compute_background_rmse(key)
        axs[0].scatter(range(errors.shape[1]), errors[0, :], label='means rmse (samples vs data)') 
        axs[0].hlines(background_mean, 0, errors.shape[1] - 1, ls='--', color='tab:blue', label='means rmse (data vs data)')
        axs[0].scatter(range(errors.shape[1]), errors[1, :], label='covs rmse (samples vs data)') 
        axs[0].hlines(background_cov, 0, errors.shape[1] - 1, ls='--', color='tab:orange', label='covs rmse (data vs data)')
        axs[0].set_yscale('log')
        axs[1].plot(losses)
        axs[1].set_yscale('log') 
        axs[1].set_xlabel('epoch') 
        axs[1].set_ylabel('reconstruction loss')
        axs[0].legend()
        return fig, axs, samples, errors 
    
    def plot_samples(self, samples, indices=None): 
        # samples is assumed to be timepoints x num_variables x num_samples. indices is a subset of range(len(timepoints)), so you can plot a subset if needed. 
        if indices is None: 
            indices = range(len(samples))
        if len(indices) > 10: 
            indices = range(0, len(indices), len(indices) // 10) 
        fig, axs = plt.subplots(10, 10, figsize=(20, 20)) 
        [(ax.set_xticks([]), ax.set_yticks([])) for ax in axs.flatten()]
        for i in indices: 
            key, subkey = jax.random.split(self.key)
            sample_idx = jax.random.choice(subkey, np.arange(samples.shape[2]), shape=(10, ), replace=False)
            dim = int(jnp.sqrt(samples.shape[1]))
            for ax_idx, j in enumerate(sample_idx): 
                axs[i, ax_idx].imshow(samples[i, :, j].reshape(dim, dim), cmap='grey') 
        return fig, axs 
    