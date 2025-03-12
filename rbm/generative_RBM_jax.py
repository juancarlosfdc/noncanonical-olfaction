import jax
import jax.numpy as jnp
from jax import random
from jax import jit 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import pandas as pd
from functools import partial

class GenerativeRBM:
    def __init__(self, n_hidden, batch_size=64, data_path=None, key=None, W_scale=0.01, digits=None):
        """
        Initializes the RBM with the given number of hidden units.
        """
        if key is None:
            key = random.PRNGKey(0)
        self.key = key
        self.data = self.load_data(data_path, digits)
        self.set_empirical_means()
        self.X_train = self.data
        self.n_visible = self.data.shape[0]  # each row is a variable
        self.n_hidden = n_hidden
        self.key, subkey = random.split(self.key)

        # Initialize weights and biases
        self.W = random.normal(subkey, shape=(n_hidden, self.n_visible)) * W_scale
        self.h_bias = jnp.zeros((n_hidden,))
        self.v_bias = self.initialize_v_bias()
        self.batch_size = batch_size
        self.persistent_chain = jax.random.bernoulli(self.key, p=0.5, shape=(self.n_visible, batch_size)).astype(float) 

    def load_data(self, data_path, digits):
        if data_path is None:
            mnist = fetch_openml('mnist_784', version=1)
            data = mnist.data
            if digits is not None: 
                data = mnist.data[mnist.target.isin([str(d) for d in digits])] 
            # Original shape is (n_samples, n_features). Transpose to have shape (n_features, n_samples).
            X = np.array(data.T, dtype=np.float32) / 255.0  # Normalize to [0,1]
            print("Data shape:", X.shape)
            X = (X > 0.5).astype(np.float32)  # Binarize the data
        else:
            X = pd.read_csv(data_path, index_col=0).values
        X_train, X_test = train_test_split(X.T, test_size=0.2, random_state=int(self.key[0]))
        return X

    def _tree_flatten(self):
        """Flatten RBM parameters for use in JAX transformations."""
        dynamic = (self.W, self.h_bias, self.v_bias, self.persistent_chain, self.key)
        static = {
            "n_visible": self.n_visible,
            "n_hidden": self.n_hidden,
            "data": self.data, 
            "mean": self.mean, 
            "cov": self.cov
        }
        return dynamic, static

    @classmethod
    def _tree_unflatten(cls, static, dynamic):
        """Reconstruct an RBM object from its flattened components."""
        W, h_bias, v_bias, persistent_chain, key = dynamic
        obj = cls(n_hidden=static["n_hidden"])
        obj.n_visible = static["n_visible"]
        obj.data = static["data"]
        obj.mean = static["mean"]
        obj.cov = static["cov"]
        obj.W = W
        obj.h_bias = h_bias
        obj.v_bias = v_bias
        obj.persistent_chain = persistent_chain
        obj.key = key
        return obj
    
    def initialize_v_bias(self):
        v_probs = jnp.sum(self.data, axis=1) / self.data.shape[1]
        v_bias_init = jnp.log((v_probs + 1e-6) / (1 - (v_probs + 1e-6)))
        return v_bias_init
    
    def set_W_to_PCs(self): 
        _, evec = jnp.linalg.eig(self.data @ self.data.T)
        self.W = jnp.real(evec[:, :self.n_hidden].T)

    def sigmoid(self, x):
        return 1.0 / (1.0 + jnp.exp(-x))

    def sample_h(self, v, W, h_bias, key):
        """Sample hidden units."""
        p_h = self.sigmoid(jnp.dot(W, v) + h_bias[:, None])
        h = random.bernoulli(key, p_h).astype(jnp.float32)
        return h, p_h

    def sample_v(self, h, W, v_bias, key):
        """Sample visible units."""
        p_v = self.sigmoid(jnp.dot(W.T, h) + v_bias[:, None])
        v = random.bernoulli(key, p_v).astype(jnp.float32)
        return v, p_v
    
    def generate(self, n_samples, W, v_bias, h_bias, gibbs_steps=100, key=None):
        """Generate samples using Gibbs sampling with JAX-friendly lax.scan."""
        if key is None:
            key = self.key  # Use class key if none provided

        # Initialize PRNG keys
        key, init_key = jax.random.split(key)

        # Initialize samples (random binary states)
        samples = random.bernoulli(init_key, p=0.5, shape=(self.n_visible, n_samples)).astype(jnp.float32)

        # Gibbs sampling step function for `lax.scan`
        def gibbs_step(carry, _):
            samples, key = carry
            key, h_key, v_key = jax.random.split(key, 3)
            h, _ = self.sample_h(samples, W, h_bias, h_key)
            new_samples, _ = self.sample_v(h, W, v_bias, v_key)
            return (new_samples, key), None

        # Run Gibbs sampling using `lax.scan`
        (final_samples, key), _ = jax.lax.scan(gibbs_step, (samples, key), None, length=gibbs_steps)

        return final_samples, key  # Return final samples and updated key


    def contrastive_divergence(self, v0, W, h_bias, v_bias, k=1, key=None):
        def body_fn(carry, _):
            vk, key = carry
            key, subkey_h, subkey_v = random.split(key, 3)
            h, _ = self.sample_h(vk, W, h_bias, subkey_h)
            vk, _ = self.sample_v(h, W, v_bias, subkey_v)
            return (vk, key), None
        (vk, key), _ = jax.lax.scan(body_fn, (v0, key), None, length=k)
        return vk, key

    def persistent_contrastive_divergence(self, batch_size, W, h_bias, v_bias, k=1, key=None):
        """Persistent chain with explicit key handling."""
        if key is None:
            key = self.key
        # Initialize persistent chain if needed
        if self.persistent_chain is None:
            key, subkey = random.split(key)
            persistent_chain = random.bernoulli(subkey, p=0.5, shape=(self.n_visible, batch_size)).astype(jnp.float32)
        else:
            persistent_chain = self.persistent_chain

        def gibbs_step(carry, _):
            samples, key = carry
            key, h_key, v_key = random.split(key, 3)
            h, _ = self.sample_h(samples, W, h_bias, h_key)
            new_samples, _ = self.sample_v(h, W, v_bias, v_key)
            return (new_samples, key), None

        (chain, key), _ = jax.lax.scan(gibbs_step, (persistent_chain, key), None, length=k)
        return chain, key

    def train_batch_pcd(self, v0, W, h_bias, v_bias, learning_rate=0.01, k=1, l2_reg=0.0, key=None):
        """Updates the RBM parameters using PCD."""
        if key is None:
            key = self.key
        key, h0key, hkkey = jax.random.split(key, 3) 
        h0, _ = self.sample_h(v0, W, h_bias, h0key)
        vk, key = self.persistent_contrastive_divergence(v0.shape[1], W, h_bias, v_bias, k, key)
        hk, _ = self.sample_h(vk, W, h_bias, hkkey)

        batch_size = v0.shape[1]
        delta_W = jnp.dot(h0, v0.T) - jnp.dot(hk, vk.T)
        delta_v_bias = jnp.sum(v0 - vk, axis=1)
        delta_h_bias = jnp.sum(h0 - hk, axis=1)

        new_W = W + learning_rate * (delta_W / batch_size - l2_reg * W)
        new_v_bias = v_bias + 0 * learning_rate * delta_v_bias / batch_size
        new_h_bias = h_bias + learning_rate * delta_h_bias / batch_size

        return new_W, new_v_bias, new_h_bias, key, vk

    def update(self, W, v_bias, h_bias, key, persistent_chain):
        self.W, self.v_bias, self.h_bias, self.key, self.persistent_chain = W, v_bias, h_bias, key, persistent_chain

    def fit(self, epochs=10, batch_size=64, learning_rate=0.01, k=1, 
        l2_reg=0.0, sample_number=1000, key=None):
        """
        Trains the RBM on the training data using jax.lax.scan for the loops.

        Parameters:
        epochs (int): Number of epochs.
        batch_size (int): Number of samples per batch.
        learning_rate (float): Learning rate.
        k (int): Number of Gibbs sampling steps per batch.
        l2_reg (float): L2 regularization coefficient.
        sample_number (int): Number of samples to generate at intervals.
        key: JAX PRNG key.

        Returns:
        losses: Array of reconstruction losses per epoch.
        final_state: Updated RBM state.
        """
        if key is None:
            key = self.key

        # Shuffle training data using JAX ops
        key, subkey = jax.random.split(key)
        num_samples = self.X_train.shape[1]
        permutation = jax.random.permutation(subkey, num_samples)
        X_train = self.X_train[:, permutation]
        
        num_batches = num_samples // batch_size
        batches = jnp.stack([X_train[:, i * batch_size:(i + 1) * batch_size] for i in range(num_batches)], axis=0)

        # Define batch step function for `lax.scan`
        def batch_step(carry, batch):
            W, v_bias, h_bias, key, persistent_chain = carry
            W, v_bias, h_bias, key, persistent_chain = self.train_batch_pcd(
                batch, W, h_bias, v_bias, learning_rate, k, l2_reg, key
            )
            v_recon, key = self.contrastive_divergence(batch, W, h_bias, v_bias, k, key)
            loss = jnp.sum((batch - v_recon) ** 2)
            return (W, v_bias, h_bias, key, persistent_chain), loss

        # Define epoch step function for `lax.scan`
        def epoch_step(carry, epoch):
            W, v_bias, h_bias, key, persistent_chain = carry
            key, subkey = jax.random.split(key)

            # Scan over batches
            (W, v_bias, h_bias, key, persistent_chain), batch_losses = jax.lax.scan(
                batch_step, (W, v_bias, h_bias, key, persistent_chain), batches
            )
            epoch_loss = jnp.sum(batch_losses)

            # Optionally generate samples every few epochs
            def gen_samples(key):
                key, subkey = jax.random.split(key)
                samples, key = self.generate(sample_number, W, v_bias, h_bias, gibbs_steps=1000, key=subkey)
                return key, samples

            def no_samples(key):
                return key, jnp.zeros((self.n_visible, sample_number))

            key, samples = jax.lax.cond(
                (epoch * 10) % epochs == 0, gen_samples, no_samples, key
            )

            return (W, v_bias, h_bias, key, persistent_chain), (epoch_loss, samples, W, v_bias, h_bias)

        # Scan over epochs
        (W, v_bias, h_bias, key, persistent_chain), epoch_results = jax.lax.scan(
            epoch_step, (self.W, self.v_bias, self.h_bias, key, self.persistent_chain), jnp.arange(epochs)
        )

        # Unpack results
        losses, samples, W, v_bias, h_bias = epoch_results

        def drop_placeholder(samples): 
            index = jnp.arange(epochs)
            return samples[index * 10 % epochs == 0]
        
        non_zero_samples = drop_placeholder(samples) 
        # Create a new updated RBM state
        self.update(W=W, v_bias=v_bias, h_bias=h_bias, key=key, persistent_chain=persistent_chain)

        return jnp.array(losses), non_zero_samples, W, v_bias, h_bias, key


    @staticmethod
    @jit 
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
        losses, samples, W, v_bias, h_bias, key = self.fit(**train_args)
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
        return fig, axs, samples, errors, W, v_bias, h_bias
    
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
    

from jax import tree_util
tree_util.register_pytree_node(GenerativeRBM,
                               GenerativeRBM._tree_flatten,
                               GenerativeRBM._tree_unflatten)