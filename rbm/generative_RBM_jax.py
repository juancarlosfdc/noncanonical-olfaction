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
from jax.experimental import io_callback 
import os 
import re 
import glob

class GenerativeRBM:
    def __init__(self, key, n_hidden, batch_size=64, data_path=None, W_scale=0.01, digits=None):
        """
        Initializes the RBM with the given number of hidden units.
        """
        self.data = self.load_data(data_path, digits)
        self.set_empirical_means()
        self.X_train = self.data
        self.n_visible = self.data.shape[0]  # each row is a variable
        self.n_hidden = n_hidden

        # Initialize weights and biases
        key, subkey = random.split(key)
        self.W = random.normal(subkey, shape=(n_hidden, self.n_visible)) * W_scale
        self.h_bias = jnp.zeros((n_hidden,))
        self.v_bias = self.initialize_v_bias()
        self.batch_size = batch_size
        key, subkey = random.split(key) 
        self.persistent_chain = jax.random.bernoulli(subkey, p=0.5, shape=(self.n_visible, batch_size)).astype(float) 

    def load_data(self, data_path, digits):
        if data_path is None:
            mnist = fetch_openml('mnist_784', version=1)
            data = mnist.data
            if digits is not None: 
                data = mnist.data[mnist.target.isin([str(d) for d in digits])] 
            # Original shape is (n_samples, n_features). Transpose to have shape (n_features, n_samples).
            X = jnp.array(data.T, dtype=np.float32) / 255.0  # Normalize to [0,1]
            print("Data shape:", X.shape)
            X = (X > 0.5).astype(np.float32)  # Binarize the data
        else:
            X = jnp.array(pd.read_csv(data_path, index_col=0).values, dtype=jnp.float32) 
        return X

    def _tree_flatten(self):
        """Flatten RBM parameters for use in JAX transformations."""
        dynamic = (self.W, self.h_bias, self.v_bias, self.persistent_chain)
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
        W, h_bias, v_bias, persistent_chain = dynamic
        # obj = cls(n_hidden=static["n_hidden"])
        obj = cls.__new__(cls) 
        obj.n_visible = static["n_visible"]
        obj.data = static["data"]
        obj.mean = static["mean"]
        obj.cov = static["cov"]
        obj.W = W
        obj.h_bias = h_bias
        obj.v_bias = v_bias
        obj.persistent_chain = persistent_chain
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

    def sample_h(self, v, W, h_bias, subkey):
        """Sample hidden units."""
        p_h = self.sigmoid(jnp.dot(W, v) + h_bias[:, None])
        h = random.bernoulli(subkey, p_h).astype(jnp.float32)
        return h, p_h

    def sample_v(self, h, W, v_bias, subkey):
        """Sample visible units."""
        p_v = self.sigmoid(jnp.dot(W.T, h) + v_bias[:, None])
        v = random.bernoulli(subkey, p_v).astype(jnp.float32)
        return v, p_v
    
    def generate(self, key, n_samples, W, v_bias, h_bias, gibbs_steps=100):
        """Generate samples using Gibbs sampling with JAX-friendly lax.scan."""

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

    def contrastive_divergence(self, key, v0, W, h_bias, v_bias, k=1):
        def body_fn(carry, _):
            vk, key = carry
            key, subkey_h, subkey_v = random.split(key, 3)
            h, _ = self.sample_h(vk, W, h_bias, subkey_h)
            vk, _ = self.sample_v(h, W, v_bias, subkey_v)
            return (vk, key), None
        (vk, key), _ = jax.lax.scan(body_fn, (v0, key), None, length=k)
        return vk, key

    def persistent_contrastive_divergence(self, key, batch_size, W, h_bias, v_bias, persistent_chain, k=1):
        """Persistent chain with explicit key handling."""

        def gibbs_step(carry, _):
            samples, key = carry
            key, h_key, v_key = random.split(key, 3)
            h, _ = self.sample_h(samples, W, h_bias, h_key)
            new_samples, _ = self.sample_v(h, W, v_bias, v_key)
            return (new_samples, key), None

        (chain, key), _ = jax.lax.scan(gibbs_step, (persistent_chain, key), None, length=k)
        return chain, key

    def train_batch_pcd(self, key, v0, W, h_bias, v_bias, persistent_chain, learning_rate=0.01, k=1, l2_reg=0.0):
        """Updates the RBM parameters using PCD."""
        key, h0key, hkkey = jax.random.split(key, 3) 
        h0, _ = self.sample_h(v0, W, h_bias, h0key)
        vk, key = self.persistent_contrastive_divergence(key, v0.shape[1], W, h_bias, v_bias, persistent_chain, k)
        hk, _ = self.sample_h(vk, W, h_bias, hkkey)

        batch_size = v0.shape[1]
        delta_W = jnp.dot(h0, v0.T) - jnp.dot(hk, vk.T)
        delta_v_bias = jnp.sum(v0 - vk, axis=1)
        delta_h_bias = jnp.sum(h0 - hk, axis=1)

        new_W = W + learning_rate * (delta_W / batch_size - l2_reg * W)
        new_v_bias = v_bias + learning_rate * delta_v_bias / batch_size
        new_h_bias = h_bias + learning_rate * delta_h_bias / batch_size

        return new_W, new_v_bias, new_h_bias, vk, key

    def update(self, W, v_bias, h_bias, persistent_chain):
        self.W, self.v_bias, self.h_bias, self.persistent_chain = W, v_bias, h_bias, persistent_chain

    def fit(self, key, output_dir, scans=1, epochs_per_scan=10, batch_size=64, learning_rate=0.01, k=1, 
        l2_reg=0.0, sample_number=1000):
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

        # Shuffle training data using JAX ops
        key, subkey = jax.random.split(key)
        num_samples = self.X_train.shape[1]
        permutation = jax.random.permutation(subkey, num_samples)
        X_train = self.X_train[:, permutation]
        init_persistent_chain = jax.random.bernoulli(subkey, p=0.5, shape=(self.n_visible, batch_size)).astype(float) 
        
        num_batches = num_samples // batch_size
        batches = jnp.stack([X_train[:, i * batch_size:(i + 1) * batch_size] for i in range(num_batches)], axis=0)

        epochs = epochs_per_scan * scans

        # Define batch step function for `lax.scan`
        def batch_step(carry, batch):
            W, v_bias, h_bias, persistent_chain, key = carry
            W, v_bias, h_bias, persistent_chain, key = self.train_batch_pcd(
                key, batch, W, h_bias, v_bias, persistent_chain, learning_rate, k, l2_reg
            )
            v_recon, key = self.contrastive_divergence(key, batch, W, h_bias, v_bias, k)
            loss = jnp.sum((batch - v_recon) ** 2)
            return (W, v_bias, h_bias, persistent_chain, key), loss
        
        # Define epoch step function for `lax.scan`
        def epoch_step(carry, epoch):
            W, v_bias, h_bias, persistent_chain, key = carry
            
            key, subkey = jax.random.split(key)

            # Scan over batches
            (W, v_bias, h_bias, persistent_chain, key), batch_losses = jax.lax.scan(
                batch_step, (W, v_bias, h_bias, persistent_chain, subkey), batches
            )
            epoch_loss = jnp.sum(batch_losses)

            def write_samples(samples, epoch):
                if not os.path.isdir(f'{output_dir}/samples'): 
                    os.mkdir(f'{output_dir}/samples')
                jnp.save(f'{output_dir}/samples/samples_{epoch}', samples)

            # optionally write samples every few epochs. 
            def gen_samples(key):
                key, subkey = jax.random.split(key)
                samples, key = self.generate(subkey, sample_number, W, v_bias, h_bias, gibbs_steps=1000)
                io_callback(write_samples, None, samples, scan * epochs_per_scan + epoch)
                jax.debug.print("epoch: {e} reconstruction error: {r}", e=epoch, r=epoch_loss)
                return key 

            key = jax.lax.cond(
                (epoch * 10) % epochs == 0, gen_samples, lambda x: x, key
            )

            return (W, v_bias, h_bias, persistent_chain, key), (epoch_loss)

        # there are two options. we can loop over epochs (slow but low mem), scan over epochs (fast but high mem). 

        # the third answer: loop over scans! That's fast and tolerable mem. 

        current_state = (self.W, self.v_bias, self.h_bias, init_persistent_chain, key)
        losses_list = []

        for scan in range(scans): 
            # Run a scan for scan_length epochs
            current_state, epoch_results = jax.lax.scan(
                epoch_step, current_state, scan * epochs_per_scan + jnp.arange(epochs_per_scan)
            )
            # Unpack epoch results: this is the second tuple returned by epoch_step. 
            losses =  epoch_results
            losses_list.append(losses)

        losses = jnp.concatenate(losses_list, axis=0)

        # Create a new updated RBM state
        W, v_bias, h_bias, persistent_chain, key = current_state
        self.update(W=W, v_bias=v_bias, h_bias=h_bias, persistent_chain=persistent_chain)

        return jnp.array(losses), key

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

    def read_samples(self, output_dir): 
        files = glob.glob(f'{output_dir}/samples/*.npy')
        files_sorted = sorted(files, key=lambda x: int(re.findall(r'samples_(\d+)\.npy', x)[0]))
        # Read each file and store in a list
        samples_list = [jnp.load(f) for f in files_sorted]
        print('reading sample files:', files_sorted) 
        concatenated_samples = jnp.stack(samples_list)
        return concatenated_samples
    
    def plot_deviations_over_time(self, output_dir, train_args):
        current_samples = glob.glob(f'{output_dir}/samples/*')
        for cs in current_samples: 
            os.remove(cs) 
        losses, key = self.fit(output_dir=output_dir, **train_args)
        samples = self.read_samples(output_dir)
        fig, axs = plt.subplots(2, 1, height_ratios=[4, 1])
        errors = jnp.array(jax.vmap(self.compute_rmse)(samples)) 
        key, subkey = jax.random.split(key) 
        background_mean, background_cov = self.compute_background_rmse(subkey)
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
    
    def plot_samples(self, key, samples, indices=None): 
        # samples is assumed to be timepoints x num_variables x num_samples. indices is a subset of range(len(timepoints)), so you can plot a subset if needed. 
        if indices is None: 
            indices = range(len(samples))
        if len(indices) > 10: 
            indices = range(0, len(indices), len(indices) // 10) 
        fig, axs = plt.subplots(10, 10, figsize=(20, 20)) 
        [(ax.set_xticks([]), ax.set_yticks([])) for ax in axs.flatten()]
        for i in indices: 
            key, subkey = jax.random.split(key)
            sample_idx = jax.random.choice(subkey, np.arange(samples.shape[2]), shape=(10, ), replace=False)
            dim = int(jnp.sqrt(samples.shape[1]))
            for ax_idx, j in enumerate(sample_idx): 
                axs[i, ax_idx].imshow(samples[i, :, j].reshape(dim, dim), cmap='grey') 
        return fig, axs 
    

from jax import tree_util
tree_util.register_pytree_node(GenerativeRBM,
                               GenerativeRBM._tree_flatten,
                               GenerativeRBM._tree_unflatten)