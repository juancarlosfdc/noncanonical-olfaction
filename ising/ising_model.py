import jax
import jax.numpy as jnp
import jax.random as jrandom
from jax import grad, value_and_grad, jit, vmap, lax, jit, tree_util, nn
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 14
from matplotlib.gridspec import GridSpec
import pandas as pd
from scipy.ndimage import maximum_filter
import matplotlib.colors as mcolors
import numpy as np

class OlfactionIsingModel:
    def __init__(self, data_path, key, fraction=1, shuffle=False):
        # Static attributes
        self.data_path = data_path
        self.key = key
        self.fraction = fraction
        self.shuffle = shuffle 
        self.data = self.load_data(key, fraction, shuffle) 
        # Compute derived attributes
        self.empirical_mean, self.empirical_cov = self.set_empirical_means()

        # Dynamic attributes (initialized later)
        self.beta = None
        self.h = None
        self.J = None
        self.burnin = None
        self.tau = None

    def set_parameters(self, beta, h, J, burnin, tau):
        self.beta = beta
        self.h = h
        self.J = J
        self.burnin = burnin
        self.tau = tau

    def _tree_flatten(self):
        children = (self.beta, self.h, self.J, self.burnin, self.tau)
        aux_data = {
            'data_path': self.data_path,
            'key': self.key,
            'fraction': self.fraction, 
            'shuffle': self.shuffle
        }
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        instance = cls(aux_data['data_path'], aux_data['key'], aux_data['fraction'], aux_data['shuffle']) 
        instance.set_parameters(*children)
        return instance
    
    def load_data(self, key, fraction, shuffle):
        data = pd.read_csv(self.data_path, index_col=0)        
        num_rows = data.shape[0]
        if fraction < 1: 
            indices = jax.random.choice(key, jnp.arange(num_rows), shape=(int(fraction * num_rows), ), replace=False)
        else: 
            indices = range(num_rows) 
        if shuffle: 
            data = data.apply(lambda row: pd.Series(np.random.permutation(row.values), index=row.index), axis=1)
        return data.iloc[indices, :]

    def energy(self, s, h, J): 
        return - (h @ s + 1/2 * s @ J @ s)
    
    def update(self, state, _):
        key, s = state
        key, subkey = jax.random.split(key)
        i = jax.random.choice(subkey, jnp.arange(len(s)))  # Randomly choose a spin
        # see https://sites.stat.washington.edu/mmp/courses/stat534/spring19/Handouts/lecture-may21.pdf for an easy overview of how conditional prob becomes sigmoid
        contribution = self.beta * (self.h[i] + self.J[i, :] @ s) 
        p_plus = nn.sigmoid(2 * contribution)
        key, subkey = jax.random.split(key)
        s = s.at[i].set(jnp.where(jax.random.uniform(subkey) < p_plus, 1, -1))
        return (key, s), s  # Return new state and spins
    
    def sample_from_ising_model(self, key, h, J, s_init=None, iters=1, beta=1):
        if s_init is None:
            s_init = jax.random.bernoulli(key, p=0.5, shape=(len(h),)) * 2 - 1  # Initialize spins (-1, 1)
        # Run Gibbs sampling with lax.scan
        self.beta, self.h, self.J = beta, h, J 
        (final_state, spins), spin_trajectory = jax.lax.scan(
            self.update, (key, s_init), None, length=iters)
        return spin_trajectory.T
    
    def compute_magnetization(self, vals): 
        return jnp.sum(vals, axis=0) 
    
    def tau_vs_beta(self, key, h, J, betas): 
        taus = []
        for beta in betas:
            vals = self.sample_from_ising_model(key, h, J, iters=1000, beta=beta)
            m = self.compute_magnetization(vals) 
            tau = self.estimate_autocorrelation_time(m)
            taus.append(tau) 
        return betas, taus
    
    def autocorrelation(self, chain, lag):
        # this can't be JIT compiled because length of array (or padding) is dynamic, depends on lag. 
        n = chain.shape[0]
        mean = jnp.mean(chain)
        var = jnp.var(chain)
        x1 = lax.dynamic_slice(chain, (0,), (n-lag,))
        x2 = lax.dynamic_slice(chain, (lag,), (n-lag,))
        return jnp.sum((x1 - mean) * (x2 - mean)) / (var * n)

    def plot_tau_vs_lag(self, chain, max_lag=200):
        lags = jnp.arange(max_lag)
        autocorrs = [self.autocorrelation(chain, lag) for lag in lags] 
        fig, ax = plt.subplots()
        ax.plot(lags, autocorrs)
        ax.set_xlabel("Lag")
        ax.set_ylabel("Autocorrelation")
        return fig, ax

    def estimate_autocorrelation_time(self, chain, max_lag=200):
        """Estimate the autocorrelation time."""
        autocorrs = [self.autocorrelation(chain, lag) for lag in range(max_lag)]
        for lag, corr in enumerate(autocorrs):
            if corr < 0.1:  # Threshold for approximate independence
                return lag
        return max_lag  # Return max_lag if no decorrelation is found
    
    def generate_independent_samples(self, key, h, J, samples, burnin=1000, tau=150):
        iterations = samples * tau + burnin 
        vals = self.sample_from_ising_model(key, h, J, iters=iterations) # remember beta is fixed to 1 here and abosrbed in the parameters. 
        independent_samples = vals[:, burnin::tau]
        return independent_samples

    def compute_MC_expectations(self, samples, h_MC, J_MC, h_current, J_current): # here we're using their clever importance-sampling-style histogram MC trick. see https://arxiv.org/pdf/0712.2437
        # return self.compute_averages(samples, weights=None, centered=False) 
        h_diff, J_diff = h_current - h_MC, J_current - J_MC 
        # should we normalize h_diff here so that the unnormalized weights are order 1? It's otherwise very easy for this to blow up badly. It would not blow up under coordinate descent: h_diff and J_diff would be near-maximally sparse. 
        # certainly you need to take small steps so that h_diff is small... maybe order 1 / P. No! It should be 1/N. 
        log_unnormalized_weights = h_diff @ samples + jnp.diagonal(samples.T @ J_diff @ samples)
        # jax.debug.print('log unnormalized weights = {mean}, {min}, {max}', mean=jnp.mean(log_unnormalized_weights), min=jnp.min(log_unnormalized_weights), max=jnp.max(log_unnormalized_weights))
        norm = jnp.linalg.norm(log_unnormalized_weights)
        norm = jnp.where(norm == 0, 1.0, norm)  # Avoid division by zero
        log_weights = log_unnormalized_weights / norm
        # alternatively: just divide by P to get a reasonable size. 
        log_weights = log_unnormalized_weights / 1 # samples.shape[1] 
        weights = jnp.exp(log_weights)
        # jax.debug.print('std_dev_weights = {s}', s=jnp.std(log_unnormalized_weights))
        Q_mean, Q_cov = self.compute_averages(samples, weights, centered=False) 
        return Q_mean, Q_cov

    @staticmethod 
    @jit 
    def compute_averages(samples, weights=None, centered=True):
        mean = jnp.average(samples, axis=1, weights=weights)
        # outer_product = jnp.einsum('ip,jp->ijp', samples, samples) # this tensor is N^2 x P entries so for N = 8000, P = 1000 it crashes badly. x 4 bytes it's 16 * 10*9 * 4 = 64 GB! 
        # cov = jnp.average(outer_product, axis=-1, weights=weights)
        cov = jax.lax.cond(centered, lambda _ : jnp.cov(m=samples, aweights=weights), lambda _ : jnp.cov(m=samples, aweights=weights) + jnp.einsum('i,j->ij', mean, mean), None) # uncentered covariance = centered covariance + product of means 
        return mean, cov
    
    def set_empirical_means(self, centered=False): 
        return self.compute_averages(self.data.values, centered=centered)

    def mask_top_k(self, arr, k, fill_value=0):
        # Get top-k indices
        flat_indices = jnp.argsort(jnp.abs(arr.ravel()))[::-1][:k]  # Indices of top k values (in magnitude) in flattened array
        mask = jnp.zeros(arr.size, dtype=bool).at[flat_indices].set(True)  # Boolean mask in flattened shape
        mask = mask.reshape(arr.shape)  # Reshape to 2D
        # Apply mask: keep top-k values, set others to fill_value
        return jnp.where(mask, arr, fill_value)

    def optimize_ising_model(self, key, h_init, J_init, eta, h_eta=None, J_eta=None, k=1, stages=1, iterations=1, num_samples=10000): 
        if h_eta is None: 
            h_eta = eta 
        if J_eta is None: 
            J_eta = eta 

        def inner_update(carry, _):
            h, J, samples, h_MC, J_MC = carry  # Include MC parameters in carry
            Q_mean, Q_cov = self.compute_MC_expectations(samples, h_MC, J_MC, h, J) # this is the uncentered second moment, which is what we're trying to match. 
            grad_h = Q_mean - self.empirical_mean
            grad_J = Q_cov - self.empirical_cov
            # jax.debug.print('Q_mean = {q}\nempirical_mean = {e}', q=Q_mean[:2], e=self.empirical_mean[:2])
            # jax.debug.print('Q_cov = {q}\nempirical_cov = {e}', q=Q_cov[:2, :2], e=self.empirical_cov[:2, :2])
            h_new = h - 0 * h_eta * grad_h
            J_new = J - J_eta * self.mask_top_k(grad_J, k) * (1 - jnp.eye(J.shape[0])) # always keep diagonals pegged to 0! 
            return (h_new, J_new, samples, h_MC, J_MC), (h_new, J_new)  # Collect trajectory

        def outer_update(carry, _):
            key, h_MC, J_MC = carry
            key, subkey = jax.random.split(key)
            # jax.debug.print('h_MC = {h}\nJ_MC={J}', h=h_MC, J=J_MC)
            samples = self.generate_independent_samples(subkey, h_MC, J_MC, samples=num_samples, burnin=self.burnin, tau=self.tau) 
            binarized_samples = (samples + 1) // 2
            (h_final, J_final, _, _, _), (h_traj, J_traj) = jax.lax.scan(
                inner_update, (h_MC, J_MC, binarized_samples, h_MC, J_MC), None, length=iterations
            )
            return (key, h_final, J_final), (h_traj, J_traj)

        (key, h_final, J_final), (h_trajectory, J_trajectory) = jax.lax.scan(
            outer_update, (key, h_init, J_init), None, length=stages
        )

        return h_final, J_final, h_trajectory, J_trajectory
    
    def rolling_max_2d(self, data, window):
        # Apply a maximum filter with a square kernel of size `window x window`
        return maximum_filter(data, size=window, mode='constant', cval=0)
    
    def plot_samples(self, samples, window=5): 
        fig, ax = plt.subplots(figsize=(8, 6), dpi=600)
        fattened_data = self.rolling_max_2d(samples, window)
        cmap = mcolors.ListedColormap(['white', 'black'])
        ax.imshow(fattened_data, aspect='auto', cmap=cmap)
        return fig, ax
    
    def background_model_differences(self, key): 
        data = self.data.values
        cols = data.shape[1]
        perm = jax.random.permutation(key, cols)
        fh_indices = perm[:cols // 2]
        sh_indices = perm[-cols//2:]
        first_half_data = data[:, fh_indices]
        second_half_data = data[:, sh_indices]
        fhm, fhc = self.compute_averages(first_half_data)
        shm, shc = self.compute_averages(second_half_data)
        return fhm - shm, (fhc - shc).flatten()

    def QC_samples(self, key, samples, mode='scatter'):
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        background_mean_diffs, background_cov_diffs = self.background_model_differences(key)
        sample_mean, sample_cov = self.compute_averages(samples) 
        sample_mean_diffs, sample_cov_diffs = sample_mean - self.empirical_mean, (sample_cov - self.empirical_cov).flatten()
        if mode=='scatter': 
            axs[0, 0].scatter(range(len(background_mean_diffs)), jnp.sort(background_mean_diffs))
            axs[0, 1].scatter(range(len(background_cov_diffs)), jnp.sort(background_cov_diffs)) 
            axs[1, 0].scatter(range(len(sample_mean_diffs)), jnp.sort(sample_mean_diffs))
            axs[1, 1].scatter(range(len(sample_cov_diffs)), jnp.sort(sample_cov_diffs))
        elif mode=='hist': 
            axs[0, 0].hist(background_mean_diffs)
            axs[0, 1].hist(background_cov_diffs) 
            axs[1, 0].hist(sample_mean_diffs)
            axs[1, 1].hist(sample_cov_diffs)
        else:
            pass 
        return fig, axs, sample_mean_diffs, sample_cov_diffs
    

    def compute_average_deviations(self, samples): 
        sample_mean, sample_cov = self.compute_averages(samples, centered=False) 
        sample_mean_diffs, sample_cov_diffs = sample_mean - self.empirical_mean, (sample_cov - self.empirical_cov).flatten()
        return jnp.sqrt(jnp.average(sample_mean_diffs**2)), jnp.sqrt(jnp.average(sample_cov_diffs**2))
    
    def compute_background_average_deviations(self, key): 
        means, covs = self.background_model_differences(key)
        return jnp.sqrt(jnp.average(means**2)), jnp.sqrt(jnp.average(covs**2))
    
    def compute_loss(self, key, h_baseline, J_baseline, h, J, samples): 
        pass # lol--need to estimate log of a partition function! 

    # def estimate_deviation(self, key, h, J): 
    #     samples = self.generate_independent_samples()

    # def estimate_deviation_timecourse(self, key, h_traj, J_traj, subsample=1): 
    #     stages, iterations, N = h_traj.shape
    #     hs = h_traj.reshape(stages * iterations, N)
    #     Js = J_traj.reshape(stages * iterations, N, N)
    #     means, covs = [], []
    #     for i in range(0, stages * iterations, subsample):
    #         key, subkey = jax.random.split(key) 
    #         samples = self.generate_independent_samples(subkey, hs[i], Js[i], samples=self.data.shape[1] // 2, burnin=self.burnin, tau=self.tau)
    #         m, c = self.compute_average_deviations((samples + 1) // 2)
    #         means.append(m)
    #         covs.append(c) 
    #     return means, covs 


    def estimate_deviation_timecourse(self, key, h_traj, J_traj, subsample=1):
        stages, iterations, N = h_traj.shape
        flattened_h = h_traj.reshape(stages * iterations, N)
        flattened_J = J_traj.reshape(stages * iterations, N, N)

        # Subsample the indices
        subsampled_h = flattened_h[::subsample]
        subsampled_J = flattened_J[::subsample]
        
        # Generate random keys for each sample
        keys = jax.random.split(key, len(subsampled_h))

        # Vectorized sample generation
        def generate_samples(k, h, J):
            return self.generate_independent_samples(
                k, h, J, 
                samples=10 * self.data.shape[1] // 2, 
                burnin=self.burnin, 
                tau=self.tau
            )

        samples = jax.vmap(generate_samples)(keys, subsampled_h, subsampled_J)

        # Convert spin configurations from {-1,1} to {0,1}
        binarized_samples = (samples + 1) // 2
        # return binarized_samples 
        # Vectorized computation of deviations
        means, covs = jax.vmap(self.compute_average_deviations)(binarized_samples)

        return np.array(means), np.array(covs)

    def plot_deviation_timecourse(self, key, h_traj, J_traj, subsample=1): 
        m, c = self.estimate_deviation_timecourse(key, h_traj, J_traj, subsample=subsample) 
        fig, ax = plt.subplots()
        ax.plot(m, label='mean deviation') 
        ax.plot(c, label='covariance deviation') 
        mfloor, cfloor = self.compute_background_average_deviations(key) 
        ax.hlines(mfloor, xmin=ax.get_xlim()[0], xmax=ax.get_xlim()[1], ls='--', label='noise floor', color='tab:blue') 
        ax.hlines(cfloor, xmin=ax.get_xlim()[1], xmax=ax.get_xlim()[1], ls='--', label='noise floor', color='tab:orange') 
        return fig, ax, m, c 


    # def optimize_ising_model(self, key, h_init, J_init, eta, stages=1, iterations=1): 
    #     h_MC = h_init 
    #     J_MC = J_init 
    #     h = h_init 
    #     J = J_init 
    #     h_traj, J_traj = [], []
    #     for s in range(stages):
    #         samples = self.generate_independent_samples(key, h_MC, J_MC, samples=10000)
    #         for t in range(iterations): 
    #             Q_mean, Q_cov = self.compute_MC_expectations((samples + 1) // 2, h_MC, J_MC, h, J) 
    #             grad_h = Q_mean - self.empirical_mean
    #             grad_J = Q_cov - self.empirical_cov 
    #             h -= eta * grad_h 
    #             J -= 0 * eta * grad_J 
    #         h_MC, J_MC = h, J 
    #         h_traj.append(h_MC)
    #         J_traj.append(J_MC)
    #     return h_MC, J_MC, jnp.array(h_traj), jnp.array(J_traj) 
    
    def fit_h_to_empirical_means(self): 
        # this function fits the field parameters h to match the mean of each spin by inverting the sigmoid function
        h = 1/2 * jnp.log(self.empirical_mean / (1 - self.empirical_mean))
        return h 

tree_util.register_pytree_node(OlfactionIsingModel,
                               OlfactionIsingModel._tree_flatten,
                               OlfactionIsingModel._tree_unflatten)