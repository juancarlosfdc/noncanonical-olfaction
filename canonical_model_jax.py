# canonical-model-jax.py
# # note: jax and classes are tricky
# still worth it imo 
# https://jax.readthedocs.io/en/latest/faq.html#strategy-1-jit-compiled-helper-function

# %%
import jax
import jax.numpy as jnp
from jax import grad, value_and_grad, jit, vmap
from jax.scipy.stats import norm, rankdata
from jax.scipy.special import digamma
from jax.scipy.special import gamma
from jax.scipy.linalg import cho_factor, cho_solve
from jax.example_libraries import optimizers
from jax.scipy.linalg import cholesky
from jax import lax 
from scipy.stats import differential_entropy
import matplotlib.pyplot as plt
from jax import jit, tree_util
from jax import nn 
plt.rcParams['font.size'] = 14
from matplotlib.gridspec import GridSpec

# %%
class OlfactorySensing:
    def __init__(self, N=100, n=2, M=30, P=1000, sigma_0=1e-2, sigma_c=2.0): 
        self.N = N
        self.n = n
        self.M = M
        self.P = P
        self.sigma_0 = sigma_0
        self.sigma_c = sigma_c
        self.set_sigma()
        self.W = None  # Initialize W as None; it may be set later with set_random_W
        self.vasicek_window = None  # This will be set when draw_cs is called

    def _tree_flatten(self):
        # Treat `W` as a dynamic value, while the rest are static
        children = (self.W,)  # W is the only dynamic value
        aux_data = {
            'N': self.N,
            'n': self.n,
            'M': self.M,
            'P': self.P,
            'sigma_0': self.sigma_0,
            'sigma_c': self.sigma_c,
            'vasicek_window': self.vasicek_window,
        }
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        # Recreate an instance of OlfactorySensing from children and aux_data
        instance = cls(
            N=aux_data['N'],
            n=aux_data['n'],
            M=aux_data['M'],
            P=aux_data['P'],
            sigma_0=aux_data['sigma_0'],
            sigma_c=aux_data['sigma_c']
        )
        instance.W = children[0]
        instance.vasicek_window = aux_data['vasicek_window']
        return instance

    def set_sigma(self): 
        self.sigma = lambda x: x / (1 + x) 

    def draw_c(self, key): 
        c = jnp.zeros(self.N)
        non_zero_indices = jax.random.choice(key, self.N, shape=(self.n,), replace=False)
        concentrations = jax.random.lognormal(key, sigma=self.sigma_c, shape=(self.n,))
        c = c.at[non_zero_indices].set(concentrations)
        return c

    def draw_cs(self, key):
        keys = jax.random.split(key, self.P)
        self.vasicek_window = jax.lax.stop_gradient(jnp.round(jnp.sqrt(self.P) + 0.5)).astype(int)
        return jnp.array([self.draw_c(k) for k in keys]).T

    def set_random_W(self, key): 
        self.W = 1 / jnp.sqrt(self.N) * jax.random.normal(key, shape=(self.M, self.N))

    def compute_activity(self, W, c, key): 
        pre_activations = W @ c
        r = self.sigma(pre_activations) + self.sigma_0 * jax.random.normal(key, shape=pre_activations.shape) 
        return r

    # @jit # This might take hours to compile.  https://jax.readthedocs.io/en/latest/control-flow.html#control-flow is helpful for the following
    def compute_entropy_of_r(self, W, c, key):
        r = self.compute_activity(W, c, key)
        entropy = self.compute_sum_of_marginal_entropies(r) - self.compute_information_of_r(r)
        return entropy

    # @jit 
    def compute_sum_of_marginal_entropies(self, r):
        compute_entropy_vmap = vmap(self._vasicek_entropy, in_axes=0)
        # Apply the vectorized function
        marginal_entropies = compute_entropy_vmap(r)
        # Sum the marginal entropies
        return jnp.sum(marginal_entropies)

    # @jit 
    def compute_information_of_r(self, r): 
        M, P = r.shape
        G = norm.ppf((rankdata(r.T, axis=0) / (P + 1)), loc=0, scale=1) # this is just ranking the data and making it normally distributed. 
        bias_correction = 0.5 * jnp.sum(digamma((P - jnp.arange(1, M + 1) + 1) / 2) - jnp.log(P / 2)) 
        cov_matrix = jnp.cov(G, rowvar=False)
        chol_decomp = cholesky(cov_matrix)
        log_det = jnp.sum(jnp.log(jnp.diag(chol_decomp)))
        I = -(log_det - bias_correction) # remember: entropy overall is sum of marginals minus information. information is sum of marginals - entropy. Kind of stupid. 
        return I
    
    def sum_covariances(self, W, c, key): 
        r = self.compute_activity(W, c, key)
        cov_r = jnp.cov(r) 
        off_diag_mask = ~jnp.eye(cov_r.shape[0], dtype=bool)
        # Extract the off-diagonal elements
        off_diag_elements = cov_r[off_diag_mask]
        # Compute the Frobenius norm of the off-diagonal elements
        frob_norm = jnp.sqrt(jnp.sum(off_diag_elements**2))
        return frob_norm 
    
    def log_det_sigma(self, W, c, key):
        r = self.compute_activity(W, c, key)
        cov_r = jnp.cov(r)  
        chol = cholesky(cov_r) 
        log_det = jnp.sum(jnp.log(jnp.diag(chol)))
        return log_det

    
    def _pad_along_last_axis(self, X):
        first_value = X[0]
        last_value = X[-1]
        # Use `lax.full_like` to create padded arrays
        Xl = lax.full_like(x=jnp.empty((self.vasicek_window,)), fill_value=first_value)
        Xr = lax.full_like(x=jnp.empty((self.vasicek_window,)), fill_value=last_value)
        return jnp.concatenate((Xl, X, Xr))

    def _vasicek_entropy(self, X):
        n = X.shape[-1]
        X = jnp.sort(X, axis=-1)
        X = self._pad_along_last_axis(X)
        start1 = 2 * self.vasicek_window
        length = self.P
        differences = lax.dynamic_slice(X, (start1,), (length, )) - lax.dynamic_slice(X, (0,), (length,))
        logs = jnp.log(n / (2 * self.vasicek_window) * differences)
        return jnp.mean(logs, axis=-1)

# Register the custom class as a PyTree with JAX
tree_util.register_pytree_node(
    OlfactorySensing,
    OlfactorySensing._tree_flatten,
    OlfactorySensing._tree_unflatten
)

def compute_rho(W, tol=None): 
    if not tol: 
        rho = 1 - jnp.sum(W == 0) / len(W.flatten())
    else: 
        rho = 1 - jnp.sum(W < tol) / len(W.flatten()) 
    return rho 


def natural_gradient_dual_space(n_steps, W_init, c_init, key, loss, lr, os, phi, psi, verbose=True):
    value_and_grad = jax.value_and_grad(loss) # loss should be a function we minimize with W coming first
    trajectory = [W_init]
    entropies = [-os.compute_entropy_of_r(W_init, c_init, key)] 
    losses = [loss(W_init, c_init, key)]
    U_init = psi(W_init) 
    U_current = U_init 
    print_interval = round(n_steps / 10) 
    for n in range(n_steps):
        key, subkey = jax.random.split(key)
        cs = os.draw_cs(subkey)
        value, grad = value_and_grad(phi(U_current), cs, subkey) 
        U_new = U_current - lr * grad # see https://vene.ro/blog/mirror-descent.html#duality-between-mirror-descent-and-natural-gradient 
        trajectory.append(phi(U_current))
        entropies.append(-os.compute_entropy_of_r(phi(U_current), cs, key))
        losses.append(loss(phi(U_current), cs, subkey)) 
        U_current = U_new
        if n % print_interval == 0 and verbose: 
            print(f"Step {n}, Loss: {value}")
    return jnp.array(trajectory), entropies, losses


def plot_W_and_activity_in_2D(W_init, r_init, W_opt, r_opt, entropy, losses, rhos, loss_label=''): 
    fig = plt.figure(figsize=(18, 5))
    # Create a GridSpec layout with 2 rows and 3 columns
    gs = GridSpec(2, 3, height_ratios=[3, 1], figure=fig)
    # Add axes
    ax1 = fig.add_subplot(gs[:, 0])  # First column spans both rows
    ax2 = fig.add_subplot(gs[:, 1])
    ax3_top = fig.add_subplot(gs[0, 2])  # Top half of the middle column
    ax3_bottom = fig.add_subplot(gs[1, 2])  # Bottom half of the middle column
    ax1.scatter(W_init[0, :], W_init[1, :], label='W_init', alpha=0.7)
    ax1.scatter(W_opt[0, :], W_opt[1, :], label='W_opt', alpha=0.7)
    ax2.scatter(r_init[0, :], r_init[1, :], label='r_init', alpha=0.7)
    ax2.scatter(r_opt[0, :], r_opt[1, :], label='r_opt', alpha=0.7)
    ax1.set_title(r'$W_{j}$')
    ax1.set_xlabel(r'$W_{1j}$')
    ax1.set_ylabel(r'$W_{2j}$')
    ax2.set_title('activity') 
    ax2.set_xlabel(r'$r_1$')
    ax2.set_ylabel(r'$r_2$')
    ax3_bottom.set_xlabel('step') 
    ax3_top.set_ylabel(r'$-H(r)$')
    ax3_top.set_title('entropy') 
    axr = ax3_top.twinx()
    ent, = ax3_top.plot(range(len(entropy)), entropy, label='entropy')
    rho, = axr.plot(rhos, color='tab:orange', label=r'$\rho$')
    lines = [ent, rho]
    labels = [line.get_label() for line in lines]
    ax3_top.legend(lines, labels)
    ax3_top.set_xticks([])
    ax3_top.set_xticklabels([])
    ax3_bottom.plot(losses, label=loss_label, color='tab:green') 
    ax3_bottom.set_title(loss_label) 
    return [ax1, ax2, ax3_top, ax3_bottom]


def plot_trajectory(Ws, ents, losses, rhos, sigma_c, gamma): 
    fig, axs = plt.subplots(2, 1, height_ratios=[3, 1])
    ent, = axs[0].plot(ents, label='entropy', color='tab:blue') 
    axs[1].plot(losses, label=r"$\log |\Sigma|$")
    axs[1].set_xlabel('step') 
    axs[0].set_ylabel('-H(r)')
    title = rf'$\sigma_c = {sigma_c:.2g}, \gamma = {gamma:.2g}$'
    axs[0].set_title(title)
    ax2 = axs[0].twinx()
    rho, = ax2.plot(rhos, color='tab:orange', linestyle='--', label=r'$\rho$')
    ax2.set_ylabel(r'$\rho$')
    lines = [ent, rho]
    labels = [line.get_label() for line in lines]
    axs[0].legend(lines, labels)
    axs[1].legend()
    return fig

def plot_W(W_init, W_opt, entropies, rhos): 
    fig = plt.figure(figsize=(18, 5))

    # Create a GridSpec layout with 2 rows and 3 columns
    gs = GridSpec(2, 3, height_ratios=[1, 1], figure=fig)

    # Add axes
    ax1 = fig.add_subplot(gs[:, 0])  # First column spans both rows
    ax2_top = fig.add_subplot(gs[0, 1])  # Top half of the middle column
    ax2_bottom = fig.add_subplot(gs[1, 1])  # Bottom half of the middle column
    ax3 = fig.add_subplot(gs[:, 2]) 

    epsilon = 1e-30 
    W_opt = jnp.where(W_opt == 0, epsilon, W_opt)
    ax1.hist(W_init.flatten(), alpha=0.7, label='W_init', density=True)
    ax1.hist(W_opt.flatten(), alpha=0.7, label='W_opt', density=True) 
    ax1.set_xlabel(r'$W_{ij}$')
    ax2_top.hist(jnp.log(W_init.flatten()), alpha=0.7, label='W_init', density=True, color='tab:blue')
    ax2_bottom.hist(jnp.log(W_opt.flatten()), alpha=0.7, label='W_opt', density=True, color='tab:orange') 
    ax2_bottom.set_xlabel(r'$\log(W_{ij})$')
    ax1.legend()
    ent, = ax3.plot(range(len(entropies)), entropies, label='entropy')
    ax3.set_title('entropy') 
    ax3.set_ylabel(r'$-H(r)$')
    ax3.set_xlabel('step') 
    axr = ax3.twinx()
    rho, = axr.plot(rhos, color='tab:orange', label=r'$\rho$')
    lines = [ent, rho]
    labels = [line.get_label() for line in lines]
    ax3.legend(lines, labels)
    return fig


# Making the class a pytree: see https://jax.readthedocs.io/en/latest/faq.html#strategy-3-making-customclass-a-pytree

