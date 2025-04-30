import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import flax.linen as nn
import jax.tree_util as jtu
from jax import vmap, lax, jit
from jax.scipy.stats import norm, rankdata
from jax.scipy.special import digamma
from jax.scipy.special import gamma
from jax.scipy.linalg import cholesky
from functools import partial
from typing import NamedTuple
from flax import struct
from functools import partial 

plt.rcParams["font.size"] = 20
DATA_PATH = "../rbm/mm_synthetic_data_non_zero_samples_9_April_2025.npy"


class HyperParams(NamedTuple):
    N: int = 100  # number of odorants. this needs to match the number of variables (rows) in the data when using the data driven model.
    n: int = 2  # sparsity of odor vectors--only applies for the Qin et al 2019 odor model
    M: int = 30  # number of odorant receptors
    L: int = 300  # number of olfactory receptor neurons
    P: int = 1000  # number of samples when we compute MI
    K: int = 50 # number of glomeruli (only relevant if computing with glomerular convergence)
    window: int = 32  # this should be set to jnp.round(jnp.sqrt(P) + 0.5).astype(int) but doing this dynamically makes hp not hashable. So change it when you change P!
    sigma_0: float = 0.01  # neural noise in Gaussian linear filter model
    sigma_c: float = 2.0  # std_dev of log normal Qin et al 2019 odorant model
    W_shape: float = 1.0 
    W_scale: str = "1 / ||c||" 
    W_init: str = "normal"
    nonlinearity: str = "sigmoid log"
    F_max: float = 25.0
    hill_exponent: int = 4
    odor_model: str = "log normal"
    activity_model: str = "glomerular convergence"
    loss: str = "jensen_shannon_loss"
    phi_w: str = "identity"
    psi_w: str = "identity"
    phi_e: str = "softplus_phi"
    psi_e: str = "softplus_psi"
    phi_g: str = "identity"
    psi_g: str = "identity"
    sigma_kappa_inv: float = 4.0 # this is for Reddy et al. 2018 antagonism model 
    rho: float = 0.0 # ditto
    canonical_E_init: bool = False 
    balanced_E_init: bool = False 
    canonical_G_init: bool = False
    binarize_c_for_MI_computation: bool = False 


class TrainingConfig(NamedTuple):
    scans: int = 10
    epochs_per_scan: int = 100
    gamma_W: float = 0.1
    gamma_E: float = 0.1
    gamma_G: float = 0.1
    gamma_T: float = 0.1
    T_hidden_dim: int = 128
    T_mode: str = "inner_product"


class LoggingConfig(NamedTuple):
    log_interval: int = None
    save_model: bool = False
    output_dir: str = "."
    config_id: str = "0"


class FullConfig(NamedTuple):
    hyperparams: HyperParams
    training: TrainingConfig
    logging: LoggingConfig
    seed: int = 0
    data_path: str = '' 


@struct.dataclass
class Params:
    W: jnp.ndarray
    E: jnp.ndarray
    G: jnp.ndarray 
    kappa_inv: jnp.ndarray
    eta: (
        jnp.ndarray
    )  # these are not going to be changed during training, but you can't hash a tuple with jax arrays, so here they shall live.


class T_estimator(nn.Module):
    hidden_dim: int = 128
    mode: str = "inner_product" 

    def g_f(self, v):
        return jnp.clip(
            jnp.log(2) - jnp.log(1 + jnp.exp(-v)), max=jnp.log(2) - 1e-6
        )  # see Table 2 of https://arxiv.org/pdf/1606.00709


    @nn.compact
    def __call__(self, c, r):
         # batch should be the first dimension here. this is contra everything else in our pipeline, so be careful! transpose just before passing in c and r
        if self.mode == "inner_product": 
            r = nn.Dense(self.hidden_dim)(r)
            r = nn.tanh(r)
            r = nn.Dense(self.hidden_dim)(r)
            r = nn.tanh(r)
            c = nn.Dense(self.hidden_dim)(c)
            c = nn.tanh(c)
            c = nn.Dense(self.hidden_dim)(c)
            c = nn.tanh(c)
            v = jnp.einsum("bi,bi->b", c, r)  # or use vmap(jnp.dot)(c, r)
            return self.g_f(v)

        elif self.mode == "concatenate": 
            # Concatenate c (P, N) and r (P, L) -> (P, N + L)
            x = jnp.concatenate([c, r], axis=1)
            x = nn.Dense(self.hidden_dim)(x)
            x = nn.tanh(x)
            x = nn.Dense(self.hidden_dim)(x)
            x = nn.tanh(x)
            v = nn.Dense(1)(x)
        return self.g_f(v)


@struct.dataclass
class TrainingState:
    p: any  # Your primal parameters
    Tp: any  # MINE network parameters
    key: jax.Array  # PRNG key
    metrics: dict


def create_one_hot_array(key, L, M):
    indices = jax.random.randint(key, (L,), 0, M)
    return jax.nn.one_hot(indices, M, dtype=float)


def sigmoid_log(x):
    return x / (jnp.abs(x) + 1) # for the cases where x can be negative


def initialize_p(rng, mean_norm_c=None, hp=None) -> Params:
    if hp is None:
        hp = HyperParams()
    W_key, E_key, G_key, z_key, eta_key = jax.random.split(rng, 5) # I would like to automatically set the scale based on odorant model... 

    phi_E = PHI_PSI_REGISTRY[hp.phi_e]
    phi_G = PHI_PSI_REGISTRY[hp.phi_g]
    phi_W = PHI_PSI_REGISTRY[hp.phi_w]

    if hp.W_init == "gamma": 
        W = float(hp.W_scale) * phi_W(jnp.clip(
            jax.random.gamma(W_key, a=hp.W_shape, shape=(hp.M, hp.N)),
            min=1e-9,
            max=1 - 1e-9,
        ))
    
    else:
        if hp.W_scale == "1 / ||c||": 
            W = (1 / mean_norm_c) * phi_W(jax.random.normal(W_key, shape=(hp.M, hp.N)))
        
        elif hp.W_scale == "1 / root(N)": 
            W = 1 / jnp.sqrt(hp.N) * phi_W(jax.random.normal(W_key, shape=(hp.M, hp.N))) # this is a bad scaling because c is log-normal distributed and sparse 

    if hp.canonical_E_init:
        if hp.balanced_E_init:
            try:
                E = jnp.repeat(
                    jnp.eye(hp.M), hp.L // hp.M, axis=0
                ).astype(float)  # this is a canonical initialization with exactly equal numbers representing each receptor
            except:
                raise ValueError(
                    "For balanced canonical initialization, set L to a postive multiple of M"
                )
        else:
            E = create_one_hot_array(E_key, hp.L, hp.M)
    else:
        E = phi_E(
            0.5 + 0.1 * jax.random.normal(E_key, shape=(hp.L, hp.M))
        )  # this is a noncanonical initialization where every neuron expresses roughly the same amount of every receptor.
    if hp.canonical_G_init: 
        G = E.T # this only works if number of receptors (hp.M) == number of glomeruli (hp.K) 
    else: 
        G = phi_G(1/hp.L * jax.random.normal(G_key, shape=(hp.K, hp.L))) # this scaling is key. Otherwise the scale of g is enormous compared to r, and loss = 0 no matter what. the signal is swamped. 
    # kappa_inv = jax.random.lognormal(kappa_inv_key, sigma=sigma_kappa_inv, shape=(hp.M, hp.N)) # see "Olfactory Encoding Model" in Reddy and Zak 2018
    eta = jax.random.lognormal(eta_key, shape=(hp.M, hp.N))
    z = jax.random.normal(z_key, shape=(hp.M, hp.N))
    kappa_inv = jnp.exp(
        hp.sigma_kappa_inv * (hp.rho * jnp.log(eta) + jnp.sqrt(1 - hp.rho**2) * z)
    )
    return hp, Params(W, E, G, kappa_inv, eta)


def initialize_training_state(subkey, hp, p_init, training_config):
    subkey_cs, subkey_activity, subkey_T, subkey_state = jax.random.split(subkey, 4)
    T = T_estimator(
        hidden_dim = training_config.T_hidden_dim,
        mode = training_config.T_mode
        )
    draw_cs = ODOR_MODEL_REGISTRY[hp.odor_model]
    activity_function = ACTIVITY_FUNCTION_REGISTRY[hp.activity_model]
    cs = draw_cs(subkey_cs, hp)
    r = activity_function(hp, p_init, cs, subkey_activity)
    if hp.loss == "jensen_shannon_loss":
        Tp_init = T.init(subkey_T, cs.T, r.T)  # this is for variational MI estimation
    else:
        Tp_init = 0  # this is a dummy for Gaussian MI estimation (not variational, so no T network needed)
    init_state = TrainingState(p_init, Tp_init, subkey_state, metrics={"mi": 0.0})
    return init_state


def phi(u):
    return 1 / (1 + jnp.exp(-u))


def psi(x):
    x = jnp.clip(x, min=1e-6, max=1 - 1e-6)
    return jnp.log(x / (1 - x))


def phi(u):
    return jax.nn.softmax(u, axis=1)


def psi(x):
    return jnp.log(jnp.clip(x, min=1e-6, max=1 - 1e-6))


def draw_c_sparse_log_normal(subkey, hp):
    c = jnp.zeros(hp.N)
    sub1, sub2 = jax.random.split(subkey)
    non_zero_indices = jax.random.choice(sub1, hp.N, shape=(hp.n,), replace=False)
    concentrations = jax.random.lognormal(sub2, sigma=hp.sigma_c, shape=(hp.n,))
    c = c.at[non_zero_indices].set(concentrations)
    return c

@partial(jax.jit, static_argnames=['hp'])
def draw_cs_sparse_log_normal(subkey, hp):
    keybatch = jax.random.split(subkey, hp.P)
    cs = vmap(draw_c_sparse_log_normal, in_axes=(0, None))(keybatch, hp).T
    return cs


def closure_draw_cs_data_driven(
    path,
):  # closures are nice https://apxml.com/courses/advanced-jax/chapter-1-advanced-jax-transformations-control-flow/closures-jax-staging
    loaded_data = jnp.load(path)
    P_total = loaded_data.shape[1]

    def draw_cs_data_driven_binary(subkey, hp):
        idx = jax.random.choice(subkey, P_total, shape=(hp.P,), replace=True)
        batch = jax.device_put(loaded_data[:, idx])  # move to device only once per call
        return batch  # shape (N, P), like the log_normal one

    def draw_cs_data_driven_log_normal(subkey, hp):
        subkey_index, subkey_concentration = jax.random.split(subkey)
        idx = jax.random.choice(subkey_index, P_total, shape=(hp.P,), replace=True)
        batch = jax.device_put(loaded_data[:, idx])  # move to device only once per call
        batch = jnp.where(
            batch,
            jax.random.lognormal(
                subkey_concentration, sigma=hp.sigma_c, shape=batch.shape
            ),
            jnp.zeros(batch.shape),
        )
        return batch

    return draw_cs_data_driven_binary, draw_cs_data_driven_log_normal


draw_cs_data_driven_binary, draw_cs_data_driven_log_normal = (
    closure_draw_cs_data_driven(DATA_PATH)
)


def compute_receptor_activity(subkey, hp, p, c):
    pre_activations = p.W @ c
    nl = NL_REGISTRY[hp.nonlinearity]
    r = nl(pre_activations) + hp.sigma_0 * jax.random.normal(
        subkey, shape=pre_activations.shape
    )
    return r


def compute_linear_filter_activity(hp, p, c, subkey):
    receptor_activations = p.W @ c  # p.W is receptor x odorant
    c_AMP = p.E @ receptor_activations  # p.E is neuron x receptor
    nl = NL_REGISTRY[hp.nonlinearity]
    r = nl(c_AMP) + hp.sigma_0 * jax.random.normal(subkey, shape=c_AMP.shape)
    return r


def linear_filter_plus_glomerular_layer(hp, p, c, subkey): 
    r = compute_linear_filter_activity(hp, p, c, subkey)
    g = p.G @ r 
    return g 


def compute_osn_firing_rate_with_antagonism(hp, p, c):
    """generalized version of equation 14 in Reddy, Zak et al. 2018. eta and kappa_inv are receptor x odorant, and the gene expression matrix p.E is neuron x receptor"""
    receptor_induced_activities = p.E @ (
        jnp.matmul((p.eta * p.kappa_inv), c) / (1 + jnp.matmul(p.kappa_inv, c))
    )
    denominator = 1 + receptor_induced_activities**-hp.hill_exponent
    F = hp.F_max / denominator
    return F


def compute_osn_firing_with_antagonism(hp, p, c, subkey):
    firing_rate = compute_osn_firing_rate_with_antagonism(hp, p, c)
    # firing = jax.random.poisson(subkey, lam=firing_rate)
    firing = firing_rate + hp.sigma_0 * jax.random.normal(
        subkey, shape=firing_rate.shape
    )
    return firing  # this should be neuron x samples, where c is odorants x samples


def compute_entropy(r, hp):
    entropy = compute_sum_of_marginal_entropies(r, hp) - compute_information(r)
    return entropy


def compute_sum_of_marginal_entropies(r, hp):
    compute_entropy_vmap = vmap(vasicek_entropy, in_axes=(0, None))
    marginal_entropies = compute_entropy_vmap(r, hp)
    return jnp.sum(marginal_entropies)


def vasicek_entropy(
    X, hp
):  # see https://github.com/scipy/scipy/blob/main/scipy/stats/_entropy.py#L378
    n = X.shape[-1]
    X = jnp.sort(X, axis=-1)
    X = pad_along_last_axis(X, hp.window)
    start1 = 2 * hp.window
    length = hp.P
    differences = lax.dynamic_slice(X, (start1,), (length,)) - lax.dynamic_slice(
        X, (0,), (length,)
    )
    logs = jnp.log(n / (2 * hp.window) * differences)
    return jnp.mean(logs, axis=-1)


def pad_along_last_axis(X, window):
    first_value = X[0]
    last_value = X[-1]
    Xl = lax.full_like(x=jnp.empty((window,)), fill_value=first_value)
    Xr = lax.full_like(x=jnp.empty((window,)), fill_value=last_value)
    return jnp.concatenate((Xl, X, Xr))


def compute_information(r):
    M, P = r.shape
    G = norm.ppf(
        (rankdata(r.T, axis=0) / (P + 1)), loc=0, scale=1
    )  # this is just ranking the data and making it normally distributed.
    bias_correction = 0.5 * jnp.sum(
        digamma((P - jnp.arange(1, M + 1) + 1) / 2) - jnp.log(P / 2)
    )
    cov_matrix = jnp.cov(G, rowvar=False)
    chol_decomp = cholesky(cov_matrix)
    log_det = jnp.sum(jnp.log(jnp.diag(chol_decomp)))
    I = -(
        log_det - bias_correction
    )  # remember: entropy overall is sum of marginals minus information.
    return I


@partial(jax.jit, static_argnames=["hp"])
def entropy_loss(p, hp, cs, subkey):
    r = compute_linear_filter_activity(subkey, hp, p, cs)
    e = compute_entropy(r, hp)
    return -e


@partial(jax.jit, static_argnames=["hp"])
def log_det_loss(p, hp, cs, subkey):
    r = compute_linear_filter_activity(subkey, hp, p, cs)
    cov = jnp.cov(r, rowvar=True)
    chol_decomp = cholesky(cov)
    log_det = jnp.sum(jnp.log(jnp.diag(chol_decomp)))
    return -log_det


def f_star_JSD(t):  # see Table 2 of https://arxiv.org/pdf/1606.00709
    return -jnp.log(2 - jnp.exp(t))


def jensen_shannon_loss(p, Tp, hp, cs, key):
    key, subkey_firing, subkey_shuffle = jax.random.split(key, 3)
    activity_function = ACTIVITY_FUNCTION_REGISTRY[hp.activity_model]
    r = activity_function(hp, p, cs, subkey_firing)
    r_shuffled = jax.random.permutation(subkey_shuffle, r, axis=1)
    if hp.binarize_c_for_MI_computation: 
        cs = (cs != 0).astype(float) 
    T_joint = T_estimator().apply(Tp, cs.T, r.T)
    T_marginal = T_estimator().apply(Tp, cs.T, r_shuffled.T)
    mi_estimate = jnp.mean(T_joint) - jnp.mean(f_star_JSD(T_marginal))
    return -mi_estimate


def flatten_metrics(metrics_list):
    """Convert list of metric dicts to dict of arrays."""
    flattened = {k: [] for k in metrics_list[0].keys()}  # Initialize with keys
    for metrics in metrics_list:
        for k, v in metrics.items():
            flattened[k].extend(v)
    return {k: jnp.array(v) for k, v in flattened.items()}


@partial(jax.jit, static_argnames=("hp"))
def update_params_T(p, Tp, hp, cs, key, gammas):
    # Split keys
    key, loss_key = jax.random.split(key)
    # parse gammas
    gamma_W, gamma_E, gamma_G, gamma_T = gammas 
    # Compute gradients for MINE network (T) and model params (E)
    if hp.loss != "jensen_shannon_loss":
        raise ValueError(
            "only the jensen shannon loss requires a T network. For other losses, use update_params"
        )

    loss = LOSS_REGISTRY[hp.loss]

    phi_W = PHI_PSI_REGISTRY[hp.phi_w]
    psi_W = PHI_PSI_REGISTRY[hp.psi_w]

    phi_E = PHI_PSI_REGISTRY[hp.phi_e]
    psi_E = PHI_PSI_REGISTRY[hp.psi_e]

    phi_G = PHI_PSI_REGISTRY[hp.phi_g]
    psi_G = PHI_PSI_REGISTRY[hp.psi_g]

    mi_estimate, grads = jax.value_and_grad(loss, argnums=(0, 1))(
        p, Tp, hp, cs, loss_key
    )
    grads_p, grads_T = grads

    # Update MINE network (T)

    Tp = jax.tree.map(lambda param, grad: param - gamma_T * grad, Tp, grads_T)
    # natural gradient in dual space is equivalent to mirror descent in primal space. A "straight through" trick.
    #  phi:unconstrained --> constrained, psi is inverse of phi.
    # see https://vene.ro/blog/mirror-descent.html#generalizing-the-projected-gradient-method-with-divergences for a very clear exposition

    U_current = psi_W(p.W)  # map to dual space
    epsilon = 1e-8  # small value to avoid division by zero
    grad_norm = jnp.linalg.norm(grads_p.W)
    normalized_grad = grads_p.W / (grad_norm + epsilon)
    U_new = U_current - gamma_W * normalized_grad  
    W_new = phi_W(U_new)  # map back to primal space
    p = p.replace(W=W_new)

    # now for expression 
    U_current = psi_E(p.E)  # map to dual space
    epsilon = 1e-8  # small value to avoid division by zeroE
    grad_norm = jnp.linalg.norm(grads_p.E)
    normalized_grad = grads_p.E / (grad_norm + epsilon)
    U_new = U_current - gamma_E * normalized_grad    # grads_p.E is correctly with respect to the primal parameters and evaluated on the primal parameters
    E_new = phi_E(U_new)  # map back to primal space
    p = p.replace(E=E_new)

    # now do this again for glomeruli... 
    U_current = psi_G(p.G)  # map to dual space
    grad_norm = jnp.linalg.norm(grads_p.G)
    normalized_grad = grads_p.G / (grad_norm + epsilon)
    U_new = U_current - gamma_G * normalized_grad # grads_p.G is correctly with respect to the primal parameters and evaluated on the primal parameters
    G_new = phi_G(U_new)  # map back to primal space
    p = p.replace(G=G_new)

    return p, Tp, -mi_estimate


def update_params(p, hp, cs, key, gamma_E):
    ''' this is old--comes from estimating MI directly a la Qin et al. 2019, not using the variational formulation '''
    # Split keys
    key, loss_key = jax.random.split(key)

    loss = LOSS_REGISTRY[hp.loss]
    phi = PHI_PSI_REGISTRY[hp.phi]
    psi = PHI_PSI_REGISTRY[hp.psi]

    # Compute gradients for MINE network (T) and model params (E)
    mi_estimate, grads_p = jax.value_and_grad(loss, argnums=(0))(
        p, hp, cs, loss_key
    )  # loss needs to have this signature

    # natural gradient in dual space is equivalent to mirror descent in primal space. A "straight through" trick.
    #  phi:unconstrained --> constrained, psi is inverse of phi.
    # see https://vene.ro/blog/mirror-descent.html#generalizing-the-projected-gradient-method-with-divergences for a very clear exposition

    U_current = psi(p.E)  # map to dual space
    U_new = (
        U_current - gamma_E * grads_p.E
    )  # grads_p.E is correctly with respect to the primal parameters and evaluated on the primal parameters
    E_new = phi(U_new)  # map back to primal space
    p = p.replace(E=E_new)

    return p, mi_estimate


def scan_step(state, hp, gammas):
    key, subkey_odors, subkey_loss = jax.random.split(state.key, 3)
    draw_cs = ODOR_MODEL_REGISTRY[hp.odor_model]
    cs = draw_cs(subkey_odors, hp)
    if hp.loss == "jensen_shannon_loss":
        p_new, Tp_new, mi_estimate = update_params_T(
            state.p, state.Tp, hp, cs, subkey_loss, gammas
        )
    else:
        p_new, mi_estimate = update_params(state.p, hp, cs, subkey_loss, gammas[0])
        Tp_new = 0
    new_metrics = {"mi": mi_estimate}
    return TrainingState(p_new, Tp_new, key, new_metrics), new_metrics


# @partial(jax.jit, static_argnames=('hp', 'epochs_per_scan'))
def scan_phase(state, hp, gammas, epochs_per_scan):
    """Process multiple epochs in a single scanned phase"""

    # Scan over epochs_per_scan steps
    def scan_closure(
        state, gammas
    ):  # this is just a trick so you can pass hp into your scan_step.
        state, new_metrics = scan_step(state, hp, gammas)
        return state, new_metrics

    final_state, metrics = jax.lax.scan(
        scan_closure, state, xs=gammas, length=epochs_per_scan
    )
    return final_state, metrics


def train_natural_gradient_scan_over_epochs(
    initial_state, hp, gammas, scans=2, epochs_per_scan=2
):
    all_metrics = []
    for scan_idx in range(scans):
        # Run scanned phase
        scan_gammas = gammas[
            scan_idx * epochs_per_scan : (scan_idx + 1) * epochs_per_scan
        ]
        state, scan_metrics = scan_phase(
            initial_state, hp, scan_gammas, epochs_per_scan
        )
        scan_metrics = jax.device_get(scan_metrics)  # Move from device to host

        all_metrics.append(scan_metrics)
        if scan_idx % 1 == 0:
            avg_mi = jnp.mean(jnp.array(scan_metrics["mi"]))
            print(f"Scan {scan_idx}: Avg MI {avg_mi:.3f}")
        initial_state = state

    return initial_state, flatten_metrics(all_metrics)


def make_alternating_gammas(epochs_per_scan, scans, gamma_T, gamma_E):
    gamma_Ts = []
    gamma_Es = []
    for session in range(scans):
        if session % 2 == 0:
            gamma_Ts.extend([gamma_T] * epochs_per_scan)
            gamma_Es.extend([0.0] * epochs_per_scan)
        else:
            gamma_Ts.extend([0.0] * epochs_per_scan)
            gamma_Es.extend([gamma_E] * epochs_per_scan)
    return jnp.array([gamma_Es, gamma_Ts]).T


def make_constant_gammas(epochs_per_scan, scans, gamma_W, gamma_E, gamma_G, gamma_T):
    gamma_Ws = [gamma_W] * (epochs_per_scan * scans) 
    gamma_Es = [gamma_E] * (epochs_per_scan * scans)
    gamma_Gs = [gamma_G] * (epochs_per_scan * scans)
    gamma_Ts = [gamma_T] * (epochs_per_scan * scans)
    return jnp.array([gamma_Ws, gamma_Es, gamma_Gs, gamma_Ts]).T


NL_REGISTRY = {
    "sigmoid log": sigmoid_log,
    "relu": jax.nn.relu,
    "tanh": jax.nn.tanh,
    "id": lambda x: x,
}

ACTIVITY_FUNCTION_REGISTRY = {
    "linear filter": compute_linear_filter_activity,
    "antagonism": compute_osn_firing_with_antagonism,
    "glomerular convergence": linear_filter_plus_glomerular_layer
}

PHI_PSI_REGISTRY = {
    "softplus_phi": lambda u: jax.nn.softmax(
        u, axis=1
    ),  # rows of expression matrix should sum to 1.
    "softplus_psi": lambda x: jnp.log(jnp.clip(x, min=1e-6, max=1 - 1e-6)),
    "bounded_phi": lambda u: 1 / (1 + jnp.exp(-u)),
    "bounded_psi": lambda x: jnp.log(
        jnp.clip(x, min=1e-6, max=1 - 1e-6) / (1 - jnp.clip(x, 1e-6, 1 - 1e-6))
    ),
    "positive_phi": jnp.exp,
    "positive_psi": jnp.log,
    "identity": lambda x: x
}

LOSS_REGISTRY = {
    "jensen_shannon_loss": jensen_shannon_loss,
    "entropy_loss": entropy_loss,
    "log_det_loss": log_det_loss,
}

ODOR_MODEL_REGISTRY = {
    "data-driven binary": draw_cs_data_driven_binary,
    "data-driven log normal": draw_cs_data_driven_log_normal,
    "log normal": draw_cs_sparse_log_normal,
}
