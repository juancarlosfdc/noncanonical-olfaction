import jax 
import jax.numpy as jnp
from jax import lax
from jax import nn
import matplotlib.pyplot as plt
from functools import partial
import argparse 


def make_block_diagonal(J_vals, N):
    J = len(J_vals)
    if J == 0:
        return jnp.zeros((N, N))
    
    K = N // J  # Base block size
    remainder = N % J
    block_sizes = [K + 1 if i < remainder else K for i in range(J)]
    
    mat = jnp.zeros((N, N))
    pos = 0 
    
    for j in range(J):
        block_size = block_sizes[j]
        block = J_vals[j] * jnp.ones(block_size) / block_size  # Scalar -> K x K diagonal block
        
        # Place the block
        end_pos = pos + block_size
        mat = mat.at[pos:end_pos, pos:end_pos].set(block)
        pos = end_pos
    
    return mat


def update(beta, h, J, state, _):
    """Gibbs sampling update step for Ising model"""
    key, s = state
    key, subkey = jax.random.split(key)
    i = jax.random.choice(subkey, jnp.arange(len(s)))  # Randomly choose a spin
    
    # Calculate probability of spin being +1
    contribution = beta * (h[i] + J[i, :] @ s)
    p_plus = nn.sigmoid(2 * contribution)
    
    # Update spin
    key, subkey = jax.random.split(key)
    s = s.at[i].set(jnp.where(jax.random.uniform(subkey) < p_plus, 1, -1))
    
    return (key, s), s  # Return new state and spins

def sample_from_ising_model(key, h, J, s_init=None, iters=1, beta=1):
    """Gibbs sampling for Ising model using JAX"""
    if s_init is None:
        # Initialize spins randomly (-1, 1)
        s_init = jax.random.bernoulli(key, p=0.5, shape=(len(h),)) * 2 - 1
    
    # Create partial function with fixed parameters
    update_fn = partial(update, beta, h, J)
    
    # Run Gibbs sampling with lax.scan
    (final_state, spins), spin_trajectory = jax.lax.scan(
        update_fn, (key, s_init), None, length=iters)
    
    return spin_trajectory.T

def tau_vs_beta(key, h, J, betas, sample_fn, compute_mag_fn, estimate_tau_fn):
    """Calculate autocorrelation times vs inverse temperatures (beta)."""
    def body_fn(carry, beta):
        key, taus = carry
        key, subkey = jax.random.split(key)
        vals = sample_fn(subkey, h, J, iters=1000, beta=beta)
        m = compute_mag_fn(vals)
        tau = estimate_tau_fn(m)
        return (key, taus + [tau]), None
    
    # Use JAX scan instead of Python loop for better JIT compatibility
    (final_key, taus), _ = lax.scan(
        lambda carry, beta: body_fn(carry, beta),
        (key, []),
        betas
    )
    return betas, jnp.array(taus)

@partial(jax.jit, static_argnames=['lag'])
def autocorrelation(chain, lag):
    """Calculate autocorrelation at given lag."""
    n = chain.shape[0]
    mean = jnp.mean(chain)
    var = jnp.var(chain)
    x1 = lax.dynamic_slice(chain, (0,), (n-lag,))
    x2 = lax.dynamic_slice(chain, (lag,), (n-lag,))
    return jnp.sum((x1 - mean) * (x2 - mean)) / (var * n)

def plot_tau_vs_lag(chain, lags=range(200)):
    """Plot autocorrelation vs lag (non-JAX plotting function)."""
    autocorrs = []
    for lag in lags: 
        autocorrs.append(autocorrelation(chain, lag))
    fig, ax = plt.subplots()
    ax.plot(lags, autocorrs)
    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorrelation")
    return fig, ax

@jax.jit
def estimate_autocorrelation_time(chain, max_lag=200, threshold=0.1):
    """JIT-compatible autocorrelation time estimation."""
    lags = jnp.arange(max_lag)
    autocorrs = jax.vmap(autocorrelation, in_axes=(None, 0))(chain, lags)
    
    # Find first lag where autocorrelation < threshold
    below_threshold = autocorrs < threshold
    return jnp.where(jnp.any(below_threshold), 
                    jnp.argmax(below_threshold), 
                    max_lag)

def parse_args():
    parser = argparse.ArgumentParser(description="Generate samples from Ising model with given parameters")
    
    parser.add_argument("--seed", type=int, default=1, help="Random seed (default: 1)")
    parser.add_argument("--N", type=int, default=1000, help="Number of odorants (default: 1000)")
    parser.add_argument("--block_size", type=int, default=10, help="Number of odorants (default: 10)")
    parser.add_argument("--gamma_shape", type=float, default=1.0, help="shape parameter for gamma distribution (default: 1.0; h will be distributed scale * gamma(shape) + loc)")
    parser.add_argument("--gamma_scale", type=float, default=1/10.0, help="gamma scaling (default = 1/10; h will be distributed scale * gamma(shape) + loc)")
    parser.add_argument("--gamma_loc", type=float, default=-0.11, help="Offset to subtract from gamma values (default: -0.11; h will be distributed scale * gamma(shape) + loc)")
    parser.add_argument("--j", type=float, default=1.05, help="j, where J = j / block_size * block diagonal matrix")
    parser.add_argument("--zero_h", action="store_true", help="Set h to zeros instead of gamma distributed values")
    parser.add_argument("--lag", type=int, default=25000, help="lag for ising model. The spins from gibbs sampling will be downsampled at this rate and burnin = 15 * lag. default = 25000") 
    parser.add_argument("--beta", type=int, default=1, help="inverse temperature for sampling. default = 1") 
    parser.add_argument("--iters", type=int, default=1000000, help="number of iterations per epoch. default = 10^6") 
    parser.add_argument("--epochs", type=int, default=1, help="number of epochs (so total samples is iters * epochs). default = 1")
    return parser.parse_args()


if __name__ == "__main__": 
    jax.config.update("jax_default_matmul_precision", "high")
    print(jax.default_backend())

    args = parse_args()
    
    # Initialize random keys
    key = jax.random.key(args.seed)
    key, subkey_h, subkey_s = jax.random.split(key, 3)
    
    # Generate h vector
    if args.zero_h:
        h = jnp.zeros(args.N)
    else:
        h = jax.random.gamma(subkey_h, a=args.gamma_shape, 
                            shape=(args.N,)) * args.gamma_scale + args.gamma_loc
    
    # Generate lambdas and J matrix
    lambdas = args.j * jnp.ones((args.block_size,))
    J = make_block_diagonal(lambdas, args.N)
    
    # Sample from Ising model
    samples = [] 
    for _ in range(args.epochs): 
        s = sample_from_ising_model(subkey_s, h, J, beta=args.beta, iters=args.iters)
        samples.append(s) 
    s = jnp.concatenate(samples, axis=1)
    magnetization = jnp.mean(s, axis=0) # data is spins x samples 

    fig, ax = plot_tau_vs_lag(magnetization, lags = range(10000, 30000, 1000)) 
    ax.hlines(0.1, xmin=ax.get_xlim()[0], xmax = ax.get_xlim()[1], color='tab:red', ls='--')
    fig.savefig('autocorrelation_vs_lag.png')

    indep_s = s[:, 15 * args.lag::args.lag]

    fig, ax = plt.subplots() 
    ax.hist(jnp.mean((indep_s + 1)//2 , axis=1))
    fig.savefig('mean_histogram.png')

    fig, ax = plt.subplots() 
    ax.imshow(indep_s, aspect='auto') 
    fig.savefig('samples.png')

    sigma = jnp.cov(indep_s)
    plt.imshow(jnp.abs(sigma * (1 - jnp.eye(args.N)))) 
    plt.colorbar()
    plt.savefig('correlation.png')

