import jax
import jax.numpy as jnp
from jax.scipy.stats import norm
import matplotlib.pyplot as plt
import argparse 

def make_block_diagonal_cov(J_vals, N):
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
        block = J_vals[j] * jnp.ones(block_size) # Scalar -> K x K diagonal block
        
        # Place the block
        end_pos = pos + block_size
        mat = mat.at[pos:end_pos, pos:end_pos].set(block)
        pos = end_pos
    
    return mat

def gaussian_copula_binary(key, means, cov, P):
    n = len(means)
    cov = cov + 1e-6 * jnp.eye(cov.shape[0])
    Z = jax.random.multivariate_normal(key, mean=jnp.zeros(n), cov=cov, shape=(P, ))
    thresholds = norm.ppf(1 - means)  # Inverse CDF
    X = (Z > thresholds).astype(int)
    return X.T, Z.T

def parse_args():
    parser = argparse.ArgumentParser(description="Generate samples from Ising model with given parameters")
    
    parser.add_argument("--seed", type=int, default=1, help="Random seed (default: 1)")
    parser.add_argument("--N", type=int, default=1000, help="Number of odorants (default: 1000)")
    parser.add_argument("--block_size", type=int, default=10, help="Number of odorants (default: 10)")
    parser.add_argument("--gamma_shape", type=float, default=1.0, help="shape parameter for gamma distribution (default: 1.0; means will be distributed scale * gamma(shape) + loc)")
    parser.add_argument("--gamma_scale", type=float, default=1/5, help="gamma scaling (default = 1/5; means will be distributed scale * gamma(shape) + loc)")
    parser.add_argument("--gamma_loc", type=float, default=-0.05, help="Offset to subtract from gamma values (default: -0.05; means will be distributed scale * gamma(shape) + loc)")
    parser.add_argument("--lambda_value", type=float, default=2, help="j, where J = j * block diagonal matrix")
    parser.add_argument("--balanced_means", action="store_true", help="Set means to 0.5")
    parser.add_argument("--lag", type=int, default=200, help="lag for ising model. The spins from gibbs samplign will be downsampled at this rate and burnin = 15 * lag") 
    parser.add_argument("--beta", type=int, default=1, help="inverse temperature for sampling. default = 1") 
    parser.add_argument("--samples", type=int, default=100000, help="number of samples. default = 10^5") 
    return parser.parse_args()

if __name__ == "__main__": 
    args = parse_args()
    
    # Initialize random keys
    key = jax.random.key(args.seed)
    key, subkey_mean, subkey_samples = jax.random.split(key, 3)
    
    # Generate h vector
    if args.balanced_means:
        means = jnp.ones(args.N) * 0.5
    else:
        means = jnp.clip(jax.random.gamma(subkey_mean, a=1, shape=(args.N,)) * args.gamma_scale + args.gamma_loc, min=0, max=0.9) 

    # Generate lambdas and J matrix
    lambdas = args.lambda_value * jnp.ones((args.block_size,))
    J = make_block_diagonal_cov(lambdas, args.N)
    X, _ = gaussian_copula_binary(subkey_samples, means, J, P=args.samples)
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs[0].hist(jnp.mean(X, axis=1))
    axs[0].set_title('means')
    im = axs[1].imshow(jnp.nan_to_num(jnp.corrcoef(X), 0))
    axs[1].set_title('correlations')
    plt.colorbar(im)
    fig.savefig('summary_stats.png')

