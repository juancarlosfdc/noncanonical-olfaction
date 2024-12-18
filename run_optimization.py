from canonical_model_jax import *
import argparse 

def phi(u): 
    return 1 / (1 + jnp.exp(-u))

def psi(x): 
    return jnp.log(x / (1 - x))

rho_vectorized = jax.vmap(lambda x: compute_rho(x, tol=1e-10), in_axes=0)

def run_optimization(N, n, M, sigma_c, P, output_dir): 
    key = jax.random.PRNGKey(0)
    W_init = jnp.clip(1 / jnp.sqrt(N) * jax.random.gamma(key, a=1, shape=(M, N)), a_min=0, a_max=(1 - 1e-10)) 
    gamma = 1
    os = OlfactorySensing(N=N, n=n, M=M, P=P, sigma_c=sigma_c)
    os.W = W_init
    os.cs = os.draw_cs(key=key)     
    Ws, ents, losses = natural_gradient_dual_space(10, W_init, os.cs, key, lambda * args: - os.log_det_sigma(*args), 1, os, phi, psi)
    rhos = rho_vectorized(Ws)
    fig = plot_trajectory(Ws, ents, losses, rhos, sigma_c=sigma_c, gamma=gamma)
    filename = f"{output_dir}/sigma_{sigma_c:.2g}_N_{N}_n_{n}_M_{M}"
    fig.savefig(filename + '.png')
    jnp.save(filename + '.npy', Ws) 
    return rhos[-1]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--sigma", type=float, required=False, help="Input sigma value", default=2.0)
    parser.add_argument("--n", type=int, required=False, help="Input n", default=2) 
    parser.add_argument("--N", type=int, required=False, help="Input N", default=100) 
    parser.add_argument("--M", type=int, required=False, help="Input M", default=30) 
    parser.add_argument("--P", type=int, required=False, help="Input P", default=50000) 
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--results_file", type=str, required=True, help="Path to results file")
    args = parser.parse_args()

    rho = run_optimization(args.N, args.n, args.M, args.sigma, args.P, args.output_dir) 

    with open(args.results_file, "a") as f:
        f.write(f"{args.N}\t{args.n}\t{args.M}\t{args.sigma}\t{args.P}\t{rho}\n")