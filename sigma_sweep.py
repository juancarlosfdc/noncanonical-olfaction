from canonical_model_jax import *

N, n, M, P = 100, 2, 30, 1000
os = OlfactorySensing(N=N, n=n, M=M, P=P, sigma_c=2.44)
key = jax.random.PRNGKey(0)
os.cs = os.draw_cs(key=key) 
W_init = 1 / jnp.sqrt(os.N) * jax.random.gamma(key, a=1, shape=(M, N))
os.W = W_init 

def phi(u): 
    return 1 / (1 + jnp.exp(-u))

def psi(x): 
    return jnp.log(x / (1 - x))

rho_vectorized = jax.vmap(lambda x: compute_rho(x, tol=1e-10), in_axes=0)

def run_sigma_sweep(sigmas): 
    N, n, M, P = 100, 2, 30, 1000
    key = jax.random.PRNGKey(0)
    W_init = 1 / jnp.sqrt(N) * jax.random.gamma(key, a=1, shape=(M, N))
    gamma = 1
    final_rhos = []
    for sigma_c in sigmas: 
        os = OlfactorySensing(N=N, n=n, M=M, P=P, sigma_c=sigma_c)
        os.W = W_init
        os.cs = os.draw_cs(key=key)     
        Ws, ents, losses = natural_gradient_dual_space(400, W_init, os.cs, key, lambda * args: - os.log_det_sigma(*args), 1, os, phi, psi)
        rhos = rho_vectorized(Ws)
        final_rhos.append(rhos[-1]) 
        fig = plot_trajectory(Ws, ents, losses, rhos, sigma_c=sigma_c, gamma=gamma)
        filename = f"sigma_sweep/sigma_{sigma_c:.2g}_gamma_{gamma:.2g}" 
        fig.savefig(filename + '.png')
        jnp.save(filename + '.npy', Ws) 
    return sigmas, final_rhos 

sigmas = jnp.linspace(2, 6, 2) 

sigmas, final_rhos = run_sigma_sweep(sigmas)

fig, ax = plt.subplots()
ax.scatter(sigmas, final_rhos) 
ax.set_xlabel(r'$\sigma_c$')
ax.set_ylabel(r'$\rho$')
fig.savefig('rho_vs_sigma.png') 
