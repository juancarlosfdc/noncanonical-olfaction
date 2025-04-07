import matplotlib.pyplot as plt

def plot_expression(E_init, E_final, mis): 
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs[0].scatter(E_init[:, 0], E_init[:, 1], alpha=0.7, label='E_init')
    axs[0].scatter(E_final[:, 0], E_final[:, 1], alpha=0.7, label='E_final') 
    # axs[0].hist(E_init[:, 0], alpha=0.7, label=r'$(E_{init})_{i, 0}$')
    # axs[0].hist(E_final[:, 0], alpha=0.7, label=r'$(E_{final})_{i, 0}$')
    axs[0].set_xlabel(r'$E_{i, 1}$')
    axs[0].set_ylabel(r'$E_{i, 2}$')
    axs[0].legend()
    axs[0].set_title('E_i')
    axs[1].plot(mis) 
    return fig, axs

def plot_activity(p_init, p_final, hp, mis, key): 
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    E_init = p_init.E 
    E_final = p_final.E 
    axs[0].scatter(E_init[:, 0], E_init[:, 1], alpha=0.7, label='E_init')
    axs[0].scatter(E_final[:, 0], E_final[:, 1], alpha=0.7, label='E_final') 
    # axs[0].hist(E_init[:, 0], alpha=0.7, label=r'$(E_{init})_{i, 0}$')
    # axs[0].hist(E_final[:, 0], alpha=0.7, label=r'$(E_{final})_{i, 0}$')
    axs[0].set_xlabel(r'$E_{i, 1}$')
    axs[0].set_ylabel(r'$E_{i, 2}$')
    axs[0].legend()
    axs[0].set_title('E_i')
    key, *subkeys = jax.random.split(key, 4) 
    cs = draw_cs_sparse_log_normal(subkeys[0], hp)
    r_init = compute_linear_filter_activity(subkeys[1], hp, p_init, cs)
    r_final = compute_linear_filter_activity(subkeys[2], hp, p_final, cs)
    axs[1].scatter(r_init[0], r_init[1], alpha=0.7, label='r_init')
    axs[1].scatter(r_final[0], r_final[1], alpha=0.7, label='r_final')
    axs[1].legend()
    axs[1].set_title('neural activity') 
    axs[2].plot(mis) 
    axs[2].set_title(r'$\widehat{MI_{JSD}}(r; c)$')
    return fig, axs, r_init, r_final 

def compute_expression_ratio_neuron(e): 
    se = jnp.sort(e) 
    return se[-1] / jnp.clip(se[-2], min=1e-3) 

def compute_expression_ratios(E): 
    ratios = vmap(compute_expression_ratio_neuron, in_axes=0)(E) 
    return ratios 

def plot_expression(E_init, E_final, mis):
    mosaic = [['E_init', 'E_final', 'MI'],
     ['hist_init', 'hist_final',       '.']]
    fig, axs = plt.subplot_mosaic(mosaic, figsize=(18, 10))
    vmin = min(E_init.min(), E_final.min())
    vmax = max(E_init.max(), E_final.max())
    im1 = axs['E_init'].imshow(E_init, vmin=vmin, vmax=vmax, aspect='auto')
    axs['E_init'].set_title(r'$E_{init}$')
    im2 = axs['E_final'].imshow(E_final, vmin=vmin, vmax=vmax, aspect='auto')
    axs['E_final'].set_title(r'$E_{final}$')
    fig.colorbar(im1, ax=[axs['E_init'], axs['E_final']], location='right', pad=0.1)
    axs['MI'].plot(mis)
    axs['MI'].set_title(r'$\widehat{MI_{JSD}}(r; c)$')
    init_ratios = jnp.log(compute_expression_ratios(E_init))
    final_ratios = jnp.log(compute_expression_ratios(E_final)) 
    axs['hist_init'].hist(init_ratios, alpha=0.7, label=rf'$\langle \log (r_{{E_{{init}}}})\rangle = {jnp.mean(init_ratios):.2f}$')
    axs['hist_final'].hist(final_ratios, alpha=0.7, label=rf'$\langle \log (r_{{E_{{final}}}})\rangle = {jnp.mean(final_ratios):.2f}$')
    [ax.legend() for ax in [axs['hist_init'], axs['hist_final']]]
    ax23 = fig.add_subplot(223, frameon = False)
    ax23.set_xticks([])
    ax23.set_yticks([])
    ax23.set_title(r"$\log(\text{expression ratio})$", loc='right')
    return fig, axs 