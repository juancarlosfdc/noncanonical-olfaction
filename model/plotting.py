import matplotlib.pyplot as plt
from model import *
import jax
import jax.numpy as jnp
from jax import vmap


def plot_activity(p_init, p_final, hp, mis, key):
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    E_init = p_init.E
    E_final = p_final.E
    axs[0].scatter(E_init[:, 0], E_init[:, 1], alpha=0.7, label="E_init")
    axs[0].scatter(E_final[:, 0], E_final[:, 1], alpha=0.7, label="E_final")
    # axs[0].hist(E_init[:, 0], alpha=0.7, label=r'$(E_{init})_{i, 0}$')
    # axs[0].hist(E_final[:, 0], alpha=0.7, label=r'$(E_{final})_{i, 0}$')
    axs[0].set_xlabel(r"$E_{i, 1}$")
    axs[0].set_ylabel(r"$E_{i, 2}$")
    axs[0].legend()
    axs[0].set_title("E_i")
    axs[0].plot([0, 1], [1, 0], color="black")
    key, *subkeys = jax.random.split(key, 4)
    draw_cs = ODOR_MODEL_REGISTRY[hp.odor_model]
    activity_function = ACTIVITY_FUNCTION_REGISTRY[hp.activity_model]
    cs = draw_cs(subkeys[0], hp)
    r_init = activity_function(hp, p_init, cs, subkeys[1])
    r_final = activity_function(hp, p_final, cs, subkeys[2])
    if r_init.shape[0] == 1:
        axs[1].hist(r_init[0], alpha=0.7, label="r_init")
        axs[1].hist(r_final[0], alpha=0.7, label="r_final")
    else:
        axs[1].scatter(r_init[0], r_init[1], alpha=0.7, label="r_init")
        axs[1].scatter(r_final[0], r_final[1], alpha=0.7, label="r_final")
        axs[1].set_ylabel(r"$r_2$", labelpad=1)
    axs[1].legend()
    axs[1].set_title("neural activity")
    axs[1].set_xlabel(r"$r_1$")
    axs[2].plot(mis)
    axs[2].set_title(r"$\widehat{MI_{JSD}}(r; c)$")
    axs[2].set_xlabel("training step")
    return fig, axs, r_init, r_final


# example call:
"""
key = jax.random.key(42) 
fig, ax, *_ = plot_activity(init_state.p, state.p, hp, metrics['mi'], key) 
fig.suptitle(f'{hp.activity_model} activity, {hp.odor_model} odor model', y=1.05)
fig.savefig('tmp.png', bbox_inches='tight')
"""


def compute_expression_ratio_neuron(e):
    se = jnp.sort(e)
    return se[-1] / jnp.clip(se[-2], min=1e-3)


def compute_expression_ratios(E):
    ratios = vmap(compute_expression_ratio_neuron, in_axes=0)(E)
    return ratios


def plot_expression(E_init, E_final, mis, metric='scatter', mi_clip=-1, c_bin=True):
    mosaic = [["E_init", "E_final", "MI"], ["hist_init", "hist_final", "."]]
    fig, axs = plt.subplot_mosaic(
        mosaic, 
        figsize=(18, 10),
        gridspec_kw={"hspace": 0.5}
    )
    axs["E_final"].sharey(axs["E_init"])
    axs["E_final"].tick_params(labelleft=False)
    axs["E_final"].tick_params(labelbottom=False)
    axs["E_init"].set_ylabel("ORNs")
    axs["E_init"].set_xlabel("ORs", labelpad=5)
    vmin = min(E_init.min(), E_final.min())
    vmax = max(E_init.max(), E_final.max())
    im1 = axs["E_init"].imshow(E_init, vmin=vmin, vmax=vmax, aspect="auto")
    axs["E_init"].set_title(r"$E_{init}$")
    im2 = axs["E_final"].imshow(E_final, vmin=vmin, vmax=vmax, aspect="auto")
    axs["E_final"].set_title(r"$E_{final}$")
    fig.colorbar(im1, ax=[axs["E_init"], axs["E_final"]], location="right", pad=0.1)
    axs["MI"].plot(jnp.clip(mis, mi_clip))
    if c_bin: 
        axs["MI"].set_title(fr"$\widehat{{MI_{{\mathrm{{JSD}}}}}}(r;\ c_{{bin}})$ (clip={mi_clip})")
    else: 
        axs["MI"].set_title(fr"$\widehat{{MI_{{\mathrm{{JSD}}}}}}(r;\ c)$ (clip={mi_clip})")
    axs["MI"].set_xlabel("epochs") 
    if metric == 'ratio_max_to_second': 
        init_ratios = jnp.log(compute_expression_ratios(E_init))
        final_ratios = jnp.log(compute_expression_ratios(E_final))
        axs["hist_init"].hist(
            init_ratios,
            alpha=0.7,
            label=rf"$\langle \log (r_{{E_{{init}}}})\rangle = {jnp.mean(init_ratios):.2f}$",
        )
        axs["hist_final"].hist(
            final_ratios,
            alpha=0.7,
            label=rf"$\langle \log (r_{{E_{{final}}}})\rangle = {jnp.mean(final_ratios):.2f}$",
        )
        ax23_title = r"$\log(\text{expression ratio})$"
    elif metric == 'max_value': 
        axs['hist_init'].hist(jnp.max(E_init, axis=1))
        axs['hist_final'].hist(jnp.max(E_final, axis=1))
        ax23_title = 'row-wise maxima'
    elif metric == 'scatter': 
        axs['hist_init'].scatter(range(E_init.shape[0]), jnp.sort(jnp.max(E_init, axis=1)))
        axs['hist_final'].scatter(range(E_final.shape[0]), jnp.sort(jnp.max(E_final, axis=1)))
        ax23_title = 'row-wise maxima'
    # [ax.legend() for ax in [axs["hist_init"], axs["hist_final"]]]
    [ax.set_xlabel("ORNs") for ax in [axs["hist_init"], axs["hist_final"]]]
    ax23 = fig.add_subplot(223, frameon=False)
    ax23.set_xticks([])
    ax23.set_yticks([])
    ax23.set_title(ax23_title, loc="right")    
    return fig, axs


# example call:
"""
fig, axs = plot_expression(init_state.p.E, state.p.E,  metrics['mi'])
fig.savefig('tmp.png')
"""
