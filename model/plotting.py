import matplotlib.pyplot as plt
from model import *
import jax
import jax.numpy as jnp
from jax import vmap


def plot_2D_activity(p_init, p_final, hp, mis, key):
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


def plot_sorted_values(E, ax): 
    max_values = jnp.max(E, axis=1)  # Get max per row
    second_max_values = jnp.sort(E, axis=1)[:, -2]  # Second-highest per row
    # Sort indices based on max_values to maintain ordering
    sorted_indices = jnp.argsort(max_values)
    ax.scatter(range(len(max_values)), max_values[sorted_indices], label='max value')
    ax.scatter(range(len(second_max_values)), second_max_values[sorted_indices], label='second-to-max', alpha=0.5)
    return ax 


def render_parameters(ax_table, hp, t): 
    keys = ["L", "M", "sigma_0", "W_shape", "sigma_c", "gamma_E", "gamma_G", "gamma_T"]
    display_keys = {"L": "L", 
                    "M": "M", 
                    "sigma_0": r"$\sigma_0$", 
                    "W_shape": r"$W_{\mathrm{shape}}$", 
                    "gamma_E": r"$\gamma_E$",
                    "gamma_G": r"$\gamma_G$",
                    "gamma_T": r"$\gamma_T$",
                    "sigma_c": r"$\sigma_c$",
                    "odor_model": "odor model",
                    "activity_model": "activity model"}
    
    cell_text = [
    [
        display_keys[key],
        str(int(getattr(hp, key))) if isinstance(getattr(hp, key), int)
        else f"{getattr(hp, key):.3f}" if isinstance(getattr(hp, key), float)
        else str(getattr(hp, key))
    ] 
    for key in keys 
    if hasattr(hp, key)
    ] 
    cell_text += [[display_keys[key], f"{getattr(t, key):.3f}"] for key in keys if hasattr(t, key)]   
    table = ax_table.table(
        cellText=cell_text,
        colLabels=["Parameter", "Value"],
        loc="center",
        cellLoc="left",
        bbox=[0, 0, 1, 1]
    )
    ax_table.axis("off")  # Hide axes
    ax_table.set_title("Hyperparameters")
    table.auto_set_font_size(False)
    table.set_fontsize(20)
    return ax_table 


def plot_expression(E_init, E_final, mis, hp, t, metric='scatter', mi_clip=-1, E_clip=1e-4, log_scale=True, key=None): 
    mosaic = [["E_init", "E_final", "MI"], ["hist_init", "hist_final", "params"]]
    fig, axs = plt.subplot_mosaic(
        mosaic, 
        figsize=(18, 10),
        gridspec_kw={"hspace": 0.5}
    )
    E_init_full, E_final_full = E_init, E_final
    axs["E_final"].sharey(axs["E_init"])
    axs["E_final"].tick_params(labelleft=False)
    axs["E_final"].tick_params(labelbottom=False)
    axs["E_init"].set_xlabel("ORs", labelpad=5)
    if E_init.shape[0] > 2 * E_init.shape[1]: 
        rows = jax.random.choice(key, E_init.shape[0], shape=(E_init.shape[1] * 2,), replace=False)
        E_init = E_init[rows, :]
        E_final = E_final[rows, :]
        axs["E_init"].set_ylabel("ORNs (downsampled)")
    else: 
        axs["E_init"].set_ylabel("ORNs")
    if log_scale: 
        E_final = jnp.maximum(E_final, E_clip)
        vmin = jnp.log10(min(E_init.min(), E_final.min())) 
        vmax = jnp.log10(max(E_init.max(), E_final.max())) # good ole log is monotonic 
        im1 = axs["E_init"].imshow(jnp.log10(E_init), vmin=vmin, vmax=vmax, aspect="auto", interpolation="nearest")
        im2 = axs["E_final"].imshow(jnp.log10(E_final), vmin=vmin, vmax=vmax, aspect="auto", interpolation="nearest")
        cbar_label = r"$\log_{10}(E_{ij})$"
    else: 
        E_final = jnp.maximum(E_final, E_clip)
        vmin = min(E_init.min(), E_final.min())
        vmax = max(E_init.max(), E_final.max()) 
        im1 = axs["E_init"].imshow(E_init, vmin=vmin, vmax=vmax, aspect="auto", interpolation="nearest")
        im2 = axs["E_final"].imshow(E_final, vmin=vmin, vmax=vmax, aspect="auto", interpolation="nearest")
        cbar_label = r"$\log(E_{ij})$" 
        
    axs["E_init"].set_title(r"$E_{init}$")
    axs["E_final"].set_title(r"$E_{final}$")
    cbar = fig.colorbar(im1, ax=[axs["E_init"], axs["E_final"]], location="right", pad=0.05)
    cbar.ax.set_title(cbar_label, fontsize=16)
    axs["MI"].plot(jnp.clip(mis, mi_clip))
    if hp.binarize_c_for_MI_computation: 
        axs["MI"].set_title(fr"$\widehat{{MI_{{\mathrm{{JSD}}}}}}(r;\ c_{{bin}})$ (clip={mi_clip})")
    else: 
        axs["MI"].set_title(fr"$\widehat{{MI_{{\mathrm{{JSD}}}}}}(r;\ c)$ (clip={mi_clip})")
    axs["MI"].set_xlabel("epochs") 
    if metric == 'ratio_max_to_second': 
        init_ratios = jnp.log(compute_expression_ratios(E_init_full))
        final_ratios = jnp.log(compute_expression_ratios(E_final_full))
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
        axs['hist_init'].hist(jnp.max(E_init_full, axis=1))
        axs['hist_final'].hist(jnp.max(E_final_full, axis=1))
        ax23_title = 'row-wise maxima'
    elif metric == 'scatter': 
        axs['hist_init'] = plot_sorted_values(E_init_full, axs['hist_init'])
        axs['hist_final'] = plot_sorted_values(E_final_full, axs['hist_final'])
        ax23_title = 'row-wise maxima'
        axs['hist_init'].legend() 
    # [ax.legend() for ax in [axs["hist_init"], axs["hist_final"]]]
    axs['hist_init'].set_title(r'$E_{init}$')
    axs['hist_final'].set_title(r'$E_{final}$')
    [ax.set_xlabel("ORNs") for ax in [axs["hist_init"], axs["hist_final"]]]
    ax23 = fig.add_subplot(223, frameon=False)
    ax23.set_xticks([])
    ax23.set_yticks([])
    ax23.set_title(ax23_title, loc="right")    
    axs["params"] = render_parameters(axs["params"], hp, t) 
    return fig, axs

def sort_rows_by_first_threshold(matrix, threshold):
    first_above_thresh = jnp.argmax(matrix > threshold[:, None], axis=1)
    all_below_thresh = ~jnp.any(matrix > threshold[:, None], axis=1)
    first_above_thresh = first_above_thresh.at[all_below_thresh].set(matrix.shape[1])  # Assign a large column index
    sorted_row_indices = jnp.argsort(first_above_thresh)
    return sorted_row_indices 

def plot_G(G_init, G_final, mis, hp, t, mi_clip=-1):
    mosaic = [["G_init", "G_final", "MI"], [".", ".", "params"]]
    fig, axs = plt.subplot_mosaic(
        mosaic, 
        figsize=(18, 10),
        gridspec_kw={"hspace": 0.5}
    )
    axs["G_init"].imshow(G_init)
    vmin = min(G_init.min(), G_final.min())
    vmax = max(G_init.max(), G_final.max()) 
    threshold = 0.1 * jnp.max(G_final, axis=1)
    indices = sort_rows_by_first_threshold(G_final, threshold)
    im1 = axs["G_init"].imshow(G_init[indices, :], vmin=vmin, vmax=vmax, aspect="auto", interpolation="nearest")
    im2 = axs["G_final"].imshow(G_final[indices, :], vmin=vmin, vmax=vmax, aspect="auto", interpolation="nearest")
    cbar_label = r"$G_{ij}$"
    axs["G_init"].set_title(r"$G_{init}$")
    axs["G_final"].set_title(r"$G_{final}$")
    cbar = fig.colorbar(im1, ax=[axs["G_init"], axs["G_final"]], location="right", pad=0.05)
    cbar.ax.set_title(cbar_label, fontsize=16)
    axs["MI"].plot(jnp.clip(mis, mi_clip))
    if hp.binarize_c_for_MI_computation: 
        axs["MI"].set_title(fr"$\widehat{{MI_{{\mathrm{{JSD}}}}}}(r;\ c_{{bin}})$ (clip={mi_clip})")
    else: 
        axs["MI"].set_title(fr"$\widehat{{MI_{{\mathrm{{JSD}}}}}}(r;\ c)$ (clip={mi_clip})")
    axs["MI"].set_xlabel("epochs") 
    axs["params"] = render_parameters(axs["params"], hp, t) 
    axs["G_init"].set_xlabel("OSNs", labelpad=5)
    axs["G_init"].set_ylabel("glomeruli")
    return fig, axs 

def plot_W(W_init, W_final, mis, hp, t, mi_clip=-1):
    mosaic = [["W_init", "W_final", "MI"], [".", ".", "params"]]
    fig, axs = plt.subplot_mosaic(
        mosaic, 
        figsize=(18, 10),
        gridspec_kw={"hspace": 0.5}
    )
    axs["W_init"].imshow(W_init)
    vmin = min(W_init.min(), W_final.min())
    vmax = max(W_init.max(), W_final.max()) 
    threshold = 0.1 * jnp.max(W_final, axis=1)
    im1 = axs["W_init"].imshow(W_init, vmin=vmin, vmax=vmax, aspect="auto", interpolation="nearest")
    im2 = axs["W_final"].imshow(W_final, vmin=vmin, vmax=vmax, aspect="auto", interpolation="nearest")
    cbar_label = r"$W_{ij}$"
    axs["W_init"].set_title(r"$W_{init}$")
    axs["W_final"].set_title(r"$W_{final}$")
    cbar = fig.colorbar(im1, ax=[axs["W_init"], axs["W_final"]], location="right", pad=0.05)
    cbar.ax.set_title(cbar_label, fontsize=16)
    axs["MI"].plot(jnp.clip(mis, mi_clip))
    if hp.binarize_c_for_MI_computation: 
        axs["MI"].set_title(fr"$\widehat{{MI_{{\mathrm{{JSD}}}}}}(r;\ c_{{bin}})$ (clip={mi_clip})")
    else: 
        axs["MI"].set_title(fr"$\widehat{{MI_{{\mathrm{{JSD}}}}}}(r;\ c)$ (clip={mi_clip})")
    axs["MI"].set_xlabel("epochs") 
    axs["params"] = render_parameters(axs["params"], hp, t) 
    axs["W_init"].set_xlabel("OSNs", labelpad=5)
    axs["W_init"].set_ylabel("glomeruli")
    return fig, axs 

def plot_activity(key, hp, p):
    key, *subkeys = jax.random.split(key, 3)
    draw_cs = ODOR_MODEL_REGISTRY[hp.odor_model]
    activity_function = ACTIVITY_FUNCTION_REGISTRY[hp.activity_model]
    cs = draw_cs(subkeys[0], hp)
    r = activity_function(hp, p, cs, subkeys[1])
    fig, ax = plt.subplots()
    im = ax.imshow(r, aspect="auto")
    fig.colorbar(im) 
    ax.set_xlabel("samples") 
    ax.set_ylabel("neurons") 
    fig.suptitle("activity") 
