"""
Contains the different visualization utilities used in jupyter notebooks.
"""

from typing import Callable, Iterable, Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm

from lid.base import LIDEstimator
from models.diffusions.sdes import Sde

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_histograms(
    *lid_sets,
    labels=None,
    colors=None,
    filename="fig_hist",
    title="Histogram of Estimated LID",
    _x_label="Estimated LID",
    _y_label="Frequency",
):
    """
    Plot histograms for multiple LID sets.

    Args:
        *lid_sets: lid_1, lid_2, ...
        labels: list of strings
        colors: list of colors (optional, same length as lid_sets)
    """

    sns.set_theme(style="darkgrid")

    # Unpack LID sets properly
    lids = [np.asarray(l).flatten() for l in lid_sets]
    n_sets = len(lids)

    # Default labels
    if labels is None:
        labels = [f"Set {i+1}" for i in range(n_sets)]

    # Handle colors
    if colors is None:
        colors = ['darkblue', 'crimson', 'teal', 'tan', 'darkviolet',
                  'olive', 'slateblue', 'coral', 'firebrick']
    if len(colors) < n_sets:
        raise ValueError(f"Not enough colors provided ({len(colors)} for {n_sets} sets).")

    # Compute shared x-range
    all_vals = np.concatenate(lids)
    xmin, xmax = np.min(all_vals), np.max(all_vals)

    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(111)
    ax.xaxis.offsetText.set_fontsize(24)

    ax.tick_params(axis='x', labelsize=24)
    ax.tick_params(axis='y', labelsize=24)

    ax.set_ylabel(_y_label, fontsize=24)
    ax.set_xlabel(_x_label, fontsize=24)

    bins = 40
    for i, (lid, label) in enumerate(zip(lids, labels)):
        sns.histplot(
            lid,
            label=label,
            color=colors[i],
            stat="count",
            kde=False,
            common_norm=False,
            element="bars",
            bins=bins,
            ax=ax,
        )

    legend = plt.legend(fontsize=18)
    plt.grid(alpha=1.0)
    plt.tight_layout()

    filename_fig = f"{filename}.png"
    plt.savefig(filename_fig, dpi=200, bbox_inches='tight')
    print('... saved', filename_fig)
    plt.show()

def plot_kde(
    *lid_sets,
    labels=None,
    colors=None,
    filename="fig_kde",
    title="KDE of Estimated LID",
    _x_label="Estimated LID",
    _y_label="Density",
):
    """
    KDE (density) plot for multiple LID sets.
    Same interface & visual style as plot_histograms.
    """

    sns.set_theme(style="darkgrid")

    # Convert LID sets to numpy
    lids = [np.asarray(l).flatten() for l in lid_sets]
    n_sets = len(lids)
    # Default labels
    if labels is None:
        labels = [f"Set {i+1}" for i in range(n_sets)]

    # Default colors (same palette as before)
    if colors is None:
        colors = ['crimson', 'darkblue', 'teal', 'tan', 'darkviolet',
                  'olive', 'slateblue', 'coral', 'firebrick']
        
    
    if len(colors) < n_sets:
        raise ValueError(
            f"Not enough colors provided ({len(colors)}) for {n_sets} LID sets."
        )

    # Shared figure settings
    #fig = plt.figure(figsize=(8, 4))
    fig = plt.figure(figsize=(5, 4))

    ax = fig.add_subplot(111)
    
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    #if np.min(lids) > 1000:
    all_vals = np.concatenate(lids)
    if np.min(all_vals) > 1000:
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))

    # Font sizes (same as histogram version)
    ax.xaxis.offsetText.set_fontsize(24)
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    ax.set_xlabel(_x_label, fontsize=20)
    ax.set_ylabel(_y_label, fontsize=20)

    # KDE plot for each LID set
    for i, (lid, label) in enumerate(zip(lids, labels)):
        if np.var(lid) < 0.01:
            lid = lid + 0.01 * np.random.randn(len(lid))
        sns.kdeplot(
            lid,
            fill=True,
            alpha=0.8,
            color=colors[i],
            label=label,
            linewidth=2.0,
            ax=ax,
        )

    # Legend
    legend = plt.legend(fontsize=13, loc='best', frameon=True)
    # legend = plt.legend(
    #     bbox_to_anchor=(0.5, 1.02),
    #     loc='lower center',
    #     borderaxespad=0,
    #     ncol=n_sets,
    #     fontsize=14,
    #     frameon=False
    # )
    legend.get_frame().set_alpha(0.8)
    
    #plt.xlim(17350, 22000) 
    
    plt.grid(alpha=1.0)
    plt.tight_layout()

    filename_fig = f"{filename}.png"
    plt.savefig(filename_fig, dpi=200, bbox_inches='tight')
    print("... saved", filename_fig)

    plt.show()


# 251204
import torch

def compute_cv_std(
    lid_matrix: torch.Tensor,
    dim: int = 0,
    eps: float = 1e-8,
) -> tuple[float, torch.Tensor]:
    """
    Compute the Coefficient of Variation (CV = std / mean) of LID estimates.
    """
    lid_matrix = lid_matrix.to(torch.float32)
    # mean and std over the scale dimension
    mu = lid_matrix.mean(dim=dim)                        # e.g. (num_points,)
    std = lid_matrix.std(dim=dim, unbiased=False)        # same shape as mu

    cv = std / (mu + eps)                                # same shape as mu
    mean_cv = cv.mean().item()
    mean_std = std.mean().item()
    return mean_cv, cv, mean_std, std

def plot_lid_on_a_grid_3d(
    data: torch.Tensor,
    lid_estimator,
    mode: Literal["with_preprocessing", "without_preprocessing"],
    argument_name: str,
    argument_values: Iterable,
    gt_lid: torch.Tensor | None = None,
    savepath: Optional[str] = None,
    lid_matrix: torch.Tensor | None = None,
    label_fmt: str = "t={:.3f}",  # or "delta={:.4f}" when you call it
    return_cv: bool = False,      # NEW: whether to compute Coefficient of Variation over argument_values
    **other_kwargs,
):
    """
    3D version of plot_lid_on_a_grid — each subplot has its own colorbar.

    Two modes:
      1) Backward-compatible mode (lid_matrix is None):
         - Uses lid_estimator + argument_name/argument_values to compute LID.
      2) Precomputed mode (lid_matrix is not None):
         - Uses rows of lid_matrix instead of calling lid_estimator.

    When gt_lid is provided, adds mean absolute error (MAE) to the titles.

    If return_cv=True, this function additionally computes the Coefficient of Variation (CV)
    of LID estimates across the given argument_values (e.g., different noise scales t).
    In that case, it returns (mean_cv, cv_per_point, lid_matrix_full).
    Otherwise it returns None (backward-compatible behavior).
    """

    import matplotlib.pyplot as plt
    from tqdm import tqdm

    argument_values = list(argument_values)
    M = len(argument_values)

    fig, axes = plt.subplots(4, 4, figsize=(18, 16), subplot_kw={"projection": "3d"})
    axes = axes.flatten()
    assert M == len(axes), "For a 4x4 grid, you need exactly 16 argument_values."

    # -------------------------------------------------
    # Optional preprocessing (old behavior)
    # -------------------------------------------------
    if lid_matrix is None:
        if mode == "with_preprocessing":
            artifact = lid_estimator.preprocess(data)
        elif mode == "without_preprocessing":
            artifact = None
        else:
            raise ValueError("Invalid mode; expected 'with_preprocessing' or 'without_preprocessing'.")
        # collect per-scale LID for later CV computation (shape: (M, N))
        collected_lid_rows = []
    else:
        # Precomputed mode: we won't use lid_estimator at all
        assert lid_matrix.shape == (
            M,
            data.shape[0],
        ), f"lid_matrix must be of shape (len(argument_values), N) = ({M}, {data.shape[0]}), got {lid_matrix.shape}"

    x, y, z = data[:, 0].cpu(), data[:, 1].cpu(), data[:, 2].cpu()

    # -------------------------------------------------
    # Main loop over grid cells
    # -------------------------------------------------
    for idx, (ax, arg_val) in enumerate(
        tqdm(
            list(zip(axes, argument_values)),
            total=M,
            desc="Computing 3D LID scatterplots"
        )
    ):
        # --- 1) Get per-point LID either from estimator (old) or from lid_matrix (new) ---
        if lid_matrix is not None:
            # Precomputed mode
            all_lid = lid_matrix[idx].cpu()
        else:
            # Backward-compatible mode: compute LID on the fly
            if mode == "with_preprocessing":
                all_lid = lid_estimator.compute_lid_from_artifact(
                    artifact,
                    **{argument_name: arg_val},
                    **other_kwargs,
                ).cpu()
            elif mode == "without_preprocessing":
                all_lid = lid_estimator.estimate_lid(
                    data,
                    **{argument_name: arg_val},
                    **other_kwargs,
                ).cpu()
            else:
                raise ValueError("Invalid mode")

            # collect for CV computation later (shape per row: (1, N))
            if return_cv:
                collected_lid_rows.append(all_lid.unsqueeze(0))

        # --- 2) MAE if ground truth is available ---
        if gt_lid is not None:
            mae = torch.mean(torch.abs(all_lid.float() - gt_lid.float().cpu())).item()
            title = f"{label_fmt.format(float(arg_val))}, MAE={mae:.3f}"
        else:
            title = label_fmt.format(float(arg_val))

        # --- 3) Scatter plot ---
        #s = ax.scatter(x, y, z, c=all_lid, cmap="cividis", s=12)
        s = ax.scatter(x, y, z, c=all_lid, cmap="coolwarm", s=12)
        #s = ax.scatter(x, y, z, c=all_lid, cmap="magma", s=12)


        ax.set_title(title, fontsize=20)
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)

        cbar = fig.colorbar(s, ax=ax, pad=0.05, fraction=0.046)
        ticklabs = cbar.ax.get_yticklabels()
        cbar.ax.set_yticklabels(ticklabs, fontsize=20)
        cbar.set_label("Estimated LID", rotation=90, labelpad=10, fontsize=20)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])

    fig.tight_layout()

    if savepath:
        plt.tight_layout()
        plt.savefig(savepath,
                    bbox_inches="tight",
                    pad_inches=0.02,
                    dpi=300,
)

    plt.show()

    # -------------------------------------------------
    # Optionally compute Coefficient of Variation (CV)
    # -------------------------------------------------
    if not return_cv:
        # Backward-compatible: no return value
        return None

    # lid_matrix_full: shape (M, N) with rows = argument_values (e.g., different t)
    if lid_matrix is not None:
        lid_matrix_full = lid_matrix.detach().cpu()
    else:
        lid_matrix_full = torch.cat(collected_lid_rows, dim=0)  # (M, N)

    mean_cv, cv_per_point, mean_std, std_per_point = compute_cv_std(lid_matrix_full, dim=0)  # CV over argument_values per point

    return mean_cv, mean_std, lid_matrix_full

#

def plot_lid_on_a_grid(
    data,
    lid_estimator: LIDEstimator,
    mode: Literal["with_preprocessing", "without_preprocessing"],
    argument_name: str,
    argument_values: Iterable,
    **other_kwargs,  # the other kwargs that will be passed to the lid estimator
):
    fig, axes = plt.subplots(4, 4, figsize=(16, 13))
    # fig, axes = plt.subplots(1, 2, figsize=(16, 16))

    assert len(argument_values) == len(axes.flatten())
    if mode == "with_preprocessing":
        artifact = lid_estimator.preprocess(data)
    for ax, arg_val in tqdm(
        zip(axes.flatten(), argument_values),
        desc="computing scatterplot",
        total=len(argument_values),
    ):  # Generate 1k points and plot them

        if mode == "with_preprocessing":
            all_lid = lid_estimator.compute_lid_from_artifact(
                artifact,
                **{argument_name: arg_val},
                **other_kwargs,
            ).cpu()
        elif mode == "without_preprocessing":
            all_lid = lid_estimator.estimate_lid(
                data, **{argument_name: arg_val}, **other_kwargs
            ).cpu()
        else:
            raise ValueError("Invalid mode")

        # clip LID values
        # 251013
        #all_lid = np.clip(all_lid, 0, lid_estimator.ambient_dim)

        #s = ax.scatter(*data.cpu().T, c=all_lid, cmap="plasma", vmin=0, vmax=2)
        #s = ax.scatter(*data.cpu().T, c=all_lid, cmap="cividis", vmin=0, vmax=2)
        s = ax.scatter(*data.cpu().T, c=all_lid, cmap="magma")

        ax.set_title(f"{argument_name}={round(arg_val, 3)}")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(s, cax=cax, orientation="vertical")

        #cbar.set_label(f"$LID({{{round(arg_val, 3)}}})(\\cdot)$", rotation=90, labelpad=15)

    fig.tight_layout()
    plt.show()


def plot_log_prob_on_a_grid(
    data: torch.Tensor,
    argument_name: str,
    argument_values: Iterable,
    sde: Sde | None = None,
    log_prob_fn: Callable | None = None,
    **log_prob_kwargs,  # log prob kwargs
):
    fig, axes = plt.subplots(4, 4, figsize=(16, 13))
    assert len(argument_values) == len(axes.flatten())

    if log_prob_fn is None:
        assert sde is not None, "sde must be provided if log_prob_fn is not provided"
        log_prob_fn = sde.log_prob
    else:
        assert sde is None, "sde must not be provided if log_prob_fn is provided"

    epsilon_cnt = 0
    for ax, arg_val in tqdm(zip(axes.flatten(), argument_values), total=len(argument_values)):
        epsilon_cnt += 1
        all_log_probs = log_prob_fn(x=data, **{argument_name: arg_val}, **log_prob_kwargs)
        # turn all_log_probs into their ranks

        heatmap = torch.exp(all_log_probs.cpu())
        # Graph the norms
        s = ax.scatter(*data.T.cpu().numpy(), c=heatmap, cmap="plasma")

        ax.set_title(f"{argument_name} = {arg_val}")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(s, cax=cax, orientation="vertical")

        # Add a label to the colorbar
        cbar.set_label("$p(\\cdot)$", rotation=90, labelpad=15)
    fig.tight_layout()


def plot_lid_trend_simple(
    data,
    lid_estimator: LIDEstimator,
    mode: Literal["with_preprocessing", "without_preprocessing"],
    argument_name: str,
    argument_values: Iterable,
    **other_kwargs,  # the other kwargs that will be passed to the lid estimator
):
    lid = []
    x_axis = []

    if mode == "with_preprocessing":
        artifact = lid_estimator.preprocess(data)
    for arg_val in tqdm(argument_values):
        if mode == "with_preprocessing":
            all_lid = lid_estimator.compute_lid_from_artifact(
                artifact,
                **{argument_name: arg_val},
                **other_kwargs,
            ).cpu()
        elif mode == "without_preprocessing":
            all_lid = lid_estimator.estimate_lid(
                data, **{argument_name: arg_val}, **other_kwargs
            ).cpu()
        else:
            raise ValueError("Invalid mode")

        lid.append(all_lid)
        x_axis.append(arg_val)

    lid = torch.stack(lid).T.cpu().numpy()
    x_axis = np.array(x_axis)

    for i in range(len(lid)):
        plt.plot(x_axis, lid[i], alpha=0.1)
    # take the average lid estimate
    avg_lid = lid.mean(axis=0)
    plt.plot(x_axis, avg_lid, color="red")

    #plt.title(f"LID estimates")
    plt.xlabel(argument_name)
    plt.ylabel(f"$LID(\\cdot; {argument_name})$")

    plt.show()
