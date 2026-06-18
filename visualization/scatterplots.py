from typing import List, Optional, Tuple

import numpy as np
import seaborn as sns
import torch
import umap
import umap.umap_ as umap
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

from .pretty import FONT_FAMILY, ColorTheme, StyleDecorator, hashlines, savable


def visualize_estimates_heatmap(
    data: np.ndarray,  # [data_size, dim]
    estimates: np.ndarray,  # [data_size]
    title: str,
    max_estimate: float | None = None,
    min_estimate: float | None = None,
    return_img: bool = False,
    alpha: float = 0.1,
    reducer: umap.UMAP | None = None,
):
    """
    This is used when we want to plot an estimand that is assigned
    to all the points of the data.

    What this function does is that it takes all the estimated values
    and then performs a umap embedding (if needed) on data, then
    provides a heatmap where the intensity of a point represents the value
    of the estimand.

    To visualize LID values, for example, we can use this method.
    """
    # train a UMAP embedding on all the data
    if data.shape[1] > 2:
        title = f"{title} (UMAP projection)"
        if reducer is None:
            reducer = umap.UMAP()
            reducer.fit(data)
        embedding = reducer.transform(data)
    else:
        embedding = data
    try:
        s = plt.scatter(
            embedding[:, 0],
            embedding[:, 1],
            alpha=alpha,
            cmap="plasma",
            c=estimates,
            vmin=min_estimate,
            vmax=max_estimate,
        )
        plt.title(title)
        # fix the colorbar to a
        plt.colorbar(s, orientation="vertical")

    finally:

        img = None
        if not return_img:
            plt.show()
        else:
            fig = plt.gcf()
            fig.canvas.draw()
            np_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(
                fig.canvas.get_width_height()[::-1] + (3,)
            )
            # create a PIL image out of the np_array of size (H x W x 3)
            img = Image.fromarray(np_array)
        plt.close()

    return img, reducer


@savable
@StyleDecorator(font_scale=2, style="whitegrid", line_style="--")
def pretty_visualize_estimates_heatmap(
    data: np.ndarray,  # [data_size, dim]
    estimates: np.ndarray,  # [data_size]
    max_estimate: float | None = None,
    min_estimate: float | None = None,
    alpha: float = 0.1,
    reducer: umap.UMAP | None = None,
    colorbar_label: str | None = None,
    figsize: Optional[tuple] = (7, 6),
    fontsize: Optional[int] = None,
    no_legend: bool = False,
    legend_fontsize: Optional[int] = None,
    legend_loc: Optional[str] = None,
    title: str | None = None,
    custom_xticks: Optional[int] = None,
    custom_yticks: Optional[int] = None,
    custom_zticks: Optional[int] = None,
    box_ratios: Optional[List[int]] = None,
    remove_ticks_x_label: bool = False,
    remove_ticks_y_label: bool = False,
    remove_ticks_z_label: bool = False,
    cbar_ticks: List[float] | None = None,
    xlim: Tuple | None = None,
    ylim: Tuple | None = None,
    zlim: Tuple | None = None,
):
    """
    This is used when we want to plot an estimand that is assigned
    to all the points of the data.

    What this function does is that it takes all the estimated values
    and then performs a umap embedding (if needed) on data, then
    provides a heatmap where the intensity of a point represents the value
    of the estimand.

    To visualize LID values, for example, we can use this method.
    """

    # train a UMAP embedding on all the data
    assert data.shape[1] in [2, 3], "Only 2 and 3 dimensional data is covered"

    colors = [ColorTheme.RED_FIRST.value, ColorTheme.BLUE_FIRST.value, ColorTheme.GOLD.value]  #
    cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)
    estimates = np.clip(estimates, min_estimate, max_estimate)

    if data.shape[1] == 2:
        fig, ax = plt.subplots(figsize=figsize)
        scatter = sns.scatterplot(
            x=data[:, 0],
            y=data[:, 1],
            palette=cmap,
            legend=None,
            s=100,
            alpha=alpha,
            edgecolor="none",
            hue=estimates,
        )

    if data.shape[1] == 3:
        # Create a figure and a 3D axis
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
        all_alphas = np.concatenate([alpha * np.ones(len(data)), np.zeros(2)])
        all_points = np.concatenate([data, np.array([[0, 0, -1000.5], [0, 0, +1000.5]])])
        all_estimates = np.concatenate([estimates, np.array([min_estimate, max_estimate])])
        ax.scatter(
            all_points[:, 0],
            all_points[:, 1],
            all_points[:, 2],
            c=all_estimates,
            cmap=cmap,
            alpha=all_alphas,
        )
        if box_ratios:
            # Adjust the aspect ratio to be equal
            ax.set_box_aspect(box_ratios)
    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)
    if zlim:
        ax.set_zlim(*zlim)
    norm = plt.Normalize(vmin=min_estimate, vmax=max_estimate)
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap, norm=norm),
        ax=ax,
        orientation="vertical",
        label=colorbar_label if colorbar_label is not None else "estimates",
    )
    if cbar_ticks:
        # Set colorbar ticks and tick labels
        tick_labels = [str(tick) for tick in cbar_ticks]
        cbar.set_ticks(cbar_ticks)
        cbar.set_ticklabels(tick_labels)

    if custom_xticks:
        ax.set_xticks(custom_xticks)
    if custom_yticks:
        ax.set_yticks(custom_yticks)
    if custom_zticks:
        ax.set_zticks(custom_zticks)

    if remove_ticks_x_label:
        # Set tick parameters for the axes to white
        ax.tick_params(axis="x", colors="white")

    if remove_ticks_y_label:
        ax.tick_params(axis="y", colors="white")

    if remove_ticks_z_label and data.shape[1] == 3:
        ax.tick_params(axis="z", colors="white")

    ax.legend(loc=legend_loc, prop={"family": FONT_FAMILY, "size": legend_fontsize})
    if no_legend:
        ax.legend_.remove()

    if title is not None:
        ax.set_title(title, fontsize=fontsize, fontdict={"family": FONT_FAMILY})

    return ax

from typing import List
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA

# assuming you already have something like this in your codebase
# from your_module import ColorTheme


def visualize_pca_clusters(
    data: List[torch.Tensor | np.ndarray] | torch.Tensor | np.ndarray,
    labels: List[str] | str | None = None,
    title: str = "PCA embeddings",
    alpha: List[float] | float = 0.1,
    colors: List[str] | None = None,
    return_img: bool = False,
    pca: PCA | None = None,
    return_pca: bool = False,
    n_components: int = 2,
) -> Image:
    """
    Visualize multiple datasets in a shared 2D PCA space.

    Args:
        data: list of tensors/arrays, or a single tensor/array.
              Each element is shape [N_i, D].
        labels: list of labels (one per dataset) or single string.
        title: plot title.
        alpha: float or list of floats (transparency per dataset).
        colors: optional list of colors; if None, ColorTheme.get_colors is used.
        return_img: if True, return a PIL.Image instead of showing the plot.
        pca: optional sklearn PCA object to reuse; if None, a new one is fit.
        return_pca: if True, return (img, pca) instead of just img.
        n_components: number of PCA components (2 by default for 2D plot).

    Returns:
        img or (img, pca) depending on return_pca.
    """

    # Normalize input shapes/types
    if isinstance(data, (torch.Tensor, np.ndarray)):
        data = [data]

    if labels is None:
        labels = [f"set {i}" for i in range(len(data))]
    elif isinstance(labels, str):
        labels = [labels]

    assert len(data) == len(labels), "Data and labels should have the same length."

    if isinstance(alpha, float):
        alpha = [alpha] * len(data)
    assert len(alpha) == len(data), "Alpha and data should have the same length."

    # Convert all to numpy
    for i in range(len(data)):
        if isinstance(data[i], torch.Tensor):
            data[i] = data[i].detach().cpu().numpy()

    # Concatenate for joint PCA fit/transform
    data_concatenated = np.concatenate(data, axis=0)  # [sum N_i, D]

    # Colors
    colors = colors or ColorTheme.get_colors(len(data))

    try:
        # If D > 2, apply PCA, else just plot directly
        if data_concatenated.shape[1] > 2:
            if pca is None:
                pca = PCA(n_components=n_components)
                pca.fit(data_concatenated)
            all_embeddings = pca.transform(data_concatenated)[:, :2]
        else:
            all_embeddings = data_concatenated

        # Split back per dataset and plot
        offset = 0
        for i, (d_i, lbl, a_i, col_i) in enumerate(zip(data, labels, alpha, colors)):
            N_i = len(d_i)
            embeddings = all_embeddings[offset:offset + N_i]
            offset += N_i

            plt.scatter(
                embeddings[:, 0],
                embeddings[:, 1],
                alpha=a_i,
                color=col_i,
                s=10,
            )
            # dummy point for legend
            plt.scatter([], [], color=col_i, label=lbl)

        plt.legend()
        plt.title(title)
        plt.xlabel("PC 1")
        plt.ylabel("PC 2")
    finally:
        img = None
        if not return_img:
            plt.show()
        else:
            fig = plt.gcf()
            fig.canvas.draw()
            np_array = np.frombuffer(
                fig.canvas.tostring_rgb(), dtype=np.uint8
            ).reshape(fig.canvas.get_width_height()[::-1] + (3,))
            img = Image.fromarray(np_array)
        plt.close()

    if return_pca:
        return img, pca
    return img


def visualize_umap_clusters(
    data: List[torch.Tensor | np.ndarray] | torch.Tensor | np.ndarray,
    labels: List[str] | str | None = None,
    title: str = "UMAP embeddings",
    alpha: List[float] | float = 0.1,
    colors: List[str] | None = None,
    return_img: bool = False,
    reducer: umap.UMAP | None = None,
    return_reducer: bool = False,
) -> Image:
    # Some checks
    if isinstance(data, (torch.Tensor, np.ndarray)):
        data = [data]
    if isinstance(labels, str):
        labels = [labels]
    assert len(data) == len(labels), "Data and labels should have the same length."
    if isinstance(alpha, float):
        alpha = [alpha] * len(data)
    assert len(alpha) == len(data), "Alpha and data should have the same length."

    # Turn everything into numpy arrays
    for i in range(len(data)):
        if isinstance(data[i], torch.Tensor):
            data[i] = data[i].cpu().numpy()

    # concatenate everything before passing to UMAP
    data_concatenated = np.concatenate(data, axis=0)
    # get the colors for visualizing the clusters
    colors = colors or ColorTheme.get_colors(len(data))

    # train a UMAP model and then visualize all the clusters
    try:
        if data_concatenated.shape[1] > 2:
            if not reducer:
                reducer = umap.UMAP()
                reducer.fit(data_concatenated)
            all_embeddings = reducer.transform(data_concatenated)
        else:
            all_embeddings = data_concatenated
        for i in range(len(data)):
            L = sum(len(d) for d in data[:i])
            R = sum(len(d) for d in data[: i + 1])
            embeddings = all_embeddings[L:R]
            plt.scatter(embeddings[:, 0], embeddings[:, 1], alpha=alpha, color=colors[i])
            plt.scatter([], [], color=colors[i], label=labels[i])
        plt.legend()
        plt.title(title)
    finally:
        img = None
        if not return_img:
            plt.show()
        else:
            fig = plt.gcf()
            fig.canvas.draw()
            np_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(
                fig.canvas.get_width_height()[::-1] + (3,)
            )
            # create a PIL image out of the np_array of size (H x W x 3)
            img = Image.fromarray(np_array)
        plt.close()
    return img if not return_reducer else (img, reducer)


# 3D scatter plotting

from typing import List, Optional, Tuple, Union
ArrayLike = Union[np.ndarray, torch.Tensor]

def _to_numpy3(x: ArrayLike) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    assert x.ndim == 2, f"Expected (N,3), got {x.shape}"
    if x.shape[1] != 3:
        raise ValueError(f"3D scatter expects 3 columns; got {x.shape[1]}.")
    return x

def _set_axes_equal(ax):
    # Equal aspect for 3D: make axes ranges equal
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0]); x_mid = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0]); y_mid = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0]); z_mid = np.mean(z_limits)
    max_range = max([x_range, y_range, z_range])
    ax.set_xlim3d([x_mid - max_range/2, x_mid + max_range/2])
    ax.set_ylim3d([y_mid - max_range/2, y_mid + max_range/2])
    ax.set_zlim3d([z_mid - max_range/2, z_mid + max_range/2])

def set_axes_equal_3d(ax, points: np.ndarray, pad: float = 0.05):
    """
    Make a 3D plot have equal scale on all axes by setting identical half-ranges
    around the data centroid. 'points' is (N,3).
    """
    pts = np.asarray(points)
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    center = (maxs + mins) / 2.0
    spans = (maxs - mins)
    # use the largest span so the data fits in a cube
    radius = 0.5 * spans.max() * (1.0 + pad)

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))  # crucial for true equal scaling in 3D


def visualize_3d_clusters(
    data: List[ArrayLike] | ArrayLike,
    labels: Optional[List[str] | str] = None,
    title: str = "3D scatter",
    alpha: List[float] | float = 1.0,
    colors: Optional[List[str]] = None,
    sizes: List[float] | float = 20,
    figsize: Tuple[int, int] = (12, 10),
    elev: float = 20.0,
    azim: float = -35.0,
    box_ratios: Optional[Tuple[float, float, float]] = None,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    zlim: Optional[Tuple[float, float]] = None,
    legend_loc: Optional[str] = "best",
    return_img: bool = False,
    savepath: Optional[str] = None,
    gt_lid: Optional[ArrayLike] = None,   # ground-truth LID (optional)
    lid_val: Optional[ArrayLike] = None,  # predicted/estimated LID (optional; drives coloring)
    cmap: str = "cividis",
):
    """
    Simple 3D scatter (no UMAP) for multiple point clouds in R^3.

    New behavior:
      - If lid_val is provided: color points by lid_val.
      - If gt_lid is also provided: compute MAE = mean(|gt_lid - lid_val|) and append to title.
      - If lid_val is NOT provided: no per-point colormap; all clusters use the given solid colors.
      - If gt_lid is provided but lid_val is NOT: raise ValueError (as requested).
      - If gt_lid is NOT provided: do not compute MAE.

    Notes:
      - When coloring by lid_val, all data arrays are concatenated and plotted together.
      - When not coloring, each data array is plotted with its own solid color and label.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image
    import torch

    # --- normalize inputs ---
    if isinstance(data, (np.ndarray, torch.Tensor)):
        data = [data]
    data_np = [_to_numpy3(d) for d in data]
    n_total = sum(x.shape[0] for x in data_np)

    if isinstance(labels, str):
        labels = [labels]
    if labels is None:
        labels = [f"set {i}" for i in range(len(data_np))]
    assert len(labels) == len(data_np), "labels length must match number of datasets"

    if isinstance(alpha, (int, float)):
        alpha = [float(alpha)] * len(data_np)
    if isinstance(sizes, (int, float)):
        sizes = [float(sizes)] * len(data_np)

    # --- rule (4): if gt_lid is passed but lid_val is not -> error ---
    if gt_lid is not None and lid_val is None:
        raise ValueError("gt_lid was provided but lid_val was not. Please pass lid_val (predictions) to compute MAE.")

    # Prepare colors for the non-colored mode
    if colors is None:
        try:
            from .pretty import ColorTheme
            colors = ColorTheme.get_colors(len(data_np))
        except Exception:
            colors = [None] * len(data_np)

    # --- set up figure ---
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    #ax.view_init(elev=elev, azim=azim)

    # Concatenate when needed (for coloring by lid_val)
    if lid_val is not None:
        X = np.concatenate(data_np, axis=0)  # (N,3)
        lid_val_np = np.asarray(lid_val).reshape(-1)
        assert lid_val_np.shape[0] == n_total, "lid_val length must match total number of points"
        sc = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=lid_val_np, cmap=cmap, s=sizes[0], alpha=alpha[0])
        cbar = plt.colorbar(sc, ax=ax, pad=0.03, fraction=0.03)
         
        ticklabs = cbar.ax.get_yticklabels()
        cbar.ax.set_yticklabels(ticklabs, fontsize=24)
        cbar.set_label("Estimated LID", rotation=90, labelpad=10, fontsize=24)
        handles = [sc]


        # MAE if gt_lid provided (rule 3 & 5)
        mae_txt = ""
        if gt_lid is not None:
            gt_lid_np = np.asarray(gt_lid).reshape(-1)
            assert gt_lid_np.shape[0] == n_total, "gt_lid length must match total number of points"
            mae = np.mean(np.abs(gt_lid_np - lid_val_np))
            mae_txt = f", MAE={mae:.3f}"
        ax.set_title(f"{title}{mae_txt}", fontsize=24)

    else:
        # No lid_val -> solid color per cluster (rule 2)
        handles = []
        for i, X in enumerate(data_np):
            h = ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=sizes[i], alpha=alpha[i], color=colors[i])
            handles.append(h)
        ax.set_title(title, fontsize=24)
        # Legend only meaningful in non-colored mode
        ax.legend(handles, labels, loc=legend_loc)

    # Limits
    all_pts = np.concatenate(data_np, axis=0)
    if xlim: ax.set_xlim(*xlim)
    if ylim: ax.set_ylim(*ylim)
    if zlim: ax.set_zlim(*zlim)

    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    ax.tick_params(axis='z', labelsize=20)

    
    # Optional save/return
    img = None
    if savepath:
        plt.tight_layout()
        plt.savefig(savepath, dpi=150)
    if not return_img:
        plt.show()
    else:
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        img = Image.fromarray(buf.reshape(h, w, 3))
    plt.close(fig)
    return img

