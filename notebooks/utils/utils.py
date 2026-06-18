from __future__ import annotations
import torch
from torch import nn
import numpy as np
from tqdm import tqdm
from PIL import Image
import io
import time

import torch
import time

import torch
import torch.autograd as autograd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm
import os, sys

import torch
from torch import autograd
from tqdm import tqdm


import torch
import torch.autograd as autograd
from tqdm import tqdm

import torch
import torch.autograd as autograd
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import os

################### 260131 
# ============================================================
# Refactor + add 3 y-axis measures:
#   (A) density drop along eigenmode (score-line-integral proxy)
#   (B) score growth along eigenmode
#   (C) diffusion-time persistence (slope of v^T H_t v vs t)
# ============================================================

import os, math
import numpy as np
import torch
import torch.autograd as autograd
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors


# -----------------------------
# 0) Hessian eigendecomposition
# -----------------------------
def compute_hessian_eig_full(sde, x, t, show_progress=True):
    """
    Computes full eigenvalues/eigenvectors of Jacobian of score (symmetrized).
    Returns:
      eigvals_hessian_flat: numpy (B*D,)
      vecs_np: numpy (B, D, D)  (columns are eigenvectors)
    NOTE: This is O(D^2) memory and O(D^3) time due to full eigendecomp.
    Recommended: batch_size=1 and small D.
    """
    x = x.detach().clone().requires_grad_(True)
    B = x.shape[0]
    D = x.numel() // B

    # t batch
    if isinstance(t, (int, float)):
        t_batch = torch.full((B,), float(t), device=x.device, dtype=x.dtype)
    else:
        t_batch = t.to(device=x.device, dtype=x.dtype)

    # Score and flattened
    score = sde.score(x, t_batch)
    score_flat = score.reshape(B, -1)

    # Build Jacobian rows (B,D,D)
    hessian_rows = []
    iterator = range(D)
    if show_progress and D > 100:
        iterator = tqdm(iterator, desc=f"Computing Jacobian of score ({D} dims)")

    for i in iterator:
        grad_i = autograd.grad(
            outputs=score_flat[:, i].sum(),
            inputs=x,
            create_graph=False,
            retain_graph=True,
        )[0]
        hessian_rows.append(grad_i.reshape(B, -1).unsqueeze(1))

    jacobian_batch = torch.cat(hessian_rows, dim=1)          # (B,D,D)
    jacobian_batch = (jacobian_batch + jacobian_batch.transpose(1, 2)) / 2.0

    # Eigen decomposition of symmetric Jacobian
    vals, vecs = torch.linalg.eigh(jacobian_batch)           # vals (B,D), vecs (B,D,D)

    # Convert to Hessian eigenvalues: H = -∂s/∂x = -J
    eigvals_hessian = -vals

    eigvals_flat = eigvals_hessian.detach().cpu().numpy().reshape(-1)
    vecs_np = vecs.detach().cpu().numpy()
    return eigvals_flat, vecs_np


# ----------------------------------------
# 1) Measure: Spatial smoothness (original)
# ----------------------------------------
def measure_spatial_smoothness(vecs_np, img_shape=(3, 32, 32)):
    """
    Lag-1 neighbor autocorrelation (Right + Down) for each eigenvector.
    Interface: takes vecs_np (B,D,D) and returns y_flat (B*D,)
    """
    C, H, W = img_shape
    B, D, _ = vecs_np.shape

    # columns are eigenvectors -> transpose to have eigenvectors as rows
    vecs_t = np.transpose(vecs_np, (0, 2, 1))  # (B, D_eig, D_flat)

    ys = []
    for b in range(B):
        v_imgs = vecs_t[b].reshape(D, C, H, W)  # (D,C,H,W)

        # neighbor products (down + right)
        diff_h = (v_imgs[:, :, :-1, :] * v_imgs[:, :, 1:, :]).sum(axis=(1, 2, 3))
        diff_w = (v_imgs[:, :, :, :-1] * v_imgs[:, :, :, 1:]).sum(axis=(1, 2, 3))
        ys.append(diff_h + diff_w)

    y = np.stack(ys, axis=0).reshape(-1)
    return y


# ---------------------------------------------------------
# Helper: chunked batched score calls with eigenvector batch
# ---------------------------------------------------------
def _score_dot_v_chunked(sde, x0, t_val, v_imgs, eps, chunk_size=64):
    """
    Compute dot products:
      a = <s(x0), v>
      b = <s(x0 + eps v), v>
    for each v in v_imgs (K,C,H,W), chunked.
    Returns:
      a_all, b_all: torch tensors (K,)
    """
    device = x0.device
    dtype = x0.dtype
    K = v_imgs.shape[0]

    # ensure unit-norm eigenvectors (in pixel space)
    v_flat = v_imgs.reshape(K, -1)
    v_norm = torch.linalg.norm(v_flat, dim=1, keepdim=True).clamp_min(1e-12)
    v_imgs_unit = (v_flat / v_norm).reshape_as(v_imgs)

    a_list, b_list = [], []
    with torch.no_grad():
        for start in range(0, K, chunk_size):
            end = min(K, start + chunk_size)
            v_chunk = v_imgs_unit[start:end]
            k = v_chunk.shape[0]

            x_rep = x0.repeat(k, 1, 1, 1)
            t_rep = torch.full((k,), float(t_val), device=device, dtype=dtype)

            s0 = sde.score(x_rep, t_rep).reshape(k, -1)
            v0 = v_chunk.reshape(k, -1)
            a = (s0 * v0).sum(dim=1)

            x_eps = x_rep + eps * v_chunk
            s1 = sde.score(x_eps, t_rep).reshape(k, -1)
            b = (s1 * v0).sum(dim=1)

            a_list.append(a.detach().cpu())
            b_list.append(b.detach().cpu())

    return torch.cat(a_list, dim=0), torch.cat(b_list, dim=0)

# -------------------------------------------------------
# 2) Measure A: Density drop proxy via score line integral
# -------------------------------------------------------
def measure_density_drop(sde, x, t, vecs_np, eps=1e-3, img_shape=(3,32,32), chunk_size=64):
    """
    Measures -Delta log p_t approx -eps * score_dot_v
    """
    # --- FIX: Explicitly get device/dtype from input x ---
    device = x.device
    dtype = x.dtype
    # -----------------------------------------------------
    
    C, H, W = img_shape
    device = x.device
    dtype = x.dtype

    # Ensure batch compatibility but strongly recommended B=1 for cost confirmation
    B = x.shape[0]
    _, D, _ = vecs_np.shape
    vecs_t = np.transpose(vecs_np, (0, 2, 1))  # (B, D, D)

    y_all = []
    for b in range(B):
        x0 = x[b:b+1].detach()  # (1,C,H,W)
        v_imgs = torch.from_numpy(vecs_t[b].reshape(D, C, H, W)).to(device=device, dtype=dtype)

        a, bdot = _score_dot_v_chunked(
            sde=sde, x0=x0, t_val=t, v_imgs=v_imgs, eps=eps, chunk_size=chunk_size
        )
        # trapezoid integral
        delta_logp = eps * 0.5 * (a + bdot)  # (D,)
        drop = (-delta_logp).numpy()
        y_all.append(drop)

    return np.concatenate(y_all, axis=0)

# ---------------------------------------------
# 3) Measure B: Score growth along eigenvector
# ---------------------------------------------
def measure_score_growth(sde, x, t, vecs_np, eps=1e-3, img_shape=(3,32,32), chunk_size=64):
    """
    Measures ||s(x + eps*v) - s(x)||
    """
    # --- FIX: Explicitly get device/dtype from input x ---
    device = x.device
    dtype = x.dtype
    # -----------------------------------------------------

    C, H, W = img_shape
    device = x.device
    dtype = x.dtype
    B = x.shape[0]
    _, D, _ = vecs_np.shape
    vecs_t = np.transpose(vecs_np, (0, 2, 1))  # (B, D, D)

    y_all = []
    for b in range(B):
        x0 = x[b:b+1].detach()  # (1,C,H,W)
        v_imgs = torch.from_numpy(vecs_t[b].reshape(D, C, H, W)).to(device=device, dtype=dtype)

        # unit norm
        v_flat = v_imgs.reshape(D, -1)
        v_norm = torch.linalg.norm(v_flat, dim=1, keepdim=True).clamp_min(1e-12)
        v_imgs_unit = (v_flat / v_norm).reshape_as(v_imgs)

        growth_list = []
        for start in range(0, D, chunk_size):
            end = min(D, start + chunk_size)
            v_chunk = v_imgs_unit[start:end]
            k = v_chunk.shape[0]

            x_rep = x0.repeat(k, 1, 1, 1)
            t_rep = torch.full((k,), float(t), device=device, dtype=dtype)

            s0 = sde.score(x_rep, t_rep).reshape(k, -1)
            s1 = sde.score(x_rep + eps * v_chunk, t_rep).reshape(k, -1)
            g = torch.linalg.norm(s1 - s0, dim=1)

            growth_list.append(g.detach().cpu())

        y_all.append(torch.cat(growth_list, dim=0).numpy())

    return np.concatenate(y_all, axis=0)


# ------------------------------------------------------------------------
# Helper: quadratic form v^T H_t v = - v^T (∂s/∂x) v via one autograd grad
# ------------------------------------------------------------------------
def _hessian_quadform_along_v(
    sde,
    x0,         # (1,C,H,W)
    t_val,
    v_imgs,     # (K,C,H,W) assumed unit norm
):
    """
    Computes q = v^T H_t v for each v in a batch (K,), where H_t = -∂s/∂x.
    Implementation:
      inner_k = <s(x), v_k>
      grad_x inner_k = (∂s/∂x)^T v_k  (same if symmetrized)
      quad_k = <grad_x inner_k, v_k> = v_k^T (∂s/∂x) v_k
      return -quad_k
    """
    device = x0.device
    dtype = x0.dtype
    K = v_imgs.shape[0]

    x_rep = x0.repeat(K, 1, 1, 1).detach().clone().requires_grad_(True)  # (K,C,H,W)
    t_rep = torch.full((K,), float(t_val), device=device, dtype=dtype)

    score = sde.score(x_rep, t_rep).reshape(K, -1)
    v_flat = v_imgs.reshape(K, -1)

    inner = (score * v_flat).sum(dim=1)  # (K,)
    grad = autograd.grad(inner.sum(), x_rep, create_graph=False, retain_graph=False)[0].reshape(K, -1)
    quad_j = (grad * v_flat).sum(dim=1)  # v^T (∂s/∂x) v

    return (-quad_j).detach()            # v^T H v


# ---------------------------------------------------------------
# 4) Measure C: Diffusion-time persistence (slope vs log t)
# ---------------------------------------------------------------
def _collect_q_over_t(
    sde,
    x0,           # (1,C,H,W)
    v_imgs_unit,  # (D,C,H,W) unit vectors
    t_list,
    chunk_size=32,
):
    """
    Collect raw Q matrix: Q[i, j] = v_i^T H_{t_j} v_i (can be negative).
    Returns:
      Q: np.ndarray (D,T) float64 WITHOUT any flooring/clamping.
    """
    D = v_imgs_unit.shape[0]
    T = len(t_list)
    Q_out = np.zeros((D, T), dtype=np.float64)

    for start in range(0, D, chunk_size):
        end = min(D, start + chunk_size)
        v_chunk = v_imgs_unit[start:end]  # (k,C,H,W)

        qs = []
        for tt in t_list:
            q = _hessian_quadform_along_v(sde, x0, tt, v_chunk)  # (k,)
            qs.append(q.detach().cpu().numpy().astype(np.float64))

        Q = np.stack(qs, axis=1)  # (k,T)
        Q_out[start:end, :] = Q

    return Q_out
def _robust_rowwise_slope(logt, Y, valid_mask, min_points=3):
    """
    Fit slope per row of Y (shape D,T) against logt (shape T,),
    using only entries where valid_mask is True.
    Returns slopes (D,) with NaN when not enough points.
    """
    D, T = Y.shape
    slopes = np.full((D,), np.nan, dtype=np.float64)

    for i in range(D):
        m = valid_mask[i]
        if m.sum() < min_points:
            continue
        x = logt[m]
        y = Y[i, m]
        x = x - x.mean()
        y = y - y.mean()
        denom = (x * x).sum()
        if denom < 1e-30:
            continue
        slopes[i] = (x * y).sum() / denom

    return slopes


def measure_time_persistence_norm_slope(
    sde,
    x,
    t,
    vecs_np,
    t_factors=(1.0, 2.0, 4.0, 8.0),
    eps_floor=1e-12,
    img_shape=(3, 32, 32),
    chunk_size=32,
    min_points=3,
    reject_sign_flips=True,
):
    """
    Robust normalized slope:
      slope of log(t * |q(t)|) vs log t,
    where q(t)=v^T H_t v may be negative.
    - Reject sign-flipping rows (optional)
    - Ignore points where |q| <= eps_floor instead of clamping
    Returns (B*D,) float32 with NaNs for invalid directions.
    """
    C, H, W = img_shape
    device = x.device
    dtype = x.dtype

    B = x.shape[0]
    _, D, _ = vecs_np.shape
    vecs_t = np.transpose(vecs_np, (0, 2, 1))  # (B,D,D)

    t_list = [float(t) * float(f) for f in t_factors]
    t_arr = np.array(t_list, dtype=np.float64)
    logt = np.log(t_arr + 1e-30)

    y_all = []
    for b in range(B):
        x0 = x[b:b+1].detach()
        v_imgs = torch.from_numpy(vecs_t[b].reshape(D, C, H, W)).to(device=device, dtype=dtype)

        # unit norm
        v_flat = v_imgs.reshape(D, -1)
        v_norm = torch.linalg.norm(v_flat, dim=1, keepdim=True).clamp_min(1e-12)
        v_imgs_unit = (v_flat / v_norm).reshape_as(v_imgs)

        Q = _collect_q_over_t(
            sde=sde, x0=x0, v_imgs_unit=v_imgs_unit,
            t_list=t_list, chunk_size=chunk_size
        )  # (D,T) raw signed

        # sign flip detection
        sign = np.sign(Q)
        # treat zeros as invalid (will be masked anyway)
        sign[sign == 0] = np.nan
        flip = np.any(sign[:, 1:] * sign[:, :-1] < 0, axis=1)  # (D,)

        # magnitude mask
        mag = np.abs(Q)
        valid = mag > float(eps_floor)  # (D,T)

        if reject_sign_flips:
            valid[flip, :] = False

        # log(t * |q|)
        TQ = mag * t_arr[None, :]
        logTQ = np.log(TQ + 1e-30)  # safe since invalid masked anyway

        slopes = _robust_rowwise_slope(logt, logTQ, valid, min_points=min_points)

        y_all.append(slopes.astype(np.float32))

    return np.concatenate(y_all, axis=0)


def measure_time_persistence_norm_slope(
    sde,
    x,
    t,
    vecs_np,
    t_factors=(1.0, 2.0, 4.0, 8.0),
    eps_floor=1e-12,
    img_shape=(3, 32, 32),
    chunk_size=32,
):
    """
    Stabilized slope: slope of log(t*q(t)) vs log t.
    If q(t) ~ 1/t (normal), then t*q(t) ~ const => slope ~ 0.
    This removes the universal diffusion scaling.
    """
    C, H, W = img_shape
    device = x.device
    dtype = x.dtype

    B = x.shape[0]
    _, D, _ = vecs_np.shape
    vecs_t = np.transpose(vecs_np, (0, 2, 1))  # (B,D,D)

    t_list = [float(t) * float(f) for f in t_factors]
    logt = np.log(np.array(t_list, dtype=np.float64) + 1e-30)

    y_all = []
    for b in range(B):
        x0 = x[b:b+1].detach()
        v_imgs = torch.from_numpy(vecs_t[b].reshape(D, C, H, W)).to(device=device, dtype=dtype)

        v_flat = v_imgs.reshape(D, -1)
        v_norm = torch.linalg.norm(v_flat, dim=1, keepdim=True).clamp_min(1e-12)
        v_imgs_unit = (v_flat / v_norm).reshape_as(v_imgs)

        Q = _collect_q_over_t(
            sde=sde, x0=x0, v_imgs_unit=v_imgs_unit,
            t_list=t_list, chunk_size=chunk_size
        )


        # stabilized quantity: t * q(t)
        tq = Q * np.array(t_list, dtype=np.float64)[None, :]
        logTQ = np.log(tq + 1e-30)

        lt = logt[None, :]
        lt_c = lt - lt.mean(axis=1, keepdims=True)
        lq_c = logTQ - logTQ.mean(axis=1, keepdims=True)
        denom = (lt_c * lt_c).sum(axis=1, keepdims=True) + 1e-30
        slope = (lt_c * lq_c).sum(axis=1) / denom.squeeze(1)  # (D,)

        y_all.append(slope.astype(np.float32))

    return np.concatenate(y_all, axis=0)


def measure_time_persistence_norm_var(
    sde,
    x,
    t,
    vecs_np,
    t_factors=(1.0, 2.0, 4.0, 8.0),
    eps_floor=1e-12,
    img_shape=(3, 32, 32),
    chunk_size=32,
    min_points=3,
    reject_sign_flips=True,
):
    """
    Robust stability:
      Var over log t of log(t*|q(t)|),
    computed only on valid points where |q|>eps_floor.
    Returns NaN for unstable/insufficient directions.
    """
    C, H, W = img_shape
    device = x.device
    dtype = x.dtype

    B = x.shape[0]
    _, D, _ = vecs_np.shape
    vecs_t = np.transpose(vecs_np, (0, 2, 1))

    t_list = [float(t) * float(f) for f in t_factors]
    t_arr = np.array(t_list, dtype=np.float64)

    y_all = []
    for b in range(B):
        x0 = x[b:b+1].detach()
        v_imgs = torch.from_numpy(vecs_t[b].reshape(D, C, H, W)).to(device=device, dtype=dtype)

        v_flat = v_imgs.reshape(D, -1)
        v_norm = torch.linalg.norm(v_flat, dim=1, keepdim=True).clamp_min(1e-12)
        v_imgs_unit = (v_flat / v_norm).reshape_as(v_imgs)

        Q = _collect_q_over_t(
            sde=sde, x0=x0, v_imgs_unit=v_imgs_unit,
            t_list=t_list, chunk_size=chunk_size
        )

        sign = np.sign(Q)
        sign[sign == 0] = np.nan
        flip = np.any(sign[:, 1:] * sign[:, :-1] < 0, axis=1)

        mag = np.abs(Q)
        valid = mag > float(eps_floor)

        if reject_sign_flips:
            valid[flip, :] = False

        TQ = mag * t_arr[None, :]
        logTQ = np.log(TQ + 1e-30)

        var = np.full((D,), np.nan, dtype=np.float64)
        for i in range(D):
            m = valid[i]
            if m.sum() < min_points:
                continue
            var[i] = np.var(logTQ[i, m])

        y_all.append(var.astype(np.float32))

    return np.concatenate(y_all, axis=0)


# ============================================================
# Measure registry + selector
# ============================================================
MEASURE_REGISTRY = {
    "smoothness": {
        "fn": lambda **kw: measure_spatial_smoothness(kw["vecs_np"], img_shape=kw.get("img_shape", (3,32,32))),
        "ylabel": "Spatial Smoothness",
        "cbar": "Smoothness",
    },
    "density_drop": {
        "fn": lambda **kw: measure_density_drop(
            kw["sde"], kw["x"], kw["t"], kw["vecs_np"],
            eps=kw.get("eps", 1e-3),
            img_shape=kw.get("img_shape", (3,32,32)),
            chunk_size=kw.get("chunk_size", 64),
        ),
        "ylabel": r"Density drop proxy  $-\Delta \log p_t$",
        "cbar": r"$-\Delta \log p_t$",
    },
    "score_growth": {
        "fn": lambda **kw: measure_score_growth(
            kw["sde"], kw["x"], kw["t"], kw["vecs_np"],
            eps=kw.get("eps", 1e-3),
            img_shape=kw.get("img_shape", (3,32,32)),
            chunk_size=kw.get("chunk_size", 64),
        ),
        "ylabel": r"Score growth  $\|s(x+\epsilon v)-s(x)\|$",
        "cbar": r"Score growth",
    },
    "time_persistence": {
        "fn": lambda **kw: measure_time_persistence_slope(
            kw["sde"], kw["x"], kw["t"], kw["vecs_np"],
            t_factors=kw.get("t_factors", (1.0,2.0,4.0,8.0)),
            img_shape=kw.get("img_shape", (3,32,32)),
            chunk_size=kw.get("chunk_size", 32),
        ),
        "ylabel": r"Time-persistence slope  $\frac{d\log(v^\top H_t v)}{d\log t}$",
        "cbar": r"Slope",
    },
}

MEASURE_REGISTRY.update({
    "time_persistence_norm_slope": {
        "fn": lambda **kw: measure_time_persistence_norm_slope(
            kw["sde"], kw["x"], kw["t"], kw["vecs_np"],
            t_factors=kw.get("t_factors", (1.0,2.0,4.0,8.0)),
            eps_floor=kw.get("eps_floor", 1e-12),
            img_shape=kw.get("img_shape", (3,32,32)),
            chunk_size=kw.get("chunk_size", 32),
        ),
        "ylabel": r"Normalized slope  $\frac{d\log(t\,v^\top H_t v)}{d\log t}$",
        "cbar": r"norm slope",
    },
    "time_persistence_norm_var": {
        "fn": lambda **kw: measure_time_persistence_norm_var(
            kw["sde"], kw["x"], kw["t"], kw["vecs_np"],
            t_factors=kw.get("t_factors", (1.0,2.0,4.0,8.0)),
            eps_floor=kw.get("eps_floor", 1e-12),
            img_shape=kw.get("img_shape", (3,32,32)),
            chunk_size=kw.get("chunk_size", 32),
        ),
        "ylabel": r"Stability  $\mathrm{Var}[\log(t\,v^\top H_t v)]$",
        "cbar": r"var",
    },
})



def compute_measure(
    measure_name: str,
    *,
    sde=None,
    x=None,
    t=None,
    vecs_np=None,
    img_shape=(3,32,32),
    **kwargs
):
    if measure_name not in MEASURE_REGISTRY:
        raise ValueError(f"Unknown measure_name={measure_name}. "
                         f"Choose from {list(MEASURE_REGISTRY.keys())}")
    fn = MEASURE_REGISTRY[measure_name]["fn"]
    return fn(sde=sde, x=x, t=t, vecs_np=vecs_np, img_shape=img_shape, **kwargs)


# ============================================================
# Plotting: create_scatter_plot / _plot_scatter_on_axis updated
# ============================================================
def create_scatter_plot(
    eigvals,
    yvals,
    sigma2_val,
    filename_prefix="analysis",
    solid_color="firebrick",
    y_label="Y",
    cbar_label="Y",
):
    """
    Generates and saves eigenvalue vs y scatter plot.
    If solid_color is None -> color by yvals with coolwarm.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    if solid_color is not None:
        c_val = solid_color
        cmap = None
        norm = None
        edgecolors = "white"
    else:
        cmap = plt.cm.get_cmap("coolwarm")
        norm = mcolors.Normalize(vmin=np.min(yvals), vmax=np.max(yvals))
        c_val = yvals
        edgecolors = "none"

    ax.scatter(
        eigvals,
        yvals,
        c=c_val,
        cmap=cmap,
        norm=norm,
        alpha=0.5,
        s=15,
        edgecolors=edgecolors,
        linewidth=0.1,
    )

    ax.set_xlabel(r"Eigenvalue $\lambda$", fontsize=22)
    ax.set_ylabel(y_label, fontsize=22)

    ax.axvline(x=1.0 / sigma2_val, color="dimgray", linestyle="--", linewidth=2.5, label=r"$1/\sigma^2$")

    if solid_color is None:
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label(cbar_label, fontsize=16)
        cbar.solids.set_alpha(1.0)

    plt.legend(fontsize=18, loc="best")
    save_path = f"{filename_prefix}_scatter.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Scatter plot saved to {save_path}")


# --- 1. Helper function for plotting logic (Updated) ---
def _plot_scatter_on_axis(
    ax, 
    eigvals, 
    smoothness, 
    sigma2_val, 
    title=None, 
    solid_color='firebrick', 
    font_scale=1.0,
    y_label="Y",
    x_scale='symlog',  # <--- 'linear' or 'symlog'
    linthresh=50.0     
):
    """
    Draws a scatter plot with SymLog scale support.
    """
    # Define colors
    if solid_color is not None:
        c_val = solid_color
        cmap = None
        norm = None
        edgecolors = 'white'
    else:
        # Gradient mode
        cmap = plt.cm.get_cmap('coolwarm')
        norm = mcolors.Normalize(vmin=np.min(smoothness), vmax=np.max(smoothness))
        c_val = smoothness
        edgecolors = 'none'

    # Scatter plot
    sc = ax.scatter(
        eigvals, 
        smoothness, 
        c=c_val, 
        cmap=cmap, 
        norm=norm, 
        alpha=0.5, 
        s=10 * font_scale, 
        edgecolors=edgecolors, 
        linewidth=0.1
    )
    
    # --- Axis Scaling ---
    if x_scale == 'symlog':
        # symlog: 対数軸だが、0付近の linthresh の範囲内は線形になる
        ax.set_xscale('symlog', linthresh=linthresh)
    
    # Adjust font sizes
    label_fs = 14 * font_scale
    tick_fs = 12 * font_scale
    
    ax.set_xlabel(r"Eigenvalue $\lambda$", fontsize=label_fs)
    ax.set_ylabel(y_label, fontsize=label_fs)
    if title:
        ax.set_title(title, fontsize=label_fs + 2)
    
    ax.tick_params(axis='both', which='major', labelsize=tick_fs)

    # Theoretical normal line
    ax.axvline(x=1.0/sigma2_val, color='dimgray', linestyle='--', linewidth=1.5, label=r'$1/\sigma^2$')
    
    # Return objects needed for colorbar (only if gradient mode)
    if solid_color is None:
        return sc, cmap, norm 
    return None, None



# ============================================================
# analyze_eigenvectors_visual updated (choose y-measure)
# ============================================================
def analyze_eigenvectors_visual(
    eigvals,
    vecs,
    yvals,
    sigma2_val,
    ref_image=None,
    num_visualize=5,
    filename_prefix="analysis",
    solid_color="firebrick",
    y_label="Y",
    cbar_label="Y",
):
    """
    1) scatter plot (eigvals vs y)
    2) show reference image (optional)
    3) visualize selected eigenvectors (same as before)
    """
    create_scatter_plot(
        eigvals=eigvals,
        yvals=yvals,
        sigma2_val=sigma2_val,
        filename_prefix=filename_prefix,
        solid_color=solid_color,
        y_label=y_label,
        cbar_label=cbar_label,
    )

    if ref_image is not None:
        img_vis = ref_image.detach().cpu().clone()
        if img_vis.dim() == 4:
            img_vis = img_vis[0]
        img_vis = img_vis.permute(1, 2, 0).numpy()
        img_vis = (img_vis + 1) / 2
        img_vis = np.clip(img_vis, 0, 1)

        fig = plt.figure(figsize=(3, 3.5))
        plt.imshow(img_vis)
        plt.title("Ref Image", fontsize=12)
        plt.axis("off")

        save_path = f"{filename_prefix}_reference.png"
        plt.savefig(save_path, bbox_inches="tight")
        plt.show()

    # eigenvector visualization logic unchanged
    C, H, W = 3, 32, 32
    if len(vecs.shape) == 3:
        target_vecs = vecs[0]
    else:
        target_vecs = vecs

    abs_sorted_indices = np.argsort(np.abs(eigvals))
    val_sorted_indices = np.argsort(eigvals)

    idx_tangent = abs_sorted_indices[:num_visualize]
    idx_normal = val_sorted_indices[-num_visualize:]
    mid_start = 100
    idx_boundary = abs_sorted_indices[mid_start: mid_start + num_visualize]

    print("Visualizing Tangent (Near Zero Lambda)...")
    visualize_selected_eigenvectors(
        indices=idx_tangent,
        eigvals=eigvals,
        vecs=target_vecs,
        title="Tangent Components (Near 0)",
        filename=f"{filename_prefix}_tangent",
        img_shape=(C, H, W),
    )

    print("Visualizing Boundary (Transition Region)...")
    visualize_selected_eigenvectors(
        indices=idx_boundary,
        eigvals=eigvals,
        vecs=target_vecs,
        title="Boundary Components",
        filename=f"{filename_prefix}_boundary",
        img_shape=(C, H, W),
    )

    print("Visualizing Normal (Large Positive Lambda)...")
    visualize_selected_eigenvectors(
        indices=idx_normal,
        eigvals=eigvals,
        vecs=target_vecs,
        title="Normal Components (Large λ)",
        filename=f"{filename_prefix}_normal",
        img_shape=(C, H, W),
    )


# ============================================================
# batch_analyze_scatter_plots updated (select y-axis measure)
# ============================================================

def batch_analyze_scatter_plots(
    sde,
    ddpm_pipe,
    x_batch,
    t_val,
    save_dir,
    grid_cols=4,
    solid_color="firebrick",
    y_measure="smoothness",
    measure_kwargs=None,
    img_shape=(3,32,32),
):
    """
    For all images in x_batch (B,C,H,W):
      1) compute Hessian eigvals + vecs
      2) compute chosen y-measure
      3) save individual scatter + target image
      4) save a grid of scatter plots
    """
    if measure_kwargs is None:
        measure_kwargs = {}

    # ---- NEW: normalize x_batch shape ----
    if isinstance(x_batch, (list, tuple)):
        x_batch = torch.stack(x_batch, dim=0)

    if not torch.is_tensor(x_batch):
        raise TypeError(f"x_batch must be a torch.Tensor, got {type(x_batch)}")

    if x_batch.dim() == 3:
        # single image (C,H,W) -> (1,C,H,W)
        x_batch = x_batch.unsqueeze(0)
    elif x_batch.dim() != 4:
        raise ValueError(f"x_batch must have shape (B,C,H,W) or (C,H,W), got {tuple(x_batch.shape)}")

    # Optional: sanity check channels
    if x_batch.shape[1] != img_shape[0]:
        raise ValueError(f"Expected C={img_shape[0]} but got x_batch.shape[1]={x_batch.shape[1]}")

    # ---- use shape[0] not len() ----
    batch_size = x_batch.shape[0]
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    batch_size = len(x_batch)
    grid_rows = math.ceil(batch_size / grid_cols)

    # noise parameters (DDPM style)
    idx = int(round(float(t_val) * 999))
    sigma2_val = float(1.0 - ddpm_pipe.scheduler.alphas_cumprod[idx].item())
    sigma_val = math.sqrt(sigma2_val)
    print(f"Batch Analysis: N={batch_size}, t={t_val}, 1/sigma^2={1.0/sigma2_val:.2f}, y={y_measure}")

    # labels
    y_label = MEASURE_REGISTRY[y_measure]["ylabel"]
    cbar_label = MEASURE_REGISTRY[y_measure]["cbar"]

    fig_grid, axes_grid = plt.subplots(
        grid_rows, grid_cols,
        figsize=(5 * grid_cols, 4 * grid_rows),
        constrained_layout=True
    )
    if batch_size == 1:
        axes_grid = np.array([axes_grid])
    axes_flat = axes_grid.flatten()

    for i in tqdm(range(batch_size), desc="Processing Batch"):
        x_single = x_batch[i:i+1]
        z_single = torch.randn_like(x_single)
        x_noisy = x_single + sigma_val * z_single

        # 1) eigendecomp
        eigvals, vecs_np = compute_hessian_eig_full(
            sde=sde,
            x=x_noisy,
            t=t_val,
            show_progress=False,
        )

        # 2) measure for y-axis
        yvals = compute_measure(
            y_measure,
            sde=sde,
            x=x_noisy,
            t=t_val,
            vecs_np=vecs_np,
            img_shape=img_shape,
            **measure_kwargs
        )

        # --- A) Save individual scatter
        fig_single, ax_single = plt.subplots(figsize=(8, 5))
        _plot_scatter_on_axis(
            ax_single, eigvals, yvals, sigma2_val,
            title=f"Sample {i} (t={t_val})",
            solid_color=solid_color,
            y_label=y_label,
        )
        ax_single.legend(fontsize=12)

        scatter_path = os.path.join(save_dir, f"scatter_{y_measure}_{i:03d}.png")
        fig_single.savefig(scatter_path, dpi=100, bbox_inches="tight")
        plt.close(fig_single)

        # --- B) Save target image
        img_vis = x_noisy.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
        img_vis = (img_vis + 1) / 2.0
        img_vis = np.clip(img_vis, 0, 1)
        image_path = os.path.join(save_dir, f"image_{i:03d}.png")
        plt.imsave(image_path, img_vis)

        # --- C) Plot onto grid
        ax_grid = axes_flat[i]
        _plot_scatter_on_axis(
            ax_grid, eigvals, yvals, sigma2_val,
            title=f"ID: {i}",
            solid_color=solid_color,
            font_scale=0.8,
            y_label=y_label,
        )

    for j in range(batch_size, len(axes_flat)):
        axes_flat[j].axis("off")

    grid_path = os.path.join(save_dir, f"scatter_grid_{y_measure}_{batch_size}samples_t{t_val}.png")
    fig_grid.savefig(grid_path, dpi=150, bbox_inches="tight")
    plt.close(fig_grid)
    print(f"Done! Saved to: {save_dir}")


# ============================================================
# visualize_selected_eigenvectors unchanged (kept verbatim)
# ============================================================
def visualize_selected_eigenvectors(
    indices,
    eigvals,
    vecs,
    title=None,
    filename="eigenvectors_plot",
    img_shape=(3, 32, 32),
):
    C, H, W = img_shape
    num_plots = len(indices)
    fig, axes = plt.subplots(1, num_plots, figsize=(3 * num_plots, 3.5))
    if num_plots == 1:
        axes = [axes]

    for i, idx in enumerate(indices):
        lam = eigvals[idx]
        if len(vecs.shape) == 3:
            v = vecs[0, :, idx]
        else:
            v = vecs[:, idx]

        v_img = v.reshape(C, H, W)
        max_val = np.abs(v_img).max() + 1e-12
        v_img = v_img / max_val
        v_img_vis = (v_img + 1) / 2
        v_img_vis = np.transpose(v_img_vis, (1, 2, 0))

        ax = axes[i]
        ax.imshow(v_img_vis)
        ax.set_title(f"$\\lambda={lam:.1f}$", fontsize=12)
        ax.axis("off")

    if title:
        plt.suptitle(title, fontsize=16)

    if not filename.endswith(".png"):
        filename += ".png"

    plt.savefig(filename, bbox_inches="tight")
    plt.show()
    print(f"Eigenvectors plot saved to {filename}")


# ============================================================
# Example usage
# ============================================================
# loader_iter = iter(loader)
# x_batch_16, _ = next(loader_iter)
# if x_batch_16.shape[0] < 16:
#     x_next, _ = next(loader_iter)
#     x_batch_16 = torch.cat([x_batch_16, x_next], dim=0)

# batch_analyze_scatter_plots(
#     sde=sde,
#     ddpm_pipe=ddpm,
#     x_batch=x_batch_16,
#     t_val=0.0001,
#     save_dir=os.path.join(PATH_FIG, "scatter", "CIFAR"),
#     grid_cols=4,
#     solid_color="darkred",
#     y_measure="density_drop",   # "smoothness" | "density_drop" | "score_growth" | "time_persistence"
#     measure_kwargs={"eps": 1e-3, "chunk_size": 64, "t_factors": (1.0,2.0,4.0,8.0)},
#     img_shape=(3,32,32),
# )

################### 260131 end

def compute_hessian_eigenvalues(sde, x, t, show_progress=True):
    """
    Computes exact eigenvalues of the Hessian of -log p_t(x).
    Compatible with both images (B, C, H, W) and flat vectors (B, D).
    
    Definition: H = - nabla^2 log p_t(x) approx - nabla s(x, t)
    
    Args:
        sde: Score-based model (must accept input of x's shape and return same shape)
        x: Input tensor of shape (Batch, ...)
        t: Time/Noise level
    """
    # Enable gradient tracking for input data
    # Keep original shape (B, C, H, W) or (B, D)
    x = x.detach().clone().requires_grad_(True)
    
    # Get batch size and total dimension D
    B = x.shape[0]
    D = x.numel() // B  # (C*H*W) or (D)
    
    # Process time t
    if isinstance(t, (int, float)):
        t_batch = torch.full((B,), t, device=x.device, dtype=x.dtype)
    else:
        t_batch = t.to(x.device)
        
    # Compute Score: Output shape matches input (B, ...)
    score = sde.score(x, t_batch)
    
    # Flatten score to (B, D) for computation
    score_flat = score.reshape(B, -1)

    hessian_rows = []
    
    # Loop over D dimensions (Takes time for images, e.g., D=3072)
    iterator = range(D)
    if show_progress and D > 100: 
        iterator = tqdm(iterator, desc=f"Computing Hessian ({D} dims)")

    for i in iterator:
        # Compute gradient of the i-th component of the score
        # autograd.grad returns shape (B, C, H, W) matching input x
        grad_i = autograd.grad(
            outputs=score_flat[:, i].sum(), 
            inputs=x, 
            create_graph=False, 
            retain_graph=True
        )[0]
        
        # Flatten gradient to (B, D) and append to list
        hessian_rows.append(grad_i.reshape(B, -1).unsqueeze(1)) # Shape: (B, 1, D)
        
    # Concatenate to create Jacobian matrix: (B, D, D)
    jacobian_batch = torch.cat(hessian_rows, dim=1)
    
    # Symmetrize (Jacobian of gradient is symmetric)
    jacobian_batch = (jacobian_batch + jacobian_batch.transpose(1, 2)) / 2
    
    # Compute eigenvalues (Eigenvalues of Jacobian)
    eigvals_jacobian = torch.linalg.eigvalsh(jacobian_batch)
    
    # Set H = -J to match the paper (Flip sign to make Normal direction positive)
    # Tangent approx 0, Normal approx +1/sigma^2
    eigvals_hessian = -1 * eigvals_jacobian
    
    return eigvals_hessian.detach().cpu().numpy().flatten()

import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
from matplotlib.lines import Line2D

import matplotlib.cm as cm
import matplotlib.colors as mcolors

def plot_eigenvalue_spectrum_colored(
    sde, x_batch, t_val, z_batch, sigma2_val,
    p_c_pairs=[], 
    filename="hessian_spectrum_colored"
):
    """
    Plots the spectrum colored by overlap with the noise vector.
    """
    batch_size = x_batch.shape[0]
    
    print("Computing eigenvalues and overlaps...")
    # Pass z_batch to the new function
    eigvals, overlaps = compute_hessian_eig_and_overlap(sde, x_batch, t_val, z_batch)
    print("Overlap Stats:")
    print(f"  Max:  {overlaps.max():.4f} (Should be close to 1.0)")
    print(f"  Mean: {overlaps.mean():.4f}")
    print(f"  Min:  {overlaps.min():.4f}")

    # --- Setup Bins ---
    q01, q99 = np.percentile(eigvals, [1, 99])
    spread = q99 - q01
    x_range = (q01 - spread * 0.1, q99 + spread * 0.1)
    
    # Create bins
    bins = np.linspace(x_range[0], x_range[1], 40)
    
    # Digitize: Find which bin each eigenvalue belongs to
    bin_indices = np.digitize(eigvals, bins)
    
    # Calculate frequency and mean overlap per bin
    bin_counts = []
    bin_overlaps = []
    bin_centers = []
    
    for i in range(1, len(bins)):
        # Indices of eigvals in this bin
        mask = bin_indices == i
        count = np.sum(mask)
        if count > 0:
            avg_overlap = np.mean(overlaps[mask])
            bin_counts.append(count / batch_size) # Normalize by batch
            bin_overlaps.append(avg_overlap)
        else:
            bin_counts.append(0)
            bin_overlaps.append(0) # Color won't matter for height 0
        
        bin_centers.append((bins[i-1] + bins[i]) / 2)

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.set_theme(style="darkgrid")
    
    # Colormap: Viridis or Coolwarm (Blue=Low Overlap, Red=High Overlap)
    cmap = plt.cm.get_cmap('RdYlBu_r') # Red for High, Blue for Low
    norm = mcolors.Normalize(vmin=0, vmax=1.0)
    
    # Draw bars manually
    ax.bar(
        bin_centers, 
        bin_counts, 
        width=(bins[1] - bins[0]), 
        color=cmap(norm(bin_overlaps)),
        edgecolor='white',
        linewidth=0.5,
        alpha=0.9
    )

    ax.set_xlabel(r"Eigenvalue $\lambda$", fontsize=18)
    ax.set_ylabel("Frequency", fontsize=18)
    ax.set_title(f"Spectrum Colored by Noise Overlap (t={t_val})", fontsize=20)
    
    # Add Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label(r"Mean Overlap $(v \cdot z)^2$", fontsize=16)
    
    # Add theoretical vertical line
    target_lambda = 1.0 / sigma2_val
    ax.axvline(x=target_lambda, color='green', linestyle='--', linewidth=2, label=r'$1/\sigma^2$')
    ax.legend(fontsize=14)

    # Save
    if filename:
        save_path = f"{filename}.png" if not filename.endswith(".png") else filename
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
        
    plt.show()
    return eigvals

def plot_eigenvalue_spectrum_with_filter(
    sde,
    x_batch,
    t_val,
    sigma2_val,
    p_c_pairs,  # List of [c, p] e.g., [[0.1, 4], [0.05, 2], [0.2, 8]]
    title=None,
    x_range=None,
    filename="hessian_spectrum_plot",
    show_filters=True  # <--- New switch: Default is OFF
):
    """
    Plots the eigenvalue spectrum with filters.
    Legend is placed on the RIGHT side, vertically stacked.
    Colors for filters can be manually adjusted inside the function.
    """
    cm = plt.cm.coolwarm

    # Color setup
    line_colors = ['rosybrown', 'maroon', cm(0.2)]
    line_colors = [ cm(0.25), 'royalblue', 'darkblue']
    line_colors = [ 'darksalmon', 'rosybrown', cm(0.25), 'royalblue', 'darkblue', 'darkslateblue']

    if len(line_colors) < len(p_c_pairs):
        line_colors = sns.color_palette("coolwarm", len(p_c_pairs))

    # 1. Setup Style
    # Using darkgrid as the base style.
    sns.set_theme(style="darkgrid")
    bg_color = sns.axes_style()['axes.facecolor']

    # Get dimension info
    batch_size = x_batch.shape[0]
    dim = x_batch.numel() // batch_size

    if title is None:
        title = "Hessian Spectrum & LHSD Filter"

    print(f"Computing eigenvalues for {batch_size} samples (Dimension={dim})...")

    # Compute eigenvalues
    eigvals = compute_hessian_eigenvalues(sde, x_batch, t_val)

    # --- Determine Plot Range ---
    # We calculate the range based on data percentiles first.
    if x_range is None:
        q01, q99 = np.percentile(eigvals, [1, 99])
        spread = q99 - q01
        
        lower = q01 - spread * 0.1
        upper = q99 + spread * 0.1
        
        # Only extend range for filters/kappa if filters are actually shown
        if show_filters:
            max_c = max([pair[0] for pair in p_c_pairs])
            max_kappa = max_c / sigma2_val
            
            lower = min(lower, -max_kappa * 0.5)
            upper = max(upper, max_kappa * 1.5)
            
        x_range = (lower, upper)
        # x_range = (lower, upper*0.2) # for investigating 3072D_900d_nonliner

    # --- Plotting Construction ---
    fig, ax1 = plt.subplots(figsize=(10, 5))

    # --- 1. Background / Secondary Axis Handling ---
    if show_filters:
        # Create secondary axes for the filter lines
        ax_bg = ax1.twinx()   # Layer 1: Background & Grid
        ax_line = ax1.twinx() # Layer 3: Lines

        # Layer Management
        ax_bg.set_zorder(1)
        ax_bg.set_facecolor(bg_color)
        ax_bg.patch.set_visible(True)

        # Make the main histogram axis transparent so the background shows through
        ax1.set_zorder(2)
        ax1.patch.set_visible(False)
        ax1.grid(False)

        ax_line.set_zorder(3)
        ax_line.patch.set_visible(False)
        ax_line.grid(False)
        ax_line.axis("off")

        # Setup right Y-axis for Filter Response
        y_limit_filter = (-0.05, 1.25)
        ax_bg.set_ylim(y_limit_filter)

        # White Grid for the secondary axis
        for y in [0.0, 0.5, 1.0]:
            ax_bg.axhline(y, color='white', linewidth=1.5)

        ax_bg.set_yticks([0.0, 0.5, 1.0])
        ax_bg.tick_params(axis='y', labelsize=22, colors='black')
        ax_bg.set_ylabel(r"Filter Response $f(\lambda)$", fontsize=24, color='black')
    
    else:
        # Standard plot mode: Just one axis
        # sns.set_theme(style="darkgrid") already sets the gray background on ax1
        ax1.grid(True, color='white', linewidth=1.5) # Ensure grid is white like the original
        ax_line = None # No secondary line axis

    # --- 2. Histogram (Main Axis) ---
    ax1.xaxis.offsetText.set_fontsize(24)
    ax1.tick_params(axis='x', labelsize=22)
    ax1.tick_params(axis='y', labelsize=22)
    ax1.set_xlabel(r"Eigenvalue $\lambda$", fontsize=24)
    ax1.set_ylabel("Frequency", fontsize=24, color='black')

    weights = np.ones_like(eigvals) / batch_size
    sns.histplot(
        x=eigvals,
        weights=weights,
        stat="count",
        bins=30,
        binrange=x_range,
        # color='darkblue',
        color=cm(1.0),
        alpha=1.0,
        label='Spectrum', # Label for Legend
        ax=ax1,
        element="bars",
        edgecolor=None
    )
    ax1.set_xlim(x_range)

    # --- 3. Filter Lines (Conditional) ---
    
    # Collect handles for legend
    all_handles = []
    all_labels = []
    
    # Get histogram handle
    h1, l1 = ax1.get_legend_handles_labels()
    all_handles.extend(h1)
    all_labels.extend(l1)

    if show_filters and ax_line is not None:
        ax_line.set_ylim(y_limit_filter)
        lam_grid = np.linspace(x_range[0], x_range[1], 1000)

        # Plot filters
        for i, (c_val, p_val) in enumerate(p_c_pairs):
            kappa_val = c_val / sigma2_val
            f_vals = 1.0 / (1.0 + (np.maximum(lam_grid, 0) / (kappa_val + 1e-20))**p_val)
            #f_vals = 1.0 / (1.0 + (np.abs(lam_grid) / (kappa_val + 1e-20))**p_val)

            # Use color from manual list
            current_color = line_colors[i % len(line_colors)]

            ax_line.plot(
                lam_grid,
                f_vals,
                color=current_color,
                linewidth=4.0,
                linestyle='-',
                label=rf'$c={c_val}, p={p_val}$' # Label for Legend
            )

            # Plot Cutoff Line
            ax_line.axvline(x=kappa_val, color='dimgray', linestyle='--', linewidth=2.5, alpha=0.7)

        # Get Filter handles (from ax_line)
        h2, l2 = ax_line.get_legend_handles_labels()
        all_handles.extend(h2)
        all_labels.extend(l2)

        # Create Cutoff proxy handle
        cutoff_handle = Line2D([0], [0], color='dimgray', linestyle='--', linewidth=2.5)
        all_handles.append(cutoff_handle)
        all_labels.append(r'Cutoff $\kappa$')

        # Unified Legend on the Right (Only when filters are shown)
        ax_line.legend(
            all_handles,
            all_labels,
            loc='upper left',
            bbox_to_anchor=(1.18, 1),
            fontsize=24,
            frameon=True
        )
        # Adjust layout to accommodate the external legend
        plt.subplots_adjust(right=0.8)

    else:
        # If filters are off, just put a simple legend for the spectrum (optional)
        # or no legend if preferred. Here we add the spectrum legend inside.
        ax1.legend(
            all_handles,
            all_labels,
            loc='best',
            fontsize=20,
            frameon=True
        )
        # No special subplots_adjust needed

    # --- Saving and Showing ---
    if filename:
        directory = os.path.dirname(filename)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        save_path = f"{filename}.png" if not filename.endswith(".png") else filename
        # bbox_inches='tight' is crucial for external legends
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"... saved {save_path}")

    plt.show()

    return eigvals

def measure_inference_time(
    estimator, 
    x: torch.Tensor, 
    t: float, 
    n_warmup: int = 2, 
    n_repeat: int = 10,
    **estimator_kwargs 
) -> float:
    """
    Measures the accurate inference time of the LID estimator.
    
    Args:
        estimator: The LID estimator instance.
        x: Input data tensor.
        t: Diffusion time or noise level (float).
        n_warmup: Number of warm-up iterations (default: 2).
        n_repeat: Number of repetitions for averaging (default: 10).
        **estimator_kwargs: Additional keyword arguments to pass to estimator.estimate_lid()
                            (e.g., hutchinson_sample_count for FLIPD).
        
    Returns:
        float: Average execution time in milliseconds.
    """
    
    # 1. Warm-up
    for _ in range(n_warmup):
        _ = estimator.estimate_lid(
            x=x, 
            t=t, 
            **estimator_kwargs 
        )
    
    # 2. Measurement
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    
    for _ in range(n_repeat):
        _ = estimator.estimate_lid(
            x=x, 
            t=t, 
            **estimator_kwargs #
        )
        
    end_event.record()
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    elapsed_time_ms = start_event.elapsed_time(end_event) / n_repeat
    
    return elapsed_time_ms


def compute_png_complexity(imgs: torch.Tensor) -> np.ndarray:
    """
    Complexity measure based on PNG compressed size.

    Args:
        imgs: [N, C, H, W] tensor in [0,1]

    Returns:
        complexity_bits: np.ndarray [N], PNG size in bits
    """
    imgs = imgs.detach().cpu()
    N = imgs.size(0)
    complexity_bits = np.empty(N, dtype=np.int32)

    for i in tqdm(range(N), desc="Computing PNG complexity"):
        img = imgs[i].clamp(0, 1)                   # [C,H,W]
        # to uint8 H×W×C
        img_np = (img.permute(1, 2, 0).numpy() * 255).astype("uint8")
        pil = Image.fromarray(img_np)

        buf = io.BytesIO()
        # `optimize=True` to make size more meaningful
        pil.save(buf, format="PNG", optimize=True)
        size_bytes = len(buf.getvalue())
        complexity_bits[i] = size_bytes * 8         # bits (scale doesn't matter, but you asked for bit-length)

    return complexity_bits




# --------------------------
# Filter / diagnostics
# --------------------------
def pick_kappa_snr(sigma2: float, c: float, eps: float = 1e-12) -> float:
    """
    Simple SNR-inspired cutoff:
        kappa(t) = c / sigma(t)^2
    where sigma2 = sigma(t)^2.
    """
    return float(c) / float(sigma2 + eps)


def hill_filter_np(lam: np.ndarray, kappa: float, p: int = 4) -> np.ndarray:
    """
    Hill filter used in LHSD:
        f(lam) = 1 / (1 + (max(lam,0)/kappa)^p)
    """
    lam_pos = np.maximum(lam, 0.0)
    k = float(kappa) + 1e-20
    return 1.0 / (1.0 + (lam_pos / k) ** int(p))


def transition_mass_np(eigvals: np.ndarray, kappa: float, eps_rel: float = 0.2) -> float:
    """
    Fraction of eigenvalues falling into a relative transition band around kappa:
        band = [kappa*(1-eps_rel), kappa*(1+eps_rel)]
    """
    k = float(kappa)
    lo, hi = k * (1.0 - eps_rel), k * (1.0 + eps_rel)
    return float(np.mean((eigvals >= lo) & (eigvals <= hi)))


def filter_variance_proxy_np(eigvals: np.ndarray, kappa: float, p: int = 4) -> float:
    """
    Variance proxy of filter responses over the spectrum:
        Var[f(lam)] = E[f^2] - (E[f])^2
    (This is a convenient scalar indicator of how much spectral mass sits in the transition region.)
    """
    f = hill_filter_np(eigvals, kappa=kappa, p=p)
    return float(np.mean(f ** 2) - (np.mean(f) ** 2))



def plot_transition_mass_over_t(
    sde,
    schedule,
    x_batch: torch.Tensor,
    t_list,
    p_c_pairs,
    eps_rel: float = 0.2,
    seed: int = 0,
    title: str | None = None,
    filename: str | None = None,
    show_progress: bool = True,
):
    """
    Plot Pr(lambda in [kappa_-, kappa_+]) vs t for each (c,p),
    where kappa(t)=c/sigma(t)^2 and [kappa_-,kappa_+]=[kappa(1-eps),kappa(1+eps)].
    """
    sns.set_theme(style="darkgrid")
    bg_color = sns.axes_style()['axes.facecolor']
    cm = plt.cm.coolwarm

    #line_colors = ['darksalmon', 'rosybrown', cm(0.25), 'royalblue', 'darkblue', 'darkslateblue']
    line_colors = ["lightsalmon","firebrick", cm(1.0), 'royalblue', 'darkblue', 'darkslateblue']
    if len(line_colors) < len(p_c_pairs):
        line_colors = sns.color_palette("coolwarm", len(p_c_pairs))

    if isinstance(t_list, (list, tuple)):
        t_vals = [float(t) for t in t_list]
    else:
        t_vals = [float(t) for t in list(t_list)]

    if title is None:
        title = r"Transition mass vs noise scale $t$"

    B = x_batch.shape[0]
    D = x_batch.numel() // B

    g = torch.Generator(device=x_batch.device)
    g.manual_seed(int(seed))

    res = {}
    for (c_val, p_val) in p_c_pairs:
        key = (float(c_val), int(p_val))
        res[key] = {"t": [], "prob": [], "kappa": []}

    iterator = tqdm(t_vals, desc=f"Sweep t (Transition mass; B={B}, D={D})") if show_progress else t_vals

    for t in iterator:
        sigma2 = float(schedule.sigma2(float(t)))
        sigma = float(np.sqrt(sigma2))

        #z = torch.randn_like(x_batch, generator=g)
        z = torch.randn(
            x_batch.shape,
            device=x_batch.device,
            dtype=x_batch.dtype,
            generator=g
        )
        x_noisy = x_batch + sigma * z

        eigvals = compute_hessian_eigenvalues(sde, x_noisy, t, show_progress=False)

        for (c_val, p_val) in p_c_pairs:
            c_val = float(c_val)
            p_val = int(p_val)
            kappa = pick_kappa_snr(sigma2=sigma2, c=c_val)

            prob = transition_mass_np(eigvals, kappa=kappa, eps_rel=eps_rel)  # already a probability

            key = (c_val, p_val)
            res[key]["t"].append(float(t))
            res[key]["prob"].append(float(prob))
            res[key]["kappa"].append(float(kappa))

    # ---- plot ----
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_facecolor(bg_color)
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=18)
    ax.set_xlabel(r"Noise scale $t$", fontsize=22)
    #ax.set_ylabel(r"$\Pr(\lambda \in [\kappa_-, \kappa_+])$", fontsize=22)
    ax.set_ylabel(r"$M(t)$", fontsize=22)

    #ax.set_title(title, fontsize=22)

    handles, labels = [], []
    for i, (c_val, p_val) in enumerate(p_c_pairs):
        key = (float(c_val), int(p_val))
        color = line_colors[i % len(line_colors)]
        t = np.array(res[key]["t"])
        y = np.array(res[key]["prob"])
        h, = ax.plot(t, y, linewidth=3.5, color=color)
        handles.append(h)
        labels.append(rf"$c={float(c_val)},\,p={int(p_val)}$")

    # ax.legend(
    #     handles, labels,
    #     loc='upper left',
    #     bbox_to_anchor=(1.02, 1.0),
    #     fontsize=18,
    #     frameon=True
    # )
    #plt.subplots_adjust(right=0.78)

    # leg = ax.legend(
    #     handles, labels,
    #     loc="lower center",
    #     bbox_to_anchor=(0.5, 1.02),
    #     ncol=min(len(p_c_pairs), 4),
    #     frameon=True,
    #     fancybox=True,
    #     framealpha=0.95,
    #     fontsize=18
    # )
    leg = ax.legend(
        handles, labels,
        frameon=True,
        fancybox=True,
        framealpha=0.95,
        fontsize=18
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.82)

    # ax.legend(
    #     bbox_to_anchor=(0.5, 1.02),
    #     loc='lower center',
    #     borderaxespad=0,
    #     ncol=len(p_c_pairs),
    #     fontsize=18,
    #     frameon=True
    # )

    if filename:
        directory = os.path.dirname(filename)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        save_path = f"{filename}.png" if not filename.endswith(".png") else filename
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"... saved {save_path}")

    plt.show()
    return res


def plot_filter_variance_proxy_over_t(
    sde,
    schedule,
    x_batch: torch.Tensor,
    t_list,
    p_c_pairs,
    seed: int = 0,
    title: str | None = None,
    filename: str | None = None,
    show_progress: bool = True,
):
    """
    Plot Var[f(lambda)] vs t for each (c,p),
    where f is the Hill filter with that (kappa(t), p).
    """
    sns.set_theme(style="darkgrid")
    bg_color = sns.axes_style()['axes.facecolor']
    cm = plt.cm.coolwarm

    line_colors = ['darksalmon', 'rosybrown', cm(0.25), 'royalblue', 'darkblue', 'darkslateblue']
    if len(line_colors) < len(p_c_pairs):
        line_colors = sns.color_palette("coolwarm", len(p_c_pairs))

    if isinstance(t_list, (list, tuple)):
        t_vals = [float(t) for t in t_list]
    else:
        t_vals = [float(t) for t in list(t_list)]

    if title is None:
        title = r"Filter variance proxy vs noise scale $t$"

    B = x_batch.shape[0]
    D = x_batch.numel() // B

    g = torch.Generator(device=x_batch.device)
    g.manual_seed(int(seed))

    res = {}
    for (c_val, p_val) in p_c_pairs:
        key = (float(c_val), int(p_val))
        res[key] = {"t": [], "vproxy": [], "kappa": []}

    iterator = tqdm(t_vals, desc=f"Sweep t (Var proxy; B={B}, D={D})") if show_progress else t_vals

    for t in iterator:
        sigma2 = float(schedule.sigma2(float(t)))
        sigma = float(np.sqrt(sigma2))

        #z = torch.randn_like(x_batch, generator=g)
        z = torch.randn(
            x_batch.shape,
            device=x_batch.device,
            dtype=x_batch.dtype,
            generator=g
        )
        x_noisy = x_batch + sigma * z

        eigvals = compute_hessian_eigenvalues(sde, x_noisy, t, show_progress=False)

        for (c_val, p_val) in p_c_pairs:
            c_val = float(c_val)
            p_val = int(p_val)
            kappa = pick_kappa_snr(sigma2=sigma2, c=c_val)

            vproxy = filter_variance_proxy_np(eigvals, kappa=kappa, p=p_val)

            key = (c_val, p_val)
            res[key]["t"].append(float(t))
            res[key]["vproxy"].append(float(vproxy))
            res[key]["kappa"].append(float(kappa))

    # ---- plot ----
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_facecolor(bg_color)
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    ax.set_xlabel(r"Noise scale $t$", fontsize=22)
    ax.set_ylabel(r"$\mathrm{Var}[f(\lambda)]$", fontsize=22)
    ax.set_title(title, fontsize=22)

    handles, labels = [], []
    for i, (c_val, p_val) in enumerate(p_c_pairs):
        key = (float(c_val), int(p_val))
        color = line_colors[i % len(line_colors)]
        t = np.array(res[key]["t"])
        y = np.array(res[key]["vproxy"])
        h, = ax.plot(t, y, linewidth=3.5, color=color)
        handles.append(h)
        labels.append(rf"$c={float(c_val)},\,p={int(p_val)}$")

    ax.legend(
        handles, labels,
        loc='upper left',
        bbox_to_anchor=(1.02, 1.0),
        fontsize=18,
        frameon=True
    )
    plt.subplots_adjust(right=0.78)

    if filename:
        directory = os.path.dirname(filename)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        save_path = f"{filename}.png" if not filename.endswith(".png") else filename
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"... saved {save_path}")

    plt.show()
    return res
