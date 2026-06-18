# Reproduce https://openreview.net/pdf?id=ZEf03Uunvk
# ------------------------------------------------------------
# IDR: Embed any low-D LIDDistribution into image space (28x28),
# then invert images back to base coordinates for visualization.
# ------------------------------------------------------------

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Tuple, Optional

import torch
import torch.nn.functional as F

# ---- Optional: PCA on FMNIST to get (mu, U) image basis ----
try:
    from torchvision import datasets, transforms
    _HAS_TORCHVISION = True
except Exception:
    _HAS_TORCHVISION = False

# Your repo base:
from .lid_base import LIDDistribution


# =========================
# 1) PCA helper (one-time)
# =========================
# run fmnist_pca

# =========================
# 2) Save/Load parameters
# =========================
@dataclass
class IDRParams:
    mu: torch.Tensor          # (784,)
    U: torch.Tensor           # (784, K)
    W: Optional[torch.Tensor] = None   # (rff_dim, d)
    b: Optional[torch.Tensor] = None   # (rff_dim,)
    sigma: float = 5.0
    rff_dim: int = 15
    bias_terms: int = 2
    out_shape: Tuple[int, int, int] = (1, 28, 28)


def save_idr_params(path: str, params: IDRParams) -> None:
    payload = {
        "mu": params.mu.cpu(),
        "U": params.U.cpu(),
        "W": None if params.W is None else params.W.cpu(),
        "b": None if params.b is None else params.b.cpu(),
        "sigma": params.sigma,
        "rff_dim": params.rff_dim,
        "bias_terms": params.bias_terms,
        "out_shape": params.out_shape,
    }
    torch.save(payload, path)


def load_idr_params(path: str, map_location: Optional[str | torch.device] = None) -> IDRParams:
    p = torch.load(path, map_location=map_location)
    return IDRParams(
        mu=p["mu"], U=p["U"],
        W=p.get("W", None), b=p.get("b", None),
        sigma=float(p.get("sigma", 5.0)),
        rff_dim=int(p.get("rff_dim", 15)),
        bias_terms=int(p.get("bias_terms", 2)),
        out_shape=tuple(p.get("out_shape", (1, 28, 28))),
    )


# =========================
# 3) IDR image wrapper
# =========================
class IDRImageWrapper(LIDDistribution):
    """
    Wrap a base LIDDistribution (low-D) and map to image space (C,H,W), default 1x28x28.

    Forward map:
        x_low ∈ R^{N×d} --φ--> φ(x) ∈ R^{N×K} --decode--> img = μ + φ(x) @ U^T
    where φ uses random Fourier features (RFF) + simple bias terms, providing a smooth,
    continuous, IDR-like embedding into image domain.

    Inversion:
        Given img, estimate φ̂ = U^T (img - μ) and solve for x by minimizing:
            L(x) = ||sin(Wx + b) - φ̂_sin||^2 + ||cos(Wx + b) - φ̂_cos||^2
        with Adam or L-BFGS. Returns base coords for visualization (e.g., Funnel in 3D).
    """

    def __init__(
        self,
        base_dist: LIDDistribution,
        mu: torch.Tensor,        # (784,)
        U: torch.Tensor,         # (784, K)
        rff_dim: Optional[int] = None,
        sigma: float = 5.0,
        bias_terms: int = 2,
        clamp01: bool = True,
        out_shape: Tuple[int, int, int] = (1, 28, 28),
        seed: Optional[int] = 1234,
    ):
        assert mu.ndim == 1 and mu.numel() == out_shape[1] * out_shape[2] * out_shape[0], \
            f"mu must match out_shape flattened, got {mu.shape} vs {out_shape}"
        assert U.ndim == 2 and U.shape[0] == mu.numel(), "U must be (flattened_pixels, K)"

        self.base = base_dist
        self.mu = mu.float()
        self.U = U.float()
        self.out_shape = out_shape
        self.clamp01 = clamp01

        self.K = U.shape[1]
        if rff_dim is None:
            rff_dim = max((self.K - bias_terms) // 2, 0)
        self.rff_dim = int(rff_dim)
        self.bias_terms = int(bias_terms)
        assert 2 * self.rff_dim + self.bias_terms <= self.K, \
            "U has too few columns for requested features."

        self._sigma = float(sigma)
        self._rng = torch.Generator()
        if seed is not None:
            self._rng.manual_seed(int(seed))
        self._W: Optional[torch.Tensor] = None  # (rff_dim, d)
        self._b: Optional[torch.Tensor] = None  # (rff_dim,)

    # ---------- RFF utils ----------
    def _init_rff(self, d: int, device: torch.device, dtype: torch.dtype) -> None:
        if self._W is None or self._W.shape[1] != d:
            self._W = torch.randn(self.rff_dim, d, generator=self._rng, device=device, dtype=dtype) / self._sigma
            self._b = torch.rand(self.rff_dim, generator=self._rng, device=device, dtype=dtype) * (2.0 * math.pi)

    def _phi(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N, d) -> φ: (N, K) with [sin, cos, bias...]
        """
        N, d = x.shape
        self._init_rff(d, device=x.device, dtype=x.dtype)

        proj = F.linear(x, self._W)  # (N, rff_dim)
        proj = proj + self._b  # broadcast

        feats = [torch.sin(proj), torch.cos(proj)]  # each (N, rff_dim)

        bias = []
        if self.bias_terms >= 1:
            bias.append(torch.ones(N, 1, device=x.device, dtype=x.dtype))
        if self.bias_terms >= 2:
            bias.append(x.pow(2).mean(dim=1, keepdim=True))  # simple low-freq anchor

        phi = torch.cat(feats + bias, dim=1)  # (N, 2*rff_dim + bias_terms)

        # fit to K (pad or crop)
        if phi.shape[1] < self.K:
            pad = torch.zeros(N, self.K - phi.shape[1], device=x.device, dtype=x.dtype)
            phi = torch.cat([phi, pad], dim=1)
        elif phi.shape[1] > self.K:
            phi = phi[:, : self.K]
        return phi

    def encode(self, x_low: torch.Tensor, return_dict: bool = False):
        """
        Map provided base points x_low (N,d) into image space (N,1,28,28).
        This mirrors the forward used by .sample(...) but bypasses base_dist.sample.
        """
        assert x_low.ndim == 2, f"encode expects (N,d), got {x_low.shape}"
        x_low = x_low.float()
        # ensure RFF matrices exist
        if self._W is None:
            # initialize using x's dimensionality
            self._init_rff(d=x_low.shape[1], device=x_low.device, dtype=x_low.dtype)

        phi = self._phi(x_low)                       # (N, K)
        img_flat = self.mu.to(x_low) + phi @ self.U.to(x_low).T  # (N, 784)
        C, H, W = self.out_shape
        imgs = img_flat.view(x_low.size(0), C, H, W)
        if self.clamp01:
            imgs = imgs.clamp_(0.0, 1.0)

        if return_dict:
            return {"samples": imgs}
        return imgs

    # ---------- Forward sampling ----------
    def sample(
        self,
        sample_shape,
        return_dict: bool = False,
        seed: Optional[int] = None,
    ):
        # base distribution always asked in (N,) form by LIDSyntheticDataset
        ret = self.base.sample(sample_shape, return_dict=True, seed=seed)
        x_low = ret["samples"].float()  # (N, d)
        lid = ret["lid"].long()
        idx = ret["idx"].long()
        N = x_low.shape[0]

        phi = self._phi(x_low)                       # (N, K)
        img_flat = self.mu.unsqueeze(0) + phi @ self.U.t()  # (N, pixels)
        C, H, W = self.out_shape
        imgs = img_flat.view(N, C, H, W)

        if self.clamp01:
            imgs = imgs.clamp_(0.0, 1.0)

        if return_dict:
            return {"samples": imgs, "lid": lid, "idx": idx}
        return imgs

    # ---------- Inversion helpers ----------
    @torch.no_grad()
    def project_to_phi(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        imgs: (N, C, H, W) in [0,1]
        returns phi_hat ≈ φ(x): (N, K) by U^T (img - mu)
        """
        N = imgs.shape[0]
        flat = imgs.view(N, -1).float()
        phi_hat = (flat - self.mu.unsqueeze(0)) @ self.U  # (N, K)
        return phi_hat

    def invert_to_base(
        self,
        imgs: torch.Tensor,
        steps: int = 200,
        lr: float = 0.2,
        optimizer: str = "adam",   # or "lbfgs"
        verbose: bool = False,
        detach: bool = True,
        batch_size: Optional[int] = None,
        init: str = "ls",          # "zero" | "ls"
    ) -> torch.Tensor:
        """
        Recover base coordinates x (N, d) from images produced by this wrapper.

        imgs: (N, C, H, W)
        returns x_est: (N, d)
        """
        device = imgs.device
        dtype = imgs.dtype

        phi_hat = self.project_to_phi(imgs).to(device=device, dtype=dtype)
        s_hat = phi_hat[:, : self.rff_dim]
        c_hat = phi_hat[:, self.rff_dim : 2 * self.rff_dim]

        # RFF was initialized when we first called _phi; ensure it's ready
        if self._W is None or self._b is None:
            # create a dummy to infer d from base; safer: call sample once before invert
            dummy = self.base.sample((1,), return_dict=True)
            _ = self._phi(dummy["samples"].float().to(device=device, dtype=dtype))
        d = self._W.shape[1]

        N = imgs.size(0)
        if batch_size is None:
            batch_size = N

        x_out = torch.empty(N, d, device=device, dtype=dtype)

        WT = self._W.t()                   # (d, rff_dim)
        pinvW = torch.linalg.pinv(self._W) # (d, rff_dim)
        b_vec = self._b.unsqueeze(0).to(device=device, dtype=dtype)  # (1, rff_dim)

        for i0 in range(0, N, batch_size):
            i1 = min(N, i0 + batch_size)
            s_t = s_hat[i0:i1]
            c_t = c_hat[i0:i1]

            # init
            if init == "ls":
                angle0 = torch.atan2(s_t, c_t)         # (B, rff_dim)
                rhs = angle0 - b_vec                   # (B, rff_dim)
                x0 = rhs @ pinvW.t()                   # (B, d)
            else:
                x0 = torch.zeros(s_t.size(0), d, device=device, dtype=dtype)

            x = x0.clone().requires_grad_(True)

            if optimizer.lower() == "lbfgs":
                opt = torch.optim.LBFGS([x], lr=lr, max_iter=steps, line_search_fn="strong_wolfe")

                def closure():
                    opt.zero_grad(set_to_none=True)
                    proj = F.linear(x, self._W) + b_vec
                    loss = F.mse_loss(torch.sin(proj), s_t) + F.mse_loss(torch.cos(proj), c_t)
                    loss.backward()
                    return loss

                opt.step(closure)
            else:
                opt = torch.optim.Adam([x], lr=lr)
                for _ in range(steps):
                    opt.zero_grad(set_to_none=True)
                    proj = F.linear(x, self._W) + b_vec
                    loss = F.mse_loss(torch.sin(proj), s_t) + F.mse_loss(torch.cos(proj), c_t)
                    loss.backward()
                    opt.step()

            if detach:
                x = x.detach()

            x_out[i0:i1] = x

            if verbose:
                with torch.no_grad():
                    proj = F.linear(x, self._W) + b_vec
                    final = F.mse_loss(torch.sin(proj), s_t) + F.mse_loss(torch.cos(proj), c_t)
                    print(f"[invert_to_base] {i0}:{i1} loss={final.item():.6f}")

        return x_out

    # ---------- Convenience: export/import params ----------
    def export_params(self) -> IDRParams:
        return IDRParams(
            mu=self.mu, U=self.U, W=self._W, b=self._b, sigma=self._sigma,
            rff_dim=self.rff_dim, bias_terms=self.bias_terms, out_shape=self.out_shape
        )

    def import_params(self, params: IDRParams) -> None:
        self.mu = params.mu.float()
        self.U = params.U.float()
        self._W = None if params.W is None else params.W.clone().float()
        self._b = None if params.b is None else params.b.clone().float()
        self._sigma = float(params.sigma)
        self.rff_dim = int(params.rff_dim)
        self.bias_terms = int(params.bias_terms)
        self.out_shape = tuple(params.out_shape)
        self.K = self.U.shape[1]


# =========================
# 4) Minimal usage example
# =========================
"""
# Example (commented — adapt paths/imports to your tree):

from data.datasets.generated import LIDSyntheticDataset
from data.distributions.funnel import Funnel
from data.distributions.utils_fmnist_pca import fit_fmnist_pca
from data.distributions.idr_image import (
    IDRImageWrapper, save_idr_params, load_idr_params
)
import torch
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

# 1) Fit FMNIST PCA basis once and cache
mu, U = fit_fmnist_pca(n_samples=5000, target_class=7, K=32, seed=42, root="./dataset/FMNIST")
# Optionally save:
# save_idr_params("./idr_fmnist7_k32.pt", IDRParams(mu=mu, U=U, sigma=5.0, rff_dim=15, bias_terms=2))

# 2) Base low-D dataset (Funnel: LID = 2)
base = Funnel()  # your distribution class

# 3) Wrap with IDR (images)
idr = IDRImageWrapper(base_dist=base, mu=mu, U=U, rff_dim=15, sigma=5.0, bias_terms=2)

# Warm-up (build W,b for current d)
_ = idr.sample((1,), return_dict=True, seed=0)

# 4) Create dataset in image space
dimg = LIDSyntheticDataset(size=5000, distribution=idr, standardize=False, seed=123)
imgs = dimg.x.to(device)      # (N, 1, 28, 28)
true_lid = dimg.lid           # tensor([2, 2, ...])

# ... run your LID estimator in image space here ...

# 5) Invert images to base coordinates
x_est = idr.invert_to_base(imgs, steps=150, lr=0.2, optimizer="adam", init="ls").cpu()  # (N, 3)

# 6) Visualize recovered 3D Funnel
fig = plt.figure(figsize=(6,5))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x_est[:,0], x_est[:,1], x_est[:,2], s=2, alpha=0.6)
ax.set_xlabel("x1 (≈ t − 4)")
ax.set_ylabel("x2 (≈ r sin θ)")
ax.set_zlabel("x3 (≈ r cos θ)")
ax.set_title("Recovered base coords from IDR’ed FMNIST images (Funnel)")
plt.tight_layout()
plt.show()
"""

