from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Type, Union, Optional

import torch
import torch.nn.functional as F  # kept for future extensions
from torch.utils.data import Dataset


# ============================================================
# Base class for modules
# ============================================================


class LIDModule:
    """
    Atomic transformation with a well-defined intrinsic dimension.

    Each module must define:
      - self.lid_dim: how many parameters (i.e. LID contribution)
      - sample_theta(batch_size, device): -> (B, lid_dim)
      - forward(x, theta): x: (B,C,H,W), theta: (B,lid_dim) -> transformed x
    """

    lid_dim: int

    def sample_theta(self, batch_size: int, device: torch.device) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


# ============================================================
# LID-0: Identity module (image as-is)
# ============================================================


class IdentityModule(LIDModule):
    """
    Identity transformation: x' = x.

    - LID contribution = 0
    - Used to represent LID-0 manifolds explicitly when needed.
    """

    def __init__(self) -> None:
        self.lid_dim = 0

    def sample_theta(self, batch_size: int, device: torch.device) -> torch.Tensor:
        # No parameters; return an empty tensor (B, 0)
        return torch.empty(batch_size, 0, device=device)

    def forward(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        # Ignore theta and return x unchanged
        return x


# ============================================================
# Example atomic modules (LID >= 1)
# ============================================================


class BrightnessModule(LIDModule):
    """
    Global brightness shift with safeguards against degenerate (too flat) images:

        x' = x + theta   (then optionally clamped and mixed with x)

    - theta is scalar per image
    - LID contribution = 1

    Args:
        eps            : max absolute brightness shift
        min_std_ratio  : minimal allowed std(y) / std(x); if violated, mix with x
        clip_min, clip_max: optional range to clamp intensities (e.g. [0,1])
    """

    def __init__(
        self,
        eps: float = 0.1,
        min_std_ratio: float = 0.6,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
    ) -> None:
        self.lid_dim = 1
        self.eps = eps
        self.min_std_ratio = min_std_ratio
        self.clip_min = clip_min
        self.clip_max = clip_max

    def sample_theta(self, batch_size: int, device: torch.device) -> torch.Tensor:
        # Uniform(-eps, eps)
        return (2 * self.eps) * torch.rand(batch_size, 1, device=device) - self.eps

    def forward(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """
        x: (B,C,H,W), theta: (B,1)
        """
        B = x.shape[0]
        b = theta.view(B, 1, 1, 1)

        # raw brightness shift
        y = x + b

        # clamp to valid range
        if self.clip_min is not None and self.clip_max is not None:
            y = torch.clamp(y, self.clip_min, self.clip_max)

        # --- safeguard: avoid too-small std (overly flat images) ---
        if self.min_std_ratio is not None:
            x_flat = x.view(B, -1)
            y_flat = y.view(B, -1)

            x_std = x_flat.std(dim=1, keepdim=True) + 1e-8  # avoid div by 0
            y_std = y_flat.std(dim=1, keepdim=True)

            ratio = y_std / x_std  # (B,1)
            # lambda in [0,1]; if ratio < min_std_ratio, pull y back towards x
            lam = (ratio / self.min_std_ratio).clamp(0.0, 1.0)
            lam = lam.view(B, 1, 1, 1)

            y = lam * y + (1.0 - lam) * x

        return y


class ContrastModule(LIDModule):
    """
    Global contrast adjustment with safeguards:

        x' = (alpha) * (x - mean(x)) + mean(x)

    - alpha = 1 + theta (clamped to [alpha_min, alpha_max])
    - LID contribution = 1

    Args:
        eps            : max deviation of alpha from 1.0 (alpha in [1-eps, 1+eps])
        alpha_min      : hard lower bound on alpha (avoid collapse to zero)
        alpha_max      : hard upper bound on alpha (avoid extreme contrast)
        min_std_ratio  : minimal allowed std(y) / std(x); mix with x if below
        max_std_ratio  : maximal allowed std(y) / std(x); mix with x if above
        clip_min, clip_max: optional range to clamp intensities
    """

    def __init__(
        self,
        eps: float = 0.2,
        alpha_min: float = 0.5,
        alpha_max: float = 1.5,
        min_std_ratio: float = 0.5,
        max_std_ratio: float = 2.0,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
    ) -> None:
        self.lid_dim = 1
        self.eps = eps
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.min_std_ratio = min_std_ratio
        self.max_std_ratio = max_std_ratio
        self.clip_min = clip_min
        self.clip_max = clip_max

    def sample_theta(self, batch_size: int, device: torch.device) -> torch.Tensor:
        # theta ~ Uniform(-eps, eps), then alpha = 1 + theta, clamped in forward
        return (2 * self.eps) * torch.rand(batch_size, 1, device=device) - self.eps

    def forward(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """
        x: (B,C,H,W), theta: (B,1)
        """
        B, C, H, W = x.shape

        # per-image mean over spatial dims (channelwise mean is also fine)
        m = x.view(B, C, -1).mean(dim=-1).view(B, C, 1, 1)

        # alpha around 1, with clamps to avoid extreme/negative values
        alpha = 1.0 + theta.view(B, 1, 1, 1)
        alpha = alpha.clamp(self.alpha_min, self.alpha_max)

        # raw contrast transform
        y = alpha * (x - m) + m

        # clamp intensities
        if self.clip_min is not None and self.clip_max is not None:
            y = torch.clamp(y, self.clip_min, self.clip_max)

        # --- safeguard: keep std within [min_std_ratio, max_std_ratio] * std(x) ---
        x_flat = x.view(B, -1)
        y_flat = y.view(B, -1)

        x_std = x_flat.std(dim=1, keepdim=True) + 1e-8
        y_std = y_flat.std(dim=1, keepdim=True)
        ratio = y_std / x_std  # (B,1)

        # target: ratio in [min_std_ratio, max_std_ratio]
        lower = self.min_std_ratio
        upper = self.max_std_ratio

        # For samples with ratio < lower or > upper, blend with x
        # define a per-sample blend coefficient lam in [0,1]:
        #   lam=1 -> use y as is; lam=0 -> revert to x.
        lam = torch.ones_like(ratio)

        # too flat: pull towards x
        mask_low = ratio < lower
        if mask_low.any():
            lam_low = (ratio[mask_low] / lower).clamp(0.0, 1.0)
            lam[mask_low] = lam_low

        # too strong contrast: also pull towards x
        mask_high = ratio > upper
        if mask_high.any():
            # if ratio == upper -> lam=1, if ratio >> upper -> lam small
            lam_high = (upper / ratio[mask_high]).clamp(0.0, 1.0)
            lam[mask_high] = lam_high

        lam = lam.view(B, 1, 1, 1)
        y = lam * y + (1.0 - lam) * x

        return y



class CircularTranslationModule(LIDModule):
    """
    Circular (wrap-around) integer-pixel translation in x and y using torch.roll.

        x' = roll(x, shifts=(ty, tx))

    - LID contribution = 2 (tx, ty)
    - Safe w.r.t. boundaries, no padding artifacts.
    """

    def __init__(self, max_shift: int = 4) -> None:
        self.lid_dim = 2
        self.max_shift = max_shift

    def sample_theta(self, batch_size: int, device: torch.device) -> torch.Tensor:
        # Sample integer shifts in [-max_shift, max_shift]
        tx = torch.randint(
            -self.max_shift, self.max_shift + 1, (batch_size, 1), device=device
        )
        ty = torch.randint(
            -self.max_shift, self.max_shift + 1, (batch_size, 1), device=device
        )
        return torch.cat([tx, ty], dim=1).float()

    def forward(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        # theta: (B, 2) -> per-sample roll
        B, C, H, W = x.shape
        tx = theta[:, 0].long()
        ty = theta[:, 1].long()
        out = torch.empty_like(x)
        for i in range(B):
            out[i] = torch.roll(
                x[i], shifts=(ty[i].item(), tx[i].item()), dims=(1, 2)
            )
        return out


# ============================================================
# SimulatedManifold object (formerly LIDManifold)
# ============================================================


@dataclass
class SimulatedManifold:
    """
    Represents a manifold generated by composing several LIDModules
    starting from a base image.

    base_image: (C, H, W) tensor
    modules: sequence of LIDModule instances

    The ground-truth LID is:
        lid_dim = sum(module.lid_dim for module in modules)

    Special case:
        - LID-0 manifold ("image as is"):
          modules can be [] or [IdentityModule()].
    """

    base_image: torch.Tensor
    modules: Sequence[LIDModule]

    @property
    def lid_dim(self) -> int:
        return sum(m.lid_dim for m in self.modules)

    def sample(
        self,
        n_samples: int,
        device: torch.device = torch.device("cpu"),
    ) -> Dict[str, torch.Tensor]:
        """
        Sample n_samples points on this manifold.

        Returns a dict with:
          - "x":     (n_samples, C, H, W) transformed images
          - "theta": (n_samples, lid_dim) concatenated parameters
                     (empty tensor if lid_dim == 0)
          - "lid":   (n_samples,) long tensor with the scalar LID
        """
        base = self.base_image.to(device).unsqueeze(0).expand(
            n_samples, -1, -1, -1
        )  # (B,C,H,W)
        x = base
        thetas: List[torch.Tensor] = []

        for m in self.modules:
            theta_m = m.sample_theta(n_samples, device)  # (B, m.lid_dim)
            thetas.append(theta_m)
            x = m.forward(x, theta_m)

        if len(thetas) > 0:
            theta_all = torch.cat(thetas, dim=1)
        else:
            theta_all = torch.empty(n_samples, 0, device=device)

        lid_val = self.lid_dim
        lid_tensor = torch.full(
            (n_samples,),
            lid_val,
            device=device,
            dtype=torch.long,
        )

        return {
            "x": x,
            "theta": theta_all,
            "lid": lid_tensor,
        }


# ============================================================
# Registry and spec handling
# ============================================================

MODULE_REGISTRY: Dict[str, Type[LIDModule]] = {
    # LID-0
    "identity": IdentityModule,
    # Intensity
    "brightness": BrightnessModule,
    "contrast": ContrastModule,
    # Spatial
    "circ_translation": CircularTranslationModule,
    # Add more modules here as needed
}

# Spec entry can be either:
#   - "brightness"
#   - {"name": "brightness", "eps": 0.05}
ModuleSpec = Union[str, Dict[str, Any]]


def build_module_from_spec(entry: ModuleSpec) -> LIDModule:
    """
    Build a LIDModule instance from a spec entry.

    entry can be:
      - a string: "brightness"
      - a dict:   {"name": "brightness", "eps": 0.05}
    """
    # Case 1: simple string → default constructor
    if isinstance(entry, str):
        name = entry
        if name not in MODULE_REGISTRY:
            raise ValueError(f"Unknown module name '{name}'")
        cls = MODULE_REGISTRY[name]
        return cls()  # default parameters

    # Case 2: dict with explicit parameters
    if isinstance(entry, dict):
        if "name" not in entry:
            raise ValueError(f"Module spec dict must contain a 'name' key, got {entry}")
        name = entry["name"]
        if name not in MODULE_REGISTRY:
            raise ValueError(f"Unknown module name '{name}'")
        cls = MODULE_REGISTRY[name]
        kwargs = {k: v for k, v in entry.items() if k != "name"}
        return cls(**kwargs)

    raise TypeError(f"Unsupported module spec type: {type(entry)}")


def build_simulated_manifolds_for_images(
    images: Dict[str, torch.Tensor],
    spec: Dict[str, List[ModuleSpec]],
) -> Dict[str, SimulatedManifold]:
    """
    Build SimulatedManifold objects for a set of images according to a given spec.

    Args:
        images:
            Mapping from image ID (e.g., "A", "B") to tensor (C, H, W).

        spec:
            Mapping from image ID to a list of module specs.
            Each module spec can be:

                - "brightness"
                - {"name": "brightness", "eps": 0.05}

    Returns:
        manifolds: dict {image_id: SimulatedManifold}
    """
    manifolds: Dict[str, SimulatedManifold] = {}

    for img_id, img in images.items():
        if img_id not in spec:
            raise ValueError(f"No spec given for image_id {img_id}")

        module_specs = spec[img_id]

        # Empty list => LID-0 manifold (image as-is)
        if len(module_specs) == 0:
            modules: List[LIDModule] = []
        else:
            modules = [build_module_from_spec(entry) for entry in module_specs]

        manifolds[img_id] = SimulatedManifold(base_image=img, modules=modules)

    return manifolds


# ============================================================
# Convenience helper for LID-0
# ============================================================


def create_lid0_manifold(image: torch.Tensor) -> SimulatedManifold:
    """
    Create a LID-0 manifold for a single image.
    The image will always be returned as-is (no transformation).
    """
    return SimulatedManifold(base_image=image, modules=[])


# ============================================================
# Dataset that samples from multiple SimulatedManifolds
# ============================================================


class SimulatedManifoldDataset(Dataset):
    """
    On-the-fly dataset sampling from multiple SimulatedManifolds.

    return_mode:
      - "x":        return only image tensor        -> batch is (B, C, H, W)
      - "x_lid":    return (x, lid)                 -> batch is tuple (B,C,H,W), (B,)
      - "full":     return dict with x, lid, theta, manifold_idx

    Sampling strategy over manifolds:

      lid_sampling = "uniform_lid":
        - each distinct LID value gets the same total probability
        - within each LID group, manifolds are sampled uniformly

      lid_sampling = "uniform_manifold":
        - each manifold (image) is equally likely

    Alternatively, you can pass `sampling_weights` to override the strategy:

      sampling_weights:
        - Dict[str, float]: keys are manifold IDs (same as in `manifolds` dict)
        - or a sequence of floats with length == number of manifolds
          in the order of `manifolds.keys()`.

        These are normalized to form probabilities and override `lid_sampling`.
    """

    def __init__(
        self,
        manifolds: Dict[str, SimulatedManifold],
        n_per_epoch: int = 50_000,
        device: torch.device = torch.device("cpu"),
        lid_sampling: str = "uniform_lid",
        return_mode: str = "x",
        sampling_weights: Optional[Union[Dict[str, float], Sequence[float], torch.Tensor]] = None,
    ) -> None:
        super().__init__()

        if len(manifolds) == 0:
            raise ValueError("manifolds dict is empty.")

        if return_mode not in ("x", "x_lid", "full"):
            raise ValueError("return_mode must be one of {'x', 'x_lid', 'full'}")

        self.device = device
        self.n_per_epoch = int(n_per_epoch)
        self.return_mode = return_mode

        # store ids and manifolds in a stable order
        self.manifold_ids: List[str] = list(manifolds.keys())
        self.manifolds: List[SimulatedManifold] = [manifolds[k] for k in self.manifold_ids]

        # tensor of LIDs per manifold
        self.lids = torch.tensor(
            [m.lid_dim for m in self.manifolds], dtype=torch.long
        )
        self.max_lid_dim = int(self.lids.max().item())

        # build sampling probabilities over manifolds
        self.probs = self._build_probs(lid_sampling, sampling_weights)

    def _build_probs(
        self,
        lid_sampling: str,
        sampling_weights: Optional[Union[Dict[str, float], Sequence[float], torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Build sampling probabilities over manifolds.

        If `sampling_weights` is provided, it overrides `lid_sampling` behavior:

          - If dict: keys are manifold_ids, values are non-negative weights.
          - If sequence / tensor: length must match number of manifolds, and
            order is assumed to follow `self.manifold_ids`.

        Otherwise, it falls back to 'uniform_manifold' or 'uniform_lid'.
        """
        num_m = len(self.manifolds)

        # ---- Custom weights override built-in strategies ----
        if sampling_weights is not None:
            if isinstance(sampling_weights, dict):
                # map dict of {id: weight} to ordered tensor
                weights_list: List[float] = []
                for mid in self.manifold_ids:
                    if mid not in sampling_weights:
                        raise ValueError(
                            f"sampling_weights dict missing entry for manifold_id '{mid}'"
                        )
                    weights_list.append(float(sampling_weights[mid]))
                probs = torch.tensor(weights_list, dtype=torch.float)
            else:
                # sequence or tensor; rely on order
                probs = torch.as_tensor(sampling_weights, dtype=torch.float)
                if probs.numel() != num_m:
                    raise ValueError(
                        f"sampling_weights length {probs.numel()} does not match "
                        f"number of manifolds {num_m}"
                    )

            if (probs < 0).any():
                raise ValueError("sampling_weights must be non-negative.")
            total = probs.sum().item()
            if total <= 0:
                raise ValueError("Sum of sampling_weights must be positive.")
            probs = probs / total
            return probs

        # ---- Built-in strategies ----
        if lid_sampling == "uniform_manifold":
            probs = torch.ones(num_m, dtype=torch.float)
            probs /= probs.sum()
            return probs

        elif lid_sampling == "uniform_lid":
            lids = self.lids
            unique_lids = lids.unique()
            k = len(unique_lids)

            probs = torch.zeros(num_m, dtype=torch.float)
            for L in unique_lids:
                idxs = (lids == L).nonzero(as_tuple=True)[0]
                group_mass = 1.0 / float(k)  # each LID value gets 1/k
                per_m = group_mass / float(len(idxs))
                probs[idxs] = per_m

            probs /= probs.sum()
            return probs

        else:
            raise ValueError(f"Unknown lid_sampling mode '{lid_sampling}'")

    def __len__(self) -> int:
        return self.n_per_epoch

    def __getitem__(self, idx: int):
        # sample manifold index according to probs
        with torch.no_grad():
            m_idx = torch.multinomial(self.probs, 1).item()

        manifold = self.manifolds[m_idx]
        out = manifold.sample(n_samples=1, device=self.device)

        x = out["x"][0]       # (C,H,W)
        lid = out["lid"][0]   # scalar

        # theta: (1, lid_dim) or (1, 0) -> flatten and pad to max_lid_dim
        theta = out["theta"].view(-1)  # (lid_dim,) or (0,)
        pad_dim = self.max_lid_dim - theta.numel()
        if pad_dim < 0:
            raise RuntimeError("Found lid_dim > max_lid_dim; inconsistent state.")
        if pad_dim > 0:
            theta = torch.cat(
                [theta, torch.zeros(pad_dim, device=self.device)], dim=0
            )  # (max_lid_dim,)

        manifold_idx = m_idx  # int index, safe for default_collate

        # -------- return according to mode --------
        if self.return_mode == "x":
            # just the image tensor (for trainers expecting tensor batches)
            return x

        if self.return_mode == "x_lid":
            # tuple -> default_collate gives (B,C,H,W) and (B,)
            return x, lid

        # full dict: all numeric types (no strings) so default_collate works
        return {
            "x": x,
            "lid": lid,
            "theta": theta,
            "manifold_idx": manifold_idx,
        }
