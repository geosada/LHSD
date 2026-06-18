import math
import torch
import torch.nn.functional as F

from .lid_base import LIDDistribution


def smoothstep(x, edge0, edge1):
    """
    Smooth interpolation (0→1) as x moves from edge0 to edge1.
    Returns tensor same shape as x.
    """
    t = (x - edge0) / (edge1 - edge0 + 1e-12)
    t = t.clamp(0., 1.)
    return t * t * (3.0 - 2.0 * t)


class Funnel(LIDDistribution):
    r"""
    2D "funnel" manifold embedded in R^3:

        t ~ Uniform(t_min, t_max)
        r = r0 * exp(-t)
        θ ~ Uniform(0, 2π)

        x1 = t - t_shift
        x2 = r * sin(θ)
        x3 = r * cos(θ)

    NEW:
        `gt_lid` is assigned as:
            - 1D   for small-radius (stick)
            - 3D   for large-radius (skirt)
            - continuous from 1→3 in between
    """

    def __init__(
        self,
        t_min: float = 0.0,
        t_max: float = 8.0,
        r0: float = 3.0,
        t_shift: float = 4.0,
        noise: float = 0.0,
        center: tuple[float, float, float] = (0.0, 0.0, 0.0),
        scale: float = 1.0,
        # NEW PARAMETERS for LID behavior
        r_stick: float = 0.20,   # radius >= r_stick → 1D stick
        r_skirt: float = 1.20,   # radius <= r_skirt → 3D skirt
    ):
        assert t_max > t_min, "Require t_max > t_min."
        assert r_skirt > 0 and r_stick > 0

        self.t_min = float(t_min)
        self.t_max = float(t_max)
        self.r0 = float(r0)
        self.t_shift = float(t_shift)
        self.noise = float(noise)
        self.center = (float(center[0]), float(center[1]), float(center[2]))
        self.scale = float(scale)

        # LID thresholds
        self.r_stick = float(r_stick)
        self.r_skirt = float(r_skirt)

    def _compute_gt_lid(self, r):
        """
        Compute GT local dimension from radius.

        small r → skirt (3D)
        large r → stick (1D)
        in between → 1..3 continuous smoothstep
        """
        # If funnel opens inwards, r decreasing/increasing varies with params,
        # but we only care about r magnitude.
        low = self.r_skirt     # 3D
        high = self.r_stick    # 1D

        # smoothstep maps: r=high→0, r=low→1
        w = smoothstep(r, high, low)
        lid = 1.0 + 2.0 * w     # maps w∈[0,1] to LID∈[1,3]

        return lid

    def sample(
        self,
        sample_shape,
        return_dict: bool = False,
        seed: int | None = None,
    ):
        # Normalize sample_shape to (N, 3)
        if isinstance(sample_shape, int):
            sample_shape = (sample_shape, 3)
        assert len(sample_shape) == 1 or (
            len(sample_shape) == 2 and sample_shape[1] == 3
        ), "Sample shape should be N x 3"
        n_samples = sample_shape[0]

        g = torch.Generator()
        if seed is not None:
            g.manual_seed(int(seed))

        # t ~ U[t_min, t_max], θ ~ U[0, 2π]
        t = torch.rand(n_samples, generator=g) * (self.t_max - self.t_min) + self.t_min
        theta = torch.rand(n_samples, generator=g) * (2.0 * math.pi)

        # Funnel radius
        r = self.r0 * torch.exp(-t)

        # Coordinates
        x1 = t - self.t_shift
        x2 = r * torch.sin(theta)
        x3 = r * torch.cos(theta)
        x = torch.stack([x1, x2, x3], dim=1)

        # Apply scale and center
        if self.scale != 1.0:
            x = self.scale * x
        if self.center != (0.0, 0.0, 0.0):
            c = torch.tensor(self.center, dtype=x.dtype)
            x = x + c

        # Add noise *after* computing LID
        if self.noise > 0.0:
            x = x + self.noise * torch.randn_like(x, generator=g)

        # NEW: ground-truth LID
        gt_lid = self._compute_gt_lid(r)     # float LID ∈ [1,3]

        if return_dict:
            return {
                "samples": x.float(),
                "gt_lid": gt_lid.float(),         # continuous GT
                "lid": gt_lid.round().long(),     # optional legacy integer field
                "idx": torch.zeros(n_samples, dtype=torch.long),
                "r": r,
                "t": t,
            }

        return x.float()
