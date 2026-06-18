import math
import torch

from .lid_base import LIDDistribution


class Funnel(LIDDistribution):
    r"""
    2D "funnel" manifold embedded in R^3:

        t ~ Uniform(t_min, t_max)         (default: [0, 8])
        r = r0 * exp(-t)                  (default r0 = 3)
        θ ~ Uniform(0, 2π)

        x1 = t - t_shift                  (default t_shift = 4)
        x2 = r * sin(θ)
        x3 = r * cos(θ)

    Intrinsic dimensionality (LID) = 2 everywhere.

    Args:
        t_min (float): lower bound for t. Default: 0.0
        t_max (float): upper bound for t. Default: 8.0
        r0 (float): base radius at t = 0. Default: 3.0
        t_shift (float): shift for x1 = t - t_shift. Default: 4.0
        noise (float): ambient isotropic Gaussian noise std. Default: 0.0
        center (tuple[float, float, float]): added offset. Default: (0., 0., 0.)
        scale (float): multiplicative scaling. Default: 1.0
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
    ):
        assert t_max > t_min, "Require t_max > t_min."
        self.t_min = float(t_min)
        self.t_max = float(t_max)
        self.r0 = float(r0)
        self.t_shift = float(t_shift)
        self.noise = float(noise)
        self.center = (float(center[0]), float(center[1]), float(center[2]))
        self.scale = float(scale)

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

        r = self.r0 * torch.exp(-t)
        x1 = t - self.t_shift
        x2 = r * torch.sin(theta)
        x3 = r * torch.cos(theta)
        x = torch.stack([x1, x2, x3], dim=1)

        if self.scale != 1.0:
            x = self.scale * x
        if self.center != (0.0, 0.0, 0.0):
            c = torch.tensor(self.center, dtype=x.dtype)
            x = x + c

        if self.noise > 0.0:
            x = x + self.noise * torch.randn_like(x, generator=g)

        if return_dict:
            return {
                "samples": x.float(),
                "lid": 2 * torch.ones(n_samples, dtype=torch.long),
                "idx": torch.zeros(n_samples, dtype=torch.long),
            }
        return x.float()

