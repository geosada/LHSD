import torch
import torch.distributions as dist

from .lid_base import LIDDistribution

# from https://openreview.net/forum?id=ZEf03Uunvk

class Spiral(LIDDistribution):
    r"""
    1D spiral manifold in R^2, parameterized as in the paper:
        t ~ Uniform(t_min, t_max)
        r = 1 / t
        x1 = r * sin(t / r)
        x2 = r * cos(t / r)

    Default [t_min, t_max] = [1, 100] reproduces the construction used in the
    "Spiral (IDR)" dataset (before any IDR/image-domain mapping).
    Intrinsic dimension (LID) is 1 everywhere.

    Args:
        t_min (float): lower bound of t (inclusive). Default: 1.0
        t_max (float): upper bound of t (inclusive). Default: 100.0
        noise (float): std of optional isotropic Gaussian noise in ambient space. Default: 0.0
        center (tuple[float, float]): optional (x, y) offset added to all samples. Default: (0.0, 0.0)
        scale (float): optional multiplicative scaling of coordinates. Default: 1.0
    """

    def __init__(
        self,
        t_min: float = 1.0,
        t_max: float = 100.0,
        noise: float = 0.0,
        center: tuple[float, float] = (0.0, 0.0),
        scale: float = 1.0,
    ):
        assert t_max > t_min > 0.0, "Require 0 < t_min < t_max."
        self.t_min = float(t_min)
        self.t_max = float(t_max)
        self.noise = float(noise)
        self.center = (float(center[0]), float(center[1]))
        self.scale = float(scale)

    def sample(
        self,
        sample_shape,
        return_dict: bool = False,
        seed: int | None = None,
    ):
        # Normalize sample_shape to (N, 2)
        if isinstance(sample_shape, int):
            sample_shape = (sample_shape, 2)
        assert len(sample_shape) == 1 or (
            len(sample_shape) == 2 and sample_shape[1] == 2
        ), "Sample shape should be N x 2"
        n_samples = sample_shape[0]
    
        # RNG
        g = torch.Generator()
        if seed is not None:
            g.manual_seed(int(seed))
    
        # t ~ Uniform[t_min, t_max]  (use torch.rand with generator)
        t = torch.rand(n_samples, generator=g) * (self.t_max - self.t_min) + self.t_min
    
        # r = 1 / t (strictly positive)
        r = 1.0 / t
    
        # x1 = r * sin(t / r), x2 = r * cos(t / r)
        angle = t / r  # == t**2
        x1 = r * torch.sin(angle)
        x2 = r * torch.cos(angle)
        x = torch.stack([x1, x2], dim=1)
    
        # optional scaling and centering
        if self.scale != 1.0:
            x = self.scale * x
        if self.center != (0.0, 0.0):
            c = torch.tensor(self.center, dtype=x.dtype)
            x = x + c
    
        # optional ambient Gaussian noise (generator supported here)
        if self.noise > 0.0:
            x = x + self.noise * torch.randn_like(x, generator=g)
    
        if return_dict:
            return {
                "samples": x.float(),
                "lid": torch.ones(n_samples, dtype=torch.long),
                "idx": torch.zeros(n_samples, dtype=torch.long),
            }
        return x.float()
    
