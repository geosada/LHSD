import math
import torch

from .lid_base import LIDDistribution


class Moon(LIDDistribution):
    r"""
    3D "moon" (crescent) manifold in R^3.

    Intrinsic dimensionality (LID) is 3 in the interior, 2 on boundary surfaces,
    and 1 where two boundaries intersect.

    Args:
        r, inner_ratio, inner_shift, noise, center, scale, batch_factor, uniform_density:
            (same as before)
        tol_surface_ratio (float): tolerance (as a fraction of r) for being "near"
            the top/bottom surfaces x3=±τ(φ). Default: 5e-3.
        tol_cylinder_ratio (float): tolerance (as a fraction of r) for being "near"
            the vertical cylindrical surfaces (outer/inner circle boundaries).
            Default: 5e-3.
    """

    def __init__(
        self,
        r: float = 3.0,
        inner_ratio: float = 0.899,
        inner_shift: float = 0.1,
        noise: float = 0.0,
        center: tuple[float, float, float] = (0.0, 0.0, 0.0),
        scale: float = 1.0,
        batch_factor: int = 4,
        uniform_density: bool = False,
        # --- NEW ---
        tol_surface_ratio: float = 5e-3,
        tol_cylinder_ratio: float = 5e-3,
    ):
        assert r > 0.0
        assert 0.0 < inner_ratio < 1.0
        self.r = float(r)
        self.inner_ratio = float(inner_ratio)
        self.inner_shift = float(inner_shift)
        self.noise = float(noise)
        self.center = (float(center[0]), float(center[1]), float(center[2]))
        self.scale = float(scale)
        self.batch_factor = int(batch_factor)
        self.uniform_density = bool(uniform_density)
        # --- NEW ---
        self.tol_surface = float(tol_surface_ratio) * self.r
        self.tol_cylinder = float(tol_cylinder_ratio) * self.r

    @staticmethod
    def _tau_from_phi(r: float, phi: torch.Tensor) -> torch.Tensor:
        # same τ(φ) as before
        return r * (0.001 + 0.2 * (0.5 * (1.0 - torch.sin(phi))))

    def _sample_crescent_2d(self, n_samples: int, g: torch.Generator) -> torch.Tensor:
        # (unchanged except for device-safe rand usage already in your file)
        r = self.r
        inner_r = self.inner_ratio * r
        inner_cy = -self.inner_shift
        tau_max = 0.201 * r

        xs = []
        need = n_samples
        while need > 0:
            m = max(need * self.batch_factor, 1024)
            phi = torch.rand(m, generator=g) * (2.0 * math.pi)
            rho = r * torch.sqrt(torch.rand(m, generator=g))

            x = rho * torch.cos(phi)
            y = rho * torch.sin(phi)

            inside_inner = (x**2 + (y - inner_cy) ** 2) < (inner_r**2)
            keep = ~inside_inner

            if keep.any():
                kept_x = x[keep]
                kept_y = y[keep]
                if self.uniform_density:
                    kept_phi = torch.atan2(kept_y, kept_x)
                    kept_tau = self._tau_from_phi(r, kept_phi)
                    acc = torch.rand(kept_tau.shape, generator=g, device=kept_tau.device) * tau_max < kept_tau
                    if acc.any():
                        kept_xy = torch.stack([kept_x[acc], kept_y[acc]], dim=1)
                        xs.append(kept_xy)
                        need -= kept_xy.shape[0]
                else:
                    kept_xy = torch.stack([kept_x, kept_y], dim=1)
                    xs.append(kept_xy)
                    need -= kept_xy.shape[0]

        xy = torch.cat(xs, dim=0)[:n_samples]
        return xy

    def _classify_gt_lid(self, x1, x2, x3, tau):
        """
        Compute GT local dimension by proximity to boundary surfaces.
        Returns gt_lid (LongTensor) and diagnostic distances.
        """
        r = self.r
        inner_r = self.inner_ratio * r
        inner_cy = -self.inner_shift

        # distances to vertical cylindrical boundaries (outer & inner circles)
        rho_outer = torch.sqrt(x1**2 + x2**2)
        rho_inner = torch.sqrt(x1**2 + (x2 - inner_cy)**2)  # note: inner center at (0, -inner_shift)
        d_outer = torch.abs(rho_outer - r)
        d_inner = torch.abs(rho_inner - inner_r)
        near_cylinder = (torch.minimum(d_outer, d_inner) <= self.tol_cylinder)

        # distance to top/bottom surfaces x3 = ±tau(φ)
        d_surface = torch.abs(torch.abs(x3) - tau)
        near_surface = d_surface <= self.tol_surface

        # combine: both → edge (1D), exactly one → surface (2D), none → interior (3D)
        gt = torch.full_like(x3, 3, dtype=torch.long)
        gt[near_surface ^ near_cylinder] = 2
        gt[near_surface & near_cylinder] = 1

        return gt, d_outer, d_inner, d_surface

    def sample(
        self,
        sample_shape,
        return_dict: bool = False,
        seed: int | None = None,
        return_debug: bool = False,   # NEW: optionally return distances
    ):
        # normalize shape
        if isinstance(sample_shape, int):
            sample_shape = (sample_shape, 3)
        assert len(sample_shape) == 1 or (len(sample_shape) == 2 and sample_shape[1] == 3), "Sample shape should be N x 3"
        n_samples = sample_shape[0]

        g = torch.Generator()
        if seed is not None:
            g.manual_seed(int(seed))

        # (x1, x2)
        xy = self._sample_crescent_2d(n_samples, g)
        x1, x2 = xy[:, 0], xy[:, 1]

        # angle & tau (before noise)
        phi = torch.atan2(x2, x1)
        tau = self._tau_from_phi(self.r, phi)

        # x3
        u = torch.rand(n_samples, generator=g) * 2.0 - 1.0
        x3 = tau * u

        # classify GT LID on the clean sample
        gt_lid, d_outer, d_inner, d_surface = self._classify_gt_lid(x1, x2, x3, tau)

        # assemble and apply transforms
        x = torch.stack([x1, x2, x3], dim=1)
        if self.scale != 1.0:
            x = self.scale * x
        if self.center != (0.0, 0.0, 0.0):
            c = torch.tensor(self.center, dtype=x.dtype)
            x = x + c

        # add noise *after* labeling, so GT is for the underlying manifold
        if self.noise > 0.0:
            x = x + self.noise * torch.randn_like(x, generator=g)

        if return_dict:
            out = {
                "samples": x.float(),
                "gt_lid": gt_lid,                # NEW
                "lid": 3 * torch.ones(n_samples, dtype=torch.long),  # legacy field
                "idx": torch.zeros(n_samples, dtype=torch.long),
                "phi": phi,                      # helpful for analysis
                "tau": tau,
            }
            if return_debug:
                out.update({
                    "d_outer": d_outer, "d_inner": d_inner, "d_surface": d_surface
                })
            return out
        return x.float()
