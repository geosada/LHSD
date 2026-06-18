# models/diffusions/sdes/sdes.py
# (Modified to add safe debug instrumentation in Sde.score)
# Comments are in English as requested.

import functools
import math
import numbers
from abc import ABC, abstractmethod
from typing import Literal

import torch
import torch.nn as nn
from tqdm import tqdm

from ..utils import batch_linspace
from .utils import HUTCHINSON_DATA_DIM_THRESHOLD, compute_trace_of_jacobian, copy_tensor_or_create


class Sde(ABC, nn.Module):
    """The forward- and reverse-SDEs defining a diffusion model.

    This closely follows the math described in Song et al. (2020).  (Available here
    https://arxiv.org/abs/2011.13456). Equation numbers in the comments throughout
    this file refer to the equations in the paper.

    This class implements a general SDE, as given by equation (5),
        dx = f(x, t)dt + g(t)dw,
    and an approximation of its reverse SDE, the true form of which is given by equation (6):
        dx = [f(x, t) - g(t)^2 grad_x log p_t(x)]dt + g(t) dw',
    where w' is a reverse Brownian motion.

    In reality, we approximate grad_x log p_t(x) with our score network, s(x, t).
    Specific choices of f(x, t) and g(t) should be implemented as subclasses.

    This code implements:
    (1) The forward and backward SDE.
    (2) The forward and backward ODE.
    (3) Log probability estimates.

    Args:
        score_net: a network corresponding to (x, t) |-> s(x, t) * sigma(t), with sigma(t)
            as defined in self.sigma(t) below

        NOTE:
        By default, this Sde.score() assumes `score_net(x,t)` returns `sigma(t) * score(x,t)`.
        If your `score_net` returns the raw score directly, set:
            self._debug_expected = "raw_score"
        (You can do this from your notebook without modifying this file further.)
    """

    def __init__(self, score_net: torch.nn.Module):
        super().__init__()
        self.score_net = score_net

        # =========================
        # DEBUG / SAFETY INSTRUMENTS
        # =========================
        # Enable/disable debug printing inside `score()`.
        self._debug_score = False

        # Print every N calls (useful when score() is called in loops).
        self._debug_score_every = 1
        self._debug_score_count = 0

        # Convention toggle:
        #   "sigma_score": score_net returns sigma(t) * score(x,t), and we divide by sigma(t).
        #   "raw_score"  : score_net returns score(x,t) directly, and we do NOT divide.
        self._debug_expected = "sigma_score"

    @abstractmethod
    def drift(self, x, t):
        """The drift coefficient f(x, t) of the forward SDE"""

    @abstractmethod
    def diff(self, t):
        """The diffusion coefficient g(t) of the forward SDE"""

    @abstractmethod
    def sigma(self, t_end, t_start=0):
        """The standard deviation of x(t_end) | x(t_start)"""

    def score(self, x, t, **score_kwargs):
        """
        The score s(x, t) of the forward SDE at time t.

        Default convention:
          - score_net(x,t) returns sigma(t) * s(x,t)
          - this function returns s(x,t) by dividing by sigma(t)

        If your score_net already returns the raw score, set:
            self._debug_expected = "raw_score"
        """

        sigma_t = self.sigma(t).to(x.device)

        t = copy_tensor_or_create(t, device=x.device)

        if t.ndim == 0:
            t = t.expand(x.shape[0])  # t should be batched for the score_net
        else:
            new_dims = x.ndim - sigma_t.ndim  # Expand sigma_t for broadcasting
            sigma_t = sigma_t.reshape(x.shape[:1] + (1,) * new_dims)

        # -------------------------
        # Debug printing (optional)
        # -------------------------
        self._debug_score_count += 1
        do_dbg = self._debug_score and (self._debug_score_count % self._debug_score_every == 0)

        if do_dbg:
            # x statistics
            x0 = x.detach()
            print(
                f"[Sde.score DEBUG] x: shape={tuple(x0.shape)} "
                f"min={float(x0.min().cpu()):.4g} max={float(x0.max().cpu()):.4g} "
                f"mean={float(x0.mean().cpu()):.4g} std={float(x0.std().cpu()):.4g}"
            )

            # t statistics (after batching)
            t0 = t.detach()
            print(
                f"[Sde.score DEBUG] t: shape={tuple(t0.shape)} "
                f"min={float(t0.min().cpu()):.4g} max={float(t0.max().cpu()):.4g} "
                f"mean={float(t0.float().mean().cpu()):.4g}"
            )

            # sigma_t statistics (broadcasted)
            st = sigma_t.detach()
            # Reduce to per-sample scalars even if sigma_t is broadcasted to (B,1,1,1,...)
            st_flat = st.view(st.shape[0], -1)[:, 0]
            print(
                f"[Sde.score DEBUG] sigma_t: shape={tuple(st.shape)} "
                f"min={float(st_flat.min().cpu()):.6g} max={float(st_flat.max().cpu()):.6g} "
                f"mean={float(st_flat.mean().cpu()):.6g} sigma2_mean={float((st_flat**2).mean().cpu()):.6g}"
            )
            print(f"[Sde.score DEBUG] convention expected: {self._debug_expected}")

        # Compute raw network output
        score_out = self.score_net(x, t, **score_kwargs)

        if do_dbg:
            y0 = score_out.detach()
            print(
                f"[Sde.score DEBUG] score_net(x,t) output: shape={tuple(y0.shape)} "
                f"min={float(y0.min().cpu()):.4g} max={float(y0.max().cpu()):.4g} "
                f"mean={float(y0.mean().cpu()):.4g} std={float(y0.std().cpu()):.4g}"
            )

        # Normalize according to chosen convention
        if self._debug_expected == "raw_score":
            # score_net already returns the score ∇ log p_t(x)
            score = score_out
        else:
            # default: score_net returns sigma(t) * score
            score = score_out / sigma_t

        if do_dbg:
            s0 = score.detach()
            print(
                f"[Sde.score DEBUG] returned score: shape={tuple(s0.shape)} "
                f"min={float(s0.min().cpu()):.4g} max={float(s0.max().cpu()):.4g} "
                f"mean={float(s0.mean().cpu()):.4g} std={float(s0.std().cpu()):.4g}"
            )

        return score

    @abstractmethod
    def solve_forward_sde(self, x_start, t_end=1.0, t_start=0.0, return_eps=False):
        """Expectation: t_start < t_end"""

    def solve_forward_ode(self, x_start, t_start=1e-4, t_end=1.0, steps=1000, **score_kwargs):
        t_start, t_end = self._match_timestep_shapes(t_start, t_end)
        assert torch.all(t_start <= t_end)

        return self._solve(x_start, t_start, t_end, steps, stochastic=False, **score_kwargs)

    def solve_reverse_sde(self, x_start, t_start=1.0, t_end=1e-4, steps=1000, **score_kwargs):
        t_start, t_end = self._match_timestep_shapes(t_start, t_end)
        assert torch.all(t_start >= t_end)

        return self._solve(x_start, t_start, t_end, steps, stochastic=True, **score_kwargs)

    def solve_reverse_ode(self, x_start, t_start=1.0, t_end=1e-4, steps=1000, **score_kwargs):
        t_start, t_end = self._match_timestep_shapes(t_start, t_end)
        assert torch.all(t_start >= t_end)

        return self._solve(x_start, t_start, t_end, steps, stochastic=False, **score_kwargs)

    @staticmethod
    def _match_timestep_shapes(t_start, t_end):
        t_start = copy_tensor_or_create(t_start)
        t_end = copy_tensor_or_create(t_end)
        if t_start.ndim > t_end.ndim:
            t_end = torch.full_like(t_start, fill_value=t_end)
        elif t_start.ndim < t_end.ndim:
            t_start = torch.full_like(t_end, fill_value=t_start)
        return t_start, t_end

    @torch.no_grad()
    def _solve(
        self,
        x_start: torch.Tensor,
        t_start: float = 1.0,
        t_end: float = 1e-4,
        steps: int = 1000,
        stochastic: bool = True,
        **score_kwargs,
    ):
        """Solve the SDE or ODE with an Euler(-Maruyama) solver.

        Note that this can be used for either the forward or backward solve, depending on whether
        t_start < t_end (forward) or t_start > t_end (reverse). Note that this method is not
        appropriate for the forward SDE; the forward SDE should have an analytical solution.

        TODO: Add predictor-corrector steps.

        Args:
            x_start (Tensor of shape (batch_size, ...)): The starting point
            t_start: The starting time
            t_end: The final time (best not set to zero for numerical stability)
            steps: The number of steps for the solver
            stochastic: Whether to use the SDE (True) or ODE (False)

        Returns:
            x_end: (Tensor of shape (batch_size, ...))
        """
        device = x_start.device
        x = x_start.detach().clone()

        ts = batch_linspace(t_start, t_end, steps=steps).to(device)
        delta_t = copy_tensor_or_create((t_end - t_start) / (steps - 1))  # Negative in reverse time

        rng = tqdm(ts, desc="Iterating the solver")
        for t in rng:
            score = self.score(x, t, **score_kwargs)
            drift = self.drift(x, t)
            diff = self.diff(t)

            if t.ndim > 0:  # diff is batched, so add dimensions for broadcasting
                new_dims = x.ndim - t.ndim
                diff = diff.reshape(x.shape[:1] + (1,) * new_dims)
                delta_t = delta_t.reshape(x.shape[:1] + (1,) * new_dims)

            if stochastic:
                # Perform an Euler-Maruyama step on the reverse SDE from equation (6)
                delta_w = delta_t.abs().sqrt() * torch.randn(x.shape).to(device)
                dx = (drift - diff**2 * score) * delta_t + diff * delta_w
            else:
                # Compute an Euler step on the reverse ODE from equation (13)
                dx = (drift - diff**2 * score / 2) * delta_t

            x += dx
        return x

    def _trace_of_drift_derivative(
        self,
        x: torch.Tensor,
        t: float,
        method: (
            Literal["hutchinson_gaussian", "hutchinson_rademacher", "deterministic"] | None
        ) = None,
        hutchinson_sample_count: int = HUTCHINSON_DATA_DIM_THRESHOLD,
        chunk_size: int = 128,
        seed: int = 42,
        verbose: bool = False,
    ):
        """
        Return the trace of the drift derivative for the log_prob calculation.

        In the generic case, we can use the Hutchinson estimator for this purpose.
        However, in many cases such as VpSDE and VeSDE, the trace of the drift derivative
        can be directly computed using the diffusion hyperparameters.

        For example,

        VP-SDE: this value is \\beta(t) \\times d where d is the dimension of the data.
        VE-SDE: this value is 0.
        """
        drift_fn = functools.partial(self.drift, t=t)
        return compute_trace_of_jacobian(
            fn=drift_fn,
            x=x,
            method=method,
            hutchinson_sample_count=hutchinson_sample_count,
            chunk_size=chunk_size,
            seed=seed,
            verbose=verbose,
        )

    def laplacian(
        self,
        x: torch.Tensor,
        t: float,
        method: (
            Literal["hutchinson_gaussian", "hutchinson_rademacher", "deterministic"] | None
        ) = None,
        hutchinson_sample_count: int = HUTCHINSON_DATA_DIM_THRESHOLD,
        chunk_size: int = 128,
        seed: int = 42,
        verbose: bool = False,
        **score_kwargs,
    ):
        """
        Computes Laplacian (trace of Jacobian of the score) via JVP-based trace estimator.

        Note:
        Here the "Hessian of log_prob" equals the Jacobian of the score function.
        """
        score_fn = functools.partial(self.score, t=t, **score_kwargs)
        return compute_trace_of_jacobian(
            fn=score_fn,
            x=x,
            method=method,
            hutchinson_sample_count=hutchinson_sample_count,
            chunk_size=chunk_size,
            seed=seed,
            verbose=verbose,
        )

    @torch.no_grad()
    def log_prob(
        self,
        x: torch.Tensor,
        t: float = 1e-4,
        t_end: float = 1.0,
        steps: int = 1000,
        verbose: bool = False,
        drift_trace_kwargs: dict = None,
        laplacian_kwargs: dict = None,
        shared_trace_kwargs: dict = None,
        **score_kwargs,
    ):
        """
        Computes log probability by solving the reverse-time ODE and integrating
        instantaneous change-of-variables.
        """
        assert t <= t_end, f"t should be less than t_end, got t={t} and t_end={t_end}"

        x = x.clone().detach()
        device = x.device
        batch_size = x.shape[0]

        if drift_trace_kwargs is None:
            drift_trace_kwargs = {}

        if laplacian_kwargs is None:
            laplacian_kwargs = {}

        if shared_trace_kwargs is None:
            shared_trace_kwargs = {}

        log_p = torch.zeros(batch_size, device=device, dtype=x.dtype)

        ts = batch_linspace(t, t_end, steps=steps).to(device)
        delta_s = copy_tensor_or_create((t_end - t) / (steps - 1))
        rng = tqdm(ts, desc="Iterating the ODE") if verbose else ts

        for s in rng:
            trace_of_drift_derivative = self._trace_of_drift_derivative(
                x=x, t=s, **(shared_trace_kwargs | drift_trace_kwargs)
            )
            trace_of_score_derivative = self.laplacian(
                x=x, t=s, **(shared_trace_kwargs | laplacian_kwargs | score_kwargs)
            )
            log_p += delta_s * (
                trace_of_drift_derivative - 0.5 * self.diff(s) ** 2 * trace_of_score_derivative
            )
            x += delta_s * (
                self.drift(x, s) - 0.5 * self.diff(s) ** 2 * self.score(x, s, **score_kwargs)
            )

        log_p += self.prior_log_prob(x, t_end)
        return log_p

    def prior_log_prob(self, x: torch.Tensor, t_end: float = 1.0):
        """Prior is Gaussian with mean 0 and std sigma(t_end)."""
        sigma = self.sigma(t_end)
        ambient_dim = x.numel() // x.shape[0]
        log_normalizing_factor = 0.5 * ambient_dim * math.log(2 * math.pi * sigma**2)
        exponential_term = -0.5 * torch.sum(x * x, dim=tuple(range(1, x.dim()))) / sigma**2
        return exponential_term - log_normalizing_factor


class VpSde(Sde):
    """The variance-preserving SDE described by Song et al. (2020) in equation (11)."""

    def __init__(
        self,
        score_net: torch.nn.Module,
        beta_min: float = 0.1,
        beta_max: float = 20,
        t_max: float = 1.0,
    ):
        self.beta_min = torch.tensor(beta_min)
        self.beta_max = torch.tensor(beta_max)
        self.t_max = torch.tensor(t_max)
        super().__init__(score_net)

    def drift(self, x, t):
        """The drift coefficient f(x, t) of the forward SDE"""
        t = copy_tensor_or_create(t, device=x.device)
        if t.ndim > 0:
            new_dims = x.ndim - t.ndim
            t = t.reshape(x.shape[:1] + (1,) * new_dims)
        return -self.beta(t) * x / 2

    def diff(self, t):
        """The diffusion coefficient g(t) of the forward SDE"""
        return torch.sqrt(self.beta(t))

    def mu_scale(self, t_end, t_start=0.0):
        """Scaling factor for the mean of x(t_end) | x(t_start)."""
        return torch.exp(-self.beta_integral(t_start, t_end) / 2)

    def sigma(self, t_end, t_start=0):
        """The standard deviation of x(t_end) | x(t_start)"""
        return torch.sqrt(1.0 - torch.exp(-self.beta_integral(t_start, t_end)))

    def beta(self, t):
        return (self.beta_max - self.beta_min) * t / self.t_max + self.beta_min

    def beta_integral(self, t_start, t_end):
        """Integrate beta(t) from t_start to t_end"""
        if not hasattr(self, "beta_diff"):
            self.beta_diff = self.beta_max - self.beta_min
        t_diff = t_end - t_start
        return self.beta_diff / (2 * self.t_max) * (t_end**2 - t_start**2) + self.beta_min * t_diff

    def _trace_of_drift_derivative(self, x: torch.Tensor, t: float):
        """Analytical trace of drift derivative for VP-SDE."""
        ambient_dim = x.numel() // x.shape[0]
        batch_size = x.shape[0]
        return (
            -0.5
            * torch.ones(batch_size, device=x.device, dtype=x.dtype)
            * ambient_dim
            * self.beta(t)
        )

    def solve_forward_sde(self, x_start, t_end=1.0, t_start=0.0, return_eps=False):
        """Solve the SDE forward from time t_start to t_end"""
        t_start, t_end = self._match_timestep_shapes(t_start, t_end)
        t_start, t_end = t_start.to(x_start.device), t_end.to(x_start.device)
        assert torch.all(t_start <= t_end)

        mu_scale = self.mu_scale(t_start=t_start, t_end=t_end)
        sigma_end = self.sigma(t_start=t_start, t_end=t_end)
        eps = torch.randn_like(x_start)

        if mu_scale.ndim > 0:
            new_dims = x_start.ndim - mu_scale.ndim
            mu_scale = mu_scale.reshape(x_start.shape[:1] + (1,) * new_dims)
            sigma_end = sigma_end.reshape(x_start.shape[:1] + (1,) * new_dims)

        x_end = mu_scale * x_start + sigma_end * eps

        if return_eps:
            return x_end, eps
        else:
            return x_end


class VeSde(Sde):
    """The variance-exploding SDE described by Song et al. (2020) in equation (9)."""

    def __init__(
        self,
        score_net: torch.nn.Module,
        sigma_min: float = 0.01,
        sigma_max: float = 50.0,
        t_max: float = 1.0,
    ):
        self.sigma_min = torch.tensor(sigma_min)
        self.sigma_max = torch.tensor(sigma_max)
        self.t_max = t_max
        super().__init__(score_net)

    def drift(self, x, t):
        return torch.zeros_like(x)

    def diff(self, t):
        sigma = self.sigma(t)
        diff = sigma * torch.sqrt(2 * (torch.log(self.sigma_max) - torch.log(self.sigma_min)))
        return diff

    def sigma(self, t_end, t_start=0.0):
        if isinstance(t_start, numbers.Number) and t_start == 0:
            return self.sigma_min * (self.sigma_max / self.sigma_min) ** t_end
        else:
            return torch.sqrt(self.sigma(t_end) ** 2 - self.sigma(t_start) ** 2)

    def _trace_of_drift_derivative(self, x: torch.Tensor, t: float):
        batch_size = x.shape[0]
        return torch.zeros(batch_size, device=x.device, dtype=x.dtype)

    def solve_forward_sde__ORIGINAL(self, x_start, t_end=1.0, t_start=0.0, return_eps=False):
        """Original version kept for reference."""
        t_start, t_end = self._match_timestep_shapes(t_start, t_end)
        t_start, t_end = t_start.to(x_start.device), t_end.to(x_start.device)
        assert torch.all(t_start <= t_end)

        sigma_end = self.sigma(t_start=t_start, t_end=t_end)
        eps = torch.randn_like(x_start)

        if sigma_end.ndim > 0:
            sigma_end = sigma_end[..., None]

        x_end = sigma_end * torch.randn_like(x_start)

        if return_eps:
            return x_end, eps
        else:
            return x_end

    def solve_forward_sde(self, x_start, t_end=1.0, t_start=0.0, return_eps=False):
        """Solve the VE forward SDE analytically: x_end = x_start + sigma_inc * eps"""
        t_start, t_end = self._match_timestep_shapes(t_start, t_end)
        t_start, t_end = t_start.to(x_start.device), t_end.to(x_start.device)
        assert torch.all(t_start <= t_end)

        sigma_inc = self.sigma(t_start=t_start, t_end=t_end)
        eps = torch.randn_like(x_start)

        if sigma_inc.ndim > 0:
            sigma_inc = sigma_inc.reshape(x_start.shape[:1] + (1,) * (x_start.ndim - 1))

        x_end = x_start + sigma_inc * eps

        if return_eps:
            return x_end, eps
        else:
            return x_end
