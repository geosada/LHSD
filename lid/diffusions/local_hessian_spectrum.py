from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Literal

import torch
from einops import rearrange
from lid import ModelBasedLIDEstimator
from models.diffusions.sdes import Sde
# If your project uses 'UnpackBatch' like NB does, import it:
from data.transforms.unpack import UnpackBatch
# --- add near the top of the file ---
from dataclasses import dataclass

import math
@dataclass
class LHSEArtifact:
    x: torch.Tensor
    delta: float
    sigma2: float
    method: str


# ----------------------------
#  VP / VE sigma^2 schedules
# ----------------------------
class Sigma2Schedule:
    def sigma2(self, delta: float) -> float:
        raise NotImplementedError
    def sigma(self, delta: float) -> float:
        return math.sqrt(self.sigma2(delta))
    def tvec(self, B: int, delta: float, device, dtype):
        return torch.full((B,), float(delta), device=device, dtype=dtype)

class VPSchedule(Sigma2Schedule):
    def __init__(self, beta_min=0.1, beta_max=20.0):
        self.beta_min, self.beta_max = beta_min, beta_max
    def sigma2(self, delta: float) -> float:
        # matches your sigma2_of_t_vp
        t = torch.tensor(delta, dtype=torch.float32)
        a2 = torch.exp(-(self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * t**2))
        return float(1.0 - a2)  # σ²(t) = 1 - α²(t)
        # :contentReference[oaicite:1]{index=1}

class VESchedule(Sigma2Schedule):
    def __init__(self, sigma_min: float, sigma_max: float):
        self.sigma_min, self.sigma_max = sigma_min, sigma_max
    def sigma2(self, delta: float) -> float:
        # log-linear VE: σ(t) = σ_min * (σ_max/σ_min)^t; use σ²(t)
        sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** float(delta)
        return float(sigma * sigma)

# ----------------------------------------
#  Shared kernels  (HVP, 2-step mini-SLQ, and heads)
# ----------------------------------------
def hvp_logp_batch(sde: Sde, x: torch.Tensor, t: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    # exactly as in your code
    x = x.detach().requires_grad_(True)
    if t.ndim == 0:
        t = t.expand(x.shape[0]).to(x.device, x.dtype)
    s = sde.score(x, t)
    g = (s * v).sum(dim=-1)
    Hv, = torch.autograd.grad(g.sum(), x, retain_graph=False, create_graph=False)
    return Hv
# :contentReference[oaicite:2]{index=2}

def _rademacher_probes(x_batch: torch.Tensor, K: int):
    B, D = x_batch.shape
    V = torch.empty(B, K, D, device=x_batch.device, dtype=x_batch.dtype).bernoulli_(0.5).mul_(2).sub_(1)
    V = V / (V.norm(dim=-1, keepdim=True) + 1e-12)
    return V
# :contentReference[oaicite:3]{index=3}

def _general_slq_per_sample(sde: Sde, x_batch: torch.Tensor, t_vec: torch.Tensor, V: torch.Tensor, f, m: int):
    """
    Vectorized m-step Lanczos/SLQ -> returns E_v[e1^T f(T_v) e1] per sample.
    Uses torch.linalg.eigh for tridiagonal eigenvalue decomposition.
    """
    B, K, D = V.shape
    out = torch.zeros(B, device=x_batch.device, dtype=x_batch.dtype)

    # Memory efficient: Loop over probes K, vectorize over Batch B
    for k in range(K):
        # Lanczos iteration storage
        alphas = []
        betas = []
        
        q_prev = None
        q_curr = V[:, k, :]  # (B, D)

        # --- Lanczos Iterations ---
        for i in range(m):
            # A is negative Hessian in this context (matches original code)
            Aq = -hvp_logp_batch(sde, x_batch, t_vec, q_curr)
            
            # alpha_i = q_i^T A q_i
            alpha = (q_curr * Aq).sum(dim=1)  # (B,)
            alphas.append(alpha)

            if i < m - 1:
                # r = A q_i - alpha_i q_i - beta_{i-1} q_{i-1}
                r = Aq - alpha[:, None] * q_curr
                if q_prev is not None:
                    r = r - betas[-1][:, None] * q_prev
                
                # beta_i = ||r||
                beta = r.norm(dim=1) + 1e-12
                betas.append(beta)

                # q_{i+1} = r / beta_i
                q_prev = q_curr
                q_curr = r / beta[:, None]

        # --- Solve Tridiagonal Eigenvalue Problem ---
        # Construct T matrix of shape (B, m, m)
        T = torch.zeros(B, m, m, device=x_batch.device, dtype=x_batch.dtype)
        
        diags = torch.stack(alphas, dim=1)      # (B, m)
        off_diags = torch.stack(betas, dim=1)   # (B, m-1)

        T.diagonal(dim1=-2, dim2=-1).copy_(diags)
        T.diagonal(offset=1, dim1=-2, dim2=-1).copy_(off_diags)
        T.diagonal(offset=-1, dim1=-2, dim2=-1).copy_(off_diags)

        # Eigendecomposition (T is symmetric)
        # eigvals: (B, m), eigvecs: (B, m, m)
        eigvals, eigvecs = torch.linalg.eigh(T)

        # --- Quadrature Approximation ---
        # Weights: (e1^T u_j)^2 -> First component of each eigenvector squared
        weights = eigvecs[:, 0, :] ** 2  # (B, m)

        # Apply function f to eigenvalues
        f_evals = f(eigvals)  # (B, m)

        # Weighted sum: sum_j (weights_j * f(lambda_j))
        est = (weights * f_evals).sum(dim=1)
        out += est

    return out / K

def _minislq_2x2_per_sample(sde: Sde, x_batch: torch.Tensor, t_vec: torch.Tensor, V: torch.Tensor, f):
    """
    Vectorized 2-step Lanczos/SLQ → returns E_v[e1^T f(T_v) e1] per sample; caller rescales by D.
    """
    B, K, D = V.shape
    out = torch.zeros(B, device=x_batch.device, dtype=x_batch.dtype)
    for k in range(K):
        q  = V[:, k, :]
        Aq = -hvp_logp_batch(sde, x_batch, t_vec, q)
        alpha1 = (q * Aq).sum(dim=1)
        r1 = Aq - alpha1[:, None] * q
        beta1 = r1.norm(dim=1) + 1e-12
        q2 = r1 / beta1[:, None]

        Aq2 = -hvp_logp_batch(sde, x_batch, t_vec, q2)
        alpha2 = (q2 * Aq2).sum(dim=1)

        disc = torch.sqrt((alpha1 - alpha2)**2 + 4*beta1**2)
        lam1 = 0.5*((alpha1 + alpha2) - disc)
        lam2 = 0.5*((alpha1 + alpha2) + disc)

        w1 = (beta1**2) / (beta1**2 + (lam1 - alpha1)**2 + 1e-20)  # (u^T e1)^2
        w2 = 1.0 - w1

        out += w1 * f(lam1) + w2 * f(lam2)
    return out / K

def DoF_SNR_sum(sde, x_batch, t_scalar, sigma2, num_probe=4, lanczos_m=2):
    B, D = x_batch.shape
    t_vec = torch.full((B,), float(t_scalar), device=x_batch.device, dtype=x_batch.dtype)
    V = _rademacher_probes(x_batch, num_probe)
    f = lambda lam: 1.0 / (1.0 + sigma2 * lam.clamp_min(0))
    # Use generalized SLQ
    avg_v = _general_slq_per_sample(sde, x_batch, t_vec, V, f, m=lanczos_m)
    return D * avg_v

def SNR_PR(sde, x_batch, t_scalar, sigma2, num_probe=4, eps=1e-12, lanczos_m=2):
    B, D = x_batch.shape
    t_vec = torch.full((B,), float(t_scalar), device=x_batch.device, dtype=x_batch.dtype)
    V = _rademacher_probes(x_batch, num_probe)
    f1 = lambda lam: 1.0 / (1.0 + sigma2 * lam.clamp_min(0))
    f2 = lambda lam: (1.0 / (1.0 + sigma2 * lam.clamp_min(0)))**2
    avg1 = _general_slq_per_sample(sde, x_batch, t_vec, V, f1, m=lanczos_m)
    avg2 = _general_slq_per_sample(sde, x_batch, t_vec, V, f2, m=lanczos_m)
    T1, T2 = D * avg1, D * avg2
    return (T1*T1) / (T2 + eps)

def PR_local(sde, x_batch, t_scalar, eps_reg=1e-3, num_probe=6, lanczos_m=2):
    B, D = x_batch.shape
    t_vec = torch.full((B,), float(t_scalar), device=x_batch.device, dtype=x_batch.dtype)
    V = _rademacher_probes(x_batch, num_probe)
    g1 = lambda lam: 1.0 / (lam.clamp_min(0) + eps_reg)
    g2 = lambda lam: 1.0 / (lam.clamp_min(0) + eps_reg)**2
    avg1 = _general_slq_per_sample(sde, x_batch, t_vec, V, g1, m=lanczos_m)
    avg2 = _general_slq_per_sample(sde, x_batch, t_vec, V, g2, m=lanczos_m)
    T1, T2 = D * avg1, D * avg2
    return (T1*T1) / (T2 + 1e-12)

def Soft_DoF(sde, x_batch, t_scalar, kappa, p=4, num_probe=8, lanczos_m=2):
    B, D = x_batch.shape
    t_vec = torch.full((B,), float(t_scalar), device=x_batch.device, dtype=x_batch.dtype)
    V = _rademacher_probes(x_batch, num_probe)
    f = lambda lam: 1.0 / (1.0 + (lam.clamp_min(0) / (kappa + 1e-20))**p)
    avg = _general_slq_per_sample(sde, x_batch, t_vec, V, f, m=lanczos_m)
    return D * avg

def Logistic_DoF(sde, x_batch, t_scalar, kappa, width=None, num_probe=6, lanczos_m=2):
    B, D = x_batch.shape
    t_vec = torch.full((B,), float(t_scalar), device=x_batch.device, dtype=x_batch.dtype)
    V = _rademacher_probes(x_batch, num_probe)
    if width is None:
        width = max(float(kappa) * 0.12, 1e-8)
    def f(lam):
        lam_pos = lam.clamp_min(0)
        z = -(lam_pos - float(kappa)) / (float(width) + 1e-12)
        return torch.sigmoid(z)
    avg = _general_slq_per_sample(sde, x_batch, t_vec, V, f, m=lanczos_m)
    return D * avg

def pick_kappa_snr(sigma2, c=1.0):   # κ ≈ c/σ²
    return float(c) / float(sigma2)

def pick_kappa_logistic(sigma2, c=0.4, w_scale=0.12):
    kappa = float(c) / float(sigma2)
    width = max(kappa * float(w_scale), 1e-8)
    return kappa, width

MethodName = Literal["DoF", "SNR_PR", "PR_local", "Soft_DoF", "Logistic_DoF"]

class LocalHessianSpectralEstimator(ModelBasedLIDEstimator):
    """
    SLQ-based LID via Hessian spectrum summaries.
    """
    Artifact = LHSEArtifact

    def __init__(
        self,
        model: Sde,
        schedule: Sigma2Schedule,
        method: MethodName,
        ambient_dim: Optional[int] = None,
        device: Optional[torch.device] = None,
        unpack: Optional[UnpackBatch] = None,
        batch_size: int = 512,
        # --- Parameters for SLQ ---
        lanczos_m: int = 10,  # Default to 10 or whatever you prefer (was hardcoded 2)
        # --------------------------
        dof_num_probe: int = 4,
        snr_num_probe: int = 8,
        pr_num_probe:  int = 6,
        soft_c: float = 0.7, soft_p: int = 4, soft_num_probe: int = 8,
        log_c: float = 0.4, log_wscale: float = 0.12, log_num_probe: int = 6,
        eps_reg: float = 1e-31
    ):
        super().__init__(ambient_dim=ambient_dim, model=model, device=device, unpack=unpack)
        self.sde: Sde = self.model
        self.schedule = schedule
        self.method = method
        self.batch_size = batch_size
        self.lanczos_m = lanczos_m  # Save m
        self.dof_num_probe  = dof_num_probe
        self.snr_num_probe  = snr_num_probe
        self.pr_num_probe   = pr_num_probe
        self.soft_c         = soft_c
        self.soft_p         = soft_p
        self.soft_num_probe = soft_num_probe
        self.log_c          = log_c
        self.log_wscale     = log_wscale
        self.log_num_probe  = log_num_probe
        self.eps_reg        = eps_reg

    def _preprocess(
        self,
        x: torch.Tensor,
        delta: float = 1e-3,
        verbose: int = 0,
        **kwargs,
    ) -> "LocalHessianSpectralEstimator.Artifact":
        assert isinstance(x, torch.Tensor), "x should be a torch.Tensor"
        x = x.to(self.device)
        sigma2 = self.schedule.sigma2(delta)
        return LHSEArtifact(x=x, delta=float(delta), sigma2=float(sigma2), method=self.method)

    def compute_lid_from_artifact(
        self,
        lid_artifact: "LocalHessianSpectralEstimator.Artifact",
        num_probe: int = 6,
        soft_c: Optional[float] = None,
        soft_p: Optional[int] = None,
        log_c: Optional[float] = None,
        log_wscale: Optional[float] = None,
        kappa: Optional[float] = None,
        width: Optional[float] = None,
        eps_reg: Optional[float] = None,
        lanczos_m: Optional[int] = None, # Allow override per call
        **kwargs,
    ) -> torch.Tensor:
        """
        Consume the Artifact and run the SLQ head selected by `method`.
        Return shape: (B,)
        """
        x = lid_artifact.x
        delta = lid_artifact.delta
        sigma2 = lid_artifact.sigma2
        
        # Resolve parameters
        soft_c      = self.soft_c     if soft_c      is None else soft_c
        soft_p      = self.soft_p     if soft_p      is None else soft_p
        log_c       = self.log_c      if log_c       is None else log_c
        log_wscale  = self.log_wscale if log_wscale  is None else log_wscale
        eps_reg     = self.eps_reg    if eps_reg     is None else eps_reg
        m_steps     = self.lanczos_m  if lanczos_m   is None else lanczos_m

        outs = []

        for i in range(0, x.shape[0], self.batch_size):
            xb = x[i:i+self.batch_size]
        
            if self.method == "DoF":
                nprobe = num_probe or self.dof_num_probe
                out = DoF_SNR_sum(self.sde, xb, delta, sigma2, num_probe=nprobe, lanczos_m=m_steps)
        
            elif self.method == "SNR_PR":
                nprobe = num_probe or self.snr_num_probe
                out = SNR_PR(self.sde, xb, delta, sigma2, num_probe=nprobe, lanczos_m=m_steps)
        
            elif self.method == "PR_local":
                nprobe = num_probe or self.pr_num_probe
                out = PR_local(self.sde, xb, delta, eps_reg=eps_reg, num_probe=nprobe, lanczos_m=m_steps)
        
            elif self.method == "Soft_DoF":
                nprobe = num_probe or self.soft_num_probe
                kappa_eff = (float(kappa) if kappa is not None
                             else pick_kappa_snr(sigma2, c=float(soft_c)))
                out = Soft_DoF(self.sde, xb, delta, kappa_eff, p=int(soft_p), num_probe=nprobe, lanczos_m=m_steps)
        
            elif self.method == "Logistic_DoF":
                nprobe = num_probe or self.log_num_probe
                if kappa is None:
                    kappa_eff, width_eff = pick_kappa_logistic(sigma2, c=float(log_c), w_scale=float(log_wscale))
                else:
                    kappa_eff = float(kappa)
                    width_eff = (float(width) if width is not None
                                 else max(kappa_eff * float(log_wscale), 1e-8))
                out = Logistic_DoF(self.sde, xb, delta, kappa_eff, width=width_eff, num_probe=nprobe, lanczos_m=m_steps)
        
            else:
                raise ValueError(self.method)
        
            outs.append(out)
        
        lids = torch.cat(outs, dim=0).clamp_min(0)
        if self.ambient_dim is not None:
            lids = lids.clamp_max(self.ambient_dim)
        return lids.cpu()