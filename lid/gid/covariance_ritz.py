import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from ..base import LIDEstimator

# ---------- utilities

def _center(X: torch.Tensor) -> torch.Tensor:
    return X - X.mean(dim=0, keepdim=True)

def _matvec_cov(Xc: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    # C v = (1/(N-1)) Xc^T (Xc v)
    # Xc: (N,D), v: (D,) or (B,D)
    if v.ndim == 1:
        xv = Xc @ v
        return (Xc.t() @ xv) / (Xc.shape[0] - 1)
    else:
        # batched: v (B,D)
        xv = Xc @ v.t()                  # (N,B)
        Cv = (Xc.t() @ xv) / (Xc.shape[0] - 1)  # (D,B)
        return Cv.t()                    # (B,D)

def _hutchinson_trace(matvec, D: int, nv: int, device, dtype) -> float:
    # E[z^T M z] = tr(M), with z Rademacher / unit expected
    zs = torch.empty(nv, D, device=device, dtype=dtype).bernoulli_(0.5).mul_(2).sub_(1)
    zs = F.normalize(zs, dim=1)  # stabilize
    Mz = matvec(zs)              # (nv,D)
    est = (zs * Mz).sum(dim=1).mean()
    return float(est)

def _power_lambda_max(matvec, D: int, iters: int, device, dtype) -> float:
    # crude λ_max via power iteration (cheap & good enough to scale Chebyshev)
    v = torch.randn(D, device=device, dtype=dtype)
    v = v / (v.norm() + 1e-12)
    lam = 0.0
    for _ in range(iters):
        w = matvec(v)                  # Cv
        n = w.norm() + 1e-20
        v = w / n
        lam = float((v * matvec(v)).sum())  # Rayleigh
    return max(lam, 1e-12)

def _chebyshev_projector_trace(
    matvec,
    D: int,
    a: float, b: float,
    lmin: float, lmax: float,
    degree: int,
    nv: int,
    device,
    dtype,
    jackson: bool = True,
) -> float:
    """
    Approximate tr P_{[a,b]}(C) ≈ tr[ sum_{j=0}^p g_j^p γ_j T_j(l(C)) ],
    where l(t) maps [lmin,lmax] -> [-1,1], T_j are Chebyshev polynomials,
    γ_j are Fourier-like coefficients of the box on [a,b] in the mapped domain,
    and g_j^p are Jackson smoothing coefficients.

    Returns an estimate of the eigenvalue count in [a,b].
    """
    # 1) Linear map l(t) from [lmin, lmax] to [-1,1]
    #    x = l(t) = (t - m)/r, with m=(lmax+lmin)/2, r=(lmax-lmin)/2
    m = 0.5 * (lmax + lmin)
    r = 0.5 * (lmax - lmin + 1e-12)

    def l_of_mat(vec):  # apply l(C) to vec: l(C) v = (C v - m v)/r
        return (matvec(vec) - m * vec) / r

    # 2) Chebyshev coefficients γ_j for indicator on [a,b] mapped to [-1,1]
    #    Following paper’s eqs: γ_0 = (1/π)(arccos(a')-arccos(b')),
    #    γ_j = (2/π) * (sin(j arccos(a')) - sin(j arccos(b'))) / j, j>0
    def _clip01(x):  # safety for numeric drift
        return max(min(x, 1.0), -1.0)
    ap = _clip01((a - m) / r)
    bp = _clip01((b - m) / r)
    ac, bc = math.acos(ap), math.acos(bp)
    gammas = [ (ac - bc) / math.pi ]  # γ_0
    for j in range(1, degree + 1):
        gam = (2.0 / math.pi) * (math.sin(j * ac) - math.sin(j * bc)) / j
        gammas.append(gam)

    # 3) Jackson coefficients g_j^p (paper’s eq. (3))
    #    g_j^p = sin((j+1)α_p)/((p+2) sin α_p) + (1 - (j+1)/(p+2)) cos(j α_p), α_p=π/(p+2)
    if jackson:
        p = degree
        ap_j = math.pi / (p + 2.0)
        denom = (p + 2.0) * math.sin(ap_j) + 1e-20
        gj = [ (math.sin((j + 1) * ap_j) / (p + 2.0) / math.sin(ap_j) +
                (1.0 - (j + 1.0)/(p + 2.0)) * math.cos(j * ap_j))
               for j in range(0, degree + 1) ]
    else:
        gj = [1.0] * (degree + 1)

    # 4) Hutchinson for tr[ sum_j g_j γ_j T_j(l(C)) ]
    #    Compute z^T T_j(l(C)) z via Chebyshev recurrence in operator form
    zs = torch.empty(nv, D, device=device, dtype=dtype).bernoulli_(0.5).mul_(2).sub_(1)
    zs = F.normalize(zs, dim=1)

    # T0(l(C)) z = z
    Tprev = zs.clone()                     # T_0
    contrib = gj[0] * gammas[0] * (zs * Tprev).sum(dim=1)  # z^T T0 z = ||z||^2 = 1 after norm

    if degree >= 1:
        # T1(l(C)) z = l(C) z
        Tcurr = l_of_mat(zs)
        contrib = contrib + gj[1] * gammas[1] * (zs * Tcurr).sum(dim=1)
        for j in range(2, degree + 1):
            # T_j = 2 l(C) T_{j-1} - T_{j-2}
            Tnext = 2.0 * l_of_mat(Tcurr) - Tprev
            contrib = contrib + gj[j] * gammas[j] * (zs * Tnext).sum(dim=1)
            Tprev, Tcurr = Tcurr, Tnext

    # Average over probes
    tr_est = contrib.mean().item()
    # This is tr(P_[a,b]) approximately = eigenvalue count in [a,b]
    return max(tr_est, 0.0)

# ---------- the estimator

class CovarianceRitzChebyshevEstimator(LIDEstimator):
    """
    Implements Ozçoban–Manguoğlu–Yetkin (arXiv:2503.09485) style ID:
      - Operator: global covariance C = (1/(N-1)) X_c^T X_c
      - Only matvecs with C
      - Hutchinson for tr(C)
      - Chebyshev/Jackson approximation to spectral projectors to count
        eigenvalues in intervals; accumulate variance until threshold.

    Usage:
        est = CovarianceRitzChebyshevEstimator(data=X, device=torch.device('cuda'))
        est.fit()  # optional (just centers)
        d_hat = est.estimate_lid(X, var_fraction=0.95)
    """

    @dataclass
    class Artifact:
        Xc: torch.Tensor
        N: int
        D: int
        device: torch.device
        dtype: torch.dtype
        lam_min_est: float
        lam_max_est: float
        total_trace: float

    def fit(self):
        # nothing to train; data-driven
        return self

    # preprocess: center and precompute rough λ_max, trace
    def _preprocess(self, x: torch.Tensor, nv_trace: int = 64, iters_power: int = 20, lam_min_floor: float = 0.0) -> "CovarianceRitzChebyshevEstimator.Artifact":
        assert x.ndim == 2, "x must be (N,D)"
        x = x.to(self.device)
        Xc = _center(x)
        N, D = Xc.shape

        def matvec(v):
            return _matvec_cov(Xc, v)

        lam_max_est = _power_lambda_max(matvec, D, iters=iters_power, device=Xc.device, dtype=Xc.dtype) * 1.05  # safe margin
        lam_min_est = lam_min_floor  # covariance is PSD; floor at ~0
        total_trace = _hutchinson_trace(matvec, D, nv=nv_trace, device=Xc.device, dtype=Xc.dtype)

        return CovarianceRitzChebyshevEstimator.Artifact(
            Xc=Xc, N=N, D=D, device=Xc.device, dtype=Xc.dtype,
            lam_min_est=lam_min_est, lam_max_est=lam_max_est,
            total_trace=total_trace
        )

    # compute: interval counting + variance accumulation
    def compute_lid_from_artifact(
        self,
        lid_artifact: "CovarianceRitzChebyshevEstimator.Artifact",
        *,
        var_fraction: float = 0.95,
        num_intervals: int = 32,
        degree: int = 64,
        nv_proj: int = 64,
        jackson: bool = True,
        interval_kind: str = "log",   # "log" or "linear"
        eps_rel: float = 1e-6
    ):
        A = lid_artifact
        Xc, D = A.Xc, A.D
        device, dtype = A.device, A.dtype

        def matvec(v):
            return _matvec_cov(Xc, v)

        # Build intervals on [lam_min_est, lam_max_est]
        lmin, lmax = A.lam_min_est, A.lam_max_est
        # avoid degenerate range
        if not math.isfinite(lmax) or lmax <= 0:
            return torch.tensor([0.0], device=device, dtype=dtype).repeat(Xc.shape[0])

        if interval_kind == "log":
            lo = max(lmin, lmax * eps_rel)
            edges = torch.logspace(math.log10(lo + 1e-20), math.log10(lmax), steps=num_intervals+1, device=device, dtype=dtype)
            edges[0] = 0.0
        else:
            edges = torch.linspace(lmin, lmax, steps=num_intervals+1, device=device, dtype=dtype)

        # 1) Total variance (trace) already estimated
        trace_total = A.total_trace

        # 2) For each interval, estimate eigenvalue COUNT via Chebyshev projector trace
        counts = []
        masses = []   # variance mass in interval ≈ count * interval midpoint (rough; paper also aggregates by intervals)
        edges_cpu = edges.detach().cpu().tolist()
        for i in range(num_intervals):
            a, b = float(edges_cpu[i]), float(edges_cpu[i+1])
            if b <= a:
                counts.append(0.0); masses.append(0.0); continue
            cnt = _chebyshev_projector_trace(
                matvec, D, a, b, lmin, lmax,
                degree=degree, nv=nv_proj, device=device, dtype=dtype, jackson=jackson
            )
            counts.append(cnt)
            midpoint = 0.5 * (a + b)
            masses.append(cnt * midpoint)

        counts = torch.tensor(counts, device=device, dtype=dtype)
        masses = torch.tensor(masses, device=device, dtype=dtype)

        # 3) Accumulate until variance fraction is met
        # NOTE: The paper accumulates by intervals; here we follow that spirit.
        # Sort intervals by descending midpoint to mimic picking largest-eigenvalue bands first.
        midpoints = 0.5 * (edges[:-1] + edges[1:])
        order = torch.argsort(midpoints, descending=True)
        cum_mass = 0.0
        cum_count = 0.0
        target = var_fraction * trace_total
        for idx in order.tolist():
            cum_mass += float(masses[idx])
            cum_count += float(counts[idx])
            if cum_mass >= target:
                break

        # Return a single global ID (same for each row, to fit your batched interface)
        d_hat = max(min(cum_count, float(D)), 0.0)
        return torch.full((Xc.shape[0],), d_hat, device=device, dtype=dtype)
