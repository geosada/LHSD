"""
image_ops.py
------------
Shared utilities for image preprocessing and spectral embedding.

Includes:
    • antialiased_downsample(imgs, target_hw, sigma)
    • dct2 / idct2  –  Orthonormal DCT-II and its inverse
    • spectral_embed_lowfreq(imgs_lo, to_hw)
"""

import math
import torch
import torch.nn.functional as F

# ======================================
# Anti-aliased Gaussian downsampling
# ======================================

def _gaussian_kernel_2d(ks: int, sigma: float, device, dtype):
    """Build a 2D Gaussian kernel normalized to sum to 1."""
    ax = torch.arange(ks, device=device, dtype=dtype) - (ks - 1) / 2
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    k = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    k = k / k.sum()
    return k


def antialiased_downsample(imgs: torch.Tensor, target_hw: int, sigma: float | None = None) -> torch.Tensor:
    """
    Downsample with Gaussian pre-blur to avoid aliasing.

    Args:
        imgs: (N, C, H, W) tensor in [0,1], square images.
        target_hw: target height=width, e.g. 16 or 8.
        sigma: pre-blur std in pixels. If None, uses Nyquist-aware default.

    Returns:
        (N, C, target_hw, target_hw)
    """
    assert imgs.ndim == 4 and imgs.shape[2] == imgs.shape[3], "Expect NCHW square images"
    N, C, H, W = imgs.shape
    assert target_hw <= H, "Downsample only (target <= source)."
    device, dtype = imgs.device, imgs.dtype

    scale = target_hw / H
    if sigma is None:
        sigma = max(0.8 / scale, 0.8)

    ks = int(2 * math.ceil(3 * sigma) + 1)
    k2d = _gaussian_kernel_2d(ks, sigma, device, dtype)
    k = k2d.view(1, 1, ks, ks).repeat(C, 1, 1, 1)
    x = F.conv2d(imgs, k, padding=ks // 2, groups=C)
    x = F.interpolate(x, size=(target_hw, target_hw), mode="bilinear", align_corners=False)
    return x.clamp_(0.0, 1.0)


# ======================================
# Orthonormal DCT utilities
# ======================================

def _dct_matrix(n: int, device, dtype):
    """Return orthonormal DCT-II matrix C (n×n)."""
    k = torch.arange(n, device=device, dtype=dtype).view(-1, 1)
    j = torch.arange(n, device=device, dtype=dtype).view(1, -1)
    C = torch.cos(math.pi / n * (k + 0.5) * j)
    C[:, 0] = C[:, 0] / math.sqrt(2.0)
    C = C * math.sqrt(2.0 / n)
    return C


def dct2(imgs: torch.Tensor):
    """2D orthonormal DCT-II applied per channel."""
    N, C, H, W = imgs.shape
    assert H == W, "Square only."
    device, dtype = imgs.device, imgs.dtype
    Cmat = _dct_matrix(H, device, dtype)
    x = imgs
    x = torch.einsum("ij,ncjk->ncik", Cmat, x)
    x = torch.einsum("ij,ncki->nckj", Cmat, x)
    return x


def idct2(coeffs: torch.Tensor):
    """Inverse of dct2 (2D orthonormal DCT-II)."""
    N, C, H, W = coeffs.shape
    assert H == W, "Square only."
    device, dtype = coeffs.device, coeffs.dtype
    Cmat = _dct_matrix(H, device, dtype)
    x = coeffs
    x = torch.einsum("ji,ncjk->ncik", Cmat, x)
    x = torch.einsum("ji,ncki->nckj", Cmat, x)
    return x


def spectral_embed_lowfreq(imgs_lo: torch.Tensor, to_hw: int) -> torch.Tensor:
    """
    Embed low-res images into higher-res grid by zero-padding low-frequency DCT block.

    Args:
        imgs_lo: (N,C,h,h) in [0,1], low-resolution images.
        to_hw: target size >= h (e.g., 32).

    Returns:
        (N,C,to_hw,to_hw) images reconstructed from low-frequency content.
    """
    assert imgs_lo.ndim == 4 and imgs_lo.shape[2] == imgs_lo.shape[3], "Expect NCHW square images"
    h = imgs_lo.shape[-1]
    assert to_hw >= h, "Embedding requires to_hw >= h."
    device, dtype = imgs_lo.device, imgs_lo.dtype

    Xh = dct2(imgs_lo)
    N, C = Xh.shape[:2]
    XH = torch.zeros(N, C, to_hw, to_hw, device=device, dtype=dtype)
    XH[:, :, :h, :h] = Xh
    out = idct2(XH).clamp_(0.0, 1.0)
    return out
