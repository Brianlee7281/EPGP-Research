"""
Characteristic variety computation and frequency sampling for linear PDEs.

The characteristic variety V of a PDE A(∂)f = 0 is:
    V = { z ∈ ℂ^n : det A(z) = 0 }  (scalar case)
    V = { z ∈ ℂ^n : ker A(z) ≠ {0} } (system case)

For S-EPGP, we sample finitely many z_1, ..., z_r ∈ V and build
basis functions φ_j(x) = B(x, z_j) exp(z_j · x).
"""

import torch
from torch import Tensor
from typing import Tuple, Optional
import math


def sample_heat_1d(
    n_freq: int,
    omega_max: float = 10.0,
    mode: str = "grid",
    device: torch.device = torch.device("cpu"),
) -> Tuple[Tensor, Tensor]:
    """
    Sample frequencies on the characteristic variety of 1D heat equation u_t = u_xx.
    
    Variety: z_t = z_x^2.  For real solutions, take z_x = iω (purely imaginary),
    giving z_t = -ω^2 (real, negative).
    
    Returns real frequencies ω_1, ..., ω_r and corresponding decay rates ω_j^2.
    Basis functions are:
        exp(-ω_j^2 t) cos(ω_j x),   exp(-ω_j^2 t) sin(ω_j x)
    
    Args:
        n_freq: number of frequency pairs (total basis size = 2*n_freq)
        omega_max: maximum frequency magnitude
        mode: "grid" for uniform, "random" for random sampling
        device: torch device
    
    Returns:
        omegas: (n_freq,) real tensor of positive frequencies
        decay_rates: (n_freq,) tensor of ω_j^2
    """
    if mode == "grid":
        # Uniform grid on (0, omega_max], excluding 0 (constant handled by DC term if needed)
        omegas = torch.linspace(omega_max / n_freq, omega_max, n_freq, device=device)
    elif mode == "random":
        omegas = torch.rand(n_freq, device=device) * omega_max
        omegas = omegas.clamp(min=0.1)  # avoid near-zero frequencies
    elif mode == "learned":
        # Initialize for gradient-based optimization
        omegas = torch.linspace(omega_max / n_freq, omega_max, n_freq, device=device)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    decay_rates = omegas ** 2
    return omegas, decay_rates


def sample_heat_2d(
    n_freq_per_dim: int,
    omega_max: float = 10.0,
    mode: str = "grid",
    device: torch.device = torch.device("cpu"),
) -> Tuple[Tensor, Tensor]:
    """
    Sample frequencies for 2D heat equation u_t = u_xx + u_yy.
    
    Variety: z_t = z_x^2 + z_y^2.
    For real solutions: z_x = iω_x, z_y = iω_y → z_t = -(ω_x^2 + ω_y^2).
    
    Returns:
        omegas: (M, 2) tensor of (ω_x, ω_y) pairs, M = n_freq_per_dim^2
        decay_rates: (M,) tensor of ω_x^2 + ω_y^2
    """
    if mode == "grid":
        ox = torch.linspace(-omega_max, omega_max, n_freq_per_dim, device=device)
        oy = torch.linspace(-omega_max, omega_max, n_freq_per_dim, device=device)
        grid_x, grid_y = torch.meshgrid(ox, oy, indexing="ij")
        omegas = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)
    elif mode == "random":
        M = n_freq_per_dim ** 2
        omegas = (torch.rand(M, 2, device=device) - 0.5) * 2 * omega_max
    else:
        raise ValueError(f"Unknown mode: {mode}")

    decay_rates = (omegas ** 2).sum(dim=-1)
    return omegas, decay_rates


def sample_wave_2d(
    n_freq_per_dim: int,
    omega_max: float = 10.0,
    c: float = 1.0,
    mode: str = "grid",
    device: torch.device = torch.device("cpu"),
) -> Tuple[Tensor, Tensor]:
    """
    Sample frequencies for 2D wave equation u_tt = c^2 (u_xx + u_yy).
    
    Variety: z_t^2 = c^2(z_x^2 + z_y^2).  (a cone)
    For real solutions: z_x = iω_x, z_y = iω_y → z_t = ±i·c·√(ω_x^2+ω_y^2).
    
    Two branches (forward/backward in time), both purely oscillatory.
    Basis: cos(ω_t t) cos(ω_x x) cos(ω_y y), etc.
    
    Returns:
        omegas_space: (M, 2) spatial frequencies
        omegas_time: (M,) temporal frequencies (positive branch; include both ±)
    """
    if mode == "grid":
        ox = torch.linspace(-omega_max, omega_max, n_freq_per_dim, device=device)
        oy = torch.linspace(-omega_max, omega_max, n_freq_per_dim, device=device)
        grid_x, grid_y = torch.meshgrid(ox, oy, indexing="ij")
        omegas_space = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)
    elif mode == "random":
        M = n_freq_per_dim ** 2
        omegas_space = (torch.rand(M, 2, device=device) - 0.5) * 2 * omega_max
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Dispersion relation: ω_t = c * |ω_spatial|
    omegas_time = c * torch.sqrt((omegas_space ** 2).sum(dim=-1))
    return omegas_space, omegas_time


def sample_maxwell_2d(
    n_freq_per_dim: int,
    omega_max: float = 10.0,
    c: float = 1.0,
    mode: str = "grid",
    device: torch.device = torch.device("cpu"),
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Sample frequencies for 2D Maxwell equations (E_x, E_y, B)(x, y, t).
    
    The characteristic variety is the SAME cone as the 2D wave equation:
        z_t^2 = c^2(z_x^2 + z_y^2)
    
    But the Noetherian multipliers are NONTRIVIAL:
    For z = (iz_x, iz_y, iz_t) on the cone, the null vector of A(z) is:
        B(z) ∝ (z_y · z_t,  -z_x · z_t,  z_x^2 + z_y^2)^T
    
    Returns:
        omegas_space: (M, 2) spatial frequencies
        omegas_time: (M,) temporal frequencies  
        multipliers: (M, 3) Noetherian multiplier vectors (complex)
    """
    omegas_space, omegas_time = sample_wave_2d(
        n_freq_per_dim, omega_max, c, mode, device
    )
    
    # Noetherian multipliers for Maxwell
    # A(z) = [[z_t, 0, -z_y], [0, z_t, z_x], [-z_y, z_x, z_t]]
    # ker A(z) for z on cone: proportional to (z_y*z_t, -z_x*z_t, z_x^2+z_y^2)
    ox = omegas_space[:, 0]  # ω_x
    oy = omegas_space[:, 1]  # ω_y
    ot = omegas_time          # ω_t = c|ω_spatial|
    
    # Multiplier: (ω_y · ω_t, -ω_x · ω_t, ω_x² + ω_y²)
    m1 = oy * ot
    m2 = -ox * ot
    m3 = ox ** 2 + oy ** 2
    multipliers = torch.stack([m1, m2, m3], dim=-1)  # (M, 3)
    
    # Normalize
    norms = multipliers.norm(dim=-1, keepdim=True).clamp(min=1e-10)
    multipliers = multipliers / norms
    
    return omegas_space, omegas_time, multipliers
