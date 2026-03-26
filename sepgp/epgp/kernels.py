"""
S-EPGP kernel construction.

The S-EPGP kernel is:
    k(x, x') = φ(x)^H · Σ · φ(x')

where φ(x) are basis functions derived from the characteristic variety,
and Σ = diag(σ_j²) are trainable weights.

For real-valued solutions, we work with real basis functions obtained by
pairing conjugate complex exponentials into cos/sin pairs.
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple


class SEPGPKernel(nn.Module):
    """
    S-EPGP kernel for scalar PDEs.
    
    Given frequencies and a PDE type, constructs basis functions φ(x)
    and the kernel k(x,x') = φ(x)^T Σ φ(x').
    
    This is equivalent to a Bayesian linear regression model:
        f(x) = Σ_j c_j φ_j(x),  c_j ~ N(0, σ_j²)
    """
    
    def __init__(
        self,
        n_basis: int,
        log_sigma: Optional[Tensor] = None,
        log_noise: float = -4.0,
    ):
        super().__init__()
        self.n_basis = n_basis
        
        # Trainable frequency weights (log scale for positivity)
        if log_sigma is None:
            self.log_sigma = nn.Parameter(torch.zeros(n_basis))
        else:
            self.log_sigma = nn.Parameter(log_sigma)
        
        # Trainable noise variance
        self.log_noise = nn.Parameter(torch.tensor(log_noise))
    
    @property
    def sigma2(self) -> Tensor:
        """Frequency weight variances σ_j²."""
        return torch.exp(2 * self.log_sigma)
    
    @property
    def noise_var(self) -> Tensor:
        """Observation noise variance."""
        return torch.exp(2 * self.log_noise)
    
    def kernel_matrix(self, Phi1: Tensor, Phi2: Optional[Tensor] = None) -> Tensor:
        """
        Compute kernel matrix K_{ij} = φ(x_i)^T Σ φ(x_j).
        
        Args:
            Phi1: (N1, n_basis) basis function evaluations at first set of points
            Phi2: (N2, n_basis) basis function evaluations at second set of points
                  If None, uses Phi1 (for training kernel matrix).
        
        Returns:
            K: (N1, N2) kernel matrix
        """
        if Phi2 is None:
            Phi2 = Phi1
        
        # K = Phi1 @ diag(σ²) @ Phi2^T
        weighted = Phi1 * self.sigma2.unsqueeze(0)  # (N1, n_basis)
        K = weighted @ Phi2.t()  # (N1, N2)
        return K


class Heat1DKernel(SEPGPKernel):
    """
    S-EPGP kernel for 1D heat equation u_t = u_xx.
    
    Basis functions:
        φ_{2j-1}(x,t) = exp(-ω_j² t) cos(ω_j x)
        φ_{2j}(x,t)   = exp(-ω_j² t) sin(ω_j x)
    
    where ω_j are frequencies on the characteristic variety.
    """
    
    def __init__(
        self,
        omegas: Tensor,
        log_noise: float = -4.0,
        learnable_freqs: bool = False,
    ):
        n_freq = omegas.shape[0]
        n_basis = 2 * n_freq  # cos + sin for each frequency
        super().__init__(n_basis=n_basis, log_noise=log_noise)
        
        if learnable_freqs:
            self.omegas = nn.Parameter(omegas.clone())
        else:
            self.register_buffer("omegas", omegas.clone())
    
    def basis(self, x: Tensor, t: Tensor) -> Tensor:
        """
        Evaluate basis functions at points (x, t).
        
        Args:
            x: (N,) spatial coordinates
            t: (N,) temporal coordinates
        
        Returns:
            Phi: (N, 2*n_freq) basis matrix
        """
        # omegas: (r,)
        # Outer products: (N, r)
        ox = x.unsqueeze(-1) * self.omegas.unsqueeze(0)       # (N, r)
        decay = torch.exp(-(self.omegas ** 2).unsqueeze(0) * t.unsqueeze(-1))  # (N, r)
        
        cos_part = decay * torch.cos(ox)  # (N, r)
        sin_part = decay * torch.sin(ox)  # (N, r)
        
        Phi = torch.cat([cos_part, sin_part], dim=-1)  # (N, 2r)
        return Phi


class Heat2DKernel(SEPGPKernel):
    """
    S-EPGP kernel for 2D heat equation u_t = u_xx + u_yy.
    
    Basis functions for each (ω_x, ω_y) pair:
        exp(-(ω_x² + ω_y²)t) · {cos(ω_x x)cos(ω_y y), cos(ω_x x)sin(ω_y y),
                                   sin(ω_x x)cos(ω_y y), sin(ω_x x)sin(ω_y y)}
    
    Since we use full grid including negative frequencies, we can use complex form
    and take real parts, or just use the exp(i(ω_x x + ω_y y)) approach.
    
    Simplified: using complex exponentials and taking real/imag parts.
    """
    
    def __init__(
        self,
        omegas: Tensor,  # (M, 2) spatial frequency pairs
        log_noise: float = -4.0,
        learnable_freqs: bool = False,
    ):
        M = omegas.shape[0]
        n_basis = 2 * M  # real + imag for each frequency
        super().__init__(n_basis=n_basis, log_noise=log_noise)
        
        if learnable_freqs:
            self.omegas = nn.Parameter(omegas.clone())
        else:
            self.register_buffer("omegas", omegas.clone())
    
    def basis(self, x: Tensor, y: Tensor, t: Tensor) -> Tensor:
        """
        Args:
            x, y, t: each (N,)
        Returns:
            Phi: (N, 2M) basis matrix
        """
        # Spatial phase: ω_x * x + ω_y * y for each frequency
        phase = (x.unsqueeze(-1) * self.omegas[:, 0].unsqueeze(0) +
                 y.unsqueeze(-1) * self.omegas[:, 1].unsqueeze(0))  # (N, M)
        
        # Temporal decay: exp(-(ω_x² + ω_y²) t)
        decay_rate = (self.omegas ** 2).sum(dim=-1)  # (M,)
        decay = torch.exp(-decay_rate.unsqueeze(0) * t.unsqueeze(-1))  # (N, M)
        
        cos_part = decay * torch.cos(phase)  # (N, M)
        sin_part = decay * torch.sin(phase)  # (N, M)
        
        Phi = torch.cat([cos_part, sin_part], dim=-1)  # (N, 2M)
        return Phi


class Wave2DKernel(SEPGPKernel):
    """
    S-EPGP kernel for 2D wave equation u_tt = c²(u_xx + u_yy).
    
    Variety: z_t = ±c|z_spatial|.  Both branches → oscillatory in time.
    
    Basis: for each (ω_x, ω_y):
        ω_t = c√(ω_x² + ω_y²)
        cos(ω_t t) cos(ω_x x + ω_y y),  sin(ω_t t) cos(ω_x x + ω_y y),
        cos(ω_t t) sin(ω_x x + ω_y y),  sin(ω_t t) sin(ω_x x + ω_y y)
    
    = 4 basis functions per spatial frequency pair.
    """
    
    def __init__(
        self,
        omegas_space: Tensor,  # (M, 2)
        c: float = 1.0,
        log_noise: float = -4.0,
        learnable_freqs: bool = False,
    ):
        M = omegas_space.shape[0]
        n_basis = 4 * M  # 4 combinations of cos/sin in space × time
        super().__init__(n_basis=n_basis, log_noise=log_noise)
        
        self.c = c
        if learnable_freqs:
            self.omegas_space = nn.Parameter(omegas_space.clone())
        else:
            self.register_buffer("omegas_space", omegas_space.clone())
    
    @property
    def omegas_time(self) -> Tensor:
        return self.c * torch.sqrt((self.omegas_space ** 2).sum(dim=-1))
    
    def basis(self, x: Tensor, y: Tensor, t: Tensor) -> Tensor:
        """
        Args:
            x, y, t: each (N,)
        Returns:
            Phi: (N, 4M)
        """
        # Spatial phase
        phase_s = (x.unsqueeze(-1) * self.omegas_space[:, 0].unsqueeze(0) +
                   y.unsqueeze(-1) * self.omegas_space[:, 1].unsqueeze(0))  # (N, M)
        
        # Temporal phase
        ot = self.omegas_time  # (M,)
        phase_t = t.unsqueeze(-1) * ot.unsqueeze(0)  # (N, M)
        
        cs = torch.cos(phase_s)  # (N, M)
        ss = torch.sin(phase_s)
        ct = torch.cos(phase_t)
        st = torch.sin(phase_t)
        
        # 4 combinations
        Phi = torch.cat([ct * cs, st * cs, ct * ss, st * ss], dim=-1)  # (N, 4M)
        return Phi


class Maxwell2DKernel(nn.Module):
    """
    S-EPGP kernel for 2D Maxwell equations.
    
    System: (E_x, E_y, B)(x, y, t) with 3 coupled equations.
    Same characteristic variety as 2D wave (cone), but with
    NONTRIVIAL Noetherian multipliers.
    
    For each frequency z on the cone, the basis function is a 3-vector:
        φ_j(x,y,t) = B(z_j) · exp(i(ω_x x + ω_y y ± ω_t t))
    
    where B(z_j) is the null vector of A(z_j).
    """
    
    def __init__(
        self,
        omegas_space: Tensor,    # (M, 2)
        multipliers: Tensor,     # (M, 3) Noetherian multipliers
        c: float = 1.0,
        log_noise: float = -4.0,
    ):
        super().__init__()
        M = omegas_space.shape[0]
        # 4 trig combos × M frequencies, but now each is a 3-vector
        n_scalar_basis = 4 * M
        
        self.c = c
        self.register_buffer("omegas_space", omegas_space)
        self.register_buffer("multipliers", multipliers)  # (M, 3)
        
        # Per-frequency weights
        self.log_sigma = nn.Parameter(torch.zeros(n_scalar_basis))
        self.log_noise = nn.Parameter(torch.tensor(log_noise))
    
    @property
    def sigma2(self) -> Tensor:
        return torch.exp(2 * self.log_sigma)
    
    @property
    def noise_var(self) -> Tensor:
        return torch.exp(2 * self.log_noise)
    
    @property
    def omegas_time(self) -> Tensor:
        return self.c * torch.sqrt((self.omegas_space ** 2).sum(dim=-1))
    
    def basis(self, x: Tensor, y: Tensor, t: Tensor) -> Tensor:
        """
        Evaluate vector-valued basis functions.
        
        Returns:
            Phi: (N, 4M, 3) — each basis function is a 3-vector
        """
        M = self.omegas_space.shape[0]
        N = x.shape[0]
        
        phase_s = (x.unsqueeze(-1) * self.omegas_space[:, 0].unsqueeze(0) +
                   y.unsqueeze(-1) * self.omegas_space[:, 1].unsqueeze(0))
        
        ot = self.omegas_time
        phase_t = t.unsqueeze(-1) * ot.unsqueeze(0)
        
        cs = torch.cos(phase_s)
        ss = torch.sin(phase_s)
        ct = torch.cos(phase_t)
        st = torch.sin(phase_t)
        
        # Scalar trig basis: (N, 4M)
        scalar_basis = torch.cat([ct * cs, st * cs, ct * ss, st * ss], dim=-1)
        
        # Multiply by Noetherian multipliers: each group of 4 uses same multiplier
        # multipliers: (M, 3) → repeat 4 times → (4M, 3)
        mult_expanded = self.multipliers.repeat(4, 1)  # (4M, 3)
        
        # Phi: (N, 4M, 3) = scalar_basis[:, :, None] * mult_expanded[None, :, :]
        Phi = scalar_basis.unsqueeze(-1) * mult_expanded.unsqueeze(0)
        
        return Phi
    
    def kernel_matrix_component(
        self, Phi1: Tensor, Phi2: Tensor, comp_i: int, comp_j: int
    ) -> Tensor:
        """
        Kernel matrix for components i, j of the vector field.
        
        K_{ij}(x, x') = Σ_h σ_h² φ_h^{(i)}(x) φ_h^{(j)}(x')
        
        Args:
            Phi1: (N1, 4M, 3) from self.basis
            Phi2: (N2, 4M, 3)
            comp_i, comp_j: component indices (0, 1, 2 for E_x, E_y, B)
        
        Returns:
            K: (N1, N2)
        """
        phi_i = Phi1[:, :, comp_i]  # (N1, 4M)
        phi_j = Phi2[:, :, comp_j]  # (N2, 4M)
        
        weighted = phi_i * self.sigma2.unsqueeze(0)  # (N1, 4M)
        K = weighted @ phi_j.t()  # (N1, N2)
        return K
