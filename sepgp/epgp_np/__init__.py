"""
S-EPGP: Sparse Ehrenpreis-Palamodov Gaussian Processes
======================================================

NumPy/SciPy implementation of S-EPGP for solving linear PDEs
with constant coefficients from data.

Core algorithm:
    1. Compute characteristic variety V of the PDE
    2. Sample frequencies z_1, ..., z_r ∈ V
    3. Build basis Φ(x) from exp-polynomial functions on V
    4. GP prior: f(x) = C^T φ(x), C ~ N(0, Σ)
    5. Condition on data via Woodbury (O(Nr²) instead of O(N³))
    6. Optimize hyperparameters via marginal log-likelihood
"""

import numpy as np
from numpy import ndarray
from scipy.optimize import minimize
from scipy.linalg import cho_factor, cho_solve
from typing import Tuple, Optional, Dict, Callable
import warnings


# ====================================================================
# Characteristic Variety Sampling
# ====================================================================

def sample_variety_heat_1d(
    n_freq: int, omega_max: float = 10.0, mode: str = "grid"
) -> ndarray:
    """
    1D heat equation u_t = u_xx.
    Variety: z_t = z_x². Purely imaginary z_x = iω → z_t = -ω².
    Returns ω_1, ..., ω_r > 0.
    """
    if mode == "grid":
        return np.linspace(omega_max / n_freq, omega_max, n_freq)
    elif mode == "random":
        return np.sort(np.random.uniform(0.1, omega_max, n_freq))
    raise ValueError(f"Unknown mode: {mode}")


def sample_variety_heat_2d(
    n_per_dim: int, omega_max: float = 10.0
) -> ndarray:
    """
    2D heat equation u_t = u_xx + u_yy.
    Variety: z_t = z_x² + z_y². Returns (M, 2) array of (ω_x, ω_y).
    """
    ox = np.linspace(-omega_max, omega_max, n_per_dim)
    oy = np.linspace(-omega_max, omega_max, n_per_dim)
    gx, gy = np.meshgrid(ox, oy, indexing="ij")
    return np.column_stack([gx.ravel(), gy.ravel()])


def sample_variety_wave_2d(
    n_per_dim: int, omega_max: float = 10.0, c: float = 1.0
) -> Tuple[ndarray, ndarray]:
    """
    2D wave equation u_tt = c²(u_xx + u_yy).
    Variety: z_t² = c²(z_x² + z_y²)  — a cone.
    Returns (omegas_space (M,2), omegas_time (M,)).
    """
    omegas_space = sample_variety_heat_2d(n_per_dim, omega_max)
    omegas_time = c * np.sqrt((omegas_space ** 2).sum(axis=1))
    return omegas_space, omegas_time


def sample_variety_maxwell_2d(
    n_per_dim: int, omega_max: float = 10.0, c: float = 1.0
) -> Tuple[ndarray, ndarray, ndarray]:
    """
    2D Maxwell: same cone as wave, but with Noetherian multipliers.
    ker A(z) ∝ (ω_y·ω_t, -ω_x·ω_t, ω_x²+ω_y²)^T.
    Returns (omegas_space, omegas_time, multipliers (M,3)).
    """
    omegas_space, omegas_time = sample_variety_wave_2d(n_per_dim, omega_max, c)
    ox, oy, ot = omegas_space[:, 0], omegas_space[:, 1], omegas_time
    mult = np.column_stack([oy * ot, -ox * ot, ox**2 + oy**2])
    norms = np.linalg.norm(mult, axis=1, keepdims=True).clip(min=1e-10)
    return omegas_space, omegas_time, mult / norms


# ====================================================================
# Basis Function Construction
# ====================================================================

def basis_heat_1d(x: ndarray, t: ndarray, omegas: ndarray) -> ndarray:
    """
    Basis for 1D heat: φ_{2j-1} = e^{-ω²t} cos(ωx), φ_{2j} = e^{-ω²t} sin(ωx).
    
    Args:
        x: (N,) spatial coords
        t: (N,) temporal coords
        omegas: (r,) frequencies
    Returns:
        Phi: (N, 2r) basis matrix
    """
    phase = np.outer(x, omegas)                          # (N, r)
    decay = np.exp(-np.outer(t, omegas ** 2))             # (N, r)
    return np.hstack([decay * np.cos(phase), decay * np.sin(phase)])


def basis_heat_2d(
    x: ndarray, y: ndarray, t: ndarray, omegas: ndarray
) -> ndarray:
    """
    Basis for 2D heat: exp(-(ωx²+ωy²)t) · {cos, sin}(ωx·x + ωy·y).
    
    Args:
        x, y, t: each (N,)
        omegas: (M, 2) frequency pairs
    Returns:
        Phi: (N, 2M)
    """
    phase = x[:, None] * omegas[:, 0] + y[:, None] * omegas[:, 1]  # (N, M)
    decay = np.exp(-np.outer(t, (omegas ** 2).sum(axis=1)))          # (N, M)
    return np.hstack([decay * np.cos(phase), decay * np.sin(phase)])


def basis_wave_2d(
    x: ndarray, y: ndarray, t: ndarray,
    omegas_space: ndarray, omegas_time: ndarray,
) -> ndarray:
    """
    Basis for 2D wave: {cos,sin}(ωt·t) × {cos,sin}(ωx·x+ωy·y).
    4 combinations per frequency → (N, 4M).
    """
    ps = x[:, None] * omegas_space[:, 0] + y[:, None] * omegas_space[:, 1]  # (N,M)
    pt = np.outer(t, omegas_time)  # (N,M)
    cs, ss = np.cos(ps), np.sin(ps)
    ct, st = np.cos(pt), np.sin(pt)
    return np.hstack([ct*cs, st*cs, ct*ss, st*ss])  # (N, 4M)


def basis_maxwell_2d(
    x: ndarray, y: ndarray, t: ndarray,
    omegas_space: ndarray, omegas_time: ndarray, multipliers: ndarray,
) -> ndarray:
    """
    Basis for 2D Maxwell: vector-valued.
    Returns (N, 4M, 3) — each basis function is a 3-vector.
    """
    ps = x[:, None] * omegas_space[:, 0] + y[:, None] * omegas_space[:, 1]
    pt = np.outer(t, omegas_time)
    cs, ss = np.cos(ps), np.sin(ps)
    ct, st = np.cos(pt), np.sin(pt)
    scalar = np.hstack([ct*cs, st*cs, ct*ss, st*ss])  # (N, 4M)
    # Each group of M uses same multiplier, repeated 4 times
    mult_exp = np.tile(multipliers, (4, 1))  # (4M, 3)
    return scalar[:, :, None] * mult_exp[None, :, :]  # (N, 4M, 3)


# ====================================================================
# GP Engine (Woodbury-based)
# ====================================================================

class SEPGP:
    """
    S-EPGP Gaussian Process with efficient Woodbury computation.
    
    Parameters θ = (log_sigma (r,), log_noise (scalar)):
        σ_j² = exp(2·log_sigma_j)     — frequency weights
        σ_n² = exp(2·log_noise)        — observation noise
    """
    
    def __init__(self, n_basis: int, jitter: float = 1e-6):
        self.n_basis = n_basis
        self.jitter = jitter
        
        # Hyperparameters (log scale)
        self.log_sigma = np.zeros(n_basis)
        self.log_noise = -4.0
        
        # Posterior (set after conditioning)
        self._mu_c = None
        self._Sigma_c = None
    
    @property
    def sigma2(self) -> ndarray:
        return np.exp(2 * self.log_sigma)
    
    @property
    def noise_var(self) -> float:
        return np.exp(2 * self.log_noise)
    
    def _pack_params(self) -> ndarray:
        return np.concatenate([self.log_sigma, [self.log_noise]])
    
    def _unpack_params(self, theta: ndarray) -> None:
        self.log_sigma = theta[:-1].copy()
        self.log_noise = float(theta[-1])
    
    def condition(self, Phi: ndarray, y: ndarray) -> None:
        """
        Condition on data. Phi: (N, r), y: (N,).
        Uses Woodbury: O(Nr² + r³) instead of O(N³).
        """
        r = Phi.shape[1]
        s2 = self.sigma2
        nv = self.noise_var
        
        # A = Σ^{-1} + σ_n^{-2} Φ^T Φ
        A = np.diag(1.0 / s2) + Phi.T @ Phi / nv
        A += self.jitter * np.eye(r)
        
        L, low = cho_factor(A)
        
        rhs = Phi.T @ y / nv
        self._mu_c = cho_solve((L, low), rhs)
        self._Sigma_c = cho_solve((L, low), np.eye(r))
    
    def predict(self, Phi_test: ndarray) -> Tuple[ndarray, ndarray]:
        """
        Returns (mean, variance) at test points.
        """
        assert self._mu_c is not None, "Call condition() first"
        mean = Phi_test @ self._mu_c
        tmp = Phi_test @ self._Sigma_c
        var = np.sum(tmp * Phi_test, axis=1) + self.noise_var
        return mean, var
    
    def marginal_log_likelihood(self, Phi: ndarray, y: ndarray) -> float:
        """
        log p(y|θ) using Woodbury identity.
        
        = -½ y^T(K+σ²I)^{-1}y - ½ log|K+σ²I| - N/2 log(2π)
        
        where K = Φ Σ Φ^T, computed efficiently via matrix determinant lemma.
        """
        N, r = Phi.shape
        s2 = self.sigma2
        nv = self.noise_var
        
        # A = Σ^{-1} + σ_n^{-2} Φ^T Φ
        A = np.diag(1.0 / s2) + Phi.T @ Phi / nv
        A += self.jitter * np.eye(r)
        
        try:
            L_A = np.linalg.cholesky(A)
        except np.linalg.LinAlgError:
            return -1e10  # return bad value if not PD
        
        # Quadratic form
        Phi_y = Phi.T @ y
        alpha = cho_solve(cho_factor(A), Phi_y)
        quad = (y @ y - Phi_y @ alpha / nv) / nv
        
        # Log determinant: log|K+σ²I| = log|A| + log|Σ| + N·log(σ²)
        log_det_A = 2 * np.sum(np.log(np.diag(L_A)))
        log_det_Sigma = np.sum(np.log(s2))
        log_det = log_det_A + log_det_Sigma + N * np.log(nv)
        
        return -0.5 * (quad + log_det + N * np.log(2 * np.pi))
    
    def optimize(
        self,
        Phi_fn: Callable[[], ndarray],
        y: ndarray,
        n_restarts: int = 1,
        maxiter: int = 200,
        verbose: bool = True,
    ) -> Dict[str, list]:
        """
        Optimize hyperparameters via L-BFGS-B on negative MLL.
        
        Args:
            Phi_fn: callable returning Phi (allows learnable frequencies)
            y: (N,) observations
        """
        def neg_mll(theta):
            self._unpack_params(theta)
            Phi = Phi_fn()
            val = -self.marginal_log_likelihood(Phi, y)
            if not np.isfinite(val):
                return 1e10
            return val
        
        best_val = np.inf
        best_theta = self._pack_params()
        
        for restart in range(n_restarts):
            theta0 = self._pack_params()
            if restart > 0:
                theta0 = theta0 + 0.5 * np.random.randn(len(theta0))
            
            res = minimize(neg_mll, theta0, method="L-BFGS-B", 
                          options={"maxiter": maxiter, "disp": verbose and restart == 0})
            
            if res.fun < best_val:
                best_val = res.fun
                best_theta = res.x.copy()
        
        self._unpack_params(best_theta)
        
        if verbose:
            print(f"  Optimized MLL: {-best_val:.2f}")
            print(f"  Noise σ²: {self.noise_var:.2e}")
            print(f"  Mean weight σ²: {self.sigma2.mean():.4f}")
        
        return {"mll": -best_val, "noise_var": self.noise_var}


# ====================================================================
# Black-Scholes Transforms
# ====================================================================

class BSParams:
    """Black-Scholes parameters."""
    def __init__(self, sigma: float, r: float, K: float, T: float):
        self.sigma = sigma
        self.r = r
        self.K = K
        self.T = T
    
    @property
    def k(self) -> float:
        return 2 * self.r / self.sigma**2
    
    @property
    def alpha(self) -> float:
        return -(self.k - 1) / 2
    
    @property
    def beta(self) -> float:
        return -((self.k + 1) / 2) ** 2


def bs_to_heat(S, t, V, p: BSParams):
    """(S, t, V) → (x, τ, v)."""
    x = np.log(S / p.K)
    tau = 0.5 * p.sigma**2 * (p.T - t)
    v = V / (p.K * np.exp(p.alpha * x + p.beta * tau))
    return x, tau, v


def heat_to_bs(x, tau, v, p: BSParams):
    """(x, τ, v) → (S, t, V)."""
    S = p.K * np.exp(x)
    t = p.T - 2 * tau / p.sigma**2
    V = p.K * np.exp(p.alpha * x + p.beta * tau) * v
    return S, t, V


def bs_call_price(S, t, p: BSParams):
    """Analytical European call price."""
    from scipy.stats import norm
    tau_bs = np.clip(p.T - t, 1e-10, None)
    d1 = (np.log(S / p.K) + (p.r + 0.5 * p.sigma**2) * tau_bs) / (p.sigma * np.sqrt(tau_bs))
    d2 = d1 - p.sigma * np.sqrt(tau_bs)
    return S * norm.cdf(d1) - p.K * np.exp(-p.r * tau_bs) * norm.cdf(d2)


def heat_ic_from_call(x, p: BSParams):
    """IC for heat eq from call payoff: v(x,0) = max(e^{(k+1)x/2} - e^{(k-1)x/2}, 0)."""
    return np.maximum(np.exp((p.k + 1) * x / 2) - np.exp((p.k - 1) * x / 2), 0.0)
