"""
Vanilla EPGP: Ehrenpreis-Palamodov Gaussian Processes
=====================================================

Unlike S-EPGP (discrete sum over finitely many frequencies),
vanilla EPGP integrates a Gaussian measure over the entire
characteristic variety to get a CLOSED-FORM covariance kernel.

The key integral for 1D heat equation u_t = u_xx:
  
  Variety: z_t = z_x^2,  parametrize z_x = iω → z_t = -ω^2
  Basis: exp(-ω^2 t + iωx)
  Measure: ω ~ N(0, ℓ^2)  (Gaussian on the variety)

  k((x,t),(x',t')) = σ^2 ∫ exp(-ω^2 t + iωx) exp(-ω^2 t' - iωx') · p(ω) dω

  where p(ω) = 1/(√(2π)ℓ) exp(-ω^2/(2ℓ^2))

This is a Gaussian integral in ω with closed-form solution:

  k = σ^2 / (ℓ√(2a)) · exp(-Δx^2 / (4a))

  where a = t + t' + 1/(2ℓ^2),  Δx = x - x'.

Advantages over S-EPGP:
  - No frequency discretization error (exact)
  - Fewer hyperparameters (σ^2, ℓ, σ_noise vs r frequency weights)
  - Kernel is smooth, no Gibbs oscillation

Disadvantages:
  - Requires the integral to have a closed form
  - Only works for PDEs where the variety has nice geometry
  - Standard GP regression O(N^3) — no low-rank Woodbury trick
"""

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from typing import Tuple, Optional, Dict
import warnings


# ====================================================================
# Closed-Form EPGP Kernels
# ====================================================================

def kernel_heat_1d(
    x1: np.ndarray, t1: np.ndarray,
    x2: np.ndarray, t2: np.ndarray,
    sigma2: float, ell: float,
) -> np.ndarray:
    """
    EPGP kernel for 1D heat equation u_t = u_xx.
    
    Gaussian measure N(0, ℓ²) on the frequency ω parametrizing
    the variety z_t = z_x² via z_x = iω.
    
    Derivation:
        ∫ exp(-ω²t + iωx) exp(-ω²t' - iωx') · 1/(√(2π)ℓ) exp(-ω²/(2ℓ²)) dω
        
        = 1/(√(2π)ℓ) · ∫ exp(-ω²·a + iω·Δx) dω
        
        where a = t + t' + 1/(2ℓ²), Δx = x - x'
        
        = 1/(√(2π)ℓ) · √(π/a) · exp(-Δx²/(4a))
        
        = 1/(ℓ·√(2a)) · exp(-Δx²/(4a))
    
    Args:
        x1, t1: (N1,) first set of points
        x2, t2: (N2,) second set of points
        sigma2: signal variance σ²
        ell: length scale ℓ of Gaussian measure on variety
    
    Returns:
        K: (N1, N2) kernel matrix
    """
    # a_{ij} = t1_i + t2_j + 1/(2ℓ²)
    a = t1[:, None] + t2[None, :] + 1.0 / (2.0 * ell**2)  # (N1, N2)
    dx = x1[:, None] - x2[None, :]  # (N1, N2)
    
    K = sigma2 / (ell * np.sqrt(2.0 * a)) * np.exp(-dx**2 / (4.0 * a))
    return K


def kernel_heat_2d(
    x1: np.ndarray, y1: np.ndarray, t1: np.ndarray,
    x2: np.ndarray, y2: np.ndarray, t2: np.ndarray,
    sigma2: float, ell: float,
) -> np.ndarray:
    """
    EPGP kernel for 2D heat equation u_t = u_xx + u_yy.
    
    Product Gaussian measure N(0, ℓ²I) on (ω_x, ω_y).
    The integral factors into a product of two 1D integrals:
    
        k = σ² / (2ℓ²a) · exp(-(Δx² + Δy²) / (4a))
    
    where a = t + t' + 1/(2ℓ²).
    
    This is a 2D Gaussian in space whose width grows with t+t'
    — exactly capturing heat diffusion spreading.
    """
    a = t1[:, None] + t2[None, :] + 1.0 / (2.0 * ell**2)
    dx = x1[:, None] - x2[None, :]
    dy = y1[:, None] - y2[None, :]
    
    K = sigma2 / (2.0 * ell**2 * a) * np.exp(-(dx**2 + dy**2) / (4.0 * a))
    return K


def kernel_heat_1d_shifted(
    x1: np.ndarray, t1: np.ndarray,
    x2: np.ndarray, t2: np.ndarray,
    sigma2: float, ell: float, mu0: float,
) -> np.ndarray:
    """
    EPGP kernel with shifted Gaussian measure N(μ₀, ℓ²) on ω.
    
    Allows the measure to concentrate around a nonzero center frequency,
    useful if the solution has a dominant oscillation frequency.
    
    k = σ²/(ℓ√(2a)) · exp(-Δx²/(4a)) · exp(iμ₀Δx · ... )
    
    Full formula:
        a = t + t' + 1/(2ℓ²)
        b_R = -μ₀/ℓ²    (real part of linear coeff)
        b_I = Δx          (imag part)
        
        k = σ²/(ℓ√(2a)) · exp(-(b_R² + b_I²)/(4a) + b_R·μ₀/ℓ² ... )
    
    Simplification (completing the square properly):
        k = σ²/(ℓ√(2a)) · exp(-Δx²/(4a) + μ₀²/(2ℓ²) - μ₀²·a/... )
    
    Actually let me derive this cleanly. With measure N(μ₀, ℓ²):
    
    ∫ exp(-ω²(t+t') + iω·Δx) · 1/(√(2π)ℓ) exp(-(ω-μ₀)²/(2ℓ²)) dω
    
    Expand: -(ω-μ₀)²/(2ℓ²) = -ω²/(2ℓ²) + ωμ₀/ℓ² - μ₀²/(2ℓ²)
    
    Combined ω² coeff: -(t+t'+1/(2ℓ²)) = -a
    Combined ω coeff: iΔx + μ₀/ℓ²  (call this β, complex)
    Constant: -μ₀²/(2ℓ²)
    
    Integral = 1/(√(2π)ℓ) · exp(-μ₀²/(2ℓ²)) · √(π/a) · exp(β²/(4a))
    
    where β = μ₀/ℓ² + iΔx
    β² = μ₀²/ℓ⁴ + 2iμ₀Δx/ℓ² - Δx²
    
    Real part of the full exponent:
        -μ₀²/(2ℓ²) + Re[β²/(4a)]
        = -μ₀²/(2ℓ²) + (μ₀²/ℓ⁴ - Δx²)/(4a)
    
    Imaginary part (gives oscillation):
        Im[β²/(4a)] = 2μ₀Δx/(4aℓ²) = μ₀Δx/(2aℓ²)
    """
    a = t1[:, None] + t2[None, :] + 1.0 / (2.0 * ell**2)
    dx = x1[:, None] - x2[None, :]
    
    # β = μ₀/ℓ² + iΔx
    # β² = (μ₀/ℓ²)² - Δx² + 2i(μ₀/ℓ²)Δx
    beta_r = mu0 / ell**2
    beta_i = dx
    
    beta2_real = beta_r**2 - beta_i**2  # (μ₀/ℓ²)² - Δx²
    beta2_imag = 2 * beta_r * beta_i     # 2(μ₀/ℓ²)Δx
    
    log_prefactor = np.log(sigma2) - 0.5 * np.log(2.0 * a) - np.log(ell)
    
    exponent_real = -mu0**2 / (2 * ell**2) + beta2_real / (4 * a)
    exponent_imag = beta2_imag / (4 * a)
    
    # Take real part of the kernel (for real-valued solutions)
    K = np.exp(log_prefactor + exponent_real) * np.cos(exponent_imag)
    return K


# ====================================================================
# EPGP Gaussian Process (standard, not Woodbury)
# ====================================================================

class EPGP:
    """
    Vanilla EPGP with closed-form kernel.
    
    Standard GP regression: O(N³) for conditioning.
    Hyperparameters: σ² (signal), ℓ (length scale), σ_n² (noise).
    
    Much fewer parameters than S-EPGP (3 vs ~r+1), but:
    - Requires closed-form kernel (limits PDE applicability)
    - O(N³) instead of O(Nr²) for large N
    """
    
    def __init__(self, kernel_fn, jitter: float = 1e-6):
        """
        Args:
            kernel_fn: str, one of 'heat_1d', 'heat_2d', 'heat_1d_shifted'
        """
        self.kernel_fn_name = kernel_fn
        self.jitter = jitter
        
        # Hyperparameters (log scale)
        self.log_sigma2 = 0.0      # log(σ²)
        self.log_ell = 0.0         # log(ℓ)
        self.log_noise = -4.0      # log(σ_n²)
        self.mu0 = 0.0             # shift for shifted kernel
        
        # Cached posterior
        self._alpha = None
        self._L = None
        self._train_data = None
    
    @property
    def sigma2(self) -> float:
        return np.exp(self.log_sigma2)
    
    @property
    def ell(self) -> float:
        return np.exp(self.log_ell)
    
    @property
    def noise_var(self) -> float:
        return np.exp(self.log_noise)
    
    def _compute_kernel(self, data1, data2) -> np.ndarray:
        """Dispatch to the right kernel function."""
        if self.kernel_fn_name == 'heat_1d':
            x1, t1 = data1
            x2, t2 = data2
            return kernel_heat_1d(x1, t1, x2, t2, self.sigma2, self.ell)
        
        elif self.kernel_fn_name == 'heat_2d':
            x1, y1, t1 = data1
            x2, y2, t2 = data2
            return kernel_heat_2d(x1, y1, t1, x2, y2, t2, self.sigma2, self.ell)
        
        elif self.kernel_fn_name == 'heat_1d_shifted':
            x1, t1 = data1
            x2, t2 = data2
            return kernel_heat_1d_shifted(x1, t1, x2, t2, self.sigma2, self.ell, self.mu0)
        
        else:
            raise ValueError(f"Unknown kernel: {self.kernel_fn_name}")
    
    def condition(self, train_data: tuple, y: np.ndarray) -> None:
        """
        Condition on training data.
        
        Args:
            train_data: tuple of arrays, e.g. (x, t) for 1D heat
            y: (N,) observations
        """
        self._train_data = train_data
        self._y_train = y
        N = y.shape[0]
        
        K = self._compute_kernel(train_data, train_data)
        K += (self.noise_var + self.jitter) * np.eye(N)
        
        self._L, self._low = cho_factor(K)
        self._alpha = cho_solve((self._L, self._low), y)
    
    def predict(self, test_data: tuple) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict at test points.
        
        Returns:
            mean: (N*,) posterior mean
            var: (N*,) posterior variance
        """
        assert self._alpha is not None, "Call condition() first"
        
        K_star = self._compute_kernel(test_data, self._train_data)  # (N*, N)
        K_ss = self._compute_kernel(test_data, test_data)            # (N*, N*)
        
        mean = K_star @ self._alpha
        
        V = cho_solve((self._L, self._low), K_star.T)  # (N, N*)
        var = np.diag(K_ss) - np.sum(K_star.T * V, axis=0) + self.noise_var
        
        return mean, var
    
    def marginal_log_likelihood(self, train_data: tuple, y: np.ndarray) -> float:
        """
        log p(y|θ) = -½ y^T K^{-1} y - ½ log|K| - N/2 log(2π)
        """
        N = y.shape[0]
        K = self._compute_kernel(train_data, train_data)
        K += (self.noise_var + self.jitter) * np.eye(N)
        
        try:
            L, low = cho_factor(K)
        except np.linalg.LinAlgError:
            return -1e10
        
        alpha = cho_solve((L, low), y)
        
        quad = y @ alpha
        log_det = 2.0 * np.sum(np.log(np.diag(L)))
        
        return -0.5 * (quad + log_det + N * np.log(2 * np.pi))
    
    def _pack_params(self) -> np.ndarray:
        if self.kernel_fn_name == 'heat_1d_shifted':
            return np.array([self.log_sigma2, self.log_ell, self.log_noise, self.mu0])
        return np.array([self.log_sigma2, self.log_ell, self.log_noise])
    
    def _unpack_params(self, theta: np.ndarray) -> None:
        self.log_sigma2 = float(theta[0])
        self.log_ell = float(theta[1])
        self.log_noise = float(theta[2])
        if len(theta) > 3:
            self.mu0 = float(theta[3])
    
    def optimize(
        self, train_data: tuple, y: np.ndarray,
        n_restarts: int = 3, maxiter: int = 200, verbose: bool = True,
    ) -> Dict:
        """Optimize hyperparameters via L-BFGS-B on negative MLL."""
        
        def neg_mll(theta):
            self._unpack_params(theta)
            val = -self.marginal_log_likelihood(train_data, y)
            return val if np.isfinite(val) else 1e10
        
        best_val = np.inf
        best_theta = self._pack_params()
        
        for restart in range(n_restarts):
            theta0 = self._pack_params()
            if restart > 0:
                theta0 = theta0 + 0.3 * np.random.randn(len(theta0))
            
            res = minimize(neg_mll, theta0, method="L-BFGS-B",
                          options={"maxiter": maxiter, "disp": False})
            
            if res.fun < best_val:
                best_val = res.fun
                best_theta = res.x.copy()
        
        self._unpack_params(best_theta)
        
        if verbose:
            print(f"  Optimized MLL: {-best_val:.2f}")
            print(f"  σ² = {self.sigma2:.4f}, ℓ = {self.ell:.4f}, σ_n² = {self.noise_var:.2e}")
            if self.kernel_fn_name == 'heat_1d_shifted':
                print(f"  μ₀ = {self.mu0:.4f}")
        
        return {"mll": -best_val}
