"""
Gaussian Process regression engine for S-EPGP.

Since the S-EPGP kernel is low-rank (k(x,x') = φ(x)^T Σ φ(x')),
we use the Woodbury identity for efficient O(Nr² + r³) computation
instead of the naive O(N³) Cholesky on the N×N kernel matrix.

This makes the method scalable to large datasets, which is critical
for 2D/3D problems.
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple, Optional
import math


class SEPGP(nn.Module):
    """
    S-EPGP Gaussian Process for scalar PDEs.
    
    Given:
        - Basis functions Φ(X) ∈ ℝ^{N×r} evaluated at training points
        - Observations y ∈ ℝ^N
        - Prior: c ~ N(0, Σ), Σ = diag(σ²)
        - Noise: y = Φc + ε, ε ~ N(0, σ_n² I)
    
    Posterior on c:
        c|y ~ N(μ_c, Σ_c)
        Σ_c = (Σ^{-1} + σ_n^{-2} Φ^T Φ)^{-1}
        μ_c = Σ_c σ_n^{-2} Φ^T y
    
    Prediction:
        f(x*) | y ~ N(φ(x*)^T μ_c,  φ(x*)^T Σ_c φ(x*))
    """
    
    def __init__(self, kernel: nn.Module, jitter: float = 1e-6):
        super().__init__()
        self.kernel = kernel
        self.jitter = jitter
        
        # Cached posterior (set after conditioning)
        self._mu_c: Optional[Tensor] = None
        self._Sigma_c: Optional[Tensor] = None
        self._Phi_train: Optional[Tensor] = None
    
    def condition(self, Phi_train: Tensor, y_train: Tensor) -> None:
        """
        Condition the GP on training data using Woodbury identity.
        
        Args:
            Phi_train: (N, r) basis functions at training points
            y_train: (N,) observations
        """
        self._Phi_train = Phi_train
        N, r = Phi_train.shape
        
        sigma2 = self.kernel.sigma2          # (r,)
        noise_var = self.kernel.noise_var     # scalar
        
        # Σ_c = (Σ^{-1} + σ_n^{-2} Φ^T Φ)^{-1}
        # This is r×r, much smaller than N×N when r << N
        Sigma_inv = torch.diag(1.0 / sigma2) + Phi_train.t() @ Phi_train / noise_var
        Sigma_inv = Sigma_inv + self.jitter * torch.eye(r, device=Phi_train.device)
        
        # Cholesky of Σ_c^{-1}
        L = torch.linalg.cholesky(Sigma_inv)
        
        # μ_c = Σ_c (σ_n^{-2} Φ^T y)
        rhs = Phi_train.t() @ y_train / noise_var  # (r,)
        mu_c = torch.cholesky_solve(rhs.unsqueeze(-1), L).squeeze(-1)  # (r,)
        
        # Store Σ_c = L^{-T} L^{-1} (via solve)
        Sigma_c = torch.cholesky_solve(torch.eye(r, device=L.device), L)
        
        self._mu_c = mu_c
        self._Sigma_c = Sigma_c
    
    def predict(
        self, Phi_test: Tensor, return_var: bool = True
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Predict at test points.
        
        Args:
            Phi_test: (N*, r) basis functions at test points
            return_var: whether to return predictive variance
        
        Returns:
            mean: (N*,) posterior mean
            var: (N*,) posterior variance (if return_var=True)
        """
        assert self._mu_c is not None, "Must call condition() first"
        
        mean = Phi_test @ self._mu_c  # (N*,)
        
        var = None
        if return_var:
            # Var = φ^T Σ_c φ + σ_n²
            # For each test point: v_i = φ_i^T Σ_c φ_i
            tmp = Phi_test @ self._Sigma_c  # (N*, r)
            var = (tmp * Phi_test).sum(dim=-1) + self.kernel.noise_var  # (N*,)
        
        return mean, var
    
    def marginal_log_likelihood(self, Phi_train: Tensor, y_train: Tensor) -> Tensor:
        """
        Log marginal likelihood for hyperparameter optimization.
        
        log p(y) = -½ y^T (K + σ²I)^{-1} y - ½ log|K + σ²I| - N/2 log(2π)
        
        Using Woodbury for efficient computation:
            K + σ²I = Φ Σ Φ^T + σ²I
        
        Matrix determinant lemma:
            |Φ Σ Φ^T + σ²I| = |Σ^{-1} + σ^{-2} Φ^T Φ| · |Σ| · |σ²I|
        
        Woodbury inverse for the quadratic form.
        """
        N, r = Phi_train.shape
        sigma2 = self.kernel.sigma2
        noise_var = self.kernel.noise_var
        
        # A = Σ^{-1} + σ_n^{-2} Φ^T Φ   (r×r)
        A = torch.diag(1.0 / sigma2) + Phi_train.t() @ Phi_train / noise_var
        A = A + self.jitter * torch.eye(r, device=Phi_train.device)
        
        L_A = torch.linalg.cholesky(A)
        
        # Quadratic form: y^T (K + σ²I)^{-1} y
        # = σ^{-2} (y^T y - y^T Φ A^{-1} Φ^T y / σ_n²)
        Phi_y = Phi_train.t() @ y_train  # (r,)
        alpha = torch.cholesky_solve(Phi_y.unsqueeze(-1), L_A).squeeze(-1)
        
        quad = (y_train @ y_train - Phi_y @ alpha / noise_var) / noise_var
        
        # Log determinant: log|K + σ²I|
        # = log|A| + log|Σ| + N log(σ²)
        # = 2 Σ log(L_A_ii) + Σ log(σ_j²) + N log(σ_n²)
        log_det_A = 2 * torch.log(torch.diag(L_A)).sum()
        log_det_Sigma = torch.log(sigma2).sum()
        log_det = log_det_A + log_det_Sigma + N * torch.log(noise_var)
        
        mll = -0.5 * (quad + log_det + N * math.log(2 * math.pi))
        return mll
    
    def get_coefficients(self) -> Tensor:
        """Return posterior mean coefficients μ_c."""
        assert self._mu_c is not None, "Must call condition() first"
        return self._mu_c.detach().clone()


class VectorSEPGP(nn.Module):
    """
    S-EPGP for vector-valued PDEs (e.g., Maxwell).
    
    Each output component shares the same frequencies but has
    different basis functions (via Noetherian multipliers).
    
    Stacks observations from all components and solves jointly.
    """
    
    def __init__(self, kernel: nn.Module, n_components: int = 3, jitter: float = 1e-6):
        super().__init__()
        self.kernel = kernel
        self.n_components = n_components
        self.jitter = jitter
        self._mu_c = None
        self._Sigma_c = None
    
    def condition(
        self,
        Phi_train: Tensor,  # (N, r, n_comp) from Maxwell kernel
        y_train: Tensor,    # (N, n_comp) observations of all components
        obs_mask: Optional[Tensor] = None,  # (N, n_comp) bool: which components observed
    ) -> None:
        """
        Condition on multi-component observations.
        
        We stack everything into a single regression problem:
        For Maxwell with components (E_x, E_y, B), if all observed at all points:
            Φ_stacked = [Φ[:,:,0]; Φ[:,:,1]; Φ[:,:,2]]  ∈ ℝ^{3N × r}
            y_stacked = [y[:,0]; y[:,1]; y[:,2]]            ∈ ℝ^{3N}
        
        (with appropriate masking for partial observations)
        """
        N, r, nc = Phi_train.shape
        
        if obs_mask is None:
            obs_mask = torch.ones(N, nc, dtype=torch.bool, device=Phi_train.device)
        
        # Stack observed data
        Phi_list = []
        y_list = []
        for c in range(nc):
            mask_c = obs_mask[:, c]
            if mask_c.any():
                Phi_list.append(Phi_train[mask_c, :, c])
                y_list.append(y_train[mask_c, c])
        
        Phi_stacked = torch.cat(Phi_list, dim=0)  # (N_total, r)
        y_stacked = torch.cat(y_list, dim=0)        # (N_total,)
        
        # Same Woodbury solve as scalar case
        N_total = Phi_stacked.shape[0]
        sigma2 = self.kernel.sigma2
        noise_var = self.kernel.noise_var
        
        Sigma_inv = (torch.diag(1.0 / sigma2) + 
                     Phi_stacked.t() @ Phi_stacked / noise_var +
                     self.jitter * torch.eye(r, device=Phi_stacked.device))
        
        L = torch.linalg.cholesky(Sigma_inv)
        rhs = Phi_stacked.t() @ y_stacked / noise_var
        
        self._mu_c = torch.cholesky_solve(rhs.unsqueeze(-1), L).squeeze(-1)
        self._Sigma_c = torch.cholesky_solve(
            torch.eye(r, device=L.device), L
        )
    
    def predict(
        self, Phi_test: Tensor, component: int, return_var: bool = True
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Predict a single component at test points.
        
        Args:
            Phi_test: (N*, r, n_comp) from kernel.basis()
            component: which component to predict (0, 1, 2)
        """
        assert self._mu_c is not None, "Must call condition() first"
        
        phi = Phi_test[:, :, component]  # (N*, r)
        mean = phi @ self._mu_c
        
        var = None
        if return_var:
            tmp = phi @ self._Sigma_c
            var = (tmp * phi).sum(dim=-1) + self.kernel.noise_var
        
        return mean, var
