"""
Hyperparameter optimization for S-EPGP via marginal log-likelihood.
"""

import torch
from torch import Tensor
from typing import Callable, Optional, Dict
import math


def optimize_mll(
    gp,
    Phi_fn: Callable,  # function that returns Phi given current kernel params
    y_train: Tensor,
    n_steps: int = 200,
    lr: float = 0.05,
    verbose: bool = True,
    print_every: int = 50,
) -> Dict[str, list]:
    """
    Optimize kernel hyperparameters by maximizing marginal log-likelihood.
    
    Args:
        gp: SEPGP instance
        Phi_fn: callable that returns Phi_train (may depend on learnable frequencies)
        y_train: (N,) observations
        n_steps: optimization steps
        lr: learning rate
        verbose: print progress
        print_every: print interval
    
    Returns:
        history: dict with 'mll', 'noise', 'sigma' lists
    """
    optimizer = torch.optim.Adam(gp.kernel.parameters(), lr=lr)
    history = {"mll": [], "noise": [], "sigma_mean": []}
    
    for step in range(n_steps):
        optimizer.zero_grad()
        
        Phi_train = Phi_fn()
        mll = gp.marginal_log_likelihood(Phi_train, y_train)
        loss = -mll  # minimize negative MLL
        
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            history["mll"].append(mll.item())
            history["noise"].append(gp.kernel.noise_var.item())
            history["sigma_mean"].append(gp.kernel.sigma2.mean().item())
        
        if verbose and (step % print_every == 0 or step == n_steps - 1):
            print(
                f"Step {step:4d} | MLL: {mll.item():10.2f} | "
                f"noise: {gp.kernel.noise_var.item():.2e} | "
                f"mean σ²: {gp.kernel.sigma2.mean().item():.4f}"
            )
    
    return history
