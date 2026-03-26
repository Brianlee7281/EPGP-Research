"""
Experiment 1: S-EPGP on 1D Heat Equation
=========================================

Solve u_t = u_xx with initial condition g(x) = exp(-50(x-0.5)²) at t=0.

This reproduces the setup from Härkönen et al. (ICML 2023) Fig. 2
and matches Brian's original GUROP research task.

The analytical solution is:
    u(x,t) = 1/√(1+200t) · exp(-50(x-0.5)² / (1+200t))

(Gaussian that spreads over time via heat kernel convolution.)
"""

import sys
sys.path.insert(0, "/home/claude")

import torch
import matplotlib.pyplot as plt
import numpy as np
from epgp import sample_heat_1d, Heat1DKernel, SEPGP, optimize_mll


# ============================================================
# Ground truth: analytical solution
# ============================================================

def true_solution(x: torch.Tensor, t: torch.Tensor, a: float = 50.0, x0: float = 0.5):
    """
    Analytical solution to u_t = u_xx with u(x,0) = exp(-a(x-x0)^2).
    Via convolution with heat kernel: u(x,t) = 1/√(1+4at) exp(-a(x-x0)²/(1+4at))
    """
    factor = 1.0 / torch.sqrt(1.0 + 4 * a * t)
    exponent = -a * (x - x0) ** 2 / (1.0 + 4 * a * t)
    return factor * torch.exp(exponent)


# ============================================================
# Setup
# ============================================================

torch.manual_seed(42)
device = torch.device("cpu")

# Training data: sample from initial condition t=0 only
N_train = 30
x_train = torch.rand(N_train, device=device)  # x ∈ [0, 1]
t_train = torch.zeros(N_train, device=device)  # all at t=0
y_train = true_solution(x_train, t_train)

# Add small noise
noise_std = 0.01
y_train = y_train + noise_std * torch.randn_like(y_train)

# Test grid: full (x, t) domain
nx, nt = 100, 50
x_test_1d = torch.linspace(0, 1, nx, device=device)
t_test_1d = torch.linspace(0, 0.02, nt, device=device)
t_grid, x_grid = torch.meshgrid(t_test_1d, x_test_1d, indexing="ij")
x_test = x_grid.flatten()
t_test = t_grid.flatten()

# True solution on test grid
y_true = true_solution(x_test, t_test)


# ============================================================
# S-EPGP: Build and optimize
# ============================================================

n_freq = 40  # number of frequency pairs → 80 basis functions
omegas, _ = sample_heat_1d(n_freq, omega_max=30.0, mode="grid", device=device)

kernel = Heat1DKernel(omegas, log_noise=-4.0, learnable_freqs=False)
gp = SEPGP(kernel)

# Basis functions at training points
Phi_train = kernel.basis(x_train, t_train)

print("=" * 60)
print("S-EPGP on 1D Heat Equation")
print("=" * 60)
print(f"Training points: {N_train} (all at t=0)")
print(f"Frequencies: {n_freq} → {2*n_freq} basis functions")
print(f"Test grid: {nx} × {nt} = {nx*nt} points")
print()

# Optimize hyperparameters
print("Optimizing hyperparameters via marginal log-likelihood...")
history = optimize_mll(
    gp, 
    lambda: kernel.basis(x_train, t_train),
    y_train, 
    n_steps=300, 
    lr=0.05,
    print_every=100,
)

# Condition and predict
Phi_train = kernel.basis(x_train, t_train)
gp.condition(Phi_train, y_train)

Phi_test = kernel.basis(x_test, t_test)
y_pred, y_var = gp.predict(Phi_test)

# Metrics
mse = ((y_pred - y_true) ** 2).mean().item()
max_err = (y_pred - y_true).abs().max().item()
print(f"\nResults:")
print(f"  MSE: {mse:.2e}")
print(f"  Max error: {max_err:.2e}")


# ============================================================
# Visualization
# ============================================================

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Row 1: True vs Predicted vs Error
y_true_grid = y_true.reshape(nt, nx).detach().numpy()
y_pred_grid = y_pred.reshape(nt, nx).detach().numpy()
y_var_grid = y_var.reshape(nt, nx).detach().numpy()

extent = [0, 1, 0, 0.02]

im0 = axes[0, 0].imshow(y_true_grid, origin="lower", extent=extent, aspect="auto", cmap="viridis")
axes[0, 0].set_title("True Solution")
axes[0, 0].set_xlabel("x")
axes[0, 0].set_ylabel("t")
plt.colorbar(im0, ax=axes[0, 0])

im1 = axes[0, 1].imshow(y_pred_grid, origin="lower", extent=extent, aspect="auto", cmap="viridis")
axes[0, 1].set_title(f"S-EPGP Prediction (MSE={mse:.2e})")
axes[0, 1].set_xlabel("x")
axes[0, 1].set_ylabel("t")
plt.colorbar(im1, ax=axes[0, 1])

err_grid = np.abs(y_pred_grid - y_true_grid)
im2 = axes[0, 2].imshow(err_grid, origin="lower", extent=extent, aspect="auto", cmap="hot")
axes[0, 2].set_title("Absolute Error")
axes[0, 2].set_xlabel("x")
axes[0, 2].set_ylabel("t")
plt.colorbar(im2, ax=axes[0, 2])

# Scatter training points on prediction plot
axes[0, 1].scatter(x_train.numpy(), t_train.numpy(), c="red", s=20, marker="x", label="Training")
axes[0, 1].legend()

# Row 2: Slices at different times + MLL convergence
time_slices = [0.0, 0.005, 0.01, 0.02]
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

for i, ts in enumerate(time_slices):
    tidx = int(ts / 0.02 * (nt - 1))
    tidx = min(tidx, nt - 1)
    axes[1, 0].plot(
        x_test_1d.numpy(), y_true_grid[tidx], "--", color=colors[i], alpha=0.7
    )
    axes[1, 0].plot(
        x_test_1d.numpy(), y_pred_grid[tidx], "-", color=colors[i], label=f"t={ts}"
    )

axes[1, 0].scatter(x_train.numpy(), y_train.numpy(), c="red", s=20, marker="x", zorder=5)
axes[1, 0].set_title("Time slices (solid=pred, dashed=true)")
axes[1, 0].set_xlabel("x")
axes[1, 0].set_ylabel("u(x,t)")
axes[1, 0].legend()

# Uncertainty
axes[1, 1].plot(x_test_1d.numpy(), y_pred_grid[0], "b-", label="mean (t=0)")
std_0 = np.sqrt(y_var_grid[0])
axes[1, 1].fill_between(
    x_test_1d.numpy(),
    y_pred_grid[0] - 2 * std_0,
    y_pred_grid[0] + 2 * std_0,
    alpha=0.3, label="±2σ"
)
axes[1, 1].scatter(x_train.numpy(), y_train.numpy(), c="red", s=20, marker="x")
axes[1, 1].set_title("Uncertainty at t=0")
axes[1, 1].set_xlabel("x")
axes[1, 1].legend()

# MLL convergence
axes[1, 2].plot(history["mll"])
axes[1, 2].set_title("Marginal Log-Likelihood")
axes[1, 2].set_xlabel("Step")
axes[1, 2].set_ylabel("MLL")

plt.suptitle("S-EPGP: 1D Heat Equation (training data at t=0 only)", fontsize=14)
plt.tight_layout()
plt.savefig("/home/claude/heat_1d_results.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nSaved: heat_1d_results.png")
