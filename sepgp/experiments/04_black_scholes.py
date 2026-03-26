"""
Experiment 4: Black-Scholes Pipeline
=====================================

1. Define BS parameters and compute true European call price
2. Transform to heat equation domain
3. Sample initial condition v(x, 0) at τ=0
4. Fit S-EPGP on heat equation
5. Transform predictions back to BS variables
6. Compare predicted vs true option prices

This is the second task from Brian's GUROP research with Prof. Raita.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch
import matplotlib.pyplot as plt
import numpy as np
from epgp import (
    sample_heat_1d, Heat1DKernel, SEPGP, optimize_mll,
    BSParams, bs_to_heat, heat_to_bs, bs_call_price, heat_ic_from_bs_call,
)


# ============================================================
# Black-Scholes parameters
# ============================================================

params = BSParams(
    sigma=0.3,   # 30% volatility
    r=0.05,      # 5% risk-free rate
    K=100.0,     # strike price
    T=1.0,       # 1 year to expiry
)

print("=" * 60)
print("Black-Scholes → Heat Equation → S-EPGP Pipeline")
print("=" * 60)
print(f"σ={params.sigma}, r={params.r}, K={params.K}, T={params.T}")
print(f"k = 2r/σ² = {params.k:.4f}")
print(f"α = {params.alpha:.4f}, β = {params.beta:.4f}")
print()


# ============================================================
# Step 1: True BS solution in original variables
# ============================================================

torch.manual_seed(42)

# Grid in BS domain
n_S = 200
n_t = 50
S_grid_1d = torch.linspace(50.0, 200.0, n_S)   # stock prices
t_grid_1d = torch.linspace(0.0, 0.99, n_t)      # times (avoid t=T exactly)

S_mesh, t_mesh = torch.meshgrid(S_grid_1d, t_grid_1d, indexing="ij")
S_flat = S_mesh.flatten()
t_flat = t_mesh.flatten()

V_true = bs_call_price(S_flat, t_flat, params)


# ============================================================
# Step 2: Transform to heat equation domain
# ============================================================

x_all, tau_all, v_all = bs_to_heat(S_flat, t_flat, V_true, params)

# Check transform roundtrip
S_back, t_back, V_back = heat_to_bs(x_all, tau_all, v_all, params)
roundtrip_err = (V_back - V_true).abs().max().item()
print(f"Transform roundtrip error: {roundtrip_err:.2e}")


# ============================================================
# Step 3: Training data — initial condition only (τ=0)
# ============================================================

# In heat domain, τ=0 corresponds to t=T (at expiry) in BS domain
# v(x, 0) = max(e^{(k+1)x/2} - e^{(k-1)x/2}, 0)

N_train = 50
x_train = torch.linspace(-2.0, 2.0, N_train)  # log-moneyness range
tau_train = torch.zeros(N_train)
v_train = heat_ic_from_bs_call(x_train, params)

# Add small noise
noise_std = 0.01 * v_train.abs().clamp(min=1e-6)
v_train_noisy = v_train + noise_std * torch.randn_like(v_train)

print(f"\nTraining: {N_train} points at τ=0 (BS expiry)")
print(f"x range: [{x_train.min():.1f}, {x_train.max():.1f}] (log-moneyness)")


# ============================================================
# Step 4: Fit S-EPGP on heat equation
# ============================================================

n_freq = 50
omegas, _ = sample_heat_1d(n_freq, omega_max=20.0, mode="grid")

kernel = Heat1DKernel(omegas, log_noise=-3.0, learnable_freqs=False)
gp = SEPGP(kernel)

print(f"\nS-EPGP: {n_freq} frequencies → {2*n_freq} basis functions")
print("Optimizing hyperparameters...")

history = optimize_mll(
    gp,
    lambda: kernel.basis(x_train, tau_train),
    v_train_noisy,
    n_steps=400,
    lr=0.03,
    print_every=100,
)

# Condition and predict on heat domain grid
Phi_train = kernel.basis(x_train, tau_train)
gp.condition(Phi_train, v_train_noisy)

# Test grid in heat domain
nx_test, ntau_test = 100, 50
x_test_1d = torch.linspace(-2.0, 2.0, nx_test)
tau_test_1d = torch.linspace(0.0, 0.5 * params.sigma**2 * params.T, ntau_test)

tau_grid, x_grid = torch.meshgrid(tau_test_1d, x_test_1d, indexing="ij")
x_test = x_grid.flatten()
tau_test = tau_grid.flatten()

Phi_test = kernel.basis(x_test, tau_test)
v_pred, v_var = gp.predict(Phi_test)


# ============================================================
# Step 5: Transform predictions back to BS domain
# ============================================================

S_pred, t_pred, V_pred = heat_to_bs(x_test, tau_test, v_pred, params)
_, _, V_true_test = heat_to_bs(x_test, tau_test, v_pred * 0, params)  # placeholder

# Compute true BS prices at the same (S, t) points
V_true_at_test = bs_call_price(S_pred.clamp(min=1e-6), t_pred.clamp(min=0, max=params.T - 1e-6), params)

# Mask out invalid regions (S ≤ 0, t outside [0, T])
valid = (S_pred > 1.0) & (t_pred > 0.0) & (t_pred < params.T - 0.01)
valid = valid & V_true_at_test.isfinite() & V_pred.isfinite()

mse_bs = ((V_pred[valid] - V_true_at_test[valid]) ** 2).mean().item()
max_err_bs = (V_pred[valid] - V_true_at_test[valid]).abs().max().item()
rel_err = ((V_pred[valid] - V_true_at_test[valid]).abs() / V_true_at_test[valid].clamp(min=1e-6)).mean().item()

print(f"\n{'='*60}")
print(f"Results in Black-Scholes domain:")
print(f"  MSE: {mse_bs:.4e}")
print(f"  Max absolute error: {max_err_bs:.4f}")
print(f"  Mean relative error: {rel_err:.4%}")


# ============================================================
# Visualization
# ============================================================

fig, axes = plt.subplots(2, 3, figsize=(18, 11))

# ---- Row 1: Heat equation domain ----

v_pred_grid = v_pred.reshape(ntau_test, nx_test).detach().numpy()
extent_heat = [-2, 2, 0, tau_test_1d[-1].item()]

axes[0, 0].imshow(v_pred_grid, origin="lower", extent=extent_heat, aspect="auto", cmap="viridis")
axes[0, 0].scatter(x_train.numpy(), tau_train.numpy(), c="red", s=20, marker="x")
axes[0, 0].set_title("S-EPGP prediction (heat domain)")
axes[0, 0].set_xlabel("x (log-moneyness)")
axes[0, 0].set_ylabel("τ")

# IC fit
axes[0, 1].plot(x_test_1d.numpy(), v_pred_grid[0], "b-", linewidth=2, label="S-EPGP")
v_true_ic = heat_ic_from_bs_call(x_test_1d, params).numpy()
axes[0, 1].plot(x_test_1d.numpy(), v_true_ic, "r--", linewidth=1.5, label="True IC")
axes[0, 1].scatter(x_train.numpy(), v_train_noisy.numpy(), c="red", s=15, marker="x", label="Training data")
axes[0, 1].set_title("Initial condition fit (τ=0)")
axes[0, 1].set_xlabel("x")
axes[0, 1].set_ylabel("v(x, 0)")
axes[0, 1].legend()

# MLL convergence
axes[0, 2].plot(history["mll"])
axes[0, 2].set_title("MLL convergence")
axes[0, 2].set_xlabel("Step")
axes[0, 2].set_ylabel("MLL")

# ---- Row 2: Black-Scholes domain ----

# Slice at a few times
t_slices_bs = [0.0, 0.25, 0.5, 0.75]
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
S_plot = torch.linspace(60, 180, 200)

for i, t_val in enumerate(t_slices_bs):
    V_true_slice = bs_call_price(S_plot, torch.full_like(S_plot, t_val), params)
    
    # Get S-EPGP prediction at this time
    x_plot, tau_plot, _ = bs_to_heat(S_plot, torch.full_like(S_plot, t_val), V_true_slice, params)
    Phi_plot = kernel.basis(x_plot, tau_plot)
    v_plot, _ = gp.predict(Phi_plot, return_var=False)
    _, _, V_plot = heat_to_bs(x_plot, tau_plot, v_plot, params)
    
    axes[1, 0].plot(S_plot.numpy(), V_true_slice.detach().numpy(), "--", color=colors[i], alpha=0.7)
    axes[1, 0].plot(S_plot.numpy(), V_plot.detach().numpy(), "-", color=colors[i], label=f"t={t_val}")

axes[1, 0].axvline(params.K, color="gray", linestyle=":", alpha=0.5, label="Strike")
axes[1, 0].set_title("BS option price (solid=pred, dashed=true)")
axes[1, 0].set_xlabel("Stock price S")
axes[1, 0].set_ylabel("Call price V")
axes[1, 0].legend()

# Error heatmap in BS domain
V_pred_grid = V_pred.reshape(ntau_test, nx_test).detach().numpy()
V_true_grid = V_true_at_test.reshape(ntau_test, nx_test).detach().numpy()
valid_grid = valid.reshape(ntau_test, nx_test).numpy()

err_grid = np.abs(V_pred_grid - V_true_grid)
err_grid[~valid_grid] = np.nan

S_grid_for_plot = S_pred.reshape(ntau_test, nx_test).detach().numpy()
t_grid_for_plot = t_pred.reshape(ntau_test, nx_test).detach().numpy()

axes[1, 1].scatter(
    S_grid_for_plot[valid_grid], t_grid_for_plot[valid_grid],
    c=err_grid[valid_grid], cmap="hot", s=1, vmin=0, vmax=5
)
axes[1, 1].set_title("Absolute error in BS domain")
axes[1, 1].set_xlabel("S")
axes[1, 1].set_ylabel("t")
axes[1, 1].set_xlim(60, 180)

# Relative error slice at t=0.5
t_mid = 0.5
V_true_mid = bs_call_price(S_plot, torch.full_like(S_plot, t_mid), params)
x_mid, tau_mid, _ = bs_to_heat(S_plot, torch.full_like(S_plot, t_mid), V_true_mid, params)
Phi_mid = kernel.basis(x_mid, tau_mid)
v_mid, v_var_mid = gp.predict(Phi_mid)
_, _, V_mid = heat_to_bs(x_mid, tau_mid, v_mid, params)

rel_err_slice = ((V_mid - V_true_mid).abs() / V_true_mid.clamp(min=1e-6)).detach().numpy()

axes[1, 2].semilogy(S_plot.numpy(), rel_err_slice)
axes[1, 2].set_title(f"Relative error at t={t_mid}")
axes[1, 2].set_xlabel("Stock price S")
axes[1, 2].set_ylabel("Relative error")
axes[1, 2].axvline(params.K, color="gray", linestyle=":", alpha=0.5)

plt.suptitle(
    f"Black-Scholes → Heat → S-EPGP → Black-Scholes\n"
    f"(σ={params.sigma}, r={params.r}, K={params.K}, T={params.T}  |  "
    f"MSE={mse_bs:.2e}, Mean rel err={rel_err:.2%})",
    fontsize=13,
)
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "black_scholes_results.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved: black_scholes_results.png")
