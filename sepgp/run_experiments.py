"""
S-EPGP Complete Experiments
============================
Phase 1a: 1D Heat Equation
Phase 1c: 2D Wave Equation  
Phase 2:  Black-Scholes Pipeline
"""

import sys, os
sys.path.insert(0, "/home/claude")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from epgp_np import (
    sample_variety_heat_1d, sample_variety_wave_2d, sample_variety_heat_2d,
    basis_heat_1d, basis_heat_2d, basis_wave_2d,
    SEPGP,
    BSParams, bs_to_heat, heat_to_bs, bs_call_price, heat_ic_from_call,
)

np.random.seed(42)

# ================================================================
# EXPERIMENT 1: 1D Heat Equation
# ================================================================

print("=" * 70)
print("EXPERIMENT 1: 1D Heat Equation  u_t = u_xx")
print("=" * 70)

def true_heat_1d(x, t, a=50.0, x0=0.5):
    """u(x,t) = 1/√(1+4at) exp(-a(x-x0)²/(1+4at))"""
    return np.exp(-a * (x - x0)**2 / (1 + 4*a*t)) / np.sqrt(1 + 4*a*t)

# Training: 30 points at t=0 only
N_train = 30
x_tr = np.random.uniform(0, 1, N_train)
t_tr = np.zeros(N_train)
y_tr = true_heat_1d(x_tr, t_tr) + 0.01 * np.random.randn(N_train)

# S-EPGP
n_freq = 40
omegas = sample_variety_heat_1d(n_freq, omega_max=30.0)
Phi_tr = basis_heat_1d(x_tr, t_tr, omegas)

gp1 = SEPGP(n_basis=2*n_freq)
print(f"Basis: {n_freq} freq → {2*n_freq} functions | Train: {N_train} pts at t=0")
print("Optimizing hyperparameters...")
gp1.optimize(lambda: Phi_tr, y_tr, maxiter=300)

gp1.condition(Phi_tr, y_tr)

# Test grid
nx, nt = 100, 50
x1d = np.linspace(0, 1, nx)
t1d = np.linspace(0, 0.02, nt)
TG, XG = np.meshgrid(t1d, x1d, indexing="ij")
x_te, t_te = XG.ravel(), TG.ravel()

Phi_te = basis_heat_1d(x_te, t_te, omegas)
y_pred, y_var = gp1.predict(Phi_te)
y_true = true_heat_1d(x_te, t_te)

mse1 = np.mean((y_pred - y_true)**2)
print(f"MSE: {mse1:.2e} | Max err: {np.max(np.abs(y_pred - y_true)):.2e}")

# --- Plot ---
fig = plt.figure(figsize=(16, 10))
gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)

Y_true_g = y_true.reshape(nt, nx)
Y_pred_g = y_pred.reshape(nt, nx)
ext = [0, 1, 0, 0.02]

ax = fig.add_subplot(gs[0, 0])
im = ax.imshow(Y_true_g, origin="lower", extent=ext, aspect="auto", cmap="viridis")
ax.set_title("True solution"); ax.set_xlabel("x"); ax.set_ylabel("t")
plt.colorbar(im, ax=ax)

ax = fig.add_subplot(gs[0, 1])
im = ax.imshow(Y_pred_g, origin="lower", extent=ext, aspect="auto", cmap="viridis")
ax.scatter(x_tr, t_tr, c="red", s=20, marker="x", label="train")
ax.set_title(f"S-EPGP (MSE={mse1:.2e})"); ax.set_xlabel("x"); ax.set_ylabel("t")
ax.legend(); plt.colorbar(im, ax=ax)

ax = fig.add_subplot(gs[0, 2])
im = ax.imshow(np.abs(Y_pred_g - Y_true_g), origin="lower", extent=ext, aspect="auto", cmap="hot")
ax.set_title("|Error|"); ax.set_xlabel("x"); ax.set_ylabel("t")
plt.colorbar(im, ax=ax)

ax = fig.add_subplot(gs[1, 0])
for i, ts in enumerate([0, 0.005, 0.01, 0.02]):
    tidx = min(int(ts / 0.02 * (nt-1)), nt-1)
    ax.plot(x1d, Y_true_g[tidx], "--", alpha=0.6)
    ax.plot(x1d, Y_pred_g[tidx], "-", label=f"t={ts}")
ax.scatter(x_tr, y_tr, c="red", s=15, marker="x", zorder=5)
ax.set_title("Time slices"); ax.set_xlabel("x"); ax.legend(fontsize=8)

ax = fig.add_subplot(gs[1, 1])
std0 = np.sqrt(y_var.reshape(nt, nx)[0])
ax.plot(x1d, Y_pred_g[0], "b-", label="mean")
ax.fill_between(x1d, Y_pred_g[0]-2*std0, Y_pred_g[0]+2*std0, alpha=0.3, label="±2σ")
ax.scatter(x_tr, y_tr, c="red", s=15, marker="x")
ax.set_title("Uncertainty (t=0)"); ax.set_xlabel("x"); ax.legend()

# Spectral weights
ax = fig.add_subplot(gs[1, 2])
ax.bar(range(len(gp1.sigma2)), gp1.sigma2, width=0.8)
ax.set_title("Learned σ² weights"); ax.set_xlabel("Basis index"); ax.set_ylabel("σ²")
ax.set_yscale("log")

fig.suptitle("S-EPGP: 1D Heat Equation (data at t=0 only)", fontsize=14)
plt.savefig("/home/claude/exp1_heat1d.png", dpi=150, bbox_inches="tight")
plt.close()
print("→ Saved exp1_heat1d.png\n")


# ================================================================
# EXPERIMENT 2: 2D Wave Equation
# ================================================================

print("=" * 70)
print("EXPERIMENT 2: 2D Wave Equation  u_tt = u_xx + u_yy")
print("=" * 70)

def true_wave_2d(x, y, t):
    """Standing wave: u = cos(πx)cos(πy)cos(√2·πt)."""
    return np.cos(np.pi*x) * np.cos(np.pi*y) * np.cos(np.sqrt(2)*np.pi*t)

# Training: first 2 time frames
n_space = 15
xs = np.linspace(0, 1, n_space)
ys = np.linspace(0, 1, n_space)
XS, YS = np.meshgrid(xs, ys, indexing="ij")
x_s, y_s = XS.ravel(), YS.ravel()

t_frames = [0.0, 0.05]
x_tr2, y_tr2, t_tr2, y_tr2_vals = [], [], [], []
for tf in t_frames:
    x_tr2.append(x_s)
    y_tr2.append(y_s)
    t_tr2.append(np.full_like(x_s, tf))
    y_tr2_vals.append(true_wave_2d(x_s, y_s, tf) + 0.01*np.random.randn(len(x_s)))

x_tr2 = np.concatenate(x_tr2)
y_tr2 = np.concatenate(y_tr2)
t_tr2 = np.concatenate(t_tr2)
val_tr2 = np.concatenate(y_tr2_vals)

# S-EPGP
n_per_dim = 8  # → 64 spatial freq pairs → 256 basis functions
omegas_sp, omegas_t = sample_variety_wave_2d(n_per_dim, omega_max=12.0)
Phi_tr2 = basis_wave_2d(x_tr2, y_tr2, t_tr2, omegas_sp, omegas_t)

gp2 = SEPGP(n_basis=4 * len(omegas_t))
print(f"Spatial freq: {n_per_dim}² = {len(omegas_t)} → {4*len(omegas_t)} basis | Train: {len(val_tr2)} pts")
print("Optimizing...")
gp2.optimize(lambda: Phi_tr2, val_tr2, maxiter=300)
gp2.condition(Phi_tr2, val_tr2)

# Test: predict at later times
fig, axes = plt.subplots(2, 4, figsize=(18, 9))
test_times = [0.0, 0.1, 0.2, 0.3]
n_test_s = 40
xt = np.linspace(0, 1, n_test_s)
yt = np.linspace(0, 1, n_test_s)
XT, YT = np.meshgrid(xt, yt, indexing="ij")
x_flat, y_flat = XT.ravel(), YT.ravel()

mse_total = 0
for col, tt in enumerate(test_times):
    t_flat = np.full_like(x_flat, tt)
    u_true = true_wave_2d(x_flat, y_flat, t_flat).reshape(n_test_s, n_test_s)
    
    Phi = basis_wave_2d(x_flat, y_flat, t_flat, omegas_sp, omegas_t)
    u_pred, _ = gp2.predict(Phi)
    u_pred = u_pred.reshape(n_test_s, n_test_s)
    
    mse_t = np.mean((u_pred - u_true)**2)
    mse_total += mse_t
    
    ext2 = [0, 1, 0, 1]
    axes[0, col].imshow(u_true, origin="lower", extent=ext2, cmap="RdBu_r", vmin=-1, vmax=1)
    axes[0, col].set_title(f"True t={tt}")
    
    axes[1, col].imshow(u_pred, origin="lower", extent=ext2, cmap="RdBu_r", vmin=-1, vmax=1)
    axes[1, col].set_title(f"S-EPGP t={tt}\nMSE={mse_t:.2e}")
    
    if tt in t_frames:
        axes[1, col].set_title(f"S-EPGP t={tt} [TRAIN]\nMSE={mse_t:.2e}")

print(f"Mean MSE across times: {mse_total/len(test_times):.2e}")

fig.suptitle("S-EPGP: 2D Wave Equation (trained on t=0, 0.05 frames)", fontsize=13)
plt.tight_layout()
plt.savefig("/home/claude/exp2_wave2d.png", dpi=150, bbox_inches="tight")
plt.close()
print("→ Saved exp2_wave2d.png\n")


# ================================================================
# EXPERIMENT 3: Black-Scholes Pipeline
# ================================================================

print("=" * 70)
print("EXPERIMENT 3: Black-Scholes → Heat → S-EPGP → Black-Scholes")
print("=" * 70)

params = BSParams(sigma=0.3, r=0.05, K=100.0, T=1.0)
print(f"σ={params.sigma}, r={params.r}, K={params.K}, T={params.T}, k={params.k:.4f}")

# Training: IC in heat domain (τ=0, i.e. BS expiry)
N_bs = 60
x_bs_tr = np.linspace(-2.0, 2.0, N_bs)
tau_bs_tr = np.zeros(N_bs)
v_bs_tr = heat_ic_from_call(x_bs_tr, params)
v_bs_tr_noisy = v_bs_tr + 0.01 * np.abs(v_bs_tr + 0.01) * np.random.randn(N_bs)

# S-EPGP on heat equation
n_freq_bs = 60
omegas_bs = sample_variety_heat_1d(n_freq_bs, omega_max=25.0)
Phi_bs_tr = basis_heat_1d(x_bs_tr, tau_bs_tr, omegas_bs)

gp3 = SEPGP(n_basis=2*n_freq_bs)
print(f"Basis: {n_freq_bs} freq → {2*n_freq_bs} functions | Train: {N_bs} pts at τ=0")
print("Optimizing...")
gp3.optimize(lambda: Phi_bs_tr, v_bs_tr_noisy, maxiter=400)
gp3.condition(Phi_bs_tr, v_bs_tr_noisy)

# Predict in heat domain and transform back
fig = plt.figure(figsize=(16, 12))
gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.35)

# -- Panel 1: IC fit in heat domain --
ax = fig.add_subplot(gs[0, 0])
x_plot = np.linspace(-2.5, 2.5, 300)
v_true_ic = heat_ic_from_call(x_plot, params)
Phi_ic = basis_heat_1d(x_plot, np.zeros_like(x_plot), omegas_bs)
v_pred_ic, v_var_ic = gp3.predict(Phi_ic)

ax.plot(x_plot, v_true_ic, "r--", lw=1.5, label="True IC")
ax.plot(x_plot, v_pred_ic, "b-", lw=2, label="S-EPGP")
ax.fill_between(x_plot, v_pred_ic - 2*np.sqrt(v_var_ic), v_pred_ic + 2*np.sqrt(v_var_ic), alpha=0.2)
ax.scatter(x_bs_tr, v_bs_tr_noisy, c="red", s=10, marker="x")
ax.set_title("Heat IC fit (τ=0)"); ax.set_xlabel("x"); ax.legend()

# -- Panel 2: Heat domain at various τ --
ax = fig.add_subplot(gs[0, 1])
tau_max = 0.5 * params.sigma**2 * params.T
for tau_val in [0, 0.01, 0.02, 0.04]:
    Phi_t = basis_heat_1d(x_plot, np.full_like(x_plot, tau_val), omegas_bs)
    v_t, _ = gp3.predict(Phi_t)
    ax.plot(x_plot, v_t, label=f"τ={tau_val:.2f}")
ax.set_title("Heat solution at various τ"); ax.set_xlabel("x"); ax.set_ylabel("v"); ax.legend(fontsize=8)

# -- Panels 3-8: BS domain comparisons --
S_plot = np.linspace(50, 200, 300)
t_slices = [0.0, 0.25, 0.5, 0.75, 0.9, 0.99]
positions = [(0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1)]

all_mse = []
for (row, col), t_val in zip(positions, t_slices):
    ax = fig.add_subplot(gs[row, col])
    
    V_true = bs_call_price(S_plot, np.full_like(S_plot, t_val), params)
    
    # Transform to heat, predict, transform back
    x_h, tau_h, _ = bs_to_heat(S_plot, np.full_like(S_plot, t_val), V_true, params)
    Phi_h = basis_heat_1d(x_h, tau_h, omegas_bs)
    v_h, v_h_var = gp3.predict(Phi_h)
    _, _, V_pred = heat_to_bs(x_h, tau_h, v_h, params)
    _, _, V_lo = heat_to_bs(x_h, tau_h, v_h - 2*np.sqrt(np.abs(v_h_var)), params)
    _, _, V_hi = heat_to_bs(x_h, tau_h, v_h + 2*np.sqrt(np.abs(v_h_var)), params)
    
    valid = np.isfinite(V_pred) & np.isfinite(V_true) & (V_true > 0.1)
    mse_slice = np.mean((V_pred[valid] - V_true[valid])**2)
    all_mse.append(mse_slice)
    
    ax.plot(S_plot, V_true, "r--", lw=1.5, label="BS true")
    ax.plot(S_plot, V_pred, "b-", lw=2, label="S-EPGP")
    ax.fill_between(S_plot, np.minimum(V_lo, V_hi), np.maximum(V_lo, V_hi), 
                    alpha=0.15, color="blue")
    ax.axvline(params.K, color="gray", ls=":", alpha=0.5)
    ax.set_title(f"t={t_val} (MSE={mse_slice:.2e})")
    ax.set_xlabel("S"); ax.set_ylabel("V")
    ax.legend(fontsize=7); ax.set_ylim(-5, max(V_true) * 1.1)

# Summary
ax = fig.add_subplot(gs[2, 2])
ax.bar(range(len(t_slices)), all_mse, tick_label=[f"t={t}" for t in t_slices])
ax.set_ylabel("MSE"); ax.set_title("MSE by time slice")
ax.set_yscale("log")
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

overall_mse = np.mean(all_mse)
fig.suptitle(
    f"Black-Scholes → Heat → S-EPGP → Black-Scholes\n"
    f"σ={params.sigma}, r={params.r}, K={params.K}, T={params.T} | "
    f"Overall MSE={overall_mse:.2e}",
    fontsize=13,
)
plt.savefig("/home/claude/exp3_black_scholes.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Overall MSE across time slices: {overall_mse:.2e}")
print("→ Saved exp3_black_scholes.png\n")

print("=" * 70)
print("ALL EXPERIMENTS COMPLETE")
print("=" * 70)
