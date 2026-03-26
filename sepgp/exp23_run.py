"""
Experiments 2 & 3: 2D Wave + Black-Scholes (fast, fixed hyperparams)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from epgp_np import (
    sample_variety_wave_2d, basis_wave_2d,
    sample_variety_heat_1d, basis_heat_1d,
    SEPGP,
    BSParams, bs_to_heat, heat_to_bs, bs_call_price, heat_ic_from_call,
)
np.random.seed(42)

# ================================================================
# EXPERIMENT 2: 2D Wave
# ================================================================
print("=" * 60)
print("2D WAVE EQUATION  u_tt = u_xx + u_yy")
print("=" * 60)

def true_wave(x, y, t):
    return np.cos(np.pi*x)*np.cos(np.pi*y)*np.cos(np.sqrt(2)*np.pi*t)

# Train: 2 frames, 10×10 grid
n_s = 10
xs = np.linspace(0,1,n_s); ys = np.linspace(0,1,n_s)
XS, YS = np.meshgrid(xs, ys, indexing="ij")
xf, yf = XS.ravel(), YS.ravel()

t_frames = [0.0, 0.05]
x_tr, y_tr, t_tr, v_tr = [], [], [], []
for tf in t_frames:
    x_tr.append(xf); y_tr.append(yf)
    t_tr.append(np.full(len(xf), tf))
    v_tr.append(true_wave(xf, yf, tf) + 0.005*np.random.randn(len(xf)))
x_tr = np.concatenate(x_tr); y_tr = np.concatenate(y_tr)
t_tr = np.concatenate(t_tr); v_tr = np.concatenate(v_tr)

# S-EPGP with fixed hyperparams
n_pd = 5  # 25 spatial freq → 100 basis
omegas_sp, omegas_t = sample_variety_wave_2d(n_pd, omega_max=8.0)
n_basis = 4 * len(omegas_t)
Phi_tr = basis_wave_2d(x_tr, y_tr, t_tr, omegas_sp, omegas_t)

gp = SEPGP(n_basis=n_basis)
# Set hyperparams manually: uniform weights, small noise
gp.log_sigma = np.full(n_basis, -1.0)  # σ² ≈ 0.135 each
gp.log_noise = -3.0  # σ_n² ≈ 0.002

print(f"{n_pd}²={len(omegas_t)} freq → {n_basis} basis | {len(v_tr)} train pts")
print(f"MLL: {gp.marginal_log_likelihood(Phi_tr, v_tr):.2f}")
gp.condition(Phi_tr, v_tr)

# Predict
fig, axes = plt.subplots(3, 5, figsize=(20, 12))
test_times = [0.0, 0.1, 0.2, 0.3, 0.4]
n_ts = 30
xt = np.linspace(0,1,n_ts); yt = np.linspace(0,1,n_ts)
XT, YT = np.meshgrid(xt, yt, indexing="ij")
xfl, yfl = XT.ravel(), YT.ravel()

for col, tt in enumerate(test_times):
    tfl = np.full_like(xfl, tt)
    u_true = true_wave(xfl, yfl, tfl).reshape(n_ts, n_ts)
    
    Phi = basis_wave_2d(xfl, yfl, tfl, omegas_sp, omegas_t)
    u_pred, _ = gp.predict(Phi)
    u_pred = u_pred.reshape(n_ts, n_ts)
    
    mse_t = np.mean((u_pred - u_true)**2)
    
    kw = dict(origin="lower", extent=[0,1,0,1], cmap="RdBu_r", vmin=-1.1, vmax=1.1)
    axes[0,col].imshow(u_true, **kw)
    axes[0,col].set_title(f"True t={tt:.1f}", fontsize=10)
    
    axes[1,col].imshow(u_pred, **kw)
    label = f"S-EPGP t={tt:.1f}"
    if tt in t_frames: label += " [TRAIN]"
    axes[1,col].set_title(f"{label}\nMSE={mse_t:.2e}", fontsize=10)
    
    axes[2,col].imshow(np.abs(u_pred-u_true), origin="lower", extent=[0,1,0,1],
                       cmap="hot", vmin=0, vmax=0.3)
    axes[2,col].set_title(f"|Error| t={tt:.1f}", fontsize=10)
    print(f"  t={tt:.1f} MSE={mse_t:.2e}")

for r in range(3):
    for c in range(5):
        if r < 2: axes[r,c].set_xticks([])
        if c > 0: axes[r,c].set_yticks([])

fig.suptitle("S-EPGP: 2D Wave Equation  (trained on t=0.0, 0.05 → extrapolating)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "exp2_wave2d.png"), dpi=150, bbox_inches="tight")
plt.close()
print("→ Saved exp2_wave2d.png\n")

# ================================================================
# EXPERIMENT 3: Black-Scholes Pipeline
# ================================================================
print("=" * 60)
print("BLACK-SCHOLES → HEAT → S-EPGP → BLACK-SCHOLES")
print("=" * 60)

params = BSParams(sigma=0.3, r=0.05, K=100.0, T=1.0)
print(f"σ={params.sigma}, r={params.r}, K={params.K}, T={params.T}")

# Train: IC in heat domain
N_bs = 80
x_bs = np.linspace(-3.0, 3.0, N_bs)
tau_bs = np.zeros(N_bs)
v_bs = heat_ic_from_call(x_bs, params)
v_bs_n = v_bs + 0.005 * (np.abs(v_bs) + 0.1) * np.random.randn(N_bs)

# S-EPGP
n_freq = 40
omegas = sample_variety_heat_1d(n_freq, omega_max=20.0)
n_b = 2 * n_freq
Phi_bs = basis_heat_1d(x_bs, tau_bs, omegas)

gp_bs = SEPGP(n_basis=n_b)
gp_bs.log_sigma = np.full(n_b, 0.0)  # σ² = 1
gp_bs.log_noise = -4.0
print(f"{n_freq} freq → {n_b} basis | {N_bs} train pts at τ=0")
print(f"MLL: {gp_bs.marginal_log_likelihood(Phi_bs, v_bs_n):.2f}")
gp_bs.condition(Phi_bs, v_bs_n)

# BS domain comparison
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

# Row 0: heat domain
x_plot = np.linspace(-3.5, 3.5, 300)

# IC fit
ax = axes[0, 0]
v_true_ic = heat_ic_from_call(x_plot, params)
Phi_ic = basis_heat_1d(x_plot, np.zeros_like(x_plot), omegas)
v_pred_ic, v_var_ic = gp_bs.predict(Phi_ic)
ax.plot(x_plot, v_true_ic, "r--", lw=1.5, label="True IC")
ax.plot(x_plot, v_pred_ic, "b-", lw=2, label="S-EPGP")
ax.fill_between(x_plot, v_pred_ic-2*np.sqrt(np.abs(v_var_ic)),
                v_pred_ic+2*np.sqrt(np.abs(v_var_ic)), alpha=0.2)
ax.scatter(x_bs, v_bs_n, c="red", s=8, marker="x")
ax.set_title("Heat IC fit (τ=0)"); ax.set_xlabel("x"); ax.legend(fontsize=8)

# Heat at various τ
ax = axes[0, 1]
for tau_val in [0, 0.005, 0.01, 0.02, 0.04]:
    Phi_t = basis_heat_1d(x_plot, np.full_like(x_plot, tau_val), omegas)
    v_t, _ = gp_bs.predict(Phi_t)
    ax.plot(x_plot, v_t, label=f"τ={tau_val:.3f}")
ax.set_title("Heat solution v(x,τ)"); ax.set_xlabel("x"); ax.legend(fontsize=7)

# Spectral weights
ax = axes[0, 2]
ax.bar(range(len(gp_bs.sigma2)), gp_bs.sigma2, width=0.8)
ax.set_title("σ² weights"); ax.set_xlabel("Basis"); ax.set_yscale("log")

# Row 1 + rest: BS domain at various t
S_plot = np.linspace(50, 200, 300)
t_slices = [0.0, 0.25, 0.5, 0.75, 0.95]
positions = [(0,3), (1,0), (1,1), (1,2), (1,3)]

all_mse = []
for (row, col), t_val in zip(positions, t_slices):
    ax = axes[row, col]
    V_true = bs_call_price(S_plot, np.full_like(S_plot, t_val), params)
    
    x_h, tau_h, _ = bs_to_heat(S_plot, np.full_like(S_plot, t_val), V_true, params)
    Phi_h = basis_heat_1d(x_h, tau_h, omegas)
    v_h, _ = gp_bs.predict(Phi_h)
    _, _, V_pred = heat_to_bs(x_h, tau_h, v_h, params)
    
    valid = np.isfinite(V_pred) & np.isfinite(V_true) & (V_true > 0.5) & (S_plot > 60) & (S_plot < 180)
    mse_s = np.mean((V_pred[valid] - V_true[valid])**2) if valid.any() else 0
    all_mse.append(mse_s)
    
    ax.plot(S_plot, V_true, "r--", lw=1.5, label="BS true")
    ax.plot(S_plot, np.clip(V_pred, -10, 200), "b-", lw=2, label="S-EPGP")
    ax.axvline(params.K, color="gray", ls=":", alpha=0.5)
    ax.set_title(f"t={t_val} (MSE={mse_s:.2e})", fontsize=10)
    ax.set_xlabel("S"); ax.set_ylabel("V")
    ax.legend(fontsize=7)
    ax.set_ylim(-5, max(V_true[valid])*1.15 if valid.any() else 100)
    print(f"  BS t={t_val}: MSE={mse_s:.2e}")

overall = np.mean(all_mse)
fig.suptitle(
    f"Black-Scholes → Heat → S-EPGP → Black-Scholes\n"
    f"(σ={params.sigma}, r={params.r}, K={params.K} | Overall MSE={overall:.2e})",
    fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "exp3_black_scholes.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"\nOverall BS MSE: {overall:.2e}")
print("→ Saved exp3_black_scholes.png")
