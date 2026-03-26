"""
Experiment 1: S-EPGP on 1D Heat Equation (fast version)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from epgp_np import sample_variety_heat_1d, basis_heat_1d, SEPGP

np.random.seed(42)

def true_heat_1d(x, t, a=50.0, x0=0.5):
    return np.exp(-a*(x-x0)**2 / (1+4*a*t)) / np.sqrt(1+4*a*t)

# Training: 30 points at t=0
N = 30
x_tr = np.random.uniform(0, 1, N)
t_tr = np.zeros(N)
y_tr = true_heat_1d(x_tr, t_tr) + 0.01*np.random.randn(N)

# S-EPGP
n_freq = 30
omegas = sample_variety_heat_1d(n_freq, omega_max=25.0)
Phi_tr = basis_heat_1d(x_tr, t_tr, omegas)

gp = SEPGP(n_basis=2*n_freq)
print(f"1D Heat | {n_freq} freq -> {2*n_freq} basis | {N} train pts at t=0")
gp.optimize(lambda: Phi_tr, y_tr, maxiter=200, verbose=True)
gp.condition(Phi_tr, y_tr)

# Test grid
nx, nt = 80, 40
x1d = np.linspace(0, 1, nx)
t1d = np.linspace(0, 0.02, nt)
TG, XG = np.meshgrid(t1d, x1d, indexing="ij")
Phi_te = basis_heat_1d(XG.ravel(), TG.ravel(), omegas)
y_pred, y_var = gp.predict(Phi_te)
y_true = true_heat_1d(XG.ravel(), TG.ravel())

mse = np.mean((y_pred - y_true)**2)
print(f"MSE: {mse:.2e} | Max err: {np.max(np.abs(y_pred-y_true)):.2e}")

# Plot
Y_true_g = y_true.reshape(nt, nx)
Y_pred_g = y_pred.reshape(nt, nx)
Y_var_g = y_var.reshape(nt, nx)
ext = [0, 1, 0, 0.02]

fig, axes = plt.subplots(2, 3, figsize=(15, 9))

im = axes[0,0].imshow(Y_true_g, origin="lower", extent=ext, aspect="auto", cmap="viridis")
axes[0,0].set_title("True solution"); axes[0,0].set_xlabel("x"); axes[0,0].set_ylabel("t")
plt.colorbar(im, ax=axes[0,0])

im = axes[0,1].imshow(Y_pred_g, origin="lower", extent=ext, aspect="auto", cmap="viridis")
axes[0,1].scatter(x_tr, t_tr, c="red", s=25, marker="x", label="train", zorder=5)
axes[0,1].set_title(f"S-EPGP prediction (MSE={mse:.2e})"); axes[0,1].set_xlabel("x"); axes[0,1].set_ylabel("t")
axes[0,1].legend(); plt.colorbar(im, ax=axes[0,1])

im = axes[0,2].imshow(np.abs(Y_pred_g-Y_true_g), origin="lower", extent=ext, aspect="auto", cmap="hot")
axes[0,2].set_title("|Error|"); axes[0,2].set_xlabel("x"); axes[0,2].set_ylabel("t")
plt.colorbar(im, ax=axes[0,2])

colors = ["#1f77b4","#ff7f0e","#2ca02c","#d62728"]
for i, ts in enumerate([0, 0.005, 0.01, 0.02]):
    tidx = min(int(ts/0.02*(nt-1)), nt-1)
    axes[1,0].plot(x1d, Y_true_g[tidx], "--", color=colors[i], alpha=0.6)
    axes[1,0].plot(x1d, Y_pred_g[tidx], "-", color=colors[i], label=f"t={ts}")
axes[1,0].scatter(x_tr, y_tr, c="red", s=15, marker="x", zorder=5)
axes[1,0].set_title("Time slices (solid=pred, dashed=true)"); axes[1,0].set_xlabel("x")
axes[1,0].legend(fontsize=8)

std0 = np.sqrt(Y_var_g[0])
axes[1,1].plot(x1d, Y_pred_g[0], "b-", label="mean (t=0)")
axes[1,1].fill_between(x1d, Y_pred_g[0]-2*std0, Y_pred_g[0]+2*std0, alpha=0.25, label="±2σ")
axes[1,1].scatter(x_tr, y_tr, c="red", s=15, marker="x")
axes[1,1].set_title("Uncertainty at t=0"); axes[1,1].set_xlabel("x"); axes[1,1].legend()

axes[1,2].bar(range(len(gp.sigma2)), gp.sigma2, width=0.8, color="#1f77b4")
axes[1,2].set_title("Learned σ² weights"); axes[1,2].set_xlabel("Basis index"); axes[1,2].set_yscale("log")

fig.suptitle("S-EPGP: 1D Heat Equation  u_t = u_xx  (data at t=0 only)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "exp1_heat1d.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved exp1_heat1d.png")
