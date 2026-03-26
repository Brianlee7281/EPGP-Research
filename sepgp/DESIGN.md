# S-EPGP: Reimplementation & Extension

## Project Overview

Reimplementation of the Ehrenpreis-Palamodov Gaussian Process (S-EPGP) framework for solving
systems of linear PDEs with constant coefficients from data. Based on Härkönen, Lange-Hegermann,
Raiţă (ICML 2023) and the B-EPGP extension (arXiv:2411.16663).

**Original repo**: https://github.com/haerski/EPGP  
**Reference**: arXiv:2212.14319 (ICML 2023), arXiv:2411.16663

---

## Core Algorithm: S-EPGP

### Mathematical Setup

Given a system of linear PDEs with constant coefficients:

```
A(∂)f = 0,   where A ∈ R^{ℓ' × ℓ},  R = ℝ[∂_x1, ..., ∂_xn]
```

**Step 1: Algebraic Preprocessing**
Compute the characteristic variety:
```
V = { z ∈ ℂ^n : det A(z) = 0 }   (for scalar case)
V = { z ∈ ℂ^n : ker A(z) ≠ {0} }  (for systems)
```
and Noetherian multipliers B_{i,j}(x, z) ∈ ℂ[x, z]^ℓ.

**Step 2: Sample Frequencies**
Choose z_1, ..., z_r ∈ V (on the characteristic variety).

**Step 3: Construct Basis Functions**
```
φ_h(x) = B_j(x, z_h) · exp(z_h · x)
```
For scalar PDEs with trivial multipliers: φ_h(x) = exp(z_h · x).

**Step 4: GP Prior**
Define f(x) = C^T · φ(x) where C_j ~ N(0, Σ), Σ = diag(σ_j²).

Covariance kernel:
```
k(x, x') = φ(x)^H · Σ · φ(x')
```

**Step 5: Conditioning (= Ridge Regression)**
Given data {(x_i, y_i)}_{i=1}^N:
```
K_{ij} = k(x_i, x_j)
posterior mean:  μ(x*) = k(x*, X) · (K + σ_noise² I)^{-1} · y
posterior var:   σ²(x*) = k(x*, x*) - k(x*, X) · (K + σ_noise² I)^{-1} · k(X, x*)
```

**Step 6: Hyperparameter Optimization**
Optimize σ_j (frequency weights), σ_noise, and optionally z_j via marginal likelihood:
```
log p(y|X) = -½ y^T (K + σ²I)^{-1} y - ½ log|K + σ²I| - N/2 log(2π)
```

---

## Characteristic Varieties for Target PDEs

### Phase 1: Scalar PDEs (trivial Noetherian multipliers)

| PDE | Equation | Variety V | Parametrization |
|-----|----------|-----------|-----------------|
| 1D Heat | u_t = u_xx | z_t = z_x² | z_x free, z_t = z_x² |
| 2D Heat | u_t = u_xx + u_yy | z_t = z_x² + z_y² | (z_x, z_y) free, z_t = z_x² + z_y² |
| 2D Wave | u_tt = u_xx + u_yy | z_t² = z_x² + z_y² | (z_x, z_y) free, z_t = ±√(z_x² + z_y²) |
| 3D Wave | u_tt = u_xx + u_yy + u_zz | z_t² = z_x² + z_y² + z_z² | cone in 4D |
| Laplace 2D | u_xx + u_yy = 0 | z_x² + z_y² = 0 | z_y = ±iz_x |
| Helmholtz | u_xx + u_yy + k²u = 0 | z_x² + z_y² = -k² | circle of radius ik |

### Phase 2: Black-Scholes Pipeline

The Black-Scholes PDE:
```
∂V/∂t + ½σ²S² ∂²V/∂S² + rS ∂V/∂S - rV = 0
```

**Transform to heat equation** via:
```
S = K·exp(x),  t = T - τ/(½σ²),  V = K·v(x,τ)·exp(-½(k-1)x - ¼(k+1)²τ)
```
where k = 2r/σ².

Then v satisfies: v_τ = v_xx (standard heat equation).

**Pipeline**:
1. Define Black-Scholes parameters (σ, r, K, T)
2. Compute true BS solution (European call via BS formula)
3. Transform true solution → heat equation domain
4. Sample initial condition g(x) = v(x, 0) at t=0
5. Fit S-EPGP to heat equation data
6. Transform prediction back to BS variables
7. Compare with true BS price

### Phase 3: PDE Systems (nontrivial Noetherian multipliers)

| System | Equations | Variables | Variety | Multipliers |
|--------|-----------|-----------|---------|-------------|
| Maxwell (2D+t) | ∂E_x/∂t = ∂B/∂y, ∂E_y/∂t = -∂B/∂x, ∂B/∂t = ∂E_x/∂y - ∂E_y/∂x | (E_x, E_y, B)(x,y,t) | z_t² = z_x² + z_y² (same as 2D wave!) | Nontrivial: B_j ∈ ℂ^3 |
| Elasticity | μΔu + (λ+μ)∇(∇·u) = ρ ∂²u/∂t² | u(x,y,t) ∈ ℝ² | Two sheets (P-wave, S-wave) | Nontrivial |

**Maxwell Noetherian multipliers** (from the EPGP paper):
For frequency z = (z_x, z_y, z_t) on the cone z_t² = z_x² + z_y², the null space of
```
A(z) = [[z_t, 0, -z_y], [0, z_t, z_x], [-z_y, z_x, z_t]]
```
gives the multiplier vectors. For each z on the cone, ker A(z) is 1-dimensional:
```
B(z) ∝ (z_y·z_t, -z_x·z_t, z_x² + z_y²)^T  (up to normalization)
```

---

## Architecture

```
epgp/
├── __init__.py
├── varieties.py          # Characteristic variety computation & sampling
├── multipliers.py        # Noetherian multiplier computation (for systems)
├── kernels.py            # S-EPGP kernel construction
├── gp.py                 # GP regression engine (conditioning, prediction, MLL)
├── optimize.py           # Hyperparameter optimization (MLL-based)
├── transforms.py         # PDE-specific transforms (e.g., BS ↔ heat)
│
├── pdes/                 # PDE definitions (registry pattern)
│   ├── __init__.py
│   ├── base.py           # Abstract PDE class
│   ├── heat.py           # 1D/2D/3D heat equation
│   ├── wave.py           # 2D/3D wave equation
│   ├── laplace.py        # Laplace, Helmholtz
│   ├── maxwell.py        # Maxwell system
│   └── black_scholes.py  # Black-Scholes (wraps heat + transform)
│
├── data/                 # Data generation & ground truth
│   ├── __init__.py
│   ├── generators.py     # Analytical solutions, numerical references
│   └── samplers.py       # IC/BC sampling, noisy observations
│
└── viz/                  # Visualization
    ├── __init__.py
    ├── plot_1d.py
    ├── plot_2d.py
    └── animate.py

experiments/
├── 01_1d_heat.py         # Phase 1a: reproduce ICML Fig 2
├── 02_2d_heat.py         # Phase 1b: melting face demo
├── 03_2d_wave.py         # Phase 1c: vibrating membrane
├── 04_black_scholes.py   # Phase 2: BS pipeline (your original task!)
├── 05_maxwell.py         # Phase 3: Maxwell system
├── 06_pinn_comparison.py # Benchmarks vs PINN
└── notebooks/
    └── demo.ipynb        # Interactive walkthrough
```

---

## Implementation Roadmap

### Phase 1a: 1D Heat Equation (foundation)
**Goal**: Exact reproduction of ICML 2023 Fig 2 results.

- [ ] `varieties.py`: Parametric sampler for z_t = z_x²
  - For 1D heat with purely imaginary frequencies: z_x = iω → z_t = -ω²
  - φ_j(x,t) = exp(-ω_j² t + iω_j x)  →  real part: exp(-ω²t)cos(ωx), sin(ωx)
  - Sample ω_1, ..., ω_r uniformly or on a grid
- [ ] `kernels.py`: Build Φ matrix and kernel k(x,x') = Φ(x)^H Σ Φ(x')
- [ ] `gp.py`: GP conditioning with Cholesky solve
- [ ] `optimize.py`: MLL optimization for σ_j, σ_noise via L-BFGS
- [ ] Test: g(x) = exp(-50(x-0.5)²) at t=0, predict for t > 0

**Key numerical detail**: For real-valued solutions, pair conjugate frequencies:
```
φ_j(x,t) = exp(-ω_j²t) · cos(ω_j·x)   (real part)
ψ_j(x,t) = exp(-ω_j²t) · sin(ω_j·x)   (imag part)
```
So the real basis has 2r functions for r frequencies.

### Phase 1b: 2D Heat Equation
- [ ] Extend variety sampler: (ω_x, ω_y) grid → z_t = -(ω_x² + ω_y²)
- [ ] Basis: φ_{j,k}(x,y,t) = exp(-(ω_j² + ω_k²)t) · [cos/sin(ω_j x)] · [cos/sin(ω_k y)]
- [ ] Test: smiley face initial condition (reproduce MATHREPO demo)

### Phase 1c: 2D Wave Equation
- [ ] Variety: z_t² = z_x² + z_y²  (cone)
- [ ] Two branches: z_t = +√(z_x²+z_y²) and z_t = -√(...)
- [ ] Basis involves cos/sin in both space and time (no exponential decay!)
- [ ] Test: vibrating membrane from first 3 frames

### Phase 2: Black-Scholes Pipeline
- [ ] `transforms.py`: BS ↔ heat variable changes
- [ ] `black_scholes.py`: Full pipeline class
- [ ] `generators.py`: BS analytical solution (European call/put)
- [ ] Test: fit from transformed IC, inverse-transform, compare to BS formula
- [ ] Measure MSE in both heat and BS domains

### Phase 3: Maxwell Equations (PDE system)
- [ ] `multipliers.py`: Compute ker A(z) for Maxwell's A matrix at each z
- [ ] Extend `kernels.py` for vector-valued GPs: k(x,x') ∈ ℝ^{3×3}
- [ ] Basis: φ_j(x,y,t) = B(z_j) · exp(z_j · (x,y,t))  ∈ ℂ³
- [ ] Covariance: k(x,x') = Σ_j σ_j² · φ_j(x) · φ_j(x')^H  ∈ ℂ^{3×3}
- [ ] For prediction of component i: use k_{ii} and cross-covariances
- [ ] Test: spiral E-field initial data (reproduce MATHREPO Maxwell demo)

---

## Key Implementation Notes

### Frequency Sampling Strategies
1. **Uniform grid** (simple, good for low-d): ω ∈ [-ω_max, ω_max] with spacing Δω
2. **Random sampling** (scalable): ω ~ N(0, length_scale²)
3. **Learned frequencies** (S-EPGP): initialize randomly, optimize via MLL
4. **Importance sampling**: concentrate near frequencies with high signal

### Numerical Considerations
- **Complex arithmetic**: PyTorch supports complex tensors (torch.cfloat, torch.cdouble)
- **Cholesky stability**: add jitter (1e-6 to 1e-4) to diagonal of K
- **Scaling**: For N data points, r frequencies → K is N×N, Φ is N×2r
  - Direct Cholesky: O(N³) — fine for N < 5000
  - For larger N: use Woodbury identity since rank(K) ≤ 2r:
    ```
    (Φ Σ Φ^H + σ²I)^{-1} = σ^{-2}I - σ^{-2}Φ(Σ^{-1} + σ^{-2}Φ^HΦ)^{-1}Φ^H σ^{-2}
    ```
    This is O(Nr² + r³) instead of O(N³) — critical for 2D/3D problems!
- **MLL gradient**: Use automatic differentiation (PyTorch autograd)

### Real-valued Solutions from Complex Basis
For a PDE with real coefficients, if z is a root then so is z̄.
Pair them: for z = a + ib,
```
Re[c · B(z) exp(z·x)] = exp(a·x)(c_R cos(b·x) - c_I sin(b·x))
```
The real basis has the same dimension but avoids complex GP regression.

---

## Comparison Targets

| Method | Params | Exact? | Data needed | Extrapolation |
|--------|--------|--------|-------------|---------------|
| S-EPGP (ours) | ~3r + 2 | Yes (by construction) | Very little | Excellent |
| PINN | ~300K | Approx | Collocation pts | Poor |
| Fourier (fixed freq) | ~2M+1 | Approx | Medium | Medium |
| Numerical solver | N/A | Approx | IC/BC (well-posed) | N/A |

---

## References

1. Härkönen, Lange-Hegermann, Raiţă. "Gaussian Process Priors for Systems of Linear PDEs with Constant Coefficients." ICML 2023. arXiv:2212.14319
2. Huang, Härkönen, Lange-Hegermann, Raiţă. "GP Priors for Boundary Value Problems of Linear PDEs." arXiv:2411.16663
3. Original code: https://github.com/haerski/EPGP
4. MATHREPO demos: https://mathrepo.mis.mpg.de/EPGP/index.html
