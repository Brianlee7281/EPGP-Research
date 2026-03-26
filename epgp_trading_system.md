# EPGP Confidence-Filtered Trading System: Mathematical Design Document

**Algebraic Gaussian Process Uncertainty Quantification for Systematic Trading**

> This document defines the complete mathematical architecture of a trading system that uses Ehrenpreis-Palamodov Gaussian Process (EPGP) posterior uncertainty as a confidence filter for trade entry and exit decisions. The system applies EPGP to financial PDEs through coordinate transformations, extracts calibrated uncertainty bands, and converts these into actionable trading signals across multiple markets.

---

## Table of Contents

1. [System Philosophy and Architecture](#1-system-philosophy-and-architecture)
2. [Mathematical Foundations: The Ehrenpreis-Palamodov Fundamental Principle](#2-mathematical-foundations)
   - 2.6 [Hardcoded Varieties for Target PDEs](#26-hardcoded-varieties-for-target-pdes) *(new)*
3. [EPGP Kernel Construction](#3-epgp-kernel-construction)
   - 3.6 [Explicit EPGP Kernel for the 1D Heat Equation](#36-explicit-epgp-kernel-for-the-1d-heat-equation) *(revised — full derivation + quadrature)*
   - 3.7 [Hyperparameter Optimization](#37-hyperparameter-optimization) *(new)*
4. [S-EPGP: Sparse Variant with Learned Frequencies](#4-s-epgp-sparse-variant)
5. [B-EPGP: Boundary-Constrained Extension](#5-b-epgp-boundary-constrained-extension)
6. [Financial PDE Pipeline: Black-Scholes to Heat Equation](#6-financial-pde-pipeline)
   - 6.7 [Numerical Stability](#67-numerical-stability) *(new)*
7. [Forward Problem: Option Pricing with Uncertainty Quantification](#7-forward-problem)
   - 7.0 [Data Sources and Pipeline](#70-data-sources-and-pipeline) *(new)*
8. [Inverse Problem: Volatility Surface Calibration via EPGP](#8-inverse-problem)
9. [Structural Break Detection: EPGP as Regime Sensor](#9-structural-break-detection)
10. [Confidence Filter: From Posterior Variance to Trading Signals](#10-confidence-filter)
11. [Market Application I: Options (Variance Risk Premium)](#11-market-application-options)
12. [Market Application II: Crypto Perpetual Funding Rate](#12-market-application-crypto)
13. [Hybrid Extensions: Beyond Linear Constant Coefficients](#13-hybrid-extensions)
14. [Industry Landscape: How Institutions Solve This Problem Today](#14-industry-landscape)
15. [The PDE Assumption: When It Holds and When It Breaks](#15-the-pde-assumption)
16. [Comparison Framework: EPGP vs Alternative Methods](#16-comparison-framework)
17. [Backtesting Architecture](#17-backtesting-architecture)
18. [Parameter Reference](#18-parameter-reference)
19. [Open Questions and Experimental Agenda](#19-open-questions)

---

## 1. System Philosophy and Architecture

### 1.1 Core Thesis

Most quantitative trading systems attempt to predict future prices or returns. This system does not. Instead, it exploits a different kind of information: **the degree to which observed market data conforms to known physical (PDE) structure, and the uncertainty inherent in regions where data is sparse or structure is breaking down.**

The key insight is that EPGP provides two outputs simultaneously:

- A **posterior mean** — the best estimate of the solution to the financial PDE given observed data.
- A **posterior variance** — a principled, calibrated measure of how confident the model is at each point.

The posterior variance is not a generic statistical uncertainty. It is structurally informed: it knows the PDE, it knows where data exists, and it knows where the PDE structure is being violated. This makes it a fundamentally different signal from anything produced by GARCH, Heston calibration, or neural networks.

### 1.2 What EPGP Uncertainty Tells Us

In regions where market data is abundant and consistent with the PDE:
- Posterior variance is small.
- The model is confident.
- Market pricing is well-explained by the known structure.

In regions where market data is sparse:
- Posterior variance is large.
- The model acknowledges ignorance.
- Trading in these regions carries unquantified risk.

In regions where market data contradicts the PDE:
- Posterior variance increases anomalously.
- The PDE structure is breaking down.
- Something has changed — regime transition, liquidity shock, structural break.

### 1.3 System Architecture

```
Layer 0: Data Ingestion
  Market data (options chain, funding rates, yield curve)
  → Sparse observation points {(x_i, y_i)}

Layer 1: PDE Transformation
  Raw financial variables → coordinate transform → constant-coefficient PDE domain
  (e.g., Black-Scholes → heat equation via log-moneyness + time reversal)

Layer 2: EPGP Engine
  Characteristic variety V computation (Macaulay2 / algebraic preprocessing)
  → EPGP kernel construction k(x, x')
  → GP posterior: mean μ(x) and variance σ²(x) conditioned on data

Layer 3: Dual Detection
  3A: Statistical regime detection (HMM on market features)
  3B: Structural regime detection (EPGP posterior variance anomalies)
  → Combined regime signal

Layer 4: Confidence Filter
  Posterior variance → confidence score → entry/exit/sizing decisions
  High confidence: trade at full size
  Low confidence: reduce size or skip
  Variance spike: exit or hedge

Layer 5: Strategy Execution
  Market-specific strategy logic (put spread, funding rate arb, etc.)
  → Order generation → Execution

Layer 6: Performance Monitoring
  Track whether confidence filter improves risk-adjusted returns
  A/B comparison: with filter vs without filter
```

### 1.4 Design Principles

**Principle 1: The EPGP layer does not generate alpha.** It generates a confidence signal that modulates existing strategies. Any strategy that has positive expected value can be improved by knowing when to trade and when to sit out.

**Principle 2: Two independent regime sensors are better than one.** HMM detects regime changes from statistical patterns in historical data. EPGP detects regime changes from structural PDE violations in current data. They use different information and fail in different ways.

**Principle 3: Same framework, multiple markets.** The EPGP confidence filter architecture is market-agnostic. The PDE and coordinate transformation change per market, but the confidence → signal pipeline is identical. This enables cross-market validation of the approach.

**Principle 4: Every claim is testable.** The system is designed for rigorous A/B testing. Strategy A (no EPGP filter) vs Strategy B (with EPGP filter) on the same data, same period, same execution assumptions.

---

## 2. Mathematical Foundations: The Ehrenpreis-Palamodov Fundamental Principle

### 2.1 Setup

Let R = C[∂₁, ..., ∂ₙ] be the polynomial ring of partial differential operators with constant coefficients, where ∂ᵢ = ∂/∂xᵢ. A system of linear PDEs with constant coefficients is encoded by a matrix A ∈ R^{ℓ' × ℓ}. We seek smooth solutions f = (f₁, ..., fℓ) : Ω → R^ℓ on a convex open domain Ω ⊆ Rⁿ satisfying:

A(∂)f = 0

The **characteristic variety** of this system is:

V = V(A) = { z ∈ Cⁿ : rank A(z) < ℓ }

This is an algebraic variety — the set of "frequencies" at which the symbol matrix A(z) drops rank.

### 2.2 The Fundamental Principle

**Theorem (Ehrenpreis 1970, Palamodov 1970).** Let A ∈ R^{ℓ' × ℓ}, Ω ⊆ Rⁿ convex and open. There exist:

(i) Irreducible components V₁, ..., Vₛ of the characteristic variety V, and

(ii) ℓ×1 polynomial vectors {Bᵢ,ⱼ(x, z)}_{i=1,...,s; j=1,...,mᵢ} in 2n variables (Noetherian multipliers),

such that every smooth solution f : Ω → Rℓ to A(∂)f = 0 can be written as:

f(x) = Σᵢ₌₁ˢ Σⱼ₌₁^{mᵢ} ∫_{Vᵢ} Bᵢ,ⱼ(x, z) · exp(⟨x, z⟩) dμᵢ,ⱼ(z)

where μᵢ,ⱼ are complex measures supported on Vᵢ.

### 2.3 Interpretation

This theorem says: **every solution is a (generalized) superposition of exponential-polynomial waves, with frequencies restricted to the characteristic variety V.**

For ODEs (n=1), V is a finite set of points (the roots of the characteristic polynomial), and the integral reduces to a finite sum — recovering the classical solution space {exp(zₖt) · tʲ}.

For PDEs (n≥2), V is typically a complex algebraic curve or surface, and the integral is genuinely continuous. This is a **nonlinear Fourier series** with frequencies drawn not from a lattice (as in classical Fourier analysis) but from an algebraic variety.

### 2.4 Examples Relevant to Finance

**Example 1: Heat equation.** The PDE ∂ₜu - ∂²ₓu = 0 has symbol z₁ - z₂², so:

V = { (z₁, z₂) ∈ C² : z₁ = z₂² }

This is a parabola in C². Solutions are superpositions of exp(z₂²t + z₂x) for z₂ ∈ C. When z₂ = iξ for real ξ, this gives exp(-ξ²t + iξx) — the standard Fourier mode of the heat equation, decaying exponentially in time.

**Example 2: Wave equation.** The PDE ∂²ₜu - ∂²ₓu = 0 has:

V = { (z₁, z₂) ∈ C² : z₁² = z₂² } = { z₁ = z₂ } ∪ { z₁ = -z₂ }

Two lines — corresponding to left-traveling and right-traveling waves.

**Example 3: 2D Heat equation (multi-asset).** The PDE ∂ₜu - ∂²ₓ₁u - 2ρ∂ₓ₁∂ₓ₂u - ∂²ₓ₂u = 0 has:

V = { (z₀, z₁, z₂) ∈ C³ : z₀ = z₁² + 2ρz₁z₂ + z₂² }

This is a 2D surface in C³, parametrizable by (z₁, z₂) ∈ C².

### 2.5 Computational Pipeline

The algebraic preprocessing — computing V and the Noetherian multipliers B — is fully algorithmic:

1. Encode the PDE system as a polynomial matrix A(z).
2. Compute the primary decomposition of the ideal generated by the maximal minors of A(z).
3. For each primary component Qᵢ, extract the variety Vᵢ and Noetherian operators Bᵢ,ⱼ.

This is implemented in the `NoetherianOperators` package in Macaulay2, via the `solvePDE` command (Ait El Manssour, Härkönen, Sturmfels 2023). However, Macaulay2 is designed for systems where the characteristic variety has nontrivial primary decomposition with multiplicities — e.g., coupled Maxwell's equations, higher-order PDE systems with confluent roots. For the PDEs used in this system, the varieties are elementary and computed by hand (see Section 2.6).

### 2.6 Hardcoded Varieties for Target PDEs

For every PDE targeted in Phases 1–4, the characteristic variety is smooth (no multiplicities), irreducible (s = 1), and has a single trivial Noetherian multiplier B = 1. No computer algebra system is needed.

| PDE | Symbol p(z) | Variety V | Noetherian multipliers | Parametrization |
|-----|-------------|-----------|----------------------|-----------------|
| 1D Heat: ∂ₜu = ∂²ₓu | z₁ - z₂² | {(z₂², z₂) : z₂ ∈ C} | B = 1 (trivial, multiplicity 1) | z₂ ∈ C freely |
| 1D Wave: ∂²ₜu = ∂²ₓu | z₁² - z₂² | {z₁ = z₂} ∪ {z₁ = -z₂} | B = 1 on each component | z₂ ∈ C on each line |
| Fokker-Planck (OU): ∂ₜp = κ∂_F(Fp) + ½σ²∂²_Fp | z₀ + κz₁ - ½σ²z₁² | {(½σ²z₁² - κz₁, z₁) : z₁ ∈ C} | B = 1 | z₁ ∈ C freely |
| 2D Heat: ∂ₜu = ∂²ₓ₁u + 2ρ∂ₓ₁∂ₓ₂u + ∂²ₓ₂u | z₀ - z₁² - 2ρz₁z₂ - z₂² | {(z₁² + 2ρz₁z₂ + z₂², z₁, z₂)} | B = 1 | (z₁, z₂) ∈ C² freely |

The full generality of primary decomposition and Noetherian operator computation is unnecessary for these cases. Macaulay2 becomes relevant only if the system is extended to coupled PDE systems (e.g., Maxwell, elasticity), systems with characteristic varieties having multiplicities (e.g., repeated roots), or user-specified arbitrary PDE systems where manual derivation is impractical. This is a Phase 5+ concern.

### 2.7 Why This Matters for Trading

The Ehrenpreis-Palamodov principle gives us the **exact function space** in which financial PDE solutions live. When we place a Gaussian process prior on this space, every realization of the GP is automatically an exact PDE solution. This is not an approximation — it is a structural guarantee. The posterior mean is an exact solution, and the posterior variance measures uncertainty *within the space of exact solutions*.

---

## 3. EPGP Kernel Construction

### 3.1 From Integral Representation to GP Kernel

The Ehrenpreis-Palamodov representation expresses solutions as:

f(x) = Σᵢ Σⱼ ∫_{Vᵢ} Bᵢ,ⱼ(x, z) · exp(⟨x, z⟩) dμᵢ,ⱼ(z)

To turn this into a GP, we parametrize the measures μᵢ,ⱼ as Gaussian measures on V:

μᵢ,ⱼ ~ Gaussian measure on Vᵢ with density proportional to exp(-||z'||² / (2σᵢ²))

where z' denotes local coordinates on Vᵢ and σᵢ is a trainable scale parameter.

### 3.2 The EPGP Covariance Kernel

With Gaussian measure parametrization, the EPGP covariance function is:

k_EPGP(x, x') = Σᵢ₌₁ˢ Σⱼ₌₁^{mᵢ} ∫_{Vᵢ} Bᵢ,ⱼ(x, z)ᴴ · Bᵢ,ⱼ(x', z) · exp(⟨x, z⟩ + ⟨x', z⟩*) · g_σᵢ(z) dz

where:
- The superscript H denotes conjugate transpose
- g_σᵢ(z) is the Gaussian density on Vᵢ with scale σᵢ
- The * denotes complex conjugation

For a real-valued GP, we take the real part: k(x, x') = Re[k_EPGP(x, x')].

### 3.3 Key Property: Exact PDE Satisfaction

**Theorem (Härkönen, Lange-Hegermann, Raiţă 2023).** If f ~ GP(0, k_EPGP), then A(∂)f = 0 almost surely. That is, every realization of the GP is an exact solution of the PDE system.

**Proof sketch.** Differentiation commutes with the integral. Applying A(∂) to exp(⟨x, z⟩) yields A(z) · exp(⟨x, z⟩). Since z ∈ V = { z : det A(z) = 0 }, the expression A(z) · B(x,z) = 0 for each Noetherian multiplier B. Therefore A(∂)f(x) = 0.

### 3.4 Hyperparameters

The EPGP kernel has the following trainable parameters:

- σᵢ² (scale parameters): One per variety component. Controls the "bandwidth" of frequencies used on each Vᵢ. Larger σᵢ → more high-frequency content → sharper features. Smaller σᵢ → smoother solutions.

- Observation noise σ₀²: Standard GP noise parameter. Accounts for measurement error in the data.

- (Optional) Mean and shift parameters for the Gaussian measure on V.

Total parameter count is typically O(s) where s is the number of irreducible components — dramatically fewer than neural networks (which need ~10⁵ parameters) or even standard GP kernels (which may need dozens of length scales).

### 3.5 GP Posterior (Standard)

Given observations y = [y₁, ..., yₘ]ᵀ at points X = [x₁, ..., xₘ], with y = f(X) + ε, ε ~ N(0, σ₀²I):

Posterior mean:    μ(x) = k(x, X) · [K(X, X) + σ₀²I]⁻¹ · y
Posterior variance: σ²(x) = k(x, x) - k(x, X) · [K(X, X) + σ₀²I]⁻¹ · k(X, x)

where K(X, X) is the m×m Gram matrix with entries k(xᵢ, xⱼ).

**Computational cost:** O(m³) for inversion, O(m²) per prediction point. For m ~ 100 (typical for sparse option chains), this is milliseconds.

### 3.6 Explicit EPGP Kernel for the 1D Heat Equation

> **This section is the computational foundation of the entire system.** Every downstream computation — posterior mean, variance, confidence scores — reduces to evaluating this kernel. Implementers should validate this section first before proceeding.

**3.6.1 Setup**

For the heat equation ∂ₜu = ∂²ₓu, the characteristic variety is V = {(z₁, z₂) ∈ C² : z₁ = z₂²}, parametrized by ω ∈ C via z = (ω², ω). The Noetherian multiplier is B = 1 (Section 2.6). An exponential solution is:

φ_ω(x, t) = exp(ω²t + ωx)

We impose a Gaussian measure on ω. Writing ω = a + bi for a, b ∈ R, the measure is:

dμ(ω) = (1/(2πσ²)) · exp(-(a² + b²)/(2σ²)) da db

**3.6.2 Kernel as integral**

k(x, t; x', t') = ∫∫_{R²} φ_ω(x, t) · φ_ω̄(x', t') · dμ(ω)

where ω̄ = a - bi (conjugate, to ensure real-valued kernel). Expanding:

φ_ω(x, t) · φ_ω̄(x', t') = exp(ω²t + ωx + ω̄²t' + ω̄x')

Substituting ω = a + bi:

- ω² = a² - b² + 2abi
- ω̄² = a² - b² - 2abi
- ω²t + ω̄²t' = (a² - b²)(t + t') + 2abi(t - t')
- ωx + ω̄x' = a(x + x') + bi(x - x')

So the full exponent in the integrand is:

E = (a² - b²)(t + t') + 2abi(t - t') + a(x + x') + bi(x - x') - (a² + b²)/(2σ²)

**3.6.3 Separating real and imaginary parts**

Define shorthand:

- T₊ = t + t',  T₋ = t - t'
- X₊ = x + x',  X₋ = x - x'
- λ = 1/(2σ²)

Real part of E:

Re(E) = a²(T₊ - λ) - b²(T₊ + λ) + aX₊

Imaginary part of E:

Im(E) = 2abT₋ + bX₋

The kernel is k = (1/(2πσ²)) · Re[∫∫ exp(E) da db].

**3.6.4 Evaluating the 2D Gaussian integral**

Write the exponent as a quadratic form in (a, b). Collecting terms:

E = -[λ - T₊]a² - [λ + T₊]b² + 2i·T₋·ab + X₊·a + iX₋·b

This is a complex quadratic form. Define the 2×2 precision matrix:

M = [ (λ - T₊),    -iT₋   ]
    [ -iT₋,    (λ + T₊) ]

and linear term vector h = [X₊, iX₋]ᵀ, so that E = -vᵀMv + hᵀv where v = (a, b)ᵀ.

The integral evaluates to the standard multivariate Gaussian result (valid when Re(M) is positive definite, i.e., λ > |T₊| which requires σ² < 1/(2|T₊|)):

∫∫ exp(E) da db = π / √det(M) · exp(¼ hᵀ M⁻¹ h)

Computing det(M):

det(M) = (λ - T₊)(λ + T₊) - (-iT₋)² = λ² - T₊² - T₋²

Since T₊² + T₋² = (t + t')² + (t - t')² = 2(t² + t'²):

det(M) = λ² - 2(t² + t'²) = 1/(4σ⁴) - 2(t² + t'²)

Computing M⁻¹:

M⁻¹ = (1/det(M)) · [ (λ + T₊),   iT₋   ]
                      [ iT₋,    (λ - T₊) ]

Computing hᵀ M⁻¹ h:

hᵀ M⁻¹ h = (1/det(M)) · [(λ + T₊)X₊² + 2iT₋ · X₊ · iX₋ + (λ - T₊)(iX₋)²]
           = (1/det(M)) · [(λ + T₊)X₊² - 2T₋X₊X₋ - (λ - T₊)X₋²]

Substituting back X₊ = x + x', X₋ = x - x', T₊ = t + t', T₋ = t - t':

**3.6.5 Final closed-form kernel**

k_heat(x, t; x', t') = (1/(2σ²)) · (1/√D) · exp(Q / (4D))

where:

D = 1/(4σ⁴) - 2(t² + t'²)

Q = (1/(2σ²) + t + t')(x + x')² - 2(t - t')(x + x')(x - x') - (1/(2σ²) - t - t')(x - x')²

**Validity condition:** D > 0, equivalently σ² < 1 / (2√(2(t² + t'²))). For small τ values typical of options (τ < 0.5), this requires σ < ~1.0, which is within the normal hyperparameter range. If this condition is violated, increase λ (decrease σ) or use the numerical quadrature fallback.

**3.6.6 Numerical quadrature fallback**

When the closed form is unwieldy, numerically unstable, or for verification, evaluate the kernel via 2D Gauss-Hermite quadrature over (a, b) ∈ R²:

```python
import numpy as np
from numpy.polynomial.hermite_e import hermegauss

def epgp_heat_kernel(x, t, xp, tp, sigma, n_quad=30):
    """Evaluate EPGP kernel for 1D heat equation via Gauss-Hermite quadrature.

    Parameters
    ----------
    x, t : float — spatial and temporal coordinate of first point
    xp, tp : float — spatial and temporal coordinate of second point
    sigma : float — kernel scale parameter (bandwidth on characteristic variety)
    n_quad : int — quadrature order per dimension (total points = n_quad²)

    Returns
    -------
    float — kernel value k(x, t; xp, tp)
    """
    nodes, weights = hermegauss(n_quad)
    # Scale: probabilist's Hermite polynomials, so nodes are for exp(-x²/2)
    a_vals = sigma * nodes
    b_vals = sigma * nodes
    w_a = weights
    w_b = weights

    result = 0.0
    for i in range(n_quad):
        a = a_vals[i]
        for j in range(n_quad):
            b = b_vals[j]
            omega = complex(a, b)
            omega_bar = complex(a, -b)
            exponent = omega**2 * t + omega * x + omega_bar**2 * tp + omega_bar * xp
            result += w_a[i] * w_b[j] * np.exp(exponent).real

    # Normalize: hermegauss weights sum to sqrt(2*pi), so divide by (2*pi*sigma^2)
    return result / (2 * np.pi * sigma**2)


def epgp_heat_gram_matrix(X, sigma, noise_var, n_quad=30):
    """Compute the Gram matrix K(X, X) + noise_var * I for a set of points.

    Parameters
    ----------
    X : ndarray of shape (m, 2) — each row is (x_i, t_i)
    sigma : float — kernel scale
    noise_var : float — observation noise variance σ₀²
    n_quad : int — quadrature order

    Returns
    -------
    ndarray of shape (m, m) — the regularized Gram matrix
    """
    m = X.shape[0]
    K = np.zeros((m, m))
    for i in range(m):
        for j in range(i, m):
            K[i, j] = epgp_heat_kernel(X[i, 0], X[i, 1], X[j, 0], X[j, 1], sigma, n_quad)
            K[j, i] = K[i, j]
    K += noise_var * np.eye(m)
    return K
```

**Performance:** Quadrature order N = 30 per dimension (900 evaluations) gives relative error < 10⁻¹² for typical parameter ranges. For m = 100 observation points, the full Gram matrix requires m(m+1)/2 × N² ≈ 4.5 × 10⁶ exponential evaluations — sub-second on modern hardware. Vectorization with NumPy broadcasting can reduce this by 10-100×.

**3.6.7 Verification checklist**

The kernel implementation must pass two sanity checks before any downstream use:

1. **PDE satisfaction:** For any fixed (x', t'), the function g(x, t) = k(x, t; x', t') must satisfy ∂ₜg = ∂²ₓg. Verify numerically:
   - Compute ∂ₜg via finite differences: (k(x, t+ε; x', t') - k(x, t-ε; x', t')) / (2ε)
   - Compute ∂²ₓg via finite differences: (k(x+ε, t; x', t') - 2k(x, t; x', t') + k(x-ε, t; x', t')) / ε²
   - Confirm |∂ₜg - ∂²ₓg| < 10⁻⁶ for ε = 10⁻⁵ across a grid of test points.

2. **Positive semi-definiteness:** The Gram matrix K(X, X) must have all non-negative eigenvalues for any set of points X. Generate 10 random point sets of size m = 50, compute K, check min eigenvalue ≥ -10⁻¹⁰ (allowing for numerical noise).

3. **Symmetry:** k(x, t; x', t') = k(x', t'; x, t). Verify |k(a; b) - k(b; a)| < 10⁻¹² on random pairs.

---

### 3.7 Hyperparameter Optimization

**3.7.1 Objective**

The log marginal likelihood for GP regression:

L(θ) = -½ yᵀ (K_θ + σ₀²I)⁻¹ y - ½ log|K_θ + σ₀²I| - m/2 log(2π)

where θ = {σ² (kernel scale), σ₀² (noise variance)} for EPGP, or θ = {z₁, ..., zᵣ, σ², σ₀²} for S-EPGP.

**3.7.2 Optimizer Selection**

| Variant | Parameters | Optimizer | Rationale |
|---------|-----------|-----------|-----------|
| EPGP | 2 (σ², σ₀²) | L-BFGS-B with bounds | Low-dimensional, smooth. Gradient available analytically. Bounded to prevent σ² → 0 or σ₀² → 0. |
| S-EPGP | 2r + 2 (frequencies + scales + noise) | Adam (1000 steps) → L-BFGS-B (refinement) | Higher-dimensional, frequency parameters can be non-smooth. Adam for exploration, L-BFGS for polishing. |

**3.7.3 Initialization Strategy**

Non-convexity means initialization matters. Use multi-start:

1. Generate N_init = 10 random starting points for θ:
   - σ² sampled log-uniformly from [0.1, 10.0]
   - σ₀² sampled log-uniformly from [10⁻⁴, 10⁻¹]
   - For S-EPGP: frequency locations z_k initialized by sampling uniformly on V within a bounding box determined by the data extent
2. Run optimizer from each starting point for a reduced number of iterations (e.g., 100 for L-BFGS).
3. Select the top 3 by marginal likelihood value.
4. Run full optimization from these 3 starting points.
5. Return the θ with highest marginal likelihood.

**3.7.4 Bounds and Constraints**

| Parameter | Lower Bound | Upper Bound | Rationale |
|-----------|-------------|-------------|-----------|
| σ² | 0.01 | 100.0 | Prevent degenerate kernel (σ² → 0 collapses to zero function) |
| σ₀² | 10⁻⁶ | 1.0 | Noise must be positive; upper bound prevents fitting pure noise |
| z_k (S-EPGP) | On V | On V | Variety constraint enforced by parametrization (e.g., z₁ = z₂² for heat equation). Only z₂ is free. |

For S-EPGP, the variety constraint is enforced not by constrained optimization but by reparametrization: optimize over the free parameter (e.g., z₂ ∈ C for heat equation) and compute the dependent parameter (z₁ = z₂²) deterministically.

**3.7.5 Gradient Computation**

Analytic gradients of the log marginal likelihood with respect to θ:

∂L/∂θⱼ = ½ yᵀ K⁻¹ (∂K/∂θⱼ) K⁻¹ y - ½ tr(K⁻¹ ∂K/∂θⱼ)

where K = K_θ + σ₀²I and K⁻¹ is computed via Cholesky (reusing the factorization from L evaluation). The kernel derivative ∂K/∂σ² is computable in closed form for the EPGP kernel (differentiate the quadrature integrand with respect to σ, or differentiate the closed-form expression from Section 3.6.5). For S-EPGP, automatic differentiation (PyTorch) handles gradients with respect to frequency parameters.

**3.7.6 Practical Defaults**

For Phase 1–2 (m < 200, EPGP with 2 hyperparameters):
- Multi-start with N_init = 10 is sufficient (10 × L-BFGS runs, each < 100ms).
- Total optimization time: < 1 second per options chain snapshot.
- This is fast enough for daily recalibration.

For S-EPGP with r = 50 frequencies (102 parameters):
- Adam with learning rate 10⁻³, 2000 steps, followed by L-BFGS refinement.
- Total optimization time: ~10–30 seconds per snapshot.
- Still feasible for daily recalibration, but too slow for intraday.

```python
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import cho_factor, cho_solve

def neg_log_marginal_likelihood(log_theta, X, y, kernel_fn, n_quad=30):
    """Negative log marginal likelihood for EPGP hyperparameter optimization.

    Parameters
    ----------
    log_theta : ndarray of shape (2,) — [log(σ²), log(σ₀²)]
    X : ndarray of shape (m, 2) — observation points (x_i, t_i)
    y : ndarray of shape (m,) — observed values
    kernel_fn : callable — kernel function k(x, t, xp, tp, sigma, n_quad)
    n_quad : int — quadrature order

    Returns
    -------
    float — negative log marginal likelihood (to be minimized)
    """
    sigma_sq, noise_var = np.exp(log_theta)
    sigma = np.sqrt(sigma_sq)
    m = len(y)

    # Build Gram matrix
    K = np.zeros((m, m))
    for i in range(m):
        for j in range(i, m):
            K[i, j] = kernel_fn(X[i, 0], X[i, 1], X[j, 0], X[j, 1], sigma, n_quad)
            K[j, i] = K[i, j]

    # Regularize
    jitter = 1e-6 * np.trace(K) / m
    K_reg = K + (noise_var + jitter) * np.eye(m)

    # Cholesky
    try:
        L, low = cho_factor(K_reg)
    except np.linalg.LinAlgError:
        return 1e10  # Return large value for non-PD matrices

    alpha = cho_solve((L, low), y)
    nll = 0.5 * y @ alpha + np.sum(np.log(np.diag(L))) + 0.5 * m * np.log(2 * np.pi)
    return nll


def optimize_epgp_hyperparams(X, y, kernel_fn, n_starts=10, n_quad=30):
    """Multi-start L-BFGS-B optimization of EPGP hyperparameters.

    Returns
    -------
    dict with keys 'sigma_sq', 'noise_var', 'nll' (best negative log marginal likelihood)
    """
    best_result = None
    best_nll = np.inf

    for _ in range(n_starts):
        log_sigma_sq_init = np.random.uniform(np.log(0.1), np.log(10.0))
        log_noise_init = np.random.uniform(np.log(1e-4), np.log(1e-1))
        x0 = np.array([log_sigma_sq_init, log_noise_init])

        result = minimize(
            neg_log_marginal_likelihood,
            x0,
            args=(X, y, kernel_fn, n_quad),
            method='L-BFGS-B',
            bounds=[(np.log(0.01), np.log(100.0)), (np.log(1e-6), np.log(1.0))],
            options={'maxiter': 200}
        )

        if result.fun < best_nll:
            best_nll = result.fun
            best_result = result

    sigma_sq, noise_var = np.exp(best_result.x)
    return {'sigma_sq': sigma_sq, 'noise_var': noise_var, 'nll': best_nll}
```

---

## 4. S-EPGP: Sparse Variant with Learned Frequencies

### 4.1 Motivation

When the integral in the EPGP kernel does not admit a closed form — which is common for PDEs more complex than the heat equation — S-EPGP replaces the continuous integral with a finite sum over learned frequency points.

### 4.2 Construction

Define the S-EPGP prior with realizations of the form:

f(x) = Σⱼ₌₁ᵐ Σᵢ₌₁ʳ Cᵢ,ⱼ · Bⱼ(x, zᵢ,ⱼ) · exp(⟨x, zᵢ,ⱼ⟩)

where:
- zᵢ,ⱼ ∈ V are spectral frequency points on the characteristic variety (learnable)
- Cᵢ,ⱼ ~ N(0, Σ) are random coefficients
- Σ = diag(σ₁², ..., σₛ²) is a diagonal covariance matrix
- m is the number of Noetherian multipliers
- r is the number of frequency points per multiplier (user-specified)

### 4.3 S-EPGP Covariance Kernel

k_S-EPGP(x, x') = φ(x)ᴴ · Σ · φ(x')

where φ(x) is the feature vector:

φ(x) = [B₁(x, z₁,₁)exp(⟨x, z₁,₁⟩), ..., Bₘ(x, zᵣ,ₘ)exp(⟨x, zᵣ,ₘ⟩)]ᵀ

### 4.4 Learning Frequencies

The spectral points {zᵢ,ⱼ} are learned by maximizing the GP marginal likelihood:

log p(y | X, θ) = -½ yᵀ(K + σ₀²I)⁻¹y - ½ log|K + σ₀²I| - m/2 log(2π)

with respect to θ = {zᵢ,ⱼ, σⱼ², σ₀²}, subject to the constraint zᵢ,ⱼ ∈ V.

The variety constraint can be enforced by parametrizing V. For example, for the heat equation with V = {z₁ = z₂²}, we optimize over z₂ freely and set z₁ = z₂².

### 4.5 Connection to Sparse Spectrum GPs

S-EPGP is precisely a Sparse Spectrum GP (Lázaro-Gredilla et al. 2010) restricted to the characteristic variety V. Standard Sparse Spectrum GPs place frequencies anywhere in Rⁿ; S-EPGP constrains them to V. This constraint encodes exact PDE satisfaction while retaining the scalability advantage of sparse methods.

### 4.6 When to Use EPGP vs S-EPGP

| Criterion | EPGP | S-EPGP |
|-----------|------|--------|
| Integral tractable? | Required (closed form or quadrature) | Not needed |
| Data size | Small to moderate (m < 1000) | Any size |
| Solution accuracy | Optimal (all frequencies) | Depends on r (number of learned points) |
| Frequency learning | No (fixed Gaussian measure) | Yes (marginal likelihood optimization) |
| Computational cost | O(m³) kernel evaluation + integral | O(m·r²) for kernel, O(m³) for GP |

For financial applications with m ~ 50-200 observation points and relatively simple PDEs (heat, wave), EPGP is preferred. S-EPGP becomes necessary for higher-dimensional problems or when the integral is intractable.

---

## 5. B-EPGP: Boundary-Constrained Extension

### 5.1 Motivation

Financial PDEs come with boundary conditions. A European put option satisfies:

- V(S, T) = max(K - S, 0) as t → T (terminal payoff)
- V(0, t) = K·exp(-r(T-t)) as S → 0 (deep in-the-money)
- V(S, t) → 0 as S → ∞ (deep out-of-the-money)

Standard EPGP satisfies the PDE but not these boundary conditions. B-EPGP (Huang, Härkönen, Lange-Hegermann, Raiţă 2024) extends EPGP to incorporate linear boundary conditions.

### 5.2 Construction

Given PDE system A(∂)f = 0 on domain Ω with boundary conditions B(∂)f|_{∂Ω} = g:

1. Compute the Ehrenpreis-Palamodov data (V, {Bᵢ,ⱼ}) for A.
2. For each boundary condition, restrict the exponential-polynomial basis to the boundary and solve for combinations that satisfy B(∂)f|_{∂Ω} = g.
3. Construct modified basis functions that satisfy both PDE and boundary conditions.
4. Build the GP kernel from these modified basis functions.

### 5.3 Key Result

**Theorem (Huang et al. 2024).** B-EPGP realizations satisfy both the PDE A(∂)f = 0 and the boundary conditions B(∂)f|_{∂Ω} = g exactly. The posterior mean is an exact solution of the boundary value problem.

### 5.4 Significance for Finance

In the heat equation domain (after BS transformation), the boundary conditions become:

- u(x, 0) = g(x) (transformed payoff at τ = 0)
- u(x, τ) → 0 as x → ±∞ (far-field decay)

B-EPGP can encode these exactly, meaning the GP prior already knows the payoff structure and far-field behavior. Data only needs to inform the solution in the interior — dramatically reducing the data requirement.

---

## 6. Financial PDE Pipeline: Black-Scholes to Heat Equation

### 6.1 The Black-Scholes PDE

For a European option with price V(S, t), underlying price S, time t, risk-free rate r, and constant volatility σ:

∂V/∂t + ½σ²S²·∂²V/∂S² + rS·∂V/∂S - rV = 0

This is a linear PDE but with **variable coefficients** (the S² and S terms). EPGP requires constant coefficients. The classical coordinate transformation resolves this.

### 6.2 Coordinate Transformation

Define new variables:

x = ln(S/K)                    (log-moneyness)
τ = ½σ²(T - t)                (reversed, scaled time)
V(S, t) = K · exp(-αx - βτ) · u(x, τ)

where:

α = -½(2r/σ² - 1) = -½(k - 1)     with k = 2r/σ²
β = -¼(2r/σ² + 1)² = -¼(k + 1)²

### 6.3 Result: Heat Equation

Under this transformation, the BS PDE becomes:

∂u/∂τ = ∂²u/∂x²

This is the 1D heat equation — linear, constant coefficients. EPGP applies directly.

### 6.4 Boundary Condition Transformation

The European put payoff V(S, T) = max(K - S, 0) transforms to:

u(x, 0) = exp(½(k-1)x) · max(1 - exp(x), 0) / K
         = max(exp(½(k+1)x) - exp(½(k-1)x), 0)    [after simplification]

The far-field conditions become:

u(x, τ) → exp(½(k+1)x) - exp(½(k-1)x)   as x → -∞ (deep ITM)
u(x, τ) → 0                                as x → +∞ (deep OTM)

### 6.5 Inverse Transformation

Given EPGP posterior mean μ_u(x, τ) and variance σ²_u(x, τ) in heat equation coordinates, transform back to BS coordinates:

μ_V(S, t) = K · exp(-αx - βτ) · μ_u(x, τ)

For the variance, using the delta method (since the transformation is smooth and monotone):

σ²_V(S, t) ≈ [K · exp(-αx - βτ)]² · σ²_u(x, τ)

This gives a confidence interval for the option price in the original BS coordinates.

### 6.6 Multi-Asset Extension (2D Heat)

For a two-asset option with prices S₁, S₂, correlation ρ:

xᵢ = ln(Sᵢ/Kᵢ),  τ = ½σ₁²(T - t)

The transformed PDE becomes:

∂u/∂τ = ∂²u/∂x₁² + 2ρ(σ₂/σ₁)·∂²u/∂x₁∂x₂ + (σ₂/σ₁)²·∂²u/∂x₂²

This is still linear constant-coefficient (assuming constant σ₁, σ₂, ρ). EPGP applies. The characteristic variety is a 2D surface in C³.

### 6.7 Numerical Stability

**6.7.1 Overflow/Underflow in Exponential Prefactor**

The inverse transformation V = K · exp(-αx - βτ) · u involves exp(-αx). For typical parameters (r = 0.05, σ = 0.20), α ≈ 0.75. At deep OTM (x = ln(S/K) = -3, corresponding to S/K ≈ 0.05), exp(-αx) ≈ exp(2.25) ≈ 9.5 — manageable. At extreme OTM (x = -6), exp(-αx) ≈ exp(4.5) ≈ 90 — still fine.

The dangerous direction is deep ITM: x = +6 gives exp(-αx) ≈ exp(-4.5) ≈ 0.01. Combined with u values that may be large, this can cause precision loss.

**Mitigation:** Work in log-space for the inverse transformation:

log V = log K - αx - βτ + log u

Compute log V first, then exponentiate only at the final step. For variance propagation:

log σ²_V = 2(log K - αx - βτ) + log σ²_u

This avoids intermediate overflow/underflow entirely.

```python
def inverse_transform_logspace(x, tau, log_u, log_var_u, K, alpha, beta):
    """Transform EPGP posterior from heat coordinates back to BS coordinates.

    Works in log-space to avoid overflow/underflow.

    Parameters
    ----------
    x : ndarray — log-moneyness values
    tau : ndarray — transformed time values
    log_u : ndarray — log of EPGP posterior mean in heat coordinates
    log_var_u : ndarray — log of EPGP posterior variance in heat coordinates
    K : float — strike price
    alpha, beta : float — BS transformation parameters

    Returns
    -------
    V : ndarray — option prices in BS coordinates
    var_V : ndarray — option price variances in BS coordinates
    """
    log_prefactor = np.log(K) - alpha * x - beta * tau
    log_V = log_prefactor + log_u
    log_var_V = 2 * log_prefactor + log_var_u
    return np.exp(log_V), np.exp(log_var_V)
```

**6.7.2 Delta Method Breakdown Near the Strike**

The delta method σ²_V ≈ [K·exp(-αx - βτ)]² · σ²_u assumes the transformation is approximately linear over the posterior uncertainty range. Near x = 0 and τ → 0 (at-the-money near expiry), the option payoff has a kink, and the heat equation solution u(x, τ) has a sharp gradient. The linear approximation breaks.

**Mitigations (in order of recommended implementation):**

- **Option 1 (recommended for initial implementation):** Exclude near-expiry options from EPGP analysis. For τ < τ_min (corresponding to DTE < 3 days), do not compute confidence scores. The confidence filter already avoids low-confidence regions; near-expiry is naturally one of them.

- **Option 2 (for validation):** Use Monte Carlo sampling from the GP posterior. Draw N = 1000 samples from p(u | data), transform each sample to V, compute empirical variance. Exact but slower (~100× cost of delta method).

- **Option 3 (for production):** Use second-order delta method (include the Hessian term). More accurate near the kink but algebraically complex. Only implement if Options 1–2 prove insufficient.

**6.7.3 Gram Matrix Conditioning**

The Gram matrix K(X, X) + σ₀²I can become ill-conditioned when observation points are very close together (near-duplicate strikes), the kernel scale σ is large relative to data spacing, or m is large (m > 200).

**Mitigations:**

1. **Jitter term:** Add K + (σ₀² + ε)I where ε = 10⁻⁶ · trace(K) / m. This is standard in GP implementations.

2. **Cholesky decomposition:** Use Cholesky (not direct inversion) for solving linear systems. Cholesky is numerically stable and provides a natural check: if the decomposition fails, the matrix is too ill-conditioned.

3. **Condition monitoring:** If cond(K + σ₀²I) > 10¹⁰, log a warning and increase σ₀² or reduce m by removing near-duplicate points.

4. **Scalability fallback:** For m > 500 (unlikely in options but possible in crypto): use sparse GP approximations (inducing points) or the S-EPGP formulation, which has O(mr²) cost instead of O(m³).

```python
import numpy as np
from scipy.linalg import cho_factor, cho_solve

def gp_posterior(K_XX, K_Xs, k_ss, y, noise_var):
    """Numerically stable GP posterior via Cholesky decomposition.

    Parameters
    ----------
    K_XX : ndarray (m, m) — Gram matrix at training points
    K_Xs : ndarray (m,) — cross-covariance between training and test point
    k_ss : float — prior variance at test point
    y : ndarray (m,) — observed values
    noise_var : float — observation noise variance σ₀²

    Returns
    -------
    mu : float — posterior mean at test point
    var : float — posterior variance at test point
    """
    m = K_XX.shape[0]
    jitter = 1e-6 * np.trace(K_XX) / m
    L_matrix = K_XX + (noise_var + jitter) * np.eye(m)

    try:
        L, low = cho_factor(L_matrix)
    except np.linalg.LinAlgError:
        # Fallback: increase jitter aggressively
        L_matrix += 1e-4 * np.eye(m)
        L, low = cho_factor(L_matrix)

    alpha = cho_solve((L, low), y)
    mu = K_Xs @ alpha

    v = cho_solve((L, low), K_Xs)
    var = k_ss - K_Xs @ v
    return mu, max(var, 0.0)  # Clamp to non-negative
```

---

## 7. Forward Problem: Option Pricing with Uncertainty Quantification

### 7.0 Data Sources and Pipeline

**7.0.1 Phase 1: Synthetic Data (No External Source)**

Generate test data from the known analytical solution to the heat equation (or BS closed-form). Add Gaussian noise with known σ₀. This provides a controlled environment where the true solution is known, enabling exact validation of the EPGP implementation.

```python
import numpy as np

def heat_equation_gaussian_ic(x, t, x0=0.5, width=50.0):
    """True solution to heat equation with Gaussian initial condition.

    u(x, 0) = exp(-width * (x - x0)^2)
    """
    return (1.0 / np.sqrt(1 + 4 * width * t)) * np.exp(
        -width * (x - x0) ** 2 / (1 + 4 * width * t)
    )

def bs_put_price(S, K, T, r, sigma):
    """Black-Scholes European put price (closed-form benchmark)."""
    from scipy.stats import norm
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

# Example: sparse noisy observations for Phase 1 testing
x_obs = np.array([0.2, 0.35, 0.5, 0.65, 0.8])
t_obs = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
noise_std = 0.01
y_obs = heat_equation_gaussian_ic(x_obs, t_obs) + np.random.normal(0, noise_std, len(x_obs))
```

**7.0.2 Phase 2+: Historical Options Data**

| Source | Coverage | Cost | Fields | Use |
|--------|----------|------|--------|-----|
| OptionsDX | SPX EOD 2020–2023 | Free (static CSV files) | Strike, expiry, bid, ask, IV, delta, gamma, theta, vega, volume, OI | Primary backtest (in-sample) |
| Polygon.io | SPX EOD 2024+ | $29/mo API | Strike, expiry, close, bid, ask, IV | Out-of-sample validation |
| CBOE | VIX, VIX3M, VVIX | Free | Daily close | Regime features for HMM |
| Yahoo Finance | SPX daily OHLCV | Free | ^GSPC | Underlying price |
| FRED | Risk-free rate (DGS10) | Free | Daily | BS transformation parameter r |

**7.0.3 Options Chain Processing**

For each trading date t:

1. Load the full options chain: all puts with 7 ≤ DTE ≤ 60.
2. Filter: bid > 0, volume > 0, 0.01 ≤ |delta| ≤ 0.50 (exclude deep ITM and near-zero delta).
3. Compute mid price: V_mid = (bid + ask) / 2.
4. For each (K, T) pair, transform to heat coordinates: x = ln(S/K), τ = ½σ²(T - t).
   - Use ATM implied vol as σ for the transformation (or VIX/100 as proxy).
   - Compute u = V_mid / (K · exp(-αx - βτ)).
5. Result: observation set {(xᵢ, τⱼ, uᵢⱼ)} ready for EPGP.

```python
import numpy as np

def options_chain_to_heat_coords(S, strikes, expiries, mid_prices, sigma_atm, r):
    """Transform options chain to heat equation coordinates.

    Parameters
    ----------
    S : float — current underlying price
    strikes : ndarray — strike prices K
    expiries : ndarray — time to expiration T (in years)
    mid_prices : ndarray — option mid prices V_mid
    sigma_atm : float — ATM implied vol (used for coordinate transform)
    r : float — risk-free rate

    Returns
    -------
    x : ndarray — log-moneyness
    tau : ndarray — transformed time
    u : ndarray — transformed option values (heat equation domain)
    """
    k_param = 2 * r / sigma_atm**2
    alpha = -0.5 * (k_param - 1)
    beta = -0.25 * (k_param + 1)**2

    x = np.log(S / strikes)
    tau = 0.5 * sigma_atm**2 * expiries
    u = mid_prices / (strikes * np.exp(-alpha * x - beta * tau))
    return x, tau, u
```

**7.0.4 Missing Data Handling**

Deep OTM strikes often have no bid (bid = 0) or very wide spreads (ask/bid > 3). These are excluded from the observation set but remain in the prediction grid — they are precisely the points where EPGP posterior variance is most informative.

When fewer than 10 valid observations remain after filtering for a given maturity, that maturity is excluded from the EPGP fit for the current date.

**7.0.5 Historical Data for Crypto (Phase 4)**

| Source | Coverage | Cost | Fields |
|--------|----------|------|--------|
| Binance API | Perpetual funding rate history | Free | Funding rate, timestamp, 8h intervals |
| Bybit API | Perpetual funding rate history | Free | Funding rate, timestamp, 8h intervals |
| OKX API | Perpetual funding rate history | Free | Funding rate, timestamp, 8h intervals |
| CoinGecko / CryptoCompare | BTC/ETH spot price history | Free | OHLCV daily |

All crypto data sources provide free API access with rate limits sufficient for historical backfill. Funding rates are deterministic (computed by the exchange formula) and publicly available for the full history of each perpetual contract.

### 7.1 Problem Statement

Given: A sparse set of observed option prices {(Kᵢ, Tⱼ, V^{mkt}_{ij})} for various strikes and maturities.
Find: The option price surface V(S, t; K, T) with pointwise uncertainty estimates.

### 7.2 Pipeline

Step 1: Transform observed prices to heat equation coordinates.
  (xᵢ, τⱼ, uᵢⱼ) = Transform(Kᵢ, Tⱼ, V^{mkt}_{ij})

Step 2: Construct EPGP kernel using the hardcoded variety V = {z₁ = z₂²} for the heat equation (Section 2.6).
  Parametrize z₂ = a + bi for a, b ∈ R. Evaluate via closed form (Section 3.6.5) or quadrature (Section 3.6.6).
  k(·, ·) = k_EPGP or k_S-EPGP. No algebraic preprocessing needed.

Step 3: Optimize hyperparameters (σ², σ₀²) by maximizing marginal likelihood.
  θ* = argmax_θ log p(u | X, θ)

Step 4: Compute posterior mean and variance on a dense grid.
  μ_u(x, τ), σ²_u(x, τ) for (x, τ) on grid

Step 5: Transform back to BS coordinates.
  μ_V(S, t; K, T) = K · exp(-αx - βτ) · μ_u(x, τ)
  σ²_V(S, t; K, T) ≈ [K · exp(-αx - βτ)]² · σ²_u(x, τ)

Step 6: Extract confidence intervals.
  95% CI for V: [μ_V - 1.96·σ_V, μ_V + 1.96·σ_V]

### 7.3 Data Efficiency

A key advantage: EPGP provides meaningful posterior uncertainty even with very few data points. With m = 10 option prices, the GP posterior knows:
- Where data constrains the solution (low variance near observed strikes/maturities)
- Where the solution is extrapolated (high variance in deep OTM or long-dated regions)
- The uncertainty is quantitatively calibrated (95% intervals should contain true values 95% of the time)

### 7.4 Implied Volatility Surface with Uncertainty

From the option price posterior, derive an implied volatility surface with uncertainty:

σ_impl(K, T) = BS⁻¹(μ_V(S, t; K, T))

The uncertainty in σ_impl can be obtained by propagating σ²_V through the inverse BS formula via the delta method:

σ²_{impl}(K, T) ≈ σ²_V / (∂V/∂σ)² = σ²_V / Vega²

This gives confidence bands on the entire implied volatility surface.

---

## 8. Inverse Problem: Volatility Surface Calibration via EPGP

### 8.1 Problem Statement

The forward problem assumes constant σ (via the BS → heat transformation). In reality, σ varies with strike and maturity. The inverse problem asks: **given market option prices, what effective volatility σ(K, T) generated them?**

### 8.2 EPGP Approach to Inversion

Following Li, Lange-Hegermann, and Raiţă (2025), we treat the PDE parameter as unknown:

∂u/∂τ = c · ∂²u/∂x²

where c is an unknown diffusion coefficient (related to local volatility). The EPGP framework is extended to jointly estimate the solution u(x, τ) and the parameter c from data.

For multiple regions of the vol surface, we can allow c to vary piecewise:

c_ATM for |x| < δ (at-the-money region)
c_OTM for x > δ (out-of-the-money puts)
c_ITM for x < -δ (in-the-money region)

Each region has its own EPGP posterior for c, with uncertainty quantification.

### 8.3 Joint Posterior

The joint posterior over solution and parameters is:

p(u, c | data) ∝ p(data | u) · p(u | c) · p(c)

where:
- p(data | u) is the likelihood (option prices given the solution)
- p(u | c) is the EPGP prior (solution must satisfy PDE with parameter c)
- p(c) is a prior on the diffusion coefficient

The posterior over c gives:
- Point estimate: E[c | data] — the most likely volatility parameter
- Uncertainty: Var[c | data] — how confident we are about that parameter
- Anomaly detection: sudden increase in Var[c | data] signals structural change

### 8.4 Iterative Refinement

For a full volatility surface σ(K, T):

1. Start with initial estimate σ₀ (e.g., ATM implied vol).
2. Transform BS → heat using σ₀.
3. Run EPGP, compute posterior mean and variance.
4. Identify regions where posterior variance is large → σ₀ is a poor fit there.
5. Update σ in those regions and re-transform.
6. Iterate until convergence.

This is a form of the iterative linearization discussed in Section 13, applied to the specific case of variable volatility.

---

## 9. Structural Break Detection: EPGP as Regime Sensor

### 9.1 Core Idea

When market data is generated by a constant-coefficient PDE, the EPGP posterior variance is small and stable. When the underlying process changes — regime transition, liquidity event, structural break — the data no longer conforms to the PDE, and the posterior variance increases.

This gives a **physics-based regime detection** mechanism, complementary to statistical methods like HMM.

### 9.2 Variance Monitoring

Define the aggregate model confidence at time t:

Φ(t) = (1/N) · Σᵢ₌₁ᴺ σ²_EPGP(xᵢ, t) / σ²_baseline(xᵢ)

where:
- σ²_EPGP(xᵢ, t) is the EPGP posterior variance at observation point xᵢ at time t
- σ²_baseline(xᵢ) is the historical average posterior variance at that point
- N is the number of observation points

Φ(t) ≈ 1 under normal conditions. Φ(t) >> 1 signals structural deviation.

### 9.3 Z-Score Anomaly Detection

Compute a rolling z-score of Φ(t):

Z_Φ(t) = (Φ(t) - μ_Φ) / σ_Φ

where μ_Φ and σ_Φ are the rolling mean and standard deviation of Φ over a lookback window (e.g., 63 trading days).

Signal interpretation:
- |Z_Φ| < 1: Normal regime. PDE structure holds. Trade normally.
- 1 < |Z_Φ| < 2: Elevated uncertainty. Reduce position size.
- |Z_Φ| > 2: Structural break likely. Exit positions or hedge.

### 9.4 Comparison with HMM Regime Detection

| Feature | HMM | EPGP Variance |
|---------|-----|---------------|
| Data used | Historical returns, VIX, term structure | Current option chain + PDE structure |
| Detection mechanism | Statistical state inference | Physical structure violation |
| Latency | Looks back (uses historical window) | Real-time (uses current snapshot) |
| False positives | Noise in features can trigger | Robust (PDE structure is stable under noise) |
| Blind spots | Misses structural changes not in features | Misses changes that don't affect PDE structure |
| Output | P(regime = k) | Confidence level Φ(t) |

### 9.5 Dual Detection Integration

Combine both signals:

Regime_combined(t) = w_HMM · P_HMM(HighVol, t) + w_EPGP · sigmoid(Z_Φ(t) - threshold)

where:
- P_HMM(HighVol, t) is the HMM posterior probability of the high-volatility regime
- sigmoid(Z_Φ(t) - threshold) converts the EPGP z-score to a [0,1] probability
- w_HMM and w_EPGP are weights (initially equal; can be optimized)

When both sensors agree, confidence in the regime signal is high. When they disagree, the system becomes conservative.

---

## 10. Confidence Filter: From Posterior Variance to Trading Signals

### 10.1 Pointwise Confidence Score

For each potential trade characterized by (strike K, maturity T, entry time t), compute:

C(K, T, t) = 1 - σ²_EPGP(K, T, t) / σ²_max

where σ²_max is the maximum posterior variance observed historically (or a theoretical upper bound). C ∈ [0, 1], where 1 = maximum confidence, 0 = no confidence.

### 10.2 Trade Entry Filter

A trade is eligible for entry only if:

C(K, T, t) > C_min  AND  Z_Φ(t) < Z_max

where C_min (e.g., 0.6) and Z_max (e.g., 2.0) are thresholds calibrated on historical data.

### 10.3 Position Sizing

When entry conditions are met, scale position size by confidence:

Size(K, T, t) = Base_Size · min(1, C(K, T, t) / C_target) · Regime_Scaler(t)

where:
- Base_Size is the Kelly-optimal position for the strategy
- C_target is the confidence level at which full size is taken (e.g., 0.8)
- Regime_Scaler ∈ [0.25, 1.0] is determined by the dual regime signal

### 10.4 Exit Signals

The EPGP confidence filter also generates exit signals:

- **Confidence degradation:** If C(K, T, t) for an open position drops below C_exit < C_min, exit.
- **Variance spike:** If Z_Φ(t) exceeds Z_emergency (e.g., 3.0), exit all positions.
- **Structural break:** If the EPGP posterior mean deviates from market prices by more than the 99% posterior interval, the model is misspecified — exit and recalibrate.

### 10.5 The A/B Test

The fundamental experiment: run the same base strategy with and without the EPGP confidence filter.

Strategy A: Base strategy (e.g., VRP) with standard entry/exit rules.
Strategy B: Same strategy, but EPGP confidence filter gates entries and sizes positions.

Compare:
- Sharpe ratio (B should be higher if filter removes bad trades)
- Maximum drawdown (B should be lower if filter avoids structural breaks)
- Win rate (B should be higher if filter avoids low-confidence trades)
- Number of trades (B will be lower — the question is whether fewer, better trades win)

---

## 11. Market Application I: Options (Variance Risk Premium)

### 11.1 Strategy Description

Systematically sell SPX put credit spreads to harvest the variance risk premium (IV > RV structurally). The EPGP confidence filter determines when and at what size to enter.

### 11.2 Data Pipeline

Weekly options chain for SPX:
- Strikes: K₁, ..., Kₙ (typically 20-50 liquid strikes)
- Maturities: T₁, ..., Tₘ (2-4 relevant expiries)
- Prices: V^{mkt}(Kᵢ, Tⱼ) (mid-quote or mark)
- Total observations per snapshot: ~50-200 points

### 11.3 EPGP Application

For each weekly snapshot:

1. Transform the options chain to heat equation coordinates: {(xᵢ, τⱼ, uᵢⱼ)}.
2. Construct EPGP kernel for the heat equation.
3. Compute posterior mean μ_u and variance σ²_u on a dense (x, τ) grid.
4. Transform back to get μ_V(K, T) and σ²_V(K, T).
5. Compute confidence scores for candidate trades.
6. Compare with regime signals.
7. Enter trades that pass all filters.

### 11.4 Specific Value of EPGP for Options

The OTM put strikes that VRP targets (10-delta) are precisely where:
- Market data is sparsest (few trades, wide bid-ask)
- EPGP posterior variance is most informative
- Standard interpolation methods (Heston, SABR) have the most model risk

EPGP tells you: "At this 10-delta strike, my confidence interval is [$2.80, $3.60]. The market mid is $3.20. This is well within my confidence band — the trade is consistent with PDE structure."

vs: "At this 10-delta strike, my confidence interval is [$1.50, $5.00]. The market mid is $3.20. I can't distinguish this from a wide range of possibilities — skip this trade."

### 11.5 VRP-Specific Metrics

Track EPGP-specific performance indicators:

- Confidence-weighted VRP: Does filtering by confidence select trades with higher realized premium?
- Variance spike frequency: How often does Z_Φ > 2 correctly predict VRP drawdowns?
- Strike-level accuracy: Do high-confidence strikes have tighter post-trade P&L distribution?

---

## 12. Market Application II: Crypto Perpetual Funding Rate

### 12.1 Strategy Description

Crypto perpetual futures have a funding rate F(t) that transfers between longs and shorts every 8 hours. When F(t) is positive, longs pay shorts — and vice versa. Funding rates exhibit mean reversion.

Strategy: When funding rate deviates significantly from its EPGP-estimated equilibrium, bet on mean reversion.

### 12.2 PDE Model

Model the funding rate dynamics as a diffusion process. The probability density p(F, t) of funding rates satisfies the Fokker-Planck equation:

∂p/∂t = -∂/∂F[μ(F)p] + ½ · ∂²/∂F²[σ²p]

For a simple Ornstein-Uhlenbeck process (linear mean reversion):

μ(F) = κ(θ - F)   (mean reversion toward θ with speed κ)
σ² = constant

This gives:

∂p/∂t = κ · ∂/∂F[(F - θ)p] + ½σ² · ∂²p/∂F²

After change of variables y = F - θ:

∂p/∂t = κ · ∂/∂y[y · p] + ½σ² · ∂²p/∂y²

This is a **linear PDE with constant coefficients** (the κ and σ² are constant within a regime). EPGP applies.

### 12.3 EPGP Application

1. Observe funding rates from multiple exchanges: {(exchange_i, time_j, F_ij)}.
2. These are sparse observations of the funding rate field.
3. Construct EPGP kernel for the Fokker-Planck equation.
4. Compute posterior mean μ_F(t) (equilibrium funding rate) and variance σ²_F(t).
5. When |F_market - μ_F| > n · σ_F (funding rate is outside confidence band), enter mean reversion trade.
6. When σ²_F spikes (structural uncertainty), reduce or skip.

### 12.4 Cross-Exchange Arbitrage Signal

With multiple exchanges:

EPGP jointly estimates the "fair" funding rate μ_F(t) from all exchange data. If exchange A has F_A >> μ_F with high confidence (low σ²_F), there is a cross-exchange mispricing opportunity. The confidence filter ensures we only act when the mispricing signal is statistically and structurally significant.

### 12.5 Advantages over Simple Statistical Approaches

A simple z-score on historical funding rates can also trigger mean reversion trades. The EPGP approach adds:

- **Physics-based equilibrium:** The mean reversion target is derived from the Fokker-Planck structure, not just historical average.
- **Adaptive confidence:** During normal periods, the confidence band is tight (aggressive trading). During regime changes, it widens (conservative).
- **Cross-exchange information fusion:** EPGP naturally handles multi-source sparse data.
- **Structural break detection:** When the OU assumption breaks (funding rate behavior changes), posterior variance increases before statistical indicators react.

---

## 13. Hybrid Extensions: Beyond Linear Constant Coefficients

### 13.1 Extension 1: Iterative Linearization for Nonlinear PDEs

**Target PDEs:** Nonlinear Black-Scholes (transaction costs), Hamilton-Jacobi-Bellman (optimal execution), nonlinear diffusion.

**Method:** Newton-Kantorovich iteration.

Given nonlinear PDE N[u] = 0:

1. Start with initial guess u₀ (e.g., linear BS solution via EPGP).
2. Linearize around u₀: L[δu] = -N[u₀], where L is the Fréchet derivative of N at u₀.
3. L is a linear constant-coefficient operator (if N has polynomial nonlinearity and u₀ is smooth). Solve with EPGP to get δu with uncertainty.
4. Update: u₁ = u₀ + δu.
5. Repeat until convergence.

**Key property:** Each iteration uses EPGP, so each intermediate solution is exact for the linearized problem, with calibrated uncertainty. The final solution has uncertainty from both linearization error and data sparsity.

**Convergence:** Under standard conditions (Lipschitz continuity of the Fréchet derivative), quadratic convergence — typically 3-5 iterations suffice.

### 13.2 Extension 2: Automated Variable-Coefficient Transformation

**Target PDEs:** Black-Scholes (before manual transformation), variable-rate models.

**Method:** Symmetry-based transformation search.

Many variable-coefficient PDEs admit **Lie point symmetries** that transform them into constant-coefficient form. The BS → heat transformation is one such symmetry.

Algorithm:
1. Given a variable-coefficient PDE, compute its Lie symmetry algebra (algorithmic, implemented in Maple/Mathematica).
2. Search for a symmetry that maps the PDE to constant-coefficient form.
3. If found: apply the transformation, then use EPGP.
4. If not found: fall back to iterative linearization or PINN.

This extends EPGP's applicability beyond manually discovered transformations to a systematic search.

### 13.3 Extension 3: Domain Decomposition

**Target:** Complex domains (non-convex regions in the (K, T) space), discontinuities.

**Method:** Schwarz-type domain decomposition.

1. Partition the domain Ω into overlapping convex subdomains Ω₁, ..., Ωₚ.
2. On each Ωᵢ, run EPGP independently.
3. On overlap regions, enforce continuity by adding observations from neighboring subdomains.
4. Iterate: update each subdomain's GP conditioned on boundary data from neighbors.

**Application in finance:** The option price surface has a kink at the strike (for short-dated options). Decompose into K < K₀ and K > K₀, run EPGP on each, and stitch at the boundary.

### 13.4 Extension 4: EPGP + PINN Hybrid

**The most general extension.** Decompose the solution:

u(x, t) = u_EPGP(x, t) + u_correction(x, t)

where:
- u_EPGP is the EPGP posterior mean for the linear constant-coefficient part
- u_correction is learned by a PINN to capture nonlinear/variable-coefficient effects

The PINN's loss function:

L = λ₁ · ||u_EPGP + u_NN - u_data||² + λ₂ · ||N[u_EPGP + u_NN]||²

where N[·] is the full (possibly nonlinear) PDE operator evaluated at collocation points.

**Advantages:**
- PINN starts from a good initial solution (EPGP), not from scratch → faster convergence
- The correction u_NN is small → requires fewer parameters
- EPGP provides uncertainty on the base solution; PINN provides point correction → hybrid uncertainty

### 13.5 Extension 5: EPGP + Latent Force Model Hybrid

**Alternative to PINN for structured nonlinearity.** The LFM (Álvarez et al. 2009) models:

L[u] = g(x, t)

where L is a linear operator (handled by EPGP) and g is an unknown forcing function modeled as a GP.

In finance: L = heat operator, g = the "market force" that causes deviations from BS (vol smile, jump effects, liquidity).

The joint model:
- u_EPGP ~ GP with EPGP kernel (satisfies L[u] = 0)
- g ~ GP with standard kernel (smooth, no PDE constraint)
- u_total = u_EPGP + L⁻¹[g] (EPGP base + LFM correction)

This provides full Bayesian uncertainty over both the PDE solution and the forcing — more principled than PINN for uncertainty quantification.

---

## 14. Industry Landscape: How Institutions Solve This Problem Today

The fundamental question this system addresses — "when to trade and when to step aside" — is not new. Every institutional trading desk has some answer to it. Understanding the existing approaches reveals exactly where EPGP offers something different and where it does not.

### 14.1 Approach 1: Statistical Filters

The most widespread approach. Firms estimate volatility using GARCH-family models, then reduce positions when volatility is abnormally elevated. Simpler desks use VIX levels, moving averages, or realized volatility thresholds as direct filters. The approach is fast, intuitive, and battle-tested over decades.

The fundamental limitation is backward-looking dependence. Statistical filters extract patterns from historical data. When an unprecedented event occurs — a global pandemic shutting down economies (March 2020), a coordinated central bank regime shift (2022), or a flash crash driven by algorithmic feedback loops (May 2010) — the filter has no historical template to match against. It reacts late, after the damage has begun.

### 14.2 Approach 2: Statistical Regime Models

Hidden Markov Models (HMMs) represent the more sophisticated version of statistical filtering. They partition the market into discrete states — typically Bull, Neutral, and Bear — and estimate posterior probabilities over these states in real time. Markov-switching GARCH extends this by allowing the volatility dynamics themselves to differ across regimes.

The advantage over simple filters is regime-conditional strategy selection: different strategies, position sizes, or risk budgets for different states. The disadvantage is the same: regime inference is still driven by historical data patterns. The model learns what past regimes "looked like" and tries to pattern-match the present. A new kind of regime — one that does not resemble any historical precedent — will be misclassified until sufficient data accumulates.

### 14.3 Approach 3: Model Calibration Error Monitoring

Quantitative derivatives desks — the closest institutional analog to this project — use a subtly different approach. Every day, they calibrate a pricing model (typically Heston, SABR, or a local volatility model) to the current options market. Calibration produces a residual: the gap between model prices and observed market prices.

When calibration error suddenly increases, it signals that the model is failing to explain the market. Traders interpret this as a warning — either the market is behaving unusually, or the model assumptions are breaking down. In practice, this triggers manual intervention: risk reduction, model review, or hedging adjustments.

The advantage is that this approach looks at the current market state directly, not historical patterns. The disadvantage is that the interpretation of calibration error is subjective and unstructured. When the Heston RMSE goes from 0.02 to 0.05, is that "somewhat worse" or "dangerously wrong"? There is no principled answer. Different traders draw different lines. The uncertainty is not quantified — it is just a residual number that a human interprets based on experience.

### 14.4 Approach 4: Pure Risk Management Rules

The simplest and most universal approach. Daily loss limits trigger automatic position liquidation. Portfolio VaR exceeding a threshold forces reduction. Drawdown hitting a predetermined level shuts down the strategy entirely.

These rules are robust precisely because they make no attempt to understand the market state. They respond to outcomes (losses), not causes. The advantage is simplicity and certainty — the rules fire when they fire, with no ambiguity. The disadvantage is that they are always reactive: the loss has already occurred before the rule activates. Avoiding a bad trade is strictly better than exiting one.

### 14.5 How EPGP Differs

Every method described above operates on **statistical signals** — patterns in historical data, price movements, or calibration residuals. EPGP operates on a **structural signal**: whether the known physical law (the PDE) still explains the observed data.

This is a fundamentally different category of information.

**An analogy.** Consider predicting whether a bridge will collapse. Statistical methods say: "Bridges of this type, built in this era, with this traffic load, have historically failed after N years — the probability is X%." EPGP says: "The force equations governing this bridge no longer match the observed stress measurements — the structure is changing." The first is empirical. The second is physical. They use different information and fail in different ways, which is why combining them is more robust than using either alone.

### 14.6 Three Specific Differentiators

**Differentiator 1: Source of information.** HMM looks at time series of VIX, returns, and term structure, matching historical patterns. EPGP looks at a single current snapshot of the options chain and asks: "Are these prices internally consistent with PDE structure?" The same market event is observed through entirely different lenses. When one sensor misses a signal, the other may catch it.

**Differentiator 2: Quantified uncertainty.** When Heston calibration error rises from 0.02 to 0.05, the threshold between "acceptable" and "dangerous" is a human judgment call. EPGP posterior variance is a probabilistically defined quantity. A 95% confidence interval widening means, precisely, that the true value lies within that interval with 95% probability. Trading rules built on this foundation have less arbitrariness — the confidence level is a number, not a feeling.

**Differentiator 3: Preemptive, not reactive.** Risk management rules activate after losses have occurred. EPGP signals uncertainty before the trade is entered. Not taking a bad trade is strictly superior to taking it and stopping out. The confidence filter prevents entry into trades where the model cannot support conviction, rather than cleaning up after conviction turns out to be misplaced.

### 14.7 Honest Limitations

EPGP's structural detection rests on the assumption that market dynamics have a PDE description at all. When markets move on purely political events — a surprise election result, an unexpected policy announcement — the movement may have nothing to do with any PDE structure. In such cases, EPGP's structural sensor is blind.

Furthermore, every existing method has decades of live trading validation. EPGP-based trading has zero live track record. The theoretical advantages are compelling but unproven. This is why the experimental agenda (Section 19) includes explicit kill criteria: if the advantages do not materialize in backtesting, the approach should be abandoned rather than rationalized.

---

## 15. The PDE Assumption: When It Holds and When It Breaks

### 19.1 What the PDE Assumes

The Black-Scholes PDE is derived from a specific set of assumptions: prices move continuously (no jumps), volatility is constant, transaction costs are zero, and no-arbitrage conditions hold. These assumptions produce the canonical equation:

∂V/∂t + ½σ²S²·∂²V/∂S² + rS·∂V/∂S - rV = 0

After coordinate transformation, this becomes the heat equation ∂u/∂τ = ∂²u/∂x², which is the domain where EPGP operates.

### 19.2 When It Holds (~90% of Trading Days)

On a typical day, the assumptions hold approximately. Prices move tick by tick without discontinuous jumps. Intraday volatility fluctuates but does not undergo regime changes within hours. SPX options are among the most liquid instruments in the world, so no-arbitrage conditions are enforced by active market makers. Transaction costs exist but are small relative to option premiums.

Under these conditions, the PDE provides a good structural description of how option prices relate to each other across strikes and maturities. The EPGP posterior variance will be small and stable. The confidence filter will say: "proceed normally."

This is why the entire options industry runs on Black-Scholes as a quoting and risk management framework. Not because it is exactly right, but because it is right enough, most of the time, to serve as a shared language.

### 19.3 When It Breaks (~10% of Trading Days)

The remaining days are where most of the risk — and opportunity — concentrates.

**Jump events.** Lehman Brothers bankruptcy (September 2008), the Flash Crash (May 2010), the SNB removing the EUR/CHF floor (January 2015), the COVID crash (March 2020). Prices gap discontinuously, violating the continuous-path assumption. The PDE, which models smooth diffusion, cannot describe a 10% overnight gap.

**Volatility regime shifts.** The 2022 rate hiking cycle did not produce a single crash, but volatility regime fundamentally changed — the VIX term structure inverted, realized-implied volatility dynamics shifted, and cross-asset correlations broke historical patterns. The constant-volatility assumption is violated not by a single event but by a sustained structural change.

**Liquidity withdrawal.** During stress events, market makers widen bid-ask spreads or step away entirely. The no-arbitrage assumption breaks down because arbitrage requires the ability to trade, and trading becomes impossible or prohibitively expensive. Option prices deviate from PDE-implied relationships because the mechanism that enforces those relationships (arbitrage trading) has stopped functioning.

### 17.4 The Vol Smile: A Permanent, Everyday Deviation

Even on normal days, the PDE is not perfectly right. Actual option prices exhibit a "volatility smile" — out-of-the-money puts are priced higher than Black-Scholes predicts. This is the market's recognition that the constant-volatility assumption underestimates tail risk. The smile has been persistent since at least the 1987 crash.

This means the EPGP posterior variance will never be zero, even on the calmest day. There is always some residual inconsistency between data and PDE structure. The relevant signal is not the level of variance but the change — when variance increases sharply relative to its recent baseline, something new is happening.

### 17.5 Why Partial Validity Is the Sweet Spot

A critical observation: **EPGP's value comes precisely from the PDE being approximately right most of the time and wrong some of the time.**

If the PDE were always exactly right, posterior variance would always be low, and the confidence filter would never trigger. There would be no bad trades to avoid.

If the PDE were always wrong, posterior variance would always be high, and the confidence filter would never allow entry. There would be no trades at all.

The real-world situation — mostly right, sometimes wrong — is exactly the regime where a confidence filter adds value. During the 90% of normal days, the filter confirms that conditions are suitable for trading. During the 10% of stressed days, the filter detects the structural breakdown and pulls back. Avoiding even a fraction of those 10% of days can meaningfully improve risk-adjusted returns, because losses in those periods are disproportionately large.

### 15.6 Historical Case Study: March 2020

In early March 2020, before the major crash, the options market was already showing signs of PDE structure breakdown. ATM implied volatility spiked from ~15% to ~40% within days, but the spike was structurally anomalous: the skew (difference between OTM put IV and ATM IV) moved in an unusual direction, the term structure inverted sharply, and the put-call parity relationship showed larger-than-normal deviations.

These are precisely the signals that EPGP posterior variance would capture: option prices that are individually valid but collectively inconsistent with the heat equation structure derived from constant-volatility Black-Scholes. The hypothesis — testable in backtesting — is that EPGP variance spiked during the first week of March, before the worst of the drawdown (March 12-23), providing an actionable early warning.

This hypothesis is falsifiable. Phase 3 of the experimental roadmap tests it directly.

---

## 16. Comparison Framework: EPGP vs Alternative Methods

### 18.1 Methods to Compare

| # | Method | Type | PDE Exact? | UQ? |
|---|--------|------|-----------|-----|
| 1 | EPGP | Algebraic GP | Yes | Yes |
| 2 | S-EPGP | Sparse algebraic GP | Yes | Yes |
| 3 | B-EPGP | Boundary algebraic GP | Yes (incl. BC) | Yes |
| 4 | GP + collocation | Generic GP | Approximate | Yes |
| 5 | PINN | Neural network | Approximate | No (or expensive) |
| 6 | Latent Force Model | ODE-constrained GP | Partial | Yes |
| 7 | Deep Kernel Learning + PDE | Hybrid NN+GP | Approximate | Yes |
| 8 | Finite Difference Method | Classical numerical | Approximate | No |
| 9 | Fourier / FFT (Carr-Madan) | Spectral | Exact (for known models) | No |
| 10 | Monte Carlo | Simulation | Approximate | Statistical |

### 18.2 Evaluation Axes

Axis 1: Accuracy. L² error between predicted and true solution (where true solution is known, e.g., BS closed-form).

Axis 2: Speed. Wall-clock time for training/fitting and prediction.

Axis 3: Data efficiency. How does accuracy degrade as the number of observations decreases from 200 to 50 to 10 to 5?

Axis 4: Extrapolation. Accuracy outside the training data region (e.g., predicting at strikes further OTM than any observed).

Axis 5: Uncertainty calibration. For methods with UQ: do 95% intervals actually contain the true value 95% of the time? Measured by coverage probability and interval width.

Axis 6: Robustness. Accuracy when data contains outliers or when the true model deviates slightly from the assumed PDE (model misspecification).

### 18.3 Test Problems

Test 1: 1D heat equation with Gaussian initial condition. Closed-form true solution available. Tests basic solver capability.

Test 2: BS put option pricing with constant vol. True solution is BS formula. Tests financial PDE pipeline.

Test 3: BS with sparse data. Only 10 option prices observed, at irregular (K, T) points. Tests data efficiency and UQ.

Test 4: BS with vol smile. True prices generated by Heston model. Constant-vol assumption is violated. Tests robustness to model misspecification and ability to detect structural deviation.

Test 5: 2D basket option. No closed-form. Tests multi-dimensional capability. MC as benchmark.

Test 6: Time series of option surfaces. A sequence of 52 weekly option surfaces. Tests real-world applicability, including regime changes and the variance monitoring system.

---

## 17. Backtesting Architecture

### 19.1 Design Principles

**Principle: Point-in-time everything.** At each backtest date t, the system has access only to information available at t. EPGP is trained only on data from date t. No future data leaks.

**Principle: Transaction costs are real.** Every simulated trade includes commission ($0.65/contract for IBKR options), slippage (25% of bid-ask spread), and funding costs.

**Principle: Survivorship bias prevention.** Use actual historical option chains, not reconstructed ones.

### 19.2 Backtest Loop

For each trading date t in [t_start, t_end]:

1. Ingest market data available at time t.
2. Run EPGP pipeline: transform → fit → posterior → inverse transform.
3. Compute confidence scores C(K, T, t) and regime signal Z_Φ(t).
4. If HMM is used: run HMM on data up to t (not including t+1).
5. Generate trade candidates based on base strategy rules.
6. Filter candidates by EPGP confidence.
7. Size positions.
8. Record entries and mark-to-market existing positions.
9. Check exit conditions (confidence degradation, variance spike, P&L stops).
10. Record daily P&L, positions, confidence metrics.

### 19.3 Benchmark Strategies

A: Naive (no filter) — Enter every candidate trade at fixed size.
B: Regime-only (HMM filter) — Enter only when HMM regime is favorable.
C: EPGP-only (confidence filter) — Enter only when EPGP confidence is high.
D: Combined (HMM + EPGP) — Enter only when both signals are favorable.

### 17.4 Performance Metrics

| Metric | Definition | Target |
|--------|-----------|--------|
| Annualized Sharpe | μ / σ (annualized) | > 0.5 |
| Maximum Drawdown | Peak-to-trough | < 25% |
| Win Rate | Fraction of profitable trades | > 60% |
| Profit Factor | Gross profit / Gross loss | > 1.5 |
| Calmar Ratio | Annualized return / Max DD | > 0.5 |
| Confidence-Accuracy Correlation | Corr(C, trade_profit) | > 0 (positive) |
| UQ Calibration | 95% interval coverage | ≈ 95% |
| Variance Spike Detection Rate | % of drawdowns preceded by Z_Φ > 2 | > 50% |

### 17.5 Overfitting Defense

- Walk-forward validation: Train on [t, t+W], test on [t+W, t+W+w], slide forward.
- Parameter sensitivity: Vary C_min, Z_max by ±20% and verify Sharpe change < ±30%.
- Multi-market validation: Same filter on options AND crypto. If it works on both, less likely overfit.

---

## 18. Parameter Reference

### 18.1 EPGP Engine Parameters

| Parameter | Symbol | Default | Range | Tunable? |
|-----------|--------|---------|-------|----------|
| Kernel scale (per variety component) | σᵢ² | 1.0 | [0.01, 100] | Yes (MLE) |
| Observation noise | σ₀² | 0.01 | [10⁻⁶, 1.0] | Yes (MLE) |
| S-EPGP frequency count | r | 50 | [10, 500] | Manual |
| B-EPGP boundary type | — | Dirichlet | {Dirichlet, Neumann, Robin} | Manual |

### 18.2 Confidence Filter Parameters

| Parameter | Symbol | Default | Range | Tunable? |
|-----------|--------|---------|-------|----------|
| Minimum confidence for entry | C_min | 0.6 | [0.3, 0.9] | Walk-forward |
| Target confidence for full size | C_target | 0.8 | [0.5, 0.95] | Walk-forward |
| Exit confidence threshold | C_exit | 0.4 | [0.2, C_min] | Walk-forward |
| Variance z-score threshold | Z_max | 2.0 | [1.5, 3.0] | Walk-forward |
| Emergency exit z-score | Z_emergency | 3.0 | [2.5, 4.0] | Fixed |
| Variance lookback window (days) | W_Φ | 63 | [21, 252] | Manual |

### 18.3 Regime Integration Parameters

| Parameter | Symbol | Default | Range | Tunable? |
|-----------|--------|---------|-------|----------|
| HMM weight | w_HMM | 0.5 | [0, 1] | Walk-forward |
| EPGP weight | w_EPGP | 0.5 | [0, 1] | Walk-forward |
| Combined threshold | — | 0.5 | [0.3, 0.7] | Walk-forward |

### 18.4 Market-Specific Parameters

**Options (VRP):**

| Parameter | Value | Notes |
|-----------|-------|-------|
| Target delta | -0.10 | 10-delta put |
| DTE range | 30-45 days | Weekly selection |
| Spread width | 50 points | Fixed dollar width |
| Max open positions | 4 | Risk limit |

**Crypto Funding Rate:**

| Parameter | Value | Notes |
|-----------|-------|-------|
| OU mean reversion speed κ | Calibrated | From historical funding |
| OU long-run mean θ | Calibrated | Rolling estimate |
| Entry threshold | 2σ_F from μ_F | Confidence band |
| Exchanges | Binance, Bybit, OKX | Cross-exchange data |
| Funding frequency | 8 hours | Standard perpetual |

---

## 19. Open Questions and Experimental Agenda

### 19.1 Fundamental Questions

Q1: Does EPGP posterior variance actually predict future P&L variance? This is the single most important empirical question. If variance is uncorrelated with outcomes, the entire confidence filter is worthless.

Q2: Does the EPGP variance spike precede regime changes detected by HMM? If EPGP detects structural breaks faster than HMM, the dual system has value. If not, EPGP adds complexity without benefit.

Q3: Is the heat equation approximation good enough for options? The BS → heat transformation assumes constant vol. Real markets have stochastic vol. How much does this approximation error contaminate the EPGP posterior?

Q4: Does the confidence filter generalize across markets? If it works for options but not for crypto (or vice versa), why? What market characteristics determine effectiveness?

### 19.2 Experimental Roadmap

Phase 1 — Proof of concept (1D heat, synthetic data):
- Implement EPGP for 1D heat equation.
- Verify posterior coverage on synthetic data.
- Benchmark against PINN and FDM.
- Establish that the math works before touching market data.

Phase 2 — Financial PDE validation (BS, historical options data):
- Implement BS → heat pipeline with real option chain data.
- Compute EPGP posterior on historical weekly snapshots.
- Analyze: does posterior variance correlate with subsequent option P&L variance?
- Run Strategy A vs Strategy C backtest on 2020-2024 data.

Phase 3 — Regime detection validation:
- Implement EPGP variance monitoring alongside HMM.
- Compare: which detects the March 2020 crash earlier? The 2022 rate shock?
- Run Strategy A vs B vs C vs D comparison.

Phase 4 — Crypto extension:
- Implement Fokker-Planck EPGP for funding rate.
- Backtest mean reversion strategy with and without confidence filter.
- Cross-validate: do the same confidence thresholds work in both markets?

Phase 5 — Hybrid extensions:
- Implement EPGP + PINN hybrid for vol smile correction.
- Implement EPGP + LFM hybrid.
- Compare hybrid approaches on Heston-generated data.

Phase 6 — Live paper trading:
- Deploy on IBKR (options) and/or exchange API (crypto) in paper mode.
- 3-month observation period before real capital.
- Track live vs backtest performance ratio.

### 19.3 Failure Modes and Kill Criteria

If Phase 1 shows EPGP posterior is poorly calibrated → debug implementation before proceeding.

If Phase 2 shows zero correlation between confidence and P&L → the approach does not work for options. Document the negative result and focus on crypto.

If Phase 3 shows EPGP variance monitoring is always slower than HMM → the structural detection has no incremental value. Simplify to HMM only.

If Phase 4 shows confidence filter does not improve crypto strategy → the approach may be market-specific or fundamentally flawed.

If Phases 2-4 all show negative results → conclude that EPGP confidence is not a useful trading signal. Write up the analysis as a negative-result study (still valuable for the community).

---

## Appendix A: Software Dependencies

- Macaulay2 (optional, Phase 5+): General algebraic preprocessing for arbitrary PDE systems. Not needed for heat, wave, or Fokker-Planck equations — these have hardcoded varieties (Section 2.6).
- Python (NumPy, SciPy): GP inference, posterior computation
- PyTorch: S-EPGP frequency learning, PINN hybrid
- GPyTorch (optional): Scalable GP inference
- IBKR TWS API: Options execution
- CCXT: Crypto exchange API (funding rates)

## Appendix B: Key References

Core EPGP:
- Härkönen, Lange-Hegermann, Raiţă. EPGP. ICML 2023. arXiv:2212.14319.
- Besginow, Lange-Hegermann. LODE-GPs. NeurIPS 2022. arXiv:2208.12515.
- Huang, Härkönen, Lange-Hegermann, Raiţă. B-EPGP. arXiv:2411.16663.
- Amo, Ghosh, Lange-Hegermann, Pokojovy. B-EPGP vs FEM. arXiv:2511.04518.
- Li, Lange-Hegermann, Raiţă. EPGP Inverse Problems. arXiv:2502.04276.
- Lange-Hegermann. Algorithmic LCGPs. NeurIPS 2018. arXiv:1801.09197.

Mathematical Foundations:
- Ehrenpreis. Fourier Analysis in Several Complex Variables. Wiley, 1970.
- Palamodov. Linear Differential Operators with Constant Coefficients. Springer, 1970.
- Hörmander. The Analysis of Linear PDE II. Springer, 1983.
- Cid-Ruiz, Homs, Sturmfels. Primary Ideals and Their Differential Equations. FoCM, 2021.
- Chen, Härkönen, Krone, Leykin. Noetherian Operators and Primary Decomposition. JSC, 2022.
- Ait El Manssour, Härkönen, Sturmfels. Linear PDE with Constant Coefficients. GMJ, 2023.

Competing Methods:
- Raissi, Perdikaris, Karniadakis. PINNs. JCP, 2019.
- Álvarez, Luengo, Lawrence. Latent Force Models. AISTATS, 2009.
- Chen, Hosseini, Owhadi, Stuart. Solving Nonlinear PDEs with GPs. JCP, 2021.
- Batlle, Darcy, Hosseini, Owhadi. Kernel Methods for Operator Learning. JCP, 2024.
- Wilson, Adams. Spectral Mixture Kernels. ICML, 2013.

Financial PDE:
- Black, Scholes. The Pricing of Options. JPE, 1973.
- Heston. A Closed-Form Solution for Options with Stochastic Volatility. RFS, 1993.
- Carr, Madan. Option Valuation Using the FFT. JCOF, 1999.
