# HMM Regime Detection Module: Mathematical Design Document

**Statistical Regime Sensor for the EPGP Confidence-Filtered Trading System**

> This document specifies the Hidden Markov Model (HMM) regime detection subsystem in full detail. The HMM serves as the statistical complement to the EPGP structural sensor. Together, they form a dual detection system: HMM detects regime changes from historical statistical patterns; EPGP detects regime changes from PDE structure violations in current market data. This document covers feature engineering, model specification, training, inference, label stability, market-specific configurations, integration with the EPGP confidence filter, and implementation.

---

## Table of Contents

1. [Role Within the EPGP Trading System](#1-role-within-the-epgp-trading-system)
2. [Hidden Markov Model: Mathematical Specification](#2-hidden-markov-model-mathematical-specification)
3. [Regime Definition: What Are We Detecting?](#3-regime-definition)
4. [Feature Engineering](#4-feature-engineering)
5. [Model Training](#5-model-training)
6. [Real-Time Inference](#6-real-time-inference)
7. [Label Stability and Post-Processing](#7-label-stability-and-post-processing)
8. [XGBoost Classifier Layer](#8-xgboost-classifier-layer)
9. [Market Application I: Options (SPX/VIX)](#9-market-application-options)
10. [Market Application II: Crypto Perpetual Funding Rate](#10-market-application-crypto)
11. [Dual Detection Integration with EPGP](#11-dual-detection-integration)
12. [Validation Framework](#12-validation-framework)
13. [Implementation Specification](#13-implementation-specification)
14. [Parameter Reference](#14-parameter-reference)
15. [Failure Modes and Diagnostics](#15-failure-modes-and-diagnostics)

---

## 1. Role Within the EPGP Trading System

### 1.1 The Dual Sensor Architecture

The EPGP trading system uses two independent regime detection mechanisms:

**Sensor A: HMM (this document).** Observes time series of market features (returns, volatility, term structure). Infers a discrete hidden state (regime) using statistical pattern matching against historical data. Output: posterior probability distribution over regimes at each time step.

**Sensor B: EPGP posterior variance (see main document, Section 9).** Observes a single snapshot of the options chain (or funding rate data). Measures whether the data conforms to PDE structure. Output: a continuous confidence score.

These sensors use fundamentally different information:

| Property | HMM | EPGP Variance |
|----------|-----|---------------|
| Input data | Time series (lookback window) | Single cross-sectional snapshot |
| Information type | Statistical patterns | Physical structure |
| Detection principle | "This looks like a past regime" | "The PDE no longer fits the data" |
| Strength | Captures complex multi-feature patterns | Detects novel structural breaks |
| Weakness | Blind to unprecedented regimes | Blind to non-PDE market dynamics |

### 1.2 Why Two Sensors?

A single sensor has blind spots. HMM fails when a new type of crisis occurs that does not resemble historical patterns. EPGP fails when the market disruption is not reflected in PDE structure (e.g., a pure sentiment shift that does not change options pricing dynamics).

Two independent sensors with different failure modes create a more robust system. When both agree — high confidence. When they disagree — the system becomes conservative.

### 1.3 Information Flow

```
Market Data (t)
    ├──→ Feature Extraction → HMM → P(regime = k | data up to t)
    │                                        │
    │                                        ▼
    │                              Dual Integration
    │                                        ▲
    │                                        │
    └──→ Options/Funding Data → EPGP → Φ(t), Z_Φ(t)

Dual Integration Output:
    → Regime_combined(t)
    → Position sizing scaler
    → Entry/exit gate
```

---

## 2. Hidden Markov Model: Mathematical Specification

### 2.1 Model Definition

A Hidden Markov Model consists of:

**Hidden states:** S = {s₁, s₂, ..., s_K} where K is the number of regimes. The state at time t is denoted qₜ ∈ S.

**Transition matrix:** A = [aᵢⱼ] where aᵢⱼ = P(qₜ = sⱼ | qₜ₋₁ = sᵢ). This is a K×K matrix with rows summing to 1. It encodes the probability of moving from regime i to regime j in one time step.

**Emission distribution:** P(oₜ | qₜ = sₖ) defines the probability of observing feature vector oₜ given the system is in state sₖ. We use multivariate Gaussian emissions:

P(oₜ | qₜ = sₖ) = N(oₜ; μₖ, Σₖ)

where μₖ ∈ R^d is the mean feature vector in state k and Σₖ ∈ R^{d×d} is the covariance matrix.

**Initial distribution:** π = [πₖ] where πₖ = P(q₁ = sₖ).

The full parameter set is θ = {A, {μₖ, Σₖ}_{k=1}^K, π}.

### 2.2 Three Core Algorithms

**Forward algorithm** — computes P(oₜ | o₁, ..., oₜ₋₁, θ), the filtered state probabilities given all observations up to time t. This is what we use for real-time inference. Complexity: O(K²T) for T time steps.

αₖ(t) = P(o₁, ..., oₜ, qₜ = sₖ | θ)
αₖ(t) = [Σⱼ αⱼ(t-1) · aⱼₖ] · P(oₜ | qₜ = sₖ)

The filtered posterior at time t:
P(qₜ = sₖ | o₁, ..., oₜ) = αₖ(t) / Σⱼ αⱼ(t)

**Forward-backward algorithm** — computes P(qₜ = sₖ | o₁, ..., oₜ) using the entire observation sequence (smoothed probabilities). Used for training, not for real-time inference (because it uses future data).

**Viterbi algorithm** — computes the most likely state sequence argmax P(q₁, ..., qₜ | o₁, ..., oₜ). Used for labeling historical data, not for real-time trading.

### 2.3 Training via Expectation-Maximization

The Baum-Welch algorithm (EM for HMMs) iterates:

**E-step:** Given current θ, compute expected sufficient statistics using the forward-backward algorithm:
- γₖ(t) = P(qₜ = sₖ | O, θ) — posterior probability of state k at time t
- ξᵢⱼ(t) = P(qₜ = sᵢ, qₜ₊₁ = sⱼ | O, θ) — posterior probability of transition i→j at time t

**M-step:** Update parameters:
- πₖ = γₖ(1)
- aᵢⱼ = Σₜ ξᵢⱼ(t) / Σₜ γᵢ(t)
- μₖ = Σₜ γₖ(t)·oₜ / Σₜ γₖ(t)
- Σₖ = Σₜ γₖ(t)·(oₜ - μₖ)(oₜ - μₖ)ᵀ / Σₜ γₖ(t)

Iterate until convergence (log-likelihood change < ε = 10⁻⁴).

### 2.4 Critical Constraint: No Look-Ahead

**For trading, we must use only the forward algorithm, never forward-backward or Viterbi on the current window.**

The forward algorithm computes P(qₜ | o₁, ..., oₜ) — it uses only past and current observations. The forward-backward algorithm uses the full sequence including future observations. Using forward-backward or full-sequence Viterbi in a backtest introduces look-ahead bias — the most dangerous error in quantitative finance.

Concretely: in `hmmlearn`, use `_do_forward_pass()` directly. Do **not** use `predict_proba()` or `predict()` — these call the forward-backward or Viterbi algorithm internally and will contaminate backtest results.

---

## 3. Regime Definition: What Are We Detecting?

### 3.1 Number of States: K = 3

We use three regimes:

**State 1: Low Volatility (LV).** Calm markets. Realized volatility is below historical average. VIX is low. Term structure is in contango (VIX < VIX3M). Option premiums are compressed. Transitions are slow — the market tends to stay calm for extended periods.

**State 2: Normal Volatility (NV).** Typical market conditions. Realized volatility is near its historical average. VIX is moderate. Mixed term structure signals. This is the "default" state that the market occupies most of the time.

**State 3: High Volatility (HV).** Stressed markets. Realized volatility is elevated. VIX is high. Term structure may be in backwardation (VIX > VIX3M). Wide bid-ask spreads. Rapid, sharp price movements. This state is relatively rare but accounts for the majority of drawdown risk.

### 3.2 Why Not 2 States? Why Not 4+?

**Two states** (calm vs stressed) are too coarse. The system cannot distinguish between "unusually calm" (where option premiums may be too cheap to sell) and "normal" (the sweet spot for premium selling). This distinction matters for strategy sizing.

**Four or more states** risk overfitting. With limited data (5-10 years of daily observations = 1,250-2,500 points), estimating 4+ state transition matrices and emission distributions leads to unstable parameter estimates. Information criteria (BIC) consistently favor K = 3 for VIX-related feature sets.

### 3.3 Regime Interpretation for Trading

| Regime | Volatility Level | VIX Range (approx.) | Trading Implication |
|--------|-----------------|---------------------|---------------------|
| LV | Low | VIX < 14 | Option premiums thin. EPGP confidence likely high (stable PDE). Reduce size — risk/reward less favorable. |
| NV | Normal | 14 ≤ VIX < 25 | Sweet spot. Adequate premiums. PDE structure stable. Trade at full confidence-adjusted size. |
| HV | High | VIX ≥ 25 | Premiums rich but tail risk elevated. PDE structure may be breaking. Reduce size aggressively. Dual sensor agreement required for any entry. |

### 3.4 Label Assignment

After training, the HMM's three states are unlabeled (they are just "State 0, 1, 2"). We assign semantic labels (LV, NV, HV) by sorting states by their mean realized volatility:

μ_RV(state 0) < μ_RV(state 1) < μ_RV(state 2) → LV, NV, HV

This labeling is done once after training and must be stable across retraining windows (see Section 7).

---

## 4. Feature Engineering

### 4.1 Design Principles

Features must satisfy three criteria:

1. **Observable at time t using only data up to t.** No forward-looking calculations.
2. **Informative about volatility regime.** Each feature should have statistically different distributions across LV, NV, and HV regimes.
3. **Low multicollinearity.** Highly correlated features waste model capacity and destabilize covariance estimation.

### 4.2 Primary Feature Set (Options Market)

**Feature 1: Realized Volatility (RV₂₁)**

σ_RV(t) = √(252 / 21 · Σᵢ₌₁²¹ r²_{t-i})

where rₜ = ln(Sₜ / Sₜ₋₁) is the daily log return. Window: 21 trading days (1 month). Annualized by √252. This is the most direct measure of current volatility.

**Feature 2: VIX Level**

The CBOE VIX index, observed daily. Represents the market's expectation of 30-day forward realized volatility as implied by SPX options. VIX captures the forward-looking component that RV misses.

**Feature 3: VIX Term Structure Slope (TS)**

TS(t) = (VIX3M(t) - VIX(t)) / VIX(t)

VIX3M is the 3-month VIX. Positive TS (contango) indicates normal conditions — longer-dated vol expectations exceed short-dated. Negative TS (backwardation) indicates stress — near-term fear exceeds medium-term, often preceding or accompanying market dislocations.

**Feature 4: VVIX (Volatility of Volatility)**

The CBOE VVIX index measures implied volatility of VIX options. High VVIX indicates uncertainty about future volatility itself — a second-order risk signal. When VVIX spikes, it means options market participants disagree about the future volatility path.

**Feature 5: Volatility Change (ΔVol)**

ΔVol(t) = σ_RV(t) - σ_RV(t - 21)

The 21-day change in realized volatility. Captures the direction and speed of volatility evolution. Positive ΔVol = volatility is increasing (potential regime transition). Negative = decreasing (potential regime stabilization).

### 4.3 Feature Set for Crypto Market

For the crypto funding rate application, the features are adapted:

**Feature 1: Realized Volatility (BTC or ETH)**

Same formula as above but computed on crypto daily returns. Note: crypto trades 24/7, so the annualization factor is √365, not √252.

**Feature 2: Funding Rate Level**

F(t) = current perpetual funding rate (8-hour). Directly analogous to VIX for options — it reflects the market's directional bias and willingness to pay for leverage.

**Feature 3: Funding Rate Term Structure**

If available across different perpetual contract types (e.g., quarterly vs perpetual), the spread serves the same role as VIX term structure.

**Feature 4: Cross-Exchange Funding Dispersion**

Disp(t) = std({F_Binance(t), F_Bybit(t), F_OKX(t)})

Standard deviation of funding rates across exchanges. High dispersion indicates fragmented market conditions — analogous to VVIX for options.

**Feature 5: Volume Change**

ΔVol_trading(t) = ln(V_21(t) / V_21(t-21))

Log ratio of 21-day average trading volume. Volume spikes often precede or accompany regime changes in crypto.

### 4.4 Feature Preprocessing

All features are standardized before being fed to the HMM:

zᵢ(t) = (fᵢ(t) - μᵢ) / σᵢ

where μᵢ and σᵢ are the expanding mean and standard deviation of feature i computed on data up to time t (not the full sample). This ensures no look-ahead bias in standardization.

For the initial training window (first W observations), use the in-window mean and standard deviation.

**Note on stationarity:** Expanding-window standardization prevents look-ahead bias but introduces non-stationarity: z-scores computed on 50 days of history have higher variance than those computed on 500 days. The HMM emission distributions are estimated from standardized features and will absorb this non-stationarity during training. However, for the live inference window, switch to a rolling 252-day window for standardization once at least 252 days of history are available (switch from expanding to rolling at t = 252). This provides stable z-score scales for real-time operation.

### 4.5 Handling Missing Data

**VIX3M:** Available from CBOE starting ~2008. For earlier periods or gaps, approximate as VIX3M ≈ VIX × 1.05 (empirical long-run ratio). Flag approximated periods.

**VVIX:** Available from CBOE starting ~2012. For earlier periods: exclude the feature and retrain with d = 4 features instead of d = 5. The HMM should be validated with and without VVIX to ensure robustness.

**Crypto features:** Funding rates are available from exchange inception (Binance: 2019, Bybit: 2020, OKX: 2020). Missing data from a single exchange is handled by using the remaining exchanges. If all exchanges are missing (exchange downtime), carry forward the last available value for up to 24 hours; beyond that, mark the day as missing and skip regime inference.

---

## 5. Model Training

### 5.1 Training Data and Window

**Minimum training window:** W_min = 504 trading days (approximately 2 years). This ensures sufficient data to observe multiple regime transitions. With K = 3 states and d = 5 features, the model has 3 + 9 + 15 + 15 + 15 = 57 free parameters (3 initial probs, 9 transition probs minus constraints, 15 means, 15 variances, 15 covariances per state × 3 states = much more). Two years of daily data provides ~500 observations, giving roughly 9 observations per parameter — tight but workable.

**Expanding window:** After the initial W_min training period, the HMM is retrained monthly using all available data up to the current date. This allows the model to learn from new regime observations as they occur.

**No sliding window.** An expanding window is preferred over a fixed-length sliding window because:
1. Early regime transitions (e.g., 2008 crisis) remain in the training set permanently. A sliding window would eventually drop them, losing valuable information about extreme states.
2. Transition probabilities become more stable as more data accumulates.

### 5.2 Initialization

The Baum-Welch algorithm is sensitive to initialization. We use a multi-start approach:

1. Run K-means clustering on the feature vectors with K = 3 clusters.
2. Use K-means centroids as initial emission means μₖ⁰.
3. Use within-cluster covariance as initial Σₖ⁰.
4. Initialize transition matrix with diagonal dominance: aᵢᵢ = 0.95, aᵢⱼ = 0.05/(K-1) for i ≠ j. This encodes the prior that regimes are persistent (markets do not switch regime every day).
5. Run Baum-Welch with n_init = 10 random perturbations of this K-means initialization. For each of the 10 runs: perturb K-means means by adding Gaussian noise with std = 0.1 · std(feature_i) per feature dimension. Transition matrix is re-initialized with the same diagonal-dominant structure each time. Random seed for K-means is fixed at 42; Baum-Welch perturbation seeds are 0 through 9.
6. Select the run with highest log-likelihood.

### 5.3 Covariance Type

Use **full** covariance matrices Σₖ (not diagonal or tied). Rationale:

- Diagonal covariance assumes features are independent within each regime. This is false — VIX and RV are positively correlated, and this correlation itself changes across regimes.
- Tied covariance (same Σ for all states) misses the key signal that the HV regime has higher variance and different correlation structure than the LV regime.
- Full covariance with d = 5 features means 15 free parameters per state (5 variances + 10 covariances). With 500+ training observations, this is estimable.

### 5.4 Convergence Criteria

- Maximum iterations: 200
- Log-likelihood convergence threshold: |L(θⁿ⁺¹) - L(θⁿ)| < 10⁻⁴
- If not converged after 200 iterations, flag a warning but use the current parameters.

### 5.5 Model Selection

After training, verify the model is sensible:

1. **State occupancy:** Each state should have occupancy > 10%. If one state has < 5% occupancy, it may be a spurious split. Consider reducing to K = 2.
2. **Transition matrix structure:** The diagonal should dominate (aᵢᵢ > 0.90 for all i). If any off-diagonal element exceeds 0.3, the states may not be well-separated.
3. **Emission separation:** The Bhattacharyya distance between each pair of emission distributions should exceed a threshold (e.g., > 0.5). If two states have nearly identical emissions, they are redundant.
4. **BIC comparison:** Train models with K = 2, 3, 4. Select the K that minimizes BIC = -2·L + p·log(T), where p is the number of parameters and T is the sequence length. This balances fit and complexity.

---

## 6. Real-Time Inference

### 6.1 Forward Algorithm Implementation

At each trading day t, given the observation vector oₜ and the forward variables from t-1:

1. Compute prediction: α̂ₖ(t) = Σⱼ αⱼ(t-1) · aⱼₖ for each state k.
2. Compute update: αₖ(t) = α̂ₖ(t) · N(oₜ; μₖ, Σₖ).
3. Normalize: P(qₜ = sₖ | o₁:ₜ) = αₖ(t) / Σⱼ αⱼ(t).

This is O(K²) per time step — negligible computation.

### 6.2 Numerical Stability: Log-Space Forward

The forward variables αₖ(t) involve products of many small probabilities, causing underflow. Compute in log-space:

log αₖ(t) = log Σⱼ exp(log αⱼ(t-1) + log aⱼₖ) + log N(oₜ; μₖ, Σₖ)

The first term uses the log-sum-exp trick:

log Σⱼ exp(xⱼ) = max(x) + log Σⱼ exp(xⱼ - max(x))

This is numerically stable for sequences of any length.

### 6.3 Output Format

At each time step t, the HMM produces a probability vector:

p(t) = [P(LV | o₁:ₜ), P(NV | o₁:ₜ), P(HV | o₁:ₜ)]

This vector sums to 1. The confidence in the current regime assignment is captured by:

- **Max probability:** max(p(t)). High max-prob means the model is confident about the current regime.
- **Entropy:** H(t) = -Σₖ pₖ(t) log pₖ(t). Low entropy = high confidence, high entropy = ambiguity between regimes.

### 6.4 Regime Assignment

The current regime is assigned as:

regime(t) = argmax_k P(qₜ = sₖ | o₁:ₜ)

For trading decisions, we use the full probability vector, not just the argmax. This allows graded responses: if P(HV) = 0.4, we partially reduce exposure rather than waiting for P(HV) to cross 0.5.

---

## 7. Label Stability and Post-Processing

### 7.1 The Label Permutation Problem

HMMs have a fundamental identifiability issue: the states are unordered. After retraining, "State 0" in the new model may correspond to "State 2" in the old model. If labels flip, the trading strategy receives inverted signals.

### 7.2 Label Alignment Protocol

After each retraining:

1. Compute the mean realized volatility μ_RV for each state using the training data.
2. Sort states by μ_RV: lowest → LV, middle → NV, highest → HV.
3. Verify alignment by computing the overlap between new and old state assignments on the last 63 trading days (3 months). If overlap > 80% for all three states, alignment is consistent.
4. If overlap < 80% for any state, flag a warning. Inspect manually before using the new model.

### 7.3 Stability Metric

Label stability uses two complementary checks:

**Check 1 — Hard-label overlap (robustness screen):**

Stability_label(t) = (1/63) · Σᵢ₌₁⁶³ 𝟙[regime_new(t-i) = regime_old(t-i)]

where regime_new and regime_old are the hard regime assignments (argmax of filtered posterior) from the newly trained and previously trained HMMs respectively.

Threshold: Stability_label ≥ 90%.

**Check 2 — Probabilistic overlap (sensitivity check):**

For each of the last 63 days, compute the symmetric KL divergence between the new and old filtered posteriors:

KL_sym(t) = ½ KL(P_new || P_old) + ½ KL(P_old || P_new)

Mean_KL = (1/63) · Σᵢ₌₁⁶³ KL_sym(t-i)

Threshold: Mean_KL < 0.3 nats (approximately equivalent to 90% label overlap for well-separated posteriors, but more sensitive to near-boundary flips).

**Decision rule:** Both checks must pass for a clean model switch. If either fails, trigger the warm-start retry. If Check 1 passes but Check 2 fails, the model is mathematically different even if labels appear the same — treat as unstable.

Possible causes of instability:
- New data includes a novel regime observation not present in the previous training set.
- EM converged to a different local optimum.

Action: if either check fails, keep the old model and retry retraining with the old model's parameters as initialization (warm start). Maximum 2 retry attempts. If both fail, see Section 7.3.1 for the forced-switch protocol.

### 7.3.1 Open Position Handling During Model Retraining

When a retraining event occurs (monthly or triggered by anomaly):

- **If Stability ≥ 90% (model switches cleanly):** No position action required. New regime signals take effect for new entries only; existing positions carry their original entry rationale until their natural exit condition is met.
- **If Stability < 90% (warm-start retry):** During the retry window, the old model remains active. All position sizing uses old model signals. New entries are paused until stability is confirmed or the retry is completed (maximum 2 retry attempts).
- **If retry fails twice:** Force switch to the new model regardless of stability score. Log a "forced model switch" event. For any open positions flagged as size > 50% of max allowed, reduce to 50% of current size as a precautionary hedge until the new model stabilizes over the next 5 trading days.
- **Cold start (first model):** No prior model exists; use K-means label assignment (by RV rank) as the reference "old model" for stability comparison.

### 7.4 Smoothing the Regime Signal

Raw forward-algorithm output can be noisy — the regime probability may oscillate between states on consecutive days. Apply minimal smoothing:

p_smooth(t) = (1 - λ) · p_raw(t) + λ · p_smooth(t-1)

where λ = 0.3 (exponential smoothing). This prevents single-day regime flickers from triggering trade actions while preserving responsiveness to genuine transitions.

**Do not over-smooth.** Large λ (e.g., 0.9) would delay regime detection by weeks, defeating the purpose. The recommended λ = 0.3 corresponds to an effective lookback of about 3 days.

---

## 8. XGBoost Classifier Layer

### 8.1 Why Add XGBoost?

The HMM emission model assumes multivariate Gaussian features within each regime. Real market features are not Gaussian — they have fat tails, skewness, and nonlinear interactions. XGBoost can capture these without parametric assumptions.

The architecture is: **HMM generates regime labels (via Viterbi on training data) → XGBoost learns to predict these labels from features.**

At inference time, XGBoost runs on the current feature vector to produce regime probabilities. This is a distillation: the HMM's generative model is compressed into a discriminative classifier that handles non-Gaussianity better.

### 8.2 Training Protocol

1. Train HMM on the full training set using Baum-Welch.
2. Run Viterbi on the training set to produce a label sequence {regime(t)}_{t=1}^T.
3. Train XGBoost on (features, labels) pairs.

**XGBoost configuration:**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| n_estimators | 200 | Enough for 5 features; more risks overfitting |
| max_depth | 4 | Captures nonlinear interactions without overfitting |
| learning_rate | 0.1 | Standard; combined with 200 trees gives good bias-variance |
| objective | multi:softprob | Outputs probability vector, not hard labels |
| num_class | 3 | LV, NV, HV |
| min_child_weight | 20 | Prevents splits on tiny subsets |
| subsample | 0.8 | Row subsampling for regularization |
| colsample_bytree | 0.8 | Feature subsampling for regularization |
| eval_metric | mlogloss | Multi-class log loss |

### 8.3 Feature Set for XGBoost

XGBoost receives the same 5 features as the HMM, plus derived features that exploit tree-based models' strength with nonlinear interactions:

**Additional features (computed point-in-time):**

- RV₅ / RV₂₁: Short-term to medium-term vol ratio. Values > 1 indicate accelerating volatility.
- |ΔVol| / RV₂₁: Normalized volatility change magnitude.
- VIX × TS: Interaction between VIX level and term structure. High VIX + backwardation = extreme stress.
- sign(ΔVol) × VIX: Signed directional signal.

Total features for XGBoost: 9 (5 primary + 4 derived).

**Validation requirement:** Before including derived features in production, verify that each improves out-of-sample Brier score by > 2% vs the base 5 features alone (measured on the held-out validation set from Section 12.1). Use built-in XGBoost feature importance (gain metric); features with gain < 0.01 are dropped. The derived features above were selected based on financial intuition — empirical validation determines which survive. Note that `sign(ΔVol) × VIX` introduces a discontinuity at ΔVol = 0; XGBoost handles this via splits but the feature may add noise near the boundary.

### 8.4 Inference

At time t:

1. Compute the 9 features using data up to t.
2. Pass through XGBoost: p_XGB(t) = XGBoost.predict_proba(features(t)).
3. Output: p_XGB(t) = [P(LV), P(NV), P(HV)].

XGBoost inference is a forward pass through decision trees — sub-millisecond, no look-ahead possible by construction.

### 8.5 HMM vs XGBoost: When to Use Which

| Criterion | HMM (forward algorithm) | XGBoost |
|-----------|------------------------|---------|
| Captures temporal dynamics | Yes (transition matrix) | No (treats each point independently) |
| Handles non-Gaussian features | No (Gaussian emissions) | Yes (tree-based, nonparametric) |
| Provides smooth transitions | Yes (probability evolves gradually) | No (can jump between states) |
| Speed | O(K²) per step | O(n_trees) per step — both negligible |

**Recommended approach:** Use XGBoost as the primary regime classifier, with the HMM transition matrix as a regularizer. Specifically:

p_combined(t) = (1 - w_HMM) · p_XGB(t) + w_HMM · A^T · p_combined(t-1)

where A is the HMM transition matrix and w_HMM = 0.3 controls how much temporal persistence to impose. This blends XGBoost's nonparametric accuracy with HMM's sequential smoothness.

---

## 9. Market Application I: Options (SPX/VIX)

### 9.1 Features

| # | Feature | Source | Frequency | Lookback |
|---|---------|--------|-----------|----------|
| 1 | Realized vol (σ_RV₂₁) | ^GSPC daily returns | Daily | 21 days |
| 2 | VIX | CBOE | Daily | Point-in-time |
| 3 | Term structure slope | VIX, VIX3M from CBOE | Daily | Point-in-time |
| 4 | VVIX | CBOE | Daily | Point-in-time |
| 5 | ΔVol | Derived from σ_RV₂₁ | Daily | 21-day difference |

### 9.2 Typical Regime Characteristics (Historical)

Based on SPX data 2006-2024:

**LV (Low Vol):** μ_RV ≈ 8-12%, VIX ≈ 10-14, TS > 0 (contango), VVIX ≈ 80-95. Examples: 2017 (extreme low vol), mid-2019, late 2023 to early 2024. Occupancy: ~30% of trading days.

**NV (Normal Vol):** μ_RV ≈ 12-20%, VIX ≈ 14-25, TS mixed, VVIX ≈ 95-115. The default state. Occupancy: ~50% of trading days.

**HV (High Vol):** μ_RV ≈ 20-80%, VIX ≈ 25-80, TS < 0 (backwardation common), VVIX > 115. Examples: Q4 2008 (GFC), August 2011 (debt ceiling), Q1 2020 (COVID), September 2022 (rate shock). Occupancy: ~20% of trading days.

### 9.3 Transition Dynamics

The typical transition matrix looks approximately like:

```
         To LV    To NV    To HV
From LV [ 0.97    0.03    0.00 ]
From NV [ 0.02    0.95    0.03 ]
From HV [ 0.00    0.05    0.95 ]
```

Key observations:
- LV → HV direct transition is extremely rare (probability ≈ 0). Markets almost always pass through NV before reaching HV.
- HV → LV direct transition is also rare. Markets de-stress gradually.
- Within-regime persistence is high (diagonal > 0.95). The average regime duration is 1/(1-aᵢᵢ) ≈ 20-33 trading days (1-1.5 months).

### 9.4 Integration with EPGP Confidence Filter

For options trading, the regime signal modulates the EPGP confidence filter as follows:

| Regime | Base Position Scaler | EPGP C_min Threshold | Behavior |
|--------|---------------------|---------------------|----------|
| LV | 0.5 | 0.5 | Trade smaller (thin premiums) but accept lower confidence |
| NV | 1.0 | 0.6 | Full size at standard confidence |
| HV | 0.25 | 0.8 | Aggressive reduction; require very high confidence to trade at all |

---

## 10. Market Application II: Crypto Perpetual Funding Rate

### 10.1 Features

| # | Feature | Source | Frequency | Lookback |
|---|---------|--------|-----------|----------|
| 1 | BTC realized vol | BTC daily returns | Daily | 21 days (calendar) |
| 2 | Funding rate (avg across exchanges) | Binance, Bybit, OKX | 8-hourly, daily avg | Point-in-time |
| 3 | Funding rate dispersion | Std across exchanges | 8-hourly, daily avg | Point-in-time |
| 4 | Volume change | BTC spot volume | Daily | 21-day log ratio |
| 5 | ΔVol | Derived from BTC σ_RV | Daily | 21-day difference |

### 10.2 Crypto-Specific Regime Characteristics

**Note:** The RV ranges below are heuristic estimates based on BTC price data circa 2019–2024 (source: Binance BTCUSDT daily OHLCV). Crypto volatility regimes have shifted over time; the HMM will learn the boundaries appropriate to the training period. These ranges serve as initialization guidance only.

**LV:** BTC RV ≈ 20-40% (crypto "calm" is equity "stressed"), funding rate near zero and stable, low dispersion. Typically during ranging/accumulation phases.

**NV:** BTC RV ≈ 40-70%, funding rate mildly positive or negative, moderate dispersion. Typical trending markets with reasonable leverage.

**HV:** BTC RV > 70%, funding rate extremely positive (mania) or extremely negative (panic), high dispersion across exchanges, volume spikes. Includes both euphoric pump phases and crash phases — distinguishing these may require K = 4 states (LV, NV, HV-Long, HV-Short), but start with K = 3 and validate.

### 10.3 Key Difference from Options Application

Crypto markets trade 24/7. The feature computation and HMM inference must account for:

- No "trading days" — use calendar days, with annualization factor √365.
- Data arrives every 8 hours (funding settlements), not daily. Options: run HMM daily. Crypto: optionally run HMM at each funding epoch (3x daily) or aggregate to daily. Start with daily for consistency.
- Weekends and holidays are normal trading periods in crypto. No calendar adjustment needed.
- Exchange downtime is possible. Handle per Section 4.5.

---

## 11. Dual Detection Integration with EPGP

### 11.1 The Combination Formula

At each decision point t, two signals are available:

- From HMM/XGBoost: p(t) = [P(LV), P(NV), P(HV)]
- From EPGP: Z_Φ(t) = z-score of aggregate posterior variance

These are combined into a single regime-confidence score:

**Step 1: Convert HMM to a risk score.**

R_HMM(t) = P(HV | o₁:ₜ) + 0.5 · P(NV | o₁:ₜ)

R_HMM ∈ [0, 1]. Value of 0 = pure LV. Value of 1 = pure HV. Linear blend for NV.

**Step 2: Convert EPGP to a risk score.**

R_EPGP(t) = sigmoid(Z_Φ(t) - Z_threshold)

where sigmoid(x) = 1 / (1 + exp(-x)) and Z_threshold = 1.5. R_EPGP ∈ [0, 1]. Low posterior variance → R_EPGP ≈ 0. High posterior variance → R_EPGP ≈ 1.

**Step 3: Combine.**

R_combined(t) = 1 - (1 - R_HMM(t)) · (1 - R_EPGP(t))

This is a "noisy-OR" combination: if either sensor detects high risk, the combined risk is high. Both sensors must agree on low risk for the combined score to be low. This is conservative by design.

**Step 4: Map to position scaler.**

Scaler(t) = max(0.1, 1 - R_combined(t))

Scaler ∈ [0.1, 1.0]. Even in the worst regime, we maintain a 10% minimum position to avoid missing a recovery entirely.

### 11.2 Agreement and Disagreement

| HMM Says | EPGP Says | R_combined | Action |
|----------|-----------|------------|--------|
| LV (safe) | Low variance (safe) | Low (~0.1) | Full position. High conviction in calm conditions. |
| LV (safe) | High variance (danger) | Moderate (~0.5) | Reduce position. EPGP sees something HMM doesn't. Possible novel structural break. |
| HV (danger) | Low variance (safe) | Moderate (~0.5) | Reduce position. HMM sees statistical stress; EPGP doesn't see PDE breakdown. Possible false alarm from HMM, or EPGP is lagging. |
| HV (danger) | High variance (danger) | High (~0.9) | Minimum position. Both sensors agree — high conviction in dangerous conditions. |

The second and third rows — sensor disagreement — are the most interesting cases. They occur when the market is in a transitional or ambiguous state. The conservative combined response (reduce position) ensures that disagreement never leads to aggressive positioning.

### 11.3 Latency Analysis

**Important: The estimates below are unvalidated hypotheses based on the structural properties of each signal. They have NOT been confirmed by backtesting. Section 19.2 (Phase 3 Roadmap) specifies the empirical validation procedure. Treat these as experimental predictions, not established facts.**

| Event Type | HMM Detection Latency | EPGP Detection Latency | Which Detects First? (Hypothesis) |
|------------|----------------------|----------------------|---------------------|
| Gradual vol increase | 5-15 days (features change slowly) | 3-7 days (vol surface distortion appears earlier) | EPGP (hypothesis) |
| Sudden crash (gap event) | 1-2 days (VIX spikes immediately) | 1 day (current snapshot is distorted) | Simultaneous (hypothesis) |
| Liquidity withdrawal | 5-10 days (may not show in VIX initially) | 1-3 days (bid-ask widening distorts PDE fit) | EPGP (hypothesis) |
| Sentiment shift (no vol change) | 3-5 days (features like TS slope react) | Not detected (PDE structure unchanged) | HMM only (hypothesis) |
| Slow structural change | 10-20 days (gradual statistical drift) | 5-10 days (cumulative PDE residual) | EPGP (hypothesis) |

These latency estimates are hypotheses to be validated in Phase 3 backtesting.

---

## 12. Validation Framework

### 12.1 HMM-Specific Validation Metrics

| Metric | Definition | Target |
|--------|-----------|--------|
| Brier Score | Mean squared error of probabilistic forecasts | < 0.25 |
| Label Stability | Agreement between consecutive training runs | ≥ 90% |
| State Occupancy Balance | Min state occupancy | > 10% |
| Transition Matrix Diagonal | Min diagonal element | > 0.90 |
| BIC Comparison | BIC(K=3) vs BIC(K=2) and BIC(K=4) | K=3 wins |
| Out-of-Sample Log-Likelihood | Log-likelihood on held-out data | Higher than K=2 baseline |

### 12.2 Brier Score Computation

The Brier score for regime k at time t is:

BS(t) = Σₖ (pₖ(t) - 𝟙[true regime = k])²

where pₖ(t) is the predicted probability and 𝟙[·] is the indicator function.

**Problem:** We do not know the "true" regime in real time. We use a proxy: the Viterbi decoding on the full sample (including future data) as the "ground truth" for validation purposes only. This gives the HMM its best possible labeling as a benchmark.

Compute BS on a rolling out-of-sample basis:
1. Train HMM on [1, t].
2. Compute forward probabilities p(t+1) using only data up to t+1.
3. Compare to Viterbi labels from [1, T] (full sample).
4. Average BS over all t.

Target: BS < 0.25 (better than predicting the marginal state distribution).

### 12.3 Incremental Value Test

The most important validation: does the HMM improve the EPGP trading system?

**Test 1: HMM vs No Regime Filter**
- Strategy A: EPGP confidence filter only (no HMM).
- Strategy B: EPGP confidence filter + HMM regime filter.
- Compare Sharpe, MDD, Calmar. If B ≈ A, HMM adds no value.

**Test 2: HMM vs EPGP Variance Only**
- Strategy C: HMM regime filter only (no EPGP).
- Strategy D: EPGP variance filter only (no HMM).
- Strategy E: Dual (HMM + EPGP).
- If E > C and E > D, the dual system has genuine synergy.

**Test 3: Regime Transition Detection Speed**
- For known historical events (March 2020 crash, 2022 rate shock):
  - How many days before peak drawdown did HMM signal P(HV) > 0.5?
  - How many days before peak drawdown did EPGP signal Z_Φ > 2?
  - Which detected earlier?

---

## 13. Implementation Specification

### 13.1 Library Selection

| Component | Library | Version | Rationale |
|-----------|---------|---------|-----------|
| HMM | hmmlearn | ≥ 0.3.0 | Standard, well-tested, supports Gaussian emissions |
| XGBoost | xgboost | ≥ 2.0.0 | Industry standard gradient boosting |
| Feature computation | pandas + numpy | Latest stable | Standard data manipulation |
| Optimization | scipy | Latest stable | L-BFGS for hyperparameters |

### 13.2 Core Class Interface

```python
from dataclasses import dataclass
from datetime import date
import numpy as np

@dataclass(frozen=True)
class RegimeState:
    """Output of regime detection at a single time step."""
    date: date
    probabilities: np.ndarray      # [P(LV), P(NV), P(HV)], shape (3,)
    regime: str                     # 'LV', 'NV', or 'HV' (argmax)
    confidence: float               # max(probabilities)
    entropy: float                  # -sum(p * log(p))
    risk_score: float               # R_HMM = P(HV) + 0.5 * P(NV)
    position_scaler: float          # Regime-based scaler ∈ [0.1, 1.0]


class RegimeDetector:
    """HMM + XGBoost regime detection module."""
    
    def __init__(self, n_states: int = 3, n_init: int = 10,
                 min_train_days: int = 504, retrain_frequency: str = 'monthly'):
        ...

    def fit(self, features: np.ndarray, dates: np.ndarray) -> None:
        """Train HMM + XGBoost on historical features.
        
        Args:
            features: (T, d) array of standardized feature vectors.
            dates: (T,) array of dates corresponding to each row.
        
        Side effects:
            - Fits HMM via Baum-Welch with multi-start initialization.
            - Generates Viterbi labels on training data.
            - Trains XGBoost on (features, Viterbi labels).
            - Assigns semantic labels (LV, NV, HV) by sorting mean RV.
            - Stores transition matrix for temporal regularization.
        
        Raises:
            ValueError: If len(features) < min_train_days.
            RuntimeError: If label stability < 90% vs previous model.
        """
        ...

    def predict(self, features_t: np.ndarray) -> RegimeState:
        """Real-time regime inference for a single time step.
        
        Args:
            features_t: (d,) array of current features (standardized).
        
        Returns:
            RegimeState with probabilities, regime, confidence, etc.
        
        Implementation:
            1. XGBoost predicts p_XGB from features_t.
            2. Blend with HMM transition: p = (1-w) * p_XGB + w * A^T * p_prev.
            3. Apply exponential smoothing: p_smooth = (1-λ)*p + λ*p_prev_smooth.
            4. Compute regime, confidence, entropy, risk_score, position_scaler.
        """
        ...

    def check_stability(self, old_model: 'RegimeDetector',
                        recent_features: np.ndarray) -> float:
        """Compare label assignments between old and new model.
        
        Returns:
            Stability score ∈ [0, 1]. Must be ≥ 0.90 to accept new model.
        """
        ...

    def get_label_series(self, features: np.ndarray) -> np.ndarray:
        """Produce Viterbi labels for a feature sequence (training use only).
        
        WARNING: This uses the Viterbi algorithm (full sequence) and must
        NEVER be used in backtesting or live trading. It is for generating
        XGBoost training labels only.
        """
        ...
```

### 13.3 Feature Computation

```python
class FeatureEngine:
    """Computes regime detection features from market data.
    
    All computations are strictly point-in-time: feature(t) uses only
    data from dates ≤ t. Expanding statistics (mean, std) for 
    standardization use all data up to t.
    """
    
    def compute_options_features(self, market_data: dict, 
                                  as_of_date: date) -> np.ndarray:
        """Compute 5-feature vector for options regime detection.
        
        Args:
            market_data: dict with keys 'spx_returns', 'vix', 'vix3m', 
                        'vvix' — all as pd.Series indexed by date.
            as_of_date: The date for which to compute features.
        
        Returns:
            np.ndarray of shape (5,): [RV21, VIX, TS, VVIX, ΔVol],
            standardized using expanding mean/std up to as_of_date.
        """
        ...

    def compute_crypto_features(self, market_data: dict,
                                 as_of_date: date) -> np.ndarray:
        """Compute 5-feature vector for crypto regime detection.
        
        Args:
            market_data: dict with keys 'btc_returns', 'funding_rates'
                        (dict of exchange → Series), 'btc_volume'.
            as_of_date: The date for which to compute features.
        
        Returns:
            np.ndarray of shape (5,): [BTC_RV21, funding_avg, 
            funding_dispersion, volume_change, ΔVol], standardized.
        """
        ...
```

### 13.4 Critical Implementation Warnings

**Warning 1: Forward algorithm only.** For real-time inference and backtesting, never call `model.predict()` or `model.predict_proba()` from hmmlearn. These use forward-backward (smoothed) probabilities. Instead, extract the forward pass manually:

```python
# CORRECT: Forward-only inference
from hmmlearn import hmm
import numpy as np

def forward_only_proba(model: hmm.GaussianHMM, 
                        observation: np.ndarray,
                        prev_log_alpha: np.ndarray) -> tuple:
    """Single-step forward algorithm. No look-ahead.
    
    Args:
        model: Trained GaussianHMM
        observation: (1, d) array — current feature vector
        prev_log_alpha: (K,) array — log forward variables from t-1
    
    Returns:
        (posterior, new_log_alpha): posterior probabilities and updated 
        log forward variables
    """
    n_states = model.n_components
    log_transmat = np.log(model.transmat_ + 1e-300)
    
    # Prediction step: log α̂_k(t) = log Σ_j α_j(t-1) * a_jk
    log_alpha_pred = np.zeros(n_states)
    for k in range(n_states):
        log_alpha_pred[k] = logsumexp(prev_log_alpha + log_transmat[:, k])
    
    # Update step: log α_k(t) = log α̂_k(t) + log P(o_t | state k)
    log_emission = np.array([
        multivariate_normal.logpdf(observation, 
                                    model.means_[k], 
                                    model.covars_[k])
        for k in range(n_states)
    ])
    new_log_alpha = log_alpha_pred + log_emission
    
    # Normalize to get posterior
    log_posterior = new_log_alpha - logsumexp(new_log_alpha)
    posterior = np.exp(log_posterior)
    
    return posterior, new_log_alpha
```

**Warning 2: Expanding standardization.** Feature standardization must use only data up to time t. Never standardize with the full-sample mean/std.

**Warning 3: Retraining schedule.** Retrain monthly on the first trading day of each month. Use all data from system start to current date. Compare new model to old model via label stability check before switching.

**Warning 4: NaN handling.** If VVIX is unavailable (pre-2012), retrain a separate HMM with d = 4 features. Maintain two model variants and select based on data availability.

---

## 14. Parameter Reference

### 14.1 HMM Parameters

| Parameter | Symbol | Default | Range | Tunable? |
|-----------|--------|---------|-------|----------|
| Number of states | K | 3 | {2, 3, 4} — select via BIC | Validated, not tuned |
| Covariance type | — | full | {diag, full} | Fixed at full |
| Number of initializations | n_init | 10 | [5, 20] | Fixed |
| Min training window | W_min | 504 days | [252, 756] | Fixed |
| Convergence tolerance | ε | 10⁻⁴ | — | Fixed |
| Max EM iterations | — | 200 | — | Fixed |
| Retraining frequency | — | Monthly | {Monthly, Quarterly} | Fixed |
| Label stability threshold | — | 90% | [80%, 95%] | Fixed |

### 14.2 XGBoost Parameters

| Parameter | Value | Range | Tunable? |
|-----------|-------|-------|----------|
| n_estimators | 200 | [100, 500] | Walk-forward |
| max_depth | 4 | [3, 6] | Walk-forward |
| learning_rate | 0.1 | [0.05, 0.2] | Walk-forward |
| min_child_weight | 20 | [10, 50] | Walk-forward |
| subsample | 0.8 | [0.6, 1.0] | Walk-forward |
| colsample_bytree | 0.8 | [0.6, 1.0] | Walk-forward |

### 14.3 Integration Parameters

| Parameter | Symbol | Default | Range | Tunable? |
|-----------|--------|---------|-------|----------|
| HMM transition weight | w_HMM | 0.3 | [0.0, 0.5] | Walk-forward |
| Exponential smoothing | λ | 0.3 | [0.1, 0.5] | Walk-forward |
| EPGP z-score threshold | Z_threshold | 1.5 | [1.0, 2.5] | Walk-forward |
| Min position scaler | — | 0.1 | [0.0, 0.25] | Fixed |

### 14.4 Feature Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| RV window | 21 trading days | Standard 1-month |
| ΔVol lookback | 21 trading days | Matches RV window |
| Standardization | Expanding (not rolling) | Prevents look-ahead |
| Crypto annualization | √365 | 24/7 trading |
| Equity annualization | √252 | Standard trading days |

---

## 15. Failure Modes and Diagnostics

### 15.1 Failure Mode: State Collapse

**Symptom:** One state has < 5% occupancy after training.
**Cause:** EM converged to a solution where one state is rarely visited.
**Diagnosis:** Check state occupancies after training.
**Fix:** Re-initialize with K-means. If persistent, reduce to K = 2. Run BIC comparison.

### 15.2 Failure Mode: Label Flip

**Symptom:** Label stability < 90% after retraining. Trading signals suddenly inverted.
**Cause:** EM converged to a permuted solution. What was "State 0" (LV) is now "State 2" (HV).
**Diagnosis:** Compare new and old state means. Check if sorting by μ_RV resolves the flip.
**Fix:** The label alignment protocol (Section 7.2) should catch this automatically. If it fails, warm-start retraining from old model parameters.

### 15.3 Failure Mode: Persistent NV

**Symptom:** HMM assigns P(NV) > 0.8 for months, never detecting LV or HV.
**Cause:** NV emission distribution is too broad, absorbing observations that should belong to LV or HV.
**Diagnosis:** Plot feature distributions by Viterbi state. If NV's σ is much larger than LV or HV, the model is under-separating.
**Fix:** Increase n_init. Try initializing with manually specified means (e.g., VIX = 12 for LV, VIX = 18 for NV, VIX = 35 for HV). Alternatively, constrain covariances during EM (regularization).

### 15.4 Failure Mode: Oscillating Regime

**Symptom:** Regime switches between LV and HV on consecutive days.
**Cause:** Features are near the decision boundary. The forward algorithm probability fluctuates around 0.5.
**Diagnosis:** Check entropy H(t). High entropy (> 0.9) indicates genuine ambiguity.
**Fix:** The exponential smoothing (λ = 0.3) should prevent this. If smoothing is insufficient, increase λ to 0.4. If still oscillating, treat the ambiguous period as NV (default to moderate risk).

### 15.5 Failure Mode: XGBoost Overfitting

**Symptom:** Training accuracy >> out-of-sample accuracy. In-sample Brier score << out-of-sample.
**Cause:** XGBoost memorized training labels instead of learning generalizable patterns.
**Diagnosis:** Compare training and validation metrics.
**Fix:** Reduce n_estimators to 100. Increase min_child_weight to 50. Add early stopping based on validation loss. Use stratified time-series cross-validation (not random splits — time series structure must be preserved).

### 15.6 Failure Mode: Data Feed Failure

**Symptom:** VIX, VVIX, or funding rate data is missing or stale.
**Cause:** Data provider outage, exchange downtime, API rate limit.
**Diagnosis:** Check data timestamps. If latest observation is > 24 hours old, flag as stale.
**Fix:** Carry forward last valid observation for up to 2 trading days (options) or 24 hours (crypto). Beyond that, switch to the EPGP-only mode (no HMM contribution to dual signal). Log the event for post-analysis.

### 15.7 Diagnostic Dashboard (Recommended)

Track and visualize daily:
- Current regime probabilities [P(LV), P(NV), P(HV)]
- Regime probability time series (trailing 252 days)
- Feature values and their z-scores
- Label stability metric (rolling 63-day)
- HMM log-likelihood trend (should be non-decreasing with more data)
- EPGP Z_Φ overlaid with HMM P(HV) — visual correlation check
