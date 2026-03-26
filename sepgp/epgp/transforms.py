"""
Transforms between Black-Scholes and heat equation domains.

The Black-Scholes PDE for a European option:
    ∂V/∂t + ½σ²S² ∂²V/∂S² + rS ∂V/∂S - rV = 0

is transformed to the standard heat equation v_τ = v_xx via:

    S = K exp(x)
    t = T - 2τ/σ²
    V(S, t) = K exp(α x + β τ) v(x, τ)

where:
    k = 2r/σ²
    α = -(k-1)/2
    β = -((k+1)/2)²
    
Forward transform: (S, t, V) → (x, τ, v)
Inverse transform: (x, τ, v) → (S, t, V)
"""

import torch
from torch import Tensor
from typing import NamedTuple
import math


class BSParams(NamedTuple):
    """Black-Scholes parameters."""
    sigma: float    # volatility
    r: float        # risk-free rate
    K: float        # strike price
    T: float        # time to expiry
    
    @property
    def k(self) -> float:
        return 2 * self.r / (self.sigma ** 2)
    
    @property
    def alpha(self) -> float:
        return -(self.k - 1) / 2
    
    @property
    def beta(self) -> float:
        return -((self.k + 1) / 2) ** 2


def bs_to_heat(
    S: Tensor, t: Tensor, V: Tensor, params: BSParams
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Transform Black-Scholes variables to heat equation variables.
    
    Args:
        S: (N,) stock prices
        t: (N,) times (in BS convention: t=0 is now, t=T is expiry)
        V: (N,) option prices
        params: BS parameters
    
    Returns:
        x: (N,) spatial coordinate in heat domain
        tau: (N,) temporal coordinate in heat domain
        v: (N,) solution value in heat domain
    """
    x = torch.log(S / params.K)
    tau = 0.5 * params.sigma ** 2 * (params.T - t)
    
    # V = K exp(αx + βτ) v  →  v = V / (K exp(αx + βτ))
    v = V / (params.K * torch.exp(params.alpha * x + params.beta * tau))
    
    return x, tau, v


def heat_to_bs(
    x: Tensor, tau: Tensor, v: Tensor, params: BSParams
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Transform heat equation variables back to Black-Scholes.
    
    Args:
        x: (N,) spatial coordinate in heat domain
        tau: (N,) temporal coordinate in heat domain
        v: (N,) solution value in heat domain
        params: BS parameters
    
    Returns:
        S: (N,) stock prices
        t: (N,) times in BS convention
        V: (N,) option prices
    """
    S = params.K * torch.exp(x)
    t = params.T - 2 * tau / (params.sigma ** 2)
    V = params.K * torch.exp(params.alpha * x + params.beta * tau) * v
    
    return S, t, V


def bs_call_price(S: Tensor, t: Tensor, params: BSParams) -> Tensor:
    """
    Analytical Black-Scholes European call price.
    
    C(S,t) = S·N(d1) - K·exp(-r(T-t))·N(d2)
    
    where:
        d1 = [ln(S/K) + (r + σ²/2)(T-t)] / (σ√(T-t))
        d2 = d1 - σ√(T-t)
    """
    tau_bs = params.T - t  # time to maturity
    tau_bs = tau_bs.clamp(min=1e-10)  # avoid division by zero
    
    d1 = (torch.log(S / params.K) + (params.r + 0.5 * params.sigma**2) * tau_bs) / (
        params.sigma * torch.sqrt(tau_bs)
    )
    d2 = d1 - params.sigma * torch.sqrt(tau_bs)
    
    # Standard normal CDF
    N = torch.distributions.Normal(0, 1)
    call = S * N.cdf(d1) - params.K * torch.exp(-params.r * tau_bs) * N.cdf(d2)
    
    return call


def heat_ic_from_bs_call(x: Tensor, params: BSParams) -> Tensor:
    """
    Initial condition for heat equation corresponding to European call payoff.
    
    At τ=0 (i.e., t=T, at expiry): V(S,T) = max(S-K, 0) = K·max(e^x - 1, 0).
    
    Transforming: v(x, 0) = max(e^x - 1, 0) / exp(αx)
                          = max(e^{(1-α)x} - e^{-αx}, 0)
    
    With α = -(k-1)/2:
        1 - α = 1 + (k-1)/2 = (k+1)/2
        -α = (k-1)/2
    
    So: v(x, 0) = max(e^{(k+1)x/2} - e^{(k-1)x/2}, 0)
    """
    kp = (params.k + 1) / 2
    km = (params.k - 1) / 2
    
    payoff = torch.exp(kp * x) - torch.exp(km * x)
    v0 = torch.clamp(payoff, min=0.0)
    return v0
