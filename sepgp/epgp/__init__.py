"""
EPGP: Ehrenpreis-Palamodov Gaussian Processes for solving PDEs.

A PyTorch implementation of S-EPGP for learning solutions of linear
PDE systems with constant coefficients from data.
"""

from .varieties import (
    sample_heat_1d,
    sample_heat_2d,
    sample_wave_2d,
    sample_maxwell_2d,
)
from .kernels import (
    Heat1DKernel,
    Heat2DKernel,
    Wave2DKernel,
    Maxwell2DKernel,
)
from .gp import SEPGP, VectorSEPGP
from .optimize import optimize_mll
from .transforms import BSParams, bs_to_heat, heat_to_bs, bs_call_price, heat_ic_from_bs_call

__all__ = [
    "sample_heat_1d", "sample_heat_2d", "sample_wave_2d", "sample_maxwell_2d",
    "Heat1DKernel", "Heat2DKernel", "Wave2DKernel", "Maxwell2DKernel",
    "SEPGP", "VectorSEPGP",
    "optimize_mll",
    "BSParams", "bs_to_heat", "heat_to_bs", "bs_call_price", "heat_ic_from_bs_call",
]
