"""
mlxmolkit.solvers — Portable fixed-point solvers and optimizers on Metal GPU.

SCFMixer: Anderson/Pulay/DIIS accelerator for any fixed-point iteration.
BatchLBFGS: GPU-parallel L-BFGS for N molecules simultaneously.
"""
from .mixer import SCFMixer, MixerConfig, run_fixed_point
from .lbfgs_metal import (
    lbfgs_direction, lbfgs_direction_batch, BatchLBFGS,
    batched_backtracking_line_search,
)
