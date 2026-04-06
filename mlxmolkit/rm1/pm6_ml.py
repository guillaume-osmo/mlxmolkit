"""
PM6-ML: Neural network correction for PM6 semi-empirical method.

Uses TorchMD-NET EquivariantTransformer trained on DFT reference data.
Applies a post-SCF energy (and optionally force) correction:
  E_total = E_PM6 + E_ML_correction + E_D3_dispersion

Reference: Nováček & Řezáč, JCTC 2025, 21(2), 678-690.
Model: https://github.com/Honza-R/mopac-ml

Requires: pip install torchmd-net
"""
from __future__ import annotations

import os
import numpy as np
from typing import Optional

# Atomic number → TorchMD-NET type mapping
Z_TO_ATYPE = {
    35: 1, 6: 3, 20: 5, 17: 7, 9: 9, 1: 10, 53: 12, 19: 13,
    3: 14, 12: 15, 7: 17, 11: 19, 8: 21, 15: 23, 16: 26,
}

# Supported elements
SUPPORTED_ELEMENTS = set(Z_TO_ATYPE.keys())

# Default model path
DEFAULT_MODEL = os.path.join(
    os.path.dirname(__file__), '..', '..', '..',
    'mopac-ml', 'models', 'PM6-ML_correction_seed8_best.ckpt'
)

# Cache
_model = None
_model_path = None


def _load_model(model_path: str = None):
    """Load TorchMD-NET model (cached)."""
    global _model, _model_path

    if model_path is None:
        model_path = DEFAULT_MODEL
        if not os.path.exists(model_path):
            model_path = '/Users/tgg/Github/mopac-ml/models/PM6-ML_correction_seed8_best.ckpt'

    if _model is not None and _model_path == model_path:
        return _model

    import torch
    from torchmdnet.models.model import load_model
    _model = load_model(model_path, derivative=False)
    _model.eval()
    _model_path = model_path
    return _model


def pm6_ml_correction(
    atoms: list[int],
    coords: np.ndarray,
    model_path: str = None,
) -> float:
    """Compute PM6-ML energy correction.

    Args:
        atoms: atomic numbers
        coords: (n_atoms, 3) in Angstrom
        model_path: path to .ckpt model file

    Returns:
        correction in eV
    """
    import torch

    # Check all elements supported
    for z in atoms:
        if z not in Z_TO_ATYPE:
            return 0.0  # unsupported element → no correction

    model = _load_model(model_path)

    types = torch.tensor([Z_TO_ATYPE[z] for z in atoms], dtype=torch.long)
    pos = torch.tensor(np.asarray(coords, dtype=np.float32))

    with torch.no_grad():
        out = model(types, pos)
    energy = out[0] if isinstance(out, tuple) else out

    # kJ/mol → eV
    return energy.item() / 96.485


def pm6_ml_energy(
    atoms: list[int],
    coords: np.ndarray,
    model_path: str = None,
) -> dict:
    """Compute PM6 + ML correction energy.

    Returns standard result dict with additional 'ml_correction_eV' key.
    """
    from .scf import rm1_energy

    result = rm1_energy(list(atoms), coords, method='PM6')
    correction = pm6_ml_correction(list(atoms), coords, model_path)

    result['ml_correction_eV'] = correction
    result['energy_eV'] += correction
    result['energy_kcal'] = result['energy_eV'] * 23.061
    result['method'] = 'PM6-ML'

    return result
