"""
PM6-ML: Neural network correction for PM6 semi-empirical method.

Uses TorchMD-NET EquivariantTransformer trained on DFT reference data.
E_total = E_PM6 + E_ML_correction

Supports batched inference for N molecules.

Reference: Nováček & Řezáč, JCTC 2025, 21(2), 678-690.
"""
from __future__ import annotations

import os
import numpy as np
from typing import Optional, List

Z_TO_ATYPE = {
    35: 1, 6: 3, 20: 5, 17: 7, 9: 9, 1: 10, 53: 12, 19: 13,
    3: 14, 12: 15, 7: 17, 11: 19, 8: 21, 15: 23, 16: 26,
}
SUPPORTED_ELEMENTS = set(Z_TO_ATYPE.keys())

_model = None
_model_path = None


def _load_model(model_path: str = None):
    global _model, _model_path
    if model_path is None:
        model_path = '/Users/tgg/Github/mopac-ml/models/PM6-ML_correction_seed8_best.ckpt'
    if _model is not None and _model_path == model_path:
        return _model
    import torch
    from torchmdnet.models.model import load_model
    _model = load_model(model_path, derivative=False)
    _model.eval()
    _model_path = model_path
    return _model


def pm6_ml_correction(atoms: list[int], coords: np.ndarray) -> float:
    """Single molecule PM6-ML energy correction (eV)."""
    import torch
    for z in atoms:
        if z not in Z_TO_ATYPE:
            return 0.0
    model = _load_model()
    types = torch.tensor([Z_TO_ATYPE[z] for z in atoms], dtype=torch.long)
    pos = torch.tensor(np.asarray(coords, dtype=np.float32))
    with torch.no_grad():
        out = model(types, pos)
    energy = out[0] if isinstance(out, tuple) else out
    return energy.item() / 96.485  # kJ/mol → eV


def pm6_ml_correction_batch(
    molecules: List[tuple],
) -> List[float]:
    """Batch PM6-ML corrections for N molecules.

    Args:
        molecules: list of (atoms, coords) tuples

    Returns:
        list of corrections in eV
    """
    import torch
    model = _load_model()
    corrections = []

    for atoms, coords in molecules:
        if any(z not in Z_TO_ATYPE for z in atoms):
            corrections.append(0.0)
            continue
        types = torch.tensor([Z_TO_ATYPE[z] for z in atoms], dtype=torch.long)
        pos = torch.tensor(np.asarray(coords, dtype=np.float32))
        with torch.no_grad():
            out = model(types, pos)
        energy = out[0] if isinstance(out, tuple) else out
        corrections.append(energy.item() / 96.485)

    return corrections


def pm6_ml_energy(atoms: list[int], coords: np.ndarray) -> dict:
    """PM6 + ML correction energy."""
    from .scf import rm1_energy
    result = rm1_energy(list(atoms), coords, method='PM6')
    correction = pm6_ml_correction(list(atoms), coords)
    result['ml_correction_eV'] = correction
    result['energy_eV'] += correction
    result['energy_kcal'] = result['energy_eV'] * 23.061
    result['method'] = 'PM6-ML'
    return result
