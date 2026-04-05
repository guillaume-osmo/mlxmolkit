"""
Batch COSMO pipeline: N molecules SMILES → sigma profiles in one call.

Uses RM1 batch SCF (Metal GPU) then vectorized COSMO cavity + sigma.
"""
from __future__ import annotations

import numpy as np
from typing import Optional


def batch_smiles_to_cosmo(
    smiles_list: list[str],
    method: str = 'RM1',
    n_surface_points: int = 194,
    epsilon: float = 78.39,
    use_metal: bool = True,
    verbose: bool = False,
) -> list[Optional[dict]]:
    """Generate COSMO surfaces for N molecules from SMILES (batched).

    Pipeline:
      1. RDKit 3D generation (per molecule)
      2. RM1 batch SCF (Metal GPU, all N at once)
      3. COSMO cavity + sigma profile (per molecule, vectorized numpy)

    Args:
        smiles_list: list of SMILES strings
        method: NDDO method ('RM1', 'AM1', 'PM3', etc.)
        n_surface_points: Lebedev points per atom
        epsilon: dielectric constant
        use_metal: use Metal GPU for SCF
        verbose: print progress

    Returns:
        list of COSMO result dicts (None for failed molecules)
    """
    from ..rm1.pipeline import _smiles_to_3d
    from ..rm1.scf import rm1_energy_batch
    from ..rm1.methods import get_params
    from .cavity import cosmo_surface
    from .sigma import full_sigma_analysis

    N = len(smiles_list)
    PARAMS = get_params(method)

    if verbose:
        print(f"Batch COSMO: {N} molecules, method={method}")

    # Step 1: Generate 3D (sequential — RDKit is fast)
    mol_data = []  # (atoms, coords) for valid mols
    valid_idx = []
    for i, smi in enumerate(smiles_list):
        r = _smiles_to_3d(smi, seed=42 + i)
        if r is None:
            continue
        atoms, coords = r
        if any(z not in PARAMS for z in atoms):
            continue
        mol_data.append((atoms, coords))
        valid_idx.append(i)

    if verbose:
        print(f"  3D generated: {len(mol_data)}/{N}")

    if not mol_data:
        return [None] * N

    # Step 2: Batch RM1 SCF (Metal GPU)
    scf_results = rm1_energy_batch(mol_data, method=method, use_metal=use_metal)

    if verbose:
        n_conv = sum(1 for r in scf_results if r['converged'])
        print(f"  SCF converged: {n_conv}/{len(mol_data)}")

    # Step 3: COSMO cavity + sigma (per molecule, numpy)
    results = [None] * N
    n_cosmo = 0

    for j, idx in enumerate(valid_idx):
        scf = scf_results[j]
        if not scf['converged']:
            continue

        atoms, coords = mol_data[j]
        try:
            cosmo_result = cosmo_surface(
                atoms, coords, scf['density'],
                n_points=n_surface_points, epsilon=epsilon,
            )
            sigma_result = full_sigma_analysis(cosmo_result, atoms)

            results[idx] = {
                'smiles': smiles_list[idx],
                'atoms': atoms,
                'coords': coords,
                'n_atoms': len(atoms),
                'method': method,
                'energy_eV': scf['energy_eV'],
                'heat_of_formation_kcal': scf['heat_of_formation_kcal'],
                **cosmo_result,
                **sigma_result,
            }
            n_cosmo += 1
        except Exception:
            pass

    if verbose:
        print(f"  COSMO computed: {n_cosmo}/{len(mol_data)}")

    return results


def batch_activity_coefficients(
    smiles_list: list[str],
    x: np.ndarray,
    T: float = 298.15,
    method: str = 'RM1',
    epsilon: float = 78.39,
    use_metal: bool = True,
) -> dict:
    """Compute activity coefficients for a mixture (batch COSMO).

    Args:
        smiles_list: SMILES for each component
        x: mole fractions
        T: temperature (K)
        method: NDDO method
        epsilon: dielectric constant
        use_metal: Metal GPU for SCF

    Returns:
        dict with gamma, lng, cosmo_data
    """
    from .cosmors import activity_coefficients

    cosmo_data = batch_smiles_to_cosmo(smiles_list, method=method,
                                        epsilon=epsilon, use_metal=use_metal)

    # Check all succeeded
    for i, r in enumerate(cosmo_data):
        if r is None:
            raise ValueError(f"Failed to compute COSMO for {smiles_list[i]}")

    x = np.asarray(x, dtype=np.float64)
    lng = activity_coefficients(cosmo_data, x, T=T)
    gamma = np.exp(lng)

    return {
        'smiles': smiles_list,
        'x': x,
        'T': T,
        'lng': lng,
        'gamma': gamma,
        'cosmo_data': cosmo_data,
    }
