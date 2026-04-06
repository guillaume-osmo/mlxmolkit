"""
Perfumery COSMO-RS pipeline: SMILES → sigma profiles → activity coefficients.

Full pipeline: SMILES → RDKit 3D → PM6 batch SCF (Metal GPU)
              → PM6-ML correction → COSMO cavity → sigma profile
              → COSMO-RS activity coefficients

Optimized for throughput: batch SCF + cached 3D + vectorized COSMO.
"""
from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional


# Common perfumery molecules
PERFUMERY_SMILES = {
    'Linalool': 'CC(=CCC/C(=C/CO)/C)C',
    'Limonene': 'CC1=CCC(CC1)C(=C)C',
    'Geraniol': 'OC/C=C(\\C)CCC=C(C)C',
    'Vanillin': 'COc1cc(C=O)ccc1O',
    'Coumarin': 'O=C1Oc2ccccc2C=C1',
    'Eugenol': 'COc1cc(CC=C)ccc1O',
    'Citral': 'CC(=CCC=C(C)C=O)C',
    'Musk_xylene': 'CC1=CC(=C(C(=C1[N+](=O)[O-])C)C)[N+](=O)[O-]',
    'Isoeugenol': 'COc1cc(/C=C/C)ccc1O',
    'Cinnamaldehyde': 'O=C/C=C/c1ccccc1',
    'Benzyl_acetate': 'CC(=O)OCc1ccccc1',
    'Ethyl_vanillin': 'CCOc1cc(C=O)ccc1O',
    'Hedione': 'CC(=O)CCCC1(CC=CC1)C(=O)OC',
    'Galaxolide': 'CC1(C)C2CC(C)(C)c3cc(ccc3C2OC1)C',
}

# Common solvents
SOLVENTS = {
    'Water': 'O',
    'Ethanol': 'CCO',
    'Dipropylene_glycol': 'CC(O)COCC(C)O',
    'Isopropyl_myristate': 'CCCCCCCCCCCCCC(=O)OC(C)C',
    'Diethyl_phthalate': 'CCOC(=O)c1ccccc1C(=O)OCC',
}


def compute_perfumery_profiles(
    molecules: Dict[str, str] = None,
    method: str = 'PM6',
    use_ml: bool = False,
    use_metal: bool = True,
) -> Dict[str, dict]:
    """Compute COSMO sigma profiles for perfumery molecules.

    Args:
        molecules: dict of {name: SMILES}. Defaults to common perfumery set.
        method: NDDO method ('PM6', 'RM1', 'PM3')
        use_ml: apply PM6-ML neural network correction
        use_metal: use Metal GPU for SCF

    Returns:
        dict of {name: cosmo_data} with sigma profiles
    """
    from ..rm1.pipeline import _smiles_to_3d
    from ..rm1.scf import rm1_energy_batch
    from ..rm1.methods import get_params
    from .cavity import cosmo_surface
    from .sigma import full_sigma_analysis

    if molecules is None:
        molecules = {**PERFUMERY_SMILES, **SOLVENTS}

    PARAMS = get_params(method)
    names = list(molecules.keys())
    smiles_list = list(molecules.values())

    # Step 1: 3D generation (cached)
    _cache = {}
    mol_data = []
    for smi in smiles_list:
        if smi in _cache:
            mol_data.append(_cache[smi])
        else:
            r = _smiles_to_3d(smi)
            if r:
                _cache[smi] = (list(r[0]), r[1])
                mol_data.append(_cache[smi])
            else:
                mol_data.append(None)

    # Step 2: Batch SCF
    valid = [(i, d) for i, d in enumerate(mol_data) if d is not None]
    scf_mols = [d for _, d in valid]
    scf_results = rm1_energy_batch(scf_mols, method=method, use_metal=use_metal)

    # Step 3: PM6-ML corrections (optional)
    ml_corrections = [0.0] * len(valid)
    if use_ml and method in ('PM6', 'PM6_SP'):
        try:
            from ..rm1.pm6_ml import pm6_ml_correction_batch
            ml_corrections = pm6_ml_correction_batch(scf_mols)
        except ImportError:
            pass

    # Step 4: COSMO + sigma
    results = {}
    for k, (idx, (atoms, coords)) in enumerate(valid):
        if not scf_results[k]['converged']:
            continue
        try:
            cr = cosmo_surface(atoms, coords, scf_results[k]['density'])
            sr = full_sigma_analysis(cr, atoms)
            cr.update(sr)
            cr['name'] = names[idx]
            cr['smiles'] = smiles_list[idx]
            cr['energy_eV'] = scf_results[k]['energy_eV']
            cr['ml_correction_eV'] = ml_corrections[k]
            cr['energy_pm6ml_eV'] = scf_results[k]['energy_eV'] + ml_corrections[k]
            cr['heat_of_formation_kcal'] = scf_results[k]['heat_of_formation_kcal']
            cr['atoms'] = atoms
            cr['coords'] = coords
            results[names[idx]] = cr
        except Exception:
            pass

    return results


def perfumery_activity_coefficients(
    profiles: Dict[str, dict],
    solvent: str = 'Ethanol',
    T: float = 298.15,
    alpha: float = 3.2e7,
) -> Dict[str, dict]:
    """Compute activity coefficients for perfumery molecules in a solvent.

    Returns dict of {name: {gamma_inf, lng, ...}} for each solute.
    """
    from .cosmors import activity_coefficients
    from . import params as P

    P.MF_ALPHA = alpha

    if solvent not in profiles:
        raise ValueError(f"Solvent '{solvent}' not in profiles. Available: {list(profiles.keys())}")

    solvent_prof = profiles[solvent]
    results = {}

    for name, prof in profiles.items():
        if name == solvent:
            continue
        try:
            x = np.array([0.999, 0.001])  # infinite dilution
            lng = activity_coefficients([solvent_prof, prof], x, T=T)
            gamma_inf = np.exp(lng[1])
            results[name] = {
                'gamma_inf': gamma_inf,
                'lng_inf': lng[1],
                'name': name,
                'smiles': prof.get('smiles', ''),
                'n_atoms': prof.get('n_atoms', len(prof.get('atoms', []))),
            }
        except Exception:
            pass

    return results
