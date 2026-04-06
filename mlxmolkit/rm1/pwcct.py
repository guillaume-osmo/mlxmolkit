"""
PWCCT (Pairwise Core-Core Terms) for PM6 nuclear repulsion.

PM6 uses a different nuclear repulsion formula than AM1/RM1:
  E_nuc = unpolcore + Z_A*Z_B * gam * (1 + 2*chi*exp(-alpha*(R + 0.0003*R^6)))
  + special terms for C-H, N-H, O-H, C-C, Si-O

PWCCT parameters (chi, alpha) are loaded from the 6889-line CSV.

Reference: Stewart, J. Mol. Model. 2007, 13, 1173-1213.
"""
from __future__ import annotations

import numpy as np
from typing import Dict, Tuple

# PWCCT parameters: (Z1, Z2) → (chi, alpha)
_PWCCT_CACHE: Dict[Tuple[int, int], Tuple[float, float]] = {}
_PWCCT_LOADED = False


def _load_pwcct(filepath: str = None):
    """Load PWCCT parameters from CSV."""
    global _PWCCT_CACHE, _PWCCT_LOADED
    if _PWCCT_LOADED:
        return

    if filepath is None:
        import os
        filepath = os.path.join(
            os.path.dirname(__file__), '..', '..', '..',
            'pyseqm_ref', 'seqm', 'params', 'PWCCT_PM6_MOPAC.csv'
        )
        # Try alternate path
        if not os.path.exists(filepath):
            filepath = '/Users/tgg/Github/pyseqm_ref/seqm/params/PWCCT_PM6_MOPAC.csv'

    try:
        with open(filepath) as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 4:
                    z1, z2 = int(parts[0]), int(parts[1])
                    chi, alpha = float(parts[2]), float(parts[3])
                    _PWCCT_CACHE[(z1, z2)] = (chi, alpha)
                    _PWCCT_CACHE[(z2, z1)] = (chi, alpha)
        _PWCCT_LOADED = True
    except FileNotFoundError:
        _PWCCT_LOADED = True  # Don't retry


def get_pwcct(z1: int, z2: int) -> Tuple[float, float]:
    """Get PWCCT (chi, alpha) for element pair."""
    _load_pwcct()
    return _PWCCT_CACHE.get((z1, z2), (0.0, 0.0))


def pm6_nuclear_repulsion(
    atoms: list[int],
    coords: np.ndarray,
    param_dict: dict,
) -> float:
    """PM6 nuclear repulsion energy with PWCCT.

    E_nuc = Σ_{A<B} [
        unpolcore
        + Z_A*Z_B * gam * (1 + 2*chi*exp(-alpha*(R_ang + 0.0003*R_ang^6)))
    ]

    For X-H pairs (X = C, N, O): different exponential form.
    """
    from .params import ANG_TO_BOHR

    EV = 27.21
    n_atoms = len(atoms)
    coords = np.asarray(coords, dtype=np.float64)
    E_nuc = 0.0

    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            pA = param_dict[atoms[i]]
            pB = param_dict[atoms[j]]
            R = np.linalg.norm(coords[i] - coords[j])
            R_bohr = R * ANG_TO_BOHR
            ZA = pA.n_valence
            ZB = pB.n_valence

            # (ss|ss) integral
            rho0A = 0.5 * EV / pA.gss if pA.gss > 0 else 0.0
            rho0B = 0.5 * EV / pB.gss if pB.gss > 0 else 0.0
            gam = EV / np.sqrt(R_bohr ** 2 + (rho0A + rho0B) ** 2)

            # Unpolarized core-core repulsion (1e-8 * ((Z_A^(1/3) + Z_B^(1/3)) / R)^12)
            atomic_num_A = float(atoms[i])
            atomic_num_B = float(atoms[j])
            unpolcore = 1e-8 * ((atomic_num_A ** (1.0/3) + atomic_num_B ** (1.0/3)) / R) ** 12

            # PWCCT correction
            chi, alp = get_pwcct(atoms[i], atoms[j])

            # Standard PM6 nuclear repulsion
            is_XH = ((atoms[i] in (6, 7, 8)) and atoms[j] == 1) or \
                     ((atoms[j] in (6, 7, 8)) and atoms[i] == 1)

            if is_XH:
                # C-H, N-H, O-H special form
                enuc = unpolcore + ZA * ZB * gam * (
                    1.0 + 2.0 * chi * np.exp(-alp * R ** 2)
                )
            else:
                # General form
                enuc = unpolcore + ZA * ZB * gam * (
                    1.0 + 2.0 * chi * np.exp(-alp * (R + 0.0003 * R ** 6))
                )

            # Gaussian corrections (same as AM1-style)
            t4 = ZA * ZB / R
            t5 = sum(pA.gauss_K[k] * np.exp(-pA.gauss_L[k] * (R - pA.gauss_M[k]) ** 2)
                     for k in range(4) if pA.gauss_K[k] != 0)
            t6 = sum(pB.gauss_K[k] * np.exp(-pB.gauss_L[k] * (R - pB.gauss_M[k]) ** 2)
                     for k in range(4) if pB.gauss_K[k] != 0)

            E_nuc += enuc + t4 * (t5 + t6)

    return E_nuc
