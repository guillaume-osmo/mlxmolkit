"""
Extended two-center integrals for d-orbitals (PM6).

For atom pairs where one or both have d-orbitals, the 22-integral sp set
is extended with d-orbital multipole interactions using rho3-rho6.

The d-orbital two-center integrals follow the SAME Klopman-Ohno-Dewar
pattern as sp: 1/sqrt(R² + (rhoA + rhoB)²) but with d-orbital
charge separations (dp, ds, dd) and additive terms (rho3-6).

This module computes the d-orbital EXTENSION to the sp 4×4×4×4 w tensor,
producing an expanded w tensor that covers all 9×9 orbital interactions.
"""
from __future__ import annotations

import numpy as np
from .params import ANG_TO_BOHR
from .d_charge_sep import compute_d_charge_separations
from .two_center_integrals import _compute_multipole_params, EV


def compute_d_two_center(
    pA, pB,
    R_bohr: float,
) -> dict:
    """Compute d-orbital two-center Coulomb integrals.

    Returns dict with:
        'dd_ss': (d_A d_A | s_B s_B) — d-electrons on A repelled by s-charge on B
        'ss_dd': (s_A s_A | d_B d_B)
        'dd_dd': (d_A d_A | d_B d_B) — d-d repulsion
        'dd_pp': (d_A d_A | p_B p_B)
        'pp_dd': (p_A p_A | d_B d_B)
        'ds_ss': (d_A s_A | s_B s_B) — dipole d-s on A with s on B
        etc.

    All in eV. These enter the Fock matrix as additional Coulomb/exchange terms.
    """
    ev = EV
    ev1 = ev / 2.0
    ev2 = ev / 4.0

    r = R_bohr

    # sp multipole params
    da_sp, qa_sp, rho0A, rho1A, rho2A = _compute_multipole_params(pA)
    db_sp, qb_sp, rho0B, rho1B, rho2B = _compute_multipole_params(pB)

    # d-orbital params
    dA = compute_d_charge_separations(pA)
    dB = compute_d_charge_separations(pB)

    rho3A = dA['rho3']  # DD0 additive term
    rho4A = dA['rho4']  # DP additive term
    rho5A = dA['rho5']  # DS additive term
    rho6A = dA['rho6']  # DD additive term
    dpA = dA['dp']       # d-orbital dipole separation
    dsA = dA['ds']       # d-s charge separation

    rho3B = dB['rho3']
    rho4B = dB['rho4']
    rho5B = dB['rho5']
    rho6B = dB['rho6']
    dpB = dB['dp']
    dsB = dB['ds']

    result = {}

    # (dd|ss) — d-orbital monopole on A, s-monopole on B
    aee_ds = (rho3A + rho0B) ** 2
    result['dd_ss'] = ev / np.sqrt(r**2 + aee_ds) if rho3A > 0 else 0.0

    # (ss|dd)
    aee_sd = (rho0A + rho3B) ** 2
    result['ss_dd'] = ev / np.sqrt(r**2 + aee_sd) if rho3B > 0 else 0.0

    # (dd|dd) — d-d monopole
    aee_dd = (rho3A + rho3B) ** 2
    result['dd_dd'] = ev / np.sqrt(r**2 + aee_dd) if rho3A > 0 and rho3B > 0 else 0.0

    # (dd|pp) — d-monopole on A, p-multipole on B (sigma)
    aee_dp = (rho3A + rho2B) ** 2
    result['dd_pp_sigma'] = ev / np.sqrt(r**2 + aee_dp) if rho3A > 0 else 0.0

    # (pp|dd) — p-multipole on A, d-monopole on B
    aee_pd = (rho2A + rho3B) ** 2
    result['pp_dd_sigma'] = ev / np.sqrt(r**2 + aee_pd) if rho3B > 0 else 0.0

    # (dd|sp dipole) — d-monopole on A, sp-dipole on B
    if rho3A > 0 and rho1B > 0:
        ade = (rho3A + rho1B) ** 2
        result['dd_sp'] = -ev1/np.sqrt((r+db_sp)**2 + ade) + ev1/np.sqrt((r-db_sp)**2 + ade)
    else:
        result['dd_sp'] = 0.0

    # (sp dipole|dd) — sp-dipole on A, d-monopole on B
    if rho1A > 0 and rho3B > 0:
        aed = (rho1A + rho3B) ** 2
        result['sp_dd'] = -ev1/np.sqrt((r-da_sp)**2 + aed) + ev1/np.sqrt((r+da_sp)**2 + aed)
    else:
        result['sp_dd'] = 0.0

    # (dp dipole|ss) — d-p dipole on A, s-monopole on B
    if rho4A > 0:
        ade_dp = (rho4A + rho0B) ** 2
        result['dp_ss'] = -ev1/np.sqrt((r+dpA)**2 + ade_dp) + ev1/np.sqrt((r-dpA)**2 + ade_dp)
    else:
        result['dp_ss'] = 0.0

    # (ds quadrupole|ss) — d-s quadrupole on A, s-monopole on B
    if rho5A > 0:
        aqe_ds = (rho5A + rho0B) ** 2
        ev1d = ev1/np.sqrt(r**2 + aqe_ds)
        result['ds_ss'] = ev2/np.sqrt((r-dsA)**2 + aqe_ds) + ev2/np.sqrt((r+dsA)**2 + aqe_ds) - ev1d
    else:
        result['ds_ss'] = 0.0

    return result


def d_two_center_fock(
    F: np.ndarray,
    P: np.ndarray,
    pA, pB,
    sA: int, sB: int,
    coordA: np.ndarray, coordB: np.ndarray,
) -> np.ndarray:
    """Add d-orbital two-center contributions to Fock matrix.

    Uses proper rho3-rho6 based Coulomb integrals for d-orbital
    interactions with all orbital types on the other atom.
    """
    R = np.linalg.norm(coordB - coordA)
    R_bohr = R * ANG_TO_BOHR
    nA_sp = min(pA.n_basis, 4)
    nB_sp = min(pB.n_basis, 4)

    d_int = compute_d_two_center(pA, pB, R_bohr)

    # d on A ← density on B
    if pA.n_basis == 9:
        # Total density on B: s + p + d
        PBs = P[sB, sB]
        PBp = sum(P[sB+k, sB+k] for k in range(1, nB_sp))
        PBd = sum(P[sB+4+k, sB+4+k] for k in range(5)) if pB.n_basis == 9 else 0

        # Coulomb: d on A from s on B → (dd|ss)
        for k in range(5):
            F[sA+4+k, sA+4+k] += PBs * d_int['dd_ss']
            F[sA+4+k, sA+4+k] += PBp * d_int['dd_pp_sigma']
            if pB.n_basis == 9:
                F[sA+4+k, sA+4+k] += PBd * d_int['dd_dd']

        # Exchange: d on A with s on B
        for k in range(5):
            F[sA+4+k, sB] -= 0.5 * P[sA+4+k, sB] * d_int['dd_ss']
            F[sB, sA+4+k] -= 0.5 * P[sB, sA+4+k] * d_int['dd_ss']

    # d on B ← density on A
    if pB.n_basis == 9:
        PAs = P[sA, sA]
        PAp = sum(P[sA+k, sA+k] for k in range(1, nA_sp))
        PAd = sum(P[sA+4+k, sA+4+k] for k in range(5)) if pA.n_basis == 9 else 0

        for k in range(5):
            F[sB+4+k, sB+4+k] += PAs * d_int['ss_dd']
            F[sB+4+k, sB+4+k] += PAp * d_int['pp_dd_sigma']
            if pA.n_basis == 9:
                F[sB+4+k, sB+4+k] += PAd * d_int['dd_dd']

        for k in range(5):
            F[sB+4+k, sA] -= 0.5 * P[sB+4+k, sA] * d_int['ss_dd']
            F[sA, sB+4+k] -= 0.5 * P[sA, sB+4+k] * d_int['ss_dd']

    return F
