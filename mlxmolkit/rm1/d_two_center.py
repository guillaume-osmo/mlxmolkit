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

    For YY pairs (both have d): uses full 2025 riYY integrals.
    For YX/XY pairs: uses monopole Coulomb approximation.
    """
    from .yy_integrals import compute_yy_integrals

    R = np.linalg.norm(coordB - coordA)
    R_bohr = R * ANG_TO_BOHR
    nA_sp = min(pA.n_basis, 4)
    nB_sp = min(pB.n_basis, 4)

    # YY case: both atoms have d-orbitals → full 2025 integrals
    if pA.n_basis == 9 and pB.n_basis == 9:
        da_A, qa_A, rho0A, rho1A, rho2A = _compute_multipole_params(pA)
        da_B, qa_B, rho0B, rho1B, rho2B = _compute_multipole_params(pB)
        dA = compute_d_charge_separations(pA)
        dB = compute_d_charge_separations(pB)

        riYY = compute_yy_integrals(
            R_bohr, da_A, da_B, qa_A, qa_B,
            dA['dp'], dB['dp'], dA['ds'], dB['ds'],
            dA['dorbdorb'], dB['dorbdorb'],
            rho0A, rho0B, rho1A, rho1B, rho2A, rho2B,
            dA['rho3'], dB['rho3'], dA['rho4'], dB['rho4'],
            dA['rho5'], dB['rho5'], dA['rho6'], dB['rho6'],
        )

        # The 2025 riYY are indexed as 45×45 packed (9-orbital lower triangle)
        # Pack: (0,0)=0, (1,0)=1, (1,1)=2, ..., (8,8)=44
        # riYY[(i*(i+1)/2+j) * 45 + (k*(k+1)/2+l)] = (ij|kl) where i>=j, k>=l

        # Apply to Fock: Coulomb + Exchange for the full 9×9 basis
        for i_bra in range(45):
            # Unpack (i_bra) → (mu, nu) where mu >= nu
            mu_off = 0
            for m in range(9):
                if m * (m + 1) // 2 + 0 <= i_bra <= m * (m + 1) // 2 + m:
                    mu_off = m
                    nu_off = i_bra - m * (m + 1) // 2
                    break

            mu = sA + mu_off
            nu = sA + nu_off

            for i_ket in range(45):
                wval = riYY[i_bra * 45 + i_ket]
                if abs(wval) < 1e-12:
                    continue

                # Unpack i_ket → (lam, sig) where lam >= sig
                lam_off = 0
                for m in range(9):
                    if m * (m + 1) // 2 + 0 <= i_ket <= m * (m + 1) // 2 + m:
                        lam_off = m
                        sig_off = i_ket - m * (m + 1) // 2
                        break

                lam = sB + lam_off
                sig = sB + sig_off

                # Coulomb
                F[mu, nu] += P[lam, sig] * wval
                if lam_off != sig_off:
                    F[mu, nu] += P[sig, lam] * wval
                if mu_off != nu_off:
                    F[nu, mu] += P[lam, sig] * wval
                    if lam_off != sig_off:
                        F[nu, mu] += P[sig, lam] * wval

                F[lam, sig] += P[mu, nu] * wval
                if mu_off != nu_off:
                    F[lam, sig] += P[nu, mu] * wval
                if lam_off != sig_off:
                    F[sig, lam] += P[mu, nu] * wval
                    if mu_off != nu_off:
                        F[sig, lam] += P[nu, mu] * wval

                # Exchange
                F[mu, lam] -= 0.5 * P[nu, sig] * wval
                F[lam, mu] -= 0.5 * P[sig, nu] * wval
        return F

    # YX/XY case: one has d, other doesn't → monopole Coulomb
    d_int = compute_d_two_center(pA, pB, R_bohr)

    if pA.n_basis == 9:
        PB_total = sum(P[sB+k, sB+k] for k in range(nB_sp))
        for k in range(5):
            F[sA+4+k, sA+4+k] += PB_total * d_int['dd_ss']
            F[sA+4+k, sB] -= 0.5 * P[sA+4+k, sB] * d_int['dd_ss']
            F[sB, sA+4+k] -= 0.5 * P[sB, sA+4+k] * d_int['dd_ss']

    if pB.n_basis == 9:
        PA_total = sum(P[sA+k, sA+k] for k in range(nA_sp))
        for k in range(5):
            F[sB+4+k, sB+4+k] += PA_total * d_int['ss_dd']
            F[sB+4+k, sA] -= 0.5 * P[sB+4+k, sA] * d_int['ss_dd']
            F[sA, sB+4+k] -= 0.5 * P[sA, sB+4+k] * d_int['ss_dd']

    return F
