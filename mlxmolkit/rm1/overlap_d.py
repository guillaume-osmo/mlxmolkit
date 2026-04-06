"""
D-orbital Slater overlap integrals for PM6.

Extends overlap.py with jcall=431 (3rd row s - 1s), jcall=5 (3rd row - 2nd row),
and jcall=6 (3rd row - 3rd row) overlap formulas.

Port of PYSEQM diat_overlap_PM6_SP.py lines 184-338.

D-orbital overlaps: S311 (d-s sigma), S321 (d-p sigma), etc.
Uses the same A-integral / B-integral framework with higher indices.
"""
from __future__ import annotations

import numpy as np
from .params import ANG_TO_BOHR


def _aintgs_ext(alpha: float, n_max: int = 7) -> np.ndarray:
    """Extended A-integrals up to index n_max (need 7 for jcall=6)."""
    if abs(alpha) < 1e-10:
        return np.zeros(n_max)
    a = np.zeros(n_max)
    a[0] = np.exp(-alpha) / alpha
    for k in range(1, n_max):
        a[k] = a[0] + k * a[k - 1] / alpha
    return a


def _bintgs_ext(beta: float, n_max: int = 7) -> np.ndarray:
    """Extended B-integrals up to index n_max."""
    b = np.zeros(n_max)
    x = beta

    if abs(x) <= 1e-6:
        for k in range(0, n_max, 2):
            b[k] = 2.0 / (k + 1)
        return b

    if abs(x) <= 0.5:
        x2, x3, x4, x5, x6 = x*x, x*x*x, x**4, x**5, x**6
        # Even: b[k] = 2/(k+1) + x²/(k+3) + x⁴/(k+5)*... series
        b[0] = 2.0 + x2/3.0 + x4/60.0 + x6/2520.0
        b[2] = 2.0/3.0 + x2/5.0 + x4/84.0 + x6/3240.0
        if n_max > 4: b[4] = 2.0/5.0 + x2/7.0 + x4/108.0 + x6/3960.0
        if n_max > 6: b[6] = 2.0/7.0 + x2/9.0 + x4/132.0 + x6/4680.0
        # Odd
        b[1] = -2.0/3.0*x - x3/15.0 - x5/420.0
        b[3] = -2.0/5.0*x - x3/21.0 - x5/540.0
        if n_max > 5: b[5] = -2.0/7.0*x - x3/27.0 - x5/660.0
        return b

    x_c = np.clip(x, -500.0, 500.0)
    tx = np.exp(x_c) / x
    tmx = -np.exp(-x_c) / x
    b[0] = tx + tmx
    for k in range(1, n_max):
        sign = 1.0 if k % 2 == 0 else -1.0
        b[k] = sign * tx + tmx + k * b[k-1] / x
    return b


def overlap_d_local(
    zA_s: float, zA_p: float, zA_d: float,
    zB_s: float, zB_p: float, zB_d: float,
    R_bohr: float,
    qnA: int, qnB: int, qnD_A: int, qnD_B: int,
) -> dict:
    """Compute d-orbital overlap integrals in local frame.

    Returns dict with keys:
      'S_ds_sigma': d(σ)-s overlap
      'S_dp_sigma': d(σ)-p(σ) overlap
      'S_dp_pi': d(π)-p(π) overlap
      'S_dd_sigma': d(σ)-d(σ) overlap
      'S_dd_pi': d(π)-d(π) overlap
      'S_dd_delta': d(δ)-d(δ) overlap
    """
    result = {
        'S_ds_sigma': 0.0, 'S_dp_sigma': 0.0, 'S_dp_pi': 0.0,
        'S_dd_sigma': 0.0, 'S_dd_pi': 0.0, 'S_dd_delta': 0.0,
    }

    if R_bohr < 1e-10:
        return result

    R = R_bohr

    # d-s overlap (atom A has d, atom B has s) — jcall=431
    if qnD_A > 0 and zA_d > 0 and zB_s > 0:
        alpha = 0.5 * R * (zA_d + zB_s)
        beta = 0.5 * R * (zA_d - zB_s)
        A = _aintgs_ext(alpha, 6)
        B = _bintgs_ext(beta, 6)

        # S111 for jcall=431: (3d_σ | 1s)
        result['S_ds_sigma'] = (
            zB_s ** 1.5 * zA_d ** 3.5 * R ** 5
            * (A[4]*B[0] + 2*B[1]*A[3] - 2*A[1]*B[3] - B[4]*A[0])
            / (np.sqrt(10.0) * 24.0)
        )

    # d-p overlap (atom A has d, atom B has p) — jcall=5
    if qnD_A > 0 and zA_d > 0 and zB_p > 0:
        alpha_dp = 0.5 * R * (zA_d + zB_p)
        beta_dp = 0.5 * R * (zA_d - zB_p)
        A = _aintgs_ext(alpha_dp, 7)
        B = _bintgs_ext(beta_dp, 7)

        # S221 for jcall=5: (3d_σ | 2p_σ)
        result['S_dp_sigma'] = (
            zB_p ** 2.5 * zA_d ** 3.5 * R ** 6
            * ((A[3]*B[0] - A[5]*B[2])
               + (A[2]*B[1] - A[4]*B[3])
               - (A[1]*B[2] - A[3]*B[4])
               - (A[0]*B[3] - A[2]*B[5]))
            / (16.0 * np.sqrt(30.0))
        )

        # S222 for jcall=5: (3d_π | 2p_π)
        result['S_dp_pi'] = (
            zB_p ** 2.5 * zA_d ** 3.5 * R ** 6
            * ((A[5]-A[3])*(B[0]-B[2])
               + (A[4]-A[2])*(B[1]-B[3])
               - (A[3]-A[1])*(B[2]-B[4])
               - (A[2]-A[0])*(B[3]-B[5]))
            / (32.0 * np.sqrt(30.0))
        )

    # d-d overlap — jcall=6
    if qnD_A > 0 and qnD_B > 0 and zA_d > 0 and zB_d > 0:
        alpha_dd = 0.5 * R * (zA_d + zB_d)
        beta_dd = 0.5 * R * (zA_d - zB_d)
        A = _aintgs_ext(alpha_dd, 7)
        B = _bintgs_ext(beta_dd, 7)

        # S111 for jcall=6: (3d_σ | 3d_σ) — "ss overlap" in d-d basis
        result['S_dd_sigma'] = (
            (zA_d * zB_d) ** 3.5 * R ** 7
            * (A[6]*B[0] - 3*B[2]*A[4] + 3*A[2]*B[4] - A[0]*B[6])
            / 1440.0
        )

        # S221 for jcall=6: (3d_π | 3d_π)
        result['S_dd_pi'] = (
            zB_d ** 3.5 * zA_d ** 3.5 * R ** 7
            * ((A[4]*B[0] - A[6]*B[2])
               - 2*(A[2]*B[2] - A[4]*B[4])
               + (A[0]*B[4] - A[2]*B[6]))
            / 480.0
        )

        # S222 for jcall=6: (3d_δ | 3d_δ)
        result['S_dd_delta'] = (
            zB_d ** 3.5 * zA_d ** 3.5 * R ** 7
            * ((A[6]-A[4])*(B[0]-B[2])
               - 2*(A[4]-A[2])*(B[2]-B[4])
               + (A[2]-A[0])*(B[4]-B[6]))
            / 960.0
        )

    return result


def overlap_d_molecular_frame(
    pA, pB,
    coordA: np.ndarray, coordB: np.ndarray,
) -> np.ndarray:
    """Compute full overlap matrix including d-orbitals in molecular frame.

    Returns (nA, nB) where nA, nB can be 1, 4, or 9.
    Order: [s, px, py, pz, dz², dxz, dyz, dx²-y², dxy]

    For sp×sp blocks: delegates to standard overlap_molecular_frame.
    For d-orbital blocks: uses d-orbital Slater overlaps + rotation.
    """
    from .overlap import overlap_molecular_frame as sp_overlap
    from .rotation import _rotation_matrix

    nA = pA.n_basis
    nB = pB.n_basis

    R_vec = coordB - coordA
    R = np.linalg.norm(R_vec)

    # Start with sp overlap
    nA_sp = min(nA, 4)
    nB_sp = min(nB, 4)
    S_sp = sp_overlap(pA, pB, coordA, coordB)

    S = np.zeros((nA, nB))
    S[:S_sp.shape[0], :S_sp.shape[1]] = S_sp

    if nA <= 4 and nB <= 4:
        return S

    if R < 1e-10:
        for k in range(min(nA, nB)):
            S[k, k] = 1.0
        return S

    R_bohr = R * ANG_TO_BOHR
    v = R_vec / R
    rot = _rotation_matrix(v)  # 3×3 rotation
    r0, r1, r2 = rot[0], rot[1], rot[2]  # sigma, pi_x, pi_y

    # Quantum numbers
    qnA = 1 if pA.Z <= 2 else 2
    qnB = 1 if pB.Z <= 2 else 2
    qnD_A = 3 if getattr(pA, 'has_d', False) and pA.zeta_d > 0 else 0
    qnD_B = 3 if getattr(pB, 'has_d', False) and pB.zeta_d > 0 else 0

    # Compute d-orbital local frame overlaps
    d_ovlp = overlap_d_local(
        pA.zeta_s, pA.zeta_p, getattr(pA, 'zeta_d', 0.0),
        pB.zeta_s, pB.zeta_p, getattr(pB, 'zeta_d', 0.0),
        R_bohr, qnA, qnB, qnD_A, qnD_B,
    )

    # d-s block: d on A, s on B (or vice versa)
    if nA == 9 and nB >= 1:
        Sds = d_ovlp['S_ds_sigma']
        if abs(Sds) > 1e-15:
            # d_σ overlaps s: project dz² along bond → c0 = r0
            # S[dz², s] = Sds * r0[k] (rotation to molecular frame)
            # In the 5 d-orbital basis: dz²=4, dxz=5, dyz=6, dx²-y²=7, dxy=8
            # Only dσ (dz² in local frame) has nonzero overlap with s
            for k in range(3):
                # dz² → molecular frame via d-orbital rotation matrix
                # Simplified: dσ projects as r0[k]² * √(3) factor
                pass  # TODO: proper d-orbital rotation (5×5 Wigner matrix)
            # For now: approximate d-s overlap using bond direction
            S[4, 0] = Sds  # dz² - s along bond

    # d-p block: d on A, p on B
    if nA == 9 and nB >= 4:
        Sdp_s = d_ovlp['S_dp_sigma']
        Sdp_p = d_ovlp['S_dp_pi']
        # dσ-pσ and dπ-pπ overlaps
        if abs(Sdp_s) > 1e-15 or abs(Sdp_p) > 1e-15:
            S[4, 1] = Sdp_s * r0[0]  # dz²-px (approximate)
            S[4, 2] = Sdp_s * r0[1]
            S[4, 3] = Sdp_s * r0[2]

    # d-d block
    if nA == 9 and nB == 9:
        Sdd_s = d_ovlp['S_dd_sigma']
        Sdd_p = d_ovlp['S_dd_pi']
        Sdd_d = d_ovlp['S_dd_delta']
        # Diagonal: each d-orbital overlaps itself
        S[4, 4] = Sdd_s    # dz²-dz² (sigma)
        S[5, 5] = Sdd_p    # dxz-dxz (pi)
        S[6, 6] = Sdd_p    # dyz-dyz (pi)
        S[7, 7] = Sdd_d    # dx²-y² (delta)
        S[8, 8] = Sdd_d    # dxy-dxy (delta)

    # Symmetrize for B-A block
    if nB == 9 and nA < 9:
        S_BA = overlap_d_molecular_frame(pB, pA, coordB, coordA)
        return S_BA.T

    return S
