# Copyright (c) 2026 Guillaume
# SPDX-License-Identifier: MIT

"""The g-xTB D4Srev dispersion, as the released binary computes it.

This replaces the GFN2-D4 energy fallback in :mod:`dispersion_d4srev` for the
one thing the SCF actually needs: the ATOM POTENTIAL ``dE/dq``, which the
binary's ``tblite_disp_d4`` container adds to ``pot%vat`` on every iteration
and which the fallback could not produce at all.

Every formula here was read out of the released binary and then checked
against it, not against the D4 literature -- the g-xTB model is D4Srev with a
SCREENED damping, and it differs from published D4 in ways that matter:

* the damping's ``a1`` is exactly ZERO, so the Becke-Johnson offset does not
  depend on the critical radius at all; ``r0`` enters only through an erf
  screen (``dftd4_damping_screened::get_2b_damp``);
* ``s8`` is zero -- the live cache's ``dispmat8`` comes back identically zero
  on every molecule tried -- so there is no ``r**-8`` term to add;
* the reference weights are gated by a TANH zeta rather than D4's exponential
  one (``dftd4_model_utils::zeta_tanh``).

Units: coordinates arrive in ANGSTROM and are converted here; ``rcov`` and
``r4r2`` come out of the model already in Bohr / atomic units.
"""

from __future__ import annotations

import os

import numpy as np
from scipy.special import erf

ANG_TO_BOHR = 1.8897261246204404

#: and s6/s8 as measured from the live cache.
D4S_A1 = 0.0
D4S_A2 = 8.5302496868
D4S_ALP = 0.5395219242
D4S_BETA = 0.654
D4S_S6 = 1.0
D4S_S8 = 0.0
#: Three-body (Axilrod-Teller-Muto) scale. Read from the live parameter block
#: of the reference implementation, and confirmed independently: with s9 = 1
#: the three-body energy falls short of the reference's by exactly this
#: factor on every one of 13 molecules, constant to 2e-11.
D4S_S9 = 1.826410691
#: Real-space cutoff of the three-body sum, in Bohr. The calculator hands the
#: constructor 40, and the constructor overwrites it with 25; the sum reads the
#: overwritten value. At 40 the largest molecules come out 1e-3 high, at 25
#: they agree to 4e-14.
D4S_CUTOFF3 = 25.0
#: `mctc_ncoord_erf`'s steepness.
D4S_KCN = 7.5
#: `sqrt(tiny)`, the guard `weight_references` uses on the normalisation.
_SQRT_TINY = 1.4916681462400413e-154

_DATA = os.path.join(os.path.dirname(__file__), "data")
_TABLES = None


def _tables():
    """The D4Srev reference tables, read once out of the released model."""
    global _TABLES
    if _TABLES is None:
        d = np.load(os.path.join(_DATA, "d4srev_tables.npz"))
        sh = lambda k: tuple(int(x) for x in d[k + "_shape"])
        _TABLES = {
            "elements": np.asarray(d["elements"], dtype=int),
            "ref": np.asarray(d["d4_ref"], dtype=int),
            "ngw": d["d4_ngw"].reshape(sh("d4_ngw"), order="F").astype(int),
            "refcn": d["d4_refcn"].reshape(sh("d4_refcn"), order="F"),
            "refq": d["d4_refq"].reshape(sh("d4_refq"), order="F"),
            "c6ref": d["d4_c6ref"].reshape(sh("d4_c6ref"), order="F"),
            "wf": d["d4_wf"].reshape(sh("d4_wf"), order="F"),
            "r4r2": np.asarray(d["d4_r4r2"], dtype=float),
            "rcov": np.asarray(d["d4_rcov"], dtype=float),
            "tanh": np.load(os.path.join(_DATA, "d4srev_tanh_params.npy")),
        }
    return _TABLES


def d4srev_coordination_number(atomic_numbers, coords_ang) -> np.ndarray:
    """`mctc_ncoord_erf` over the MODEL's own covalent radii, in Bohr."""
    t = _tables()
    Z = np.asarray(atomic_numbers, dtype=int)
    sp = _species_index(Z, t)
    rc = t["rcov"][sp]
    xyz = np.asarray(coords_ang, dtype=np.float64) * ANG_TO_BOHR
    r = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=2)
    rr = rc[:, None] + rc[None, :]
    np.fill_diagonal(r, 1.0)                       # keep the erf finite
    w = 0.5 * (1.0 + erf(-D4S_KCN * (r - rr) / rr))
    np.fill_diagonal(w, 0.0)
    return w.sum(axis=1)


def _species_index(Z, t):
    """Map each atom onto its row in the frozen per-element tables."""
    lut = {int(z): k for k, z in enumerate(t["elements"])}
    missing = sorted({int(z) for z in Z} - lut.keys())
    if missing:
        raise ValueError(
            "d4srev tables do not cover element(s) %s; the shipped tables "
            "stop below that atomic number" % missing)
    return np.array([lut[int(z)] for z in Z], dtype=int)


def d4srev_weights(atomic_numbers, cn, qat):
    """``gw`` and its charge derivative -- `d4srev::weight_references`.

    Both come back shaped ``(mref, nat, nat)``, indexed ``(iref, iat, jat)``
    exactly as the binary's ``gw`` cube is.
    """
    t = _tables()
    Z = np.asarray(atomic_numbers, dtype=int)
    sp = _species_index(Z, t)
    nat = len(Z)
    ref, ngw, refcn, refq, wf = (t["ref"], t["ngw"], t["refcn"], t["refq"],
                                 t["wf"])
    mref = int(ref[sp].max())
    cn = np.asarray(cn, dtype=float)
    q = np.asarray(qat, dtype=float)

    gw = np.zeros((mref, nat, nat))
    dgw = np.zeros((mref, nat, nat))
    for i in range(nat):
        zi = sp[i]
        ta, tb, tc, td = t["tanh"][:, int(Z[i]) - 1]
        nr = int(ref[zi])
        # `weight_cn` depends on jat only through wf(izp, jzp), so the whole
        # gaussian block is one (nr, nat) evaluation rather than nat of them.
        kk = np.arange(1, int(ngw[:nr, zi].max()) + 1)[:, None, None]
        w = wf[zi, sp][None, None, :]                        # (1,1,nat)
        dd = (cn[i] - refcn[:nr, zi])[None, :, None]         # (1,nr,1)
        block = np.exp(-((kk * w) * dd * dd))
        mask = (kk <= ngw[:nr, zi][None, :, None])
        g = np.where(mask, block, 0.0).sum(axis=0)           # (nr, nat)

        nrm = g.sum(axis=0)
        good = np.abs(nrm) > _SQRT_TINY
        inv = np.where(good, 1.0 / np.where(good, nrm, 1.0), 0.0)
        gwk = g * inv[None, :]
        # The exceptional branch: all the weight on the largest reference CN.
        bad = (~good)[None, :] | ~np.isfinite(gwk) | (np.abs(gwk) > 1e300)
        top = np.abs(refcn[:nr, zi].max() - refcn[:nr, zi]) < 1e-12
        gwk = np.where(bad, np.where(top[:, None], 1.0, 0.0), gwk)

        den = ta + np.tanh(td + refq[:nr, zi] * tc) * tb
        zeta = (ta + np.tanh(td + q[i] * tc) * tb) / den
        dzeta = ((tc * tb) / np.cosh(td + q[i] * tc) ** 2) / den
        gw[:nr, i, :] = zeta[:, None] * gwk
        dgw[:nr, i, :] = dzeta[:, None] * gwk
    return gw, dgw


def d4srev_weights_dcn(atomic_numbers, cn, qat):
    """``dgw/dcn`` -- the CN derivative of `d4srev::weight_references`.

    Only ``gwk`` depends on cn; the tanh zeta gate depends on q alone.  With

        gsum_r = sum_k exp(-(k*wf) * (cn_i - refcn_r)**2)
        gwk_r  = gsum_r / sum_s gsum_s

    the quotient rule gives

        dgwk_r/dcn = norm * (dgsum_r - gwk_r * sum_s dgsum_s)

    ⚠️ The EXCEPTIONAL branch (the normalisation underflowed, or the weight
    came out non-finite, and all the weight was forced onto the reference with
    the largest CN) is a CONSTANT in cn, so its derivative is zero -- not the
    quotient rule's value.  Getting that wrong is silent on well-behaved
    molecules and wrong exactly where the guard exists to help.
    """
    t = _tables()
    Z = np.asarray(atomic_numbers, dtype=int)
    sp = _species_index(Z, t)
    nat = len(Z)
    ref, ngw, refcn, refq, wf = (t["ref"], t["ngw"], t["refcn"], t["refq"],
                                 t["wf"])
    mref = int(ref[sp].max())
    cn = np.asarray(cn, dtype=float)
    q = np.asarray(qat, dtype=float)
    dgw = np.zeros((mref, nat, nat))
    for i in range(nat):
        zi = sp[i]
        ta, tb, tc, td = t["tanh"][:, int(Z[i]) - 1]
        nr = int(ref[zi])
        kk = np.arange(1, int(ngw[:nr, zi].max()) + 1)[:, None, None]
        w = wf[zi, sp][None, None, :]
        dd = (cn[i] - refcn[:nr, zi])[None, :, None]
        block = np.exp(-((kk * w) * dd * dd))
        mask = (kk <= ngw[:nr, zi][None, :, None])
        g = np.where(mask, block, 0.0).sum(axis=0)
        dg = np.where(mask, block * (-2.0 * (kk * w) * dd), 0.0).sum(axis=0)
        nrm = g.sum(axis=0)
        good = np.abs(nrm) > _SQRT_TINY
        inv = np.where(good, 1.0 / np.where(good, nrm, 1.0), 0.0)
        gwk = g * inv[None, :]
        bad = (~good)[None, :] | ~np.isfinite(gwk) | (np.abs(gwk) > 1e300)
        dgwk = inv[None, :] * (dg - gwk * dg.sum(axis=0)[None, :])
        dgwk = np.where(bad, 0.0, dgwk)
        den = ta + np.tanh(td + refq[:nr, zi] * tc) * tb
        zeta = (ta + np.tanh(td + q[i] * tc) * tb) / den
        dgw[:nr, i, :] = zeta[:, None] * dgwk
    return dgw


def d4srev_c6(atomic_numbers, gw, dgw):
    """``c6`` and ``dc6/dq`` -- `d4srev::get_atomic_c6` and `get_2b_derivs`.

    ``dc6dq(i, j)`` is the derivative with respect to atom ``i``'s charge only,
    which is why it is NOT symmetric.
    """
    t = _tables()
    Z = np.asarray(atomic_numbers, dtype=int)
    sp = _species_index(Z, t)
    nat = len(Z)
    ref, c6ref = t["ref"], t["c6ref"]
    c6 = np.zeros((nat, nat))
    dc6dq = np.zeros((nat, nat))
    for i in range(nat):
        ni = int(ref[sp[i]])
        for j in range(nat):
            nj = int(ref[sp[j]])
            a = c6ref[:ni, :nj, sp[i], sp[j]]
            gj = gw[:nj, j, i]
            c6[i, j] = gw[:ni, i, j] @ a @ gj
            dc6dq[i, j] = dgw[:ni, i, j] @ a @ gj
    return c6, dc6dq


def d4srev_dispersion_matrices(atomic_numbers, coords_ang):
    """``dispmat6`` and ``dispmat8`` -- `tblite_disp_d4::get_dispersion_matrix`.

    ``mat8`` is all zeros while ``s8`` is zero; it is returned anyway so the
    caller's contraction reads the same as the binary's.
    """
    t = _tables()
    Z = np.asarray(atomic_numbers, dtype=int)
    sp = _species_index(Z, t)
    r42 = t["r4r2"][sp]
    xyz = np.asarray(coords_ang, dtype=np.float64) * ANG_TO_BOHR
    r = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=2)
    r0 = np.sqrt(3.0 * np.outer(r42, r42))
    np.fill_diagonal(r, 1.0)
    den = r + 0.5 * (1.0 + erf(-D4S_ALP * (r - D4S_BETA * r0))) * (
        D4S_A2 + D4S_A1 * r0)
    d2 = den * den
    d6 = d2 * d2 * d2
    mat6 = -D4S_S6 / d6
    mat8 = -D4S_S8 / (d6 * d2)
    np.fill_diagonal(mat6, 0.0)
    np.fill_diagonal(mat8, 0.0)
    return mat6, mat8


def d4srev_energy(atomic_numbers, coords_ang, qat, cn=None) -> np.ndarray:
    """Per-atom D4Srev dispersion energy -- `tblite_disp_d4::get_energy`.

        E_i = 0.5 * sum_j ( c6_ij*mat6_ij + c8_ij*mat8_ij )
        c8_ij = 3 * r4r2_i * r4r2_j * c6_ij

    Inputs: `coords_ang` in ANGSTROM, `qat` in e.  Returns Hartree per atom;
    the total is its sum.
    """
    t = _tables()
    Z = np.asarray(atomic_numbers, dtype=int)
    sp = _species_index(Z, t)
    if cn is None:
        cn = d4srev_coordination_number(Z, coords_ang)
    gw, dgw = d4srev_weights(Z, cn, qat)
    c6, _dq = d4srev_c6(Z, gw, dgw)
    mat6, mat8 = d4srev_dispersion_matrices(Z, coords_ang)
    r42 = t["r4r2"][sp]
    c8 = 3.0 * np.outer(r42, r42) * c6
    return 0.5 * (c6 * mat6 + c8 * mat8).sum(axis=1)


def d4srev_gradient(atomic_numbers, coords_ang, qat, cn=None) -> np.ndarray:
    """dE/dR for the D4Srev dispersion, in HARTREE PER BOHR, shaped (nat, 3).

    Two pieces, and the second is the one an energy-only port never needs:

      * the explicit r-dependence of the damped matrix,
            d(den)/dr = 1 - (alp*a2/sqrt(pi)) * exp(-u**2)
        with mat6 = -s6/den**6 so d(mat6)/dr = 6*s6/den**7 * d(den)/dr;
      * the COORDINATION-NUMBER CHAIN, because c6 depends on the geometry
        through cn: dE/dcn_i = 0.5 * sum_j dc6dcn(i,j) * mat6(i,j), contracted
        with the D4 model's own erf CN derivative (k = 7.5 over ITS rcov, not
        the calculator's).

    ⚠️ `coords_ang` is in ANGSTROM; everything inside is BOHR and the result
    is per BOHR, which is the binary's convention.  With s8 = 0 the r**-8 half
    contributes nothing and is left out rather than multiplied by zero.
    """
    t = _tables()
    Z = np.asarray(atomic_numbers, dtype=int)
    sp = _species_index(Z, t)
    nat = len(Z)
    if cn is None:
        cn = d4srev_coordination_number(Z, coords_ang)
    gw, _dgq = d4srev_weights(Z, cn, qat)
    dgc = d4srev_weights_dcn(Z, cn, qat)
    c6, _dq = d4srev_c6(Z, gw, _dgq)
    _c6b, dc6dcn = d4srev_c6(Z, gw, dgc)
    mat6, _mat8 = d4srev_dispersion_matrices(Z, coords_ang)

    xyz = np.asarray(coords_ang, dtype=np.float64) * ANG_TO_BOHR
    r42 = t["r4r2"][sp]
    rc = t["rcov"][sp]
    d = xyz[:, None, :] - xyz[None, :, :]
    r = np.linalg.norm(d, axis=2)
    np.fill_diagonal(r, 1.0)
    r0 = np.sqrt(3.0 * np.outer(r42, r42))
    u = -D4S_ALP * (r - D4S_BETA * r0)
    den = r + 0.5 * (1.0 + erf(u)) * (D4S_A2 + D4S_A1 * r0)
    dden = 1.0 - (D4S_ALP * D4S_A2 / np.sqrt(np.pi)) * np.exp(-u * u)
    dmat6 = 6.0 * D4S_S6 / den ** 7 * dden
    np.fill_diagonal(dmat6, 0.0)
    pref = 0.5 * c6 * dmat6 / r
    grad = (np.einsum("ij,ijk->ik", pref, d)
            - np.einsum("ij,ijk->jk", pref, d))

    dEdcn = 0.5 * (dc6dcn * mat6).sum(axis=1)
    r0c = rc[:, None] + rc[None, :]
    uc = -D4S_KCN * (r - r0c) / r0c
    dcn = (1.0 / np.sqrt(np.pi)) * np.exp(-uc * uc) * (-D4S_KCN / r0c)
    np.fill_diagonal(dcn, 0.0)
    prefc = (dEdcn[:, None] + dEdcn[None, :]) * dcn / r
    grad = grad + (np.einsum("ij,ijk->ik", prefc, d)
                   - np.einsum("ij,ijk->jk", prefc, d))
    return grad


def d4srev_atom_potential(atomic_numbers, coords_ang, qat, cn=None):
    """``pot%vat`` from `tblite_disp_d4::get_potential`, in Hartree per e.

    Inputs: ``coords_ang`` in ANGSTROM, ``qat`` the atomic charges in e.  If
    ``cn`` is omitted the model's own erf coordination number is built here --
    it is NOT the calculator's CN and the two are not interchangeable.
    """
    t = _tables()
    Z = np.asarray(atomic_numbers, dtype=int)
    sp = _species_index(Z, t)
    if cn is None:
        cn = d4srev_coordination_number(Z, coords_ang)
    gw, dgw = d4srev_weights(Z, cn, qat)
    _c6, dc6dq = d4srev_c6(Z, gw, dgw)
    mat6, mat8 = d4srev_dispersion_matrices(Z, coords_ang)
    r42 = t["r4r2"][sp]
    return ((mat6 + 3.0 * np.outer(r42, r42) * mat8) * dc6dq).sum(axis=1)
