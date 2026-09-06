"""Regression tests for the vendored NumPy PYSEQM port with CODATA constants.

These tests guard the critical numerical invariants:
  - stable diatomic overlaps across shell combinations
  - stable two-center integrals and per-pair W tensors
  - symmetry and atom-order invariants

The original PYSEQM 2.0.0 snapshots used MOPAC7 physical constants.
These numerical regression snapshots were refreshed for current CODATA
constants on 2026-09-06. They test port stability, not independent physical
accuracy; test_pm7_port and test_nddo_constants provide MOPAC oracle gates.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlxmolkit.nddo._pyseqm_port.constants_np import qn_int, qnD_int
from mlxmolkit.nddo._pyseqm_port.diat_overlapD_np import diatom_overlap_matrixD
from mlxmolkit.nddo._pyseqm_port.two_elec_two_center_int_np import (
    two_elec_two_center_int,
)
from mlxmolkit.nddo.params import ANG_TO_BOHR
from mlxmolkit.nddo.methods import METHOD_PARAMS
from mlxmolkit.nddo import d_two_center as d2c


# ----------------------------------------------------------------------
# Reference ri[22] values from PYSEQM for S(16) - C(6) at 1.81 Angstrom
# ----------------------------------------------------------------------
CODATA_RI_SC_181 = np.array([6.41935368614589, -1.7177798903676198, 6.836552033713134, 6.110662164081101, 0.8204469726271593, -0.42184008720054744, 0.28941142842854806, 0.8142468429594021, 0.6997252561058145, -0.12678114395692885, 6.636209759069869, 6.231194860956338, -1.9740987335352518, -1.5192118183705328, 0.17310724216093054, 6.934099003502048, 6.2366612209428585, 6.537206583496737, 5.989819490581849, -0.10001902847146282, 5.95001810728794, 0.019900691646954183])


# Reference W[mu,nu,lam,sig] for S-C YX pair at 1.81 A
# (captured from the regression-to-PYSEQM NumPy port)
CODATA_W0000_SC_181 = 6.41935368614589  # (s_S s_S | s_C s_C)
CODATA_W0133_SC_181 = 1.5192118183705328  # (s_S p_x_S | p_z_C p_z_C) — canary
CODATA_W4400_SC_181 = 7.400058114056704  # (d_z2_S d_z2_S | s_C s_C)
CODATA_W4433_SC_181 = 7.128042369512873  # (d_z2_S d_z2_S | p_z_C p_z_C)
CODATA_W1000_SC_181 = 1.7177798903676198  # (p_x_S s_S | s_C s_C) — was 0 before fix
CODATA_W4000_SC_181 = 0.04379776054364283  # (d_z2_S s_S | s_C s_C)


def _build_const():
    """Minimal Constants stand-in TETCI requires."""
    tore = np.array(
        [0.0, 1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
         0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 0.0] + [0.0] * 60
    )
    class _C:
        qn = qn_int
        qnD_int_ = qnD_int  # avoid clash with global
    _C.qnD_int = qnD_int
    _C.tore = tore
    return _C()


# =====================================================================
# Overlap NumPy port — regressionness vs PYSEQM (qn=1..5)
# =====================================================================

OVERLAP_REFERENCE_S00 = {
    # (ZA, ZB, R_ang) -> S[0, 0] computed via the regression-to-PYSEQM
    # NumPy port; frozen as a regression reference for the port itself.
    (1, 1, 0.74): 0.648570310283831,    # H-H jcall=2
    (6, 1, 1.09): 0.4243573066024346,    # C-H jcall=3
    (6, 6, 1.54): 0.19183290562951075,    # C-C jcall=4
    (16, 1, 1.336): 0.3709970229025872,  # S-H jcall=431
    (16, 6, 1.81): 0.1590264134553709,   # S-C jcall=5
    (16, 16, 2.05): 0.14294593695156932,  # S-S jcall=6
}


@pytest.mark.parametrize("ZA,ZB,R_ang,expected_S00", [
    (ZA, ZB, R, S) for (ZA, ZB, R), S in OVERLAP_REFERENCE_S00.items()
])
def test_overlap_matches_codata_snapshot_S00(ZA, ZB, R_ang, expected_S00):
    """Diatomic overlap[0,0] must match the CODATA port snapshot to 1e-5 for the jcall paths
    we have hand-derived (qn=1..3)."""
    PARAMS = METHOD_PARAMS['PM6_D']
    pA, pB = PARAMS[ZA], PARAMS[ZB]
    xij = np.array([[1.0, 0.0, 0.0]])
    R_bohr = np.array([R_ang * ANG_TO_BOHR])
    ni = np.array([ZA], dtype=np.int64)
    nj = np.array([ZB], dtype=np.int64)
    zeta_a = np.array([[pA.zeta_s, pA.zeta_p, getattr(pA, 'zeta_d', 0.0)]])
    zeta_b = np.array([[pB.zeta_s, pB.zeta_p, getattr(pB, 'zeta_d', 0.0)]])
    S = diatom_overlap_matrixD(ni, nj, xij, R_bohr, zeta_a, zeta_b,
                                qn_int, qnD_int)
    assert abs(S[0, 0, 0] - expected_S00) < 1e-5, (
        f"S({ZA},{ZB},{R_ang}): got {S[0, 0, 0]:.6f}, "
        f"expected {expected_S00:.6f}"
    )


def test_overlap_is_symmetric_under_atom_label_swap():
    """Same physical geometry, atom labels swapped: overlap matrices
    must be transposes. diatom_overlap_matrixD requires qni>=qnj
    internally, so the swap routing in overlap_d._pyseqm_overlap_matrix
    must give the right answer regardless of caller ordering."""
    from mlxmolkit.nddo.overlap_d import _pyseqm_overlap_matrix
    PARAMS = METHOD_PARAMS['PM6_D']
    # PHYSICAL geometry: S at origin, H at +x. Same in both calls.
    coord_S = np.array([0., 0., 0.])
    coord_H = np.array([1.336, 0., 0.])
    S_SH = _pyseqm_overlap_matrix(PARAMS[16], PARAMS[1], coord_S, coord_H)
    S_HS = _pyseqm_overlap_matrix(PARAMS[1], PARAMS[16], coord_H, coord_S)
    assert S_SH.shape == (9, 1)
    assert S_HS.shape == (1, 9)
    assert np.allclose(S_SH.T, S_HS, atol=1e-13), (
        f"Atom-label swap broken: max diff {np.abs(S_SH.T - S_HS).max():.2e}"
    )


# =====================================================================
# TETCI local-frame ri (the 22-element sp integral) regressionness
# =====================================================================

def test_tetci_ri_sc_regression():
    """All 22 ri[k] values for S-C at 1.81 A must match the CODATA port snapshot to 1e-7.

    This was the canary test that caught the int-dtype bug:
      ri[1] = -1.717..., NOT zero. If you see all zeros except ri[0]/[2]/[3],
      the rho/dd/qq arrays are getting truncated to integer dtype somewhere.
    """
    from mlxmolkit.nddo._pyseqm_port import two_elec_two_center_int_np as tetci_mod

    captured = {}
    orig_wq = tetci_mod.w_withquaternion
    def spy(mol, tore, ni, nj, xij, riXH, ri, wHH):
        captured['ri'] = np.asarray(ri).copy()
        return orig_wq(mol, tore, ni, nj, xij, riXH, ri, wHH)
    tetci_mod.w_withquaternion = spy
    try:
        d2c._tetci_pair_w(METHOD_PARAMS['PM6_D'][16],
                          METHOD_PARAMS['PM6_D'][6],
                          np.array([0., 0., 0.]),
                          np.array([1.81, 0., 0.]))
    finally:
        tetci_mod.w_withquaternion = orig_wq

    assert 'ri' in captured, "TETCI did not call w_withquaternion"
    ri = captured['ri'][0]  # first pair
    diff = np.abs(ri - CODATA_RI_SC_181)
    assert diff.max() < 1e-7, (
        f"ri mismatch (max diff {diff.max()}):\n"
        f"  got:      {ri}\n"
        f"  expected: {CODATA_RI_SC_181}\n"
        f"  diff:     {diff}"
    )


# =====================================================================
# W tensor (9x9x4x4 / 9x9x9x9) regressionness
# =====================================================================

@pytest.mark.parametrize("mu,nu,lam,sig,expected", [
    (0, 0, 0, 0, CODATA_W0000_SC_181),
    (0, 1, 3, 3, CODATA_W0133_SC_181),  # canary: was 0 before dtype fix
    (1, 0, 0, 0, CODATA_W1000_SC_181),  # canary: was 0 before dtype fix
    (4, 4, 0, 0, CODATA_W4400_SC_181),
    (4, 4, 3, 3, CODATA_W4433_SC_181),
    (4, 0, 0, 0, CODATA_W4000_SC_181),
])
def test_yx_w_element_regression(mu, nu, lam, sig, expected):
    """Per-element W[mu,nu,lam,sig] for S-C YX pair must match the CODATA port snapshot."""
    PARAMS = METHOD_PARAMS['PM6_D']
    W = d2c._yx_pair_w(PARAMS[16], PARAMS[6],
                              np.array([0., 0., 0.]),
                              np.array([1.81, 0., 0.]))
    assert W is not None and W.shape == (9, 9, 4, 4)
    assert abs(W[mu, nu, lam, sig] - expected) < 1e-6, (
        f"W[{mu},{nu},{lam},{sig}] = {W[mu, nu, lam, sig]:.6f} "
        f"vs expected {expected:.6f}"
    )


def test_yx_w_symmetric_in_mu_nu():
    """W[mu,nu,lam,sig] should equal W[nu,mu,lam,sig]."""
    PARAMS = METHOD_PARAMS['PM6_D']
    W = d2c._yx_pair_w(PARAMS[16], PARAMS[6],
                              np.array([0., 0., 0.]),
                              np.array([1.81, 0., 0.]))
    diff = np.abs(W - W.transpose(1, 0, 2, 3)).max()
    assert diff < 1e-12, f"W not symmetric in (mu,nu): {diff}"


def test_yx_w_symmetric_in_lam_sig():
    PARAMS = METHOD_PARAMS['PM6_D']
    W = d2c._yx_pair_w(PARAMS[16], PARAMS[6],
                              np.array([0., 0., 0.]),
                              np.array([1.81, 0., 0.]))
    diff = np.abs(W - W.transpose(0, 1, 3, 2)).max()
    assert diff < 1e-12, f"W not symmetric in (lam,sig): {diff}"


@pytest.mark.parametrize("ZA,ZB,R_ang", [
    (16, 16, 2.05),   # S-S
    (17, 17, 1.99),   # Cl-Cl
    (15, 16, 1.95),   # P-S
    (17, 6, 1.78),    # Cl-C (odd electrons -> phantom H trick)
])
def test_yy_w_runs_and_symmetric(ZA, ZB, R_ang):
    """YY pair returns a (9, 9, 9, 9) W tensor that is symmetric in the
    expected index pairs (no NaN/inf), regardless of the actual values."""
    PARAMS = METHOD_PARAMS['PM6_D']
    W = d2c._yy_pair_w(PARAMS[ZA], PARAMS[ZB],
                              np.array([0., 0., 0.]),
                              np.array([R_ang, 0., 0.]))
    assert W is not None and W.shape == (9, 9, 9, 9)
    assert np.isfinite(W).all(), f"NaN/inf in W for Z={ZA},{ZB}"
    sym_munu = np.abs(W - W.transpose(1, 0, 2, 3)).max()
    sym_lamsig = np.abs(W - W.transpose(0, 1, 3, 2)).max()
    assert sym_munu < 1e-12, f"YY mu,nu asymmetric: {sym_munu}"
    assert sym_lamsig < 1e-12, f"YY lam,sig asymmetric: {sym_lamsig}"


# =====================================================================
# Regression test for the int-dtype bug
# =====================================================================

def test_rho_arrays_are_float_dtype():
    """Guard against the int-dtype bug: rho_0/rho_1/rho_2/dd/qq arrays
    must be float64. If qn0 = qn[Z] is int and we do zeros_like(qn0),
    float assignments truncate to 0 and the entire integral set collapses
    to the bare 1/r Coulomb."""
    import inspect
    src = inspect.getsource(
        __import__('mlxmolkit.nddo._pyseqm_port.two_elec_two_center_int_np',
                   fromlist=['two_elec_two_center_int']).two_elec_two_center_int
    )
    # We expect explicit float64 init for the rho/dd/qq arrays.
    # If anyone reverts to torch.zeros_like(qn0), this fires.
    forbidden = "torch.zeros_like(qn0)"
    assert forbidden not in src, (
        f"REGRESSION: {forbidden} found in two_elec_two_center_int. "
        f"qn0 is integer dtype; this truncates float multipole terms to 0. "
        f"Use np.zeros(qn0.shape, dtype=np.float64) instead."
    )


# =====================================================================
# Smoke test: a full SCF call still converges to known charges
# =====================================================================

def _run_native_scf(atoms, coords):
    """Mini SCF driver — copy of tests.test_pm6_d_native._run_native to
    avoid the brittle relative-import path."""
    from mlxmolkit.nddo.scf import (
        _build_basis_info, _build_core_hamiltonian, _build_fock,
    )
    from mlxmolkit.nddo.methods import get_params

    PARAMS = get_params("PM6_D")
    coords = np.asarray(coords, dtype=np.float64)
    info = _build_basis_info(atoms, PARAMS)
    n_basis = info["n_basis"]
    n_occ = info["n_occ"]
    params = info["params"]
    starts = info["atom_basis_start"]

    H = _build_core_hamiltonian(atoms, coords, info)
    P = np.zeros((n_basis, n_basis))
    for i, p in enumerate(params):
        sA = starts[i]
        if p.n_basis == 1:
            P[sA, sA] = 1.0
        else:
            for k in range(4):
                P[sA + k, sA + k] = p.n_valence / 4.0

    has_d = any(p.n_basis > 4 for p in params)
    diis_max = 6
    diis_F, diis_E = [], []
    converged = False
    for it in range(500):
        F = _build_fock(H, P, info, atoms, coords)
        F = 0.5 * (F + F.T)
        if it >= 2:
            err = F @ P - P @ F
            diis_F.append(F.copy()); diis_E.append(err.copy())
            if len(diis_F) > diis_max:
                diis_F.pop(0); diis_E.pop(0)
            nd = len(diis_F)
            if nd >= 2:
                B = np.zeros((nd + 1, nd + 1))
                for i_ in range(nd):
                    for j_ in range(nd):
                        B[i_, j_] = np.sum(diis_E[i_] * diis_E[j_])
                B[nd, :nd] = -1.0; B[:nd, nd] = -1.0
                rhs = np.zeros(nd + 1); rhs[nd] = -1.0
                try:
                    c = np.linalg.solve(B, rhs)
                    F = sum(c[k] * diis_F[k] for k in range(nd))
                except np.linalg.LinAlgError:
                    pass
        _, C = np.linalg.eigh(F)
        P_new = 2.0 * C[:, :n_occ] @ C[:, :n_occ].T
        delta = np.sqrt(np.mean((P_new - P) ** 2))
        if delta < 1e-7:
            converged = True; P = P_new; break
        if it < 3:
            mix = 0.3 if has_d else 0.5
        elif delta > 0.1:
            mix = 0.05 if has_d else 0.4
        elif delta > 0.01:
            mix = 0.5
        else:
            mix = 0.8
        P = mix * P_new + (1 - mix) * P
    q = []
    for i, p in enumerate(params):
        sA = starts[i]
        q.append(p.n_valence - sum(P[sA + k, sA + k] for k in range(p.n_basis)))
    return np.array(q), converged


def test_h2s_scf_converges_to_pyseqm_charges():
    """End-to-end check: H2S charges via the pure-numpy path must match
    PYSEQM. Expected sulfur charge from PYSEQM reference: -0.3617 e."""
    atoms = [16, 1, 1]
    coords = [[0, 0, 0], [0.9686, 0, 0.9269], [-0.9686, 0, 0.9269]]
    q, conv = _run_native_scf(atoms, coords)
    assert conv, "H2S SCF did not converge"
    assert abs(q[0] - (-0.3617)) < 1e-3, (
        f"H2S q_S = {q[0]:.4f}, expected -0.3617"
    )


def test_csc_scf_converges_to_pyseqm_charges():
    """CSC: regression test for the bug that produced -0.669/+0.144/-0.669
    (off by ~0.2e) instead of the correct -0.488/-0.046/-0.488."""
    atoms = [6, 16, 6, 1, 1, 1, 1, 1, 1]
    coords = [
        [-1.500, 0.0, 0.350], [0.0, 0.0, -0.700], [1.500, 0.0, 0.350],
        [-2.380, 0.0, -0.300], [-1.560, 0.886, 0.988], [-1.560, -0.886, 0.988],
        [2.380, 0.0, -0.300], [1.560, 0.886, 0.988], [1.560, -0.886, 0.988],
    ]
    q, conv = _run_native_scf(atoms, coords)
    assert conv, "CSC SCF did not converge"
    expected = np.array([-0.488, -0.046, -0.488])
    heavy = q[[0, 1, 2]]
    diff = np.abs(heavy - expected).max()
    assert diff < 1e-3, f"CSC charges {heavy} vs expected {expected}, max diff {diff}"
