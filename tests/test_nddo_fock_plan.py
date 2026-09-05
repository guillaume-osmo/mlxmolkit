"""The Fock plan: fused d-pair tensors, batched sp corners, and the plan-based build.

`_fock_plan` hoists everything geometry-only out of the SCF loop. The one claim
that is not obvious by inspection is the fused d-pair tensor: that
`d_two_center_fock` plus the sp routine equals ONE contraction against
`d_pair_effective_w`. It is an algebraic identity (both are linear in the
tensor and the routine subtracts exactly the sp corner), and it is checked
here on random densities for every pair kind so a change to either routine
that breaks it fails a test rather than an S-bearing energy.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlxmolkit.nddo import scf
from mlxmolkit.nddo.d_two_center import d_pair_effective_w, d_two_center_fock
from mlxmolkit.nddo.methods import get_params
from mlxmolkit.nddo.rotation import rotate_integrals_to_molecular_frame
from mlxmolkit.nddo.rotation_batch import rotate_pairs

P6 = get_params("PM6")
RNG = np.random.default_rng(7)

# (Z_A, Z_B): one pair of each kind the fused tensor covers
KINDS = {"YY": (16, 16), "YX": (16, 6), "XY": (6, 16), "YH": (16, 1), "HY": (1, 16)}
COORDS = (np.array([0.0, 0.0, 0.0]), np.array([1.31, 0.62, -0.44]))


def _contract(Weff, P, sA, sB, nA, nB, n_basis):
    """The plain NDDO pair contraction -- what _build_fock does per group."""
    F = np.zeros((n_basis, n_basis))
    P_AA = P[sA:sA + nA, sA:sA + nA]
    P_BB = P[sB:sB + nB, sB:sB + nB]
    P_AB = P[sA:sA + nA, sB:sB + nB]
    F[sA:sA + nA, sA:sA + nA] += np.einsum('abcd,cd->ab', Weff, P_BB)
    F[sB:sB + nB, sB:sB + nB] += np.einsum('abcd,ab->cd', Weff, P_AA)
    K = -0.5 * np.einsum('abcd,bd->ac', Weff, P_AB)
    F[sA:sA + nA, sB:sB + nB] += K
    F[sB:sB + nB, sA:sA + nA] += K.T
    return F


@pytest.mark.parametrize("kind", list(KINDS))
def test_fused_tensor_equals_sp_routine_plus_d_routine(kind):
    zA, zB = KINDS[kind]
    pA, pB = P6[zA], P6[zB]
    rA, rB = COORDS
    nA, nB = pA.n_basis, pB.n_basis
    n_basis = nA + nB
    sA, sB = 0, nA
    w, _, _ = rotate_integrals_to_molecular_frame(pA, pB, rA, rB)
    Weff = d_pair_effective_w(pA, pB, rA, rB, w)
    assert Weff is not None and Weff.shape == (nA, nA, nB, nB)
    for _ in range(3):
        M = RNG.standard_normal((n_basis, n_basis))
        P = M + M.T
        ref = scf._pair_fock_twocentre(np.zeros((n_basis, n_basis)), P, pA, pB, sA, sB, rA, rB, w=w)
        # _pair_fock_twocentre already calls d_two_center_fock for a d pair
        got = _contract(Weff, P, sA, sB, nA, nB, n_basis)
        scale = np.abs(ref).max()
        assert np.abs(got - ref).max() < 1e-12 * scale, kind


def test_the_sp_corner_of_a_d_pair_is_the_ordinary_batched_rotation():
    """Both the scalar and the batched rotation clamp a 9-function atom to its
    sp block; the plan relies on it to rotate every pair in one call."""
    rA, rB = COORDS
    for zA, zB in ((16, 6), (6, 16), (16, 1), (1, 16), (16, 16)):
        pA, pB = P6[zA], P6[zB]
        w_scalar, _, _ = rotate_integrals_to_molecular_frame(pA, pB, rA, rB)
        w_batch = rotate_pairs([(pA, pB)], [(rA, rB)])[0]
        assert np.abs(w_batch - w_scalar).max() < 1e-12 * max(1.0, np.abs(w_scalar).max())


def _reference_fock(H, P, info, atoms, coords):
    """The pre-plan build: one-centre formulas atom by atom, then every pair
    through the scalar pair routine. This is the specification the batched
    build is measured against."""
    params = info['params']
    starts = info['atom_basis_start']
    F = H.copy()
    for i, p in enumerate(params):
        s = starts[i]
        n_sp = min(p.n_basis, 4)
        Pss = P[s, s]
        if p.n_basis == 1:
            F[s, s] += Pss * p.gss * 0.5
            continue
        Ppp = sum(P[s + k, s + k] for k in range(1, 4))
        F[s, s] += Pss * p.gss * 0.5 + Ppp * (p.gsp - 0.5 * p.hsp)
        f1 = p.gsp - 0.5 * p.hsp
        f2 = 1.5 * p.hsp - 0.5 * p.gsp
        fd = 1.25 * p.gp2 - 0.25 * p.gpp
        foff = 0.75 * p.gpp - 1.25 * p.gp2
        for k in range(1, 4):
            F[s + k, s + k] += Pss * f1 + P[s + k, s + k] * p.gpp * 0.5 + (Ppp - P[s + k, s + k]) * fd
            F[s, s + k] += P[s, s + k] * f2
            F[s + k, s] += P[s + k, s] * f2
            for l in range(k + 1, 4):
                F[s + k, s + l] += P[s + k, s + l] * foff
                F[s + l, s + k] += P[s + l, s + k] * foff
        if p.n_basis == 9:
            from mlxmolkit.nddo.fock_d import fock_d_one_center
            F = fock_d_one_center(F, P, scf._one_centre_d_w(p), s, n_basis=9)
    n = len(atoms)
    for i in range(n):
        for j in range(i + 1, n):
            F = scf._pair_fock_twocentre(F, P, params[i], params[j], starts[i], starts[j],
                                         coords[i], coords[j])
    return F


@pytest.mark.parametrize("mol", ["CH3SH", "H2S+CH4", "ethanol"])
def test_build_fock_matches_the_scalar_reference(mol):
    if mol == "CH3SH":
        atoms = [6, 16, 1, 1, 1, 1]
        X = np.array([[0.0, 0.0, 0.0], [1.82, 0.0, 0.0], [-0.36, 1.03, 0.0],
                      [-0.36, -0.51, 0.89], [-0.36, -0.51, -0.89], [2.15, 1.27, 0.0]])
    elif mol == "H2S+CH4":       # two sulfur-free and sulfur-bearing fragments: YH, YX and XY pairs
        atoms = [16, 1, 1, 6, 1, 1, 1, 1]
        X = np.array([[0.0, 0.0, 0.0], [0.96, 0.93, 0.0], [-0.96, 0.93, 0.0],
                      [3.5, 0.0, 0.0], [4.13, 0.63, 0.63], [4.13, -0.63, -0.63],
                      [2.87, 0.63, -0.63], [2.87, -0.63, 0.63]])
    else:
        atoms = [6, 6, 8, 1, 1, 1, 1, 1, 1]
        X = np.array([[0.0, 0.0, 0.0], [1.52, 0.0, 0.0], [2.0, 1.33, 0.0], [-0.39, 1.02, 0.0],
                      [-0.39, -0.51, 0.88], [-0.39, -0.51, -0.88], [1.9, -0.52, 0.88],
                      [1.9, -0.52, -0.88], [2.96, 1.35, 0.0]])
    info = scf._build_basis_info(atoms, P6)
    H = scf._build_core_hamiltonian(atoms, X, info)
    n = info['n_basis']
    for _ in range(2):
        M = RNG.standard_normal((n, n))
        P = M + M.T
        ref = _reference_fock(H, P, info, atoms, X)
        got = scf._build_fock(H, P, info, atoms, X)
        assert np.abs(got - ref).max() < 1e-11 * np.abs(ref).max()
    # and the plan is reusable: a second density through the same plan
    plan = scf._fock_plan(info, atoms, X)
    got2 = scf._build_fock(H, P, info, atoms, X, plan=plan)
    assert np.array_equal(got2, scf._build_fock(H, P, info, atoms, X))


def test_precompute_pair_w_covers_every_pair():
    atoms = [16, 6, 1, 1]
    X = np.array([[0.0, 0.0, 0.0], [1.8, 0.0, 0.0], [-0.5, 1.2, 0.0], [2.3, 0.9, 0.5]])
    info = scf._build_basis_info(atoms, P6)
    pw = scf.precompute_pair_w(atoms, X, info)
    assert set(pw) == {(i, j) for i in range(4) for j in range(i + 1, 4)}
    w, _, _ = rotate_integrals_to_molecular_frame(P6[16], P6[6], X[0], X[1])
    assert np.abs(pw[(0, 1)] - w).max() < 1e-12 * np.abs(w).max()
