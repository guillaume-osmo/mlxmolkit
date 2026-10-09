"""Unit tests for mlxmolkit.conformer_gate on RDKit-made and deliberately broken geometry."""
from __future__ import annotations

import numpy as np
import pytest

Chem = pytest.importorskip("rdkit.Chem")
from rdkit.Chem import rdDistGeom  # noqa: E402

from mlxmolkit import conformer_gate as cg  # noqa: E402
from mlxmolkit.dg_extract import extract_dg_params, get_bounds_matrix  # noqa: E402

CAMPHOR = "C[C@@]12CC[C@@H](C1(C)C)CC2=O"
SULFOXIDE = "Cc1ccc([S@@](C)=O)cc1"
ALLOOCIMENE = "C/C=C(C)/C=C/C=C(C)C"
ANISOLE = "COc1ccccc1"


def _prepare(smi):
    mol = Chem.AddHs(Chem.MolFromSmiles(smi))
    bm = get_bounds_matrix(mol, use_macrocycle14config=True)
    extract_dg_params(mol, bm)  # same in-place stereo perception as the pipeline
    return mol, cg.build_gate_params(mol, bm)


def _rdkit_conformers(mol, n=4, seed=7):
    p = rdDistGeom.ETKDGv3()
    p.randomSeed = seed
    ids = list(rdDistGeom.EmbedMultipleConfs(mol, n, p))
    assert ids
    return [mol.GetConformer(i).GetPositions() for i in ids]


def _final(params, coords, **kw):
    packed = cg.pack_gate_params([params])
    n = params.n_atoms
    starts = np.arange(len(coords) + 1) * n
    return cg.check_final(np.concatenate(coords).ravel(), starts,
                          np.zeros(len(coords), dtype=int), packed, **kw)


@pytest.mark.parametrize("smi", [CAMPHOR, SULFOXIDE, ALLOOCIMENE, ANISOLE,
                                 "CC(=O)Oc1ccccc1C(=O)O", "C[C@H](O)C(C)CC"])
def test_rdkit_conformers_pass(smi):
    mol, params = _prepare(smi)
    res = _final(params, _rdkit_conformers(mol))
    assert res.passed.all(), [cg.cause_name(c) for c in res.cause]
    assert res.stereo_ok.all()


@pytest.mark.parametrize("smi", [CAMPHOR, SULFOXIDE])
def test_mirror_image_fails_chirality(smi):
    mol, params = _prepare(smi)
    mirrored = [x * np.array([-1.0, 1.0, 1.0]) for x in _rdkit_conformers(mol)]
    res = _final(params, mirrored)
    assert (res.cause == cg.EmbedFailureCause.CHECK_CHIRAL_CENTERS2).all()
    assert not res.stereo_ok.any()


def test_selection_mirrors_rdkit():
    _, camphor = _prepare(CAMPHOR)
    assert len(camphor.chiral_idx) == 2           # both tagged bridgeheads
    assert set(camphor.chiral_lb) <= {5.0, -100.0}   # 4-coordinate bounds [5, 100]
    _, sulfoxide = _prepare(SULFOXIDE)
    (row,) = sulfoxide.chiral_idx
    assert row[0] == row[4]                       # 3-coordinate S: centre is the 4th vertex
    assert abs(sulfoxide.chiral_lb[0]) in (2.0, 100.0)  # github #5883 lower bound 2.0
    _, alloocimene = _prepare(ALLOOCIMENE)
    assert len(alloocimene.dbstereo_idx) == 2
    _, decalin = _prepare("C1CCC2CCCCC2C1")       # untagged fusion carbons are tested
    assert len(decalin.tetra_idx) == 2


def test_flipped_double_bond_fails_stereo():
    mol, params = _prepare(ALLOOCIMENE)
    x = _rdkit_conformers(mol, n=1)[0].copy()
    # Reflect the substituents of the first stereo double bond's begin atom
    # through the plane containing the bond and perpendicular to the sp2 plane:
    # that swaps cis and trans and nothing else.
    ref, b, e, _ = params.dbstereo_idx[0]
    side = [n.GetIdx() for n in mol.GetAtomWithIdx(int(b)).GetNeighbors() if n.GetIdx() != e]
    axis = x[e] - x[b]
    axis /= np.linalg.norm(axis)
    normal = np.cross(axis, np.cross(x[side[0]] - x[b], axis))
    normal /= np.linalg.norm(normal)
    moving = set()
    stack = list(side)
    while stack:  # every atom on the begin side of the double bond
        a = stack.pop()
        if a in moving or a in (b, e):
            continue
        moving.add(a)
        stack += [n.GetIdx() for n in mol.GetAtomWithIdx(a).GetNeighbors()]
    for a in moving:
        d = x[a] - x[b]
        x[a] = x[a] - 2 * np.dot(d, normal) * normal
    res = _final(params, [x])
    assert res.cause[0] == cg.EmbedFailureCause.BAD_DOUBLE_BOND_STEREO
    assert not res.stereo_ok[0]


def test_linear_ether_angle_fails():
    mol, params = _prepare(ANISOLE)
    x = _rdkit_conformers(mol, n=1)[0].copy()
    c_me, o, c_ar = 0, 1, 2
    u = x[o] - x[c_ar]
    u /= np.linalg.norm(u)
    shift = x[o] + u * np.linalg.norm(x[c_me] - x[o]) - x[c_me]
    for a in [c_me] + [n.GetIdx() for n in mol.GetAtomWithIdx(c_me).GetNeighbors() if n.GetIdx() != o]:
        x[a] += shift  # methyl moved rigidly onto the Ar-O axis: C-O-C = 180 deg
    res = _final(params, [x])
    assert res.cause[0] == cg.EmbedFailureCause.LINEAR_ANGLE
    assert res.stereo_ok[0]


def test_short_bond_fails_and_mmff_tolerance_is_wider():
    mol, params = _prepare("CCCC")
    x = _rdkit_conformers(mol, n=1)[0].copy()
    c1, c2 = 1, 2
    d = x[c2] - x[c1]
    lb = params.bond_lb[[i for i, (a, b) in enumerate(params.bond_idx) if {a, b} == {c1, c2}][0]]
    target = lb - 0.18  # 0.18 A short: beyond the ETK tolerance, within the MMFF one
    moving = [c2, 3] + [n.GetIdx() for a in (c2, 3) for n in mol.GetAtomWithIdx(a).GetNeighbors()
                        if n.GetAtomicNum() == 1]
    delta = d / np.linalg.norm(d) * (target - np.linalg.norm(d))
    for a in set(moving):
        x[a] += delta
    assert _final(params, [x]).cause[0] == cg.EmbedFailureCause.BOND_LENGTH
    assert _final(params, [x], after_mmff=True).passed[0]


def test_non_finite_is_reported_first():
    mol, params = _prepare(CAMPHOR)
    x = _rdkit_conformers(mol, n=1)[0].copy()
    x[3, 1] = np.nan
    assert _final(params, [x]).cause[0] == cg.EmbedFailureCause.NON_FINITE


def test_dg_stage_energy_and_chirality():
    mol, params = _prepare(CAMPHOR)
    confs = _rdkit_conformers(mol, n=3)
    packed = cg.pack_gate_params([params])
    n = params.n_atoms
    x4 = np.concatenate([np.hstack([c, np.zeros((n, 1))]) for c in
                         (confs[0], confs[1], confs[2] * np.array([-1.0, 1.0, 1.0]))])
    energy = np.array([0.0, 0.06 * n, 0.0])  # 0.06/atom is above RDKit's 0.05
    cause = cg.check_dg_stage(x4.ravel(), np.arange(4) * n, np.zeros(3, dtype=int), packed, energy)
    assert cause[0] == cg.PASS
    assert cause[1] == cg.EmbedFailureCause.FIRST_MINIMIZATION
    assert cause[2] == cg.EmbedFailureCause.CHECK_CHIRAL_CENTERS


def test_mixed_molecules_are_routed_by_conf_mol():
    mol_a, pa = _prepare(CAMPHOR)
    mol_b, pb = _prepare(SULFOXIDE)
    a, b = _rdkit_conformers(mol_a, n=2), _rdkit_conformers(mol_b, n=2)
    mirror = np.array([-1.0, 1.0, 1.0])
    coords = [b[0], a[0] * mirror, b[1] * mirror, a[1]]
    conf_mol = np.array([1, 0, 1, 0])
    sizes = np.array([pa.n_atoms, pb.n_atoms])[conf_mol]
    starts = np.concatenate([[0], np.cumsum(sizes)])
    res = cg.check_final(np.concatenate(coords).ravel(), starts, conf_mol,
                         cg.pack_gate_params([pa, pb]))
    assert res.passed.tolist() == [True, False, False, True]
    assert res.stereo_ok.tolist() == [True, False, False, True]
