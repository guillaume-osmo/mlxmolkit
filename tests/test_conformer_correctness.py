"""Correctness regressions for generate_conformers_nk (2026-10-09 audit).

Every returned conformer is checked by an oracle independent of the pipeline's
own gate: RDKit's AssignStereochemistryFrom3D on a copy of
AddHs(MolFromSmiles(smiles)) carrying the conformer's coordinates, compared
element by element with the stereo the input defines (nitrogen centres
excluded), plus the largest bond angle at any non-sp atom.
"""
from __future__ import annotations

import numpy as np
import pytest

Chem = pytest.importorskip("rdkit.Chem")
from rdkit.Chem import AllChem, rdMolTransforms  # noqa: E402
from rdkit.Geometry import Point3D  # noqa: E402

from mlxmolkit.conformer_gate import EmbedFailureCause  # noqa: E402
from mlxmolkit.conformer_pipeline_v2 import generate_conformers_nk  # noqa: E402

# Molecules the audit found returned with wrong stereo or a linear angle,
# with the number of conformers that makes a pre-fix failure near certain.
CAMPHOR = "C[C@@]12CC[C@@H](C1(C)C)CC2=O"                 # 22/100 ent-camphor
SULFOXIDE = "Cc1ccc([S@@](C)=O)cc1"                       # 3/100 inverted in ETK
ALLOOCIMENE = "C/C=C(C)/C=C/C=C(C)C"                      # 4/100 wrong E/Z, 10/100 linear C-C=C
NORBORNYL = "CNC(=O)Cn1c(=O)n(C2CCN([C@H]3C[C@@H]4CC[C@H]3C4)CC2)c2ccccc21"  # 21/100
COLCHICINE = "COc1ccc2c(c(=O)c1)[C@@H](NC(C)=O)CCc1cc(OC)c(OC)c(OC)c1-2"   # 24/100 C-O-C 180 deg
PARTIAL = "C[C@H](O)C(C)CC"                               # one centre specified, one not
REGRESSION = {CAMPHOR: 40, SULFOXIDE: 100, ALLOOCIMENE: 100, NORBORNYL: 30,
              COLCHICINE: 30, PARTIAL: 20}

_TET = (Chem.ChiralType.CHI_TETRAHEDRAL_CW, Chem.ChiralType.CHI_TETRAHEDRAL_CCW)


class _Oracle:
    def __init__(self, smi: str):
        ref = Chem.MolFromSmiles(smi)
        self.mol = Chem.AddHs(ref)
        self.centres = [(a.GetIdx(), a.GetChiralTag()) for a in self.mol.GetAtoms()
                        if a.GetChiralTag() in _TET and a.GetAtomicNum() != 7]
        ct = Chem.Mol(ref)
        Chem.SetBondStereoFromDirections(ct)
        self.bonds = []
        for b in ct.GetBonds():
            if b.GetBondType() == Chem.BondType.DOUBLE and b.GetStereo() in (
                    Chem.BondStereo.STEREOCIS, Chem.BondStereo.STEREOTRANS):
                s0, s1 = b.GetStereoAtoms()
                self.bonds.append((s0, b.GetBeginAtomIdx(), b.GetEndAtomIdx(), s1,
                                   b.GetStereo() == Chem.BondStereo.STEREOCIS))
        self.angles = [(i, a.GetIdx(), k) for a in self.mol.GetAtoms()
                       if a.GetHybridization() != Chem.HybridizationType.SP
                       for n, i in enumerate(x.GetIdx() for x in a.GetNeighbors())
                       for k in [x.GetIdx() for x in a.GetNeighbors()][n + 1:]]

    def with_coords(self, xyz) -> Chem.Mol:
        m = Chem.Mol(self.mol)
        conf = Chem.Conformer(m.GetNumAtoms())
        for i, p in enumerate(np.asarray(xyz, float)):
            conf.SetAtomPosition(i, Point3D(*map(float, p)))
        m.RemoveAllConformers()
        m.AddConformer(conf, assignId=True)
        return m

    def wrong_stereo(self, xyz) -> list:
        m = self.with_coords(xyz)
        probe = Chem.Mol(m)
        Chem.AssignStereochemistryFrom3D(probe)
        bad = [f"atom {i}" for i, tag in self.centres
               if probe.GetAtomWithIdx(i).GetChiralTag() != tag]
        conf = m.GetConformer()
        bad += [f"bond {b}={c}" for a, b, c, d, cis in self.bonds
                if (abs(rdMolTransforms.GetDihedralDeg(conf, a, b, c, d)) < 90.0) != cis]
        return bad

    def max_angle(self, xyz) -> float:
        x = np.asarray(xyz, float)
        t = np.asarray(self.angles)
        u, v = x[t[:, 0]] - x[t[:, 1]], x[t[:, 2]] - x[t[:, 1]]
        c = np.sum(u * v, 1) / np.linalg.norm(u, axis=1) / np.linalg.norm(v, axis=1)
        return float(np.degrees(np.arccos(np.clip(c, -1, 1))).max())


def test_oracle_detects_an_inverted_conformer():
    """The oracle must be able to fail: a mirror image is caught."""
    res = generate_conformers_nk([CAMPHOR], 1, variant="ETKDGv3")
    x = res.molecules[0].positions_3d[0]
    oracle = _Oracle(CAMPHOR)
    assert oracle.wrong_stereo(x) == []
    assert len(oracle.wrong_stereo(x * np.array([-1.0, 1.0, 1.0]))) == 2


@pytest.mark.parametrize("run_mmff", [False, True], ids=["etk", "mmff"])
def test_no_returned_conformer_has_wrong_stereo_or_a_linear_angle(run_mmff):
    smiles = list(REGRESSION)
    res = generate_conformers_nk(smiles, [REGRESSION[s] for s in smiles],
                                 variant="ETKDGv3", run_mmff=run_mmff, seed=42)
    for smi, mol in zip(smiles, res.molecules):
        oracle = _Oracle(smi)
        assert len(mol.positions_3d) == REGRESSION[smi], (smi, mol.n_failed_by_cause)
        for j, x in enumerate(mol.positions_3d):
            assert oracle.wrong_stereo(x) == [], (smi, j)
            assert oracle.max_angle(x) <= 175.0, (smi, j, oracle.max_angle(x))
        assert all(mol.stereo_ok)
        assert all(c is None for c in mol.fail_cause)
        assert mol.n_attempted >= len(mol.positions_3d) + sum(mol.n_failed_by_cause.values())


def test_output_does_not_depend_on_chunking():
    smiles = [CAMPHOR, SULFOXIDE, ALLOOCIMENE, "CC(=O)Oc1ccccc1C(=O)O", PARTIAL,
              "OB(O)c1ccccc1"]
    runs = [generate_conformers_nk(smiles, 5, variant="ETKDGv3", run_mmff=True,
                                   seed=11, max_confs_per_batch=b) for b in (7, 400)]
    assert runs[0].n_batches > runs[1].n_batches
    for a, b in zip(runs[0].molecules, runs[1].molecules):
        assert len(a.positions_3d) == len(b.positions_3d)
        for p, q in zip(a.positions_3d, b.positions_3d):
            np.testing.assert_array_equal(p, q)
        assert a.energies == b.energies
        assert a.n_attempted == b.n_attempted


def test_seed_changes_the_output_and_fixes_it():
    a, b, c = (generate_conformers_nk([CAMPHOR], 2, variant="ETKDGv3", seed=s)
               for s in (1, 1, 2))
    np.testing.assert_array_equal(a.molecules[0].positions_3d[0], b.molecules[0].positions_3d[0])
    assert not np.array_equal(a.molecules[0].positions_3d[0], c.molecules[0].positions_3d[0])


def test_mmff_is_skipped_only_for_the_molecule_it_cannot_type():
    boronic, aspirin = "OB(O)c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O"
    res = generate_conformers_nk([boronic, aspirin], 3, variant="ETKDGv3",
                                 run_mmff=True, max_confs_per_batch=100)
    b, a = res.molecules
    assert not b.mmff_applied and b.mmff_error
    assert len(b.positions_3d) == 3          # still embedded, with ETK geometry
    assert a.mmff_applied and a.mmff_error is None
    oracle = _Oracle(aspirin)
    for x, e in zip(a.positions_3d, a.energies):
        m = oracle.with_coords(x)
        ff = AllChem.MMFFGetMoleculeForceField(m, AllChem.MMFFGetMoleculeProperties(m))
        assert abs(ff.CalcEnergy() - e) < 0.05      # the energy is aspirin's MMFF94 energy
        grad = np.linalg.norm(np.asarray(ff.CalcGrad()).reshape(-1, 3), axis=1)
        assert grad.max() < 0.5                      # at an MMFF minimum, not an ETK geometry


def test_partially_specified_molecule():
    res = generate_conformers_nk([PARTIAL], 10, variant="ETKDGv3", seed=3)
    mol = res.molecules[0]
    assert len(mol.positions_3d) == 10
    oracle = _Oracle(PARTIAL)
    assert [i for i, _ in oracle.centres] == [1]     # C3 is left unspecified
    for x in mol.positions_3d:
        assert oracle.wrong_stereo(x) == []


def test_impossible_stereo_returns_nothing_within_the_attempt_budget():
    # 7-azabicyclo[2.2.1]heptane with both bridgeheads specified the same way:
    # no geometry realises it (RDKit returns no conformer either).
    smi = "c1cncc(CN2[C@H]3CC[C@H]2CC3)c1"
    res = generate_conformers_nk([smi], 3, variant="ETKDGv3", max_rounds=8)
    mol = res.molecules[0]
    assert mol.positions_3d == []
    n_atoms = Chem.AddHs(Chem.MolFromSmiles(smi)).GetNumAtoms()
    assert 0 < mol.n_attempted <= 10 * n_atoms
    assert sum(mol.n_failed_by_cause.values()) == mol.n_attempted


def test_return_failed_adds_rejected_attempts_without_changing_accepted_ones():
    base = generate_conformers_nk([CAMPHOR], 6, variant="ETKDGv3", seed=5)
    dbg = generate_conformers_nk([CAMPHOR], 6, variant="ETKDGv3", seed=5, return_failed=True)
    m0, m1 = base.molecules[0], dbg.molecules[0]
    ok = [p for p, c in zip(m1.positions_3d, m1.fail_cause) if c is None]
    assert len(ok) == len(m0.positions_3d)
    for p, q in zip(ok, m0.positions_3d):
        np.testing.assert_array_equal(p, q)
    failed = [(c, s) for c, s in zip(m1.fail_cause, m1.fail_stage) if c is not None]
    assert len(failed) == sum(m1.n_failed_by_cause.values())
    assert all(isinstance(c, EmbedFailureCause) and s in ("dg", "etk", "mmff") for c, s in failed)


def test_three_atom_molecules_do_not_crash_the_batch():
    """Fewer atoms than the 4D embedding has coordinates: used to raise IndexError
    in the metric-matrix start and take every molecule in the call down with it."""
    smis = ["S=C=S", "O=[Si]=O", "[N-]=[N+]=O", "CC(=O)Oc1ccccc1C(=O)O"]
    res = generate_conformers_nk(smis, n_confs_per_mol=2, variant="srETKDGv3", seed=7)
    assert len(res.molecules) == len(smis)
    assert all(len(m.positions_3d) > 0 for m in res.molecules)
