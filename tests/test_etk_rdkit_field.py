"""The ETK stage against RDKit's ETKDG 3D force field, and what it fixed.

The field is held term by term against tests/_etk_rdkit_oracle.py (RDKit's
construct3DForceField rebuilt from the 2026.03 source). The regressions are
the defects parity and RDKit's BFGS removed: aryl-ether C-O-C angles driven to
a 180 deg saddle, bent sp centres, and (with RDKit >= 2026.03 bounds) ring
double bonds flipped to the wrong E/Z.
"""
import warnings

import numpy as np
import pytest

from mlxmolkit import conformer_pipeline_v2 as cp
from mlxmolkit import etk_extract as ee
from mlxmolkit import etk_metal as em
from mlxmolkit import shared_batch as sb
from mlxmolkit.dg_extract import get_bounds_matrix

Chem = pytest.importorskip("rdkit.Chem")
rdDistGeom = pytest.importorskip("rdkit.Chem.rdDistGeom")
O = pytest.importorskip("tests._etk_rdkit_oracle")

ARYL_ETHER = "Cn1cc(-c2cnc(N)c(OCc3c(Cl)cccc3Cl)n2)cn1"
COLCHICINE = "COc1cc2c(c(OC)c1OC)-c1ccc(OC)c(=O)cc1[C@@H](NC(C)=O)CC2"
ALLOOCIMENE = "C/C=C(C)/C=C/C=C(C)C"
PARITY_SMILES = [
    ARYL_ETHER, COLCHICINE, ALLOOCIMENE,
    "N#Cc1cnc(Nc2ccccn2)s1",                      # nitrile: linear-centre angle term
    "C=C=CC(=O)OC",                               # allene + ester (impropers, SP2 O)
    "N[C@@H](C(=O)N[C@@H]1C(=O)N2C(C(=O)O)=C(Cl)CC[C@H]12)c1ccccc1",  # 4-ring
    "CC(C)=C1C/C=C(\\C)CC/C=C(\\C)CC1",          # 11-ring, stereo ring double bonds
    "C1CC1c1ccc2ccccc2c1",                        # 3-ring (1-3 on bonded pairs), fused rings
]
GROUPS = ("torsion", "improper", "1-2", "1-3", "long_range", "total")


def _bond_and_angle_pairs(mol):
    bonds = {tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))) for b in mol.GetBonds()}
    angles = set()
    for a in mol.GetAtoms():
        nb = [x.GetIdx() for x in a.GetNeighbors()]
        angles |= {tuple(sorted((nb[i], nb[j]))) for i in range(len(nb)) for j in range(i + 1, len(nb))}
    return bonds, angles


def _subset(params, mol, group):
    """The params of one oracle group; the fixed-window family splits by pair."""
    import copy
    p = copy.deepcopy(params)
    bonds, angles = _bond_and_angle_pairs(mol)
    for term, (attrs, _) in sb._ETK_TERM_FIELDS.items():
        n = len(getattr(p, attrs[0]))
        if group == "total":
            keep = np.ones(n, bool)
        elif term in ("torsion", "improper"):
            keep = np.full(n, group == term)
        elif term == "dist12":
            keep = np.full(n, group == "1-2")
        elif term in ("dist13", "angle"):
            keep = np.full(n, group == "1-3")
        else:  # fixed windows: 1-3 at improper centres, else long-range
            i1, i2 = getattr(p, attrs[0]), getattr(p, attrs[1])
            kind = ["1-3" if tuple(sorted((int(a), int(b)))) in angles else "long_range"
                    for a, b in zip(i1, i2)]
            keep = np.array([k == group for k in kind], bool)
        for a in attrs:
            setattr(p, a, getattr(p, a)[keep])
    return p


def _embedded(smi, seed=42):
    m = Chem.AddHs(Chem.MolFromSmiles(smi))
    p = rdDistGeom.ETKDGv3()
    p.randomSeed = seed
    assert rdDistGeom.EmbedMolecule(m, p) == 0
    return m


def test_oracle_gradient_is_the_derivative_of_its_energy():
    m = _embedded("C=C=CC(=O)Oc1ccccc1C#N")
    x0 = m.GetConformer().GetPositions()
    x = x0 + np.random.default_rng(0).normal(scale=0.05, size=x0.shape)
    b = get_bounds_matrix(m, use_macrocycle14config=True)
    ref = O.rdkit_etk_energy_terms(m, b, x0, x)
    h = 1e-6
    for g in ("torsion", "improper", "1-3"):
        fd = np.zeros_like(x)
        for i in range(x.shape[0]):
            for d in range(3):
                xp, xm = x.copy(), x.copy()
                xp[i, d] += h
                xm[i, d] -= h
                fd[i, d] = (O.rdkit_etk_energy_terms(m, b, x0, xp)[g][0]
                            - O.rdkit_etk_energy_terms(m, b, x0, xm)[g][0]) / (2 * h)
        np.testing.assert_allclose(ref[g][1], fd, atol=1e-6 * max(1.0, np.abs(fd).max()))


@pytest.mark.parametrize("variant", ["ETKDGv3", "srETKDGv3", "ETDG"])
def test_etk_field_matches_rdkit_term_by_term(variant):
    """Energy and gradient of every term group equal RDKit's to float32 precision."""
    mols, xr, xe, params, bmats = [], [], [], [], []
    rng = np.random.default_rng(1)
    f32 = lambda a: np.asarray(a, np.float32).astype(np.float64)  # noqa: E731
    for smi in PARITY_SMILES:
        m = _embedded(smi)
        b = f32(get_bounds_matrix(m, use_macrocycle14config=ee.ETKDG_VARIANTS[variant][4]))
        x0 = m.GetConformer().GetPositions()
        mols.append(m); bmats.append(b)
        xr.append(f32(x0)); xe.append(f32(x0 + rng.normal(scale=0.05, size=x0.shape)))
        params.append(ee.extract_etk_params(m, b, variant=variant))
    nat = np.array([m.GetNumAtoms() for m in mols])
    cas = np.concatenate([[0], np.cumsum(nat)])
    ref = [O.rdkit_etk_energy_terms(m, b, a, x, variant) for m, b, a, x in zip(mols, bmats, xr, xe)]
    for g in GROUPS:
        conc = sb.concat_etk_params([_subset(p, m, g) for p, m in zip(params, mols)])
        batch = sb.pack_per_conformer_etk_batch(
            conc, nat, np.arange(len(mols)), np.concatenate([a.ravel() for a in xr]).astype(np.float32))
        e, grad = em.etk_energy_and_gradient(batch, np.concatenate([a.ravel() for a in xe]).astype(np.float32))
        grad = grad.reshape(-1, 3)
        for c, smi in enumerate(PARITY_SMILES):
            er, gr = ref[c][g]
            gm = grad[cas[c]:cas[c + 1]]
            assert abs(float(e[c]) - er) <= 1e-4 * max(abs(er), 1.0), (variant, g, smi, float(e[c]), er)
            # 2e-3: a near-planar flat-ring torsion's cos(phi) rounds to 1 in float32
            assert np.abs(gm - gr).max() <= 2e-3 * max(np.abs(gr).max(), 1.0), (variant, g, smi)


def _etk_outputs(smi, n, variant="ETKDGv3"):
    """Positions and ETK energies of n attempts at the end of ETK, before the gate."""
    rec = []
    orig = cp.etk_minimize_shared

    def spy(batch, pos, **kw):
        out = orig(batch, pos, **kw)
        rec.append((np.array(batch.conf_atom_starts), out[0].reshape(-1, 3).copy(), out[1].copy()))
        return out

    cp.etk_minimize_shared = spy
    try:
        cp.generate_conformers_nk([smi], n, variant=variant, seed=42, max_rounds=1, oversample=0.0,
                                  return_failed=True)
    finally:
        cp.etk_minimize_shared = orig
    xs, es = [], []
    for cas, x, e in rec:
        xs += [x[cas[c]:cas[c + 1]] for c in range(len(cas) - 1)]
        es += list(e)
    return xs, np.array(es)


def _max_angle_at_non_sp(mol, x):
    best = 0.0
    for a in mol.GetAtoms():
        if a.GetHybridization() == Chem.HybridizationType.SP:
            continue
        nb = [n.GetIdx() for n in a.GetNeighbors()]
        for i in range(len(nb)):
            for j in range(i + 1, len(nb)):
                u, v = x[nb[i]] - x[a.GetIdx()], x[nb[j]] - x[a.GetIdx()]
                c = u @ v / np.linalg.norm(u) / np.linalg.norm(v)
                best = max(best, float(np.degrees(np.arccos(np.clip(c, -1, 1)))))
    return best


# Attempts out of 100 (seed 42) with an angle > 175 deg at a non-SP atom at the
# end of ETK, before the gate. Measured: aryl ether 10 (L-BFGS: 62; RDKit's own
# ETKDGv3 returns 15/100), colchicine 0 (L-BFGS: 29; RDKit 1/100), alloocimene
# 31 (L-BFGS: 36; RDKit's own ETK makes these too and its LINEAR_DOUBLE_BOND
# check rejects 61 per 100 accepted conformers). The limits leave room for
# platform noise but sit well below the L-BFGS saddle rates.
@pytest.mark.parametrize("smi,max_linear,max_median_e", [
    (ARYL_ETHER, 20, 0.0),
    (COLCHICINE, 5, 5.0),
    (ALLOOCIMENE, 45, 0.5),
])
def test_etk_does_not_park_angles_on_the_linear_saddle(smi, max_linear, max_median_e):
    mol = Chem.AddHs(Chem.MolFromSmiles(smi))
    xs, es = _etk_outputs(smi, 100)
    assert len(xs) == 100
    linear = np.array([_max_angle_at_non_sp(mol, x) > 175.0 for x in xs])
    assert linear.sum() <= max_linear, linear.sum()
    # Energies at the end of ETK in the normal range: the aryl ether's non-linear
    # attempts end at about -1.5 (the saddle left them at +40 to +84).
    assert np.median(es[~linear]) <= max_median_e, np.median(es[~linear])
    assert np.isfinite(es).all()


@pytest.mark.parametrize("smi,centre", [("CC#N", 1), ("C=C=CC", 1)])
def test_sp_centres_end_linear(smi, centre):
    mol = Chem.AddHs(Chem.MolFromSmiles(smi))
    res = cp.generate_conformers_nk([smi], 8, variant="ETKDGv3", seed=42)
    nb = [n.GetIdx() for n in mol.GetAtomWithIdx(centre).GetNeighbors()]
    assert len(res.molecules[0].positions_3d) == 8
    for pos in res.molecules[0].positions_3d:
        x = np.asarray(pos, float)
        u, v = x[nb[0]] - x[centre], x[nb[1]] - x[centre]
        assert np.degrees(np.arccos(u @ v / np.linalg.norm(u) / np.linalg.norm(v))) > 178.0


@pytest.mark.skipif(cp._rdkit_version() < cp._RING_DB_BOUNDS_FIXED,
                    reason="RDKit < 2026.03 gives ring double bonds wrong 1-4 bounds")
@pytest.mark.parametrize("smi", [
    "C=C1CC/C=C(\\C)CC[C@@H]2[C@@H]1CC2(C)C",
    "CC(C)=C1C/C=C(\\C)CC/C=C(\\C)CC1",
    "CC(C)=C1C/C=C(\\C)CC/C=C(\\C)CC1=O",
])
def test_ring_double_bonds_keep_their_ez(smi):
    """Once 0/8 (all attempts BAD_DOUBLE_BOND_STEREO); now 8/8 in 10 attempts."""
    res = cp.generate_conformers_nk([smi], 8, variant="srETKDGv3", seed=42)
    mr = res.molecules[0]
    assert len(mr.positions_3d) == 8
    assert mr.n_attempted <= 40
    assert mr.n_failed_by_cause.get("BAD_DOUBLE_BOND_STEREO", 0) <= 4


def test_old_rdkit_ring_double_bond_bounds_warn_once(monkeypatch):
    monkeypatch.setattr(cp, "_rdkit_version", lambda: (2025, 9))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cp._warn_ring_stereo_double_bonds(
            [Chem.AddHs(Chem.MolFromSmiles(s)) for s in ("C/C1=C\\CCCCCCC1", "CC/C=C/C1CC/C=C/CC1")])
    assert len([x for x in w if issubclass(x.category, RuntimeWarning)]) == 1
    monkeypatch.setattr(cp, "_rdkit_version", lambda: (2026, 3))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cp._warn_ring_stereo_double_bonds([Chem.AddHs(Chem.MolFromSmiles("C/C1=C\\CCCCCCC1"))])
    assert not w


def test_etk_slices_respect_the_hessian_budget():
    n = np.array([10, 10, 30, 5, 40])
    sl = cp._etk_slices(n, budget=(3 * 30) ** 2 + (3 * 10) ** 2)
    assert sl == [(0, 2), (2, 4), (4, 5)]
    assert cp._etk_slices(np.array([100]), budget=1) == [(0, 1)]
    assert cp._etk_slices(np.zeros(0, int)) == []
