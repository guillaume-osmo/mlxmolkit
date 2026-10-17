"""Conformer optimisers: a molecule's result must not depend on what it is batched with.

Each test runs one batch of molecules of different sizes (8 to 33 atoms, one
with a stereocentre, several conformers of some) with MLX's buffer cache full
of NaN, then reruns every molecule on its own from the same starting
coordinates. The two must agree bitwise: positions, energies, convergence
flags, and gradient norms where reported.

The shared-constraint kernels (``dg_minimize_shared``, ``etk_minimize_shared``)
give every conformer its own threadgroup, and ``parallel_grad`` is a gather
with no atomics, so both gradient modes must be exact.

The host-driven loops (``dg_minimize_batch``, ``etk_minimize_batch``,
``lbfgs_minimize_batch``, ``bfgs_minimize_batch``) evaluate every molecule in
one Metal dispatch, which is exact, and take the line-search slope (and the
reported gradient norm) on each molecule's slice of the packed float32
vectors. Accelerate rounds a float32 ``np.dot`` differently when the slice
does not start on a 16-byte boundary, so a 3-D molecule's slope used to depend
on whether the number of atoms packed before it was a multiple of 4, and the
optimiser amplified the last-bit difference (up to 2.6e-3 A in positions).
Those reductions now go through ``bfgs_metal._slice_dot``. ``ALIGNED`` keeps
every slice on a 16-byte boundary, ``MISALIGNED`` does not; both must be
exact. 4-D DG slices are always aligned (4 floats per atom).

The single-molecule ``bfgs_metal.lbfgs_minimize`` / ``bfgs_minimize`` are not
the reference for the batched BFGS: they are a separate implementation (energy
summed on the GPU, a float32 step kernel, a different steepest-descent
fallback) and agree with it only to rounding. The batch of one is the
reference instead.

Not covered on purpose: ``generate_conformers_nk`` end to end, which is batch
dependent by design (iteration caps from the largest molecule, retries that
depend on the batch, one RNG stream).
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import rdDistGeom

from mlxmolkit.bfgs_batch_metal import bfgs_minimize_batch, lbfgs_minimize_batch
from mlxmolkit.conformer_metal import DEFAULT_LBFGS_M, dg_minimize_shared
from mlxmolkit.dg_extract import batch_dg_params, extract_dg_params, get_bounds_matrix
from mlxmolkit.dg_minimize_metal import dg_minimize_batch
from mlxmolkit.etk_extract import batch_etk_params, extract_etk_params
from mlxmolkit.etk_metal import etk_minimize_shared
from mlxmolkit.etk_minimize_metal import etk_minimize_batch
from mlxmolkit.shared_batch import add_etk_to_batch, init_random_positions, pack_shared_dg_batch
from tests._mlx_cache import poison_buffer_cache

SMILES = {
    "acetic_acid": "CC(=O)O",                            # 8 atoms
    "ethanol": "CCO",                                    # 9
    "caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",          # 24
    "ibuprofen": "CC(C)CC1=CC=C(C=C1)[C@@H](C)C(=O)O",  # 33, one chiral term
}

# Shared-constraint kernels: (molecule, conformers).
SHARED = [("ethanol", 2), ("caffeine", 1), ("ibuprofen", 3)]

# Host loops: one system per entry, a repeated name is another conformer.
# Atoms packed before each entry: 0, 24, 48, 56 vs 0, 9, 33, 66, 99.
ALIGNED = ["caffeine", "caffeine", "acetic_acid", "ibuprofen"]
MISALIGNED = ["ethanol", "caffeine", "ibuprofen", "ibuprofen", "ethanol"]


@pytest.fixture(scope="module")
def mols():
    """name -> (n_atoms, DGParams, ETKParams, plain-DG conformer coordinates)."""
    need = Counter(ALIGNED) | Counter(MISALIGNED)
    for name, k in SHARED:
        need[name] = max(need[name], k)
    out = {}
    for name, smi in SMILES.items():
        mol = Chem.AddHs(Chem.MolFromSmiles(smi))
        bounds = get_bounds_matrix(mol)
        # No torsion preferences, so ETK still has work to do from here.
        ids = rdDistGeom.EmbedMultipleConfs(
            mol, need[name], randomSeed=11,
            useExpTorsionAnglePrefs=False, useBasicKnowledge=False,
        )
        if len(ids) != need[name]:
            pytest.skip(f"embedding failed for {smi}")
        confs = [c.GetPositions().astype(np.float32).ravel() for c in mol.GetConformers()]
        out[name] = (
            mol.GetNumAtoms(),
            extract_dg_params(mol, bounds),
            extract_etk_params(mol, bounds),
            confs,
        )
    return out


def _slices(sizes):
    ends = np.cumsum(sizes)
    return [slice(int(e - s), int(e)) for s, e in zip(sizes, ends)]


def _conformers(mols, names):
    """The j-th occurrence of a name takes that molecule's j-th conformer."""
    seen = Counter()
    out = []
    for name in names:
        out.append(mols[name][3][seen[name]])
        seen[name] += 1
    return out


def _assert_system_matches(batch, alone, i, sl):
    np.testing.assert_array_equal(batch.positions[sl], alone.positions)
    np.testing.assert_array_equal(batch.energies[i], alone.energies[0])
    np.testing.assert_array_equal(batch.grad_norms[i], alone.grad_norms[0])
    assert batch.converged[i] == alone.converged[0]


# ---------------------------------------------------------------------------
# Shared-constraint kernels
# ---------------------------------------------------------------------------

def _assert_shared_matches(batch_out, alone_outs, counts, dim):
    out_pos, energies, statuses = batch_out
    assert np.isfinite(out_pos).all() and np.isfinite(energies).all()
    pos_sl = _slices([n * k * dim for n, k in counts])
    conf_sl = _slices([k for _, k in counts])
    for (ref_pos, ref_e, ref_s), ps, cs in zip(alone_outs, pos_sl, conf_sl):
        np.testing.assert_array_equal(out_pos[ps], ref_pos)
        np.testing.assert_array_equal(energies[cs], ref_e)
        np.testing.assert_array_equal(statuses[cs], ref_s)


@pytest.mark.parametrize("parallel_grad", [False, True], ids=["serial_grad", "parallel_grad"])
def test_shared_dg_matches_one_molecule_at_a_time(mols, parallel_grad):
    dg = [mols[name][1] for name, _ in SHARED]
    ks = [k for _, k in SHARED]
    counts = [(mols[name][0], k) for name, k in SHARED]
    batch = pack_shared_dg_batch(dg, ks, dim=4)
    pos = init_random_positions(batch, seed=42)
    kw = dict(max_iters=100, parallel_grad=parallel_grad)

    alone = [
        dg_minimize_shared(pack_shared_dg_batch([p], [k], dim=4), pos[sl], **kw)
        for p, k, sl in zip(dg, ks, _slices([n * k * 4 for n, k in counts]))
    ]

    n = pos.size
    poison_buffer_cache(n, 3 * n, 2 * DEFAULT_LBFGS_M * n)
    _assert_shared_matches(dg_minimize_shared(batch, pos, **kw), alone, counts, dim=4)


@pytest.mark.parametrize("parallel_grad", [False, True], ids=["serial_grad", "parallel_grad"])
def test_shared_etk_matches_one_molecule_at_a_time(mols, parallel_grad):
    def pack(names_ks):
        b = pack_shared_dg_batch([mols[nm][1] for nm, _ in names_ks], [k for _, k in names_ks], dim=3)
        add_etk_to_batch(b, [mols[nm][2] for nm, _ in names_ks])
        return b

    counts = [(mols[name][0], k) for name, k in SHARED]
    per_mol = [np.concatenate(mols[name][3][:k]) for name, k in SHARED]
    kw = dict(max_iters=300, parallel_grad=parallel_grad)

    alone = [etk_minimize_shared(pack([nk]), p, **kw) for nk, p in zip(SHARED, per_mol)]

    pos = np.concatenate(per_mol)
    n = pos.size
    poison_buffer_cache(n, 3 * n, 2 * DEFAULT_LBFGS_M * n)
    _assert_shared_matches(etk_minimize_shared(pack(SHARED), pos, **kw), alone, counts, dim=3)


# ---------------------------------------------------------------------------
# Host-driven L-BFGS / BFGS loops
# ---------------------------------------------------------------------------

def test_dg_minimize_batch_matches_batch_of_one(mols):
    # MISALIGNED in 3-D, but 4 floats per atom keeps every slice aligned.
    names = MISALIGNED
    dg = [mols[nm][1] for nm in names]
    sl = _slices([mols[nm][0] * 4 for nm in names])
    x0 = np.random.default_rng(42).standard_normal(sl[-1].stop).astype(np.float32)
    kw = dict(max_iters=200)

    alone = [dg_minimize_batch(batch_dg_params([p]), x0=x0[s], **kw) for p, s in zip(dg, sl)]

    poison_buffer_cache(x0.size, x0.size // 4)
    batch = dg_minimize_batch(batch_dg_params(dg), x0=x0, **kw)
    assert np.isfinite(batch.positions).all()
    for i, (ref, s) in enumerate(zip(alone, sl)):
        _assert_system_matches(batch, ref, i, s)


@pytest.mark.parametrize("names", [
    pytest.param(ALIGNED, id="aligned"),
    pytest.param(MISALIGNED, id="misaligned"),
])
def test_etk_minimize_batch_matches_batch_of_one(mols, names):
    etk = [mols[nm][2] for nm in names]
    n_atoms = [mols[nm][0] for nm in names]
    sl = _slices([n * 3 for n in n_atoms])
    # From the plain-DG coordinates alone the misaligned layout happens to
    # come out exact; noise gives every system a line search worth the name.
    rng = np.random.default_rng(3)
    x0 = [(c + 0.3 * rng.standard_normal(c.size)).astype(np.float32) for c in _conformers(mols, names)]
    kw = dict(max_iters=200)

    alone = [
        etk_minimize_batch(batch_etk_params([p], np.array([0, n], np.int32)), x, **kw)
        for p, n, x in zip(etk, n_atoms, x0)
    ]

    atom_starts = np.concatenate([[0], np.cumsum(n_atoms)]).astype(np.int32)
    x0 = np.concatenate(x0)
    poison_buffer_cache(x0.size, x0.size // 3)
    batch = etk_minimize_batch(batch_etk_params(etk, atom_starts), x0, **kw)
    assert np.isfinite(batch.positions).all()
    for i, (ref, s) in enumerate(zip(alone, sl)):
        _assert_system_matches(batch, ref, i, s)


def _spring_terms(mol_name):
    """Pairs and mid-bound targets for the bfgs_batch_metal spring energy."""
    mol = Chem.AddHs(Chem.MolFromSmiles(SMILES[mol_name]))
    n = mol.GetNumAtoms()
    bounds = rdDistGeom.GetMoleculeBoundsMatrix(mol)
    i, j = np.triu_indices(n, 1)
    lb, ub = bounds[j, i], bounds[i, j]
    keep = (ub > 0) & (ub < 100)
    pairs = np.stack([i[keep], j[keep]], axis=1).astype(np.int32)
    return pairs, ((lb[keep] + ub[keep]) / 2).astype(np.float32), n


@pytest.mark.parametrize("minimize, names", [
    pytest.param(lbfgs_minimize_batch, ALIGNED, id="lbfgs-aligned"),
    pytest.param(lbfgs_minimize_batch, MISALIGNED, id="lbfgs-misaligned"),
    pytest.param(bfgs_minimize_batch, ALIGNED, id="bfgs-aligned"),
    pytest.param(bfgs_minimize_batch, MISALIGNED, id="bfgs-misaligned"),
])
def test_bfgs_batch_matches_batch_of_one(minimize, names):
    pairs, targets, counts = zip(*[_spring_terms(nm) for nm in names])
    sl = _slices([n * 3 for n in counts])
    x0 = (2.0 * np.random.default_rng(42).standard_normal(sl[-1].stop)).astype(np.float32)
    kw = dict(max_iters=300)

    alone = [
        minimize([p], [t], [n], x0=x0[s], **kw)
        for p, t, n, s in zip(pairs, targets, counts, sl)
    ]

    poison_buffer_cache(x0.size, x0.size // 3)
    batch = minimize(list(pairs), list(targets), list(counts), x0=x0, **kw)
    assert np.isfinite(batch.positions).all()
    for i, (ref, s) in enumerate(zip(alone, sl)):
        _assert_system_matches(batch, ref, i, s)
