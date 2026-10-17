"""MMFF: a molecule's result must not depend on what it is batched with.

Every batched path here packs several molecules (and several conformers each)
into one kernel launch. Each test computes the batch once with MLX's buffer
cache full of NaN, then compares every slot with the same molecule computed
on its own. A wrong offset, a slot written for the wrong length, or an output
read before it is written shows up as a mismatch or a NaN.

See also tests/test_metal_multi_mol_padding.py, which pins the padded
multi-molecule optimiser specifically.
"""
from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import rdDistGeom

from mlxmolkit.mmff_metal_kernel import (
    mmff_energy_grad_metal,
    mmff_energy_grad_metal_mega,
    pack_multi_mol_for_metal,
    pack_params_for_metal,
)
from mlxmolkit.mmff_minimize import mmff_minimize_nk
from mlxmolkit.mmff_params import extract_mmff_params
from tests._mlx_cache import poison_buffer_cache

# Different sizes on purpose (9, 24, 33 atoms) and different conformer counts,
# so slot offsets and lengths all differ.
BATCH = [("CCO", 2), ("CN1C=NC2=C1C(=O)N(C(=O)N2C)C", 1), ("CC(C)CC1=CC=C(C=C1)C(C)C(=O)O", 3)]


def _embed(smiles, n_confs, seed=11):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    if len(rdDistGeom.EmbedMultipleConfs(mol, n_confs, randomSeed=seed)) != n_confs:
        pytest.skip(f"embedding failed for {smiles}")
    return mol


@pytest.fixture(scope="module")
def batch():
    mols = [_embed(smi, k) for smi, k in BATCH]
    params = [extract_mmff_params(m) for m in mols]
    pos = [
        np.stack([c.GetPositions() for c in m.GetConformers()]).astype(np.float32)
        for m in mols
    ]  # per molecule (C, N, 3)
    return mols, params, pos


def _single_molecule(params, pos):
    idx_buf, param_buf, meta = pack_params_for_metal(params)
    e, g = mmff_energy_grad_metal(idx_buf, param_buf, meta, mx.array(pos))
    mx.eval(e, g)
    return np.array(e), np.array(g).reshape(pos.shape)


@pytest.mark.parametrize("layout", ["packed", "padded"])
def test_mega_kernel_matches_single_molecule_kernel(batch, layout):
    _, params, pos = batch
    counts = [p.shape[0] for p in pos]
    dims = [p.n_atoms * 3 for p in params]
    idx_buf, param_buf, all_meta, dispatch, offsets, _ = pack_multi_mol_for_metal(
        params, counts
    )
    if layout == "packed":
        flat = np.concatenate([p.reshape(-1) for p in pos])
        starts = np.array(offsets)
    else:
        stride = max(dims)
        slots = [np.pad(c.reshape(-1), (0, stride - d)) for p, d in zip(pos, dims) for c in p]
        flat = np.concatenate(slots)
        starts = np.arange(sum(counts)) * stride
        offsets = mx.array(starts.astype(np.uint32))

    poison_buffer_cache(flat.size)
    e, g = mmff_energy_grad_metal_mega(
        idx_buf, param_buf, all_meta, dispatch, offsets,
        mx.array(flat), sum(counts), flat.size,
    )
    mx.eval(e, g)
    e, g = np.array(e), np.array(g)
    assert np.isfinite(e).all() and np.isfinite(g).all()

    owned = np.zeros(flat.size, bool)
    cid = 0
    for p, d, xyz in zip(params, dims, pos):
        e_ref, g_ref = _single_molecule(p, xyz)
        for k in range(xyz.shape[0]):
            s = starts[cid]
            np.testing.assert_allclose(e[cid], e_ref[k], rtol=1e-6, atol=1e-5)
            np.testing.assert_allclose(
                g[s : s + d], g_ref[k].reshape(-1), rtol=1e-5, atol=1e-4
            )
            owned[s : s + d] = True
            cid += 1
    np.testing.assert_array_equal(g[~owned], 0.0)


@pytest.mark.parametrize("use_lbfgs", [False, True], ids=["bfgs", "lbfgs"])
def test_minimize_nk_batch_matches_one_molecule_at_a_time(batch, use_lbfgs):
    """Each conformer owns a threadgroup, so batching should change nothing."""
    _, params, pos = batch
    counts = [p.shape[0] for p in pos]
    kw = dict(max_iters=300, use_lbfgs=use_lbfgs)

    alone = [mmff_minimize_nk([p], [xyz.shape[0]], xyz.reshape(-1), **kw) for p, xyz in zip(params, pos)]

    flat = np.concatenate([p.reshape(-1) for p in pos])
    poison_buffer_cache(flat.size, 3 * flat.size)
    out_pos, energies, converged = mmff_minimize_nk(params, counts, flat, **kw)

    cursor, cid = 0, 0
    for (ref_pos, ref_e, ref_conv), xyz in zip(alone, pos):
        n = xyz.size
        k = xyz.shape[0]
        np.testing.assert_allclose(out_pos[cursor : cursor + n], ref_pos, atol=1e-5)
        np.testing.assert_allclose(energies[cid : cid + k], ref_e, atol=1e-5)
        np.testing.assert_array_equal(converged[cid : cid + k], ref_conv)
        cursor += n
        cid += k
    assert np.isfinite(energies).all()
