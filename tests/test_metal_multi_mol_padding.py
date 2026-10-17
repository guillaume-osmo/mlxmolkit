"""Padded multi-molecule MMFF: the tail of each conformer's slot must be zero.

``mmff_optimize_metal_multi_mol`` lays every conformer out at stride
``max_dim`` so the batch is one ``(total_confs, max_dim)`` matrix for the
L-BFGS dot products. The mega-kernel only writes each conformer's own
``n_atoms*3`` entries, so before its output was zero-initialised the tail
``[n_atoms*3, max_dim)`` of every smaller molecule's slot was whatever MLX's
buffer cache last held. A NaN there poisons that conformer's L-BFGS
direction, the energy guard never accepts a step, and the conformer comes
back silently frozen at its input geometry with ``converged=False``.
"""
from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import rdDistGeom

from mlxmolkit.mmff_metal_kernel import (
    mmff_energy_grad_metal_mega,
    mmff_optimize_metal_fused,
    pack_multi_mol_for_metal,
)
from mlxmolkit.mmff_metal_optimizer import (
    mmff_optimize_metal_fused_multi_mol,
    mmff_optimize_metal_multi_mol,
)
from mlxmolkit.mmff_params import extract_mmff_params

# 9 and 33 atoms: ethanol's slot carries a 72-entry padded tail.
SMALL, LARGE = "CCO", "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"


def _embed(smiles, n_confs=1, seed=7):
    mols = []
    for smi in smiles:
        mol = Chem.AddHs(Chem.MolFromSmiles(smi))
        if len(rdDistGeom.EmbedMultipleConfs(mol, n_confs, randomSeed=seed)) != n_confs:
            pytest.skip(f"embedding failed for {smi}")
        mols.append(mol)
    return mols


def _pollute_buffer_cache(n_floats, n_buffers=64):
    """Make the next allocations of this size reuse NaN-filled buffers.

    Freed MLX buffers go back to a cache and are handed out again without
    being cleared, which is how uninitialised kernel outputs end up holding
    stale data in practice. Enough buffers that the gradient output gets one
    even after the optimiser's own allocations of the same size.
    """
    junk = [mx.full((n_floats,), float("nan")) for _ in range(n_buffers)]
    mx.eval(*junk)
    del junk


def _padded_inputs(mols):
    params = [extract_mmff_params(m) for m in mols]
    dims = [p.n_atoms * 3 for p in params]
    max_dim = max(dims)
    packed = pack_multi_mol_for_metal(params, [1] * len(mols))
    pos = np.zeros((len(mols), max_dim), np.float32)
    for c, m in enumerate(mols):
        pos[c, : dims[c]] = m.GetConformer().GetPositions().ravel()
    offsets = mx.array(np.arange(len(mols), dtype=np.uint32) * max_dim)
    return packed, dims, max_dim, pos, offsets


def test_mega_kernel_zeroes_the_padded_tail():
    mols = _embed([SMALL, LARGE])
    packed, dims, max_dim, pos, offsets = _padded_inputs(mols)
    idx_buf, param_buf, all_meta, dispatch, packed_offsets, _ = packed
    total = len(mols) * max_dim

    # Same conformers in the unpadded layout: the owned entries must not
    # depend on the stride.
    pos_packed = np.concatenate([pos[c, : dims[c]] for c in range(len(mols))])
    _, g_packed = mmff_energy_grad_metal_mega(
        idx_buf, param_buf, all_meta, dispatch, packed_offsets,
        mx.array(pos_packed), len(mols), int(sum(dims)),
    )
    g_packed = np.array(g_packed)

    for _ in range(3):
        _pollute_buffer_cache(total)
        e, g = mmff_energy_grad_metal_mega(
            idx_buf, param_buf, all_meta, dispatch, offsets,
            mx.array(pos.ravel()), len(mols), total,
        )
        mx.eval(e, g)
        g = np.array(g).reshape(len(mols), max_dim)
        np.testing.assert_array_equal(g[0, dims[0]:], 0.0)
        np.testing.assert_array_equal(g[0, : dims[0]], g_packed[: dims[0]])
        np.testing.assert_array_equal(g[1], g_packed[dims[0]:])
        assert np.isfinite(np.array(e)).all()


def test_fused_optimizer_kernel_zeroes_the_padded_tail():
    mols = _embed([SMALL, LARGE])
    packed, dims, max_dim, pos, offsets = _padded_inputs(mols)
    idx_buf, param_buf, all_meta, dispatch, _, _ = packed
    total = len(mols) * max_dim

    _pollute_buffer_cache(total)
    pos_out, energy_out, grad_out = mmff_optimize_metal_fused(
        idx_buf, param_buf, all_meta, dispatch, offsets,
        mx.array(pos.ravel()), len(mols), total, max_iters=20,
    )
    mx.eval(pos_out, energy_out, grad_out)
    for buf in (pos_out, grad_out):
        tail = np.array(buf).reshape(len(mols), max_dim)[0, dims[0]:]
        np.testing.assert_array_equal(tail, 0.0)
    assert np.isfinite(np.array(energy_out)).all()


@pytest.mark.parametrize(
    "optimize",
    [mmff_optimize_metal_multi_mol, mmff_optimize_metal_fused_multi_mol],
    ids=["lbfgs", "fused"],
)
def test_mixed_size_batch_matches_per_molecule(optimize):
    """Batching with a larger molecule must not change a small one's result."""
    alone = [
        optimize([(m, None)]).mol_results[0]
        for m in _embed([SMALL, LARGE], n_confs=3)
    ]

    mols = _embed([SMALL, LARGE], n_confs=3)
    max_dim = max(m.GetNumAtoms() for m in mols) * 3
    _pollute_buffer_cache(sum(m.GetNumConformers() for m in mols) * max_dim)
    batched = optimize([(m, None) for m in mols]).mol_results

    for smi, a, b in zip((SMALL, LARGE), alone, batched):
        # Padding changes only the reduction length of the L-BFGS dot
        # products, so agreement is to float32 summation order, not bitwise.
        # The stale-tail failure froze ethanol at its input geometry: ~5.7
        # kcal/mol and ~0.5 A off.
        np.testing.assert_allclose(b.energies, a.energies, atol=1e-4, err_msg=smi)
        np.testing.assert_allclose(b.positions, a.positions, atol=1e-3, err_msg=smi)
        if a.converged is not None:
            np.testing.assert_array_equal(b.converged, a.converged, err_msg=smi)
