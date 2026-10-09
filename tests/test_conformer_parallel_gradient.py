"""The 32-lane gradients of the DG / ETK / MMFF kernels against the serial ones.

Each parallel gradient sums every component in the serial loop's term order,
so the optimisers must take bit-identical trajectories, not merely close ones:
L-BFGS amplifies a last-bit difference in the gradient into a different
local minimum for a few conformers, which a tolerance would hide.
"""
import numpy as np
import pytest

from mlxmolkit.conformer_metal import build_atom_term_csr, dg_minimize_shared
from mlxmolkit.dg_extract import extract_dg_params, get_bounds_matrix, metric_matrix_positions
from mlxmolkit.etk_extract import extract_etk_params
from mlxmolkit.etk_metal import etk_minimize_shared
from mlxmolkit.mmff_minimize import mmff_minimize_nk
from mlxmolkit.mmff_params import extract_mmff_params
from mlxmolkit.shared_batch import add_etk_to_batch, init_random_positions, pack_shared_dg_batch

Chem = pytest.importorskip("rdkit.Chem")

# Stereocentres (chiral terms), rings (impropers), rotors (torsions).
SMILES = [
    "C[C@]12CC[C@H]3[C@@H](CC=C4C[C@@H](O)CC[C@@]34C)[C@@H]1CC[C@@H]2O",
    "CC1(C)S[C@@H]2[C@H](NC(=O)Cc3ccccc3)C(=O)N2[C@H]1C(=O)O",
    "CC(C)NC[C@H](O)COc1cccc2ccccc12",
]


def _mols():
    mols = [Chem.AddHs(Chem.MolFromSmiles(s)) for s in SMILES]
    bmats = [get_bounds_matrix(m) for m in mols]
    return mols, bmats


def test_csr_lists_each_atoms_terms_in_serial_order():
    # mol 0: 3 atoms, pairs (0,1) (0,2) (1,2); mol 1: 2 atoms, pair (0,1)
    pairs = np.array([[0, 1], [0, 2], [1, 2], [0, 1]])
    pair_starts = np.array([0, 3, 4])
    quads = np.array([[2, 0, 1, 2]])  # degenerate on purpose: atom 2 in two roles
    quad_starts = np.array([0, 1, 1])
    slot_base, off, ent, partner = build_atom_term_csr(
        np.array([3, 2]), [(pair_starts, pairs), (quad_starts, quads)])
    assert slot_base.tolist() == [0, 6, 10]

    def slot(m, t, a):
        n = [3, 2][m]
        return slot_base[m] + t * n + a

    def entries(m, t, a):
        s = slot(m, t, a)
        return [(int(e) >> 2, int(e) & 3) for e in ent[off[s]:off[s + 1]]], partner[off[s]:off[s + 1]].tolist()

    assert entries(0, 0, 0) == ([(0, 0), (1, 0)], [1, 2])
    assert entries(0, 0, 1) == ([(0, 1), (2, 0)], [0, 2])
    assert entries(0, 0, 2) == ([(1, 1), (2, 1)], [0, 1])
    assert entries(1, 0, 0) == ([(3, 0)], [1])
    assert entries(1, 0, 1) == ([(3, 1)], [0])
    assert entries(0, 1, 2)[0] == [(0, 0), (0, 3)]
    assert entries(0, 1, 0)[0] == [(0, 1)]
    assert entries(1, 1, 0)[0] == []


@pytest.mark.parametrize("dim", [4, 3])
def test_dg_parallel_gradient_is_bit_identical_to_serial(dim):
    mols, bmats = _mols()
    dgs = [extract_dg_params(m, b, dim=dim) for m, b in zip(mols, bmats)]
    batch = pack_shared_dg_batch(dgs, [4] * len(dgs), dim=dim)
    assert len(batch.chiral_idx1) > 0
    if dim == 4:
        pos = metric_matrix_positions(batch, bmats, seed=7, dim=4)
    else:
        pos = init_random_positions(batch, seed=7)
    ser = dg_minimize_shared(batch, pos, max_iters=400, parallel_grad=False)
    par = dg_minimize_shared(batch, pos, max_iters=400, parallel_grad=True)
    np.testing.assert_array_equal(par[2], ser[2])
    np.testing.assert_array_equal(par[1], ser[1])
    np.testing.assert_array_equal(par[0], ser[0])
    # and run to run
    np.testing.assert_array_equal(dg_minimize_shared(batch, pos, max_iters=400)[0], par[0])


@pytest.mark.parametrize("optimizer", ["bfgs", "lbfgs"])
def test_etk_parallel_gradient_is_bit_identical_to_serial(optimizer):
    mols, bmats = _mols()
    dgs = [extract_dg_params(m, b, dim=4) for m, b in zip(mols, bmats)]
    etks = [extract_etk_params(m, b, variant="ETKDGv3") for m, b in zip(mols, bmats)]
    k = [3] * len(mols)
    batch4 = pack_shared_dg_batch(dgs, k, dim=4)
    pos4 = metric_matrix_positions(batch4, bmats, seed=11, dim=4)
    dg_out, _, _ = dg_minimize_shared(batch4, pos4, max_iters=600)
    batch3 = pack_shared_dg_batch(dgs, k, dim=3)
    add_etk_to_batch(batch3, etks)
    assert len(batch3.etk_torsion_idx) > 0 and len(batch3.etk_improper_idx) > 0
    assert len(batch3.etk_dist14_idx1) > 0
    pos3 = np.concatenate([
        dg_out[int(batch4.conf_atom_starts[c]) * 4:int(batch4.conf_atom_starts[c + 1]) * 4]
        .reshape(-1, 4)[:, :3].ravel()
        for c in range(batch3.n_confs_total)
    ]).astype(np.float32)
    ser = etk_minimize_shared(batch3, pos3, max_iters=300, parallel_grad=False, optimizer=optimizer)
    par = etk_minimize_shared(batch3, pos3, max_iters=300, parallel_grad=True, optimizer=optimizer)
    np.testing.assert_array_equal(par[2], ser[2])
    np.testing.assert_array_equal(par[1], ser[1])
    np.testing.assert_array_equal(par[0], ser[0])


@pytest.mark.parametrize("use_lbfgs", [False, True])
def test_mmff_parallel_gradient_is_bit_identical_to_serial(use_lbfgs):
    from rdkit.Chem import AllChem

    params, pos, counts = [], [], []
    for i, smi in enumerate(SMILES):
        mol = Chem.AddHs(Chem.MolFromSmiles(smi))
        cids = AllChem.EmbedMultipleConfs(mol, numConfs=3, randomSeed=5 + i)
        params.append(extract_mmff_params(mol))
        counts.append(len(cids))
        for cid in cids:
            pos.append(mol.GetConformer(cid).GetPositions().astype(np.float32).ravel())
    pos = np.concatenate(pos)
    ser = mmff_minimize_nk(params, counts, pos, max_iters=400, use_lbfgs=use_lbfgs, parallel_grad=False)
    par = mmff_minimize_nk(params, counts, pos, max_iters=400, use_lbfgs=use_lbfgs, parallel_grad=True)
    np.testing.assert_array_equal(par[2], ser[2])
    np.testing.assert_array_equal(par[1], ser[1])
    np.testing.assert_array_equal(par[0], ser[0])


def test_mmff_kernel_compiles_once_for_every_batch_shape(monkeypatch):
    from rdkit.Chem import AllChem
    from mlxmolkit import mmff_minimize
    from mlxmolkit.conformer_metal import GRAD_GATHER

    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    AllChem.EmbedMultipleConfs(mol, numConfs=3, randomSeed=1)
    p = extract_mmff_params(mol)
    real = mmff_minimize._get_mmff_kernel_tg(GRAD_GATHER)
    calls = []

    def spy(**kw):
        calls.append(kw)
        return real(**kw)

    monkeypatch.setitem(mmff_minimize._mmff_kernel_tg, GRAD_GATHER, spy)
    for k in (1, 2, 3):
        pos = np.concatenate([mol.GetConformer(c).GetPositions().astype(np.float32).ravel() for c in range(k)])
        mmff_minimize_nk([p], [k], pos, max_iters=5)
    # A template argument is part of the compiled library's key: total_pos_size
    # used to be one, so every new batch shape paid a fresh Metal compile.
    assert len(calls) == 3
    assert all(not kw.get("template") for kw in calls)
