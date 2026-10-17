"""Batched L-BFGS: every system in the batch is its own optimisation.

``lbfgs_batch`` gives each system its own history, line search, step
acceptance and stopping point, with one batched evaluation per iteration.
These tests pin that independence (different dimensions, conditioning and
starting points converge to their own answers at their own pace; a failing
system leaves the others bit-identical), the masking of padded coordinates
(NaN there must not leak), the converged <=> grad_norm contract, and the
float32 line-search design: with Hager & Zhang's float64 noise floor a stiff
quadratic stalls on energy noise, while the float32 floor converges in about
as many iterations as scipy needs in float64.
"""
from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem, rdForceFieldHelpers

from mlxmolkit.batch_lbfgs import CONVERGED, LINE_SEARCH_FAILED, MAX_ITERS, lbfgs_batch
from mlxmolkit.mmff_metal_optimizer import mmff_optimize_metal_lbfgs
from tests._mlx_cache import poison_buffer_cache


def _quadratics(dims, conds, seed=0):
    """Padded batch of 0.5 x'Ax - b'x with known minimisers; NaN in the
    padded gradient entries, which the mask must neutralise."""
    rng = np.random.default_rng(seed)
    B, D = len(dims), max(dims)
    A = np.zeros((B, D, D), np.float32)
    b = np.zeros((B, D), np.float32)
    mask = np.zeros((B, D), bool)
    solutions = []
    for i, (n, c) in enumerate(zip(dims, conds)):
        Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
        A[i, :n, :n] = (Q * np.geomspace(1, c, n)) @ Q.T
        b[i, :n] = rng.standard_normal(n)
        mask[i, :n] = True
        solutions.append(np.linalg.solve(A[i, :n, :n].astype(np.float64), b[i, :n]))
    A_mx, b_mx = mx.array(A), mx.array(b)
    junk = mx.array(np.where(mask, 0.0, np.nan).astype(np.float32))

    def fun(X):
        AX = mx.squeeze(A_mx @ X[:, :, None], -1)
        return 0.5 * mx.sum(X * AX, -1) - mx.sum(b_mx * X, -1), AX - b_mx + junk

    return fun, mask, solutions


def _rosenbrock(X):
    x0, x1 = X[:, :-1], X[:, 1:]
    f = mx.sum(100 * (x1 - x0**2) ** 2 + (1 - x0) ** 2, -1)
    z = mx.zeros((X.shape[0], 1))
    g = mx.concatenate([-400 * x0 * (x1 - x0**2) - 2 * (1 - x0), z], 1)
    return f, g + mx.concatenate([z, 200 * (x1 - x0**2)], 1)


def test_padded_quadratics_converge_independently():
    dims, conds = [5, 12, 40, 23], [1, 10, 1e2, 1e3]
    fun, mask, solutions = _quadratics(dims, conds)
    D = max(dims)
    poison_buffer_cache()
    res = lbfgs_batch(fun, np.zeros((len(dims), D)), mask=mask, grad_tol=1e-4, max_iters=500)

    assert (res.status == CONVERGED).all(), res.status
    for i, (n, x_star) in enumerate(zip(dims, solutions)):
        np.testing.assert_allclose(res.x[i, :n], x_star, atol=1e-4)
        np.testing.assert_array_equal(res.x[i, n:], 0.0)  # padding never moves
    # Each system stopped at its own iteration, not the slowest one's.
    assert len(set(res.n_iters.tolist())) == len(dims), res.n_iters
    assert res.n_iters.max() < res.n_evals <= 2 * res.n_iters.max()


def test_float32_noise_floor_is_what_lets_a_stiff_problem_converge():
    """With f_rel_noise=1e-6 (Hager-Zhang's float64 default) the energy test
    cannot see the last 1e-6 of descent in float32 and the line search
    backtracks to steps of 1e-5: stuck at |g| ~ 1.5e-2. The float32 floor
    converges in 131 iterations, against 115 for scipy's float64 L-BFGS-B."""
    fun, _, (x_star,) = _quadratics([23], [1e3])
    kw = dict(grad_tol=1e-4, max_iters=500)
    good = lbfgs_batch(fun, np.zeros((1, 23)), **kw)
    stalled = lbfgs_batch(fun, np.zeros((1, 23)), f_rel_noise=1e-6, **kw)

    assert good.status[0] == CONVERGED and good.n_iters[0] < 200
    np.testing.assert_allclose(good.x[0], x_star, atol=1e-4)
    assert stalled.status[0] == MAX_ITERS and stalled.grad_norm[0] > 1e-3


def test_rosenbrock_from_different_starts():
    x0 = np.random.default_rng(0).uniform(-2, 2, (8, 10)).astype(np.float32)
    x0[0] = -1.2
    res = lbfgs_batch(_rosenbrock, x0, grad_tol=1e-3, max_iters=2000)
    assert (res.status == CONVERGED).all(), res.status
    np.testing.assert_allclose(res.x, 1.0, atol=1e-3)


def test_a_failing_system_leaves_the_others_bit_identical():
    x0 = np.random.default_rng(0).uniform(-2, 2, (4, 10)).astype(np.float32)
    reference = lbfgs_batch(_rosenbrock, x0[:3], grad_tol=1e-3, max_iters=2000)

    def with_a_nan_system(X):
        f, g = _rosenbrock(X)
        return mx.where(mx.arange(X.shape[0]) == 3, float("nan"), f), g

    res = lbfgs_batch(with_a_nan_system, x0, grad_tol=1e-3, max_iters=2000)
    assert res.status[3] == LINE_SEARCH_FAILED
    np.testing.assert_array_equal(res.x[:3], reference.x)
    np.testing.assert_array_equal(res.n_iters[:3], reference.n_iters)


def test_converged_is_exactly_grad_norm_below_tol():
    """Including systems cut off by max_iters."""
    dims, conds = [5, 23, 31], [1, 1e3, 1e4]
    fun, mask, _ = _quadratics(dims, conds)
    tol = 1e-4
    res = lbfgs_batch(fun, np.zeros((3, max(dims))), mask=mask, grad_tol=tol, max_iters=60)
    assert set(res.status.tolist()) == {CONVERGED, MAX_ITERS}
    np.testing.assert_array_equal(res.converged, res.grad_norm <= tol)


# ---------------------------------------------------------------------------
# MMFF, through the padded multi-molecule kernel
# ---------------------------------------------------------------------------

SMILES = ["CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O", "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"]


def _embedded(smiles, n_confs=2):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    if len(AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, randomSeed=3)) != n_confs:
        pytest.skip(f"embedding failed for {smiles}")
    return mol


def _rdkit_energies(mol):
    props = rdForceFieldHelpers.MMFFGetMoleculeProperties(mol)
    return np.array([
        rdForceFieldHelpers.MMFFGetMoleculeForceField(mol, props, confId=c.GetId()).CalcEnergy()
        for c in mol.GetConformers()
    ])


def test_mmff_lbfgs_reaches_rdkit_minima():
    """Every conformer converges, and in float64 RDKit energy is no worse
    than RDKit's own minimiser from the same start (measured: within
    3e-6 kcal/mol, or a lower minimum)."""
    mols = [_embedded(s) for s in SMILES]
    reference = [_embedded(s) for s in SMILES]
    for mol in reference:
        rdForceFieldHelpers.MMFFOptimizeMoleculeConfs(mol, maxIters=10000)

    poison_buffer_cache()
    res = mmff_optimize_metal_lbfgs([(m, None) for m in mols], grad_tol=1e-4)

    for smi, mol, ref, mol_res in zip(SMILES, mols, reference, res.mol_results):
        assert mol_res.converged.all(), smi
        np.testing.assert_array_equal(mol_res.converged, mol_res.grad_norms <= 1e-4)
        # Positions were written back, so this is the returned geometry.
        assert (_rdkit_energies(mol) <= _rdkit_energies(ref) + 1e-3).all(), smi


def test_mmff_lbfgs_batch_matches_each_molecule_alone():
    """Padding changes only the length of float32 row sums, so padded
    molecules agree to rounding amplified by the optimiser (measured 2.7e-5
    kcal/mol, 4.7e-4 A); the largest molecule, unpadded, is bitwise."""
    alone = [mmff_optimize_metal_lbfgs([(_embedded(s), None)]).mol_results[0] for s in SMILES]
    batch = mmff_optimize_metal_lbfgs([(_embedded(s), None) for s in SMILES]).mol_results
    for smi, a, b in zip(SMILES, alone, batch):
        np.testing.assert_array_equal(b.converged, a.converged, err_msg=smi)
        np.testing.assert_allclose(b.energies, a.energies, atol=2e-4, err_msg=smi)
        np.testing.assert_allclose(b.positions, a.positions, atol=5e-3, err_msg=smi)
    np.testing.assert_array_equal(batch[-1].positions, alone[-1].positions)
