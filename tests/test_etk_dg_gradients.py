"""ETK and DG Metal energy kernels: the gradient must be the energy's gradient.

etk_energy_metal computed the dihedral as -phi(IUPAC) but multiplied dE/dphi
by the derivative of +phi, and took the chain-rule terms for the middle atoms
from the GROMACS form (r_ij = -b1, r_kl = -b3) without flipping the signs of
the dot products. Torsion gradients pointed the wrong way (cosine -0.96
against finite differences; the old dihedral-form improper shared the bug
before it was replaced by RDKit's UFF inversion term, also checked here), so
etk_minimize_batch stopped after two iterations reporting converged
(caffeine: E = 126, |g| = 583). The energies were right: cos(k*phi) is even.

dg_energy_metal added the fourth-dimension penalty only on the coord-3
thread, but only the coord-0 thread writes the energy, so the energy left the
term out while the gradient included it.
"""
from __future__ import annotations

import dataclasses

import mlx.core as mx
import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import rdDistGeom

from mlxmolkit.dg_energy_metal import make_dg4d_energy_grad
from mlxmolkit.dg_extract import batch_dg_params, extract_dg_params, get_bounds_matrix
from mlxmolkit.etk_energy_metal import make_etk_energy_grad
from mlxmolkit.etk_extract import batch_etk_params, extract_etk_params
from mlxmolkit.etk_minimize_metal import etk_minimize_batch

CAFFEINE = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"


def _fd_gradient(energy, x, h=1e-3):
    return np.array([(energy(x + h * e) - energy(x - h * e)) / (2 * h) for e in np.eye(x.size)])


def _energy_fn(fn):
    return lambda y: float(np.sum(np.array(fn(mx.array(y.astype(np.float32)))[0]), dtype=np.float64))


@pytest.fixture(scope="module")
def caffeine():
    mol = Chem.AddHs(Chem.MolFromSmiles(CAFFEINE))
    etk = extract_etk_params(mol, get_bounds_matrix(mol))
    conf = Chem.Mol(mol)
    # No torsion preferences, so the ETK terms are far from their minima.
    rdDistGeom.EmbedMultipleConfs(
        conf, 1, randomSeed=11, useExpTorsionAnglePrefs=False, useBasicKnowledge=False
    )
    return mol.GetNumAtoms(), etk, conf.GetConformer().GetPositions().reshape(-1)


def _only(params, family):
    """The ETK parameters with every term family but one emptied."""
    kw = {
        f.name: getattr(params, f.name)[:0]
        for f in dataclasses.fields(params)
        if f.name != "n_atoms" and f.name.split("_")[0] != family
    }
    return dataclasses.replace(params, **kw)


# The flat-bottom distance terms were right and have kinks at their bounds,
# where central differences are not a fair reference.
@pytest.mark.parametrize("family", ["torsion", "improper"])
def test_etk_term_gradient_matches_finite_differences(caffeine, family):
    n_atoms, etk, x = caffeine
    fn, *_ = make_etk_energy_grad(
        batch_etk_params([_only(etk, family)], np.array([0, n_atoms], dtype=np.int32))
    )
    g = np.array(fn(mx.array(x.astype(np.float32)))[1]).astype(np.float64)
    g_fd = _fd_gradient(_energy_fn(fn), x)
    assert np.linalg.norm(g_fd) > 1.0  # the term is active here
    # float32 central differences resolve these to ~1e-5 relative.
    np.testing.assert_allclose(g, g_fd, atol=1e-3 * np.abs(g_fd).max())



@pytest.mark.parametrize(
    "V, signs",
    [
        ([0, 0, 0, 0, 1.0, 0], [1] * 6),
        ([0, 0, 0, 0, 0, 1.0], [1] * 6),
        ([2.0, 0.5, 1.0, 0.7, 0.3, 0.8], [1, -1, 1, -1, -1, 1]),
    ],
    ids=["5-fold", "6-fold", "all-six-mixed-signs"],
)
def test_etk_high_fold_torsion_gradients(caffeine, V, signs):
    """RDKit's ETK torsion gradient (CrystalFF TorsionAngleM6 and
    TorsionAngleContribs, every release through 2026.09.1) takes the 6-fold
    term with the 5-fold constants and sin(6x)/sin(x) ending in 6 instead of
    6 cos(x): rdkit/rdkit#8715, fix in #9699. Caffeine's real torsions have no
    5- or 6-fold terms, so force them here; this kernel must not share it."""
    n_atoms, etk, x = caffeine
    torsions = _only(etk, "torsion")
    n = len(torsions.torsion_V)
    torsions = dataclasses.replace(
        torsions,
        torsion_V=np.tile(np.asarray(V, np.float32), (n, 1)),
        torsion_signs=np.tile(np.asarray(signs, np.int32), (n, 1)),
    )
    fn, *_ = make_etk_energy_grad(
        batch_etk_params([torsions], np.array([0, n_atoms], dtype=np.int32))
    )
    g = np.array(fn(mx.array(x.astype(np.float32)))[1]).astype(np.float64)
    g_fd = _fd_gradient(_energy_fn(fn), x)
    np.testing.assert_allclose(g, g_fd, atol=1e-3 * np.abs(g_fd).max())

def test_etk_minimize_batch_no_longer_stops_on_a_wrong_gradient(caffeine):
    n_atoms, etk, x = caffeine
    res = etk_minimize_batch(
        batch_etk_params([etk], np.array([0, n_atoms], dtype=np.int32)),
        x.astype(np.float32), max_iters=300,
    )
    assert float(res.energies[0]) < 1e-3  # was 126 after 2 iterations
    assert res.converged[0]


def test_dg4d_energy_includes_the_fourth_dimension_penalty():
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    system = batch_dg_params([extract_dg_params(mol, get_bounds_matrix(mol))])
    x = np.random.default_rng(0).standard_normal(mol.GetNumAtoms() * 4)
    penalty = np.sum(x[3::4] ** 2)

    energies = {}
    for w in (0.0, 1.0, 10.0):
        fn, *_ = make_dg4d_energy_grad(system, fourth_dim_weight=w)
        energy = _energy_fn(fn)
        energies[w] = energy(x)
        g = np.array(fn(mx.array(x.astype(np.float32)))[1]).astype(np.float64)
        g_fd = _fd_gradient(energy, x)
        np.testing.assert_allclose(g, g_fd, atol=1e-3 * np.abs(g_fd).max(), err_msg=f"w={w}")

    for w in (1.0, 10.0):
        np.testing.assert_allclose(energies[w] - energies[0.0], w * penalty, rtol=1e-5)
