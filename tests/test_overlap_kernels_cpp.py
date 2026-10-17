"""The two overlap kernels added to `_multipole_cpp`.

⚠️ `multipole_matrices_cpp(basis)[0]` already returns the contracted overlap,
and pays for three dipole and six quadrupole matrices plus two extra rows of
the 1-D recurrence to do it. `overlap_matrix_cpp` is the same arithmetic
truncated to order zero -- `S[i][j]` depends only on rows <= i, so the
truncation cannot change a bit.

⚠️ `overlap_coeff_vjp_cpp` is the derivative `overlap_gradient_cpp` does NOT
provide. That one differentiates the two CENTRES; in g-xTB the q-vSZP
contraction depends on the EEQ-BC charges, which depend on the geometry, so a
force assembled from positions alone is wrong by tens of percent while the
energy stays exact. This was measured on the Torch port: 2.5e-03 Ha/A against
a largest component of 9.1e-03.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from mlxmolkit.xtb.gxtb_basis import build_gxtb_qvszp_basis
from mlxmolkit.xtb.multipole_integrals_cpp import (
    CPP_AVAILABLE,
    _basis_arrays,
    multipole_matrices_cpp,
    overlap_coeff_vjp_cpp,
    overlap_matrix_cpp,
)

pytestmark = pytest.mark.skipif(not CPP_AVAILABLE, reason="_multipole_cpp not built")

#: Chosen by ELEMENT, not by size: sulfur and chlorine are the ones carrying d
#: shells, and a C/H/O-only fixture would exercise s and p alone.
MOLECULES = {
    "water": ([8, 1, 1], [[0.0, 0.0, 0.117], [0.0, 0.755, -0.471],
                          [0.0, -0.755, -0.471]]),
    "h2s": ([16, 1, 1], [[0.0, 0.0, 0.1], [0.0, 0.96, -0.8], [0.0, -0.96, -0.8]]),
    "ch3cl": ([6, 17, 1, 1, 1], [[0.0, 0.0, 0.0], [0.0, 0.0, 1.78],
                                 [1.03, 0.0, -0.36], [-0.51, 0.89, -0.36],
                                 [-0.51, -0.89, -0.36]]),
}


def _cao(name):
    Z, X = MOLECULES[name]
    return list(build_gxtb_qvszp_basis(np.asarray(Z, dtype=np.intp),
                                       np.asarray(X)).cao_basis)


@pytest.mark.parametrize("name", sorted(MOLECULES))
def test_overlap_only_is_bit_identical_to_the_multipole_kernel(name):
    cao = _cao(name)
    full = np.asarray(multipole_matrices_cpp(cao)[0])
    only = np.asarray(overlap_matrix_cpp(cao))
    assert np.array_equal(full.view(np.int64), only.view(np.int64)), name


@pytest.mark.parametrize("name", sorted(MOLECULES))
def test_coefficient_vjp_matches_finite_differences(name):
    """⚠️ At h = 1e-03 the agreement is 1e-10 and it gets WORSE as h shrinks --
    that is finite-difference roundoff, so the derivative is the exact one and
    the check is what limits the digits. A single tight h would have read as a
    1e-07 error in the kernel."""
    cao = _cao(name)
    n = len(cao)
    rng = np.random.default_rng(11)
    G = rng.standard_normal((n, n))
    G = 0.5 * (G + G.T)
    gc = np.asarray(overlap_coeff_vjp_cpp(cao, G))
    _c, _l, offs, _a, coeffs = _basis_arrays(cao)
    assert gc.shape == coeffs.shape

    def loss(c):
        bfs = [dataclasses.replace(bf, coeffs=np.asarray(c[offs[i]:offs[i + 1]]))
               for i, bf in enumerate(cao)]
        return float((G * np.asarray(overlap_matrix_cpp(bfs))).sum())

    h = 1.0e-3
    for i in rng.choice(len(coeffs), size=6, replace=False):
        cp, cm = coeffs.copy(), coeffs.copy()
        cp[i] += h
        cm[i] -= h
        num = (loss(cp) - loss(cm)) / (2.0 * h)
        assert abs(num - gc[i]) <= 1.0e-7 * max(abs(num), 1.0), (name, int(i))


def test_the_vjp_is_linear_in_the_incoming_gradient():
    """A VJP must be linear in `gbar`; an accidental square or an absolute
    value would still pass a single finite-difference check at one point."""
    cao = _cao("h2s")
    n = len(cao)
    rng = np.random.default_rng(3)
    A = 0.5 * (rng.standard_normal((n, n)) + rng.standard_normal((n, n)).T)
    B = 0.5 * (rng.standard_normal((n, n)) + rng.standard_normal((n, n)).T)
    ga = np.asarray(overlap_coeff_vjp_cpp(cao, A))
    gb = np.asarray(overlap_coeff_vjp_cpp(cao, B))
    gab = np.asarray(overlap_coeff_vjp_cpp(cao, 2.0 * A - 3.0 * B))
    assert np.allclose(gab, 2.0 * ga - 3.0 * gb, atol=1e-12)


def test_the_diagonal_block_is_not_double_counted():
    """⚠️ The forward writes both `S[u][v]` and `S[v][u]` from one evaluation,
    so off the diagonal the adjoint must collect BOTH entries of `gbar` and on
    the diagonal exactly one. Feeding a gradient supported only on the
    diagonal isolates that branch."""
    cao = _cao("water")
    n = len(cao)
    G = np.zeros((n, n))
    G[2, 2] = 1.0
    gc = np.asarray(overlap_coeff_vjp_cpp(cao, G))
    _c, _l, offs, _a, coeffs = _basis_arrays(cao)

    def loss(c):
        bfs = [dataclasses.replace(bf, coeffs=np.asarray(c[offs[i]:offs[i + 1]]))
               for i, bf in enumerate(cao)]
        return float((G * np.asarray(overlap_matrix_cpp(bfs))).sum())

    h = 1.0e-3
    i = int(offs[2])
    cp, cm = coeffs.copy(), coeffs.copy()
    cp[i] += h
    cm[i] -= h
    num = (loss(cp) - loss(cm)) / (2.0 * h)
    assert abs(num - gc[i]) <= 1.0e-7 * max(abs(num), 1.0)
