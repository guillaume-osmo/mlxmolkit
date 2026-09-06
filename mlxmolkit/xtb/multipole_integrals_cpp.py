# Copyright (c) 2026 Guillaume
# SPDX-License-Identifier: MIT

"""Optional native float64 CPU multipole integral kernel."""

from __future__ import annotations

import numpy as np

from .basis import BasisFunction

try:
    from . import _multipole_cpp
except Exception:  # pragma: no cover - depends on local extension build
    _multipole_cpp = None


CPP_AVAILABLE = _multipole_cpp is not None

_NOT_BUILT = (
    "mlxmolkit.xtb._multipole_cpp is not built. Run "
    "`python setup.py build_ext --inplace` first."
)


def multipole_matrices_cpp(
    basis: list[BasisFunction],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute CAO multipole matrices with the optional C++ kernel."""

    if _multipole_cpp is None:
        raise ImportError(
            "mlxmolkit.xtb._multipole_cpp is not built. Run "
            "`python setup.py build_ext --inplace` in the osmo env."
        )

    centers, lxyz, offsets, alphas, coeffs = _basis_arrays(basis)
    return _multipole_cpp.multipole_matrices_from_arrays(
        centers,
        lxyz,
        offsets,
        alphas,
        coeffs,
    )


def overlap_matrix_cpp(basis):
    """The contracted CAO overlap alone, from the C++ kernel.

    WARNING: `multipole_matrices_cpp(basis)[0]` returns the same matrix -- and
    pays for three dipole and six quadrupole matrices to do it, plus the two
    extra rows of the 1-D recurrence that only the quadrupoles need. The two
    are bit-identical; it is the caller that has to pick the cheaper one.
    """
    if _multipole_cpp is None:
        raise ImportError(_NOT_BUILT)
    centers, lxyz, offsets, alphas, coeffs = _basis_arrays(basis)
    return _multipole_cpp.overlap_from_arrays(
        centers, lxyz, offsets, alphas, coeffs
    )


def overlap_coeff_vjp_cpp(basis, gbar):
    """Adjoint of the contracted overlap w.r.t. the contraction coefficients.

    `gbar` is the incoming gradient on S, shape (n, n); the result is one
    number per PRIMITIVE, in the order `_basis_arrays` flattens them.

    This is the derivative `overlap_gradient_cpp` does not provide: that one
    differentiates the two CENTRES. In g-xTB the q-vSZP contraction depends on
    the EEQ-BC charges, which depend on the geometry, so a force built from
    positions alone is wrong by tens of percent while the energy stays exact.
    """
    if _multipole_cpp is None:
        raise ImportError(_NOT_BUILT)
    centers, lxyz, offsets, alphas, coeffs = _basis_arrays(basis)
    return _multipole_cpp.overlap_coeff_vjp_from_arrays(
        centers, lxyz, offsets, alphas, coeffs, np.ascontiguousarray(gbar, dtype=np.float64)
    )


def multipole_gradient_cpp(
    basis: list[BasisFunction],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute CAO multipole derivative tensors with the C++ kernel."""

    if _multipole_cpp is None:
        raise ImportError(
            "mlxmolkit.xtb._multipole_cpp is not built. Run "
            "`python setup.py build_ext --inplace` in the osmo env."
        )

    centers, lxyz, offsets, alphas, coeffs = _basis_arrays(basis)
    return _multipole_cpp.multipole_gradients_from_arrays(
        centers,
        lxyz,
        offsets,
        alphas,
        coeffs,
    )


def mmompop_chain_gradient_cpp(
    P: np.ndarray,
    S: np.ndarray,
    dpint: np.ndarray,
    qpint: np.ndarray,
    aoat: np.ndarray,
    coords_bohr: np.ndarray,
    dSA: np.ndarray,
    dSB: np.ndarray,
    dDA: np.ndarray,
    dDB: np.ndarray,
    dQA: np.ndarray,
    dQB: np.ndarray,
    dE_dip: np.ndarray,
    dE_qp: np.ndarray,
) -> np.ndarray:
    """Contract Mulliken multipole derivatives into an AES gradient."""

    if _multipole_cpp is None:
        raise ImportError(
            "mlxmolkit.xtb._multipole_cpp is not built. Run "
            "`python setup.py build_ext --inplace` in the osmo env."
        )

    return _multipole_cpp.mmompop_chain_gradient(
        np.asarray(P, dtype=np.float64),
        np.asarray(S, dtype=np.float64),
        np.asarray(dpint, dtype=np.float64),
        np.asarray(qpint, dtype=np.float64),
        np.asarray(aoat, dtype=np.intp),
        np.asarray(coords_bohr, dtype=np.float64),
        np.asarray(dSA, dtype=np.float64),
        np.asarray(dSB, dtype=np.float64),
        np.asarray(dDA, dtype=np.float64),
        np.asarray(dDB, dtype=np.float64),
        np.asarray(dQA, dtype=np.float64),
        np.asarray(dQB, dtype=np.float64),
        np.asarray(dE_dip, dtype=np.float64),
        np.asarray(dE_qp, dtype=np.float64),
    )


def mmompop_cpp(
    P: np.ndarray,
    S: np.ndarray,
    dpint: np.ndarray,
    qpint: np.ndarray,
    aoat: np.ndarray,
    coords_bohr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Mulliken cumulative atomic multipoles with the C++ kernel."""

    if _multipole_cpp is None:
        raise ImportError(
            "mlxmolkit.xtb._multipole_cpp is not built. Run "
            "`python setup.py build_ext --inplace` in the osmo env."
        )

    return _multipole_cpp.mmompop_from_arrays(
        np.asarray(P, dtype=np.float64),
        np.asarray(S, dtype=np.float64),
        np.asarray(dpint, dtype=np.float64),
        np.asarray(qpint, dtype=np.float64),
        np.asarray(aoat, dtype=np.intp),
        np.asarray(coords_bohr, dtype=np.float64),
    )


def overlap_gradient_cpp(
    basis: list[BasisFunction],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute CAO overlap derivative tensors with the C++ kernel."""

    if _multipole_cpp is None:
        raise ImportError(
            "mlxmolkit.xtb._multipole_cpp is not built. Run "
            "`python setup.py build_ext --inplace` in the osmo env."
        )

    centers, lxyz, offsets, alphas, coeffs = _basis_arrays(basis)
    return _multipole_cpp.overlap_gradients_from_arrays(
        centers,
        lxyz,
        offsets,
        alphas,
        coeffs,
    )


def _basis_arrays(
    basis: list[BasisFunction],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(basis)
    centers = np.empty((n, 3), dtype=np.float64)
    lxyz = np.empty((n, 3), dtype=np.int32)
    offsets = np.empty(n + 1, dtype=np.intp)
    offsets[0] = 0
    total_prims = 0
    for i, bf in enumerate(basis):
        centers[i] = bf.center
        lxyz[i] = bf.l_xyz
        total_prims += len(bf.alphas)
        offsets[i + 1] = total_prims

    alphas = np.empty(total_prims, dtype=np.float64)
    coeffs = np.empty(total_prims, dtype=np.float64)
    for i, bf in enumerate(basis):
        start = offsets[i]
        stop = offsets[i + 1]
        alphas[start:stop] = bf.alphas
        coeffs[start:stop] = bf.coeffs
    return centers, lxyz, offsets, alphas, coeffs
