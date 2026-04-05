"""
ddCOSMO-inspired COSMO solver with smooth switching function.

Improvements over simple COSMO (cavity.py):
1. Smooth 5th-degree polynomial switching function (xTB convention)
   - No sharp burial cutoff → smoother sigma profiles
   - Width parameter eta=0.2 Å
2. Weighted surface charges: q(i) = ui(i) * q_raw(i)
   - Partially buried segments contribute proportionally
3. Better cavity area from weighted quadrature

Reference: Stahn et al., J. Phys. Chem. A 2023, 127, 5555-5567
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist
from .params import VDW_RADII, CAVITY_SCALING, EPSILON_WATER, BOHR_TO_ANG
from .lebedev import get_lebedev_grid


def _switching_function(t: np.ndarray, eta: float = 0.2) -> np.ndarray:
    """Smooth 5th-degree polynomial switching function (xTB convention).

    χ(t) = 1  if t <= 1-eta  (fully inside neighbor → buried)
    χ(t) = 0  if t >= 1      (fully outside → exposed)
    χ(t) = smooth transition for 1-eta < t < 1

    Args:
        t: normalized distance t = |r - r_j| / R_j
        eta: switching width (default 0.2)

    Returns:
        chi: switching function values in [0, 1]
    """
    chi = np.zeros_like(t)

    # Fully buried: t <= 1 - eta
    chi[t <= 1.0 - eta] = 1.0

    # Transition region: 1-eta < t < 1
    mask = (t > 1.0 - eta) & (t < 1.0)
    if np.any(mask):
        # Normalize to [0, 1] range
        s = (t[mask] - (1.0 - eta)) / eta
        # 5th degree: smooth C2 transition from 1 to 0
        chi[mask] = 1.0 - 10.0 * s**3 + 15.0 * s**4 - 6.0 * s**5

    return chi


def build_ddcosmo_cavity(
    atoms: list[int],
    coords: np.ndarray,
    n_points_per_atom: int = 194,
    scaling: float = CAVITY_SCALING,
    eta: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build COSMO cavity with smooth switching function.

    Returns:
        seg_pos: (n_seg, 3) segment positions in Angstrom
        seg_area: (n_seg,) segment areas (weighted by ui)
        seg_normal: (n_seg, 3) outward normals
        seg_atom: (n_seg,) atom index
        seg_ui: (n_seg,) switching function values (0=buried, 1=exposed)
    """
    coords = np.asarray(coords, dtype=np.float64)
    n_atoms = len(atoms)

    sphere_pts, sphere_weights = get_lebedev_grid(n_points_per_atom)
    n_pts = len(sphere_pts)

    radii = np.array([VDW_RADII.get(z, 2.0) * scaling for z in atoms])

    all_pos = []
    all_area = []
    all_normal = []
    all_atom_idx = []
    all_ui = []

    for i in range(n_atoms):
        r_i = radii[i]
        pts = coords[i] + r_i * sphere_pts  # (n_pts, 3)

        # Compute switching function fi(n) = Σ_j χ(t_n^j)
        fi = np.zeros(n_pts)
        for j in range(n_atoms):
            if j == i:
                continue
            # Normalized distance: t = |pt - center_j| / R_j
            dists = np.linalg.norm(pts - coords[j], axis=1)
            t = dists / radii[j]
            fi += _switching_function(t, eta)

        # Exposure indicator: ui = max(0, 1 - fi)
        ui = np.maximum(0.0, 1.0 - fi)

        # Keep segments with ui > threshold (partially or fully exposed)
        threshold = 0.001
        mask = ui > threshold

        if np.sum(mask) == 0:
            continue

        # Weighted areas: weight * r² * ui
        areas = sphere_weights[mask] * r_i * r_i * ui[mask]

        all_pos.append(pts[mask])
        all_area.append(areas)
        all_normal.append(sphere_pts[mask])
        all_atom_idx.append(np.full(np.sum(mask), i, dtype=np.int32))
        all_ui.append(ui[mask])

    if not all_pos:
        return (np.zeros((0, 3)), np.zeros(0), np.zeros((0, 3)),
                np.zeros(0, dtype=np.int32), np.zeros(0))

    return (np.vstack(all_pos), np.concatenate(all_area),
            np.vstack(all_normal), np.concatenate(all_atom_idx),
            np.concatenate(all_ui))


def _jacobi_diis_solve(
    A: np.ndarray,
    rhs: np.ndarray,
    max_iter: int = 200,
    tol: float = 1e-8,
    n_diis: int = 20,
) -> tuple[np.ndarray, int, bool]:
    """Jacobi iteration with DIIS extrapolation for A·q = rhs.

    Split A = D + O (diagonal + off-diagonal).
    Iterate: q_new = D⁻¹ · (rhs - O · q)
    DIIS: extrapolate from error history for faster convergence.

    O(n²) per iteration (matvec) vs O(n³) for dense solve.

    Args:
        A: (n, n) coefficient matrix
        rhs: (n,) right-hand side
        max_iter: maximum Jacobi iterations
        tol: convergence tolerance (rms of increment)
        n_diis: DIIS history size

    Returns:
        q: (n,) solution
        n_iter: iterations used
        converged: whether tolerance was reached
    """
    n = len(rhs)
    D_inv = 1.0 / np.diag(A)  # (n,)
    O = A.copy()
    np.fill_diagonal(O, 0.0)   # off-diagonal only

    # Initial guess: D⁻¹ · rhs
    q = D_inv * rhs

    # DIIS history
    diis_q = []
    diis_err = []

    for iteration in range(max_iter):
        # Jacobi step: q_new = D⁻¹ · (rhs - O · q)
        residual = rhs - O @ q
        q_new = D_inv * residual

        # Error = increment
        dq = q_new - q
        rms_dq = np.sqrt(np.mean(dq * dq))

        if rms_dq < tol:
            return q_new, iteration + 1, True

        # DIIS extrapolation (after a few plain Jacobi steps)
        if iteration >= 2:
            diis_q.append(q_new.copy())
            diis_err.append(dq.copy())

            if len(diis_q) > n_diis:
                diis_q.pop(0)
                diis_err.pop(0)

            nd = len(diis_q)
            if nd >= 2:
                # Build DIIS B matrix: B[i,j] = err_i · err_j
                B = np.zeros((nd + 1, nd + 1))
                for i in range(nd):
                    for j in range(i, nd):
                        B[i, j] = np.dot(diis_err[i], diis_err[j])
                        B[j, i] = B[i, j]
                B[:nd, nd] = -1.0
                B[nd, :nd] = -1.0

                rhs_diis = np.zeros(nd + 1)
                rhs_diis[nd] = -1.0

                try:
                    c = np.linalg.solve(B, rhs_diis)
                    q_new = sum(c[i] * diis_q[i] for i in range(nd))
                except np.linalg.LinAlgError:
                    pass  # Skip DIIS this step, use plain Jacobi

        # SOR-like damped update (omega=0.5 for stability with dense Coulomb)
        q = 0.5 * q_new + 0.5 * q

    return q, max_iter, False


def ddcosmo_charges(
    atoms: list[int],
    coords: np.ndarray,
    mulliken_charges: np.ndarray,
    seg_pos: np.ndarray,
    seg_area: np.ndarray,
    seg_ui: np.ndarray,
    epsilon: float = EPSILON_WATER,
    solver: str = 'auto',
) -> np.ndarray:
    """Solve COSMO equation with ddCOSMO-weighted segments.

    Args:
        solver: 'jacobi' for Jacobi/DIIS iterative,
                'direct' for np.linalg.solve,
                'auto' picks based on system size (Jacobi for n>400)
    """
    n_seg = len(seg_pos)
    if n_seg == 0:
        return np.zeros(0)

    seg_pos_b = seg_pos / BOHR_TO_ANG
    seg_area_b = seg_area / (BOHR_TO_ANG ** 2)
    coords_b = coords / BOHR_TO_ANG

    # A matrix
    dist = cdist(seg_pos_b, seg_pos_b)
    np.fill_diagonal(dist, 1.0)
    A = 1.0 / dist
    np.fill_diagonal(A, 1.07 * np.sqrt(4.0 * np.pi / np.maximum(seg_area_b, 1e-30)))

    # Phi potential
    dist_ac = np.maximum(cdist(seg_pos_b, coords_b), 1e-10)
    Phi = (mulliken_charges[np.newaxis, :] / dist_ac).sum(axis=1)

    # Weighted RHS
    rhs = -Phi * seg_ui

    # Solve
    if solver == 'auto':
        # Direct (numpy Accelerate) is faster for n < 3000 on Apple Silicon
        # Jacobi/DIIS only for very large systems where O(n³) dominates
        solver = 'jacobi' if n_seg > 3000 else 'direct'

    if solver == 'jacobi':
        q, n_iter, converged = _jacobi_diis_solve(A, rhs, max_iter=200, tol=1e-8)
    else:
        q = np.linalg.solve(A, rhs)

    keps = (epsilon - 1.0) / (epsilon + 0.5)
    return keps * q


def ddcosmo_surface(
    atoms: list[int],
    coords: np.ndarray,
    density: np.ndarray,
    n_points: int = 194,
    epsilon: float = EPSILON_WATER,
    eta: float = 0.2,
) -> dict:
    """Complete ddCOSMO surface calculation.

    Like cosmo_surface but with smooth switching function.
    """
    from .cavity import _mulliken_charges
    from ..rm1.params import RM1_PARAMS

    coords = np.asarray(coords, dtype=np.float64)

    seg_pos, seg_area, seg_normal, seg_atom, seg_ui = build_ddcosmo_cavity(
        atoms, coords, n_points_per_atom=n_points, eta=eta,
    )

    n_basis_per = [RM1_PARAMS[z].n_basis for z in atoms]
    mulliken = _mulliken_charges(atoms, density, n_basis_per)

    seg_charge = ddcosmo_charges(
        atoms, coords, mulliken, seg_pos, seg_area, seg_ui, epsilon=epsilon,
    )

    seg_sigma = np.zeros_like(seg_charge)
    nonzero = seg_area > 1e-30
    seg_sigma[nonzero] = seg_charge[nonzero] / seg_area[nonzero]

    cavity_area = np.sum(seg_area)
    r_dot_n = np.sum((seg_pos - np.mean(coords, axis=0)) * seg_normal, axis=1)
    cavity_volume = np.abs(np.sum(r_dot_n * seg_area) / 3.0)

    return {
        'seg_pos': seg_pos, 'seg_area': seg_area,
        'seg_charge': seg_charge, 'seg_sigma': seg_sigma,
        'seg_normal': seg_normal, 'seg_atom': seg_atom,
        'seg_ui': seg_ui, 'mulliken_charges': mulliken,
        'cavity_area': cavity_area, 'cavity_volume': cavity_volume,
        'n_seg': len(seg_pos),
    }
