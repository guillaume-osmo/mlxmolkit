"""
Sigma profile generation and averaging for COSMO-RS.

Port of openCOSMO-RS molecules.py sigma averaging algorithm.

Steps:
1. Gaussian-weight averaging of raw sigma over local region (r_av = 0.5 Å)
2. Classify segments by (sigma, element, HB type)
3. Bin into sigma profile histogram
"""
from __future__ import annotations

import numpy as np
from .params import (R_AV, SIGMA_GRID, SIGMA_GRID_STEP,
                     HB_DONOR_ELEMENTS, HB_ACCEPTOR_ELEMENTS)


def average_sigma(
    seg_pos: np.ndarray,
    seg_area: np.ndarray,
    seg_sigma_raw: np.ndarray,
    r_av: float = R_AV,
) -> np.ndarray:
    """Gaussian-weighted averaging of sigma charges.

    σ_av(i) = Σ_j σ_raw(j) · A_j · w(i,j) / Σ_j A_j · w(i,j)
    where w(i,j) = r_av² / (r_seg_i² + r_av²) · exp(-d²_ij / (r_seg_i² + r_av²))

    Matches openCOSMO-RS molecules.py calculate_averaged_sigmas.
    """
    n_seg = len(seg_pos)
    r_av_sq = r_av * r_av

    # Effective segment radius from area: A = π·r²  →  r² = A/π
    r_seg_sq = seg_area / np.pi

    # 1/(r_seg² + r_av²) for each segment
    inv_rad = 1.0 / (r_seg_sq + r_av_sq)

    # Pairwise squared distances
    diff = seg_pos[:, np.newaxis, :] - seg_pos[np.newaxis, :, :]  # (n, n, 3)
    dist_sq = np.sum(diff * diff, axis=2)  # (n, n)

    # Gaussian weights: exp(-d² / (r_seg² + r_av²))
    # Note: use inv_rad of target segment i (row)
    exp_arr = np.exp(-dist_sq * inv_rad[:, np.newaxis])  # (n, n)

    # Weight matrix: r_av² · inv_rad · exp
    weight = r_av_sq * inv_rad[:, np.newaxis] * exp_arr  # (n, n)

    # Weighted average
    numerator = (seg_sigma_raw * r_seg_sq) @ weight.T  # (n,)
    denominator = r_seg_sq @ weight.T  # (n,)

    sigma_av = numerator / (denominator + 1e-30)
    return sigma_av


def compute_sigma_profile(
    seg_area: np.ndarray,
    seg_sigma: np.ndarray,
    sigma_grid: np.ndarray = SIGMA_GRID,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute sigma profile p(σ) = histogram of segment areas by sigma.

    Args:
        seg_area: (n_seg,) areas in Angstrom²
        seg_sigma: (n_seg,) averaged sigma in e/Angstrom²
        sigma_grid: grid of sigma values for binning

    Returns:
        sigma_grid: (n_bins,) sigma values
        profile: (n_bins,) area density p(σ) in Angstrom²
    """
    n_bins = len(sigma_grid)
    profile = np.zeros(n_bins)

    # Linear interpolation into nearest bins
    for k in range(len(seg_sigma)):
        s = seg_sigma[k]
        a = seg_area[k]

        # Find position in grid
        idx_float = (s - sigma_grid[0]) / SIGMA_GRID_STEP
        idx = int(np.floor(idx_float))

        if idx < 0:
            profile[0] += a
        elif idx >= n_bins - 1:
            profile[-1] += a
        else:
            frac = idx_float - idx
            profile[idx] += a * (1.0 - frac)
            profile[idx + 1] += a * frac

    return sigma_grid, profile


def classify_segments(
    atoms: list[int],
    seg_sigma: np.ndarray,
    seg_atom: np.ndarray,
    adjacency: np.ndarray = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Classify segments as HB donor, HB acceptor, or non-HB.

    Args:
        atoms: atomic numbers
        seg_sigma: averaged sigma values
        seg_atom: atom index per segment
        adjacency: (n_atoms, n_atoms) bond connectivity (optional)

    Returns:
        seg_hb_type: (n_seg,) 0=non-HB, 1=HB-donor, 2=HB-acceptor
        seg_element: (n_seg,) element number (H bonded to O → 108 convention)
    """
    n_seg = len(seg_sigma)
    seg_hb_type = np.zeros(n_seg, dtype=np.int32)
    seg_element = np.array([atoms[seg_atom[i]] for i in range(n_seg)], dtype=np.int32)

    for i in range(n_seg):
        z = atoms[seg_atom[i]]

        if z in HB_DONOR_ELEMENTS and seg_sigma[i] < 0:
            # H atom with negative sigma → HB donor
            seg_hb_type[i] = 1
            # Modify element number: H bonded to O → 108, to N → 107
            if adjacency is not None:
                atom_idx = seg_atom[i]
                for j in range(len(atoms)):
                    if adjacency[atom_idx, j] > 0 and atoms[j] in HB_ACCEPTOR_ELEMENTS:
                        seg_element[i] = 100 + atoms[j]
                        break

        elif z in HB_ACCEPTOR_ELEMENTS and seg_sigma[i] > 0:
            # Electronegative atom with positive sigma → HB acceptor
            seg_hb_type[i] = 2

    return seg_hb_type, seg_element


def full_sigma_analysis(cosmo_result: dict, atoms: list[int]) -> dict:
    """Complete sigma profile analysis from COSMO surface data.

    Args:
        cosmo_result: output from cavity.cosmo_surface()
        atoms: atomic numbers

    Returns:
        dict with sigma profile, HB classification, and statistics
    """
    seg_pos = cosmo_result['seg_pos']
    seg_area = cosmo_result['seg_area']
    seg_sigma_raw = cosmo_result['seg_sigma']
    seg_atom = cosmo_result['seg_atom']

    # Average sigma
    seg_sigma_av = average_sigma(seg_pos, seg_area, seg_sigma_raw)

    # Sigma profile
    sigma_grid, profile = compute_sigma_profile(seg_area, seg_sigma_av)

    # HB classification
    seg_hb_type, seg_element = classify_segments(atoms, seg_sigma_av, seg_atom)

    # Sigma moments
    total_area = np.sum(seg_area)
    p_norm = profile / (total_area + 1e-30)
    sigma_moment_0 = total_area
    sigma_moment_1 = np.sum(sigma_grid * p_norm) * total_area  # dipole
    sigma_moment_2 = np.sum(sigma_grid**2 * p_norm) * total_area  # variance

    return {
        'sigma_grid': sigma_grid,
        'sigma_profile': profile,
        'seg_sigma_av': seg_sigma_av,
        'seg_hb_type': seg_hb_type,
        'seg_element': seg_element,
        'total_area': total_area,
        'sigma_moment_0': sigma_moment_0,
        'sigma_moment_1': sigma_moment_1,
        'sigma_moment_2': sigma_moment_2,
    }
