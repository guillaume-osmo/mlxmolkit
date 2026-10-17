# Copyright (c) 2026 Guillaume
# SPDX-License-Identifier: MIT

"""Repair the segment areas — and therefore σ — in an ``xtb`` ``.cosmo`` file.

``xtb`` writes a *formally* correct TURBOMOLE ``.cosmo`` file whose segment
areas are wrong. In ``xtb/src/solv/cosmo.f90::writeCosmoFile``::

    zeta(ii) = w(ig) * ui(ig, iat) * dot_product(basis(:, ig), s(:, iat))
    area(ii) = w(ig) * autoaa**2  * self%rvdw(iat)**2

The charge carries the ddCOSMO exposure factor ``ui`` — the fraction of the
Lebedev patch that is not buried inside a neighbouring sphere — and the area
does not. Every partially buried grid point is therefore billed at its *full*
patch area. Two consequences:

* the cavity area is too large (measured 1.35–1.92x, mean 1.71x over a
  7-molecule set: water, methanol, acetic acid, phenol, vanillin, octane,
  hedione), so anything that consumes areas — Klamt averaging, the
  combinatorial term, the misfit energy — is scaled wrong; and
* ``σ = charge/area`` becomes ``ui · σ_true``, i.e. σ is deflated *precisely
  in the sphere-overlap band*, which is where polar-hydrogen donor σ lives.
  That distorts the σ-profile **shape**, not just its amplitude.

A tell that needs no reference calculation: the ``area`` column of a ``.cosmo``
file takes only a handful of distinct values (Lebedev weight classes x distinct
radii — 12 values across 232 segments for water). A physical segment area has
to vary continuously with burial.

Reconstructing ``ui`` from the file is *exact*, not approximate: ``mkfiui``
accumulates ``fsw(|p - R_j| / rvdw_j)`` over a neighbour list, and ``fsw`` is
identically zero for ``t >= 1``, so summing over every other atom gives the
same number as xtb's cutoff list.

Restoring ``ui`` recovers roughly 43% of the excess area (1.71x -> 1.41x). The
remainder is ddCOSMO's own smoothing convention (``se = -1``, ``eta = 0.2``):
``fsw`` still counts points *inside* a neighbour sphere (0.8 <= t < 1) as
partly exposed. That is upstream ddX#165, not a bookkeeping slip, and a hard
indicator recovers the exact union-of-spheres area to 0.1%.

A hard indicator is nevertheless the wrong fix here: it strands 22–52% of the
screening charge on segments it declares buried. :func:`correct_cosmo_areas`
therefore keeps every xtb charge untouched and rescales ``w r^2 ui`` **per
atom** so that each atom's segment areas sum to its exact exposed area. Charge
is conserved exactly, the total area matches the union-of-spheres ground truth,
and every charged segment keeps a finite σ.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mlxmolkit.xtb.cosmo_sigma import CosmoSegments

# xtb/src/mctc/convert.f90
AUTOAA = 0.52917726

# xtb/src/solv/ddcosmo/core.f90: `se` is a module parameter, `eta` comes from
# TDomainDecompositionInput, which cosmo.f90::initCosmo fills with 0.2.
DDCOSMO_SE = -1.0
DDCOSMO_ETA = 0.2


def ddcosmo_switch(t: np.ndarray, se: float = DDCOSMO_SE, eta: float = DDCOSMO_ETA) -> np.ndarray:
    """ddCOSMO's 5th-degree switching function ``fsw`` (core.f90).

    ``se`` shifts the switching band: -1 interior, 0 centered, +1 exterior.
    xtb uses -1, so the band is ``0.8 <= t < 1.0`` — entirely *inside* the
    neighbouring sphere.
    """
    t = np.asarray(t, dtype=np.float64)
    x = t - (se + 1.0) * eta / 2.0
    flow = 1.0 - eta
    a = 15.0 * eta - 12.0
    b = 10.0 * eta * eta - 15.0 * eta + 6.0
    poly = ((x - 1.0) ** 2 * (1.0 - x) * (6.0 * x * x + a * x + b)) / eta**5
    return np.where(x >= 1.0, 0.0, np.where(x <= flow, 1.0, poly))


def exposure_factors(seg: CosmoSegments, eta: float = DDCOSMO_ETA) -> np.ndarray:
    """Rebuild the per-segment ddCOSMO exposure factor ``ui`` that xtb dropped.

    Exactly ``mkfiui``: ``fi = sum_j fsw(|p - R_j| / rvdw_j)`` over neighbours,
    then ``ui = 1 - fi`` where ``fi <= 1`` and 0 otherwise.
    """
    pos = np.asarray(seg.segments_xyz_bohr, dtype=np.float64)  # Bohr
    at_xyz = np.asarray(seg.atom_coords_bohr, dtype=np.float64)  # Bohr
    r_bohr = np.asarray(seg.atom_radii, dtype=np.float64) / AUTOAA
    owner = np.asarray(seg.segments_atom, dtype=np.intp) - 1  # to 0-based

    fi = np.zeros(pos.shape[0], dtype=np.float64)
    for j in range(at_xyz.shape[0]):
        t = np.linalg.norm(pos - at_xyz[j], axis=1) / r_bohr[j]
        contrib = ddcosmo_switch(t, eta=eta)
        contrib[owner == j] = 0.0  # a sphere is not its own neighbour
        fi += contrib
    return np.where(fi <= 1.0, 1.0 - fi, 0.0)


def exposed_area_per_atom(
    coords_ang: np.ndarray,
    radii_ang: np.ndarray,
    *,
    n_points: int = 20000,
) -> np.ndarray:
    """Exact exposed area of each sphere in a union of spheres, in Å².

    This is the *probe-radius-zero* surface — the union of the scaled van der
    Waals spheres — because that is the cavity ddCOSMO actually builds. The
    solvent-excluded helpers in :mod:`mlxmolkit.surface` and
    :mod:`mlxmolkit.connolly` roll a probe and so describe a different surface.

    Deterministic Fibonacci quadrature. Standard error on a per-atom exposed
    fraction is ``sqrt(p(1-p)/n)``, i.e. ~0.35% at the default 20k points.
    """
    coords = np.asarray(coords_ang, dtype=np.float64)
    radii = np.asarray(radii_ang, dtype=np.float64)

    i = np.arange(n_points) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n_points)
    theta = np.pi * (1.0 + 5.0**0.5) * i
    unit = np.stack(
        [np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], axis=1
    )

    out = np.zeros(radii.size, dtype=np.float64)
    for k in range(radii.size):
        pts = coords[k] + radii[k] * unit
        # (n_points, n_atoms) distances, then "inside any other sphere?"
        d = np.linalg.norm(pts[:, None, :] - coords[None, :, :], axis=2)
        inside = d < radii[None, :]
        inside[:, k] = False
        exposed = 1.0 - inside.any(axis=1).mean()
        out[k] = 4.0 * np.pi * radii[k] ** 2 * exposed
    return out


@dataclass
class CorrectedCosmo:
    """Corrected areas and σ, plus the diagnostics that justify the change."""

    ui: np.ndarray  # (n_seg,) reconstructed ddCOSMO exposure factor
    segments_area: np.ndarray  # (n_seg,) Å², corrected
    segments_sigma: np.ndarray  # (n_seg,) e/Å², corrected
    atom_scale: np.ndarray  # (n_atoms,) per-atom calibration factor k
    area_raw: float  # Å², as xtb reported
    area_ui: float  # Å², ui restored, ddCOSMO smoothing kept
    area_exact: float  # Å², union-of-spheres ground truth
    charge_conserved: bool
    mode: str

    @property
    def overcount(self) -> float:
        """How much too large xtb's own cavity area was."""
        return self.area_raw / self.area_exact

    def summary(self) -> str:
        return (
            f"area xtb {self.area_raw:.2f} Å² / +ui {self.area_ui:.2f} / "
            f"exact {self.area_exact:.2f} → over-count {self.overcount:.3f}x; "
            f"mode={self.mode}, charge conserved={self.charge_conserved}"
        )


def correct_cosmo_areas(
    seg: CosmoSegments,
    *,
    mode: str = "calibrated",
    n_points: int = 20000,
    eta: float = DDCOSMO_ETA,
) -> CorrectedCosmo:
    """Correct the segment areas of a parsed ``.cosmo`` file, and with them σ.

    Modes:

    ``"ui"``
        Restore only the factor xtb dropped: ``area = w r^2 ui``. Faithful to
        ddCOSMO's own convention, still ~1.41x above the true cavity area.
    ``"calibrated"`` (default)
        As ``"ui"``, then rescale each atom's segments so they sum to that
        atom's exact exposed area. Total and per-atom areas become exact;
        charges are untouched, so σ inherits the whole correction.

    Charges are never modified: ``sum(charge)`` is conserved by construction,
    which is exactly what a hard-indicator mask fails to do.
    """
    if mode not in ("ui", "calibrated"):
        raise ValueError(f"mode must be 'ui' or 'calibrated', got {mode!r}")

    ui = exposure_factors(seg, eta=eta)
    area_raw_seg = np.asarray(seg.segments_area, dtype=np.float64)
    charge = np.asarray(seg.segments_charge, dtype=np.float64)
    owner = np.asarray(seg.segments_atom, dtype=np.intp) - 1

    area_ui_seg = area_raw_seg * ui
    exact_at = exposed_area_per_atom(
        np.asarray(seg.atom_coords_bohr) * AUTOAA,
        np.asarray(seg.atom_radii),
        n_points=n_points,
    )

    scale = np.ones(exact_at.size, dtype=np.float64)
    if mode == "calibrated":
        for j in range(exact_at.size):
            s = area_ui_seg[owner == j].sum()
            if s > 1e-12:
                scale[j] = exact_at[j] / s
    area_fix_seg = area_ui_seg * scale[owner]

    sigma = np.zeros_like(charge)
    np.divide(charge, area_fix_seg, out=sigma, where=area_fix_seg > 1e-12)

    return CorrectedCosmo(
        ui=ui,
        segments_area=area_fix_seg,
        segments_sigma=sigma,
        atom_scale=scale,
        area_raw=float(area_raw_seg.sum()),
        area_ui=float(area_ui_seg.sum()),
        area_exact=float(exact_at.sum()),
        charge_conserved=bool(
            np.isclose(charge[area_fix_seg > 1e-12].sum(), charge.sum(), atol=1e-10)
        ),
        mode=mode,
    )
