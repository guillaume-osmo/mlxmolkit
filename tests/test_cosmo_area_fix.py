"""Tests for the xtb .cosmo segment-area repair.

No xtb binary is needed: a synthetic :class:`CosmoSegments` is built the way
``writeCosmoFile`` builds one (segment area = w r^2 with sum(w) = 4pi over the
sphere, and only points with ui > 0 emitted), so the tests exercise the real
reconstruction arithmetic rather than a mock.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mlxmolkit.xtb.cosmo_area_fix import (
    AUTOAA,
    correct_cosmo_areas,
    ddcosmo_switch,
    exposed_area_per_atom,
    exposure_factors,
)
from mlxmolkit.xtb.cosmo_sigma import CosmoSegments, parse_xtb_cosmo


def _fib(n: int) -> np.ndarray:
    i = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    theta = np.pi * (1.0 + 5.0**0.5) * i
    return np.stack(
        [np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], axis=1
    )


def _synthetic_cosmo(
    coords_ang: np.ndarray, radii_ang: np.ndarray, n_grid: int = 590
) -> CosmoSegments:
    """Build a .cosmo-shaped object exactly as xtb would write one."""
    coords_bohr = np.asarray(coords_ang) / AUTOAA
    radii = np.asarray(radii_ang, dtype=np.float64)
    unit = _fib(n_grid)
    w = 4.0 * np.pi / n_grid  # sum(w) = 4pi, as xtb's Lebedev weights do

    seg_atom, seg_xyz, seg_area = [], [], []
    for k, r in enumerate(radii):
        pts = coords_bohr[k] + (r / AUTOAA) * unit
        seg_atom.extend([k + 1] * n_grid)
        seg_xyz.append(pts)
        seg_area.extend([w * r * r] * n_grid)

    dummy = CosmoSegments(
        epsilon=float("inf"),
        fepsi=0.5,
        area=float(np.sum(seg_area)),
        volume=float("nan"),
        total_screening_charge=0.0,
        total_energy_hartree=float("nan"),
        dielectric_energy_hartree=float("nan"),
        atom_radii=radii,
        atom_coords_bohr=coords_bohr,
        atom_z=[8] * len(radii),
        segments_atom=np.asarray(seg_atom, dtype=np.intp),
        segments_xyz_bohr=np.concatenate(seg_xyz, axis=0),
        segments_charge=np.zeros(len(seg_atom)),
        segments_area=np.asarray(seg_area),
        segments_sigma=np.zeros(len(seg_atom)),
        segments_potential=np.zeros(len(seg_atom)),
        cosmo_text="",
    )

    # xtb emits only the points with ui > 0; give them a non-degenerate charge.
    ui = exposure_factors(dummy)
    keep = ui > 0.0
    rng = np.random.default_rng(20260826)
    charge = ui[keep] * rng.normal(0.0, 1e-3, size=int(keep.sum()))
    area = dummy.segments_area[keep]
    return CosmoSegments(
        epsilon=dummy.epsilon,
        fepsi=dummy.fepsi,
        area=float(area.sum()),
        volume=float("nan"),
        total_screening_charge=float(charge.sum()),
        total_energy_hartree=float("nan"),
        dielectric_energy_hartree=float("nan"),
        atom_radii=radii,
        atom_coords_bohr=coords_bohr,
        atom_z=dummy.atom_z,
        segments_atom=dummy.segments_atom[keep],
        segments_xyz_bohr=dummy.segments_xyz_bohr[keep],
        segments_charge=charge,
        segments_area=area,
        segments_sigma=charge / area,
        segments_potential=np.zeros(int(keep.sum())),
        cosmo_text="",
    )


class TestSwitchFunction:
    def test_plateaus_match_the_fortran_definition(self):
        # se = -1 => x == t; fsw is 1 below 1 - eta and 0 from 1 upwards.
        assert ddcosmo_switch(np.array([0.0, 0.5, 0.79])).tolist() == [1.0, 1.0, 1.0]
        assert ddcosmo_switch(np.array([1.0, 1.5, 4.0])).tolist() == [0.0, 0.0, 0.0]

    def test_switching_band_is_continuous_and_monotone(self):
        t = np.linspace(0.8, 1.0, 41)
        f = ddcosmo_switch(t)
        assert np.isclose(f[0], 1.0)
        assert np.isclose(f[-1], 0.0, atol=1e-12)
        assert np.all(np.diff(f) <= 1e-12)

    def test_band_lies_inside_the_neighbour_sphere(self):
        # This is the ddX#165 point: the smoothing never reaches outside t = 1,
        # so points genuinely inside a neighbour still count as partly exposed.
        assert ddcosmo_switch(np.array([0.9])).item() == pytest.approx(0.5, abs=0.1)


class TestExposedArea:
    def test_isolated_sphere_is_a_full_sphere(self):
        area = exposed_area_per_atom(np.zeros((1, 3)), np.array([1.7]), n_points=40000)
        assert area[0] == pytest.approx(4 * np.pi * 1.7**2, rel=1e-6)

    def test_two_overlapping_spheres_match_the_analytic_cap(self):
        # Equal radii r at separation d: the buried cap has height r - d/2, so
        # the buried fraction is (r - d/2) / (2r).
        r, d = 1.0, 1.0
        coords = np.array([[0.0, 0.0, 0.0], [d, 0.0, 0.0]])
        area = exposed_area_per_atom(coords, np.array([r, r]), n_points=200000)
        expected = 4 * np.pi * r**2 * (1.0 - (r - d / 2) / (2 * r))
        assert area[0] == pytest.approx(expected, rel=2e-3)
        assert area[1] == pytest.approx(expected, rel=2e-3)


class TestCorrection:
    coords = np.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0], [-0.5, 1.3, 0.0]])
    radii = np.array([1.72, 1.30, 1.30])

    def test_raw_area_column_carries_no_burial_information(self):
        """The defect, stated as a property of the file itself."""
        seg = _synthetic_cosmo(self.coords, self.radii)
        # One area value per distinct radius, regardless of how buried a
        # segment is — the signature of the missing ui factor.
        assert len(np.unique(np.round(seg.segments_area, 12))) == len(np.unique(self.radii))

    def test_ui_mode_restores_the_dropped_factor_only(self):
        seg = _synthetic_cosmo(self.coords, self.radii)
        out = correct_cosmo_areas(seg, mode="ui", n_points=40000)
        ui = exposure_factors(seg)
        np.testing.assert_allclose(out.segments_area, seg.segments_area * ui, rtol=1e-12)
        assert out.area_ui < out.area_raw
        np.testing.assert_allclose(out.atom_scale, 1.0)

    def test_calibrated_mode_reproduces_the_exact_cavity_area(self):
        seg = _synthetic_cosmo(self.coords, self.radii)
        out = correct_cosmo_areas(seg, mode="calibrated", n_points=40000)
        assert out.segments_area.sum() == pytest.approx(out.area_exact, rel=1e-9)
        assert out.overcount > 1.2  # xtb was at least 20% too large here

    def test_calibrated_mode_is_exact_per_atom_not_just_in_total(self):
        seg = _synthetic_cosmo(self.coords, self.radii)
        out = correct_cosmo_areas(seg, mode="calibrated", n_points=40000)
        exact_at = exposed_area_per_atom(
            seg.atom_coords_bohr * AUTOAA, seg.atom_radii, n_points=40000
        )
        owner = seg.segments_atom - 1
        for j in range(len(seg.atom_radii)):
            got = out.segments_area[owner == j].sum()
            assert got == pytest.approx(exact_at[j], rel=1e-9)

    def test_charge_is_never_touched(self):
        """A hard-indicator mask strands charge; this must not."""
        seg = _synthetic_cosmo(self.coords, self.radii)
        out = correct_cosmo_areas(seg, mode="calibrated", n_points=40000)
        assert out.charge_conserved
        finite = out.segments_area > 1e-12
        assert finite.all(), "every charged segment must keep a finite area"
        assert seg.segments_charge.sum() == pytest.approx(
            (out.segments_sigma * out.segments_area).sum(), abs=1e-12
        )

    def test_sigma_is_inflated_and_reshaped_not_merely_rescaled(self):
        seg = _synthetic_cosmo(self.coords, self.radii)
        out = correct_cosmo_areas(seg, mode="calibrated", n_points=40000)
        rms_raw = np.sqrt(np.average(seg.segments_sigma**2, weights=seg.segments_area))
        rms_fix = np.sqrt(np.average(out.segments_sigma**2, weights=out.segments_area))
        assert rms_fix > rms_raw
        # Per-atom factors differ, so the map from raw to corrected sigma is not
        # a single scalar — that is why the profile shape changes.
        ratio = out.segments_sigma / seg.segments_sigma
        assert ratio.std() > 1e-3

    def test_atom_scale_is_below_one_for_every_atom(self):
        # Restoring ui overshoots (ddCOSMO smoothing), so calibration always
        # shrinks the area further.
        seg = _synthetic_cosmo(self.coords, self.radii)
        out = correct_cosmo_areas(seg, mode="calibrated", n_points=40000)
        assert np.all(out.atom_scale < 1.0)

    def test_rejects_an_unknown_mode(self):
        seg = _synthetic_cosmo(self.coords, self.radii)
        with pytest.raises(ValueError, match="mode must be"):
            correct_cosmo_areas(seg, mode="hard")


class TestRealXtbFile:
    """Regression on a real ``xtb --gfn 2 --tmcosmo infinity`` water file.

    Recorded with xtb 6.7.1 (conda). The numbers below are the defect itself,
    so they are asserted, not tolerated.
    """

    path = Path(__file__).parent / "data" / "water_gfn2_tmcosmo_inf.cosmo"

    def test_area_column_has_only_lebedev_weight_classes(self):
        seg = parse_xtb_cosmo(self.path)
        # 228 segments, 2 distinct radii (O, H) — 12 distinct area values means
        # the column is w(g) * r^2 and nothing else.
        assert len(seg.segments_area) == 228
        assert len(np.unique(np.round(seg.segments_area, 9))) == 12

    def test_xtb_overcounts_the_cavity_area(self):
        seg = parse_xtb_cosmo(self.path)
        out = correct_cosmo_areas(seg, mode="calibrated", n_points=200000)
        assert out.area_raw == pytest.approx(60.2295, rel=1e-4)
        assert out.area_exact == pytest.approx(44.5, rel=0.01)
        assert out.overcount == pytest.approx(1.352, rel=0.01)
        # Restoring ui alone leaves ddCOSMO's smoothing excess behind.
        assert out.area_ui / out.area_exact == pytest.approx(1.186, rel=0.01)

    def test_correction_conserves_charge_and_lifts_sigma(self):
        seg = parse_xtb_cosmo(self.path)
        out = correct_cosmo_areas(seg, mode="calibrated", n_points=200000)
        assert out.charge_conserved
        rms_raw = np.sqrt(np.average(seg.segments_sigma**2, weights=seg.segments_area))
        rms_fix = np.sqrt(np.average(out.segments_sigma**2, weights=out.segments_area))
        assert rms_fix / rms_raw == pytest.approx(1.39, rel=0.05)
