#!/usr/bin/env python3
"""Quantify xtb's .cosmo segment-area defect on real files.

Usage:  xtb_cosmo_area_report.py <dir-or-file> [...]
Each argument is either an ``xtb.cosmo`` file or a directory searched for one.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from mlxmolkit.xtb.cosmo_area_fix import correct_cosmo_areas, exposure_factors
from mlxmolkit.xtb.cosmo_sigma import parse_xtb_cosmo

GRID = np.arange(-0.030, 0.0301, 0.001)  # e/Å², our 61-bin convention


def profile(sigma: np.ndarray, area: np.ndarray) -> np.ndarray:
    idx = np.clip(np.digitize(sigma, GRID) - 1, 0, len(GRID) - 1)
    out = np.zeros(len(GRID))
    np.add.at(out, idx, area)
    return out


def collect(args: list[str]) -> list[Path]:
    paths: list[Path] = []
    for a in args:
        p = Path(a)
        if p.is_dir():
            paths.extend(sorted(p.rglob("xtb.cosmo")))
        elif p.is_file():
            paths.append(p)
    return paths


def main() -> int:
    paths = collect(sys.argv[1:])
    if not paths:
        print(__doc__)
        return 1
    hdr = (
        f"{'molecule':<16}{'nat':>4}{'nseg':>6}{'A_xtb':>9}{'A_ui':>9}"
        f"{'A_exact':>9}{'xtb/ex':>8}{'ui/ex':>7}{'%ui<1':>7}"
        f"{'rms_raw':>10}{'rms_fix':>10}{'shape_r':>9}"
    )
    print(hdr)
    over = []
    for path in paths:
        seg = parse_xtb_cosmo(path)
        out = correct_cosmo_areas(seg, mode="calibrated")
        ui = out.ui
        rms_raw = np.sqrt(np.average(seg.segments_sigma**2, weights=seg.segments_area))
        rms_fix = np.sqrt(np.average(out.segments_sigma**2, weights=out.segments_area))
        pr = profile(seg.segments_sigma, seg.segments_area)
        pf = profile(out.segments_sigma, out.segments_area)
        r = np.corrcoef(pr / pr.sum(), pf / pf.sum())[0, 1]
        name = path.parent.name if path.name == "xtb.cosmo" else path.stem
        print(
            f"{name:<16}{len(seg.atom_radii):>4}{len(seg.segments_area):>6}"
            f"{out.area_raw:>9.2f}{out.area_ui:>9.2f}{out.area_exact:>9.2f}"
            f"{out.overcount:>8.3f}{out.area_ui / out.area_exact:>7.3f}"
            f"{100 * np.mean(ui < 0.999):>7.1f}{rms_raw:>10.5f}{rms_fix:>10.5f}{r:>9.4f}"
        )
        over.append(out.overcount)
    print(f"\nmean over-count over {len(over)} files: {np.mean(over):.3f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
