#!/usr/bin/env python3
"""Compare our g-xTB reconstruction's atomic charges against the real binary.

The oracle is the released g-xTB build of xtb, which is not available for
macOS-arm64 here (the shipped ``bin/xtb`` is missing from the local unpack —
only ``libxtb.dylib`` and ``cpx`` survive, and neither conda's xtb 6.7.1 nor
tblite 0.4.0 knows ``--gxtb``). It *is* installed on the Linux box ``union``
at ``~/tools/gxtb/xtb-6.7.1/bin/xtb`` (6.7.1, 26dd68d, built by the g-xTB
author). Our own implementation needs Apple MLX, so it cannot run there.

Hence a split run, with the geometry as the contract between the halves:

    prepare  build deterministic RDKit geometries, write xyz + a sha256 manifest
    remote   rsync the xyz files to the oracle host, run xtb --gxtb, pull logs
    compare  parse oracle charges, run gxtb_energy locally, tabulate the gap

``compare`` never regenerates geometry: it reads the same xyz files the oracle
saw and re-checks their hashes, so a geometry mismatch is an error rather than
a silent bias.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_HOST = "union"
DEFAULT_REMOTE_XTB = "~/tools/gxtb/xtb-6.7.1/bin/xtb"
DEFAULT_REMOTE_DIR = "~/gxtb_oracle"

# Elements the reconstruction has parameters for.
ALLOWED_Z = {1, 6, 7, 8, 9, 16, 17}

SYMBOLS = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F", 15: "P", 16: "S", 17: "Cl", 35: "Br", 53: "I"}

# Flag combinations to score. "base" is gxtb_energy's own defaults; D4Srev is
# off throughout because it needs the optional dftd4 package and cannot move a
# charge — it is an additive energy term outside the SCF.
#
# NOTE: `use_first_order_offsite` is gated behind `use_first_order` in
# scf_gxtb.gxtb_energy (`if use_first_order and use_first_order_offsite`), so
# passing it alone is a no-op. tools/gxtb_flag_sweep.py has an "+offsite" arm
# that does exactly that and therefore re-measures "base" — the first-order
# term has never actually been scored. Every combo here that wants the offsite
# piece sets both flags.
COMBOS: dict[str, dict[str, object]] = {
    "base": {},
    "+first": {"use_first_order": True},
    "+first+offsite": {"use_first_order": True, "use_first_order_offsite": True},
    "+third3": {"use_third_order": True},
    "+twobody3": {"use_twobody_third_order": True},
    "+fourth4": {"use_fourth_order": True},
    "+exchange": {"use_exchange": True},
    "+mfx": {"use_mfx_exchange": True},
    "+aes": {"use_aes": True},
    "+aniso_h0": {"use_aniso_h0": True},
    "+acp_h": {"use_acp_hamiltonian": True},
    # MFX is the single largest charge term (see the run log); stack on top of it.
    "+mfx+first": {"use_mfx_exchange": True, "use_first_order": True},
    "+mfx+first+off": {
        "use_mfx_exchange": True,
        "use_first_order": True,
        "use_first_order_offsite": True,
    },
    "+mfx+aes": {"use_mfx_exchange": True, "use_aes": True},
    "+mfx+fourth4": {"use_mfx_exchange": True, "use_fourth_order": True},
    "+mfx+aniso_h0": {"use_mfx_exchange": True, "use_aniso_h0": True},
    "+mfx+acp_h": {"use_mfx_exchange": True, "use_acp_hamiltonian": True},
    # Is the oracle-fitted carbon-p patch still earning its place once the MFX
    # term it was standing in for is switched on?
    "-cpatch": {"use_carbon_plevel_shift": False},
    "+mfx-cpatch": {"use_mfx_exchange": True, "use_carbon_plevel_shift": False},
    "all_on": {
        "use_first_order": True,
        "use_first_order_offsite": True,
        "use_third_order": True,
        "use_twobody_third_order": True,
        "use_fourth_order": True,
        "use_exchange": True,
    },
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_xyz(path: Path, atoms: list[int], coords: np.ndarray, comment: str = "") -> None:
    with path.open("w", encoding="utf-8") as fh:
        fh.write(f"{len(atoms)}\n{comment}\n")
        for z, xyz in zip(atoms, coords):
            fh.write(f"{SYMBOLS.get(int(z), str(z))} {xyz[0]: .12f} {xyz[1]: .12f} {xyz[2]: .12f}\n")


def read_xyz(path: Path) -> tuple[list[int], np.ndarray]:
    lines = path.read_text().splitlines()
    n = int(lines[0].split()[0])
    rev = {v: k for k, v in SYMBOLS.items()}
    atoms, coords = [], np.zeros((n, 3))
    for i, ln in enumerate(lines[2 : 2 + n]):
        p = ln.split()
        atoms.append(rev[p[0]])
        coords[i] = [float(p[1]), float(p[2]), float(p[3])]
    return atoms, coords


# ---------------------------------------------------------------- prepare


def embed(smiles: str, seed: int = 42) -> tuple[list[int], np.ndarray] | None:
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    if AllChem.EmbedMolecule(mol, params) != 0:
        return None
    AllChem.MMFFOptimizeMolecule(mol, maxIters=2000)
    conf = mol.GetConformer()
    atoms = [a.GetAtomicNum() for a in mol.GetAtoms()]
    if not set(atoms) <= ALLOWED_Z:
        return None
    coords = np.array(
        [[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
         for i in range(mol.GetNumAtoms())]
    )
    return atoms, coords


def cmd_prepare(args: argparse.Namespace) -> int:
    out = Path(args.run_dir)
    (out / "xyz").mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(Path(args.smiles_csv).open()))
    manifest = []
    for i, row in enumerate(rows):
        if len(manifest) >= args.n:
            break
        smiles = row["smiles"]
        got = embed(smiles, seed=args.seed)
        if got is None:
            print(f"  skip (embed/elements): {smiles}")
            continue
        atoms, coords = got
        if len(atoms) > args.max_atoms:
            continue
        name = f"mol{i:04d}"
        path = out / "xyz" / f"{name}.xyz"
        write_xyz(path, atoms, coords, comment=smiles)
        manifest.append(
            {"name": name, "smiles": smiles, "n_atoms": len(atoms), "sha256": _sha256(path)}
        )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"prepared {len(manifest)} geometries in {out/'xyz'}")
    return 0


# ---------------------------------------------------------------- remote

REMOTE_SCRIPT = r"""
set -u
RUN="$1"; XTB="$2"; JOBS="$3"
cd "$RUN"
mkdir -p logs
export LD_LIBRARY_PATH="$(dirname "$(dirname "$XTB")")/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
ls xyz/*.xyz | xargs -P "$JOBS" -I{} sh -c '
  f="{}"; b=$(basename "$f" .xyz)
  if [ -s "logs/$b.log" ]; then exit 0; fi
  d=$(mktemp -d)
  cp "$f" "$d/mol.xyz"
  ( cd "$d" && '"$XTB"' mol.xyz --gxtb --acc 0.2 ) > "logs/$b.log" 2>&1
  rm -rf "$d"
'
n=$(ls logs/*.log 2>/dev/null | wc -l)
echo "logs written: $n"
grep -l "normal termination" logs/*.log 2>/dev/null | wc -l | sed 's/^/normal terminations: /'
"""


def cmd_remote(args: argparse.Namespace) -> int:
    run = Path(args.run_dir)
    remote_run = f"{args.remote_dir}/{run.name}"
    print(f"rsync -> {args.host}:{remote_run}")
    subprocess.run(["ssh", args.host, f"mkdir -p {remote_run}"], check=True)
    subprocess.run(
        ["rsync", "-a", "--delete", f"{run}/xyz/", f"{args.host}:{remote_run}/xyz/"], check=True
    )
    script = REMOTE_SCRIPT
    print(f"running xtb --gxtb on {args.host} ({args.jobs} parallel)")
    proc = subprocess.run(
        ["ssh", args.host, "bash", "-s", "--", remote_run, args.remote_xtb, str(args.jobs)],
        input=script,
        text=True,
        capture_output=True,
    )
    print(proc.stdout.strip() or proc.stderr.strip()[-2000:])
    (run / "logs").mkdir(exist_ok=True)
    subprocess.run(["rsync", "-a", f"{args.host}:{remote_run}/logs/", f"{run}/logs/"], check=True)
    got = len(list((run / "logs").glob("*.log")))
    print(f"pulled {got} logs into {run/'logs'}")
    return 0


# ---------------------------------------------------------------- compare

_CHARGE_HDR = "Atomic charges and shell populations"
_ROW = re.compile(r"^\s*(\d+)\s+(\d+)\s+([A-Za-z]+)\s+([-+0-9.Ee]+)\s+(.+)$")


def parse_oracle_charges(log: str) -> tuple[list[int], np.ndarray] | None:
    lines = log.splitlines()
    try:
        start = next(i for i, ln in enumerate(lines) if _CHARGE_HDR in ln)
    except StopIteration:
        return None
    z, q, seen_header = [], [], False
    for ln in lines[start + 1 :]:
        s = ln.strip()
        if s.startswith("#"):
            seen_header = True
            continue
        if not seen_header:
            continue
        if s.startswith("---"):
            if q:
                break
            continue
        m = _ROW.match(ln)
        if not m:
            continue
        z.append(int(m.group(2)))
        q.append(float(m.group(4)))
    return (z, np.asarray(q)) if q else None


def cmd_compare(args: argparse.Namespace) -> int:
    from mlxmolkit.xtb.scf_gxtb import gxtb_energy

    run = Path(args.run_dir)
    manifest = {m["name"]: m for m in json.loads((run / "manifest.json").read_text())}
    combos = {k: COMBOS[k] for k in (args.combos.split(",") if args.combos else COMBOS)}

    per_ours: dict[str, list[np.ndarray]] = {k: [] for k in combos}
    per_ref: dict[str, list[np.ndarray]] = {k: [] for k in combos}
    per_combo_z: dict[str, list[np.ndarray]] = {k: [] for k in combos}
    n_ok = n_fail = 0
    rows = []
    for name, meta in sorted(manifest.items()):
        log_path = run / "logs" / f"{name}.log"
        xyz_path = run / "xyz" / f"{name}.xyz"
        if not log_path.exists():
            continue
        if _sha256(xyz_path) != meta["sha256"]:
            raise SystemExit(f"{xyz_path} changed since prepare — oracle saw other coordinates")
        parsed = parse_oracle_charges(log_path.read_text())
        if parsed is None:
            n_fail += 1
            continue
        z_ref, q_ref = parsed
        atoms, coords = read_xyz(xyz_path)
        if z_ref != atoms:
            raise SystemExit(f"{name}: oracle atom order differs from the xyz")

        line = {"name": name, "smiles": meta["smiles"], "nat": len(atoms)}
        for label, kw in combos.items():
            try:
                res = gxtb_energy(atoms, coords, use_d4srev=False, **kw)
                q = np.asarray(res["atom_charges"], dtype=float)
                if not res["converged"]:
                    line[label] = float("nan")
                    continue
            except Exception as exc:  # a reconstruction term can still blow up
                line[label] = float("nan")
                if args.verbose:
                    print(f"  {name} {label}: {type(exc).__name__}: {exc}")
                continue
            per_ours[label].append(q)
            per_ref[label].append(q_ref)
            per_combo_z[label].append(np.asarray(atoms))
            line[label] = float(np.abs(q - q_ref).mean())
        rows.append(line)
        n_ok += 1
        if args.verbose:
            print(f"  {name} {meta['smiles']}")

    print(f"\n{n_ok} molecules compared, {n_fail} oracle logs unusable\n")
    print(f"{'combo':<16}{'MAE':>9}{'RMSE':>9}{'max':>9}{'slope':>8}{'icept':>9}{'r':>8}{'n_at':>7}")
    best = None
    for label in combos:
        if not per_ours[label]:
            continue
        ours = np.concatenate(per_ours[label])
        ref = np.concatenate(per_ref[label])
        d = ours - ref
        mae, rmse = np.abs(d).mean(), np.sqrt((d**2).mean())
        # ours = a * oracle + b over every atom. a < 1 means we systematically
        # under-polarise; a near 1 with a large MAE means the error is
        # element-structured rather than a global scale.
        a, b = np.polyfit(ref, ours, 1)
        r = float(np.corrcoef(ref, ours)[0, 1])
        print(f"{label:<16}{mae:>9.5f}{rmse:>9.5f}{np.abs(d).max():>9.5f}"
              f"{a:>8.4f}{b:>9.5f}{r:>8.5f}{d.size:>7}")
        if best is None or mae < best[1]:
            best = (label, mae)
    if best:
        print(f"\nbest combo: {best[0]}  MAE {best[1]:.5f} e")

    # Element-resolved residual for the best combo — where the error lives.
    if best:
        label = best[0]
        d = np.concatenate(per_ours[label]) - np.concatenate(per_ref[label])
        z = np.concatenate(per_combo_z[label])
        print(f"\nper-element residual (ours - oracle), combo {label}:")
        print(f"{'Z':>4}{'sym':>5}{'n':>7}{'mean':>10}{'MAE':>10}{'sd':>10}")
        for zz in sorted(set(z.tolist())):
            m = z == zz
            print(f"{zz:>4}{SYMBOLS.get(zz, '?'):>5}{m.sum():>7}"
                  f"{d[m].mean():>10.5f}{np.abs(d[m]).mean():>10.5f}{d[m].std():>10.5f}")

    out = run / "charge_comparison.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nper-molecule MAEs -> {out}")
    return 0


def parse_oracle_shells(log: str) -> list[tuple[int, tuple[float, ...]]] | None:
    """(Z, shell populations) per atom, as xtb --gxtb prints them."""
    lines = log.splitlines()
    try:
        start = next(i for i, ln in enumerate(lines) if _CHARGE_HDR in ln)
    except StopIteration:
        return None
    out, seen_header = [], False
    for ln in lines[start + 1 :]:
        s = ln.strip()
        if s.startswith("#"):
            seen_header = True
            continue
        if not seen_header:
            continue
        if s.startswith("---"):
            if out:
                break
            continue
        m = _ROW.match(ln)
        if not m:
            continue
        out.append((int(m.group(2)), tuple(float(x) for x in m.group(5).split())))
    return out or None


def cmd_shells(args: argparse.Namespace) -> int:
    """Shell-resolved population comparison — localises the error to (element, l).

    An atomic charge sums over shells and can hide a compensating pair of shell
    errors. The oracle prints p(s)/p(p)/p(d) per atom, so compare those.
    """
    from mlxmolkit.xtb.params_gxtb import GXTB_PARAMS
    from mlxmolkit.xtb.scf_gxtb import gxtb_energy

    run = Path(args.run_dir)
    manifest = {m["name"]: m for m in json.loads((run / "manifest.json").read_text())}
    kw = COMBOS[args.combo]

    # (Z, l) -> list of (ours, oracle)
    acc: dict[tuple[int, int], list[tuple[float, float]]] = {}
    n_ok = 0
    for name, meta in sorted(manifest.items()):
        log_path = run / "logs" / f"{name}.log"
        xyz_path = run / "xyz" / f"{name}.xyz"
        if not log_path.exists():
            continue
        if _sha256(xyz_path) != meta["sha256"]:
            raise SystemExit(f"{xyz_path} changed since prepare")
        ref = parse_oracle_shells(log_path.read_text())
        if ref is None:
            continue
        atoms, coords = read_xyz(xyz_path)
        try:
            res = gxtb_energy(atoms, coords, use_d4srev=False, **kw)
        except Exception as exc:
            print(f"  {name}: {type(exc).__name__}: {exc}")
            continue
        if not res["converged"]:
            continue
        basis = res["basis"]
        qsh = np.asarray(res["shell_charges"], dtype=float)
        shell_atom = np.asarray(basis.shell_atom)
        shell_l = np.asarray(basis.shell_l)
        for ai, (z_ref, pops_ref) in enumerate(ref):
            occ = np.asarray(GXTB_PARAMS.reference_population(z_ref), dtype=float)
            for k, ish in enumerate(np.where(shell_atom == ai)[0]):
                lval = int(shell_l[ish])
                if k >= len(pops_ref):
                    continue
                ours = float(occ[lval] - qsh[ish])
                acc.setdefault((z_ref, lval), []).append((ours, float(pops_ref[k])))
        n_ok += 1

    print(f"\ncombo {args.combo}: {n_ok} molecules, shell-resolved populations\n")
    print(f"{'Z':>4}{'sym':>5}{'l':>3}{'n':>7}{'mean_ours':>11}{'mean_ref':>11}"
          f"{'mean_diff':>11}{'MAE':>10}{'sd':>10}")
    for (z, lval) in sorted(acc):
        pairs = np.asarray(acc[(z, lval)])
        d = pairs[:, 0] - pairs[:, 1]
        print(f"{z:>4}{SYMBOLS.get(z, '?'):>5}{'spdf'[lval]:>3}{len(pairs):>7}"
              f"{pairs[:, 0].mean():>11.5f}{pairs[:, 1].mean():>11.5f}"
              f"{d.mean():>11.5f}{np.abs(d).mean():>10.5f}{d.std():>10.5f}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", default="gxtb_charge_run")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare")
    p.add_argument("--smiles-csv", default=str(REPO_ROOT / "tests/data/perfumery_benchmark_100.csv"))
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--max-atoms", type=int, default=40)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("remote")
    p.add_argument("--host", default=DEFAULT_HOST)
    p.add_argument("--remote-xtb", default=DEFAULT_REMOTE_XTB)
    p.add_argument("--remote-dir", default=DEFAULT_REMOTE_DIR)
    p.add_argument("--jobs", type=int, default=8)
    p.set_defaults(func=cmd_remote)

    p = sub.add_parser("compare")
    p.add_argument("--combos", default="base")
    p.add_argument("--verbose", action="store_true")
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("shells")
    p.add_argument("--combo", default="base", choices=sorted(COMBOS))
    p.set_defaults(func=cmd_shells)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
