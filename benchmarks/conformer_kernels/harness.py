"""Stage-timed runs of ``generate_conformers_nk``, optionally recording every
kernel call so another build of the kernels can be replayed on byte-identical
inputs (see ``replay.py``).

The pipeline looks its stage functions up in the ``conformer_pipeline_v2``
module namespace at call time, so wrapping those names times each GPU stage
without touching the pipeline. Every wrapped function ends in ``mx.eval`` and
``np.array``, so the wall time of a call includes its GPU work.

Which ``mlxmolkit`` is measured is decided by ``PYTHONPATH`` alone; this file
never inserts a path, so the same script times the old and the new kernels.

Example::

    PYTHONPATH=<checkout> python harness.py --set probe500 --k 3 --mmff 1 \\
        --repeats 3 --warmup 1 --out times.json
    PYTHONPATH=<old export> python harness.py --set eq200 --k 3 --mmff 1 \\
        --record calls_old.pkl
"""
from __future__ import annotations

import argparse
import copy
import json
import pickle
import statistics
import time
from collections import defaultdict
from pathlib import Path

import mlx.core as mx
import numpy as np
import pandas as pd
from rdkit import RDLogger

import mlxmolkit
from mlxmolkit import conformer_pipeline_v2 as cpv2

RDLogger.DisableLog("rdApp.*")

WORK = Path("/Users/guillaume-osmo/Github/mlxmolkit-topu/benchmarks/topu_lbvs/work")
# Compounds whose stereo is geometrically unrealisable: DG never converges on
# them, so they measure the retry budget, not the kernels.
IMPOSSIBLE = {"CHEMBL104700", "CHEMBL107360", "CHEMBL1076484"}

SMALL20 = [
    "CC(=O)Oc1ccccc1C(=O)O",
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "Cn1c(=O)c2c(ncn2C)n(c1=O)C",
    "COc1ccc2cc(ccc2c1)[C@@H](C)C(=O)O",
    "CN1C(=O)CN=C(c2ccccc21)c3ccc(Cl)cc3",
    "Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1",
    "C[C@]12CC[C@H]3[C@@H](CC=C4C[C@@H](O)CC[C@@]34C)[C@@H]1CC[C@@H]2O",
    "CC(C)NC[C@H](O)COc1cccc2ccccc12",
    "O=C(O)C[C@@H](N)C(=O)N[C@@H](Cc1ccccc1)C(=O)OC",
    "CCN(CC)CCNC(=O)c1ccc(N)cc1",
    "CC(=O)Nc1ccc(O)cc1",
    "CN1CCC[C@H]1c1cccnc1",
    "CC1(C)S[C@@H]2[C@H](NC(=O)Cc3ccccc3)C(=O)N2[C@H]1C(=O)O",
    "OC[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@@H]1O",
    "CCCCCCCCCCCCCCCC(=O)O",
    "c1ccc2c(c1)ccc1ccccc12",
    "CC(C)(C)NCC(O)c1ccc(O)c(CO)c1",
    "COc1cc2c(cc1OC)C(=O)C(CC1CCN(Cc3ccccc3)CC1)C2",
    "CN1C2CCC1C(C(=O)OC)C(OC(=O)c1ccccc1)C2",
    "CC(=O)OC1=CC=CC=C1C(=O)O",
]

STAGES = ("dg1", "dg_retry", "collapse", "etk", "mmff")


def isomer_smiles(which: str) -> list[str]:
    """Isomeric SMILES for a named set, in a fixed order."""
    if which == "small20":
        return list(SMALL20)
    st = pd.read_parquet(WORK / "structures.parquet", columns=["ID", "identity_ok", "isomers"])
    st = st.set_index("ID")
    ids = [l.strip() for l in (WORK / "ids_probe500.txt").read_text().splitlines() if l.strip()]
    st = st.loc[ids]
    st = st[st.identity_ok]
    out = []
    for cid, iso in zip(st.index, st.isomers):
        if which == "eq200" and cid in IMPOSSIBLE:
            continue
        out.extend(s for s in iso.split("|") if s)
    if which == "eq200":
        return out[:200]
    if which == "probe500":
        return out
    raise ValueError(which)


class Recorder:
    """Wraps the three kernel entry points of the pipeline."""

    def __init__(self, record: bool):
        self.record = record
        self.times: dict[str, list[float]] = defaultdict(list)
        self.calls: list[dict] = []
        self._chunk_dg_calls = 0
        self._orig = {}

    def install(self):
        for name in ("dg_minimize_shared", "etk_minimize_shared", "mmff_minimize_nk", "_process_chunk"):
            self._orig[name] = getattr(cpv2, name)
        cpv2.dg_minimize_shared = self._dg
        cpv2.etk_minimize_shared = self._etk
        cpv2.mmff_minimize_nk = self._mmff
        cpv2._process_chunk = self._chunk

    def _chunk(self, *a, **kw):
        self._chunk_dg_calls = 0
        return self._orig["_process_chunk"](*a, **kw)

    def _timed(self, stage, fn, args, kwargs):
        rec_args = copy.deepcopy(args) if self.record else None
        t = time.perf_counter()
        out = fn(*args, **kwargs)
        self.times[stage].append(time.perf_counter() - t)
        if self.record:
            self.calls.append({
                "stage": stage, "args": rec_args, "kwargs": dict(kwargs),
                "out": tuple(np.array(o, copy=True) for o in out),
            })
        return out

    def _dg(self, batch, pos, **kw):
        self._chunk_dg_calls += 1
        if kw.get("fourth_dim_weight") == 1.0 and kw.get("max_iters") == 200:
            stage = "collapse"
        elif self._chunk_dg_calls == 1:
            stage = "dg1"
        else:
            stage = "dg_retry"
        return self._timed(stage, self._orig["dg_minimize_shared"], (batch, pos), kw)

    def _etk(self, batch, pos, **kw):
        return self._timed("etk", self._orig["etk_minimize_shared"], (batch, pos), kw)

    def _mmff(self, params, counts, pos, **kw):
        return self._timed("mmff", self._orig["mmff_minimize_nk"], (params, counts, pos), kw)


def run_once(smiles, k, mmff, mcpb, rec: Recorder):
    rec.times.clear()
    t = time.perf_counter()
    res = cpv2.generate_conformers_nk(
        smiles, n_confs_per_mol=k, variant="ETKDGv3", run_mmff=bool(mmff),
        max_confs_per_batch=mcpb,
    )
    wall = time.perf_counter() - t
    stage = {s: float(sum(rec.times.get(s, []))) for s in STAGES}
    return res, wall, stage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, choices=["probe500", "eq200", "small20"])
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--mmff", type=int, default=1)
    ap.add_argument("--mcpb", type=int, default=100000, help="max_confs_per_batch (fixed so chunks match)")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--record", type=str, default=None, help="pickle every kernel call of the last repeat")
    ap.add_argument("--result", type=str, default=None, help="npz of final per-conformer pipeline outputs")
    ap.add_argument("--out", type=str, default=None, help="json timing summary")
    ap.add_argument("--label", type=str, default="")
    args = ap.parse_args()

    smiles = isomer_smiles(args.set)
    rec = Recorder(record=False)
    rec.install()
    for _ in range(args.warmup):
        run_once(smiles, args.k, args.mmff, args.mcpb, rec)

    runs = []
    res = None
    for r in range(args.repeats):
        rec.record = bool(args.record) and r == args.repeats - 1
        rec.calls.clear()
        res, wall, stage = run_once(smiles, args.k, args.mmff, args.mcpb, rec)
        runs.append({"wall": wall, **stage})
        print(json.dumps({"rep": r, "wall": round(wall, 4), **{s: round(v, 4) for s, v in stage.items()}}), flush=True)

    summary = {
        "label": args.label, "set": args.set, "k": args.k, "mmff": args.mmff, "mcpb": args.mcpb,
        "n_smiles": len(smiles), "n_conformers": res.total_conformers, "n_batches": res.n_batches,
        "mlxmolkit_file": mlxmolkit.__file__, "mlx": mx.__version__,
        "median": {key: statistics.median(r[key] for r in runs) for key in runs[0]},
        "runs": runs,
        "converged": int(sum(sum(m.converged) for m in res.molecules)),
    }
    print(json.dumps({"median": {k_: round(v, 4) for k_, v in summary["median"].items()},
                      "converged": summary["converged"], "n_conformers": summary["n_conformers"],
                      "n_batches": summary["n_batches"], "pkg": summary["mlxmolkit_file"]}), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=1))
    if args.record:
        with open(args.record, "wb") as fh:
            pickle.dump({"set": args.set, "k": args.k, "mmff": args.mmff, "mcpb": args.mcpb,
                         "smiles": smiles, "calls": rec.calls}, fh, protocol=pickle.HIGHEST_PROTOCOL)
    if args.result:
        pos, en, conv, mol = [], [], [], []
        for i, m in enumerate(res.molecules):
            for p, e, c in zip(m.positions_3d, m.energies, m.converged):
                pos.append(np.asarray(p, np.float32).ravel()); en.append(e); conv.append(c); mol.append(i)
        np.savez(args.result, pos=np.concatenate(pos), sizes=np.array([len(p) for p in pos]),
                 energy=np.array(en), converged=np.array(conv), mol=np.array(mol))


if __name__ == "__main__":
    main()
