"""Replay recorded kernel calls (``harness.py --record``) on the kernels that
``PYTHONPATH`` resolves, and compare them conformer by conformer with the
recorded outputs.

Every call is replayed on the recorded inputs, so a difference is a property
of the kernel alone and never one inherited from an upstream stage.

Per conformer it reports: converged status (equal or not), energy difference
``|dE| / max(|E_ref|, 1)`` (relative above 1, absolute below: DG energies of
converged conformers sit at 1e-6 to 1e-3, where a pure relative error is
meaningless), and the coordinate RMS difference (4D for DG stages).

Example::

    PYTHONPATH=<checkout> python replay.py calls_old.pkl --csv cmp.csv --time 3
"""
from __future__ import annotations

import argparse
import json
import pickle
import statistics
import time

import numpy as np

import mlxmolkit
from mlxmolkit import conformer_metal, etk_metal, mmff_minimize

FN = {
    "dg1": conformer_metal.dg_minimize_shared,
    "dg_retry": conformer_metal.dg_minimize_shared,
    "collapse": conformer_metal.dg_minimize_shared,
    "etk": etk_metal.etk_minimize_shared,
    "mmff": mmff_minimize.mmff_minimize_nk,
}


def conformer_slices(call):
    """(start, stop) coordinate slices and dim for every conformer of a call."""
    if call["stage"] == "mmff":
        params, counts, _ = call["args"]
        sizes = [p.n_atoms * 3 for p, k in zip(params, counts) for _ in range(k)]
        dim = 3
    else:
        batch = call["args"][0]
        dim = batch.dim
        starts = batch.conf_atom_starts
        sizes = [int(starts[c + 1] - starts[c]) * dim for c in range(batch.n_confs_total)]
    edges = np.concatenate([[0], np.cumsum(sizes)])
    return [(int(edges[i]), int(edges[i + 1])) for i in range(len(sizes))], dim


def status_of(call, out):
    s = np.asarray(out[2])
    # mmff_minimize_nk returns a converged bool; the DG/ETK kernels a status int.
    return (~s.astype(bool)).astype(np.int32) if call["stage"] == "mmff" else s.astype(np.int32)


def compare(call, ref, new, rows):
    slices, dim = conformer_slices(call)
    s_ref, s_new = status_of(call, ref), status_of(call, new)
    e_ref, e_new = np.asarray(ref[1], np.float64), np.asarray(new[1], np.float64)
    p_ref, p_new = np.asarray(ref[0], np.float64), np.asarray(new[0], np.float64)
    for c, (a, b) in enumerate(slices):
        d = (p_new[a:b] - p_ref[a:b]).reshape(-1, dim)
        rows.append({
            "stage": call["stage"], "conf": c,
            "status_ref": int(s_ref[c]), "status_new": int(s_new[c]),
            "e_ref": float(e_ref[c]), "e_new": float(e_new[c]),
            "de_scaled": float(abs(e_new[c] - e_ref[c]) / max(abs(e_ref[c]), 1.0)),
            "rms": float(np.sqrt(np.mean(np.sum(d * d, axis=1)))),
            "bitwise_pos": bool(np.array_equal(np.asarray(ref[0][a:b]), np.asarray(new[0][a:b]))),
        })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("record")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--time", type=int, default=0, help="also time each call: median of N after 1 warm-up")
    ap.add_argument("--stages", default=",".join(FN))
    ap.add_argument("--json", default=None)
    ap.add_argument("--override", default=None, help="kwarg override, e.g. use_lbfgs=1")
    ap.add_argument("--save", default=None, help="pickle the replayed outputs")
    ap.add_argument("--ref", default=None, help="compare against outputs saved by --save instead of the recording")
    args = ap.parse_args()

    with open(args.record, "rb") as fh:
        rec = pickle.load(fh)
    want = set(args.stages.split(","))
    refs = None
    if args.ref:
        with open(args.ref, "rb") as fh:
            refs = pickle.load(fh)
    override = {}
    if args.override:
        key, val = args.override.split("=")
        override[key] = type(True)(int(val)) if key == "use_lbfgs" else float(val)
    rows: list[dict] = []
    timing: dict[str, float] = {}
    saved = []
    for i, call in enumerate(rec["calls"]):
        if call["stage"] not in want:
            continue
        fn = FN[call["stage"]]
        kwargs = dict(call["kwargs"])
        if call["stage"] == "mmff":
            kwargs.update(override)
        call = dict(call, kwargs=kwargs)
        new = fn(*call["args"], **kwargs)
        saved.append((i, tuple(np.array(o, copy=True) for o in new)))
        ref = call["out"] if refs is None else dict(refs)[i]
        compare(call, ref, new, rows)
        if args.time:
            fn(*call["args"], **kwargs)
            ts = []
            for _ in range(args.time):
                t = time.perf_counter()
                fn(*call["args"], **kwargs)
                ts.append(time.perf_counter() - t)
            timing[call["stage"]] = timing.get(call["stage"], 0.0) + statistics.median(ts)

    if args.save:
        with open(args.save, "wb") as fh:
            pickle.dump(saved, fh, protocol=pickle.HIGHEST_PROTOCOL)
    import pandas as pd
    df = pd.DataFrame(rows)
    if args.csv:
        df.to_csv(args.csv, index=False)
    summary = {"pkg": mlxmolkit.__file__, "record": args.record, "stages": {}}
    for stage, g in df.groupby("stage", sort=False):
        summary["stages"][stage] = {
            "n": int(len(g)),
            "status_mismatch": int((g.status_ref != g.status_new).sum()),
            "unconverged_ref": int((g.status_ref != 0).sum()),
            "unconverged_new": int((g.status_new != 0).sum()),
            "de_gt_1e-4": int((g.de_scaled > 1e-4).sum()),
            "rms_gt_1e-3": int((g.rms > 1e-3).sum()),
            "max_de_scaled": float(g.de_scaled.max()),
            "max_rms": float(g.rms.max()),
            "median_rms": float(g.rms.median()),
            "bitwise_identical_pos": int(g.bitwise_pos.sum()),
            "time_s": timing.get(stage),
        }
    print(json.dumps(summary, indent=1))
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(summary, fh, indent=1)


if __name__ == "__main__":
    main()
