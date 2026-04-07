"""
Preprocess SPICE-2.0.1 → sharded numpy files for streaming training.

Step 1: Scan all molecules, split by molecule ID into train/val/test
Step 2: Oversample rare elements (P, S, Br, Si, B, I) for balanced training
Step 3: Save as .npz shards (~5000 graphs each) for mlx-data streaming

Usage:
    python -m mlxmolkit.ml.shard_spice --spice /path/to/SPICE-2.0.1.hdf5 --out data/spice_shards/
"""
from __future__ import annotations

import os
import sys
import time
import h5py
import numpy as np
from collections import Counter

from .graph_builder import build_edges_np, expnorm_rbf, _one_hot_z, ELEMENT_TO_IDX

BOHR2ANG = 0.529177249
SUPPORTED = set(ELEMENT_TO_IDX.keys())
ATOMIC_MASS = {
    1: 1.008, 3: 6.94, 5: 10.81, 6: 12.011, 7: 14.007, 8: 15.999,
    9: 18.998, 11: 22.990, 12: 24.305, 14: 28.086, 15: 30.974,
    16: 32.06, 17: 35.45, 19: 39.098, 20: 40.078, 35: 79.904, 53: 126.904,
}
# Rare elements to oversample (fewer training examples → higher MAE)
RARE_ELEMENTS = {5, 14, 15, 16, 35, 53}  # B, Si, P, S, Br, I

ELEM_NAMES = {
    1: 'H', 3: 'Li', 5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F',
    11: 'Na', 12: 'Mg', 14: 'Si', 15: 'P', 16: 'S', 17: 'Cl',
    19: 'K', 20: 'Ca', 35: 'Br', 53: 'I',
}


def _mol_to_arrays(z, coords, charges, cutoff=5.0, n_rbf=64):
    """Convert one conformation to flat numpy arrays for sharding.

    Returns dict with:
        node_features: (N, n_feat) float32
        edge_index: (2, E) int32
        edge_features: (E, n_rbf) float32
        charges: (N,) float32
        n_atoms: int
        formal_charge: float
    """
    N = len(z)
    nf = _one_hot_z(z)
    src, dst, dists = build_edges_np(coords, cutoff)

    if len(src) == 0:
        src = np.array([0], dtype=np.int32)
        dst = np.array([0], dtype=np.int32)
        dists = np.array([0.0], dtype=np.float32)

    ef = expnorm_rbf(dists, n_rbf, cutoff)
    ei = np.stack([src, dst], axis=0)

    return {
        "node_features": nf,
        "edge_index": ei,
        "edge_features": ef,
        "charges": charges.astype(np.float32),
        "n_atoms": np.int32(N),
        "formal_charge": np.float32(round(float(charges.sum()))),
    }


def scan_and_split_molecules(
    hdf5_path: str,
    max_atoms: int = 50,
    max_mw: float = 500.0,
    train_frac: float = 0.80,
    val_frac: float = 0.10,
    seed: int = 42,
) -> tuple[list[str], list[str], list[str], dict]:
    """Scan SPICE, filter, and split molecule keys by identity.

    Returns:
        (train_keys, val_keys, test_keys, stats_dict)
    """
    print("Scanning SPICE molecules...")
    t0 = time.time()
    valid_keys = []
    elem_counts = Counter()
    has_rare = []

    with h5py.File(hdf5_path, "r") as f:
        all_keys = list(f.keys())
        for key in all_keys:
            grp = f[key]
            if "atomic_numbers" not in grp or "mbis_charges" not in grp:
                continue
            z = np.array(grp["atomic_numbers"], dtype=np.int32)
            n_at = len(z)
            if n_at < 3 or n_at > max_atoms:
                continue
            z_set = set(z.tolist())
            if not z_set.issubset(SUPPORTED):
                continue
            mw = sum(ATOMIC_MASS.get(int(zi), 100) for zi in z)
            if mw > max_mw:
                continue

            # Check for conformations
            charges = grp["mbis_charges"]
            if charges.ndim >= 2 and charges.shape[0] == 0:
                continue

            valid_keys.append(key)
            for zi in z:
                elem_counts[int(zi)] += 1
            has_rare.append(bool(z_set & RARE_ELEMENTS))

    dt = time.time() - t0
    print(f"Found {len(valid_keys)} valid molecules in {dt:.1f}s")
    print(f"Element distribution:")
    for z_val in sorted(elem_counts.keys()):
        name = ELEM_NAMES.get(z_val, f"Z{z_val}")
        cnt = elem_counts[z_val]
        rare_tag = " ★ RARE" if z_val in RARE_ELEMENTS else ""
        print(f"  {name:3s} (Z={z_val:2d}): {cnt:8d} atoms{rare_tag}")

    # Split by molecule
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(valid_keys))
    nt = int(len(valid_keys) * train_frac)
    nv = int(len(valid_keys) * val_frac)

    train_keys = [valid_keys[i] for i in idx[:nt]]
    val_keys = [valid_keys[i] for i in idx[nt:nt + nv]]
    test_keys = [valid_keys[i] for i in idx[nt + nv:]]

    # Count how many rare-element molecules in each split
    train_rare = sum(1 for i in idx[:nt] if has_rare[i])
    print(f"\nSplit: train={len(train_keys)} (rare={train_rare}), "
          f"val={len(val_keys)}, test={len(test_keys)}")

    stats = {
        "total": len(valid_keys),
        "train": len(train_keys),
        "val": len(val_keys),
        "test": len(test_keys),
        "elem_counts": dict(elem_counts),
    }
    return train_keys, val_keys, test_keys, stats


def shard_molecules(
    hdf5_path: str,
    mol_keys: list[str],
    out_dir: str,
    prefix: str = "train",
    max_conf: int = 3,
    oversample_rare: int = 2,
    shard_size: int = 5000,
    cutoff: float = 5.0,
    n_rbf: int = 64,
):
    """Convert molecules to graph arrays and save as .npz shards.

    Args:
        oversample_rare: repeat molecules with rare elements this many times
    """
    os.makedirs(out_dir, exist_ok=True)

    shard_data = []
    shard_idx = 0
    total_graphs = 0
    total_atoms = 0
    t0 = time.time()

    with h5py.File(hdf5_path, "r") as f:
        for ki, key in enumerate(mol_keys):
            grp = f[key]
            z = np.array(grp["atomic_numbers"], dtype=np.int32)
            coords_all = np.array(grp["conformations"]) * BOHR2ANG
            charges_all = np.array(grp["mbis_charges"])

            if coords_all.ndim == 2:
                coords_all = coords_all[np.newaxis]
                charges_all = charges_all[np.newaxis]
            if charges_all.ndim == 3 and charges_all.shape[2] == 1:
                charges_all = charges_all[:, :, 0]
            if coords_all.shape[0] == 0:
                continue

            # Determine how many times to include (oversample rare)
            z_set = set(z.tolist())
            repeats = oversample_rare if (z_set & RARE_ELEMENTS) else 1

            nc = min(max_conf, coords_all.shape[0])
            for _ in range(repeats):
                for ci in range(nc):
                    d = _mol_to_arrays(
                        z.tolist(),
                        coords_all[ci].astype(np.float32),
                        charges_all[ci].astype(np.float32),
                        cutoff=cutoff,
                        n_rbf=n_rbf,
                    )
                    shard_data.append(d)
                    total_atoms += int(d["n_atoms"])

                    if len(shard_data) >= shard_size:
                        _save_shard(shard_data, out_dir, prefix, shard_idx)
                        total_graphs += len(shard_data)
                        shard_idx += 1
                        shard_data = []

            if (ki + 1) % 10000 == 0:
                elapsed = time.time() - t0
                print(f"  {prefix}: {ki+1}/{len(mol_keys)} mols, "
                      f"{total_graphs + len(shard_data)} graphs, "
                      f"{elapsed:.0f}s")

    # Save final partial shard
    if shard_data:
        _save_shard(shard_data, out_dir, prefix, shard_idx)
        total_graphs += len(shard_data)
        shard_idx += 1

    dt = time.time() - t0
    print(f"  {prefix}: {len(mol_keys)} mols → {total_graphs} graphs "
          f"({total_atoms} atoms) in {shard_idx} shards, {dt:.1f}s")
    return total_graphs


def _save_shard(graphs: list[dict], out_dir: str, prefix: str, idx: int):
    """Pack list of graph dicts into a single .npz shard.

    Variable-size graphs are stored with offsets for reconstruction.
    """
    all_nf = []
    all_ei_src = []
    all_ei_dst = []
    all_ef = []
    all_charges = []
    all_n_atoms = []
    all_n_edges = []
    all_formal = []

    for g in graphs:
        all_nf.append(g["node_features"])
        all_ei_src.append(g["edge_index"][0])
        all_ei_dst.append(g["edge_index"][1])
        all_ef.append(g["edge_features"])
        all_charges.append(g["charges"])
        all_n_atoms.append(g["n_atoms"])
        all_n_edges.append(len(g["edge_index"][0]))
        all_formal.append(g["formal_charge"])

    path = os.path.join(out_dir, f"{prefix}_{idx:05d}.npz")
    np.savez_compressed(
        path,
        node_features=np.concatenate(all_nf, axis=0),
        ei_src=np.concatenate(all_ei_src),
        ei_dst=np.concatenate(all_ei_dst),
        edge_features=np.concatenate(all_ef, axis=0),
        charges=np.concatenate(all_charges),
        n_atoms=np.array(all_n_atoms, dtype=np.int32),
        n_edges=np.array(all_n_edges, dtype=np.int32),
        formal_charges=np.array(all_formal, dtype=np.float32),
    )


def load_shard_as_graphs(shard_path: str):
    """Load a .npz shard back into list of GraphData objects."""
    import mlx.core as mx
    from mlx_graphs.data import GraphData

    d = np.load(shard_path)
    nf_all = d["node_features"]
    src_all = d["ei_src"]
    dst_all = d["ei_dst"]
    ef_all = d["edge_features"]
    q_all = d["charges"]
    n_atoms = d["n_atoms"]
    n_edges = d["n_edges"]
    fc_all = d["formal_charges"]

    graphs = []
    atom_off = 0
    edge_off = 0

    for i in range(len(n_atoms)):
        na = int(n_atoms[i])
        ne = int(n_edges[i])

        nf = nf_all[atom_off:atom_off + na]
        src = src_all[edge_off:edge_off + ne]
        dst = dst_all[edge_off:edge_off + ne]
        ef = ef_all[edge_off:edge_off + ne]
        q = q_all[atom_off:atom_off + na]
        fc = fc_all[i]

        ei = np.stack([src, dst], axis=0).astype(np.int32)

        g = GraphData(
            edge_index=mx.array(ei),
            node_features=mx.array(nf),
            edge_features=mx.array(ef),
            node_labels=mx.array(q.reshape(-1, 1)),
            graph_labels=mx.array(np.array([fc], dtype=np.float32)),
        )
        graphs.append(g)

        atom_off += na
        edge_off += ne

    return graphs


def preprocess_spice(
    hdf5_path: str,
    out_dir: str,
    max_atoms: int = 50,
    max_mw: float = 500.0,
    max_conf: int = 3,
    oversample_rare: int = 3,
    shard_size: int = 5000,
    seed: int = 42,
):
    """Full preprocessing pipeline: scan → split → shard."""
    print("=" * 60)
    print("SPICE-2.0.1 → Sharded Training Data")
    print("=" * 60)

    train_keys, val_keys, test_keys, stats = scan_and_split_molecules(
        hdf5_path, max_atoms=max_atoms, max_mw=max_mw, seed=seed,
    )

    print(f"\nSharding train set (oversample rare ×{oversample_rare})...")
    n_train = shard_molecules(
        hdf5_path, train_keys, out_dir, "train",
        max_conf=max_conf, oversample_rare=oversample_rare,
        shard_size=shard_size,
    )

    print(f"\nSharding val set...")
    n_val = shard_molecules(
        hdf5_path, val_keys, out_dir, "val",
        max_conf=max_conf, oversample_rare=1,
        shard_size=shard_size,
    )

    print(f"\nSharding test set...")
    n_test = shard_molecules(
        hdf5_path, test_keys, out_dir, "test",
        max_conf=max_conf, oversample_rare=1,
        shard_size=shard_size,
    )

    # Save split metadata
    np.savez(
        os.path.join(out_dir, "metadata.npz"),
        train_keys=np.array(train_keys, dtype=object),
        val_keys=np.array(val_keys, dtype=object),
        test_keys=np.array(test_keys, dtype=object),
        n_train=n_train, n_val=n_val, n_test=n_test,
    )

    print(f"\n{'='*60}")
    print(f"Done! {n_train + n_val + n_test} total graphs")
    print(f"  Train: {n_train} graphs (rare ×{oversample_rare})")
    print(f"  Val:   {n_val} graphs")
    print(f"  Test:  {n_test} graphs")
    print(f"  Saved to: {out_dir}/")
    print(f"{'='*60}")

    return out_dir


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--spice", default="/Users/tgg/Downloads/SPICE-2.0.1.hdf5")
    p.add_argument("--out", default="/Users/tgg/Github/mlxmolkit_phase1/data/spice_shards")
    p.add_argument("--max-atoms", type=int, default=50)
    p.add_argument("--max-conf", type=int, default=3)
    p.add_argument("--oversample-rare", type=int, default=3)
    args = p.parse_args()

    preprocess_spice(args.spice, args.out,
                     max_atoms=args.max_atoms,
                     max_conf=args.max_conf,
                     oversample_rare=args.oversample_rare)
