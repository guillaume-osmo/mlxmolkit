"""
Streaming fine-tune: load shards one at a time via mlx.data, never hold all in RAM.

Usage:
    python -m mlxmolkit.ml.finetune_stream \
        --shards data/spice_shards/ \
        --model models/schnet_charge_50k.npz \
        --epochs 5
"""
from __future__ import annotations

import os
import sys
import glob
import time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from .schnet_charge import SchNetCharge
from .shard_spice import load_shard_as_graphs
from .train import charge_loss, evaluate
from .graph_builder import ELEMENT_TO_IDX

from mlx_graphs.loaders import Dataloader

ELEM_NAMES = {
    1: 'H', 5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 14: 'Si',
    15: 'P', 16: 'S', 17: 'Cl', 35: 'Br', 53: 'I',
}


def stream_train_epoch(
    model: SchNetCharge,
    optimizer,
    loss_and_grad,
    shard_dir: str,
    batch_size: int = 64,
    shuffle_shards: bool = True,
):
    """Train one epoch by streaming shards from disk.

    Loads one shard at a time → train on it → release memory → next shard.
    """
    shard_files = sorted(glob.glob(os.path.join(shard_dir, "train_*.npz")))
    if shuffle_shards:
        np.random.shuffle(shard_files)

    total_loss = 0.0
    total_batches = 0
    total_graphs = 0

    for shard_path in shard_files:
        # Load one shard into memory
        graphs = load_shard_as_graphs(shard_path)
        total_graphs += len(graphs)

        loader = Dataloader(graphs, batch_size=batch_size, shuffle=True)
        for batch in loader:
            loss, grads = loss_and_grad(model, batch)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss)

            total_loss += float(loss)
            total_batches += 1

        # Release shard memory
        del graphs

    avg_loss = total_loss / max(1, total_batches)
    return avg_loss, total_graphs


def load_val_test(shard_dir: str, split: str = "val", max_graphs: int = 5000):
    """Load val or test shards (small enough to fit in RAM)."""
    shard_files = sorted(glob.glob(os.path.join(shard_dir, f"{split}_*.npz")))
    graphs = []
    for sf in shard_files:
        graphs.extend(load_shard_as_graphs(sf))
        if len(graphs) >= max_graphs:
            break
    return graphs[:max_graphs]


def per_element_report(model, test_graphs, n_max=1000):
    """Per-element MAE breakdown."""
    rev_map = {v: k for k, v in ELEMENT_TO_IDX.items()}
    elem_errs = {}

    for g in test_graphs[:n_max]:
        pred = model(g)
        mx.eval(pred)
        p = np.array(pred)
        t = np.array(g.node_labels).flatten()
        nf = np.array(g.node_features)
        for ai in range(len(t)):
            z_idx = int(np.argmax(nf[ai, :19]))
            z_val = rev_map.get(z_idx, 0)
            name = ELEM_NAMES.get(z_val, f"Z{z_val}")
            elem_errs.setdefault(name, []).append(abs(p[ai] - t[ai]))

    print("Per-element MAE:")
    for name in sorted(elem_errs.keys()):
        errs = elem_errs[name]
        print(f"  {name:3s}: MAE={np.mean(errs):.4f} e  (n={len(errs):,})")
    return elem_errs


def finetune_stream(
    shard_dir: str,
    model_path: str | None = None,
    hidden: int = 128,
    n_layers: int = 4,
    n_rbf: int = 64,
    batch_size: int = 64,
    lr: float = 2e-4,
    epochs: int = 5,
    save_path: str | None = None,
):
    """Stream-based fine-tuning on all shards."""
    print("=" * 60)
    print("Streaming Fine-Tune on Full SPICE Shards")
    print("=" * 60)

    # Count shards
    train_shards = sorted(glob.glob(os.path.join(shard_dir, "train_*.npz")))
    val_shards = sorted(glob.glob(os.path.join(shard_dir, "val_*.npz")))
    test_shards = sorted(glob.glob(os.path.join(shard_dir, "test_*.npz")))
    print(f"Shards: train={len(train_shards)}, val={len(val_shards)}, test={len(test_shards)}")

    # Load model
    model = SchNetCharge(hidden=hidden, n_layers=n_layers, n_rbf=n_rbf)
    if model_path and os.path.exists(model_path):
        model.load_weights(model_path)
        print(f"Loaded weights from {model_path}")
    else:
        print("Training from scratch (no pretrained weights)")

    # Load val + test (small, fits in RAM)
    print("Loading val/test data...")
    val_graphs = load_val_test(shard_dir, "val", max_graphs=5000)
    test_graphs = load_val_test(shard_dir, "test", max_graphs=5000)
    print(f"Val: {len(val_graphs)} graphs, Test: {len(test_graphs)} graphs")

    # Pre-training evaluation
    print("\n--- Before fine-tuning ---")
    pre = evaluate(model, test_graphs[:2000], batch_size=64)
    print(f"  Test MAE: {pre['mae']:.4f} e, RMSE: {pre['rmse']:.4f} e")

    # Optimizer
    optimizer = optim.Adam(learning_rate=lr)
    loss_and_grad = nn.value_and_grad(model, charge_loss)

    best_mae = pre["mae"]

    # Training loop: stream shards
    print(f"\n--- Fine-tuning ({epochs} epochs, LR={lr}, batch={batch_size}) ---")
    for epoch in range(epochs):
        t0 = time.time()

        avg_loss, n_graphs = stream_train_epoch(
            model, optimizer, loss_and_grad, shard_dir,
            batch_size=batch_size, shuffle_shards=True,
        )

        dt = time.time() - t0
        speed = n_graphs / dt

        # Validate
        val_m = evaluate(model, val_graphs[:2000], batch_size=64)

        print(f"  Epoch {epoch}: loss={avg_loss:.6f}, "
              f"val MAE={val_m['mae']:.4f} e, "
              f"{dt:.0f}s ({speed:.0f} graphs/s, {n_graphs:,} graphs)")

        if val_m["mae"] < best_mae:
            best_mae = val_m["mae"]
            if save_path:
                model.save_weights(save_path)
                print(f"    → Saved best (MAE={best_mae:.4f} e)")

    # Final test evaluation
    print(f"\n{'='*60}")
    print("FINAL TEST EVALUATION (unseen molecules)")
    print(f"{'='*60}")
    post = evaluate(model, test_graphs, batch_size=64)
    print(f"  MAE:       {post['mae']:.4f} e")
    print(f"  RMSE:      {post['rmse']:.4f} e")
    print(f"  Max Error: {post['max_error']:.4f} e")
    print(f"  Atoms:     {post['n_atoms']:,}")

    imp = (pre['mae'] - post['mae']) / pre['mae'] * 100 if pre['mae'] > 0 else 0
    print(f"\n  Improvement: {pre['mae']:.4f} → {post['mae']:.4f} e ({imp:+.1f}%)")

    # Per-element
    print()
    per_element_report(model, test_graphs, n_max=2000)

    # Save final
    if save_path:
        final_path = save_path.replace(".npz", "_final.npz")
        model.save_weights(final_path)
        print(f"\nFinal model saved → {final_path}")

    return model


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--shards", default="/Users/tgg/Github/mlxmolkit_phase1/data/spice_shards")
    p.add_argument("--model", default="/Users/tgg/Github/mlxmolkit_phase1/models/schnet_charge_50k.npz")
    p.add_argument("--save", default="/Users/tgg/Github/mlxmolkit_phase1/models/schnet_charge_full.npz")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=64)
    args = p.parse_args()

    finetune_stream(
        shard_dir=args.shards,
        model_path=args.model,
        save_path=args.save,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
    )
