"""
Training loop for SchNet charge model on MLX Metal.

Uses mlx.nn.value_and_grad for automatic differentiation
and mlx.optimizers.Adam with cosine decay schedule.
"""
from __future__ import annotations

import os
import time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_graphs.data import GraphData, GraphDataBatch
from mlx_graphs.loaders import Dataloader

from .schnet_charge import SchNetCharge


def charge_loss(model: SchNetCharge, batch: GraphDataBatch) -> mx.array:
    """Compute charge prediction loss.

    Loss = MSE(predicted_charges, target_charges)

    The neutrality constraint is built into the model (hard constraint),
    so we don't need an explicit neutrality loss term.
    """
    pred = model(batch)                     # (N_total,)
    target = batch.node_labels.squeeze()    # (N_total,)
    return mx.mean((pred - target) ** 2)


def evaluate(
    model: SchNetCharge,
    dataset: list[GraphData],
    batch_size: int = 32,
) -> dict:
    """Evaluate model on a dataset.

    Returns:
        dict with 'mae', 'rmse', 'max_error' in electron units
    """
    all_pred = []
    all_target = []

    loader = Dataloader(dataset, batch_size=batch_size, shuffle=False)
    for batch in loader:
        pred = model(batch)
        target = batch.node_labels.squeeze()
        mx.eval(pred, target)
        all_pred.append(np.array(pred))
        all_target.append(np.array(target))

    pred = np.concatenate(all_pred)
    target = np.concatenate(all_target)
    diff = pred - target

    return {
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
        "max_error": float(np.max(np.abs(diff))),
        "n_atoms": len(pred),
    }


def train_charge_model(
    train_data: list[GraphData],
    val_data: list[GraphData],
    hidden: int = 128,
    n_layers: int = 4,
    n_rbf: int = 64,
    delta_ml: bool = False,
    batch_size: int = 32,
    lr: float = 1e-3,
    epochs: int = 300,
    patience: int = 30,
    checkpoint_dir: str = "checkpoints",
    verbose: bool = True,
) -> tuple[SchNetCharge, dict]:
    """Train charge prediction model on MLX Metal.

    Args:
        train_data: list of GraphData for training
        val_data: list of GraphData for validation
        hidden: hidden dimension
        n_layers: number of CFConv blocks
        n_rbf: number of RBF basis functions
        delta_ml: Δ-ML mode (expects PM6 charges in node features)
        batch_size: training batch size
        lr: initial learning rate
        epochs: max number of epochs
        patience: early stopping patience
        checkpoint_dir: directory for saving checkpoints
        verbose: print training progress

    Returns:
        (trained_model, history_dict)
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Model
    model = SchNetCharge(
        hidden=hidden,
        n_layers=n_layers,
        n_rbf=n_rbf,
        delta_ml=delta_ml,
        enforce_neutrality=True,
    )

    # Optimizer with cosine decay
    n_batches = max(1, len(train_data) // batch_size)
    total_steps = epochs * n_batches
    schedule = optim.cosine_decay(lr, total_steps)
    optimizer = optim.Adam(learning_rate=schedule)

    # Compile loss + grad
    loss_and_grad_fn = nn.value_and_grad(model, charge_loss)

    # Training state
    best_val_mae = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_mae": [], "val_rmse": [], "lr": []}

    for epoch in range(epochs):
        t0 = time.time()
        epoch_loss = 0.0
        n_batches_actual = 0

        train_loader = Dataloader(train_data, batch_size=batch_size, shuffle=True)

        for batch in train_loader:
            loss, grads = loss_and_grad_fn(model, batch)

            # Update parameters
            optimizer.update(model, grads)

            # CRITICAL: force evaluation to prevent memory accumulation
            mx.eval(model.parameters(), optimizer.state, loss)

            epoch_loss += float(loss)
            n_batches_actual += 1

        avg_loss = epoch_loss / max(1, n_batches_actual)
        dt = time.time() - t0

        # Validation
        val_metrics = evaluate(model, val_data, batch_size)
        val_mae = val_metrics["mae"]
        val_rmse = val_metrics["rmse"]

        history["train_loss"].append(avg_loss)
        history["val_mae"].append(val_mae)
        history["val_rmse"].append(val_rmse)
        history["lr"].append(float(schedule(optimizer.step)))

        if verbose and (epoch % 5 == 0 or epoch == epochs - 1):
            print(
                f"Epoch {epoch:4d} | loss {avg_loss:.6f} | "
                f"val MAE {val_mae:.4f} e | val RMSE {val_rmse:.4f} e | "
                f"{dt:.1f}s"
            )

        # Early stopping
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_counter = 0
            # Save best checkpoint
            ckpt_path = os.path.join(checkpoint_dir, "best_model.npz")
            model.save_weights(ckpt_path)
            if verbose:
                print(f"  → Saved best model (MAE={best_val_mae:.4f} e)")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch} (patience={patience})")
                break

    # Load best model
    best_path = os.path.join(checkpoint_dir, "best_model.npz")
    if os.path.exists(best_path):
        model.load_weights(best_path)

    history["best_val_mae"] = best_val_mae
    return model, history


def quick_overfit_test(n_molecules: int = 5, n_steps: int = 200):
    """Quick sanity check: overfit on a tiny dataset.

    Creates random molecules with random charges and trains to near-zero loss.
    Verifies the training pipeline works end-to-end on MLX.
    """
    from .graph_builder import build_graph

    print("=" * 60)
    print("Quick Overfit Test: SchNet Charge Model on MLX Metal")
    print("=" * 60)

    # Generate random molecules
    np.random.seed(42)
    graphs = []
    for i in range(n_molecules):
        n_atoms = np.random.randint(3, 12)
        atoms = np.random.choice([1, 6, 7, 8], size=n_atoms)
        coords = np.random.randn(n_atoms, 3).astype(np.float32) * 1.5
        # Random charges summing to ~0
        charges = np.random.randn(n_atoms).astype(np.float32) * 0.3
        charges -= charges.mean()  # neutralize

        g = build_graph(atoms, coords, charges=charges, cutoff=5.0, n_rbf=32)
        graphs.append(g)

    print(f"Created {n_molecules} random molecules")
    print(f"Graph 0: {graphs[0].node_features.shape[0]} atoms, "
          f"{graphs[0].edge_index.shape[1]} edges")

    # Small model for speed
    model = SchNetCharge(hidden=64, n_layers=2, n_rbf=32)
    optimizer = optim.Adam(learning_rate=1e-3)
    loss_fn = nn.value_and_grad(model, charge_loss)

    loader = Dataloader(graphs, batch_size=n_molecules, shuffle=False)

    for step in range(n_steps):
        for batch in loader:
            loss, grads = loss_fn(model, batch)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss)

        if step % 20 == 0 or step == n_steps - 1:
            print(f"  Step {step:4d}: loss = {float(loss):.6f}")

    final_loss = float(loss)
    success = final_loss < 0.01
    print(f"\nFinal loss: {final_loss:.6f}")
    print(f"{'✓ PASSED' if success else '✗ FAILED'}: overfit test "
          f"({'loss < 0.01' if success else 'loss too high'})")
    return success


if __name__ == "__main__":
    quick_overfit_test()
