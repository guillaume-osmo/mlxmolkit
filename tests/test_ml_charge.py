"""
Tests for ML charge prediction pipeline on MLX Metal.

1. Graph builder unit tests
2. SchNet forward pass shape tests
3. Overfit test (5 molecules)
4. Training loop integration test
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import mlx.core as mx
from mlxmolkit.ml.graph_builder import build_graph, build_edges_np, gaussian_rbf, expnorm_rbf
from mlxmolkit.ml.schnet_charge import SchNetCharge
from mlxmolkit.ml.train import quick_overfit_test, charge_loss, evaluate


def test_graph_builder():
    """Test graph construction from coordinates."""
    print("=" * 60)
    print("Test: Graph Builder")
    print("=" * 60)

    # Water
    atoms = [8, 1, 1]
    coords = np.array([[0, 0, 0.117], [0, 0.757, -0.469], [0, -0.757, -0.469]])
    charges = np.array([-0.8, 0.4, 0.4])

    g = build_graph(atoms, coords, charges=charges, cutoff=5.0, n_rbf=64)

    assert g.node_features.shape == (3, 19), f"Wrong nf shape: {g.node_features.shape}"
    assert g.edge_index.shape[0] == 2, f"Wrong ei shape: {g.edge_index.shape}"
    assert g.edge_features.shape[1] == 64, f"Wrong ef shape: {g.edge_features.shape}"
    assert g.node_labels.shape == (3, 1), f"Wrong nl shape: {g.node_labels.shape}"

    n_edges = g.edge_index.shape[1]
    assert n_edges == 6, f"Expected 6 edges (3 atoms, all pairs), got {n_edges}"

    print(f"  Water: {g.node_features.shape[0]} nodes, {n_edges} edges")
    print(f"  node_features: {g.node_features.shape}")
    print(f"  edge_features: {g.edge_features.shape}")
    print(f"  node_labels: {g.node_labels.shape}")
    print("  ✓ PASSED")
    return True


def test_edges():
    """Test edge construction."""
    print("=" * 60)
    print("Test: Edge Construction")
    print("=" * 60)

    coords = np.array([[0, 0, 0], [1, 0, 0], [0, 6, 0]], dtype=np.float32)  # 3rd atom far away
    src, dst, dists = build_edges_np(coords, cutoff=5.0)

    # Should have edges only between atoms 0-1 (distance 1.0), not 0-2 or 1-2 (distance > 5)
    assert len(src) == 2, f"Expected 2 edges, got {len(src)}"
    assert dists[0] < 1.1, f"Expected d≈1.0, got {dists[0]}"

    print(f"  3 atoms, cutoff=5.0: {len(src)} edges")
    print(f"  Edge pairs: {list(zip(src, dst))}")
    print(f"  Distances: {dists}")
    print("  ✓ PASSED")
    return True


def test_rbf():
    """Test RBF expansion."""
    print("=" * 60)
    print("Test: RBF Expansion")
    print("=" * 60)

    dists = np.array([0.5, 1.0, 2.5, 4.9], dtype=np.float32)

    g_rbf = gaussian_rbf(dists, n_rbf=32, cutoff=5.0)
    assert g_rbf.shape == (4, 32), f"Wrong shape: {g_rbf.shape}"
    assert g_rbf.min() >= 0, "Gaussian RBF should be non-negative"

    e_rbf = expnorm_rbf(dists, n_rbf=32, cutoff=5.0)
    assert e_rbf.shape == (4, 32), f"Wrong shape: {e_rbf.shape}"
    assert e_rbf.min() >= 0, "ExpNorm RBF should be non-negative"
    # At cutoff (4.9), envelope should be near zero
    assert e_rbf[3].max() < 0.1, f"RBF at d=4.9 should be small: {e_rbf[3].max()}"

    print(f"  Gaussian RBF: {g_rbf.shape}, max={g_rbf.max():.4f}")
    print(f"  ExpNorm RBF: {e_rbf.shape}, max={e_rbf.max():.4f}")
    print("  ✓ PASSED")
    return True


def test_schnet_forward():
    """Test SchNet forward pass shapes."""
    print("=" * 60)
    print("Test: SchNet Forward Pass")
    print("=" * 60)

    # Create simple graph
    atoms = [6, 1, 1, 1, 1]
    coords = np.array([
        [0, 0, 0], [1.09, 0, 0], [-0.36, 1.03, 0],
        [-0.36, -0.51, 0.89], [-0.36, -0.51, -0.89]
    ], dtype=np.float32)
    charges = np.array([-0.4, 0.1, 0.1, 0.1, 0.1])

    g = build_graph(atoms, coords, charges=charges, n_rbf=32)

    model = SchNetCharge(hidden=64, n_layers=2, n_rbf=32)
    q = model(g)
    mx.eval(q)

    assert q.shape == (5,), f"Expected (5,), got {q.shape}"

    # Check neutrality constraint
    q_sum = float(mx.sum(q))
    assert abs(q_sum) < 0.01, f"Charges should sum to ~0, got {q_sum}"

    q_np = np.array(q)
    print(f"  CH4: {len(atoms)} atoms → charges {q_np}")
    print(f"  Sum of charges: {q_sum:.6f} (neutrality constraint)")
    print("  ✓ PASSED")
    return True


def test_schnet_delta_ml():
    """Test Δ-ML mode (PM6 charges as input)."""
    print("=" * 60)
    print("Test: SchNet Δ-ML Mode")
    print("=" * 60)

    atoms = [8, 1, 1]
    coords = np.array([[0, 0, 0.117], [0, 0.757, -0.469], [0, -0.757, -0.469]])
    pm6_charges = np.array([-0.5, 0.25, 0.25])

    g = build_graph(atoms, coords, pm6_charges=pm6_charges, n_rbf=32)
    assert g.node_features.shape == (3, 20), f"Δ-ML should have 20 features (19+1), got {g.node_features.shape}"

    model = SchNetCharge(hidden=64, n_layers=2, n_rbf=32, delta_ml=True)
    q = model(g)
    mx.eval(q)

    assert q.shape == (3,), f"Expected (3,), got {q.shape}"
    print(f"  Water Δ-ML: charges = {np.array(q)}")
    print(f"  node_features shape: {g.node_features.shape} (19 one-hot + 1 PM6 charge)")
    print("  ✓ PASSED")
    return True


def test_overfit():
    """Run the overfit test."""
    return quick_overfit_test(n_molecules=5, n_steps=200)


if __name__ == "__main__":
    tests = [
        test_graph_builder,
        test_edges,
        test_rbf,
        test_schnet_forward,
        test_schnet_delta_ml,
        test_overfit,
    ]

    passed = 0
    failed = 0
    for t in tests:
        try:
            if t():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
        print()

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    print("=" * 60)
