"""
Test PM6-ML MLX port against PyTorch reference.

Verifies that the MLX Metal implementation produces identical results
to the original TorchMD-NET PyTorch model.
"""
import os
import sys
import numpy as np

# Handle OMP conflict between torch and mlx
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Water geometry (Angstrom)
WATER_Z = [8, 1, 1]
WATER_COORDS = np.array([
    [0.0000,  0.0000,  0.1173],
    [0.0000,  0.7572, -0.4692],
    [0.0000, -0.7572, -0.4692],
], dtype=np.float64)

# Ethanol geometry
ETHANOL_Z = [6, 6, 8, 1, 1, 1, 1, 1, 1]
ETHANOL_COORDS = np.array([
    [-0.7516,  0.0089, -0.0348],
    [ 0.7516, -0.0089,  0.0348],
    [ 1.1547, -0.0250,  1.3850],
    [-1.1683, -0.9555,  0.2620],
    [-1.1057,  0.2454, -1.0375],
    [-1.1287,  0.7819,  0.6249],
    [ 1.1287, -0.7819, -0.6249],
    [ 1.1057, -0.2454,  1.0375],
    [ 2.0972, -0.0500,  1.4300],
], dtype=np.float64)

CKPT = "/Users/tgg/Github/mopac-ml/models/PM6-ML_correction_seed8_best.ckpt"


def run_torch_reference(atoms, coords):
    """Run PyTorch reference model, return energy in kJ/mol."""
    import torch
    from torchmdnet.models.model import load_model

    Z_TO_ATYPE = {
        35: 1, 6: 3, 20: 5, 17: 7, 9: 9, 1: 10, 53: 12, 19: 13,
        3: 14, 12: 15, 7: 17, 11: 19, 8: 21, 15: 23, 16: 26,
    }
    model = load_model(CKPT, derivative=False)
    model.eval()

    types = torch.tensor([Z_TO_ATYPE[z] for z in atoms], dtype=torch.long)
    pos = torch.tensor(np.asarray(coords, dtype=np.float32))
    with torch.no_grad():
        out = model(types, pos)
    energy = out[0] if isinstance(out, tuple) else out
    return energy.item()  # kJ/mol


def run_mlx(atoms, coords):
    """Run MLX model, return energy in kJ/mol."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from mlxmolkit.rm1.pm6_ml_mlx import PM6ML_MLX

    model = PM6ML_MLX.from_checkpoint(CKPT)
    return model(atoms, coords)  # kJ/mol


def test_water():
    """Test Water: compare MLX vs PyTorch."""
    print("=" * 60)
    print("Water (H2O) — PM6-ML Energy Correction")
    print("=" * 60)

    torch_kj = run_torch_reference(WATER_Z, WATER_COORDS)
    mlx_kj = run_mlx(WATER_Z, WATER_COORDS)

    torch_ev = torch_kj / 96.485
    mlx_ev = mlx_kj / 96.485
    diff_ev = abs(mlx_ev - torch_ev)
    rel_err = abs(diff_ev / (torch_ev + 1e-30))

    print(f"  PyTorch:  {torch_kj:12.6f} kJ/mol = {torch_ev:10.6f} eV")
    print(f"  MLX:      {mlx_kj:12.6f} kJ/mol = {mlx_ev:10.6f} eV")
    print(f"  Diff:     {diff_ev:.6e} eV  ({rel_err:.4%} relative)")
    print()

    assert diff_ev < 0.01, f"Energy difference too large: {diff_ev:.6f} eV"
    print("  ✓ Water PASSED (< 0.01 eV difference)")
    return True


def test_ethanol():
    """Test Ethanol: compare MLX vs PyTorch."""
    print("=" * 60)
    print("Ethanol (C2H5OH) — PM6-ML Energy Correction")
    print("=" * 60)

    torch_kj = run_torch_reference(ETHANOL_Z, ETHANOL_COORDS)
    mlx_kj = run_mlx(ETHANOL_Z, ETHANOL_COORDS)

    torch_ev = torch_kj / 96.485
    mlx_ev = mlx_kj / 96.485
    diff_ev = abs(mlx_ev - torch_ev)
    rel_err = abs(diff_ev / (torch_ev + 1e-30))

    print(f"  PyTorch:  {torch_kj:12.6f} kJ/mol = {torch_ev:10.6f} eV")
    print(f"  MLX:      {mlx_kj:12.6f} kJ/mol = {mlx_ev:10.6f} eV")
    print(f"  Diff:     {diff_ev:.6e} eV  ({rel_err:.4%} relative)")
    print()

    assert diff_ev < 0.01, f"Energy difference too large: {diff_ev:.6f} eV"
    print("  ✓ Ethanol PASSED (< 0.01 eV difference)")
    return True


def test_npz_roundtrip():
    """Test save/load .npz gives identical results."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from mlxmolkit.rm1.pm6_ml_mlx import PM6ML_MLX

    print("=" * 60)
    print("NPZ roundtrip test")
    print("=" * 60)

    model1 = PM6ML_MLX.from_checkpoint(CKPT)
    e1 = model1(WATER_Z, WATER_COORDS)

    npz_path = "/tmp/pm6ml_test.npz"
    model1.save_npz(npz_path)
    model2 = PM6ML_MLX.from_npz(npz_path)
    e2 = model2(WATER_Z, WATER_COORDS)

    diff = abs(e1 - e2)
    print(f"  From ckpt: {e1:.6f} kJ/mol")
    print(f"  From npz:  {e2:.6f} kJ/mol")
    print(f"  Diff:      {diff:.2e} kJ/mol")

    assert diff < 1e-4, f"NPZ roundtrip difference too large: {diff}"
    print("  ✓ NPZ roundtrip PASSED")

    os.remove(npz_path)
    return True


if __name__ == "__main__":
    passed = 0
    failed = 0

    for test in [test_water, test_ethanol, test_npz_roundtrip]:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            failed += 1
        print()

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
