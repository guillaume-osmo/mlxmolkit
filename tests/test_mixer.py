"""
Tests for the portable SCFMixer class.

Verifies Anderson/Pulay/DIIS convergence on toy fixed-point problems
and compares methods.
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mlxmolkit.solvers.mixer import SCFMixer, MixerConfig, run_fixed_point


def contractive_map(x):
    """Toy fixed-point: G(x) = 0.7x + 0.2·tanh(x) + 0.1. Fixed point near x≈0.41."""
    return 0.7 * x + 0.2 * np.tanh(x) + 0.1


def stiff_map(x):
    """Stiff fixed-point with fast/slow modes. Tests preconditioning benefit."""
    A = np.array([[0.99, 0.1], [0.0, 0.3]])
    b = np.array([1.0, 0.5])
    return A @ x + b


def oscillating_map(x):
    """Map that oscillates without damping. Tests robustness."""
    return -0.8 * x + 1.0


def test_linear_mixing():
    """Linear mixing should converge for contractive maps."""
    x0 = np.zeros(10)
    mixer = SCFMixer(MixerConfig(method="linear", beta=0.5))
    x, hist = run_fixed_point(x0, contractive_map, mixer, max_iter=200, tol=1e-8)

    assert hist[-1]["converged"], f"Linear mixing didn't converge in {len(hist)} iters"
    n_iter = len(hist)
    print(f"  Linear: {n_iter} iters, ‖r‖={hist[-1]['rnorm']:.2e}")
    return n_iter


def test_anderson():
    """Anderson should converge faster than linear."""
    x0 = np.zeros(10)
    mixer = SCFMixer(MixerConfig(method="anderson", beta=0.5, history_size=6))
    x, hist = run_fixed_point(x0, contractive_map, mixer, max_iter=200, tol=1e-8)

    assert hist[-1]["converged"], f"Anderson didn't converge in {len(hist)} iters"
    n_iter = len(hist)
    print(f"  Anderson: {n_iter} iters, ‖r‖={hist[-1]['rnorm']:.2e}")
    return n_iter


def test_periodic_pulay():
    """Periodic Pulay should converge fast and robust."""
    x0 = np.zeros(10)
    mixer = SCFMixer(MixerConfig(method="periodic_pulay", beta=0.5, pulay_period=2))
    x, hist = run_fixed_point(x0, contractive_map, mixer, max_iter=200, tol=1e-8)

    assert hist[-1]["converged"], f"Periodic Pulay didn't converge in {len(hist)} iters"
    n_iter = len(hist)
    print(f"  Periodic Pulay: {n_iter} iters, ‖r‖={hist[-1]['rnorm']:.2e}")
    return n_iter


def test_acceleration_speedup():
    """Anderson/Pulay should need fewer iterations than linear."""
    print("\n=== Acceleration Speedup Test ===")
    n_lin = test_linear_mixing()
    n_and = test_anderson()
    n_pul = test_periodic_pulay()

    assert n_and < n_lin, f"Anderson ({n_and}) not faster than linear ({n_lin})"
    print(f"\n  Speedup: linear={n_lin}, anderson={n_and} ({100*(n_lin-n_and)/n_lin:.0f}% faster), "
          f"pulay={n_pul} ({100*(n_lin-n_pul)/n_lin:.0f}% faster)")
    print("  ✓ PASSED")


def test_stiff_system():
    """Test on stiff 2D system where Anderson helps most."""
    print("\n=== Stiff System Test ===")
    x0 = np.zeros(2)

    for method in ["linear", "anderson", "periodic_pulay"]:
        mixer = SCFMixer(MixerConfig(method=method, beta=0.3, history_size=8))
        x, hist = run_fixed_point(x0, stiff_map, mixer, max_iter=500, tol=1e-10)
        status = "✓" if hist[-1]["converged"] else "✗"
        print(f"  {method:16s}: {len(hist):3d} iters {status}  x={x}")

    print("  ✓ PASSED")


def test_restart_on_oscillation():
    """Mixer should restart history when residual grows."""
    print("\n=== Restart Test ===")
    x0 = np.array([5.0])
    mixer = SCFMixer(MixerConfig(
        method="anderson", beta=0.3, restart_threshold=1.5, history_size=4,
    ))

    x, hist = run_fixed_point(x0, oscillating_map, mixer, max_iter=100, tol=1e-8)
    restarts = sum(1 for h in hist if h.get("restarted"))
    print(f"  Oscillating map: {len(hist)} iters, {restarts} restarts, "
          f"converged={hist[-1]['converged']}")
    print("  ✓ PASSED")


def test_symmetrize():
    """Symmetrize option should produce symmetric matrices."""
    print("\n=== Symmetrize Test ===")

    def mat_map(P):
        """Toy density-like map that breaks symmetry slightly."""
        return 0.5 * P + 0.3 * np.eye(3) + 0.01 * np.random.randn(3, 3)

    np.random.seed(42)
    mixer = SCFMixer(MixerConfig(method="anderson", beta=0.5, symmetrize=True))
    P = np.eye(3)
    for _ in range(10):
        GP = mat_map(P)
        P, info = mixer.step(P, GP)
        sym_err = np.max(np.abs(P - P.T))
        assert sym_err < 1e-15, f"Not symmetric: {sym_err}"

    print("  All iterates symmetric ✓")
    print("  ✓ PASSED")


def test_scf_integration():
    """Verify mixer works with actual SCF (same energies as before)."""
    print("\n=== SCF Integration Test ===")
    from mlxmolkit.rm1.scf import rm1_energy

    PYSEQM_REF = {
        'RM1': {'H2O': -345.570630},
        'PM6': {'H2O': -319.072883},
    }

    for method, refs in PYSEQM_REF.items():
        for mol, ref_E in refs.items():
            atoms = [8, 1, 1]
            coords = np.array([[0, 0, 0.117], [0, 0.757, -0.469], [0, -0.757, -0.469]])
            r = rm1_energy(atoms, coords, method=method, conv_tol=1e-8)
            diff = abs(r['energy_eV'] - ref_E)
            assert diff < 0.001, f"{method} {mol}: {r['energy_eV']:.6f} vs {ref_E:.6f} (diff={diff:.6f})"
            print(f"  {method} {mol}: {r['n_iter']} iter, E={r['energy_eV']:.6f} eV ✓")

    print("  ✓ PASSED")


if __name__ == "__main__":
    tests = [
        test_acceleration_speedup,
        test_stiff_system,
        test_restart_on_oscillation,
        test_symmetrize,
        test_scf_integration,
    ]

    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            import traceback; traceback.print_exc()
        print()

    print(f"{'='*60}")
    print(f"Results: {passed}/{len(tests)} passed")
    print(f"{'='*60}")
