"""Check every tensor entry, symmetry and live batch integration of the Metal rotation."""
import numpy as np
import mlx.core as mx
import pytest
from mlxmolkit.nddo.rotation_batch import rotate_xx_batch
from mlxmolkit.nddo.rotation_metal import rotate_xx_batch_fused_metal


@pytest.mark.parametrize('n', [0, 1, 37, 4096])
def test_fused_rotation_matches_double_reference(n):
    rng = np.random.default_rng(562)
    ri = rng.normal(size=(n, 22))
    rotations = np.linalg.qr(rng.normal(size=(n, 3, 3)))[0]
    args = [ri, rotations[:, 0], rotations[:, 1], rotations[:, 2]]
    result = np.asarray(rotate_xx_batch_fused_metal(*[mx.array(a.astype(np.float32)) for a in args]))
    reference = rotate_xx_batch(*args)
    np.testing.assert_allclose(result, reference, rtol=2e-6, atol=2e-6)
    np.testing.assert_array_equal(result, result.swapaxes(1, 2))
    np.testing.assert_array_equal(result, result.swapaxes(3, 4))


def test_opt_in_batch_rotations_preserve_energy(monkeypatch):
    from mlxmolkit.nddo.scf import nddo_energy_batch
    from tests.test_nddo_batch_parity import _embed
    mols = [_embed(s) for s in ['CCO', 'c1ccccc1', 'CSC', 'CC(=O)OCC']]
    monkeypatch.setenv('MLXMOLKIT_BATCH_ROTATION_METAL', '0')
    ref = nddo_energy_batch(mols, method='PM6', density_solver='eigh')
    monkeypatch.setenv('MLXMOLKIT_BATCH_ROTATION_METAL', '1')
    actual = nddo_energy_batch(mols, method='PM6', density_solver='eigh')
    for a, b in zip(actual, ref):
        assert a['converged'] and b['converged']
        np.testing.assert_allclose(a['charges'], b['charges'], atol=1e-4, rtol=0)
        assert abs(a['heat_of_formation_kcal'] - b['heat_of_formation_kcal']) < .01
