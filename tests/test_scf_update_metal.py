"""Fused Metal density update versus the explicit MLX expressions it replaces."""
import mlx.core as mx
import numpy as np
import pytest
from mlxmolkit.nddo.scf_update_metal import density_update


@pytest.mark.parametrize('width', [6, 18, 35, 64])
@pytest.mark.parametrize('iteration', [0, 1, 2, 19])
def test_density_update_matches_unfused(width, iteration):
    rng = np.random.default_rng(572)
    old = rng.normal(size=(4, width, width)).astype(np.float32)
    new = old.copy()
    new[1] += 1e-3
    new[2] += 1e-7
    new[3] -= 1e-2
    previous = mx.array(old)
    proposed = mx.array(new)
    before = mx.array([False, False, True, True])
    counts = mx.array([100, 100, 1, 3], dtype=mx.int32)
    p, conv, iters, diff = density_update(previous, proposed, before, counts, iteration, 1e-5)
    reference_dp = mx.max(mx.abs(proposed - previous), axis=(-2, -1))
    now = reference_dp < 1e-5
    reference_p = .5 * proposed + .5 * previous if iteration < 2 else proposed
    reference_p = mx.where(before[:, None, None], previous, reference_p)
    expected = [reference_p, before | now, mx.where(now & ~before, iteration + 1, counts), reference_dp]
    for actual, wanted in zip([p, conv, iters, diff], expected):
        np.testing.assert_allclose(np.asarray(actual), np.asarray(wanted), rtol=0, atol=1e-7)


@pytest.mark.parametrize('fused', ['0', '1'])
def test_completed_density_is_frozen_while_another_molecule_iterates(monkeypatch, fused):
    from mlxmolkit.nddo import scf
    calls = 0

    def changing_eigenvectors(matrix, **kwargs):
        nonlocal calls
        calls += 1
        n = matrix.shape[-1]
        c = np.broadcast_to(np.eye(n), (2, n, n)).copy()
        # Molecule zero is stationary through iteration four, then its
        # proposed occupied subspace changes. Molecule one keeps moving.
        for mol, theta in [(0, .7 if calls > 4 else 0.), (1, calls * .17)]:
            cs, sn = np.cos(theta), np.sin(theta)
            c[mol, 0, 0] = c[mol, -1, -1] = cs
            c[mol, 0, -1], c[mol, -1, 0] = -sn, sn
        return mx.broadcast_to(mx.arange(n), (2, n)), mx.array(c.astype(np.float32))

    monkeypatch.setattr(scf, '_batched_symmetric_eigh', changing_eigenvectors)
    monkeypatch.setenv('MLXMOLKIT_SCF_FUSED_UPDATE', fused)
    x = np.array([[0., 0., 0.], [.76, .59, 0.], [-.76, .59, 0.]])
    result = scf.rm1_energy_batch_mlx([([8, 1, 1], x), ([8, 1, 1], x.copy())],
                                   method='PM6', max_iter=6, density_solver='eigh')
    assert result[0]['converged']
    assert not result[1]['converged']
    np.testing.assert_allclose(result[0]['density'], np.diag([2., 2., 2., 2., 0., 0.]), atol=1e-6)
