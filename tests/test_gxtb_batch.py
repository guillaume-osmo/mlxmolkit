"""Batch scheduling and mixed precision must preserve the validated solver."""
import json
from pathlib import Path
import numpy as np
import pytest
from mlxmolkit.xtb.gxtb_batch import gxtb_energy_batch, _refine_eigenvectors
from mlxmolkit.xtb.scf_gxtb import gxtb_energy


@pytest.mark.parametrize('backend', ['cpu', 'mlx'])
def test_batch_preserves_order_and_oracle_charges(backend):
    if backend == 'mlx':
        pytest.importorskip('mlx_addons')
    cases = json.loads((Path(__file__).parent / 'data/gxtb_oracle_charges.json').read_text())['molecules']
    mols = [(c['atoms'], np.asarray(c['coords_ang'])) for c in cases]
    expected = [gxtb_energy(z, x) for z, x in mols]
    actual = gxtb_energy_batch(mols, backend=backend)
    assert len(actual) == len(expected)
    for case, a, b in zip(cases, actual, expected):
        assert a['converged'] and b['converged']
        np.testing.assert_allclose(a['atom_charges'], b['atom_charges'], atol=2e-6, rtol=0)
        np.testing.assert_allclose(a['atom_charges'], case['oracle_charges'], atol=1e-4, rtol=0)
        assert abs(a['energy_hartree'] - b['energy_hartree']) < 1e-7
        assert a['batch_eigensolver']['backend'] == backend


def test_degenerate_eigenvalues_and_bad_seeds_are_checked():
    rng = np.random.default_rng(561)
    q, _ = np.linalg.qr(rng.normal(size=(6, 6)))
    a = np.stack([q @ np.diag([1, 1, 2, 3, 4, 5]) @ q.T, np.diag(np.arange(6.))])
    seeds = np.stack([q.astype(np.float32), np.ones((6, 6), dtype=np.float32)])
    e, v, bad = _refine_eigenvectors(a, seeds)
    assert bad[1]
    np.testing.assert_allclose(a @ v, v * e[:, None, :], atol=1e-10)
    np.testing.assert_allclose(v.swapaxes(-1, -2) @ v, np.broadcast_to(np.eye(6),a.shape), atol=1e-10)


def test_empty_and_invalid_backend():
    assert gxtb_energy_batch([]) == []
    with pytest.raises(ValueError, match='backend'):
        gxtb_energy_batch([], backend='typo')


def test_cache_scopes_restore_on_completion_and_error():
    from mlxmolkit.xtb import gxtb_scf as solver
    saved = solver._AES_STATIC_CACHE
    z = [8, 1, 1]
    x = np.array([[0., 0., 0.], [.76, .59, 0.], [-.76, .59, 0.]])
    results = gxtb_energy_batch([(z, x.copy()) for _ in range(12)])
    assert all(r['converged'] for r in results)
    assert solver._AES_STATIC_CACHE is saved
    with pytest.raises(NotImplementedError):
        gxtb_energy_batch([(z, x)], charge=1)
    assert solver._AES_STATIC_CACHE is saved
