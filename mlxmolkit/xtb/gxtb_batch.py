"""Experimental batch coordinator for the validated g-xTB SCF.

Chemical operators remain in the scalar state machine. Only generalized
eigensolves are grouped by basis size. The optional MLX backend uses GPU
Jacobi seeds, double-precision refinement, and residual-gated CPU fallback.
This is not yet a fully GPU-resident g-xTB solver.
"""
from collections import defaultdict

import numpy as np


def _refine_eigenvectors(a, vectors):
    """Refine float32 orthogonal eigenvectors in float64, checking every result."""
    v = np.asarray(vectors, dtype=np.float64)
    n = a.shape[-1]
    eye = np.eye(n)
    for _ in range(3):
        # Newton orthogonalization restores V.T V without changing the span.
        for _ in range(2):
            v = v @ (1.5 * eye - 0.5 * (v.swapaxes(-1, -2) @ v))
        t = v.swapaxes(-1, -2) @ a @ v
        e = np.diagonal(t, axis1=-2, axis2=-1).copy()
        gap = e[:, None, :] - e[:, :, None]
        rotation = np.divide(t, gap, out=np.zeros_like(t), where=np.abs(gap) > 1e-7)
        v = v @ (eye + rotation)
    for _ in range(2):
        v = v @ (1.5 * eye - 0.5 * (v.swapaxes(-1, -2) @ v))
    av = a @ v
    e = np.sum(v * av, axis=-2)
    residual = np.max(np.abs(av - v * e[:, None, :]), axis=(-2, -1))
    orth = np.max(np.abs(v.swapaxes(-1, -2) @ v - eye), axis=(-2, -1))
    scale = np.maximum(1., np.max(np.abs(a), axis=(-2, -1)))
    bad = (~np.isfinite(residual)) | (residual > 5e-11 * scale) | (orth > 5e-11)
    if np.any(bad):
        e[bad], v[bad] = np.linalg.eigh(a[bad])
    order = np.argsort(e, axis=-1)
    e = np.take_along_axis(e, order, axis=-1)
    v = np.take_along_axis(v, order[:, None, :], axis=-1)
    return e, v, bad


def gxtb_energy_batch(molecules, *, backend='cpu', **kwargs):
    """Return ordered g-xTB results for (atomic_numbers, coords_ang) pairs.

    ``backend='mlx'`` requires mlx-addons. Each result records eigensolver
    usage; matrices beyond its GPU limit and failed refinements use float64
    CPU eigh. Chemistry flags and convergence tolerance use validated defaults.
    """
    if backend not in {'cpu', 'mlx'}:
        raise ValueError("backend must be 'cpu' or 'mlx'")
    if not molecules:
        return []
    try:
        from scipy.linalg import cholesky
    except ImportError:
        def cholesky(a, **_kwargs):
            return np.linalg.cholesky(a)
    from . import gxtb_scf as solver
    if backend == 'mlx':
        import mlx.core as mx
        from mlx_addons.linalg import jacobi_eigh, JACOBI_MAX_N
    cfg = dict(solver.SOLVE_KWARGS)
    cfg.update(kwargs)
    streams = [solver._gxtb_energy_steps(z, np.asarray(x, dtype=np.float64), **cfg)
               for z, x in molecules]
    results = [None] * len(streams)
    counts = [dict(gpu_eigh=0, cpu_eigh=0, refined_fallback=0) for _ in streams]
    metrics = {}
    requests = {}
    # The scalar solver caps geometry caches at eight entries. Interleaving
    # larger batches would evict every molecule before its next iteration.
    # Give each state machine its own cache scope, restoring module state on
    # every suspension and on errors. Element-only tables stay shared.
    cache_names = ('_H0_BASIS_CACHE', '_REP_STATIC_CACHE', '_D4_MOL_CACHE',
                   '_CHOL_CACHE', '_TB3_TAU_CACHE', '_MP_CACHE_FAST',
                   '_AES_GAB_CACHE', '_AES_STATIC_CACHE', '_KMAT_SH_CACHE',
                   '_AES_GEO_CACHE', '_MFX_SHELLSUM_PLAN', '_KFOCK_STATIC',
                   '_AES_MSTACK')
    caches = [{name: {} for name in cache_names} for _ in streams]

    def advance(i, value=None, first=False):
        prior = {name: getattr(solver, name) for name in cache_names}
        for name in cache_names:
            setattr(solver, name, caches[i][name])
        try:
            requests[i] = next(streams[i]) if first else streams[i].send(value)
        except StopIteration as done:
            results[i] = done.value
            results[i]['batch_eigensolver'] = dict(backend=backend, **counts[i])
            requests.pop(i, None)
        finally:
            for name, cache in prior.items():
                setattr(solver, name, cache)

    try:
        for i in range(len(streams)):
            advance(i, first=True)
        while requests:
            groups = defaultdict(list)
            answers = {}
            for i, (f, s) in requests.items():
                old = metrics.get(i)
                if old is None or old[0] is not s:
                    try:
                        li = np.ascontiguousarray(np.linalg.inv(cholesky(s, lower=True, check_finite=False)))
                    except np.linalg.LinAlgError:
                        answers[i] = solver._solve_generalized(f, s)
                        counts[i]['cpu_eigh'] += 1
                        continue
                    metrics[i] = (s, li)
                else:
                    li = old[1]
                a = li @ f @ li.T
                # The scalar LAPACK call uses the lower triangle.
                a = np.tril(a) + np.tril(a, -1).T
                groups[len(a)].append((i, a, li))
            for n, group in groups.items():
                a = np.stack([row[1] for row in group])
                if backend == 'mlx' and n <= JACOBI_MAX_N:
                    _, seeds = jacobi_eigh(mx.array(a.astype(np.float32)))
                    mx.eval(seeds)
                    e, v, fallback = _refine_eigenvectors(a, np.asarray(seeds))
                    for k, (i, _, _) in enumerate(group):
                        counts[i]['gpu_eigh'] += 1
                        counts[i]['refined_fallback'] += int(fallback[k])
                else:
                    e, v = np.linalg.eigh(a)
                    for i, _, _ in group:
                        counts[i]['cpu_eigh'] += 1
                for k, (i, _, li) in enumerate(group):
                    answers[i] = (e[k], li.T @ v[k])
            for i in list(requests):
                advance(i, answers[i])
        return results
    finally:
        for stream in streams:
            stream.close()
