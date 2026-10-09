"""The fused Metal CHEESE kernels must agree with the reference MLX path."""

import itertools

import mlx.core as mx
import numpy as np
import pytest

from mlxmolkit.cheese import cheese_batch, cheese_similarity_matrix_mlx
from mlxmolkit.cheese_fused import cheese_max_similarity_metal, cheese_similarity_matrix_metal

TOL = 1.0e-4  # float32 summation order; similarities live in [0, 1]


def _molecules(n, seed, min_atoms=3, max_atoms=40):
    rng = np.random.default_rng(seed)
    atoms, coords, charges = [], [], []
    for _ in range(n):
        k = int(rng.integers(min_atoms, max_atoms + 1))
        atoms.append(rng.choice([6, 6, 6, 7, 8, 9, 16, 17], k))
        coords.append(rng.normal(scale=2.5, size=(k, 3)))
        q = rng.normal(scale=0.3, size=k)
        charges.append(q - q.mean())
    return cheese_batch(atoms, coords, charges)


@pytest.fixture(scope="module")
def batches():
    return _molecules(13, 1), _molecules(37, 2)


@pytest.mark.parametrize(
    "shape_metric,esp_metric,mapped,weights",
    list(itertools.product(["tanimoto", "carbo"], ["carbo", "tanimoto"], [True, False], [(1.0, 1.0), (0.3, 0.7), (1.0, 0.0)])),
)
def test_matrix_matches_reference(batches, shape_metric, esp_metric, mapped, weights):
    probe, ref = batches
    kw = dict(shape_metric=shape_metric, electrostatic_metric=esp_metric, map_electrostatic_to_unit=mapped,
              shape_weight=weights[0], electrostatic_weight=weights[1])
    want = cheese_similarity_matrix_mlx(probe, ref, **kw)
    got = cheese_similarity_matrix_metal(probe, ref, **kw)
    for channel in ("shape", "electrostatic", "combined"):
        assert float(mx.max(mx.abs(getattr(want, channel) - getattr(got, channel)))) < TOL, channel


@pytest.mark.parametrize("query_block", [1, 4, 13, 64])
def test_max_matches_reference_max_over_probes(batches, query_block):
    probe, ref = batches
    want = cheese_similarity_matrix_mlx(probe, ref)
    got = cheese_max_similarity_metal(probe, ref, query_block=query_block)
    assert float(mx.max(mx.abs(mx.max(want.shape, axis=0) - got.shape))) < TOL
    assert float(mx.max(mx.abs(mx.max((want.electrostatic + 1.0) / 2.0, axis=0) - got.electrostatic))) < TOL
    # The combined maximum is over per-overlay combined scores, NOT the mean of
    # the per-channel maxima.
    assert float(mx.max(mx.abs(mx.max(want.combined, axis=0) - got.combined))) < TOL


def test_self_similarity_is_one(batches):
    probe, _ = batches
    got = cheese_similarity_matrix_metal(probe, probe)
    diag = np.diag(np.array(got.shape))
    assert np.allclose(diag, 1.0, atol=TOL)


def test_non_prefix_mask_is_refused(batches):
    probe, ref = batches
    mask = np.array(probe.mask).copy()
    mask[0, 0] = 0  # a hole before a real atom
    bad = type(probe)(probe.atomic_numbers, probe.coords, probe.charges, mx.array(mask), probe.ids)
    with pytest.raises(ValueError, match="prefix masks"):
        cheese_similarity_matrix_metal(bad, ref)
