"""Fingerprint Tanimoto and neighbour lists: a row must not depend on its batch.

Tanimoto on bit fingerprints is a ratio of two integer popcounts. The kernels
compute ``float(inter) / (float(union) + 1e-12f)``; for ``union >= 1`` the
``1e-12`` is below one float32 ulp, so every result is the correctly rounded
``inter / union`` and the numpy reference below reproduces it bit for bit.
All comparisons here are therefore exact.

Pinned:

* The full self-similarity matrix (16x16 tiled kernel, the default) against
  one-row queries (which take the naive kernel, since ``Na != Nb``) and
  against numpy, with ``N`` not a multiple of the tile so rows sit in full
  tiles, in the ragged last tile, and on every tile boundary. The docstring
  of ``tanimoto_matrix_metal_u32`` warns of tile-boundary issues; this is
  where they would show.
* ``upper_triangle_only`` returns exactly ``triu(full, 1)``, with the lower
  triangle written as zeros (not left as whatever the buffer held).
* The neighbour-list builders bake the cutoff into Metal source under a
  fixed kernel name. Calling them with a new cutoff in the same process must
  not reuse the kernel compiled for the previous one.
* Cutoffs given as an ``int`` or a numpy scalar must work like the float.
  An int used to be pasted into the source as ``1f`` (not a valid Metal
  literal), and the compiled path passed np.float32 straight to mx.array,
  which rejects it. In ``tanimoto_blockwise`` the broken kernel was then
  cached under ``1 == 1.0``, so a later ``cutoff=1.0`` failed too.

Each kernel output is computed right after filling MLX's buffer cache with
NaN, so an entry the kernel never writes shows up as a mismatch.
"""
from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from mlxmolkit.butina_metal import build_neighbor_list_metal
from mlxmolkit.fused_tanimoto_nlist import fused_neighbor_list_metal
from mlxmolkit.tanimoto_blockwise import tanimoto_neighbors_blockwise
from mlxmolkit.tanimoto_metal_u32 import TILE, tanimoto_matrix_metal_u32
from tests._mlx_cache import poison_buffer_cache

# 3 full 16-row tiles plus a ragged one of 5.
N = 3 * TILE + 5
ZERO_ROW, DUPLICATE_OF_0 = 7, N - 1


def _fingerprints(n, nwords, seed=0):
    """Families of related fingerprints, so similarities spread over ~0.4-1.

    Each row is a random parent with 2-40% of its bits flipped; unrelated
    rows sit near 1/3. One row is all zeros (union 0) and one duplicates
    row 0 (similarity exactly 1).
    """
    rng = np.random.default_rng(seed)
    nbits = 32 * nwords
    parents = rng.random((n // 4 + 1, nbits)) < 0.5
    bits = parents[rng.integers(0, len(parents), n)]
    bits ^= rng.random((n, nbits)) < rng.uniform(0.02, 0.4, (n, 1))
    bits[ZERO_ROW] = False
    bits[DUPLICATE_OF_0] = bits[0]
    return np.packbits(bits, axis=1, bitorder="little").view(np.uint32)


def _ref_tanimoto(fp):
    bits = np.unpackbits(fp.view(np.uint8), axis=1, bitorder="little").astype(np.int64)
    inter = bits @ bits.T
    union = bits.sum(1)[:, None] + bits.sum(1)[None, :] - inter
    out = np.zeros(inter.shape, np.float32)
    np.divide(inter.astype(np.float32), union.astype(np.float32), out=out, where=union > 0)
    return out


def _ref_neighbours(sim, cutoff):
    """CSR of ``sim[i, j] >= cutoff, j != i``, each row's indices ascending."""
    hit = sim >= np.float32(cutoff)
    np.fill_diagonal(hit, False)
    offsets = np.zeros(len(sim) + 1, np.int32)
    np.cumsum(hit.sum(1), out=offsets[1:])
    return offsets, np.nonzero(hit)[1].astype(np.int32)


@pytest.mark.parametrize("nwords", [7, 32, 64], ids=["224bit", "1024bit", "2048bit"])
def test_tiled_matrix_rows_match_one_row_queries_and_reference(nwords):
    fp_np = _fingerprints(N, nwords)
    fp = mx.array(fp_np)
    ref = _ref_tanimoto(fp_np)

    poison_buffer_cache(N * N)
    full = np.array(tanimoto_matrix_metal_u32(fp))
    np.testing.assert_array_equal(full, ref)

    for i in range(N):
        poison_buffer_cache()
        row = np.array(tanimoto_matrix_metal_u32(fp[i : i + 1], fp))
        np.testing.assert_array_equal(row[0], full[i], err_msg=f"row {i}")


@pytest.mark.parametrize("use_tiled", [True, False], ids=["tiled", "naive"])
def test_upper_triangle_is_exactly_the_strict_upper_part(use_tiled):
    fp = mx.array(_fingerprints(N, 32))
    full = np.array(tanimoto_matrix_metal_u32(fp, use_tiled=use_tiled))

    poison_buffer_cache(N * N)
    upper = np.array(
        tanimoto_matrix_metal_u32(fp, use_tiled=use_tiled, upper_triangle_only=True)
    )
    np.testing.assert_array_equal(upper, np.triu(full, 1))


_NEIGHBOUR_LISTS = {
    "butina_metal": lambda fp, c: build_neighbor_list_metal(tanimoto_matrix_metal_u32(fp), c),
    "fused_jit": lambda fp, c: fused_neighbor_list_metal(fp, c, compiled=False),
    "fused_compiled": lambda fp, c: fused_neighbor_list_metal(fp, c, compiled=True),
    # Small blocks so the 53 rows span several column and row tiles.
    "blockwise": lambda fp, c: tanimoto_neighbors_blockwise(fp, c, block_size=16),
}


def _assert_neighbours(path, fp, cutoff, ref_sim):
    poison_buffer_cache()
    offsets, indices = _NEIGHBOUR_LISTS[path](fp, cutoff)
    ref_offsets, ref_indices = _ref_neighbours(ref_sim, cutoff)
    np.testing.assert_array_equal(offsets, ref_offsets, err_msg=f"cutoff {cutoff!r}")
    np.testing.assert_array_equal(indices, ref_indices, err_msg=f"cutoff {cutoff!r}")


@pytest.mark.parametrize("path", list(_NEIGHBOUR_LISTS))
def test_neighbour_list_follows_each_new_cutoff(path):
    fp_np = _fingerprints(N, 32)
    ref_sim = _ref_tanimoto(fp_np)
    fp = mx.array(fp_np)
    # The two cutoffs must select different pairs, or the test proves nothing.
    assert _ref_neighbours(ref_sim, 0.5)[0][-1] > _ref_neighbours(ref_sim, 0.8)[0][-1] > 0

    for cutoff in (0.5, 0.8, 0.5):
        _assert_neighbours(path, fp, cutoff, ref_sim)


@pytest.mark.parametrize("path", list(_NEIGHBOUR_LISTS))
def test_cutoff_given_as_int_or_numpy_scalar(path):
    """Each spelling of a cutoff, then the float 1.0 that an int 1 equals."""
    fp_np = _fingerprints(N, 32)
    fp, ref_sim = mx.array(fp_np), _ref_tanimoto(fp_np)
    for cutoff in (1, np.float32(0.8), np.float64(0.5), 1.0):
        _assert_neighbours(path, fp, cutoff, ref_sim)
