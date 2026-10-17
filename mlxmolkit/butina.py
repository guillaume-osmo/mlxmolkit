"""
Butina clustering (greedy) using a CSR neighbor list.

Pipeline (nvMolKit-style): Morgan (CPU) → Fused Tanimoto→CSR (Metal) → Butina greedy (CPU).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import mlx.core as mx


@dataclass
class ButinaResult:
    clusters: List[Tuple[int, ...]]
    cutoff: float


def butina_from_neighbor_list_csr(
    offsets: np.ndarray,
    indices: np.ndarray,
    n: int,
    cutoff: float,
    *,
    reordering: bool = True,
) -> ButinaResult:
    """
    Butina greedy clustering from a CSR neighbor list.

    Reproduces ``rdkit.ML.Cluster.Butina.ClusterData`` exactly -- same clusters,
    same member order within a cluster, same cluster order -- for a neighbor
    list built with the same threshold.

    Parameters
    ----------
    offsets, indices : np.ndarray
        CSR neighbor list.  ``indices[offsets[i]:offsets[i+1]]`` must be the
        neighbors of ``i`` in **ascending order**, excluding ``i`` itself.
    n : int
        Number of points.
    cutoff : float
        Recorded on the result; the thresholding already happened upstream.
    reordering : bool, default True
        Mirrors RDKit's ``reordering`` flag.  ``True`` re-picks the next cluster
        centroid by the number of *still unassigned* neighbors; ``False`` picks
        centroids once, from the initial degrees.

        Note the default differs from RDKit's (``False``).  ``True`` is both the
        better clustering and the cheap path here, since counts are maintained
        incrementally; pass ``reordering=False`` for bit-exact parity with a
        default RDKit call.

    Notes
    -----
    Ties on neighbor count are broken by **highest index**, matching RDKit's
    ``sorted_indices.sort(reverse=True)`` on ``(count, index)`` pairs.  RDKit
    counts a point as its own neighbor, so its counts are ``degree + 1``; the
    constant offset does not affect the ordering.

    The ``reordering=True`` path updates counts by walking the *removed*
    members' neighbors -- O(cluster_size x avg_degree) -- instead of re-sorting
    all remaining points, which is what RDKit does.
    """
    offsets = np.asarray(offsets)
    indices = np.asarray(indices)

    counts = (offsets[1:] - offsets[:-1]).astype(np.int64)
    alive = np.ones(n, dtype=np.bool_)
    clusters: List[Tuple[int, ...]] = []

    def _emit(center: int) -> np.ndarray:
        """Form the cluster seeded at ``center`` and mark its members assigned."""
        nbrs = indices[offsets[center]:offsets[center + 1]]
        alive_nbrs = nbrs[alive[nbrs]]
        members = np.empty(1 + len(alive_nbrs), dtype=np.int64)
        members[0] = center
        members[1:] = alive_nbrs
        alive[members] = False
        clusters.append(tuple(members.tolist()))
        return members

    if reordering:
        masked_counts = counts.copy()
        while True:
            # argmax over the reversed view -> highest index among the maxima,
            # which is RDKit's tie-break.
            best = n - 1 - int(np.argmax(masked_counts[::-1]))
            if masked_counts[best] <= 0:
                # Nothing left with a live neighbor; the rest are singletons.
                break

            members = _emit(best)
            masked_counts[members] = -1

            for m in members:
                m_nbrs = indices[offsets[m]:offsets[m + 1]]
                alive_m_nbrs = m_nbrs[alive[m_nbrs]]
                if len(alive_m_nbrs) > 0:
                    counts[alive_m_nbrs] -= 1
                    masked_counts[alive_m_nbrs] = counts[alive_m_nbrs]
    else:
        # Descending degree, ties by descending index; centroids fixed up front.
        order = np.lexsort((-np.arange(n, dtype=np.int64), -counts))
        for center in order:
            if counts[center] <= 0:
                break  # sorted, so everything after this is isolated too
            if not alive[center]:
                continue
            _emit(int(center))

    # Isolated points, in RDKit's pop order (descending index).
    for s in np.flatnonzero(alive)[::-1]:
        clusters.append((int(s),))

    return ButinaResult(clusters=clusters, cutoff=cutoff)


def butina_from_similarity_matrix(
    sim: np.ndarray,
    cutoff: float,
    *,
    reordering: bool = True,
) -> ButinaResult:
    """Butina from a dense similarity matrix (for testing).

    ``reordering`` is forwarded to :func:`butina_from_neighbor_list_csr`.
    """
    N = sim.shape[0]
    nbrs = []
    for i in range(N):
        js = np.where(sim[i] >= cutoff)[0]
        js = js[js != i]
        nbrs.append(js)

    offsets = np.zeros(N + 1, dtype=np.int32)
    for i, js in enumerate(nbrs):
        offsets[i + 1] = offsets[i] + len(js)
    indices = np.concatenate(nbrs) if any(len(j) > 0 for j in nbrs) else np.array([], dtype=np.int64)
    return butina_from_neighbor_list_csr(
        offsets, indices, N, cutoff, reordering=reordering
    )


def butina_tanimoto_mlx(
    fp_bytes: mx.array,
    cutoff: float,
    *,
    block_size: int | None = None,
    max_memory_bytes: int | None = None,
    reordering: bool = True,
) -> ButinaResult:
    """
    Full pipeline: fp uint8 → uint32 → Tanimoto → CSR → Butina greedy.

    Automatically selects the best strategy based on N:

    * **N <= 100k**: Single-dispatch fused Metal kernel (fastest).
    * **N > 100k**: Divide-and-conquer blockwise tiling with memory-adaptive
      block sizes so the full N×N matrix is never materialised.

    Parameters
    ----------
    fp_bytes : mx.array, shape (N, nbytes), dtype uint8
    cutoff : float
        Similarity threshold.
    block_size : int, optional
        Override tile side-length for the blockwise path.
    max_memory_bytes : int, optional
        Memory budget for one tile (blockwise path).
    reordering : bool, default True
        Centroid-selection rule, see :func:`butina_from_neighbor_list_csr`.
        Pass ``False`` for bit-exact parity with a default RDKit
        ``Butina.ClusterData`` call at ``distThresh = 1 - cutoff``.
    """
    from .fp_uint32 import fp_uint8_to_uint32

    fp_u32 = fp_uint8_to_uint32(fp_bytes)
    N = int(fp_u32.shape[0])

    # For moderate N the single-dispatch fused kernel is fastest
    if N <= 100_000:
        from .fused_tanimoto_nlist import fused_neighbor_list_metal
        offsets, indices = fused_neighbor_list_metal(fp_u32, cutoff)
    else:
        from .tanimoto_blockwise import tanimoto_neighbors_blockwise
        offsets, indices = tanimoto_neighbors_blockwise(
            fp_u32, cutoff,
            block_size=block_size,
            max_memory_bytes=max_memory_bytes,
        )

    return butina_from_neighbor_list_csr(
        offsets, indices, N, cutoff, reordering=reordering
    )
