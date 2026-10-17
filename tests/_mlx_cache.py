"""Fill MLX's buffer cache with NaN so uninitialised kernel output shows up.

MLX recycles freed buffers without clearing them. A request for ``s`` bytes is
served from a cached buffer of size in ``[s, min(2s, s + 2*page))``, so a
kernel that never writes part of its output returns whatever that buffer last
held. On a fresh process that is usually zeros, which is why such bugs pass
tests and then surface mid-pipeline as NaN.

``poison_buffer_cache()`` caches NaN buffers on a sqrt(2)-spaced ladder of
sizes, which covers every request up to ``max_bytes`` (64 KB by default, the
range where the 2x window applies). Above that the reuse window is only two
pages wide, so pass the exact element counts of the large outputs you care
about. Every size gets ``copies`` buffers because the code under test makes
its own same-size allocations (inputs, temporaries) before the one that
matters: the MMFF padded-gradient bug needed 16.
"""
from __future__ import annotations

import mlx.core as mx

_NAN = float("nan")


def _ladder(max_bytes: int) -> list[int]:
    sizes, b = [], 4.0
    while b <= max_bytes * 1.0001:
        sizes.append(max(4, int(round(b)) // 4 * 4))
        b *= 2**0.5
    return sorted(set(sizes))


def poison_buffer_cache(
    *exact_float_counts: int, copies: int = 32, max_bytes: int = 65536
) -> None:
    """Leave NaN-filled float32 buffers in MLX's cache for the next allocations."""
    counts = [s // 4 for s in _ladder(max_bytes)] + [int(n) for n in exact_float_counts]
    junk = [mx.full((n,), _NAN) for n in counts if n > 0 for _ in range(copies)]
    mx.eval(*junk)
    del junk
