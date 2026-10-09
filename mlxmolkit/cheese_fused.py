"""Fused Metal kernels for CHEESE shape + electrostatic similarity.

The reference path in :mod:`mlxmolkit.cheese` builds every atom-pair term as a
tensor of shape ``(n_probe, n_ref, atoms, atoms[, 9])`` before summing it. That
is memory-bandwidth bound: on 26-heavy-atom molecules it scores about 0.14 M
molecule pairs per second, and a 16 x 1024 tile already needs a large buffer.

Here one GPU thread owns one (probe, reference) molecule pair, walks both atom
lists in registers, and accumulates the Gaussian shape overlap and the ESP-Sim
charge overlap in one pass. Nothing per atom pair is ever written to memory.

Two entry points:

* :func:`cheese_similarity_matrix_metal` - drop-in for
  :func:`mlxmolkit.cheese.cheese_similarity_matrix_mlx`: the full
  ``(n_probe, n_ref)`` shape / electrostatic / combined matrices.
* :func:`cheese_max_similarity_metal` - for screening: per reference molecule,
  the MAXIMUM similarity over all probes, per channel. The probe axis is reduced
  on the GPU, so the probe x reference matrix is never stored either. The
  combined maximum is taken over per-overlay combined scores, not assembled from
  the per-channel maxima.

Formulas are identical to :mod:`mlxmolkit.cheese` (same atom Gaussians, same
ESP-Sim 3 x 3 Gaussian expansion of 1/r, same metric definitions and clipping),
so results agree with the reference to float32 accumulation order.
"""

from __future__ import annotations

import numpy as np
import mlx.core as mx

from mlxmolkit.cheese import (
    CheeseBatch,
    CheeseSimilarityResult,
    _ESPSIM_GAUSS_A_NP,
    _ESPSIM_GAUSS_B_NP,
    atom_gaussian_parameters_mlx,
    electrostatic_self_overlap_mlx,
    gaussian_shape_self_overlap_mlx,
)

_SHAPE_METRICS = {"tanimoto": 0, "carbo": 1}
_ESP_METRICS = {"carbo": 0, "tanimoto": 1}


def _esp_terms() -> str:
    """The 9 ESP-Sim terms folded to 6 unique ones (the 3 x 3 table is symmetric)."""

    a, b = _ESPSIM_GAUSS_A_NP, _ESPSIM_GAUSS_B_NP
    if not (np.allclose(a, a.T) and np.allclose(b, b.T)):
        raise AssertionError("ESP-Sim coefficient tables are expected to be symmetric")
    terms = []
    for i in range(3):
        for j in range(i, 3):
            mult = 1.0 if i == j else 2.0
            terms.append(f"{mult * a[i, j]:.10e}f * metal::exp({b[i, j]:.10e}f * d2)")
    return " + ".join(terms)


_PAIR_LOOP = """
    float so = 0.0f;
    float eo = 0.0f;
    const uint pa = p_off[i];
    const uint na = p_off[i + 1] - pa;
    const uint rb = r_off[j];
    const uint nb = r_off[j + 1] - rb;
    for (uint a = 0; a < na; ++a) {
        const uint ia = pa + a;
        const float ax = p_xyz[3 * ia], ay = p_xyz[3 * ia + 1], az = p_xyz[3 * ia + 2];
        const float ea = p_exp[ia], aa = p_amp[ia], qa = p_chg[ia];
        for (uint b = 0; b < nb; ++b) {
            const uint ib = rb + b;
            const float dx = ax - r_xyz[3 * ib];
            const float dy = ay - r_xyz[3 * ib + 1];
            const float dz = az - r_xyz[3 * ib + 2];
            const float d2 = dx * dx + dy * dy + dz * dz;
            const float eb = r_exp[ib];
            const float es = metal::max(ea + eb, 1.0e-12f);
            const float pref = PI15 / (es * metal::sqrt(es));
            so += aa * r_amp[ib] * pref * metal::exp(-(ea * eb / es) * d2);
            eo += qa * r_chg[ib] * (ESP_TERMS);
        }
    }
"""

_SIM_FROM_OVERLAPS = """
    const float eps = 1.0e-8f;
    float shape;
    if (SHAPE_METRIC == 0) {
        shape = so / metal::max(p_sself[i] + r_sself[j] - so, eps);
    } else {
        const float den = metal::sqrt(metal::max(p_sself[i], eps) * metal::max(r_sself[j], eps));
        shape = so / metal::max(den, eps);
    }
    shape = metal::clamp(shape, 0.0f, 1.0f);
    float esp;
    if (ESP_METRIC == 0) {
        const float den = metal::sqrt(metal::max(p_eself[i], eps) * metal::max(r_eself[j], eps));
        esp = den > eps ? metal::clamp(eo / den, -1.0f, 1.0f) : 0.0f;
    } else {
        esp = eo / metal::max(p_eself[i] + r_eself[j] - eo, eps);
    }
    // Map to [0, 1] with ESP-Sim's published ranges, as _renormalize_electrostatic_mlx.
    const float esp_unit = (ESP_METRIC == 0) ? (esp + 1.0f) * 0.5f : (esp + (1.0f / 3.0f)) * 0.75f;
    const float combined = (weights[0] * shape + weights[1] * (MAP_ESP ? esp_unit : esp)) / (weights[0] + weights[1]);
"""

_MATRIX_SOURCE = (
    """
    const uint j = thread_position_in_grid.x;
    const uint i = thread_position_in_grid.y;
    const uint n_ref = r_sself_shape[0];
    const uint n_probe = p_sself_shape[0];
    if (i >= n_probe || j >= n_ref) { return; }
"""
    + _PAIR_LOOP
    + _SIM_FROM_OVERLAPS
    + """
    const uint o = i * n_ref + j;
    out_shape[o] = shape;
    out_esp[o] = esp;
    out_combined[o] = combined;
"""
)

_MAX_SOURCE = """
    const uint j = thread_position_in_grid.x;
    const uint blk = thread_position_in_grid.y;
    const uint n_ref = r_sself_shape[0];
    const uint n_probe = p_sself_shape[0];
    if (j >= n_ref) { return; }
    const uint i0 = blk * QBLOCK;
    const uint i1 = metal::min(i0 + QBLOCK, n_probe);
    float best_shape = -INFINITY, best_esp = -INFINITY, best_comb = -INFINITY;
    for (uint i = i0; i < i1; ++i) {
""" + _PAIR_LOOP + _SIM_FROM_OVERLAPS + """
        best_shape = metal::max(best_shape, shape);
        best_esp = metal::max(best_esp, MAP_ESP ? esp_unit : esp);
        best_comb = metal::max(best_comb, combined);
    }
    const uint o = blk * n_ref + j;
    out_shape[o] = best_shape;
    out_esp[o] = best_esp;
    out_combined[o] = best_comb;
"""

_INPUTS = [
    "p_off", "p_xyz", "p_exp", "p_amp", "p_chg", "p_sself", "p_eself",
    "r_off", "r_xyz", "r_exp", "r_amp", "r_chg", "r_sself", "r_eself",
    "weights",
]
_OUTPUTS = ["out_shape", "out_esp", "out_combined"]
_HEADER = (
    f"constant float PI15 = {np.pi ** 1.5:.10e}f;\n"
    f"#define ESP_TERMS {_esp_terms()}\n"
)

_kernels: dict[str, object] = {}


def _kernel(name: str, source: str):
    if name not in _kernels:
        _kernels[name] = mx.fast.metal_kernel(
            name=name,
            input_names=_INPUTS,
            output_names=_OUTPUTS,
            source=source,
            header=_HEADER,
        )
    return _kernels[name]


def _packed(batch: CheeseBatch, gaussian_alpha: float, vdw_scale: float, default_radius: float):
    """Flatten a padded batch to contiguous per-atom arrays plus CSR offsets.

    Padding is dropped, so a tile padded to its largest molecule costs nothing
    extra. ``cheese_batch`` builds prefix masks; anything else is refused rather
    than silently mis-indexed.
    """

    mask = np.asarray(batch.mask).astype(bool)
    counts = mask.sum(axis=1)
    if not np.array_equal(mask, np.arange(mask.shape[1])[None, :] < counts[:, None]):
        raise ValueError("cheese fused kernels need prefix masks (real atoms first), as built by cheese_batch")
    exponent, amplitude = atom_gaussian_parameters_mlx(
        batch.atomic_numbers, gaussian_alpha=gaussian_alpha, vdw_scale=vdw_scale, default_radius=default_radius
    )
    offsets = np.zeros(len(counts) + 1, dtype=np.uint32)
    np.cumsum(counts, out=offsets[1:])
    flat = mx.array(np.flatnonzero(mask.reshape(-1)).astype(np.uint32))
    coords = mx.take(mx.reshape(mx.array(batch.coords, dtype=mx.float32), (-1, 3)), flat, axis=0)
    return (
        mx.array(offsets),
        mx.reshape(coords, (-1,)),
        mx.take(mx.reshape(exponent.astype(mx.float32), (-1,)), flat),
        mx.take(mx.reshape(amplitude.astype(mx.float32), (-1,)), flat),
        mx.take(mx.reshape(mx.array(batch.charges, dtype=mx.float32), (-1,)), flat),
        gaussian_shape_self_overlap_mlx(
            batch, gaussian_alpha=gaussian_alpha, vdw_scale=vdw_scale, default_radius=default_radius
        ).astype(mx.float32),
        electrostatic_self_overlap_mlx(batch).astype(mx.float32),
    )


def _template(shape_metric: str, electrostatic_metric: str, map_electrostatic_to_unit: bool, **extra):
    shape_metric, electrostatic_metric = shape_metric.lower(), electrostatic_metric.lower()
    if shape_metric not in _SHAPE_METRICS:
        raise ValueError("shape_metric must be 'tanimoto' or 'carbo'")
    if electrostatic_metric not in _ESP_METRICS:
        raise ValueError("electrostatic_metric must be 'carbo' or 'tanimoto'")
    return [
        ("SHAPE_METRIC", _SHAPE_METRICS[shape_metric]),
        ("ESP_METRIC", _ESP_METRICS[electrostatic_metric]),
        ("MAP_ESP", bool(map_electrostatic_to_unit)),
        *extra.items(),
    ]


def _weights(shape_weight: float, electrostatic_weight: float) -> mx.array:
    if float(shape_weight) + float(electrostatic_weight) <= 0:
        raise ValueError("at least one CHEESE similarity weight must be positive")
    return mx.array([float(shape_weight), float(electrostatic_weight)], dtype=mx.float32)


def cheese_similarity_matrix_metal(
    probe: CheeseBatch,
    reference: CheeseBatch | None = None,
    *,
    shape_weight: float = 1.0,
    electrostatic_weight: float = 1.0,
    map_electrostatic_to_unit: bool = True,
    electrostatic_metric: str = "carbo",
    shape_metric: str = "tanimoto",
    gaussian_alpha: float = 2.7,
    vdw_scale: float = 1.0,
    default_radius: float = 1.80,
) -> CheeseSimilarityResult:
    """Fused-kernel equivalent of :func:`mlxmolkit.cheese.cheese_similarity_matrix_mlx`.

    ``electrostatic`` is returned unmapped (signed Carbo or ESP-Sim Tanimoto),
    exactly like the reference; ``combined`` uses the mapped value when
    ``map_electrostatic_to_unit`` is set.
    """

    reference = probe if reference is None else reference
    p = _packed(probe, gaussian_alpha, vdw_scale, default_radius)
    r = _packed(reference, gaussian_alpha, vdw_scale, default_radius)
    n_p, n_r = int(p[5].shape[0]), int(r[5].shape[0])
    outs = _kernel("cheese_fused_matrix", _MATRIX_SOURCE)(
        inputs=[*p, *r, _weights(shape_weight, electrostatic_weight)],
        template=_template(shape_metric, electrostatic_metric, map_electrostatic_to_unit),
        grid=(n_r, n_p, 1),
        threadgroup=(min(32, n_r), min(8, n_p), 1),
        output_shapes=[(n_p, n_r)] * 3,
        output_dtypes=[mx.float32] * 3,
    )
    return CheeseSimilarityResult(shape=outs[0], electrostatic=outs[1], combined=outs[2])


def cheese_max_similarity_metal(
    probe: CheeseBatch,
    reference: CheeseBatch,
    *,
    query_block: int = 32,
    shape_weight: float = 1.0,
    electrostatic_weight: float = 1.0,
    map_electrostatic_to_unit: bool = True,
    electrostatic_metric: str = "carbo",
    shape_metric: str = "tanimoto",
    gaussian_alpha: float = 2.7,
    vdw_scale: float = 1.0,
    default_radius: float = 1.80,
) -> CheeseSimilarityResult:
    """Per reference molecule, the maximum similarity over every probe.

    Returns arrays of shape ``(n_reference,)``. ``electrostatic`` is the maximum
    of the mapped [0, 1] value when ``map_electrostatic_to_unit`` is set (the
    unit used in ``combined``), else of the raw metric. ``combined`` is the
    maximum over per-overlay combined scores.

    ``query_block`` probes are handled by each thread; the remaining reduction
    over blocks is one small ``mx.max``.
    """

    p = _packed(probe, gaussian_alpha, vdw_scale, default_radius)
    r = _packed(reference, gaussian_alpha, vdw_scale, default_radius)
    n_p, n_r = int(p[5].shape[0]), int(r[5].shape[0])
    q_block = max(1, int(query_block))
    n_blocks = (n_p + q_block - 1) // q_block
    outs = _kernel("cheese_fused_max", _MAX_SOURCE)(
        inputs=[*p, *r, _weights(shape_weight, electrostatic_weight)],
        template=_template(shape_metric, electrostatic_metric, map_electrostatic_to_unit, QBLOCK=q_block),
        grid=(n_r, n_blocks, 1),
        threadgroup=(min(64, n_r), 1, 1),
        output_shapes=[(n_blocks, n_r)] * 3,
        output_dtypes=[mx.float32] * 3,
    )
    return CheeseSimilarityResult(*(mx.max(o, axis=0) for o in outs))
