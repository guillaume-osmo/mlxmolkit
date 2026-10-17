"""Fused density mixing, convergence reduction and iteration bookkeeping.

One threadgroup owns one molecule. Keeping iteration counters on Metal removes
one host synchronization per SCF iteration from the eigensolver path.
"""
import mlx.core as mx

_KERNEL = None
_SOURCE = r'''
uint lane = thread_position_in_threadgroup.x;
uint mol = threadgroup_position_in_grid.x;
int width = config[0];
int iteration = config[1];
int size = width * width;
int base = mol * size;
float worst = 0.0f;
for (int k = lane; k < size; k += 256) {
    float old_p = previous[base + k];
    float new_p = proposed[base + k];
    float diff = abs(new_p - old_p);
    worst = isnan(diff) ? INFINITY : max(worst, diff);
    density[base + k] = previous_converged[mol] ? old_p :
        (iteration < 2 ? 0.5f * new_p + 0.5f * old_p : new_p);
}
threadgroup float partial[256];
partial[lane] = worst;
threadgroup_barrier(mem_flags::mem_threadgroup);
for (uint stride = 128; stride > 0; stride >>= 1) {
    if (lane < stride) partial[lane] = max(partial[lane], partial[lane + stride]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
if (lane == 0) {
    float change = partial[0];
    bool now = change < tolerance[0];
    bool before = previous_converged[mol];
    converged[mol] = before || now;
    iterations[mol] = now && !before ? iteration + 1 : previous_iterations[mol];
    delta[mol] = change;
}
'''


def density_update(previous, proposed, converged, iterations, iteration, tolerance):
    """Return density, cumulative convergence, first convergence iteration, dP."""
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = mx.fast.metal_kernel(
            name='nddo_density_update',
            input_names=['previous', 'proposed', 'previous_converged',
                         'previous_iterations', 'config', 'tolerance'],
            output_names=['density', 'converged', 'iterations', 'delta'],
            source=_SOURCE,
        )
    n, width, _ = previous.shape
    return _KERNEL(
        inputs=[previous, proposed, converged, iterations,
                mx.array([width, iteration], dtype=mx.int32),
                mx.array([tolerance], dtype=mx.float32)],
        output_shapes=[previous.shape, (n,), (n,), (n,)],
        output_dtypes=[mx.float32, mx.bool_, mx.int32, mx.float32],
        grid=(n * 256, 1, 1), threadgroup=(256, 1, 1),
    )
