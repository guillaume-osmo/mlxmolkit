"""
3D ETK (Experimental Torsion Knowledge) energy and gradient on Metal.

Five energy terms for stage 5 of ETKDG:
  1. CSD torsion preferences: 6-term Fourier series
     E = Σ_{k=1}^{6} V_k * (1 + sign_k * cos(k * φ)) / 2
  2. Improper (planarity), RDKit's UFF inversion: E = w * (1 - sin Y)
  3. 1-2 distance constraints: flat-bottom, 0.5 * w * (d - bound)^2
  4. 1-3 distance constraints: flat-bottom, 0.5 * w * (d - bound)^2
  5. long-range distance constraints: flat-bottom, 0.5 * w * (d - bound)^2
  (No linear-centre angle constraints: that term exists only in etk_metal.)

All threads compute both energy AND gradient (parallel scatter).
One thread per (global_atom, 3D_coord).
"""
from __future__ import annotations

import numpy as np
import mlx.core as mx

from mlxmolkit.etk_extract import BatchedETKSystem

# fmt: off
_ETK_ENERGY_GRAD_SOURCE = """
uint tid = thread_position_in_grid.x;
uint n_atoms_total = params_buf[0];
uint n_mols = params_buf[1];

if (tid >= n_atoms_total * 3u) {{ return; }}

uint global_atom = tid / 3u;
uint coord = tid % 3u;

// Find molecule
uint mol_id = 0;
for (uint mm = 0; mm < n_mols; mm++) {{
    if (global_atom < (uint)atom_starts[mm + 1]) {{
        mol_id = mm;
        break;
    }}
}}

float grad_val = 0.0f;
float e_contrib = 0.0f;

// ======== CSD Torsion terms ========
uint tor_start = (uint)torsion_starts[mol_id];
uint tor_end   = (uint)torsion_starts[mol_id + 1];

for (uint ti = tor_start; ti < tor_end; ti++) {{
    uint i0 = (uint)torsion_idx[ti * 4 + 0];
    uint i1 = (uint)torsion_idx[ti * 4 + 1];
    uint i2 = (uint)torsion_idx[ti * 4 + 2];
    uint i3 = (uint)torsion_idx[ti * 4 + 3];

    bool involved = (global_atom == i0 || global_atom == i1 ||
                     global_atom == i2 || global_atom == i3);
    if (!involved) continue;

    // Vectors along the torsion
    float b1x = pos[i1*3+0] - pos[i0*3+0];
    float b1y = pos[i1*3+1] - pos[i0*3+1];
    float b1z = pos[i1*3+2] - pos[i0*3+2];

    float b2x = pos[i2*3+0] - pos[i1*3+0];
    float b2y = pos[i2*3+1] - pos[i1*3+1];
    float b2z = pos[i2*3+2] - pos[i1*3+2];

    float b3x = pos[i3*3+0] - pos[i2*3+0];
    float b3y = pos[i3*3+1] - pos[i2*3+1];
    float b3z = pos[i3*3+2] - pos[i2*3+2];

    // n1 = b1 × b2, n2 = b2 × b3
    float n1x = b1y*b2z - b1z*b2y;
    float n1y = b1z*b2x - b1x*b2z;
    float n1z = b1x*b2y - b1y*b2x;

    float n2x = b2y*b3z - b2z*b3y;
    float n2y = b2z*b3x - b2x*b3z;
    float n2z = b2x*b3y - b2y*b3x;

    float n1_len = sqrt(n1x*n1x + n1y*n1y + n1z*n1z + 1e-12f);
    float n2_len = sqrt(n2x*n2x + n2y*n2y + n2z*n2z + 1e-12f);
    float b2_len = sqrt(b2x*b2x + b2y*b2y + b2z*b2z + 1e-12f);

    float cos_phi = (n1x*n2x + n1y*n2y + n1z*n2z) / (n1_len * n2_len);
    cos_phi = clamp(cos_phi, -1.0f, 1.0f);

    // m1 = n1 × b2_hat
    float b2hx = b2x / b2_len, b2hy = b2y / b2_len, b2hz = b2z / b2_len;
    float m1x = n1y*b2hz - n1z*b2hy;
    float m1y = n1z*b2hx - n1x*b2hz;
    float m1z = n1x*b2hy - n1y*b2hx;

    float sin_phi = (m1x*n2x + m1y*n2y + m1z*n2z) / (n1_len * n2_len);
    float phi = atan2(sin_phi, cos_phi);

    // 6-term Fourier: E = Σ V_k * (1 + sign_k * cos(k*φ)) / 2
    float E_tor = 0.0f;
    float dE_dphi = 0.0f;
    for (uint k = 0; k < 6u; k++) {{
        float Vk = torsion_V[ti * 6 + k];
        if (Vk == 0.0f) continue;
        float sk = (float)torsion_signs[ti * 6 + k];
        float kf = (float)(k + 1);
        E_tor += Vk * (1.0f + sk * cos(kf * phi)) * 0.5f;
        dE_dphi += Vk * (-sk * kf * sin(kf * phi)) * 0.5f;
    }}

    if (coord == 0u && global_atom == i0) e_contrib += E_tor;

    // Gradient: dE/dpos = dE/dphi * dphi/dpos
    // Using the standard torsion gradient formulas
    float n1_sq = n1x*n1x + n1y*n1y + n1z*n1z + 1e-12f;
    float n2_sq = n2x*n2x + n2y*n2y + n2z*n2z + 1e-12f;

    // dphi/dp0 = -b2_len / n1²  * n1
    // dphi/dp3 =  b2_len / n2²  * n2
    // dphi/dp1 = (b1·b2/(b2²)-1) * dphi/dp0 - (b3·b2/b2²) * dphi/dp3
    // dphi/dp2 = (b3·b2/(b2²)-1) * dphi/dp3 - (b1·b2/b2²) * dphi/dp0

    float b2_sq = b2x*b2x + b2y*b2y + b2z*b2z + 1e-12f;
    float b1_dot_b2 = b1x*b2x + b1y*b2y + b1z*b2z;
    float b3_dot_b2 = b3x*b2x + b3y*b2y + b3z*b2z;

    float f0 = -b2_len / n1_sq;
    float f3 =  b2_len / n2_sq;
    float f1a = b1_dot_b2 / b2_sq - 1.0f;
    float f1b = -b3_dot_b2 / b2_sq;
    float f2a = b3_dot_b2 / b2_sq - 1.0f;
    float f2b = -b1_dot_b2 / b2_sq;

    float dp0, dp1, dp2, dp3;
    if (coord == 0u) {{
        dp0 = f0 * n1x; dp3 = f3 * n2x;
    }} else if (coord == 1u) {{
        dp0 = f0 * n1y; dp3 = f3 * n2y;
    }} else {{
        dp0 = f0 * n1z; dp3 = f3 * n2z;
    }}
    dp1 = f1a * dp0 + f1b * dp3;
    dp2 = f2a * dp3 + f2b * dp0;

    if (global_atom == i0) grad_val += dE_dphi * dp0;
    else if (global_atom == i1) grad_val += dE_dphi * dp1;
    else if (global_atom == i2) grad_val += dE_dphi * dp2;
    else grad_val += dE_dphi * dp3;
}}

// ======== Improper: RDKit's UFF inversion, E = w * (1 - sin Y) ========
// (I, J = sp2 centre, K, L); Y is the angle of J->L to the plane (I, J, K).
uint imp_start = (uint)improper_starts[mol_id];
uint imp_end   = (uint)improper_starts[mol_id + 1];

for (uint ii = imp_start; ii < imp_end; ii++) {{
    uint iI = (uint)improper_i[ii * 4 + 0];
    uint iJ = (uint)improper_i[ii * 4 + 1];
    uint iK = (uint)improper_i[ii * 4 + 2];
    uint iL = (uint)improper_i[ii * 4 + 3];

    bool involved = (global_atom == iI || global_atom == iJ ||
                     global_atom == iK || global_atom == iL);
    if (!involved) continue;

    float u[3], v[3], w3[3];
    for (uint d = 0; d < 3u; d++) {{
        u[d] = pos[iI*3+d] - pos[iJ*3+d];
        v[d] = pos[iK*3+d] - pos[iJ*3+d];
        w3[d] = pos[iL*3+d] - pos[iJ*3+d];
    }}
    float cx = u[1]*v[2]-u[2]*v[1], cy = u[2]*v[0]-u[0]*v[2], cz = u[0]*v[1]-u[1]*v[0];
    float lc = sqrt(cx*cx + cy*cy + cz*cz);
    float lw = sqrt(w3[0]*w3[0] + w3[1]*w3[1] + w3[2]*w3[2]);
    if (lc < 1e-8f || lw < 1e-8f) continue;
    float c = clamp((cx*w3[0] + cy*w3[1] + cz*w3[2]) / (lc * lw), -1.0f, 1.0f);
    float sy = sqrt(max(1.0f - c*c, 0.0f));
    float wt = improper_w[ii];
    if (coord == 0u && global_atom == iJ) e_contrib += wt * c*c / (1.0f + sy);
    if (sy < 1e-8f) continue;
    float dE = wt * c / sy;
    float nh[3] = {cx/lc, cy/lc, cz/lc};
    float wh[3] = {w3[0]/lw, w3[1]/lw, w3[2]/lw};
    float gw[3], gc[3];
    for (uint d = 0; d < 3u; d++) {{ gw[d] = (nh[d]-c*wh[d])/lw; gc[d] = (wh[d]-c*nh[d])/lc; }}
    float gu[3] = {v[1]*gc[2]-v[2]*gc[1], v[2]*gc[0]-v[0]*gc[2], v[0]*gc[1]-v[1]*gc[0]};
    float gv[3] = {gc[1]*u[2]-gc[2]*u[1], gc[2]*u[0]-gc[0]*u[2], gc[0]*u[1]-gc[1]*u[0]};
    if (global_atom == iI) grad_val += dE * gu[coord];
    else if (global_atom == iK) grad_val += dE * gv[coord];
    else if (global_atom == iL) grad_val += dE * gw[coord];
    else grad_val -= dE * (gu[coord] + gv[coord] + gw[coord]);
}}

// ======== 1-2 Distance constraints ========
uint d12_start = (uint)dist12_starts[mol_id];
uint d12_end   = (uint)dist12_starts[mol_id + 1];

for (uint di = d12_start; di < d12_end; di++) {{
    uint a = (uint)d12_i1[di];
    uint b = (uint)d12_i2[di];

    bool is_a = (a == global_atom);
    bool is_b = (b == global_atom);
    if (!is_a && !is_b) continue;

    float dx = pos[a*3+0] - pos[b*3+0];
    float dy = pos[a*3+1] - pos[b*3+1];
    float dz = pos[a*3+2] - pos[b*3+2];
    float d = sqrt(dx*dx + dy*dy + dz*dz + 1e-12f);

    float lb = d12_lb_arr[di];
    float ub = d12_ub_arr[di];
    float w = d12_w[di];

    float my_diff;
    if (coord == 0u) my_diff = dx;
    else if (coord == 1u) my_diff = dy;
    else my_diff = dz;
    if (!is_a) my_diff = -my_diff;

    if (d < lb) {{
        float diff = d - lb;
        if (coord == 0u && is_a) e_contrib += 0.5f * w * diff * diff;
        grad_val += w * diff * my_diff / d;
    }} else if (d > ub) {{
        float diff = d - ub;
        if (coord == 0u && is_a) e_contrib += 0.5f * w * diff * diff;
        grad_val += w * diff * my_diff / d;
    }}
}}

// ======== 1-3 Distance constraints ========
uint d13_start = (uint)dist13_starts[mol_id];
uint d13_end   = (uint)dist13_starts[mol_id + 1];

for (uint di = d13_start; di < d13_end; di++) {{
    uint a = (uint)d13_i1[di];
    uint b = (uint)d13_i2[di];

    bool is_a = (a == global_atom);
    bool is_b = (b == global_atom);
    if (!is_a && !is_b) continue;

    float dx = pos[a*3+0] - pos[b*3+0];
    float dy = pos[a*3+1] - pos[b*3+1];
    float dz = pos[a*3+2] - pos[b*3+2];
    float d = sqrt(dx*dx + dy*dy + dz*dz + 1e-12f);

    float lb = d13_lb_arr[di];
    float ub = d13_ub_arr[di];
    float w = d13_w[di];

    float my_diff;
    if (coord == 0u) my_diff = dx;
    else if (coord == 1u) my_diff = dy;
    else my_diff = dz;
    if (!is_a) my_diff = -my_diff;

    if (d < lb) {{
        float diff = d - lb;
        if (coord == 0u && is_a) e_contrib += 0.5f * w * diff * diff;
        grad_val += w * diff * my_diff / d;
    }} else if (d > ub) {{
        float diff = d - ub;
        if (coord == 0u && is_a) e_contrib += 0.5f * w * diff * diff;
        grad_val += w * diff * my_diff / d;
    }}
}}

// ======== 1-4 Distance constraints ========
uint d14_start = (uint)dist14_starts[mol_id];
uint d14_end   = (uint)dist14_starts[mol_id + 1];

for (uint di = d14_start; di < d14_end; di++) {{
    uint a = (uint)d14_i1[di];
    uint b = (uint)d14_i2[di];

    bool is_a = (a == global_atom);
    bool is_b = (b == global_atom);
    if (!is_a && !is_b) continue;

    float dx = pos[a*3+0] - pos[b*3+0];
    float dy = pos[a*3+1] - pos[b*3+1];
    float dz = pos[a*3+2] - pos[b*3+2];
    float d = sqrt(dx*dx + dy*dy + dz*dz + 1e-12f);

    float lb = d14_lb_arr[di];
    float ub = d14_ub_arr[di];
    float w = d14_w[di];

    float my_diff;
    if (coord == 0u) my_diff = dx;
    else if (coord == 1u) my_diff = dy;
    else my_diff = dz;
    if (!is_a) my_diff = -my_diff;

    if (d < lb) {{
        float diff = d - lb;
        if (coord == 0u && is_a) e_contrib += 0.5f * w * diff * diff;
        grad_val += w * diff * my_diff / d;
    }} else if (d > ub) {{
        float diff = d - ub;
        if (coord == 0u && is_a) e_contrib += 0.5f * w * diff * diff;
        grad_val += w * diff * my_diff / d;
    }}
}}

grad_out[tid] = grad_val;
if (coord == 0u) {{
    energy_parts[global_atom] = e_contrib;
}}
"""
# fmt: on

_etk_kernel = None


def _get_etk_kernel():
    global _etk_kernel
    if _etk_kernel is None:
        _etk_kernel = mx.fast.metal_kernel(
            name="etk_energy_grad",
            input_names=[
                "pos", "params_buf",
                "atom_starts",
                "torsion_idx", "torsion_V", "torsion_signs", "torsion_starts",
                "improper_i", "improper_w", "improper_starts",
                "d12_i1", "d12_i2", "d12_lb_arr", "d12_ub_arr", "d12_w",
                "dist12_starts",
                "d13_i1", "d13_i2", "d13_lb_arr", "d13_ub_arr", "d13_w",
                "dist13_starts",
                "d14_i1", "d14_i2", "d14_lb_arr", "d14_ub_arr", "d14_w",
                "dist14_starts",
            ],
            output_names=["grad_out", "energy_parts"],
            source=_ETK_ENERGY_GRAD_SOURCE,
            ensure_row_contiguous=True,
        )
    return _etk_kernel


def _ensure_nonempty_1d(arr, dtype):
    """Return at least a 1-element array for Metal kernel (avoids empty buffer)."""
    if len(arr) == 0:
        return mx.zeros((1,), dtype=dtype)
    return mx.array(arr.ravel(), dtype=dtype)


def _ensure_nonempty_2d(arr, dtype, cols):
    """Return at least a 1-row array for Metal kernel, flattened."""
    if len(arr) == 0:
        return mx.zeros((cols,), dtype=dtype)
    return mx.array(arr.reshape(-1), dtype=dtype)


def make_etk_energy_grad(
    system: BatchedETKSystem,
):
    """
    Build a batched 3D ETK energy+gradient function backed by a single Metal kernel.

    Returns:
        fn(pos_flat) → (energy_parts, grad)
        where pos_flat is (n_atoms_total * 3,) float32.

        Also returns (atom_starts, n_atoms_total, n_mols) for the caller.
    """
    n_atoms_total = system.n_atoms_total
    n_mols = system.n_mols
    total_coords = n_atoms_total * 3

    params_buf = mx.array([n_atoms_total, n_mols], dtype=mx.uint32)
    atom_starts_mx = mx.array(system.atom_starts, dtype=mx.int32)

    # Torsion terms (flattened 2D arrays)
    torsion_idx_mx = _ensure_nonempty_2d(system.torsion_idx, mx.int32, 4)
    torsion_V_mx = _ensure_nonempty_2d(system.torsion_V, mx.float32, 6)
    torsion_signs_mx = _ensure_nonempty_2d(system.torsion_signs, mx.int32, 6)
    torsion_starts_mx = mx.array(system.torsion_term_starts, dtype=mx.int32)

    # Improper torsion terms
    improper_idx_mx = _ensure_nonempty_2d(system.improper_idx, mx.int32, 4)
    improper_w_mx = _ensure_nonempty_1d(system.improper_weight, mx.float32)
    improper_starts_mx = mx.array(system.improper_term_starts, dtype=mx.int32)

    # 1-4 distance terms
    d12_i1_mx = _ensure_nonempty_1d(system.dist12_idx1, mx.int32)
    d12_i2_mx = _ensure_nonempty_1d(system.dist12_idx2, mx.int32)
    d12_lb_mx = _ensure_nonempty_1d(system.dist12_lb, mx.float32)
    d12_ub_mx = _ensure_nonempty_1d(system.dist12_ub, mx.float32)
    d12_w_mx = _ensure_nonempty_1d(system.dist12_weight, mx.float32)
    dist12_starts_mx = mx.array(system.dist12_term_starts, dtype=mx.int32)

    # 1-3 distance terms
    d13_i1_mx = _ensure_nonempty_1d(system.dist13_idx1, mx.int32)
    d13_i2_mx = _ensure_nonempty_1d(system.dist13_idx2, mx.int32)
    d13_lb_mx = _ensure_nonempty_1d(system.dist13_lb, mx.float32)
    d13_ub_mx = _ensure_nonempty_1d(system.dist13_ub, mx.float32)
    d13_w_mx = _ensure_nonempty_1d(system.dist13_weight, mx.float32)
    dist13_starts_mx = mx.array(system.dist13_term_starts, dtype=mx.int32)

    # 1-4 distance terms
    d14_i1_mx = _ensure_nonempty_1d(system.dist14_idx1, mx.int32)
    d14_i2_mx = _ensure_nonempty_1d(system.dist14_idx2, mx.int32)
    d14_lb_mx = _ensure_nonempty_1d(system.dist14_lb, mx.float32)
    d14_ub_mx = _ensure_nonempty_1d(system.dist14_ub, mx.float32)
    d14_w_mx = _ensure_nonempty_1d(system.dist14_weight, mx.float32)
    dist14_starts_mx = mx.array(system.dist14_term_starts, dtype=mx.int32)

    kernel = _get_etk_kernel()

    def energy_grad_fn(pos_flat: mx.array) -> tuple[mx.array, mx.array]:
        grad_out, energy_parts = kernel(
            inputs=[
                pos_flat, params_buf,
                atom_starts_mx,
                torsion_idx_mx, torsion_V_mx, torsion_signs_mx, torsion_starts_mx,
                improper_idx_mx, improper_w_mx, improper_starts_mx,
                d12_i1_mx, d12_i2_mx, d12_lb_mx, d12_ub_mx, d12_w_mx,
                dist12_starts_mx,
                d13_i1_mx, d13_i2_mx, d13_lb_mx, d13_ub_mx, d13_w_mx,
                dist13_starts_mx,
                d14_i1_mx, d14_i2_mx, d14_lb_mx, d14_ub_mx, d14_w_mx,
                dist14_starts_mx,
            ],
            grid=(total_coords, 1, 1),
            threadgroup=(min(256, total_coords), 1, 1),
            output_shapes=[(total_coords,), (n_atoms_total,)],
            output_dtypes=[mx.float32, mx.float32],
        )
        return energy_parts, grad_out

    return energy_grad_fn, system.atom_starts, n_atoms_total, n_mols
