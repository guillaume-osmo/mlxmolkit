"""
N×k parallel DG conformer generation with shared constraints.

One Metal threadgroup per conformer (C threadgroups total, where C = Σ k_i).
Each threadgroup has TPM=32 threads that parallelize energy computation,
the gradient, line search, and L-BFGS two-loop recursion.

The gradient is spread over all lanes without atomics: phase 1 strides the
distance terms over the lanes and stores each term's prefactor; phase 2 gives
each lane whole atoms and sums their terms through a "terms per atom" CSR
index, in the serial loop's order. The result is deterministic and
bit-identical to the original thread-0 gradient (``parallel_grad=False``).
Float atomics were measured as well (``_grad_mode=GRAD_ATOMIC``): faster, but
the reordered sums send a few conformers to different minima.

Constraints are stored ONCE per molecule and shared across k conformers
via ``conf_to_mol`` indirection.  Only positions differ between conformers
of the same molecule.

Adapted from shivampatel10/mlxmolkit's dg_lbfgs.py threadgroup model.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import mlx.core as mx

from .shared_batch import SharedConstraintBatch

DEFAULT_TPM = 32
DEFAULT_LBFGS_M = 8

# ---------------------------------------------------------------------------
# Metal kernel source (MSL)
# ---------------------------------------------------------------------------

_MSL_HEADER = """
// ---- Constants ----
constant float TOLX = 1.2e-6f;
constant float FUNCTOL = 1e-4f;
constant float MOVETOL = 1e-6f;
constant float MAX_STEP_FACTOR = 100.0f;
constant int MAX_LS_ITERS = 1000;

// ---- Distance violation energy (LOCAL indices + atom_off) ----
inline float dist_violation_e(
    const device float* pos, int i1, int i2,
    float lb2, float ub2, float wt, int dim, int atom_off
) {
    float d2 = 0.0f;
    for (int d = 0; d < dim; d++) {
        float diff = pos[(atom_off + i1) * dim + d] - pos[(atom_off + i2) * dim + d];
        d2 += diff * diff;
    }
    float e = 0.0f;
    if (d2 > ub2) {
        float val = d2 / ub2 - 1.0f;
        e = wt * val * val;
    } else if (d2 < lb2) {
        float val = 2.0f * lb2 / (lb2 + d2) - 1.0f;
        e = wt * val * val;
    }
    return e;
}

// ---- Distance violation gradient (LOCAL indices + atom_off) ----
inline void dist_violation_g(
    const device float* pos, device float* grad,
    int i1, int i2, float lb2, float ub2, float wt, int dim, int atom_off
) {
    float d2 = 0.0f;
    float diff[4];
    for (int d = 0; d < dim; d++) {
        diff[d] = pos[(atom_off + i1) * dim + d] - pos[(atom_off + i2) * dim + d];
        d2 += diff[d] * diff[d];
    }
    float pf = 0.0f;
    if (d2 > ub2) {
        pf = wt * 4.0f * (d2 / ub2 - 1.0f) / ub2;
    } else if (d2 < lb2) {
        float l2d2 = d2 + lb2;
        pf = wt * 8.0f * lb2 * (1.0f - 2.0f * lb2 / l2d2) / (l2d2 * l2d2);
    }
    if (pf != 0.0f) {
        for (int d = 0; d < dim; d++) {
            float g = pf * diff[d];
            grad[(atom_off + i1) * dim + d] += g;
            grad[(atom_off + i2) * dim + d] -= g;
        }
    }
}

// ---- Chiral violation energy (LOCAL indices + atom_off) ----
inline float chiral_violation_e(
    const device float* pos,
    int i1, int i2, int i3, int i4,
    float vol_lower, float vol_upper, float wt, int dim, int atom_off
) {
    float v1[3], v2[3], v3[3];
    for (int d = 0; d < 3; d++) {
        v1[d] = pos[(atom_off+i1)*dim+d] - pos[(atom_off+i4)*dim+d];
        v2[d] = pos[(atom_off+i2)*dim+d] - pos[(atom_off+i4)*dim+d];
        v3[d] = pos[(atom_off+i3)*dim+d] - pos[(atom_off+i4)*dim+d];
    }
    float cx = v2[1]*v3[2] - v2[2]*v3[1];
    float cy = v2[2]*v3[0] - v2[0]*v3[2];
    float cz = v2[0]*v3[1] - v2[1]*v3[0];
    float vol = v1[0]*cx + v1[1]*cy + v1[2]*cz;
    float e = 0.0f;
    if (vol < vol_lower) { float d = vol - vol_lower; e = wt * d * d; }
    else if (vol > vol_upper) { float d = vol - vol_upper; e = wt * d * d; }
    return e;
}

// ---- Chiral violation gradient (LOCAL indices + atom_off) ----
inline void chiral_violation_g(
    const device float* pos, device float* grad,
    int i1, int i2, int i3, int i4,
    float vol_lower, float vol_upper, float wt, int dim, int atom_off
) {
    float v1[3], v2[3], v3[3];
    for (int d = 0; d < 3; d++) {
        v1[d] = pos[(atom_off+i1)*dim+d] - pos[(atom_off+i4)*dim+d];
        v2[d] = pos[(atom_off+i2)*dim+d] - pos[(atom_off+i4)*dim+d];
        v3[d] = pos[(atom_off+i3)*dim+d] - pos[(atom_off+i4)*dim+d];
    }
    float cx = v2[1]*v3[2] - v2[2]*v3[1];
    float cy = v2[2]*v3[0] - v2[0]*v3[2];
    float cz = v2[0]*v3[1] - v2[1]*v3[0];
    float vol = v1[0]*cx + v1[1]*cy + v1[2]*cz;
    float pf = 0.0f;
    if (vol < vol_lower) pf = 2.0f * wt * (vol - vol_lower);
    else if (vol > vol_upper) pf = 2.0f * wt * (vol - vol_upper);
    if (pf == 0.0f) return;
    float g1x = pf * cx, g1y = pf * cy, g1z = pf * cz;
    float g2x = pf * (v3[1]*v1[2] - v3[2]*v1[1]);
    float g2y = pf * (v3[2]*v1[0] - v3[0]*v1[2]);
    float g2z = pf * (v3[0]*v1[1] - v3[1]*v1[0]);
    float g3x = pf * (v2[2]*v1[1] - v2[1]*v1[2]);
    float g3y = pf * (v2[0]*v1[2] - v2[2]*v1[0]);
    float g3z = pf * (v2[1]*v1[0] - v2[0]*v1[1]);
    int o = atom_off;
    grad[(o+i1)*dim+0]+=g1x; grad[(o+i1)*dim+1]+=g1y; grad[(o+i1)*dim+2]+=g1z;
    grad[(o+i2)*dim+0]+=g2x; grad[(o+i2)*dim+1]+=g2y; grad[(o+i2)*dim+2]+=g2z;
    grad[(o+i3)*dim+0]+=g3x; grad[(o+i3)*dim+1]+=g3y; grad[(o+i3)*dim+2]+=g3z;
    grad[(o+i4)*dim+0]-=(g1x+g2x+g3x);
    grad[(o+i4)*dim+1]-=(g1y+g2y+g3y);
    grad[(o+i4)*dim+2]-=(g1z+g2z+g3z);
}

// ---- Fourth dimension energy/gradient ----
inline float fourth_dim_e(const device float* pos, int idx, float wt, int dim, int atom_off) {
    if (dim != 4) return 0.0f;
    float w = pos[(atom_off + idx) * dim + 3];
    return wt * w * w;
}
inline void fourth_dim_g(const device float* pos, device float* grad, int idx, float wt, int dim, int atom_off) {
    if (dim != 4) return;
    float w = pos[(atom_off + idx) * dim + 3];
    grad[(atom_off + idx) * dim + 3] += 2.0f * wt * w;
}

// ---- Threadgroup parallel primitives ----
inline float tg_reduce_sum(threadgroup float* s, uint tid, uint n) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = n / 2; stride > 0; stride >>= 1) {
        if (tid < stride) s[tid] += s[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float result = s[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return result;
}
inline float tg_reduce_max(threadgroup float* s, uint tid, uint n) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = n / 2; stride > 0; stride >>= 1) {
        if (tid < stride) s[tid] = max(s[tid], s[tid + stride]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float result = s[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return result;
}
inline float parallel_dot(const device float* a, const device float* b,
    int n, uint tid, uint tpm, threadgroup float* s) {
    float sum = 0.0f;
    for (int i = (int)tid; i < n; i += (int)tpm) sum += a[i] * b[i];
    s[tid] = sum;
    return tg_reduce_sum(s, tid, tpm);
}
inline void parallel_saxpy(device float* a, float alpha, const device float* b,
    int n, uint tid, uint tpm) {
    for (int i = (int)tid; i < n; i += (int)tpm) a[i] += alpha * b[i];
    threadgroup_barrier(mem_flags::mem_device);
}
inline void parallel_scale(device float* a, float alpha, int n, uint tid, uint tpm) {
    for (int i = (int)tid; i < n; i += (int)tpm) a[i] *= alpha;
    threadgroup_barrier(mem_flags::mem_device);
}
inline void parallel_copy(device float* dst, const device float* src, int n, uint tid, uint tpm) {
    for (int i = (int)tid; i < n; i += (int)tpm) dst[i] = src[i];
    threadgroup_barrier(mem_flags::mem_device);
}
inline void parallel_set(device float* a, float val, int n, uint tid, uint tpm) {
    for (int i = (int)tid; i < n; i += (int)tpm) a[i] = val;
    threadgroup_barrier(mem_flags::mem_device);
}
inline void parallel_neg_copy(device float* dst, const device float* src, int n, uint tid, uint tpm) {
    for (int i = (int)tid; i < n; i += (int)tpm) dst[i] = -src[i];
    threadgroup_barrier(mem_flags::mem_device);
}

// ---- Gradient, all lanes, in two phases ----
// Phase 1 (terms strided over lanes): the scalar prefactor pf of every
// distance term, stored per conformer in my_pf.
// Phase 2 (atoms strided over lanes): each lane sums the terms that touch the
// atoms it owns, read from csr_off/csr_ent. The index lists, per (molecule,
// term type, atom), the terms touching the atom as ``term * 4 + role`` in
// increasing term order (then role): the order in which the serial loop adds
// into that atom. Every contribution is formed with the serial helper's
// expressions, so each gradient component is the same float sum taken in the
// same order -- deterministic, and bit-identical to the thread-0 gradient.
// Slot of (mol, type T, local atom a) = slot0 + T * n_atoms + a.
// Input pointers are template parameters: MLX hands small inputs to the
// kernel in the constant address space and large ones in device.
template <typename PairsT, typename BoundsT>
inline void dg_grad_phase1(
    const device float* pos, device float* my_pf, uint tid, uint tpm,
    int dist_start, int dist_end, int atom_off, int dim,
    PairsT dist_pairs, BoundsT dist_bounds
) {
    for (int t = dist_start + (int)tid; t < dist_end; t += (int)tpm) {
        int i1 = dist_pairs[t*2], i2 = dist_pairs[t*2+1];
        float lb2 = dist_bounds[t*3], ub2 = dist_bounds[t*3+1], wt = dist_bounds[t*3+2];
        float d2 = 0.0f;
        for (int d = 0; d < dim; d++) {
            float diff = pos[(atom_off + i1) * dim + d] - pos[(atom_off + i2) * dim + d];
            d2 += diff * diff;
        }
        float pf = 0.0f;
        if (d2 > ub2) {
            pf = wt * 4.0f * (d2 / ub2 - 1.0f) / ub2;
        } else if (d2 < lb2) {
            float l2d2 = d2 + lb2;
            pf = wt * 8.0f * lb2 * (1.0f - 2.0f * lb2 / l2d2) / (l2d2 * l2d2);
        }
        my_pf[t - dist_start] = pf;
    }
}

template <typename OffT, typename EntT, typename PartT, typename QuadT, typename CBoundT, typename FourT>
inline void dg_grad_phase2(
    const device float* pos, device float* grad, const device float* my_pf, uint tid, uint tpm,
    int n_atoms, int atom_off, int dim, int slot0, int dist_start,
    OffT csr_off, EntT csr_ent, PartT csr_partner,
    QuadT chiral_quads, CBoundT chiral_bounds, float chiral_weight,
    FourT fourth_idx_arr, float fourth_dim_weight
) {
    for (int a = (int)tid; a < n_atoms; a += (int)tpm) {
        float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float own[4];
        for (int d = 0; d < dim; d++) own[d] = pos[(atom_off + a) * dim + d];
        int s = slot0 + a;
        for (int e = csr_off[s]; e < csr_off[s + 1]; e++) {
            int en = csr_ent[e]; int other = csr_partner[e];
            int t = en >> 2; int role = en & 3;
            float pf = my_pf[t - dist_start];
            if (pf == 0.0f) continue;
            // diff = pos[i1] - pos[i2], formed exactly as the serial helper does.
            for (int d = 0; d < dim; d++) {
                float po = pos[(atom_off + other) * dim + d];
                float diff = (role == 0) ? (own[d] - po) : (po - own[d]);
                float g = pf * diff;
                if (role == 0) acc[d] += g; else acc[d] -= g;
            }
        }
        s = slot0 + n_atoms + a;
        for (int e = csr_off[s]; e < csr_off[s + 1]; e++) {
            int en = csr_ent[e]; int t = en >> 2; int role = en & 3;
            int i1 = chiral_quads[t*4], i2 = chiral_quads[t*4+1];
            int i3 = chiral_quads[t*4+2], i4 = chiral_quads[t*4+3];
            float vol_lower = chiral_bounds[t*2], vol_upper = chiral_bounds[t*2+1];
            int o = atom_off;
            float v1[3], v2[3], v3[3];
            for (int d = 0; d < 3; d++) {
                v1[d] = pos[(o+i1)*dim+d] - pos[(o+i4)*dim+d];
                v2[d] = pos[(o+i2)*dim+d] - pos[(o+i4)*dim+d];
                v3[d] = pos[(o+i3)*dim+d] - pos[(o+i4)*dim+d];
            }
            float cx = v2[1]*v3[2] - v2[2]*v3[1];
            float cy = v2[2]*v3[0] - v2[0]*v3[2];
            float cz = v2[0]*v3[1] - v2[1]*v3[0];
            float vol = v1[0]*cx + v1[1]*cy + v1[2]*cz;
            float pf = 0.0f;
            if (vol < vol_lower) pf = 2.0f * chiral_weight * (vol - vol_lower);
            else if (vol > vol_upper) pf = 2.0f * chiral_weight * (vol - vol_upper);
            if (pf == 0.0f) continue;
            float g1x = pf * cx, g1y = pf * cy, g1z = pf * cz;
            float g2x = pf * (v3[1]*v1[2] - v3[2]*v1[1]);
            float g2y = pf * (v3[2]*v1[0] - v3[0]*v1[2]);
            float g2z = pf * (v3[0]*v1[1] - v3[1]*v1[0]);
            float g3x = pf * (v2[2]*v1[1] - v2[1]*v1[2]);
            float g3y = pf * (v2[0]*v1[2] - v2[2]*v1[0]);
            float g3z = pf * (v2[1]*v1[0] - v2[0]*v1[1]);
            if (role == 0) { acc[0] += g1x; acc[1] += g1y; acc[2] += g1z; }
            else if (role == 1) { acc[0] += g2x; acc[1] += g2y; acc[2] += g2z; }
            else if (role == 2) { acc[0] += g3x; acc[1] += g3y; acc[2] += g3z; }
            else {
                acc[0] -= (g1x + g2x + g3x);
                acc[1] -= (g1y + g2y + g3y);
                acc[2] -= (g1z + g2z + g3z);
            }
        }
        if (dim == 4) {
            s = slot0 + 2 * n_atoms + a;
            for (int e = csr_off[s]; e < csr_off[s + 1]; e++) {
                int idx = fourth_idx_arr[csr_ent[e] >> 2];
                float w = pos[(atom_off + idx) * dim + 3];
                acc[3] += 2.0f * fourth_dim_weight * w;
            }
        }
        for (int d = 0; d < dim; d++) grad[(atom_off + a) * dim + d] = acc[d];
    }
}

#if GRAD_MODE == 2
// ---- Gradient, all lanes, terms strided over lanes with float atomics ----
// Measurement-only variant: the order of the atomic adds varies run to run.
#define GADD(p, v) atomic_fetch_add_explicit((device atomic_float*)(p), (v), memory_order_relaxed)
inline void dist_g_atomic(const device float* pos, device float* grad,
    int i1, int i2, float lb2, float ub2, float wt, int dim, int atom_off) {
    float d2 = 0.0f; float diff[4];
    for (int d = 0; d < dim; d++) { diff[d] = pos[(atom_off+i1)*dim+d] - pos[(atom_off+i2)*dim+d]; d2 += diff[d]*diff[d]; }
    float pf = 0.0f;
    if (d2 > ub2) pf = wt * 4.0f * (d2 / ub2 - 1.0f) / ub2;
    else if (d2 < lb2) { float l2d2 = d2 + lb2; pf = wt * 8.0f * lb2 * (1.0f - 2.0f * lb2 / l2d2) / (l2d2 * l2d2); }
    if (pf != 0.0f) for (int d = 0; d < dim; d++) { float g = pf*diff[d];
        GADD(&grad[(atom_off+i1)*dim+d], g); GADD(&grad[(atom_off+i2)*dim+d], -g); }
}
inline void chiral_g_atomic(const device float* pos, device float* grad,
    int i1, int i2, int i3, int i4, float vl, float vu, float wt, int dim, int o) {
    float v1[3], v2[3], v3[3];
    for (int d = 0; d < 3; d++) { v1[d]=pos[(o+i1)*dim+d]-pos[(o+i4)*dim+d]; v2[d]=pos[(o+i2)*dim+d]-pos[(o+i4)*dim+d]; v3[d]=pos[(o+i3)*dim+d]-pos[(o+i4)*dim+d]; }
    float cx=v2[1]*v3[2]-v2[2]*v3[1], cy=v2[2]*v3[0]-v2[0]*v3[2], cz=v2[0]*v3[1]-v2[1]*v3[0];
    float vol=v1[0]*cx+v1[1]*cy+v1[2]*cz; float pf=0.0f;
    if (vol<vl) pf=2.0f*wt*(vol-vl); else if (vol>vu) pf=2.0f*wt*(vol-vu);
    if (pf==0.0f) return;
    float g1[3]={pf*cx,pf*cy,pf*cz};
    float g2[3]={pf*(v3[1]*v1[2]-v3[2]*v1[1]),pf*(v3[2]*v1[0]-v3[0]*v1[2]),pf*(v3[0]*v1[1]-v3[1]*v1[0])};
    float g3[3]={pf*(v2[2]*v1[1]-v2[1]*v1[2]),pf*(v2[0]*v1[2]-v2[2]*v1[0]),pf*(v2[1]*v1[0]-v2[0]*v1[1])};
    for (int d=0; d<3; d++) { GADD(&grad[(o+i1)*dim+d], g1[d]); GADD(&grad[(o+i2)*dim+d], g2[d]);
        GADD(&grad[(o+i3)*dim+d], g3[d]); GADD(&grad[(o+i4)*dim+d], -(g1[d]+g2[d]+g3[d])); }
}
#endif
"""

# Main kernel body — one threadgroup per CONFORMER, TPM threads per threadgroup
# conf_to_mol indirection for shared constraints
_MSL_DG_BODY = r"""
    uint tid = thread_position_in_threadgroup.x;   // 0..TPM-1
    uint conf_idx = threadgroup_position_in_grid.x; // which conformer
    const uint tpm = TPM;
    const int lbfgs_m = LBFGS_M;

    threadgroup float shared[TPM];

    // Config
    int n_confs_cfg = (int)config[0];
    int max_iters = (int)config[1];
    float grad_tol = config[2];
    float chiral_weight = config[3];
    float fourth_dim_weight = config[4];
    const int dim = DIM;  // compiled per dimension (config[5] is the same value)
    int total_pos_size = (int)config[6];

    if ((int)conf_idx >= n_confs_cfg) return;

    // ---- Shared constraint indirection ----
    int mol_idx = conf_to_mol[conf_idx];
    int atom_off = conf_atom_starts[conf_idx]; // this conformer's atom offset
    int n_atoms = mol_n_atoms[mol_idx];
    int n_vars = n_atoms * dim;

    // Constraint boundaries (per molecule — SHARED across conformers)
    int dist_start = dist_term_starts[mol_idx];
    int dist_end = dist_term_starts[mol_idx + 1];
    int chiral_start_t = chiral_term_starts[mol_idx];
    int chiral_end_t = chiral_term_starts[mol_idx + 1];
    int fourth_start_t = fourth_term_starts_arr[mol_idx];
    int fourth_end_t = fourth_term_starts_arr[mol_idx + 1];

    // L-BFGS history offset for this conformer
    int lbfgs_start = lbfgs_history_starts[conf_idx];

    // Copy initial positions to output (parallel)
    parallel_copy(&out_pos[atom_off * dim], &pos[atom_off * dim], n_vars, tid, tpm);

    // Working pointers (each conformer has its own slice)
    device float* my_pos = &out_pos[atom_off * dim];
    device float* my_grad = &work_grad[atom_off * dim];
    device float* my_dir = &work_dir[atom_off * dim];
    device float* my_old_pos = &work_scratch[atom_off * dim];
    device float* my_old_grad = &work_scratch[total_pos_size + atom_off * dim];
    device float* my_q = &work_scratch[2 * total_pos_size + atom_off * dim];

    device float* my_S = &work_lbfgs[lbfgs_start];
    device float* my_Y = &work_lbfgs[lbfgs_start + lbfgs_m * n_vars];
    device float* my_rho = &work_rho[conf_idx * lbfgs_m];

    // ---- Gradient of the current positions into my_grad (all lanes return) ----
#if GRAD_MODE == 1
    device float* my_pf = &work_pf[conf_term_base[conf_idx]];
    #define DG_GRADIENT() \
        dg_grad_phase1(out_pos, my_pf, tid, tpm, dist_start, dist_end, atom_off, dim, \
            dist_pairs, dist_bounds); \
        threadgroup_barrier(mem_flags::mem_device); \
        dg_grad_phase2(out_pos, work_grad, my_pf, tid, tpm, n_atoms, atom_off, dim, \
            csr_slot_base[mol_idx], dist_start, csr_off, csr_ent, csr_partner, \
            chiral_quads, chiral_bounds, chiral_weight, fourth_idx_arr, fourth_dim_weight); \
        threadgroup_barrier(mem_flags::mem_device);
#elif GRAD_MODE == 2
    #define DG_GRADIENT() \
        parallel_set(my_grad, 0.0f, n_vars, tid, tpm); \
        for (int t = dist_start + (int)tid; t < dist_end; t += (int)tpm) \
            dist_g_atomic(out_pos, work_grad, dist_pairs[t*2], dist_pairs[t*2+1], \
                dist_bounds[t*3], dist_bounds[t*3+1], dist_bounds[t*3+2], dim, atom_off); \
        for (int t = chiral_start_t + (int)tid; t < chiral_end_t; t += (int)tpm) \
            chiral_g_atomic(out_pos, work_grad, chiral_quads[t*4], chiral_quads[t*4+1], \
                chiral_quads[t*4+2], chiral_quads[t*4+3], chiral_bounds[t*2], chiral_bounds[t*2+1], \
                chiral_weight, dim, atom_off); \
        if (dim == 4) for (int t = fourth_start_t + (int)tid; t < fourth_end_t; t += (int)tpm) { \
            int a4 = fourth_idx_arr[t]; \
            GADD(&work_grad[(atom_off+a4)*dim+3], 2.0f*fourth_dim_weight*out_pos[(atom_off+a4)*dim+3]); } \
        threadgroup_barrier(mem_flags::mem_device);
#else
    // Reference: thread 0 adds every term serially.
    #define DG_GRADIENT() \
        parallel_set(my_grad, 0.0f, n_vars, tid, tpm); \
        if (tid == 0) { \
            for (int t = dist_start; t < dist_end; t++) \
                dist_violation_g(out_pos, work_grad, dist_pairs[t*2], dist_pairs[t*2+1], \
                    dist_bounds[t*3], dist_bounds[t*3+1], dist_bounds[t*3+2], dim, atom_off); \
            for (int t = chiral_start_t; t < chiral_end_t; t++) \
                chiral_violation_g(out_pos, work_grad, \
                    chiral_quads[t*4], chiral_quads[t*4+1], chiral_quads[t*4+2], chiral_quads[t*4+3], \
                    chiral_bounds[t*2], chiral_bounds[t*2+1], chiral_weight, dim, atom_off); \
            for (int t = fourth_start_t; t < fourth_end_t; t++) \
                fourth_dim_g(out_pos, work_grad, fourth_idx_arr[t], fourth_dim_weight, dim, atom_off); \
        } \
        threadgroup_barrier(mem_flags::mem_device);
#endif

    // ---- Initial energy (parallel) + gradient ----
    float local_energy = 0.0f;
    for (int t = dist_start + (int)tid; t < dist_end; t += (int)tpm)
        local_energy += dist_violation_e(out_pos, dist_pairs[t*2], dist_pairs[t*2+1],
            dist_bounds[t*3], dist_bounds[t*3+1], dist_bounds[t*3+2], dim, atom_off);
    for (int t = chiral_start_t + (int)tid; t < chiral_end_t; t += (int)tpm)
        local_energy += chiral_violation_e(out_pos,
            chiral_quads[t*4], chiral_quads[t*4+1], chiral_quads[t*4+2], chiral_quads[t*4+3],
            chiral_bounds[t*2], chiral_bounds[t*2+1], chiral_weight, dim, atom_off);
    for (int t = fourth_start_t + (int)tid; t < fourth_end_t; t += (int)tpm)
        local_energy += fourth_dim_e(out_pos, fourth_idx_arr[t], fourth_dim_weight, dim, atom_off);
    shared[tid] = local_energy;
    float energy = tg_reduce_sum(shared, tid, tpm);

    DG_GRADIENT();

    // ---- nvMolKit gradient scaling: 0.1x, halve while max > 10 ----
    parallel_scale(my_grad, 0.1f, n_vars, tid, tpm);
    for (int sc = 0; sc < 20; sc++) {
        float lmx = 0.0f;
        for (int i = (int)tid; i < n_vars; i += (int)tpm) {
            float a = abs(my_grad[i]); if (a > lmx) lmx = a;
        }
        shared[tid] = lmx;
        if (tg_reduce_max(shared, tid, tpm) <= 10.0f) break;
        parallel_scale(my_grad, 0.5f, n_vars, tid, tpm);
    }

    parallel_neg_copy(my_dir, my_grad, n_vars, tid, tpm);

    float local_sum_sq = 0.0f;
    for (int i = (int)tid; i < n_vars; i += (int)tpm) local_sum_sq += my_pos[i] * my_pos[i];
    shared[tid] = local_sum_sq;
    float sum_sq = tg_reduce_sum(shared, tid, tpm);
    float max_step = MAX_STEP_FACTOR * max(sqrt(sum_sq), (float)n_vars);

    int status = 1;
    int hist_count = 0;
    int hist_idx = 0;

    for (int iter = 0; iter < max_iters && status == 1; iter++) {
        parallel_copy(my_old_pos, my_pos, n_vars, tid, tpm);
        float old_energy = energy;

        float local_dir_sq = 0.0f;
        for (int i = (int)tid; i < n_vars; i += (int)tpm) local_dir_sq += my_dir[i] * my_dir[i];
        shared[tid] = local_dir_sq;
        float dir_norm = sqrt(tg_reduce_sum(shared, tid, tpm));
        if (dir_norm > max_step) parallel_scale(my_dir, max_step / dir_norm, n_vars, tid, tpm);

        float slope = parallel_dot(my_dir, my_grad, n_vars, tid, tpm, shared);

        float local_test_max = 0.0f;
        for (int i = (int)tid; i < n_vars; i += (int)tpm) {
            float ad = abs(my_dir[i]);
            float ap = max(abs(my_pos[i]), 1.0f);
            float t = ad / ap;
            if (t > local_test_max) local_test_max = t;
        }
        shared[tid] = local_test_max;
        float lambda_min = MOVETOL / max(tg_reduce_max(shared, tid, tpm), 1e-30f);

        float lam = 1.0f, prev_lam = 1.0f, prev_e = old_energy;
        bool ls_done = false;

        for (int ls_iter = 0; ls_iter < MAX_LS_ITERS && !ls_done; ls_iter++) {
            if (lam < lambda_min) { parallel_copy(my_pos, my_old_pos, n_vars, tid, tpm); ls_done = true; break; }

            for (int i = (int)tid; i < n_vars; i += (int)tpm)
                my_pos[i] = my_old_pos[i] + lam * my_dir[i];
            threadgroup_barrier(mem_flags::mem_device);

            float local_trial_e = 0.0f;
            for (int t = dist_start + (int)tid; t < dist_end; t += (int)tpm)
                local_trial_e += dist_violation_e(out_pos, dist_pairs[t*2], dist_pairs[t*2+1],
                    dist_bounds[t*3], dist_bounds[t*3+1], dist_bounds[t*3+2], dim, atom_off);
            for (int t = chiral_start_t + (int)tid; t < chiral_end_t; t += (int)tpm)
                local_trial_e += chiral_violation_e(out_pos,
                    chiral_quads[t*4], chiral_quads[t*4+1], chiral_quads[t*4+2], chiral_quads[t*4+3],
                    chiral_bounds[t*2], chiral_bounds[t*2+1], chiral_weight, dim, atom_off);
            for (int t = fourth_start_t + (int)tid; t < fourth_end_t; t += (int)tpm)
                local_trial_e += fourth_dim_e(out_pos, fourth_idx_arr[t], fourth_dim_weight, dim, atom_off);
            shared[tid] = local_trial_e;
            float trial_e = tg_reduce_sum(shared, tid, tpm);

            if (trial_e - old_energy <= FUNCTOL * lam * slope) {
                energy = trial_e; ls_done = true;
            } else {
                float tmp_lam;
                if (ls_iter == 0) {
                    tmp_lam = -slope / (2.0f * (trial_e - old_energy - slope));
                } else {
                    float rhs1 = trial_e - old_energy - lam * slope;
                    float rhs2 = prev_e - old_energy - prev_lam * slope;
                    float lam_sq = lam * lam, lam2_sq = prev_lam * prev_lam;
                    float denom_v = lam - prev_lam;
                    if (abs(denom_v) < 1e-30f) { tmp_lam = 0.5f * lam; }
                    else {
                        float a = (rhs1/lam_sq - rhs2/lam2_sq) / denom_v;
                        float b = (-prev_lam*rhs1/lam_sq + lam*rhs2/lam2_sq) / denom_v;
                        if (abs(a) < 1e-30f) tmp_lam = (abs(b)>1e-30f) ? -slope/(2.0f*b) : 0.5f*lam;
                        else {
                            float disc = b*b - 3.0f*a*slope;
                            if (disc < 0.0f) tmp_lam = 0.5f*lam;
                            else if (b <= 0.0f) tmp_lam = (-b+sqrt(disc))/(3.0f*a);
                            else tmp_lam = -slope/(b+sqrt(disc));
                        }
                    }
                }
                tmp_lam = clamp(tmp_lam, 0.1f * lam, 0.5f * lam);
                prev_lam = lam; prev_e = trial_e; lam = tmp_lam;
            }
        }

        if (!ls_done) parallel_copy(my_pos, my_old_pos, n_vars, tid, tpm);

        for (int i = (int)tid; i < n_vars; i += (int)tpm)
            my_old_pos[i] = my_pos[i] - my_old_pos[i]; // s_k
        threadgroup_barrier(mem_flags::mem_device);

        float local_tolx = 0.0f;
        for (int i = (int)tid; i < n_vars; i += (int)tpm) {
            float t = abs(my_old_pos[i]) / max(abs(my_pos[i]), 1.0f);
            if (t > local_tolx) local_tolx = t;
        }
        shared[tid] = local_tolx;
        if (tg_reduce_max(shared, tid, tpm) < TOLX) { status = 0; break; }

        parallel_copy(my_old_grad, my_grad, n_vars, tid, tpm);

        float local_new_e = 0.0f;
        for (int t = dist_start + (int)tid; t < dist_end; t += (int)tpm)
            local_new_e += dist_violation_e(out_pos, dist_pairs[t*2], dist_pairs[t*2+1],
                dist_bounds[t*3], dist_bounds[t*3+1], dist_bounds[t*3+2], dim, atom_off);
        for (int t = chiral_start_t + (int)tid; t < chiral_end_t; t += (int)tpm)
            local_new_e += chiral_violation_e(out_pos,
                chiral_quads[t*4], chiral_quads[t*4+1], chiral_quads[t*4+2], chiral_quads[t*4+3],
                chiral_bounds[t*2], chiral_bounds[t*2+1], chiral_weight, dim, atom_off);
        for (int t = fourth_start_t + (int)tid; t < fourth_end_t; t += (int)tpm)
            local_new_e += fourth_dim_e(out_pos, fourth_idx_arr[t], fourth_dim_weight, dim, atom_off);
        shared[tid] = local_new_e;
        energy = tg_reduce_sum(shared, tid, tpm);

        DG_GRADIENT();

        float local_grad_test = 0.0f;
        for (int i = (int)tid; i < n_vars; i += (int)tpm) {
            float t = abs(my_grad[i]) * max(abs(my_pos[i]), 1.0f);
            if (t > local_grad_test) local_grad_test = t;
        }
        shared[tid] = local_grad_test;
        if (tg_reduce_max(shared, tid, tpm) / max(energy, 1.0f) < grad_tol) { status = 0; break; }

        // L-BFGS update
        for (int i = (int)tid; i < n_vars; i += (int)tpm)
            my_q[i] = my_grad[i] - my_old_grad[i]; // y_k
        threadgroup_barrier(mem_flags::mem_device);

        float ys_dot = parallel_dot(my_q, my_old_pos, n_vars, tid, tpm, shared);

        if (ys_dot > 1e-10f) {
            int slot = hist_idx % lbfgs_m;
            parallel_copy(&my_S[slot * n_vars], my_old_pos, n_vars, tid, tpm);
            parallel_copy(&my_Y[slot * n_vars], my_q, n_vars, tid, tpm);
            if (tid == 0) my_rho[slot] = 1.0f / ys_dot;
            threadgroup_barrier(mem_flags::mem_device);
            hist_idx++;
            if (hist_count < lbfgs_m) hist_count++;
        }

        // Two-loop recursion
        parallel_copy(my_q, my_grad, n_vars, tid, tpm);
        device float* my_alpha = &work_alpha[conf_idx * lbfgs_m];

        for (int j = hist_count - 1; j >= 0; j--) {
            int slot = (hist_idx - 1 - (hist_count - 1 - j)) % lbfgs_m;
            if (slot < 0) slot += lbfgs_m;
            float alpha_j = my_rho[slot] * parallel_dot(&my_S[slot*n_vars], my_q, n_vars, tid, tpm, shared);
            if (tid == 0) my_alpha[j] = alpha_j;
            threadgroup_barrier(mem_flags::mem_device);
            parallel_saxpy(my_q, -alpha_j, &my_Y[slot*n_vars], n_vars, tid, tpm);
        }
        if (hist_count > 0) {
            int newest = (hist_idx - 1) % lbfgs_m;
            if (newest < 0) newest += lbfgs_m;
            float sy = parallel_dot(&my_S[newest*n_vars], &my_Y[newest*n_vars], n_vars, tid, tpm, shared);
            float yy = parallel_dot(&my_Y[newest*n_vars], &my_Y[newest*n_vars], n_vars, tid, tpm, shared);
            parallel_scale(my_q, sy / max(yy, 1e-30f), n_vars, tid, tpm);
        }
        for (int j = 0; j < hist_count; j++) {
            int slot = (hist_idx - 1 - (hist_count - 1 - j)) % lbfgs_m;
            if (slot < 0) slot += lbfgs_m;
            float beta_j = my_rho[slot] * parallel_dot(&my_Y[slot*n_vars], my_q, n_vars, tid, tpm, shared);
            float alpha_j = my_alpha[j];
            parallel_saxpy(my_q, alpha_j - beta_j, &my_S[slot*n_vars], n_vars, tid, tpm);
        }
        parallel_neg_copy(my_dir, my_q, n_vars, tid, tpm);
    }

    if (tid == 0) {
        out_energies[conf_idx] = energy;
        out_statuses[conf_idx] = status;
    }
"""

# Gradient modes compiled into the kernel (GRAD_MODE):
#   0 = thread 0 adds every term serially (reference; 31 lanes idle).
#   1 = every lane gathers the terms of the atoms it owns through a CSR
#       "terms per atom" index (default; deterministic, same order as 0).
#   2 = terms strided over lanes, float atomic adds (measurement only:
#       the order of the adds, hence the low bits, varies run to run).
GRAD_SERIAL, GRAD_GATHER, GRAD_ATOMIC = 0, 1, 2

# Cache: (tpm, lbfgs_m, grad_mode, dim) -> compiled kernel
_dg_kernel_cache: dict[tuple, object] = {}


def build_atom_term_csr(
    mol_n_atoms: np.ndarray,
    term_types: list[tuple[np.ndarray, np.ndarray]],
    local_terms: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Index, per molecule and term type, the terms that touch each atom.

    Parameters
    ----------
    mol_n_atoms : (N,) int
        Atoms per molecule.
    term_types : list of (term_starts, atoms)
        One entry per term type, in the order the serial gradient visits the
        types. ``term_starts`` is the (N+1,) per-molecule range into the
        type's term arrays; ``atoms`` is (n_terms, R) LOCAL atom indices, one
        column per role (column order = order the serial code adds them).

    Returns
    -------
    slot_base : (N+1,) int32
        Slot of (molecule m, type T, local atom a) is
        ``slot_base[m] + T * mol_n_atoms[m] + a``.
    offsets : (n_slots+1,) int32
        Entries of a slot are ``entries[offsets[slot]:offsets[slot+1]]``.
    entries : (n_incidences,) int32
        ``term * 4 + role`` with ``term`` the global term index of its type
        (or, with ``local_terms``, the index within its molecule), ordered by
        term then role: the order in which a serial loop over the terms
        accumulates into that atom.
    partner : (n_incidences,) int32
        For two-atom terms, the LOCAL index of the other atom (saves the
        kernel a dependent load); 0 for other term types.
    """
    n_atoms = np.asarray(mol_n_atoms, dtype=np.int64)
    n_mols = len(n_atoms)
    n_types = len(term_types)
    slot_base = np.zeros(n_mols + 1, dtype=np.int64)
    np.cumsum(n_types * n_atoms, out=slot_base[1:])
    n_slots = int(slot_base[-1])
    keys, ents, partners = [], [], []
    for t_type, (starts, atoms) in enumerate(term_types):
        atoms = np.asarray(atoms, dtype=np.int64)
        n_terms = int(starts[-1]) if len(starts) else 0
        if n_terms == 0:
            continue
        atoms = atoms.reshape(n_terms, -1)
        n_roles = atoms.shape[1]
        mol = np.repeat(np.arange(n_mols), np.diff(np.asarray(starts, dtype=np.int64)))
        base = slot_base[mol] + t_type * n_atoms[mol]
        keys.append((base[:, None] + atoms).ravel())
        term = np.arange(n_terms, dtype=np.int64)
        if local_terms:
            term = term - np.asarray(starts, dtype=np.int64)[mol]
        ents.append((term[:, None] * 4 + np.arange(n_roles, dtype=np.int64)[None, :]).ravel())
        partners.append(atoms[:, ::-1].ravel() if n_roles == 2 else np.zeros(atoms.size, dtype=np.int64))
    offsets = np.zeros(n_slots + 1, dtype=np.int32)
    if not keys:
        empty = np.zeros(1, dtype=np.int32)
        return slot_base.astype(np.int32), offsets, empty, empty
    key = np.concatenate(keys)
    ent = np.concatenate(ents)
    if ent.max() >= 2**31:
        raise ValueError("too many terms for the int32 term*4+role encoding")
    order = np.argsort(key, kind="stable")
    np.cumsum(np.bincount(key, minlength=n_slots), out=offsets[1:])
    return (slot_base.astype(np.int32), offsets, ent[order].astype(np.int32),
            np.concatenate(partners)[order].astype(np.int32))


def _cached_on_batch(batch, name: str, arrays: tuple, build):
    """Memoise ``build()`` on *batch*, keyed on the identity of *arrays*.

    The pipeline runs three DG minimisations on one batch; the index arrays a
    CSR is built from are never modified in place, so array identity is the
    key (the cache holds references, so an id cannot be recycled).
    """
    hit = getattr(batch, name, None)
    if hit is not None and len(hit[0]) == len(arrays) and all(a is b for a, b in zip(hit[0], arrays)):
        return hit[1]
    value = build()
    try:
        setattr(batch, name, (arrays, value))
    except AttributeError:
        pass
    return value


def _dg_grad_csr(batch: SharedConstraintBatch):
    """CSR of DG terms per atom: types (distance, chiral, fourth-dim)."""
    arrays = (batch.mol_n_atoms, batch.dist_term_starts, batch.dist_idx1, batch.dist_idx2,
              batch.chiral_term_starts, batch.chiral_idx1, batch.chiral_idx2, batch.chiral_idx3,
              batch.chiral_idx4, batch.fourth_term_starts, batch.fourth_idx)
    return _cached_on_batch(batch, "_dg_grad_csr_cache", arrays, lambda: _build_dg_grad_csr(batch))


def _build_dg_grad_csr(batch: SharedConstraintBatch):
    return build_atom_term_csr(batch.mol_n_atoms, [
        (batch.dist_term_starts, np.stack([batch.dist_idx1, batch.dist_idx2], axis=1)),
        (batch.chiral_term_starts, np.stack([
            batch.chiral_idx1, batch.chiral_idx2, batch.chiral_idx3, batch.chiral_idx4], axis=1)),
        (batch.fourth_term_starts, np.asarray(batch.fourth_idx).reshape(-1, 1)),
    ])


def _build_dg_kernel(
    tpm: int = DEFAULT_TPM,
    lbfgs_m: int = DEFAULT_LBFGS_M,
    grad_mode: int = GRAD_GATHER,
    dim: int = 4,
):
    """Compile the DG L-BFGS Metal kernel with shared constraints."""
    header = _MSL_HEADER.replace("TPM", str(tpm)).replace("LBFGS_M", str(lbfgs_m))
    header = f"#define GRAD_MODE {int(grad_mode)}\n#define DIM {int(dim)}\n" + header
    source = _MSL_DG_BODY.replace("TPM", str(tpm)).replace("LBFGS_M", str(lbfgs_m))

    return mx.fast.metal_kernel(
        name=f"dg_lbfgs_shared_g{int(grad_mode)}_d{int(dim)}",
        input_names=[
            "pos", "config",
            "conf_to_mol", "conf_atom_starts", "mol_n_atoms",
            "dist_term_starts", "dist_pairs", "dist_bounds",
            "chiral_term_starts", "chiral_quads", "chiral_bounds",
            "fourth_term_starts_arr", "fourth_idx_arr",
            "lbfgs_history_starts",
            "csr_slot_base", "csr_off", "csr_ent", "csr_partner", "conf_term_base",
        ],
        output_names=[
            "out_pos", "out_energies", "out_statuses",
            "work_grad", "work_dir", "work_scratch",
            "work_lbfgs", "work_rho", "work_alpha", "work_pf",
        ],
        header=header,
        source=source,
        ensure_row_contiguous=True,
    )


def _get_dg_kernel(
    tpm: int = DEFAULT_TPM,
    lbfgs_m: int = DEFAULT_LBFGS_M,
    grad_mode: int = GRAD_GATHER,
    dim: int = 4,
):
    key = (tpm, lbfgs_m, int(grad_mode), int(dim))
    if key not in _dg_kernel_cache:
        _dg_kernel_cache[key] = _build_dg_kernel(tpm, lbfgs_m, grad_mode, dim)
    return _dg_kernel_cache[key]


def dg_minimize_shared(
    batch: SharedConstraintBatch,
    positions: np.ndarray,
    *,
    max_iters: int = 200,
    grad_tol: float = 1e-4,
    chiral_weight: float = 1.0,
    fourth_dim_weight: float = 0.1,
    tpm: int = DEFAULT_TPM,
    lbfgs_m: int = DEFAULT_LBFGS_M,
    parallel_grad: bool = True,
    _grad_mode: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run DG L-BFGS on all C conformers in parallel with shared constraints.

    Parameters
    ----------
    batch : SharedConstraintBatch
        N molecules × k conformers with shared constraints.
    positions : np.ndarray, shape (n_atoms_total * dim,)
        Initial positions (random 4D).
    parallel_grad : bool
        True (default): all TPM lanes compute the gradient. Each lane owns
        atoms ``tid, tid + TPM, ...`` and gathers the terms that touch them
        through a per-molecule "terms per atom" CSR index, so no two lanes
        write the same component and no atomics are needed. Every component
        is summed in the same term order as the serial loop, so the result
        is deterministic and agrees with ``parallel_grad=False``.
        False: thread 0 computes the whole gradient serially while the other
        lanes wait (the original kernel, kept as a reference).

    Returns
    -------
    out_positions : np.ndarray
        Optimized positions.
    energies : np.ndarray, shape (C,)
        Final energy per conformer.
    statuses : np.ndarray, shape (C,)
        0 = converged, 1 = max_iters reached.
    """
    grad_mode = _grad_mode if _grad_mode is not None else (GRAD_GATHER if parallel_grad else GRAD_SERIAL)
    C = batch.n_confs_total
    dim = batch.dim
    total_pos_size = int(batch.conf_atom_starts[-1]) * dim

    # Pack config (config[6] = total_pos_size for kernel scratch indexing)
    config = np.array([
        C, max_iters, grad_tol, chiral_weight, fourth_dim_weight, dim,
        total_pos_size,
    ], dtype=np.float32)

    # Pack constraint arrays (LOCAL indices, interleaved)
    n_dist = len(batch.dist_idx1)
    if n_dist > 0:
        dist_pairs = np.stack([batch.dist_idx1, batch.dist_idx2], axis=1).flatten().astype(np.int32)
        dist_bounds = np.stack([batch.dist_lb2, batch.dist_ub2, batch.dist_weight], axis=1).flatten().astype(np.float32)
    else:
        dist_pairs = np.zeros(2, dtype=np.int32)
        dist_bounds = np.zeros(3, dtype=np.float32)

    n_chiral = len(batch.chiral_idx1)
    if n_chiral > 0:
        chiral_quads = np.stack([
            batch.chiral_idx1, batch.chiral_idx2,
            batch.chiral_idx3, batch.chiral_idx4,
        ], axis=1).flatten().astype(np.int32)
        chiral_bounds = np.stack([
            batch.chiral_vol_lower, batch.chiral_vol_upper,
        ], axis=1).flatten().astype(np.float32)
    else:
        chiral_quads = np.zeros(4, dtype=np.int32)
        chiral_bounds = np.zeros(2, dtype=np.float32)

    if grad_mode == GRAD_GATHER:
        csr_slot_base, csr_off, csr_ent, csr_partner = _dg_grad_csr(batch)
    else:
        csr_slot_base = csr_off = csr_ent = csr_partner = np.zeros(1, dtype=np.int32)
    # Per-conformer slice of the distance-term prefactor scratch (phase 1).
    n_dist_c = np.diff(batch.dist_term_starts.astype(np.int64))[batch.conf_to_mol]
    conf_term_base = np.zeros(C + 1, dtype=np.int32)
    np.cumsum(n_dist_c, out=conf_term_base[1:])
    total_pf = int(conf_term_base[-1]) if grad_mode == GRAD_GATHER else 0

    # L-BFGS history starts per conformer
    n_vars_c = batch.mol_n_atoms[batch.conf_to_mol].astype(np.int64) * dim
    lbfgs_starts = np.zeros(C + 1, dtype=np.int32)
    np.cumsum(2 * lbfgs_m * n_vars_c, out=lbfgs_starts[1:])
    total_lbfgs = int(lbfgs_starts[-1])

    # Convert to MLX
    kernel = _get_dg_kernel(tpm, lbfgs_m, grad_mode, dim)
    results = kernel(
        inputs=[
            mx.array(positions),
            mx.array(config),
            mx.array(batch.conf_to_mol),
            mx.array(batch.conf_atom_starts),
            mx.array(batch.mol_n_atoms),
            mx.array(batch.dist_term_starts),
            mx.array(dist_pairs),
            mx.array(dist_bounds),
            mx.array(batch.chiral_term_starts),
            mx.array(chiral_quads),
            mx.array(chiral_bounds),
            mx.array(batch.fourth_term_starts),
            mx.array(batch.fourth_idx),
            mx.array(lbfgs_starts),  # (C+1,) — full array to avoid shape aliasing
            mx.array(csr_slot_base),
            mx.array(csr_off),
            mx.array(csr_ent),
            mx.array(csr_partner),
            mx.array(conf_term_base),
        ],
        grid=(C * tpm, 1, 1),  # total threads = C threadgroups × TPM
        threadgroup=(tpm, 1, 1),
        output_shapes=[
            (total_pos_size,),    # out_pos
            (C,),                 # out_energies
            (C,),                 # out_statuses
            (total_pos_size,),    # work_grad
            (total_pos_size,),    # work_dir
            (3 * total_pos_size,),  # work_scratch (old_pos, old_grad, q)
            (max(1, total_lbfgs),),  # work_lbfgs (S + Y history)
            (max(1, C * lbfgs_m),),  # work_rho
            (max(1, C * lbfgs_m),),  # work_alpha
            (max(1, total_pf),),     # work_pf (distance-term prefactors)
        ],
        output_dtypes=[
            mx.float32, mx.float32, mx.int32,
            mx.float32, mx.float32, mx.float32,
            mx.float32, mx.float32, mx.float32, mx.float32,
        ],
    )
    mx.eval(results[0], results[1], results[2])

    out_pos = np.array(results[0])
    energies = np.array(results[1])
    statuses = np.array(results[2])

    return out_pos, energies, statuses
