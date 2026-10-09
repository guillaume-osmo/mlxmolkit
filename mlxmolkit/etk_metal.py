"""
N×k parallel ETK (3D torsion) minimization with shared constraints.

Same TPM=32 threadgroup architecture as conformer_metal.py (DG stage),
but with ETK energy terms: CSD torsion preferences (6-term Fourier),
improper torsion (planarity), and 1-4 distance constraints.

Constraint indices are LOCAL [0, n_atoms_mol). The kernel pre-adds
``atom_off = conf_atom_starts[conf_idx]`` before calling helpers.
Helpers operate on GLOBAL indices and stay unchanged from shivampatel10.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import mlx.core as mx

from .conformer_metal import (
    GRAD_GATHER,
    GRAD_SERIAL,
    _cached_on_batch,
    build_atom_term_csr,
)
from .shared_batch import SharedConstraintBatch

DEFAULT_TPM = 32
DEFAULT_LBFGS_M = 8

# ---------------------------------------------------------------------------
# ETK MSL helpers (from shivampatel10, unchanged — operate on global indices)
# ---------------------------------------------------------------------------

_ETK_HEADER = """
constant float TOLX = 1.2e-6f;
constant float FUNCTOL = 1e-4f;
constant float MOVETOL = 1e-6f;
constant float MAX_STEP_FACTOR = 100.0f;
constant int MAX_LS_ITERS = 1000;

// ---- Torsion cos(phi) ----
inline float calc_cos_phi(const device float* pos, int i1, int i2, int i3, int i4, int dim) {
    float r1[3], r2[3], r3[3], r4[3];
    for (int d=0;d<3;d++) {
        r1[d] = pos[i1*dim+d] - pos[i2*dim+d];
        r2[d] = pos[i3*dim+d] - pos[i2*dim+d];
        r3[d] = -r2[d];
        r4[d] = pos[i4*dim+d] - pos[i3*dim+d];
    }
    float t1x=r1[1]*r2[2]-r1[2]*r2[1], t1y=r1[2]*r2[0]-r1[0]*r2[2], t1z=r1[0]*r2[1]-r1[1]*r2[0];
    float t2x=r3[1]*r4[2]-r3[2]*r4[1], t2y=r3[2]*r4[0]-r3[0]*r4[2], t2z=r3[0]*r4[1]-r3[1]*r4[0];
    float comb = (t1x*t1x+t1y*t1y+t1z*t1z)*(t2x*t2x+t2y*t2y+t2z*t2z);
    if (comb < 1e-16f) return 0.0f;
    return clamp((t1x*t2x+t1y*t2y+t1z*t2z)*rsqrt(comb), -1.0f, 1.0f);
}

// ---- 6-term Fourier torsion energy (Chebyshev recurrence) ----
inline float torsion_e(float c,
    float V0,float V1,float V2,float V3,float V4,float V5,
    float s0,float s1,float s2,float s3,float s4,float s5
) {
    float c2=c*c, c3=c*c2, c4=c*c3, c5=c*c4, c6=c*c5;
    return V0*(1.0f+s0*c) + V1*(1.0f+s1*(2.0f*c2-1.0f))
         + V2*(1.0f+s2*(4.0f*c3-3.0f*c)) + V3*(1.0f+s3*(8.0f*c4-8.0f*c2+1.0f))
         + V4*(1.0f+s4*(16.0f*c5-20.0f*c3+5.0f*c))
         + V5*(1.0f+s5*(32.0f*c6-48.0f*c4+18.0f*c2-1.0f));
}

// ---- Torsion gradient (full 4-atom, from shivampatel10) ----
inline void torsion_g(const device float* pos, device float* grad,
    int i1,int i2,int i3,int i4,
    float V0,float V1,float V2,float V3,float V4,float V5,
    float s0,float s1,float s2,float s3,float s4,float s5, int dim
) {
    float r1[3],r2[3],r3[3],r4[3];
    for (int d=0;d<3;d++) {
        r1[d]=pos[i1*dim+d]-pos[i2*dim+d]; r2[d]=pos[i3*dim+d]-pos[i2*dim+d];
        r3[d]=-r2[d]; r4[d]=pos[i4*dim+d]-pos[i3*dim+d];
    }
    float t0x=r1[1]*r2[2]-r1[2]*r2[1],t0y=r1[2]*r2[0]-r1[0]*r2[2],t0z=r1[0]*r2[1]-r1[1]*r2[0];
    float t1x=r3[1]*r4[2]-r3[2]*r4[1],t1y=r3[2]*r4[0]-r3[0]*r4[2],t1z=r3[0]*r4[1]-r3[1]*r4[0];
    float d02=t0x*t0x+t0y*t0y+t0z*t0z, d12=t1x*t1x+t1y*t1y+t1z*t1z;
    if (d02<1e-16f||d12<1e-16f) return;
    float inv0=rsqrt(max(d02,1e-16f)),inv1=rsqrt(max(d12,1e-16f));
    float tnx0=t0x*inv0,tny0=t0y*inv0,tnz0=t0z*inv0;
    float tnx1=t1x*inv1,tny1=t1y*inv1,tnz1=t1z*inv1;
    float cp=clamp(tnx0*tnx1+tny0*tny1+tnz0*tnz1,-1.0f,1.0f);
    float sp2=1.0f-cp*cp, sp=sqrt(max(sp2,0.0f));
    float c=cp,c2=c*c,c3=c*c2,c4=c*c3;
    float dE=-s0*V0*sp-2.0f*s1*V1*(2.0f*c*sp)-3.0f*s2*V2*(4.0f*c2*sp-sp)
        -4.0f*s3*V3*(8.0f*c3*sp-4.0f*c*sp)-5.0f*s4*V4*(16.0f*c4*sp-12.0f*c2*sp+sp)
        -6.0f*s5*V5*(32.0f*c4*c*sp-32.0f*c3*sp+6.0f*c*sp);
    float st;
    if (abs(sp)>1e-8f) st=-dE/sp; else st=-dE/max(abs(cp),1e-16f)*sign(cp+1e-30f);
    float dcx0=inv0*(tnx1-cp*tnx0),dcy0=inv0*(tny1-cp*tny0),dcz0=inv0*(tnz1-cp*tnz0);
    float dcx1=inv1*(tnx0-cp*tnx1),dcy1=inv1*(tny0-cp*tny1),dcz1=inv1*(tnz0-cp*tnz1);
    float g1x=st*(dcz0*r2[1]-dcy0*r2[2]),g1y=st*(dcx0*r2[2]-dcz0*r2[0]),g1z=st*(dcy0*r2[0]-dcx0*r2[1]);
    float g4x=st*(dcy1*r3[2]-dcz1*r3[1]),g4y=st*(dcz1*r3[0]-dcx1*r3[2]),g4z=st*(dcx1*r3[1]-dcy1*r3[0]);
    float g2x=st*(dcy0*(r2[2]-r1[2])+dcz0*(r1[1]-r2[1])+dcy1*(-r4[2])+dcz1*r4[1]);
    float g2y=st*(dcx0*(r1[2]-r2[2])+dcz0*(r2[0]-r1[0])+dcx1*r4[2]+dcz1*(-r4[0]));
    float g2z=st*(dcx0*(r2[1]-r1[1])+dcy0*(r1[0]-r2[0])+dcx1*(-r4[1])+dcy1*r4[0]);
    float g3x=st*(dcy0*r1[2]+dcz0*(-r1[1])+dcy1*(r4[2]-r3[2])+dcz1*(r3[1]-r4[1]));
    float g3y=st*(dcx0*(-r1[2])+dcz0*r1[0]+dcx1*(r3[2]-r4[2])+dcz1*(r4[0]-r3[0]));
    float g3z=st*(dcx0*r1[1]+dcy0*(-r1[0])+dcx1*(r4[1]-r3[1])+dcy1*(r3[0]-r4[0]));
    grad[i1*dim+0]+=g1x;grad[i1*dim+1]+=g1y;grad[i1*dim+2]+=g1z;
    grad[i2*dim+0]+=g2x;grad[i2*dim+1]+=g2y;grad[i2*dim+2]+=g2z;
    grad[i3*dim+0]+=g3x;grad[i3*dim+1]+=g3y;grad[i3*dim+2]+=g3z;
    grad[i4*dim+0]+=g4x;grad[i4*dim+1]+=g4y;grad[i4*dim+2]+=g4z;
}

// ---- Improper torsion (planarity) energy: E = w * (1 - cos(2ω)) ----
// Identity: with c = cos(ω) = calc_cos_phi(ic,i0,i1,i2),  1-cos(2ω) = 2-2c^2,
// which is exactly torsion_e(c, V=[0,w,0,0,0,0], s=[0,-1,0,0,0,0]).  Delegating to the
// (finite-difference-verified) torsion energy/gradient keeps E and dE/dx consistent.
// The previous standalone improper_g used a Blondel-Karplus dφ/dx that did NOT match this
// ω definition (FD error ~150) — which silently zeroed the ETK refinement.
inline float improper_e(const device float* pos,
    int ic,int i0,int i1,int i2, float wt, int dim
) {
    float c = calc_cos_phi(pos, ic,i0,i1,i2, dim);
    return torsion_e(c, 0.0f,wt,0.0f,0.0f,0.0f,0.0f, 0.0f,-1.0f,0.0f,0.0f,0.0f,0.0f);
}

// ---- Improper torsion gradient (delegates to fixed torsion_g) ----
inline void improper_g(const device float* pos, device float* grad,
    int ic,int i0,int i1,int i2, float wt, int dim
) {
    torsion_g(pos, grad, ic,i0,i1,i2,
        0.0f,wt,0.0f,0.0f,0.0f,0.0f, 0.0f,-1.0f,0.0f,0.0f,0.0f,0.0f, dim);
}

// ---- 1-4 distance constraint: flat-bottom harmonic ----
inline float dist14_e(const device float* pos, int a, int b,
    float lb, float ub, float wt, int dim
) {
    float d2=0.0f;
    for (int d=0;d<3;d++){float df=pos[a*dim+d]-pos[b*dim+d]; d2+=df*df;}
    float dist=sqrt(d2+1e-12f);
    if (dist<lb){float v=dist-lb; return wt*v*v;}
    if (dist>ub){float v=dist-ub; return wt*v*v;}
    return 0.0f;
}

inline void dist14_g(const device float* pos, device float* grad,
    int a, int b, float lb, float ub, float wt, int dim
) {
    float df[3]; float d2=0.0f;
    for (int d=0;d<3;d++){df[d]=pos[a*dim+d]-pos[b*dim+d]; d2+=df[d]*df[d];}
    float dist=sqrt(d2+1e-12f);
    float pf=0.0f;
    if (dist<lb) pf=wt*2.0f*(dist-lb)/dist;
    else if (dist>ub) pf=wt*2.0f*(dist-ub)/dist;
    if (pf!=0.0f) {
        for (int d=0;d<3;d++){float g=pf*df[d]; grad[a*dim+d]+=g; grad[b*dim+d]-=g;}
    }
}

// ---- Torsion gradient as four role vectors (phase 1 of the parallel gradient) ----
// Same expressions as torsion_g, but the vectors it would add to atoms
// i1..i4 are stored to out[0..11] instead. A degenerate torsion, which
// torsion_g skips, stores -0.0f: x + (-0.0f) == x for every x, so adding it
// leaves the sum bit-identical to not adding anything.
inline void torsion_g_roles(const device float* pos, device float* out,
    int i1,int i2,int i3,int i4,
    float V0,float V1,float V2,float V3,float V4,float V5,
    float s0,float s1,float s2,float s3,float s4,float s5, int dim
) {
    float r1[3],r2[3],r3[3],r4[3];
    for (int d=0;d<3;d++) {
        r1[d]=pos[i1*dim+d]-pos[i2*dim+d]; r2[d]=pos[i3*dim+d]-pos[i2*dim+d];
        r3[d]=-r2[d]; r4[d]=pos[i4*dim+d]-pos[i3*dim+d];
    }
    float t0x=r1[1]*r2[2]-r1[2]*r2[1],t0y=r1[2]*r2[0]-r1[0]*r2[2],t0z=r1[0]*r2[1]-r1[1]*r2[0];
    float t1x=r3[1]*r4[2]-r3[2]*r4[1],t1y=r3[2]*r4[0]-r3[0]*r4[2],t1z=r3[0]*r4[1]-r3[1]*r4[0];
    float d02=t0x*t0x+t0y*t0y+t0z*t0z, d12=t1x*t1x+t1y*t1y+t1z*t1z;
    if (d02<1e-16f||d12<1e-16f) { for (int k=0;k<12;k++) out[k]=-0.0f; return; }
    float inv0=rsqrt(max(d02,1e-16f)),inv1=rsqrt(max(d12,1e-16f));
    float tnx0=t0x*inv0,tny0=t0y*inv0,tnz0=t0z*inv0;
    float tnx1=t1x*inv1,tny1=t1y*inv1,tnz1=t1z*inv1;
    float cp=clamp(tnx0*tnx1+tny0*tny1+tnz0*tnz1,-1.0f,1.0f);
    float sp2=1.0f-cp*cp, sp=sqrt(max(sp2,0.0f));
    float c=cp,c2=c*c,c3=c*c2,c4=c*c3;
    float dE=-s0*V0*sp-2.0f*s1*V1*(2.0f*c*sp)-3.0f*s2*V2*(4.0f*c2*sp-sp)
        -4.0f*s3*V3*(8.0f*c3*sp-4.0f*c*sp)-5.0f*s4*V4*(16.0f*c4*sp-12.0f*c2*sp+sp)
        -6.0f*s5*V5*(32.0f*c4*c*sp-32.0f*c3*sp+6.0f*c*sp);
    float st;
    if (abs(sp)>1e-8f) st=-dE/sp; else st=-dE/max(abs(cp),1e-16f)*sign(cp+1e-30f);
    float dcx0=inv0*(tnx1-cp*tnx0),dcy0=inv0*(tny1-cp*tny0),dcz0=inv0*(tnz1-cp*tnz0);
    float dcx1=inv1*(tnx0-cp*tnx1),dcy1=inv1*(tny0-cp*tny1),dcz1=inv1*(tnz0-cp*tnz1);
    float g1x=st*(dcz0*r2[1]-dcy0*r2[2]),g1y=st*(dcx0*r2[2]-dcz0*r2[0]),g1z=st*(dcy0*r2[0]-dcx0*r2[1]);
    float g4x=st*(dcy1*r3[2]-dcz1*r3[1]),g4y=st*(dcz1*r3[0]-dcx1*r3[2]),g4z=st*(dcx1*r3[1]-dcy1*r3[0]);
    float g2x=st*(dcy0*(r2[2]-r1[2])+dcz0*(r1[1]-r2[1])+dcy1*(-r4[2])+dcz1*r4[1]);
    float g2y=st*(dcx0*(r1[2]-r2[2])+dcz0*(r2[0]-r1[0])+dcx1*r4[2]+dcz1*(-r4[0]));
    float g2z=st*(dcx0*(r2[1]-r1[1])+dcy0*(r1[0]-r2[0])+dcx1*(-r4[1])+dcy1*r4[0]);
    float g3x=st*(dcy0*r1[2]+dcz0*(-r1[1])+dcy1*(r4[2]-r3[2])+dcz1*(r3[1]-r4[1]));
    float g3y=st*(dcx0*(-r1[2])+dcz0*r1[0]+dcx1*(r3[2]-r4[2])+dcz1*(r4[0]-r3[0]));
    float g3z=st*(dcx0*r1[1]+dcy0*(-r1[0])+dcx1*(r4[1]-r3[1])+dcy1*(r3[0]-r4[0]));
    out[0]=g1x; out[1]=g1y; out[2]=g1z;
    out[3]=g2x; out[4]=g2y; out[5]=g2z;
    out[6]=g3x; out[7]=g3y; out[8]=g3z;
    out[9]=g4x; out[10]=g4y; out[11]=g4z;
}

// ---- Flat-bottom distance prefactor (phase 1): dist14_g's pf ----
inline float dist14_pf(const device float* pos, int a, int b, float lb, float ub, float wt, int dim) {
    float df[3]; float d2=0.0f;
    for (int d=0;d<3;d++){df[d]=pos[a*dim+d]-pos[b*dim+d]; d2+=df[d]*df[d];}
    float dist=sqrt(d2+1e-12f);
    float pf=0.0f;
    if (dist<lb) pf=wt*2.0f*(dist-lb)/dist;
    else if (dist>ub) pf=wt*2.0f*(dist-ub)/dist;
    return pf;
}

// ---- Phase 2: one lane per atom, terms gathered through the CSR index ----
// csr_off/csr_ent list, per (molecule, term type, atom), the terms touching
// the atom as ``term * 4 + role`` in increasing term order, then role: the
// order in which the serial loop adds into it, so the sum is bit-identical.
// Type order: torsion, improper, 1-2, 1-3, 1-4 (the serial loop's order).
// Input pointers are template parameters: MLX hands small inputs to the
// kernel in the constant address space and large ones in device.
template <typename OffT, typename EntT>
inline void etk_add_pair_terms(thread float* acc, const thread float* own,
    const device float* pos, const device float* pf_arr, int t_start,
    OffT csr_off, EntT csr_ent2,
    int slot, int atom_off, int dim
) {
    for (int e = csr_off[slot]; e < csr_off[slot + 1]; e++) {
        int en = csr_ent2[2*e]; int other = csr_ent2[2*e+1];
        int role = en & 3;
        float pf = pf_arr[(en >> 2) - t_start];
        if (pf == 0.0f) continue;
        // df = pos[a] - pos[b], formed exactly as dist14_g does.
        for (int d = 0; d < 3; d++) {
            float po = pos[(atom_off + other) * dim + d];
            float df = (role == 0) ? (own[d] - po) : (po - own[d]);
            float g = pf * df;
            if (role == 0) acc[d] += g; else acc[d] -= g;
        }
    }
}

// ---- Reductions over the TPM lanes ----
// With TPM == 32 the threadgroup is exactly one SIMD-group, so the reduction
// runs in registers with xor-shuffles and needs no threadgroup memory or
// barriers. The shuffle distances (16, 8, 4, 2, 1) are the strides of the
// tree below, so each partial sum pairs the same operands in the same order
// and every lane ends with the tree's exact result (float + is commutative).
inline float reduce_sum(float v, threadgroup float* s, uint tid, uint n) {
#if TPM == 32
    v += simd_shuffle_xor(v, 16);
    v += simd_shuffle_xor(v, 8);
    v += simd_shuffle_xor(v, 4);
    v += simd_shuffle_xor(v, 2);
    v += simd_shuffle_xor(v, 1);
    return v;
#else
    threadgroup_barrier(mem_flags::mem_threadgroup);
    s[tid] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = n / 2; stride > 0; stride >>= 1) {
        if (tid < stride) s[tid] += s[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    return s[0];
#endif
}
inline float reduce_max(float v, threadgroup float* s, uint tid, uint n) {
#if TPM == 32
    v = max(v, simd_shuffle_xor(v, 16));
    v = max(v, simd_shuffle_xor(v, 8));
    v = max(v, simd_shuffle_xor(v, 4));
    v = max(v, simd_shuffle_xor(v, 2));
    v = max(v, simd_shuffle_xor(v, 1));
    return v;
#else
    threadgroup_barrier(mem_flags::mem_threadgroup);
    s[tid] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = n / 2; stride > 0; stride >>= 1) {
        if (tid < stride) s[tid] = max(s[tid], s[tid + stride]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    return s[0];
#endif
}
// Element-wise helpers: lane tid only touches elements tid, tid+tpm, ...,
// the same elements every helper and every element-wise loop of the kernel
// gives it, so no barrier is needed between them (a thread sees its own
// writes). A barrier is needed only where a lane reads elements another lane
// wrote: positions before an energy/gradient evaluation, and the gradient,
// which phase 2 writes atom by atom.
inline float parallel_dot(const device float* a, const device float* b,
    int n, uint tid, uint tpm, threadgroup float* s) {
    float sum = 0.0f;
    for (int i = (int)tid; i < n; i += (int)tpm) sum += a[i] * b[i];
    return reduce_sum(sum, s, tid, tpm);
}
inline void parallel_saxpy(device float* a, float alpha, const device float* b,
    int n, uint tid, uint tpm) {
    for (int i = (int)tid; i < n; i += (int)tpm) a[i] += alpha * b[i];
}
inline void parallel_scale(device float* a, float alpha, int n, uint tid, uint tpm) {
    for (int i = (int)tid; i < n; i += (int)tpm) a[i] *= alpha;
}
inline void parallel_copy(device float* dst, const device float* src, int n, uint tid, uint tpm) {
    for (int i = (int)tid; i < n; i += (int)tpm) dst[i] = src[i];
}
inline void parallel_set(device float* a, float val, int n, uint tid, uint tpm) {
    for (int i = (int)tid; i < n; i += (int)tpm) a[i] = val;
}
inline void parallel_neg_copy(device float* dst, const device float* src, int n, uint tid, uint tpm) {
    for (int i = (int)tid; i < n; i += (int)tpm) dst[i] = -src[i];
}
"""

# ---------------------------------------------------------------------------
# ETK kernel body — conf_to_mol + atom_off adaptation
# Pre-adds atom_off to LOCAL constraint indices before calling helpers
# ---------------------------------------------------------------------------

_ETK_BODY = r"""
    uint tid = thread_position_in_threadgroup.x;
    uint conf_idx = threadgroup_position_in_grid.x;
    const uint tpm = TPM;
    const int lbfgs_m = LBFGS_M;
    threadgroup float shared[TPM];

    int n_confs_cfg = (int)config[0];
    int max_iters = (int)config[1];
    float grad_tol_v = config[2];
    const int dim = DIM;  // compiled for 3D (config[3] is the same value)
    int total_pos_size = (int)config[4];

    if ((int)conf_idx >= n_confs_cfg) return;

    // Shared constraint indirection
    int mol_idx = conf_to_mol[conf_idx];
    int atom_off = conf_atom_starts[conf_idx];
    int n_atoms = mol_n_atoms[mol_idx];
    int n_vars = n_atoms * dim;

    // Constraint ranges (per molecule — SHARED)
    // term_starts = [torsion | improper | 1-2 | 1-3 | 1-4] ranges, (N+1) each
    const int ts = (int)config[5];
    int tor_s = term_starts[mol_idx], tor_e = term_starts[mol_idx+1];
    int imp_s = term_starts[ts+mol_idx], imp_e = term_starts[ts+mol_idx+1];
    int d12_s = term_starts[2*ts+mol_idx], d12_e = term_starts[2*ts+mol_idx+1];
    int d13_s = term_starts[3*ts+mol_idx], d13_e = term_starts[3*ts+mol_idx+1];
    int d14_s = term_starts[4*ts+mol_idx], d14_e = term_starts[4*ts+mol_idx+1];

    int lbfgs_start = lbfgs_history_starts[conf_idx];

    parallel_copy(&out_pos[atom_off*dim], &pos[atom_off*dim], n_vars, tid, tpm);
    threadgroup_barrier(mem_flags::mem_device);  // energy reads every lane's atoms

    device float* my_pos = &out_pos[atom_off*dim];
    device float* my_grad = &work_grad[atom_off*dim];
    device float* my_dir = &work_dir[atom_off*dim];
    device float* my_old_pos = &work_scratch[atom_off*dim];
    device float* my_old_grad = &work_scratch[total_pos_size + atom_off*dim];
    device float* my_q = &work_scratch[2*total_pos_size + atom_off*dim];
    device float* my_S = &work_lbfgs[lbfgs_start];
    device float* my_Y = &work_lbfgs[lbfgs_start + lbfgs_m*n_vars];
    // L-BFGS scalars: every lane computes identical values, so each keeps its own copy.
    float my_rho[LBFGS_M];
    float my_alpha[LBFGS_M];

    // ---- Gradient of the current positions into my_grad (all lanes return) ----
#if GRAD_MODE == 1
    device float* my_scr = &work_scratch[3*total_pos_size + conf_scr_base[conf_idx]];
    const int n_tor = tor_e - tor_s, n_imp = imp_e - imp_s;
    const int n12 = d12_e - d12_s, n13 = d13_e - d13_s;
    device float* scr_tor = my_scr;
    device float* scr_imp = my_scr + 12 * n_tor;
    device float* scr_d12 = scr_imp + 12 * n_imp;
    device float* scr_d13 = scr_d12 + n12;
    device float* scr_d14 = scr_d13 + n13;
    const int slot0 = csr_slot_base[mol_idx];
    #define ETK_GRADIENT() \
        for (int t=tor_s+(int)tid;t<tor_e;t+=(int)tpm) { \
            int a1=torsion_quads[t*4]+atom_off,a2=torsion_quads[t*4+1]+atom_off; \
            int a3=torsion_quads[t*4+2]+atom_off,a4=torsion_quads[t*4+3]+atom_off; \
            torsion_g_roles(out_pos,&scr_tor[12*(t-tor_s)],a1,a2,a3,a4, \
                torsion_V[t*6],torsion_V[t*6+1],torsion_V[t*6+2], \
                torsion_V[t*6+3],torsion_V[t*6+4],torsion_V[t*6+5], \
                torsion_signs_arr[t*6],torsion_signs_arr[t*6+1],torsion_signs_arr[t*6+2], \
                torsion_signs_arr[t*6+3],torsion_signs_arr[t*6+4],torsion_signs_arr[t*6+5],dim); \
        } \
        for (int t=imp_s+(int)tid;t<imp_e;t+=(int)tpm) { \
            int ic=improper_quads[t*4]+atom_off,i0=improper_quads[t*4+1]+atom_off; \
            int i1=improper_quads[t*4+2]+atom_off,i2=improper_quads[t*4+3]+atom_off; \
            torsion_g_roles(out_pos,&scr_imp[12*(t-imp_s)],ic,i0,i1,i2, \
                0.0f,improper_w[t],0.0f,0.0f,0.0f,0.0f, 0.0f,-1.0f,0.0f,0.0f,0.0f,0.0f, dim); \
        } \
        for (int t=d12_s+(int)tid;t<d12_e;t+=(int)tpm) \
            scr_d12[t-d12_s]=dist14_pf(out_pos,d12_pairs[t*2]+atom_off,d12_pairs[t*2+1]+atom_off, \
                d12_bounds[t*3],d12_bounds[t*3+1],d12_bounds[t*3+2],dim); \
        for (int t=d13_s+(int)tid;t<d13_e;t+=(int)tpm) \
            scr_d13[t-d13_s]=dist14_pf(out_pos,d13_pairs[t*2]+atom_off,d13_pairs[t*2+1]+atom_off, \
                d13_bounds[t*3],d13_bounds[t*3+1],d13_bounds[t*3+2],dim); \
        for (int t=d14_s+(int)tid;t<d14_e;t+=(int)tpm) \
            scr_d14[t-d14_s]=dist14_pf(out_pos,d14_pairs[t*2]+atom_off,d14_pairs[t*2+1]+atom_off, \
                d14_bounds[t*3],d14_bounds[t*3+1],d14_bounds[t*3+2],dim); \
        threadgroup_barrier(mem_flags::mem_device); \
        for (int a=(int)tid;a<n_atoms;a+=(int)tpm) { \
            float acc[3]={0.0f,0.0f,0.0f}; \
            float own[3]; for (int d=0;d<3;d++) own[d]=out_pos[(atom_off+a)*dim+d]; \
            int sl=slot0+a; \
            for (int e=csr_off[sl];e<csr_off[sl+1];e++) { \
                int en=csr_ent2[2*e]; const device float* v=&scr_tor[12*((en>>2)-tor_s)+3*(en&3)]; \
                acc[0]+=v[0]; acc[1]+=v[1]; acc[2]+=v[2]; \
            } \
            sl+=n_atoms; \
            for (int e=csr_off[sl];e<csr_off[sl+1];e++) { \
                int en=csr_ent2[2*e]; const device float* v=&scr_imp[12*((en>>2)-imp_s)+3*(en&3)]; \
                acc[0]+=v[0]; acc[1]+=v[1]; acc[2]+=v[2]; \
            } \
            etk_add_pair_terms(acc,own,out_pos,scr_d12,d12_s,csr_off,csr_ent2,slot0+2*n_atoms+a,atom_off,dim); \
            etk_add_pair_terms(acc,own,out_pos,scr_d13,d13_s,csr_off,csr_ent2,slot0+3*n_atoms+a,atom_off,dim); \
            etk_add_pair_terms(acc,own,out_pos,scr_d14,d14_s,csr_off,csr_ent2,slot0+4*n_atoms+a,atom_off,dim); \
            for (int d=0;d<3;d++) my_grad[a*dim+d]=acc[d]; \
        } \
        threadgroup_barrier(mem_flags::mem_device);
#else
    // Reference: thread 0 adds every term serially.
    #define ETK_GRADIENT() \
        parallel_set(my_grad, 0.0f, n_vars, tid, tpm); \
        threadgroup_barrier(mem_flags::mem_device); \
        if (tid == 0) { \
            for (int t=tor_s;t<tor_e;t++){int a1=torsion_quads[t*4]+atom_off,a2=torsion_quads[t*4+1]+atom_off,a3=torsion_quads[t*4+2]+atom_off,a4=torsion_quads[t*4+3]+atom_off; \
                torsion_g(out_pos,work_grad,a1,a2,a3,a4,torsion_V[t*6],torsion_V[t*6+1],torsion_V[t*6+2],torsion_V[t*6+3],torsion_V[t*6+4],torsion_V[t*6+5], \
                    torsion_signs_arr[t*6],torsion_signs_arr[t*6+1],torsion_signs_arr[t*6+2],torsion_signs_arr[t*6+3],torsion_signs_arr[t*6+4],torsion_signs_arr[t*6+5],dim);} \
            for (int t=imp_s;t<imp_e;t++){int ic=improper_quads[t*4]+atom_off,i0=improper_quads[t*4+1]+atom_off,i1=improper_quads[t*4+2]+atom_off,i2=improper_quads[t*4+3]+atom_off; \
                improper_g(out_pos,work_grad,ic,i0,i1,i2,improper_w[t],dim);} \
            for (int t=d12_s;t<d12_e;t++){int a=d12_pairs[t*2]+atom_off,b=d12_pairs[t*2+1]+atom_off; \
                dist14_g(out_pos,work_grad,a,b,d12_bounds[t*3],d12_bounds[t*3+1],d12_bounds[t*3+2],dim);} \
            for (int t=d13_s;t<d13_e;t++){int a=d13_pairs[t*2]+atom_off,b=d13_pairs[t*2+1]+atom_off; \
                dist14_g(out_pos,work_grad,a,b,d13_bounds[t*3],d13_bounds[t*3+1],d13_bounds[t*3+2],dim);} \
            for (int t=d14_s;t<d14_e;t++){int a=d14_pairs[t*2]+atom_off,b=d14_pairs[t*2+1]+atom_off; \
                dist14_g(out_pos,work_grad,a,b,d14_bounds[t*3],d14_bounds[t*3+1],d14_bounds[t*3+2],dim);} \
        } \
        threadgroup_barrier(mem_flags::mem_device);
#endif

    // ---- Energy (parallel) + gradient ----
    float local_e = 0.0f;

    // Torsion energy (pre-add atom_off to LOCAL indices)
    for (int t=tor_s+(int)tid; t<tor_e; t+=(int)tpm) {
        int a1=torsion_quads[t*4]+atom_off, a2=torsion_quads[t*4+1]+atom_off;
        int a3=torsion_quads[t*4+2]+atom_off, a4=torsion_quads[t*4+3]+atom_off;
        float cp=calc_cos_phi(out_pos, a1,a2,a3,a4, dim);
        local_e += torsion_e(cp, torsion_V[t*6],torsion_V[t*6+1],torsion_V[t*6+2],
            torsion_V[t*6+3],torsion_V[t*6+4],torsion_V[t*6+5],
            torsion_signs_arr[t*6],torsion_signs_arr[t*6+1],torsion_signs_arr[t*6+2],
            torsion_signs_arr[t*6+3],torsion_signs_arr[t*6+4],torsion_signs_arr[t*6+5]);
    }
    // Improper energy
    for (int t=imp_s+(int)tid; t<imp_e; t+=(int)tpm) {
        int ic=improper_quads[t*4]+atom_off, i0=improper_quads[t*4+1]+atom_off;
        int i1=improper_quads[t*4+2]+atom_off, i2=improper_quads[t*4+3]+atom_off;
        local_e += improper_e(out_pos, ic,i0,i1,i2, improper_w[t], dim);
    }
    // 1-2 bond distance energy
    for (int t=d12_s+(int)tid; t<d12_e; t+=(int)tpm) {
        int a=d12_pairs[t*2]+atom_off, b=d12_pairs[t*2+1]+atom_off;
        local_e += dist14_e(out_pos, a,b, d12_bounds[t*3],d12_bounds[t*3+1],d12_bounds[t*3+2], dim);
    }
    // 1-3 angle distance energy
    for (int t=d13_s+(int)tid; t<d13_e; t+=(int)tpm) {
        int a=d13_pairs[t*2]+atom_off, b=d13_pairs[t*2+1]+atom_off;
        local_e += dist14_e(out_pos, a,b, d13_bounds[t*3],d13_bounds[t*3+1],d13_bounds[t*3+2], dim);
    }
    // 1-4 distance energy
    for (int t=d14_s+(int)tid; t<d14_e; t+=(int)tpm) {
        int a=d14_pairs[t*2]+atom_off, b=d14_pairs[t*2+1]+atom_off;
        local_e += dist14_e(out_pos, a,b, d14_bounds[t*3],d14_bounds[t*3+1],d14_bounds[t*3+2], dim);
    }
    float energy = reduce_sum(local_e, shared, tid, tpm);

    ETK_GRADIENT();

    // ---- L-BFGS loop (identical to DG kernel) ----
    parallel_neg_copy(my_dir, my_grad, n_vars, tid, tpm);
    float lss=0.0f; for (int i=(int)tid;i<n_vars;i+=(int)tpm) lss+=my_pos[i]*my_pos[i];
    float max_step=MAX_STEP_FACTOR*max(sqrt(reduce_sum(lss, shared, tid, tpm)),(float)n_vars);
    int status=1, hist_count=0, hist_idx=0;

    for (int iter=0; iter<max_iters && status==1; iter++) {
        parallel_copy(my_old_pos, my_pos, n_vars, tid, tpm);
        float old_energy = energy;

        float ld2=0.0f; for (int i=(int)tid;i<n_vars;i+=(int)tpm) ld2+=my_dir[i]*my_dir[i];
        float dn=sqrt(reduce_sum(ld2, shared, tid, tpm));
        if (dn>max_step) parallel_scale(my_dir,max_step/dn,n_vars,tid,tpm);
        float slope=parallel_dot(my_dir,my_grad,n_vars,tid,tpm,shared);
        float ltm=0.0f; for (int i=(int)tid;i<n_vars;i+=(int)tpm){float t=abs(my_dir[i])/max(abs(my_pos[i]),1.0f);if(t>ltm)ltm=t;}
        float lmin=MOVETOL/max(reduce_max(ltm, shared, tid, tpm),1e-30f);

        float lam=1.0f,prev_lam=1.0f,prev_e=old_energy; bool ls_done=false;
        for (int ls=0; ls<MAX_LS_ITERS && !ls_done; ls++) {
            if (lam<lmin){parallel_copy(my_pos,my_old_pos,n_vars,tid,tpm);ls_done=true;break;}
            for (int i=(int)tid;i<n_vars;i+=(int)tpm) my_pos[i]=my_old_pos[i]+lam*my_dir[i];
            threadgroup_barrier(mem_flags::mem_device);

            // Trial energy (parallel)
            float lte=0.0f;
            for (int t=tor_s+(int)tid;t<tor_e;t+=(int)tpm){
                int a1=torsion_quads[t*4]+atom_off,a2=torsion_quads[t*4+1]+atom_off,a3=torsion_quads[t*4+2]+atom_off,a4=torsion_quads[t*4+3]+atom_off;
                float cp=calc_cos_phi(out_pos,a1,a2,a3,a4,dim);
                lte+=torsion_e(cp,torsion_V[t*6],torsion_V[t*6+1],torsion_V[t*6+2],torsion_V[t*6+3],torsion_V[t*6+4],torsion_V[t*6+5],
                    torsion_signs_arr[t*6],torsion_signs_arr[t*6+1],torsion_signs_arr[t*6+2],torsion_signs_arr[t*6+3],torsion_signs_arr[t*6+4],torsion_signs_arr[t*6+5]);}
            for (int t=imp_s+(int)tid;t<imp_e;t+=(int)tpm){
                int ic=improper_quads[t*4]+atom_off,i0=improper_quads[t*4+1]+atom_off,i1=improper_quads[t*4+2]+atom_off,i2=improper_quads[t*4+3]+atom_off;
                lte+=improper_e(out_pos,ic,i0,i1,i2,improper_w[t],dim);}
            for (int t=d12_s+(int)tid;t<d12_e;t+=(int)tpm){
                int a=d12_pairs[t*2]+atom_off,b=d12_pairs[t*2+1]+atom_off;
                lte+=dist14_e(out_pos,a,b,d12_bounds[t*3],d12_bounds[t*3+1],d12_bounds[t*3+2],dim);}
            for (int t=d13_s+(int)tid;t<d13_e;t+=(int)tpm){
                int a=d13_pairs[t*2]+atom_off,b=d13_pairs[t*2+1]+atom_off;
                lte+=dist14_e(out_pos,a,b,d13_bounds[t*3],d13_bounds[t*3+1],d13_bounds[t*3+2],dim);}
            for (int t=d14_s+(int)tid;t<d14_e;t+=(int)tpm){
                int a=d14_pairs[t*2]+atom_off,b=d14_pairs[t*2+1]+atom_off;
                lte+=dist14_e(out_pos,a,b,d14_bounds[t*3],d14_bounds[t*3+1],d14_bounds[t*3+2],dim);}
            float trial_e=reduce_sum(lte, shared, tid, tpm);

            if (trial_e-old_energy<=FUNCTOL*lam*slope){energy=trial_e;ls_done=true;}
            else {
                float tl;
                if (ls==0) tl=-slope/(2.0f*(trial_e-old_energy-slope));
                else {float r1=trial_e-old_energy-lam*slope,r2=prev_e-old_energy-prev_lam*slope;
                    float ls2=lam*lam,l2s=prev_lam*prev_lam,dv=lam-prev_lam;
                    if(abs(dv)<1e-30f)tl=0.5f*lam;
                    else{float a=(r1/ls2-r2/l2s)/dv,b=(-prev_lam*r1/ls2+lam*r2/l2s)/dv;
                        if(abs(a)<1e-30f)tl=(abs(b)>1e-30f)?-slope/(2.0f*b):0.5f*lam;
                        else{float disc=b*b-3.0f*a*slope;if(disc<0.0f)tl=0.5f*lam;else if(b<=0.0f)tl=(-b+sqrt(disc))/(3.0f*a);else tl=-slope/(b+sqrt(disc));}}}
                tl=clamp(tl,0.1f*lam,0.5f*lam);prev_lam=lam;prev_e=trial_e;lam=tl;
            }
        }
        if (!ls_done) parallel_copy(my_pos,my_old_pos,n_vars,tid,tpm);

        for (int i=(int)tid;i<n_vars;i+=(int)tpm) my_old_pos[i]=my_pos[i]-my_old_pos[i];
        threadgroup_barrier(mem_flags::mem_device);
        float ltx=0.0f; for (int i=(int)tid;i<n_vars;i+=(int)tpm){float t=abs(my_old_pos[i])/max(abs(my_pos[i]),1.0f);if(t>ltx)ltx=t;}
        if(reduce_max(ltx, shared, tid, tpm)<TOLX){status=0;break;}

        parallel_copy(my_old_grad,my_grad,n_vars,tid,tpm);

        // `energy` already holds trial_e of the accepted step, computed at these
        // exact positions by the same code: re-evaluating it gave the same bits.

        ETK_GRADIENT();

        float lgt=0.0f; for (int i=(int)tid;i<n_vars;i+=(int)tpm){float t=abs(my_grad[i])*max(abs(my_pos[i]),1.0f);if(t>lgt)lgt=t;}
        if(reduce_max(lgt, shared, tid, tpm)/max(energy,1.0f)<grad_tol_v){status=0;break;}

        // L-BFGS update
        for (int i=(int)tid;i<n_vars;i+=(int)tpm) my_q[i]=my_grad[i]-my_old_grad[i];
        threadgroup_barrier(mem_flags::mem_device);
        float ys=parallel_dot(my_q,my_old_pos,n_vars,tid,tpm,shared);
        if (ys>1e-10f){int sl=hist_idx%lbfgs_m;parallel_copy(&my_S[sl*n_vars],my_old_pos,n_vars,tid,tpm);
            parallel_copy(&my_Y[sl*n_vars],my_q,n_vars,tid,tpm);my_rho[sl]=1.0f/ys;hist_idx++;if(hist_count<lbfgs_m)hist_count++;}

        parallel_copy(my_q,my_grad,n_vars,tid,tpm);
        for (int j=hist_count-1;j>=0;j--){int sl=(hist_idx-1-(hist_count-1-j))%lbfgs_m;if(sl<0)sl+=lbfgs_m;
            float aj=my_rho[sl]*parallel_dot(&my_S[sl*n_vars],my_q,n_vars,tid,tpm,shared);
            my_alpha[j]=aj;
            parallel_saxpy(my_q,-aj,&my_Y[sl*n_vars],n_vars,tid,tpm);}
        if (hist_count>0){int nw=(hist_idx-1)%lbfgs_m;if(nw<0)nw+=lbfgs_m;
            float sy=parallel_dot(&my_S[nw*n_vars],&my_Y[nw*n_vars],n_vars,tid,tpm,shared);
            float yy=parallel_dot(&my_Y[nw*n_vars],&my_Y[nw*n_vars],n_vars,tid,tpm,shared);
            parallel_scale(my_q,sy/max(yy,1e-30f),n_vars,tid,tpm);}
        for (int j=0;j<hist_count;j++){int sl=(hist_idx-1-(hist_count-1-j))%lbfgs_m;if(sl<0)sl+=lbfgs_m;
            float bj=my_rho[sl]*parallel_dot(&my_Y[sl*n_vars],my_q,n_vars,tid,tpm,shared);
            parallel_saxpy(my_q,my_alpha[j]-bj,&my_S[sl*n_vars],n_vars,tid,tpm);}
        parallel_neg_copy(my_dir,my_q,n_vars,tid,tpm);
    }
    if (tid==0){out_energies[conf_idx]=energy;out_statuses[conf_idx]=status;}
"""

# ---------------------------------------------------------------------------
# Kernel build + dispatch
# ---------------------------------------------------------------------------

_etk_kernel_cache: dict[tuple, object] = {}


def _get_etk_kernel(tpm: int = DEFAULT_TPM, lbfgs_m: int = DEFAULT_LBFGS_M, grad_mode: int = GRAD_GATHER):
    key = (tpm, lbfgs_m, int(grad_mode))
    if key not in _etk_kernel_cache:
        header = (f"#define GRAD_MODE {int(grad_mode)}\n#define DIM 3\n"
                  + _ETK_HEADER.replace("TPM", str(tpm)).replace("LBFGS_M", str(lbfgs_m)))
        source = _ETK_BODY.replace("TPM", str(tpm)).replace("LBFGS_M", str(lbfgs_m))
        _etk_kernel_cache[key] = mx.fast.metal_kernel(
            name=f"etk_lbfgs_shared_g{int(grad_mode)}",
            input_names=[
                "pos", "config",
                "conf_to_mol", "conf_atom_starts", "mol_n_atoms",
                "term_starts",
                "torsion_quads", "torsion_V", "torsion_signs_arr",
                "improper_quads", "improper_w",
                "d12_pairs", "d12_bounds",
                "d13_pairs", "d13_bounds",
                "d14_pairs", "d14_bounds",
                "lbfgs_history_starts",
                "csr_slot_base", "csr_off", "csr_ent2", "conf_scr_base",
            ],
            # Metal allows 31 buffers per kernel: the gradient scratch
            # follows work_scratch in the same buffer.
            output_names=[
                "out_pos", "out_energies", "out_statuses",
                "work_grad", "work_dir", "work_scratch",
                "work_lbfgs",
            ],
            header=header,
            source=source,
            ensure_row_contiguous=True,
        )
    return _etk_kernel_cache[key]


def _starts(batch, name):
    v = getattr(batch, name)
    return v if v is not None else np.zeros(batch.n_mols + 1, dtype=np.int32)


def _pairs(i1, i2):
    if i1 is None or len(i1) == 0:
        return np.zeros((0, 2), dtype=np.int32)
    return np.stack([i1, i2], axis=1)


def _quads(q):
    return np.zeros((0, 4), dtype=np.int32) if q is None or len(q) == 0 else np.asarray(q).reshape(-1, 4)


def _etk_grad_csr(batch: SharedConstraintBatch):
    """CSR of ETK terms per atom, types in the serial gradient's order:
    torsion, improper, 1-2, 1-3, 1-4."""
    arrays = (batch.mol_n_atoms, batch.etk_torsion_term_starts, batch.etk_torsion_idx,
              batch.etk_improper_term_starts, batch.etk_improper_idx,
              batch.etk_dist12_term_starts, batch.etk_dist12_idx1, batch.etk_dist12_idx2,
              batch.etk_dist13_term_starts, batch.etk_dist13_idx1, batch.etk_dist13_idx2,
              batch.etk_dist14_term_starts, batch.etk_dist14_idx1, batch.etk_dist14_idx2)
    return _cached_on_batch(batch, "_etk_grad_csr_cache", arrays, lambda: build_atom_term_csr(
        batch.mol_n_atoms, [
            (_starts(batch, "etk_torsion_term_starts"), _quads(batch.etk_torsion_idx)),
            (_starts(batch, "etk_improper_term_starts"), _quads(batch.etk_improper_idx)),
            (_starts(batch, "etk_dist12_term_starts"), _pairs(batch.etk_dist12_idx1, batch.etk_dist12_idx2)),
            (_starts(batch, "etk_dist13_term_starts"), _pairs(batch.etk_dist13_idx1, batch.etk_dist13_idx2)),
            (_starts(batch, "etk_dist14_term_starts"), _pairs(batch.etk_dist14_idx1, batch.etk_dist14_idx2)),
        ]))


def etk_minimize_shared(
    batch: SharedConstraintBatch,
    positions: np.ndarray,
    *,
    max_iters: int = 300,
    grad_tol: float = 1e-4,
    tpm: int = DEFAULT_TPM,
    lbfgs_m: int = DEFAULT_LBFGS_M,
    parallel_grad: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run ETK L-BFGS on all C conformers in parallel with shared constraints.

    Requires ETK fields in *batch* (call ``add_etk_to_batch`` first).

    Parameters
    ----------
    batch : SharedConstraintBatch
    positions : np.ndarray, shape (n_atoms_total * 3,)
        3D positions from DG stage (after 4D→3D extraction).
    parallel_grad : bool
        True (default): all TPM lanes compute the gradient, in two phases.
        Phase 1 strides the terms over the lanes and stores, per term, the
        vectors it adds to its atoms (torsion/improper) or its scalar
        prefactor (1-2/1-3/1-4 distances). Phase 2 gives each lane whole atoms
        and sums those contributions through a per-molecule "terms per atom"
        CSR index, in the serial loop's order: no atomics, deterministic, and
        bit-identical to ``parallel_grad=False``.
        False: thread 0 computes the gradient serially (reference).

    Returns
    -------
    out_positions, energies, statuses
    """
    grad_mode = GRAD_GATHER if parallel_grad else GRAD_SERIAL
    C = batch.n_confs_total
    dim = 3  # ETK is always 3D
    total_pos_size = int(batch.conf_atom_starts[-1]) * dim

    config = np.array([C, max_iters, grad_tol, dim, total_pos_size, batch.n_mols + 1], dtype=np.float32)

    # Pack torsion terms
    nt = len(batch.etk_torsion_idx) if batch.etk_torsion_idx is not None else 0
    if nt > 0:
        tor_quads = batch.etk_torsion_idx.flatten().astype(np.int32)
        tor_V = batch.etk_torsion_V.flatten().astype(np.float32)
        tor_signs = batch.etk_torsion_signs.flatten().astype(np.float32)
    else:
        tor_quads = np.zeros(4, dtype=np.int32)
        tor_V = np.zeros(6, dtype=np.float32)
        tor_signs = np.zeros(6, dtype=np.float32)

    # Pack improper terms
    ni = len(batch.etk_improper_idx) if batch.etk_improper_idx is not None else 0
    if ni > 0:
        imp_quads = batch.etk_improper_idx.flatten().astype(np.int32)
        imp_w = batch.etk_improper_weight.astype(np.float32)
    else:
        imp_quads = np.zeros(4, dtype=np.int32)
        imp_w = np.zeros(1, dtype=np.float32)

    # Pack 1-2 bond distance terms
    def _pack_dist(idx1, idx2, lb, ub, w):
        if idx1 is not None and len(idx1) > 0:
            p = np.stack([idx1, idx2], axis=1).flatten().astype(np.int32)
            b = np.stack([lb, ub, w], axis=1).flatten().astype(np.float32)
            return p, b
        return np.zeros(2, dtype=np.int32), np.zeros(3, dtype=np.float32)

    d12_pairs, d12_bounds = _pack_dist(
        batch.etk_dist12_idx1, batch.etk_dist12_idx2,
        batch.etk_dist12_lb, batch.etk_dist12_ub, batch.etk_dist12_weight)
    d13_pairs, d13_bounds = _pack_dist(
        batch.etk_dist13_idx1, batch.etk_dist13_idx2,
        batch.etk_dist13_lb, batch.etk_dist13_ub, batch.etk_dist13_weight)
    d14_pairs, d14_bounds = _pack_dist(
        batch.etk_dist14_idx1, batch.etk_dist14_idx2,
        batch.etk_dist14_lb, batch.etk_dist14_ub, batch.etk_dist14_weight)

    # L-BFGS history
    n_vars_c = batch.mol_n_atoms[batch.conf_to_mol].astype(np.int64) * dim
    lbfgs_starts = np.zeros(C + 1, dtype=np.int32)
    np.cumsum(2 * lbfgs_m * n_vars_c, out=lbfgs_starts[1:])
    total_lbfgs = int(lbfgs_starts[-1])

    # Gradient scratch per conformer: 12 floats per torsion/improper (the
    # four role vectors) + 1 prefactor per 1-2/1-3/1-4 term.
    if grad_mode == GRAD_GATHER:
        csr_slot_base, csr_off, csr_ent, csr_partner = _etk_grad_csr(batch)
        csr_ent2 = np.stack([csr_ent, csr_partner], axis=1).ravel()
        per_mol = (12 * np.diff(_starts(batch, "etk_torsion_term_starts").astype(np.int64))
                   + 12 * np.diff(_starts(batch, "etk_improper_term_starts").astype(np.int64))
                   + np.diff(_starts(batch, "etk_dist12_term_starts").astype(np.int64))
                   + np.diff(_starts(batch, "etk_dist13_term_starts").astype(np.int64))
                   + np.diff(_starts(batch, "etk_dist14_term_starts").astype(np.int64)))
    else:
        csr_slot_base = csr_off = np.zeros(1, dtype=np.int32)
        csr_ent2 = np.zeros(2, dtype=np.int32)
        per_mol = np.zeros(batch.n_mols, dtype=np.int64)
    conf_scr_base = np.zeros(C + 1, dtype=np.int32)
    np.cumsum(per_mol[batch.conf_to_mol], out=conf_scr_base[1:])
    total_scr = int(conf_scr_base[-1])

    kernel = _get_etk_kernel(tpm, lbfgs_m, grad_mode)
    results = kernel(
        inputs=[
            mx.array(positions),
            mx.array(config),
            mx.array(batch.conf_to_mol),
            mx.array(batch.conf_atom_starts),
            mx.array(batch.mol_n_atoms),
            mx.array(np.concatenate([
                _starts(batch, "etk_torsion_term_starts"), _starts(batch, "etk_improper_term_starts"),
                _starts(batch, "etk_dist12_term_starts"), _starts(batch, "etk_dist13_term_starts"),
                _starts(batch, "etk_dist14_term_starts")]).astype(np.int32)),
            mx.array(tor_quads),
            mx.array(tor_V),
            mx.array(tor_signs),
            mx.array(imp_quads),
            mx.array(imp_w),
            mx.array(d12_pairs),
            mx.array(d12_bounds),
            mx.array(d13_pairs),
            mx.array(d13_bounds),
            mx.array(d14_pairs),
            mx.array(d14_bounds),
            mx.array(lbfgs_starts[:-1]),
            mx.array(csr_slot_base),
            mx.array(csr_off),
            mx.array(csr_ent2),
            mx.array(conf_scr_base),
        ],
        grid=(C * tpm, 1, 1),
        threadgroup=(tpm, 1, 1),
        output_shapes=[
            (total_pos_size,), (C,), (C,),
            (total_pos_size,), (total_pos_size,), (3 * total_pos_size + total_scr,),
            (max(1, total_lbfgs),),
        ],
        output_dtypes=[
            mx.float32, mx.float32, mx.int32,
            mx.float32, mx.float32, mx.float32,
            mx.float32,
        ],
    )
    mx.eval(results[0], results[1], results[2])
    return np.array(results[0]), np.array(results[1]), np.array(results[2])
