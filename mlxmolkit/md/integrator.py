"""Batched Langevin dynamics with X-H constraints on the Apple GPU (OpenMM LangevinMiddle scheme).

Per step (OpenMM's LangevinMiddleIntegrator):
  v += dt F/m;  x' = x + dt/2 v;  v = a v + sqrt(1 - a^2) sqrt(kT/m) R;  x'' = x' + dt/2 v;
  SHAKE(x'' | reference x);  v = (x''_constrained - x) / dt
Constraints are solved per cluster (a heavy atom and its hydrogens -- OpenFF constrains X-H
bonds only), one GPU thread per cluster, iterated SHAKE.
The neighbour list is rebuilt every `nlist_every` steps on the GPU (all pairs within
cutoff + skin by minimum image, exceptions excluded): for ~3 nm boxes the cell grid is
2x2x2, so all-pairs costs the same as cells and is simpler.
"""

import mlx.core as mx
import numpy as np

from mlxmolkit.md import forcefield as ff
from mlxmolkit.md import kernels

KB = 0.0083144626  # kJ/mol/K

_NLIST_SRC = """
    uint gid = thread_position_in_grid.x;
    uint N = meta[0], B = meta[1], MAXNB = meta[2], MAXEX = meta[3];
    if (gid >= N * B) return;
    uint b = gid / N;
    uint li = gid - b * N;
    float3 Lb = float3(box[3 * b], box[3 * b + 1], box[3 * b + 2]);
    float3 invL = 1.0f / Lb;
    float rl2 = rl[0] * rl[0];
    float3 pi = float3(pos[3 * gid], pos[3 * gid + 1], pos[3 * gid + 2]);
    const device int* ex = excl + li * MAXEX;
    int nex = nexcl[li];
    uint cnt = 0;
    device int* out = nbr + (size_t)gid * MAXNB;
    for (uint lj = 0; lj < N; lj++) {
        if (lj == li) continue;
        uint gj = b * N + lj;
        float3 d = pi - float3(pos[3 * gj], pos[3 * gj + 1], pos[3 * gj + 2]);
        d -= Lb * rint(d * invL);
        if (dot(d, d) >= rl2) continue;
        bool skip = false;
        for (int e = 0; e < nex; e++) { if (ex[e] == (int)lj) { skip = true; break; } }
        if (skip) continue;
        if (cnt < MAXNB) out[cnt] = (int)lj;
        cnt++;
    }
    nnb[gid] = (int)min(cnt, MAXNB);
    overflow[gid] = (int)(cnt > MAXNB);
"""

_SHAKE_SRC = """
    uint gid = thread_position_in_grid.x;
    uint C = meta[0], B = meta[1], N = meta[2], ITERS = meta[3];
    if (gid >= C * B) return;
    uint b = gid / C;
    uint c = gid - b * C;
    uint off = b * N;
    int center = cl_center[c];
    int nh = cl_n[c];
    float mc = 1.0f / inv_m[center];
    float3 rc = float3(xnew[3 * (off + center)], xnew[3 * (off + center) + 1], xnew[3 * (off + center) + 2]);
    float3 oc = float3(xref[3 * (off + center)], xref[3 * (off + center) + 1], xref[3 * (off + center) + 2]);
    float3 rh[4]; float3 oh[4]; float d2[4]; float imh[4]; int ha[4];
    for (int k = 0; k < nh; k++) {
        ha[k] = cl_h[4 * c + k];
        uint a = off + ha[k];
        rh[k] = float3(xnew[3 * a], xnew[3 * a + 1], xnew[3 * a + 2]);
        oh[k] = float3(xref[3 * a], xref[3 * a + 1], xref[3 * a + 2]);
        d2[k] = cl_d[4 * c + k] * cl_d[4 * c + k];
        imh[k] = inv_m[ha[k]];
    }
    float imc = inv_m[center];
    for (uint it = 0; it < ITERS; it++) {
        for (int k = 0; k < nh; k++) {
            float3 s = rh[k] - rc;
            float3 so = oh[k] - oc;
            float diff = d2[k] - dot(s, s);
            float g = diff / (2.0f * dot(s, so) * (imc + imh[k]));
            rh[k] += g * imh[k] * so;
            rc -= g * imc * so;
        }
    }
    xout[3 * (off + center)] = rc.x; xout[3 * (off + center) + 1] = rc.y; xout[3 * (off + center) + 2] = rc.z;
    for (int k = 0; k < nh; k++) {
        uint a = off + ha[k];
        xout[3 * a] = rh[k].x; xout[3 * a + 1] = rh[k].y; xout[3 * a + 2] = rh[k].z;
    }
"""

_nlist_k = mx.fast.metal_kernel(name="md_nlist_allpairs", input_names=["pos", "box", "excl", "nexcl", "rl", "meta"],
                                output_names=["nbr", "nnb", "overflow"], source=_NLIST_SRC)
_shake_k = mx.fast.metal_kernel(name="md_shake_clusters",
                                input_names=["xnew", "xref", "inv_m", "cl_center", "cl_n", "cl_h", "cl_d", "meta"],
                                output_names=["xout"], source=_SHAKE_SRC)

_RATTLE_V_SRC = """
    uint gid = thread_position_in_grid.x;
    uint C = meta[0], B = meta[1], N = meta[2], ITERS = meta[3];
    if (gid >= C * B) return;
    uint b = gid / C;
    uint c = gid - b * C;
    uint off = b * N;
    int center = cl_center[c];
    int nh = cl_n[c];
    float imc = inv_m[center];
    uint ac = off + center;
    float3 rc = float3(x[3 * ac], x[3 * ac + 1], x[3 * ac + 2]);
    float3 vc = float3(v[3 * ac], v[3 * ac + 1], v[3 * ac + 2]);
    float3 rh[4]; float3 vh[4]; float imh[4]; int ha[4];
    for (int k = 0; k < nh; k++) {
        ha[k] = cl_h[4 * c + k];
        uint a = off + ha[k];
        rh[k] = float3(x[3 * a], x[3 * a + 1], x[3 * a + 2]);
        vh[k] = float3(v[3 * a], v[3 * a + 1], v[3 * a + 2]);
        imh[k] = inv_m[ha[k]];
    }
    for (uint it = 0; it < ITERS; it++) {
        for (int k = 0; k < nh; k++) {
            float3 s = rh[k] - rc;
            float g = dot(s, vh[k] - vc) / (dot(s, s) * (imc + imh[k]));
            vh[k] -= g * imh[k] * s;
            vc += g * imc * s;
        }
    }
    vout[3 * ac] = vc.x; vout[3 * ac + 1] = vc.y; vout[3 * ac + 2] = vc.z;
    for (int k = 0; k < nh; k++) {
        uint a = off + ha[k];
        vout[3 * a] = vh[k].x; vout[3 * a + 1] = vh[k].y; vout[3 * a + 2] = vh[k].z;
    }
"""
_rattle_k = mx.fast.metal_kernel(name="md_rattle_velocities",
                                 input_names=["x", "v", "inv_m", "cl_center", "cl_n", "cl_h", "meta"],
                                 output_names=["vout"], source=_RATTLE_V_SRC)


class Dynamics:
    def __init__(self, t: ff.Topology, d, B: int, dt_ps: float = 0.004, temperature: float = 298.0, friction: float = 1.0,
                 skin: float = 0.1, nlist_every: int = 10, shake_iters: int = 20, maxnb: int = 1400, seed: int = 0):
        self.t, self.B, self.dt, self.T = t, B, dt_ps, temperature
        self.pme = ff.PME(t, np.diag(np.asarray(d["box"])))
        self.bonded = kernels.BondedTerms(t)
        m = np.asarray(d["masses"], np.float64)
        self.inv_m = mx.array((1.0 / m).astype(np.float32))
        self.sqrt_kt_m = mx.array(np.sqrt(KB * temperature / m).astype(np.float32))[None, :, None]
        self.a = float(np.exp(-friction * dt_ps))
        self.skin, self.nlist_every, self.maxnb, self.shake_iters = skin, nlist_every, maxnb, shake_iters
        n = t.n
        # exclusion lists for the neighbour builder
        ex = [[] for _ in range(n)]
        for i, j in t.exc_set:
            ex[i].append(j)
            ex[j].append(i)
        maxex = max(len(e) for e in ex)
        excl = np.full((n, maxex), -1, np.int32)
        for i, e in enumerate(ex):
            excl[i, : len(e)] = e
        self.excl, self.nexcl, self.maxex = mx.array(excl.reshape(-1)), mx.array(np.array([len(e) for e in ex], np.int32)), maxex
        # constraint clusters (heavy atom + its H)
        cons = np.asarray(d["constraints"])
        clusters: dict[int, list[tuple[int, float]]] = {}
        for i, j, dist in cons:
            i, j = int(i), int(j)
            heavy, h = (i, j) if m[i] > m[j] else (j, i)
            clusters.setdefault(heavy, []).append((h, float(dist)))
        assert max(len(v) for v in clusters.values()) <= 4
        cen = np.array(sorted(clusters), np.int32)
        cl_h = np.full((len(cen), 4), -1, np.int32)
        cl_d = np.zeros((len(cen), 4), np.float32)
        for c, a in enumerate(cen):
            for k, (h, dist) in enumerate(clusters[int(a)]):
                cl_h[c, k], cl_d[c, k] = h, dist
        self.n_constraints = len(cons)
        self.cl = (mx.array(cen), mx.array(np.array([len(clusters[int(a)]) for a in cen], np.int32)), mx.array(cl_h.reshape(-1)), mx.array(cl_d.reshape(-1)), len(cen))
        cmask = np.zeros(n, bool)
        cmask[cen] = True
        cmask[cl_h[cl_h >= 0]] = True
        self.cmask = mx.array(cmask)[None, :, None]
        self.key = mx.random.key(seed)

    def nlist(self, x: mx.array, box: mx.array):
        B, N = x.shape[0], x.shape[1]
        meta = mx.array(np.array([N, B, self.maxnb, self.maxex], np.uint32))
        nbr, nnb, ovf = _nlist_k(inputs=[x.reshape(-1), box.reshape(-1), self.excl, self.nexcl, mx.array(np.array([self.t.cutoff + self.skin], np.float32)), meta],
                                 grid=(B * N, 1, 1), threadgroup=(256, 1, 1),
                                 output_shapes=[(B * N * self.maxnb,), (B * N,), (B * N,)], output_dtypes=[mx.int32, mx.int32, mx.int32])
        return nbr, nnb, ovf

    def forces(self, x, box, nbr, nnb):
        ed, fd = kernels.direct_fused(x, box, nbr, nnb, self.maxnb, self.t)
        er, fr = ff.pme_energy_forces_fused(self.pme, x, box)
        eb, fb = self.bonded.energy_forces(x)
        v = box[:, 0] * box[:, 1] * box[:, 2]
        return ed + er + eb + ff.self_energy(self.t) + self.t.e_disp_v0 * self.t.v0 / v, fd + fr + fb

    def shake(self, xnew, xref):
        B, N = xnew.shape[0], xnew.shape[1]
        cen, cn, ch, cd, C = self.cl
        meta = mx.array(np.array([C, B, N, self.shake_iters], np.uint32))
        (xo,) = _shake_k(inputs=[xnew.reshape(-1), xref.reshape(-1), self.inv_m, cen, cn, ch, cd, meta], grid=(C * B, 1, 1), threadgroup=(256, 1, 1),
                         output_shapes=[(B * N * 3,)], output_dtypes=[mx.float32], init_value=0.0)
        return mx.where(self.cmask, xo.reshape(B, N, 3), xnew)  # atoms in no cluster keep xnew

    def rattle_v(self, x, v):
        """Remove velocity components along the X-H constraints (RATTLE velocity stage)."""
        B, N = x.shape[0], x.shape[1]
        cen, cn, ch, cd, C = self.cl
        meta = mx.array(np.array([C, B, N, self.shake_iters], np.uint32))
        (vo,) = _rattle_k(inputs=[x.reshape(-1), v.reshape(-1), self.inv_m, cen, cn, ch, meta], grid=(C * B, 1, 1), threadgroup=(256, 1, 1),
                          output_shapes=[(B * N * 3,)], output_dtypes=[mx.float32], init_value=0.0)
        return mx.where(self.cmask, vo.reshape(B, N, 3), v)

    def step(self, x, v, f, box, nbr, nnb):
        dt = self.dt
        v = self.rattle_v(x, v + dt * f * self.inv_m[None, :, None])
        x1 = x + 0.5 * dt * v
        self.key, sub = mx.random.split(self.key)
        v = self.a * v + np.sqrt(1.0 - self.a**2) * self.sqrt_kt_m * mx.random.normal(v.shape, key=sub)
        v = self.rattle_v(x1, v)  # thermostat noise along a constraint must not survive the step
        x2u = x1 + 0.5 * dt * v
        x2 = self.shake(x2u, x)
        # carry the post-noise velocity plus the constraint correction. (x2 - x)/dt would be the
        # AVERAGE of the pre- and post-noise velocities, which halves the thermostat noise every
        # step: measured <T> 225 K for a 298 K target before this fix.
        v = self.rattle_v(x2, v + (x2 - x2u) / dt)
        e, f = self.forces(x2, box, nbr, nnb)
        return x2, v, f, e

    def kinetic_temperature(self, v):
        m = 1.0 / self.inv_m
        ke = 0.5 * (m[None, :, None] * v * v).sum(axis=(1, 2))
        dof = 3 * self.t.n - self.n_constraints - 3
        return 2.0 * ke / (dof * KB)
