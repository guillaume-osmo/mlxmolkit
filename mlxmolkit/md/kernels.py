"""Fused Metal kernel for the real-space nonbonded term (OpenMM NonbondedForce, PME direct part).

One thread per atom over a padded full (directed) neighbour list: Lorentz-Berthelot LJ with
OpenMM's potential switch, plus erfc(alpha r)/r Coulomb, energy and analytic force accumulated
in registers (no atomics; each pair is visited from both ends, energy halved). Exception
pairs are absent from the list -- their terms live in forcefield.exception_energy.

MSL has no erfc: Numerical Recipes' erfcc (fractional error < 1.2e-7 everywhere) instead of
the Abramowitz-Stegun 7.1.26 polynomial, whose 1.5e-7 ABSOLUTE error summed over ~8e5 pairs
can drift by kJ/mol and fail the 1e-4 parity gate against OpenMM.
"""

import mlx.core as mx
import numpy as np

_SRC = """
    uint gid = thread_position_in_grid.x;
    uint N = meta[0];
    uint B = meta[1];
    uint MAXNB = meta[2];
    if (gid >= N * B) return;
    uint b = gid / N;
    uint li = gid - b * N;
    float3 Lb = float3(box[3 * b], box[3 * b + 1], box[3 * b + 2]);
    float3 invL = 1.0f / Lb;
    float rc = cst[0], rs = cst[1], alpha = cst[2], ke = cst[3];
    float rc2 = rc * rc;
    float3 pi = float3(pos[3 * gid], pos[3 * gid + 1], pos[3 * gid + 2]);
    float qi = q[gid];  // per-box charges (solute charges already scaled by lambda_elec)
    float si = sig[li];
    float ei = eps[li];
    int ai = alch[li];
    float lam = lam_s[b];
    float asc = cst[4];
    float3 f = float3(0.0f);
    float e = 0.0f;
    int n = nnb[gid];
    const device int* nl = nbr + (size_t)gid * MAXNB;
    for (int k = 0; k < n; k++) {
        uint lj = (uint)nl[k];
        uint gj = b * N + lj;
        float3 d = pi - float3(pos[3 * gj], pos[3 * gj + 1], pos[3 * gj + 2]);
        d -= Lb * rint(d * invL);
        float r2 = dot(d, d);
        if (r2 >= rc2) continue;
        float r = sqrt(r2);
        float invr = 1.0f / r;
        // Lorentz-Berthelot LJ with OpenMM's potential switch
        float s = 0.5f * (si + sig[lj]);
        float ep = sqrt(ei * eps[lj]);
        float elj, dlj;
        if (ai != alch[lj]) {
            // solute-solvent: openmmtools soft-core (a=b=1, c=6): U = lam 4 eps x (x-1), x = 1/(alpha (1-lam) + (r/s)^6)
            float rs6 = r2 / (s * s); rs6 = rs6 * rs6 * rs6;
            float x = 1.0f / (asc * (1.0f - lam) + rs6);
            elj = lam * 4.0f * ep * x * (x - 1.0f);
            dlj = -lam * 4.0f * ep * (2.0f * x - 1.0f) * x * x * 6.0f * rs6 * invr;  // dE/dr
        } else {
            float sr2 = s * s / r2;
            float sr6 = sr2 * sr2 * sr2;
            elj = 4.0f * ep * (sr6 * sr6 - sr6);
            dlj = 4.0f * ep * (-12.0f * sr6 * sr6 + 6.0f * sr6) * invr;  // dE/dr
        }
        if (rs > 0.0f && r > rs) {
            float x = (r - rs) / (rc - rs);
            float sw = 1.0f - x * x * x * (10.0f - 15.0f * x + 6.0f * x * x);
            float dsw = -30.0f * x * x * (1.0f - 2.0f * x + x * x) / (rc - rs);
            dlj = dlj * sw + elj * dsw;
            elj = elj * sw;
        }
        // erfc(alpha r)/r Coulomb, Numerical Recipes erfcc
        float z = alpha * r;
        float tt = 1.0f / (1.0f + 0.5f * z);
        float ex = exp(-z * z);
        float erfcv = tt * exp(-z * z - 1.26551223f + tt * (1.00002368f + tt * (0.37409196f + tt * (0.09678418f
                    + tt * (-0.18628806f + tt * (0.27886807f + tt * (-1.13520398f + tt * (1.48851587f
                    + tt * (-0.82215223f + tt * 0.17087277f)))))))));
        float qq = ke * qi * q[gj];
        float ec = qq * erfcv * invr;
        float dc = -qq * (erfcv * invr + 1.1283791671f * alpha * ex) * invr;  // dE/dr
        f += -(dlj + dc) * invr * d;
        e += 0.5f * (elj + ec);
    }
    frc[3 * gid] = f.x;
    frc[3 * gid + 1] = f.y;
    frc[3 * gid + 2] = f.z;
    epot[gid] = e;
"""

_kernel = mx.fast.metal_kernel(
    name="md_direct_nb",
    input_names=["pos", "q", "sig", "eps", "alch", "lam_s", "nbr", "nnb", "box", "cst", "meta"],
    output_names=["frc", "epot"],
    source=_SRC,
)


_BSPLINE5 = """
    // order-5 cardinal B-spline weights (and derivatives) for fractional offset w, OpenMM's recursion
    inline void bspline5(float w, thread float* data, thread float* ddata) {
        data[0] = 1.0f - w; data[1] = w; data[2] = 0.0f; data[3] = 0.0f; data[4] = 0.0f;
        for (int k = 3; k < 5; k++) {
            float div = 1.0f / (k - 1);
            data[k - 1] = div * w * data[k - 2];
            for (int l = 1; l < k - 1; l++)
                data[k - l - 1] = div * ((w + l) * data[k - l - 2] + (k - l - w) * data[k - l - 1]);
            data[0] = div * (1.0f - w) * data[0];
        }
        ddata[0] = -data[0];
        for (int j = 1; j < 5; j++) ddata[j] = data[j - 1] - data[j];
        float div = 0.25f;
        data[4] = div * w * data[3];
        for (int l = 1; l < 4; l++)
            data[4 - l] = div * ((w + l) * data[3 - l] + (5 - l - w) * data[4 - l]);
        data[0] = div * (1.0f - w) * data[0];
    }
"""

_SPREAD_SRC = """
    uint gid = thread_position_in_grid.x;
    uint N = meta[0], B = meta[1];
    if (gid >= N * B) return;
    uint b = gid / N;
    uint li = gid - b * N;
    int K0 = (int)meta[2], K1 = (int)meta[3], K2 = (int)meta[4];
    float3 Lb = float3(box[3 * b], box[3 * b + 1], box[3 * b + 2]);
    float3 p = float3(pos[3 * gid], pos[3 * gid + 1], pos[3 * gid + 2]);
    float3 fr = p / Lb; fr -= floor(fr);
    float3 u = fr * float3(K0, K1, K2);
    int3 bs = int3(floor(u));
    float3 w = u - float3(bs);
    float wx[5], wy[5], wz[5], dx[5], dy[5], dz[5];
    bspline5(w.x, wx, dx); bspline5(w.y, wy, dy); bspline5(w.z, wz, dz);
    float qi = q[gid];
    device atomic_float* G = (device atomic_float*)grid + (size_t)b * K0 * K1 * K2;
    for (int a = 0; a < 5; a++) {
        int ix = (bs.x + a - 4 + K0) % K0;
        for (int c = 0; c < 5; c++) {
            int iy = (bs.y + c - 4 + K1) % K1;
            float wxy = qi * wx[a] * wy[c];
            for (int e = 0; e < 5; e++) {
                int iz = (bs.z + e - 4 + K2) % K2;
                atomic_fetch_add_explicit(G + (ix * K1 + iy) * K2 + iz, wxy * wz[e], memory_order_relaxed);
            }
        }
    }
"""

_GATHER_SRC = """
    uint gid = thread_position_in_grid.x;
    uint N = meta[0], B = meta[1];
    if (gid >= N * B) return;
    uint b = gid / N;
    uint li = gid - b * N;
    int K0 = (int)meta[2], K1 = (int)meta[3], K2 = (int)meta[4];
    float3 Lb = float3(box[3 * b], box[3 * b + 1], box[3 * b + 2]);
    float3 p = float3(pos[3 * gid], pos[3 * gid + 1], pos[3 * gid + 2]);
    float3 fr = p / Lb; fr -= floor(fr);
    float3 u = fr * float3(K0, K1, K2);
    int3 bs = int3(floor(u));
    float3 w = u - float3(bs);
    float wx[5], wy[5], wz[5], dx[5], dy[5], dz[5];
    bspline5(w.x, wx, dx); bspline5(w.y, wy, dy); bspline5(w.z, wz, dz);
    const device float* P = phi + (size_t)b * K0 * K1 * K2;
    float3 g = float3(0.0f);
    for (int a = 0; a < 5; a++) {
        int ix = (bs.x + a - 4 + K0) % K0;
        for (int c = 0; c < 5; c++) {
            int iy = (bs.y + c - 4 + K1) % K1;
            for (int e = 0; e < 5; e++) {
                int iz = (bs.z + e - 4 + K2) % K2;
                float v = P[(ix * K1 + iy) * K2 + iz];
                g.x += v * dx[a] * wy[c] * wz[e];
                g.y += v * wx[a] * dy[c] * wz[e];
                g.z += v * wx[a] * wy[c] * dz[e];
            }
        }
    }
    float pref = pre[b] * q[gid];
    frc[3 * gid] = -pref * g.x * K0 / Lb.x;
    frc[3 * gid + 1] = -pref * g.y * K1 / Lb.y;
    frc[3 * gid + 2] = -pref * g.z * K2 / Lb.z;
"""

_spread = mx.fast.metal_kernel(name="md_pme_spread", input_names=["pos", "q", "box", "meta"], output_names=["grid"],
                               source=_SPREAD_SRC, header=_BSPLINE5, atomic_outputs=True)
_gather = mx.fast.metal_kernel(name="md_pme_gather", input_names=["pos", "q", "box", "phi", "pre", "meta"], output_names=["frc"],
                               source=_GATHER_SRC, header=_BSPLINE5)


def pme_spread(x: mx.array, box_len: mx.array, q: mx.array, grid: tuple) -> mx.array:
    """q: [N] (same charges in every box) or [B, N] (per-box, e.g. lambda-scaled)."""
    B, N = x.shape[0], x.shape[1]
    q = mx.broadcast_to(q, (B, N)) if q.ndim == 1 else q
    meta = mx.array(np.array([N, B, *grid], np.uint32))
    (Q,) = _spread(inputs=[x.reshape(-1), q.reshape(-1), box_len.reshape(-1), meta], grid=(B * N, 1, 1), threadgroup=(256, 1, 1),
                   output_shapes=[(B * grid[0] * grid[1] * grid[2],)], output_dtypes=[mx.float32], init_value=0.0)
    return Q.reshape(B, *grid)


def pme_gather(x: mx.array, box_len: mx.array, q: mx.array, phi: mx.array, pref: mx.array, grid: tuple) -> mx.array:
    B, N = x.shape[0], x.shape[1]
    q = mx.broadcast_to(q, (B, N)) if q.ndim == 1 else q
    meta = mx.array(np.array([N, B, *grid], np.uint32))
    (F,) = _gather(inputs=[x.reshape(-1), q.reshape(-1), box_len.reshape(-1), phi.reshape(-1), pref, meta], grid=(B * N, 1, 1),
                   threadgroup=(256, 1, 1), output_shapes=[(B * N * 3,)], output_dtypes=[mx.float32])
    return F.reshape(B, N, 3)


_TERM_HEAD = """
    uint tid = thread_position_in_grid.x;
    uint N = meta[0], B = meta[1], T = meta[2];
    if (tid >= T * B) return;
    uint b = tid / T;
    uint m = tid - b * T;
    uint off = b * N;
    device atomic_float* F = (device atomic_float*)frc;
"""

_ADD = """
    inline void addf(device atomic_float* F, uint a, float3 v) {
        atomic_fetch_add_explicit(F + 3 * a, v.x, memory_order_relaxed);
        atomic_fetch_add_explicit(F + 3 * a + 1, v.y, memory_order_relaxed);
        atomic_fetch_add_explicit(F + 3 * a + 2, v.z, memory_order_relaxed);
    }
    inline float3 P(const device float* pos, uint a) { return float3(pos[3 * a], pos[3 * a + 1], pos[3 * a + 2]); }
    inline float erfc_nr(float z) {
        float tt = 1.0f / (1.0f + 0.5f * z);
        return tt * exp(-z * z - 1.26551223f + tt * (1.00002368f + tt * (0.37409196f + tt * (0.09678418f
               + tt * (-0.18628806f + tt * (0.27886807f + tt * (-1.13520398f + tt * (1.48851587f
               + tt * (-0.82215223f + tt * 0.17087277f)))))))));
    }
"""

_BOND_SRC = _TERM_HEAD + """
    uint i = off + idx[2 * m], j = off + idx[2 * m + 1];
    float3 d = P(pos, j) - P(pos, i);
    float r = length(d);
    float dr = r - prm[2 * m];
    float k = prm[2 * m + 1];
    float3 g = (k * dr / r) * d;  // dE/dr_j
    addf(F, i, g);
    addf(F, j, -g);
    atomic_store_explicit((device atomic_float*)ene + tid, 0.5f * k * dr * dr, memory_order_relaxed);
"""

_ANGLE_SRC = _TERM_HEAD + """
    uint i = off + idx[3 * m], j = off + idx[3 * m + 1], k = off + idx[3 * m + 2];
    float3 u = P(pos, i) - P(pos, j), v = P(pos, k) - P(pos, j);
    float lu = length(u), lv = length(v);
    float3 uh = u / lu, vh = v / lv;
    float c = dot(uh, vh);
    float3 pu = vh - c * uh, pv = uh - c * vh;
    float s = length(pu);
    float theta = atan2(s, c);
    float dE = prm[2 * m + 1] * (theta - prm[2 * m]);
    float3 fi = (dE / (lu * s)) * pu;  // -dE/dtheta * dtheta/du, dtheta/du = -pu / (|u| sin)
    float3 fk = (dE / (lv * s)) * pv;
    addf(F, i, fi);
    addf(F, k, fk);
    addf(F, j, -(fi + fk));
    atomic_store_explicit((device atomic_float*)ene + tid, 0.5f * prm[2 * m + 1] * (theta - prm[2 * m]) * (theta - prm[2 * m]), memory_order_relaxed);
"""

_TORSION_SRC = _TERM_HEAD + """
    uint a0 = off + idx[4 * m], a1 = off + idx[4 * m + 1], a2 = off + idx[4 * m + 2], a3 = off + idx[4 * m + 3];
    float3 b1 = P(pos, a1) - P(pos, a0), b2 = P(pos, a2) - P(pos, a1), b3 = P(pos, a3) - P(pos, a2);
    float3 n1 = cross(b1, b2), n2 = cross(b2, b3);
    float lb2 = length(b2);
    // forcefield.bonded_energy uses phi = atan2((n1 x b2/|b2|).n2, n1.n2) = -phi_std (IUPAC)
    float phi_std = atan2(lb2 * dot(b1, n2), dot(n1, n2));
    float phi = -phi_std;
    float per = prm[3 * m], ph = prm[3 * m + 1], kt = prm[3 * m + 2];
    float dEdphi = -kt * per * sin(per * phi - ph);
    float dEdstd = -dEdphi;  // phi = -phi_std
    float nn1 = dot(n1, n1), nn2 = dot(n2, n2);
    float3 g0 = -(lb2 / nn1) * n1;  // dphi_std / dr0 (Blondel-Karplus)
    float3 g3 = (lb2 / nn2) * n2;   // dphi_std / dr3
    float p12 = dot(b1, b2) / (lb2 * lb2), p32 = dot(b3, b2) / (lb2 * lb2);
    // Blondel & Karplus (1996) with F = r0 - r1 = -b1, G = r1 - r2 = -b2, H = r3 - r2 = b3:
    float3 g1 = -(1.0f + p12) * g0 + p32 * g3;
    float3 g2 = p12 * g0 - (1.0f + p32) * g3;
    addf(F, a0, -dEdstd * g0);
    addf(F, a1, -dEdstd * g1);
    addf(F, a2, -dEdstd * g2);
    addf(F, a3, -dEdstd * g3);
    atomic_store_explicit((device atomic_float*)ene + tid, kt * (1.0f + cos(per * phi - ph)), memory_order_relaxed);
"""

_EXC_SRC = _TERM_HEAD + """
    uint li = idx[2 * m], lj = idx[2 * m + 1];
    uint i = off + li, j = off + lj;
    float3 d = P(pos, j) - P(pos, i);
    float r = length(d);
    float invr = 1.0f / r;
    float qq = prm[3 * m], s = prm[3 * m + 1], ep = prm[3 * m + 2];
    float alpha = cst[0], ke = cst[1];
    if (esol[m]) qq *= lam_e[b] * lam_e[b];  // intra-solute exception: both charges scaled
    float e = ke * qq * invr;
    float dEdr = -ke * qq * invr * invr;
    if (ep != 0.0f) {
        float sr = s * invr;
        float sr6 = sr * sr * sr * sr * sr * sr;
        e += 4.0f * ep * (sr6 * sr6 - sr6);
        dEdr += 4.0f * ep * (-12.0f * sr6 * sr6 + 6.0f * sr6) * invr;
    }
    // PME exclusion correction: remove the reciprocal-space interaction of every exception pair
    float qiqj = ke * q[i] * q[j];  // per-box (scaled) charges
    float z = alpha * r;
    float erfv = 1.0f - erfc_nr(z);
    e += -qiqj * erfv * invr;
    dEdr += -qiqj * (1.1283791671f * alpha * exp(-z * z) * invr - erfv * invr * invr);
    float3 g = (dEdr * invr) * d;  // dE/dr_j
    addf(F, i, g);
    addf(F, j, -g);
    atomic_store_explicit((device atomic_float*)ene + tid, e, memory_order_relaxed);
"""


def _term_kernel(name, src, inputs):
    return mx.fast.metal_kernel(name=name, input_names=inputs, output_names=["frc", "ene"], source=src, header=_ADD, atomic_outputs=True)


_bond_k = _term_kernel("md_bonds", _BOND_SRC, ["pos", "idx", "prm", "meta"])
_angle_k = _term_kernel("md_angles", _ANGLE_SRC, ["pos", "idx", "prm", "meta"])
_tors_k = _term_kernel("md_torsions", _TORSION_SRC, ["pos", "idx", "prm", "meta"])
_exc_k = _term_kernel("md_exceptions", _EXC_SRC, ["pos", "idx", "prm", "q", "cst", "esol", "lam_e", "meta"])


class BondedTerms:
    """Packed per-term parameters for the bonded + exception kernels (shared by every box)."""

    def __init__(self, t):
        self.t = t
        self.bond = (mx.array(np.asarray(t.bond_idx).reshape(-1).astype(np.uint32)), mx.array(np.stack([np.asarray(t.bond_r0), np.asarray(t.bond_k)], 1).reshape(-1).astype(np.float32)), int(t.bond_idx.shape[0]))
        self.angle = (mx.array(np.asarray(t.angle_idx).reshape(-1).astype(np.uint32)), mx.array(np.stack([np.asarray(t.angle_t0), np.asarray(t.angle_k)], 1).reshape(-1).astype(np.float32)), int(t.angle_idx.shape[0]))
        self.tors = (mx.array(np.asarray(t.tors_idx).reshape(-1).astype(np.uint32)), mx.array(np.stack([np.asarray(t.tors_n), np.asarray(t.tors_phase), np.asarray(t.tors_k)], 1).reshape(-1).astype(np.float32)), int(t.tors_idx.shape[0]))
        self.exc = (mx.array(np.asarray(t.exc_idx).reshape(-1).astype(np.uint32)), mx.array(np.stack([np.asarray(t.exc_qq), np.asarray(t.exc_sigma), np.asarray(t.exc_eps)], 1).reshape(-1).astype(np.float32)), int(t.exc_idx.shape[0]))
        self.cst = mx.array(np.array([t.alpha, 138.935456], np.float32))

    def energy_forces(self, x: mx.array, qb: mx.array | None = None, lam_e: mx.array | None = None,
                      exc_solute: mx.array | None = None) -> tuple[mx.array, mx.array]:
        """Bonds + angles + torsions + exceptions (with PME exclusion correction): energy [B], forces [B, N, 3].
        qb: per-box charges [B, N]; exc_solute: per-exception flag scaling its charge product by lam_e^2."""
        B, N = x.shape[0], x.shape[1]
        qb = mx.broadcast_to(self.t.q[None], (B, N)) if qb is None else qb
        lam_e = mx.ones((B,), dtype=mx.float32) if lam_e is None else lam_e
        esol = mx.zeros((self.exc[2],), dtype=mx.int32) if exc_solute is None else exc_solute
        flat = x.reshape(-1)
        F = mx.zeros((B * N * 3,), dtype=mx.float32)
        E = mx.zeros((B,), dtype=mx.float32)
        for kern, (idx, prm, T), extra in ((_bond_k, self.bond, []), (_angle_k, self.angle, []), (_tors_k, self.tors, []),
                                           (_exc_k, self.exc, [qb.reshape(-1), self.cst, esol, lam_e])):
            meta = mx.array(np.array([N, B, T], np.uint32))
            frc, ene = kern(inputs=[flat, idx, prm, *extra, meta], grid=(B * T, 1, 1), threadgroup=(256, 1, 1),
                            output_shapes=[(B * N * 3,), (B * T,)], output_dtypes=[mx.float32, mx.float32], init_value=0.0)
            F = F + frc
            E = E + ene.reshape(B, T).sum(-1)
        return E, F.reshape(B, N, 3)


def build_full_nlist(pairs_per_box: list[np.ndarray], n: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Padded full (directed) neighbour lists [B, N, MAXNB] + counts [B, N] from i<j pair lists."""
    rows_cols = [(np.concatenate([p[:, 0], p[:, 1]]), np.concatenate([p[:, 1], p[:, 0]])) for p in pairs_per_box]
    counts = [np.bincount(r, minlength=n) for r, _ in rows_cols]
    maxnb = int(max(c.max() for c in counts))
    nbr = np.zeros((len(pairs_per_box), n, maxnb), np.int32)
    for b, ((rows, cols), c) in enumerate(zip(rows_cols, counts)):
        order = np.argsort(rows, kind="stable")
        rows, cols = rows[order], cols[order]
        starts = np.concatenate([[0], np.cumsum(c)[:-1]])
        slot = np.arange(rows.size) - np.repeat(starts, c)
        nbr[b, rows, slot] = cols
    return nbr, np.stack(counts).astype(np.int32), maxnb


def direct_fused(x: mx.array, box_len: mx.array, nbr: mx.array, nnb: mx.array, maxnb: int, t,
                 qb: mx.array | None = None, alch: mx.array | None = None, lam_s: mx.array | None = None,
                 softcore_alpha: float = 0.5) -> tuple[mx.array, mx.array]:
    """Real-space energy [B] and forces [B, N, 3] for every box in one launch.

    qb: per-box charges [B, N] (default: the topology charges in every box); alch: per-atom
    region flag [N] (pairs with different flags get the soft-core at that box's lam_s [B]).
    """
    B, N = x.shape[0], x.shape[1]
    qb = mx.broadcast_to(t.q[None], (B, N)) if qb is None else qb
    alch = mx.zeros((N,), dtype=mx.int32) if alch is None else alch
    lam_s = mx.ones((B,), dtype=mx.float32) if lam_s is None else lam_s
    cst = mx.array(np.array([t.cutoff, t.switch, t.alpha, 138.935456, softcore_alpha], np.float32))
    meta = mx.array(np.array([N, B, maxnb], np.uint32))
    frc, epot = _kernel(
        inputs=[x.reshape(-1), qb.reshape(-1), t.sigma, t.eps, alch, lam_s, nbr.reshape(-1), nnb.reshape(-1), box_len.reshape(-1), cst, meta],
        grid=(B * N, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(B * N * 3,), (B * N,)],
        output_dtypes=[mx.float32, mx.float32],
    )
    return epot.reshape(B, N).sum(-1), frc.reshape(B, N, 3)
