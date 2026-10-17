"""OpenMM-equivalent force field in MLX, batched over independent boxes.

Ports exactly the functional forms OpenMM uses for an OpenFF system exported with
`combine_nonbonded_forces=True` (HarmonicBondForce, HarmonicAngleForce,
PeriodicTorsionForce, NonbondedForce with PME + switched LJ + exceptions + dispersion
correction), so a system can be validated term by term against OpenMM's Reference
platform before it is integrated.

Layout: positions are [B, N, 3] (B independent boxes sharing one topology, e.g. the
lambda windows of one solute). Units are OpenMM's: nm, kJ/mol, elementary charge.
Forces come from mx.grad of the energy, so they are consistent with it by construction.
"""

from dataclasses import dataclass

import mlx.core as mx
import numpy as np

ONE_4PI_EPS0 = 138.935456  # kJ mol^-1 nm e^-2, OpenMM's constant
PME_ORDER = 5  # OpenMM's B-spline order


@dataclass
class Topology:
    """Static parameters of one box (shared by every box of the batch)."""

    n: int
    bond_idx: mx.array
    bond_r0: mx.array
    bond_k: mx.array
    angle_idx: mx.array
    angle_t0: mx.array
    angle_k: mx.array
    tors_idx: mx.array
    tors_n: mx.array
    tors_phase: mx.array
    tors_k: mx.array
    q: mx.array
    sigma: mx.array
    eps: mx.array
    exc_idx: mx.array
    exc_qq: mx.array
    exc_sigma: mx.array
    exc_eps: mx.array
    exc_set: set
    cutoff: float
    switch: float
    alpha: float
    grid: tuple
    e_disp_v0: float  # dispersion-correction energy at the reference volume
    v0: float

    @classmethod
    def from_npz(cls, d) -> "Topology":
        f = lambda k, t=mx.float32: mx.array(np.asarray(d[k]), dtype=t)  # noqa: E731
        box = np.asarray(d["box"])
        exc = np.asarray(d["exc_idx"])
        return cls(
            n=len(d["q"]),
            bond_idx=f("bond_idx", mx.int32), bond_r0=f("bond_r0"), bond_k=f("bond_k"),
            angle_idx=f("angle_idx", mx.int32), angle_t0=f("angle_t0"), angle_k=f("angle_k"),
            tors_idx=f("tors_idx", mx.int32), tors_n=f("tors_n"), tors_phase=f("tors_phase"), tors_k=f("tors_k"),
            q=f("q"), sigma=f("sigma"), eps=f("eps"),
            exc_idx=f("exc_idx", mx.int32), exc_qq=f("exc_qq"), exc_sigma=f("exc_sigma"), exc_eps=f("exc_eps"),
            exc_set={(int(min(a, b)), int(max(a, b))) for a, b in exc},
            cutoff=float(d["cutoff"]), switch=float(d["switch"]),
            alpha=float(d["pme_alpha"]), grid=tuple(int(x) for x in d["pme_grid"]),
            e_disp_v0=float(d["e_disp"]), v0=float(box[0, 0] * box[1, 1] * box[2, 2]),
        )


def neighbor_pairs(pos: np.ndarray, box_len: np.ndarray, cutoff: float, exc_set: set) -> np.ndarray:
    """All i<j pairs within cutoff (minimum image), excluding exception pairs. CPU, for validation."""
    n = len(pos)
    out = []
    for i in range(n - 1):
        d = pos[i + 1 :] - pos[i]
        d -= box_len * np.round(d / box_len)
        r2 = (d * d).sum(1)
        js = np.nonzero(r2 < cutoff * cutoff)[0] + i + 1
        out.extend((i, int(j)) for j in js if (i, int(j)) not in exc_set)
    return np.array(out, dtype=np.int32)


def _min_image(d: mx.array, box_len: mx.array) -> mx.array:
    return d - box_len * mx.round(d / box_len)


def _erfc(x: mx.array) -> mx.array:
    return 1.0 - mx.erf(x)


def bonded_energy(x: mx.array, t: Topology) -> mx.array:
    """[B] harmonic bonds + harmonic angles + periodic torsions."""
    b = x[:, t.bond_idx[:, 1]] - x[:, t.bond_idx[:, 0]]
    r = mx.sqrt((b * b).sum(-1))
    e = (0.5 * t.bond_k * (r - t.bond_r0) ** 2).sum(-1)

    a0, a1, a2 = (x[:, t.angle_idx[:, k]] for k in range(3))
    u, v = a0 - a1, a2 - a1
    cross = mx.sqrt((mx.linalg.cross(u, v) ** 2).sum(-1))
    theta = mx.arctan2(cross, (u * v).sum(-1))
    e = e + (0.5 * t.angle_k * (theta - t.angle_t0) ** 2).sum(-1)

    p0, p1, p2, p3 = (x[:, t.tors_idx[:, k]] for k in range(4))
    b1, b2, b3 = p1 - p0, p2 - p1, p3 - p2
    n1, n2 = mx.linalg.cross(b1, b2), mx.linalg.cross(b2, b3)
    m1 = mx.linalg.cross(n1, b2 / mx.sqrt((b2 * b2).sum(-1, keepdims=True)))
    phi = mx.arctan2((m1 * n2).sum(-1), (n1 * n2).sum(-1))
    # OpenMM's PeriodicTorsionForce: k (1 + cos(n phi - phase)); phi sign as in OpenMM (IUPAC)
    e = e + (t.tors_k * (1.0 + mx.cos(t.tors_n * phi - t.tors_phase))).sum(-1)
    return e


def direct_energy(x: mx.array, box_len: mx.array, pairs: mx.array, t: Topology) -> mx.array:
    """[B] real-space LJ (switched) + erfc Coulomb over the pair list, cutoff applied."""
    i, j = pairs[..., 0], pairs[..., 1]
    xi = mx.take_along_axis(x, i[..., None], axis=1) if pairs.ndim == 3 else x[:, i]
    xj = mx.take_along_axis(x, j[..., None], axis=1) if pairs.ndim == 3 else x[:, j]
    d = _min_image(xj - xi, box_len[:, None, :])
    r2 = (d * d).sum(-1)
    r = mx.sqrt(r2)
    inside = r < t.cutoff
    sig = 0.5 * (t.sigma[i] + t.sigma[j])
    eps = mx.sqrt(t.eps[i] * t.eps[j])
    sr6 = (sig * sig / r2) ** 3
    lj = 4.0 * eps * (sr6 * sr6 - sr6)
    if t.switch > 0:
        s = mx.clip((r - t.switch) / (t.cutoff - t.switch), 0.0, 1.0)
        lj = lj * (1.0 - s**3 * (10.0 - 15.0 * s + 6.0 * s * s))
    coul = ONE_4PI_EPS0 * t.q[i] * t.q[j] * _erfc(t.alpha * r) / r
    return mx.where(inside, lj + coul, 0.0).sum(-1)


def exception_energy(x: mx.array, t: Topology) -> mx.array:
    """[B] exceptions (1-4 pairs, full strength, no cutoff) minus the reciprocal-space
    contribution of every exception pair (PME exclusion correction), as OpenMM does."""
    i, j = t.exc_idx[:, 0], t.exc_idx[:, 1]
    d = x[:, j] - x[:, i]  # OpenMM: exceptions are not minimum-imaged by default
    r = mx.sqrt((d * d).sum(-1))
    sr6 = (t.exc_sigma / r) ** 6
    e = ONE_4PI_EPS0 * t.exc_qq / r + 4.0 * t.exc_eps * (sr6 * sr6 - sr6)
    e = mx.where(t.exc_eps != 0, e, ONE_4PI_EPS0 * t.exc_qq / r)
    e = mx.where((t.exc_qq != 0) | (t.exc_eps != 0), e, 0.0)
    corr = -ONE_4PI_EPS0 * t.q[i] * t.q[j] * mx.erf(t.alpha * r) / r
    return (e + corr).sum(-1)


def self_energy(t: Topology) -> float:
    return float(-ONE_4PI_EPS0 * t.alpha / np.sqrt(np.pi) * float((t.q * t.q).sum()))


def _bspline_moduli(k: int, order: int = PME_ORDER) -> np.ndarray:
    """|b(m)|^2 for the Euler exponential spline (Essmann et al. 1995)."""
    # M_n evaluated at integer points 1..n-1
    def m_n(u, n):
        if n == 2:
            return np.where((u >= 0) & (u <= 2), 1.0 - np.abs(u - 1.0), 0.0)
        return (u * m_n(u, n - 1) + (n - u) * m_n(u - 1.0, n - 1)) / (n - 1)

    mk = m_n(np.arange(0, order, dtype=float) + 1.0, order)[: order - 1]
    m = np.arange(k)
    den = np.zeros(k, dtype=complex)
    for kk in range(order - 1):
        den += mk[kk] * np.exp(2j * np.pi * m * kk / k)
    mod = np.abs(den) ** 2
    # OpenMM fixes the zeros of the moduli by interpolation from neighbours
    for idx in np.nonzero(mod < 1e-7)[0]:
        mod[idx] = 0.5 * (mod[(idx - 1) % k] + mod[(idx + 1) % k])
    return mod


def _bspline_weights(w: mx.array, order: int = PME_ORDER) -> mx.array:
    """Cardinal B-spline weights for fractional offsets w in [0,1): returns [..., order]."""
    data = [1.0 - w, w] + [mx.zeros_like(w)] * (order - 2)
    for k in range(3, order + 1):
        div = 1.0 / (k - 1)
        new = [None] * order
        new[k - 1] = div * w * data[k - 2]
        for jj in range(1, k - 1):
            new[k - jj - 1] = div * ((w + jj) * data[k - jj - 2] + (k - jj - w) * data[k - jj - 1])
        new[0] = div * (1.0 - w) * data[0]
        for jj in range(k, order):
            new[jj] = data[jj]
        data = new
    return mx.stack(data, axis=-1)


def _bspline_weights_and_derivs(w: mx.array, order: int = PME_ORDER) -> tuple[mx.array, mx.array]:
    """Order-n B-spline weights and their derivatives d/du (Essmann: M_n' = M_{n-1}(u) - M_{n-1}(u-1))."""
    low = _bspline_weights(w, order - 1)  # [..., order-1]
    zero = mx.zeros_like(w)
    lo = [low[..., k] for k in range(order - 1)]
    dw = [-lo[0]] + [lo[k - 1] - lo[k] for k in range(1, order - 1)] + [lo[order - 2]]
    return _bspline_weights(w, order), mx.stack([d if d is not None else zero for d in dw], axis=-1)


class PME:
    """Smooth PME reciprocal space (orthorhombic boxes), batched; matches OpenMM's formulation."""

    def __init__(self, t: Topology, box_len: np.ndarray):
        self.t = t
        k1, k2, k3 = t.grid
        mods = [_bspline_moduli(k) for k in (k1, k2, k3)]
        self.mod = [mx.array(m, dtype=mx.float32) for m in mods]
        self.mi = [np.where(np.arange(k) <= k // 2, np.arange(k), np.arange(k) - k) for k in (k1, k2)] + [np.arange(k3 // 2 + 1)]

    def energy(self, x: mx.array, box_len: mx.array) -> mx.array:
        t = self.t
        k = mx.array(t.grid, dtype=mx.float32)
        B, N = x.shape[0], x.shape[1]
        frac = x / box_len[:, None, :]
        frac = frac - mx.floor(frac)
        u = frac * k  # [B, N, 3]
        base = mx.stop_gradient(mx.floor(u))  # grid cell: piecewise constant, no gradient
        w = u - base
        wts = [_bspline_weights(w[..., d]) for d in range(3)]  # each [B, N, order]
        # grid indices: OpenMM spreads to base - order + 1 + j ... use the standard (base + j - order + 1) mod K
        offs = mx.arange(PME_ORDER) - (PME_ORDER - 1)
        idx = [((base[..., d].astype(mx.int32)[..., None] + offs + t.grid[d]) % t.grid[d]) for d in range(3)]
        q = t.q[None, :, None, None, None]
        vals = q * wts[0][:, :, :, None, None] * wts[1][:, :, None, :, None] * wts[2][:, :, None, None, :]
        flat = (idx[0][:, :, :, None, None] * (t.grid[1] * t.grid[2]) + idx[1][:, :, None, :, None] * t.grid[2] + idx[2][:, :, None, None, :])
        flat = flat + (mx.arange(B) * (t.grid[0] * t.grid[1] * t.grid[2]))[:, None, None, None, None]
        Q = mx.zeros((B * t.grid[0] * t.grid[1] * t.grid[2],), dtype=mx.float32)
        Q = Q.at[flat.reshape(-1)].add(vals.reshape(-1)).reshape(B, *t.grid)
        F = mx.fft.rfftn(Q, axes=(1, 2, 3))
        # influence function
        m1 = mx.array(self.mi[0], dtype=mx.float32)[:, None, None]
        m2 = mx.array(self.mi[1], dtype=mx.float32)[None, :, None]
        m3 = mx.array(self.mi[2], dtype=mx.float32)[None, None, :]
        # |S(m)|^2 for every box at once; never index the complex array (MLX's GPU scatter,
        # used by the backward of indexing, does not support complex64).
        s2 = mx.real(F) ** 2 + mx.imag(F) ** 2  # [B, K1, K2, K3//2+1]
        L = box_len[:, :, None, None, None]  # [B, 3, 1, 1, 1]
        m2sum = (m1 / L[:, 0]) ** 2 + (m2 / L[:, 1]) ** 2 + (m3 / L[:, 2]) ** 2  # [B, K1, K2, K3h]
        denom = self.mod[0][:, None, None] * self.mod[1][None, :, None] * self.mod[2][None, None, : t.grid[2] // 2 + 1]
        eterm = mx.where(m2sum > 0, mx.exp(-(np.pi**2) * m2sum / (t.alpha**2)) / (mx.maximum(m2sum, 1e-12) * denom), 0.0)
        # rfft half-spectrum: double every kz plane except kz=0 and (even grid) the Nyquist plane
        nyquist = (m3 == t.grid[2] // 2) if t.grid[2] % 2 == 0 else mx.zeros(m3.shape, dtype=mx.bool_)
        mult = mx.where((m3 == 0) | nyquist, 1.0, 2.0)
        vol = box_len[:, 0] * box_len[:, 1] * box_len[:, 2]
        return 0.5 * ONE_4PI_EPS0 / (np.pi * vol) * (mult * eterm * s2).sum(axis=(1, 2, 3))


def pme_energy_forces(pme: PME, x: mx.array, box_len: mx.array) -> tuple[mx.array, mx.array]:
    """Analytic smooth-PME reciprocal energy [B] and forces [B, N, 3] (no autodiff).

    E = (ke / (2 pi V)) sum_m G(m) |F[Q](m)|^2 = (ke Ng / (2 pi V)) sum_k Q(k) phi(k),
    phi = IFFT[G F[Q]]; dE/dr_i = (ke Ng / (pi V)) q_i sum_k dW_i(k)/dr_i phi(k).
    """
    t = pme.t
    K = t.grid
    B, N = x.shape[0], x.shape[1]
    kf = mx.array(K, dtype=mx.float32)
    frac = x / box_len[:, None, :]
    frac = frac - mx.floor(frac)
    u = frac * kf
    base = mx.floor(u)
    w = u - base
    wd = [_bspline_weights_and_derivs(w[..., dd]) for dd in range(3)]  # (weights, derivs) per axis, [B, N, P]
    offs = mx.arange(PME_ORDER) - (PME_ORDER - 1)
    idx = [((base[..., dd].astype(mx.int32)[..., None] + offs + K[dd]) % K[dd]) for dd in range(3)]
    flat = idx[0][:, :, :, None, None] * (K[1] * K[2]) + idx[1][:, :, None, :, None] * K[2] + idx[2][:, :, None, None, :]
    flat = flat + (mx.arange(B) * (K[0] * K[1] * K[2]))[:, None, None, None, None]
    flat = flat.reshape(B, N, -1)
    wx, wy, wz = wd[0][0], wd[1][0], wd[2][0]
    dx, dy, dz = wd[0][1], wd[1][1], wd[2][1]
    W = (wx[:, :, :, None, None] * wy[:, :, None, :, None] * wz[:, :, None, None, :]).reshape(B, N, -1)
    q = t.q[None, :, None]
    Q = mx.zeros((B * K[0] * K[1] * K[2],), dtype=mx.float32).at[flat.reshape(-1)].add((q * W).reshape(-1)).reshape(B, *K)
    F = mx.fft.rfftn(Q, axes=(1, 2, 3))
    m1 = mx.array(pme.mi[0], dtype=mx.float32)[:, None, None]
    m2 = mx.array(pme.mi[1], dtype=mx.float32)[None, :, None]
    m3 = mx.array(pme.mi[2], dtype=mx.float32)[None, None, :]
    Lb = box_len[:, :, None, None, None]
    m2sum = (m1 / Lb[:, 0]) ** 2 + (m2 / Lb[:, 1]) ** 2 + (m3 / Lb[:, 2]) ** 2
    denom = pme.mod[0][:, None, None] * pme.mod[1][None, :, None] * pme.mod[2][None, None, : K[2] // 2 + 1]
    G = mx.where(m2sum > 0, mx.exp(-(np.pi**2) * m2sum / (t.alpha**2)) / (mx.maximum(m2sum, 1e-12) * denom), 0.0)
    nyq = (m3 == K[2] // 2) if K[2] % 2 == 0 else mx.zeros(m3.shape, dtype=mx.bool_)
    mult = mx.where((m3 == 0) | nyq, 1.0, 2.0)
    vol = box_len[:, 0] * box_len[:, 1] * box_len[:, 2]
    energy = 0.5 * ONE_4PI_EPS0 / (np.pi * vol) * (mult * G * (mx.real(F) ** 2 + mx.imag(F) ** 2)).sum(axis=(1, 2, 3))
    phi = mx.fft.irfftn(F * G, s=K, axes=(1, 2, 3))  # [B, K1, K2, K3]
    ph = phi.reshape(-1)[flat.reshape(-1)].reshape(B, N, PME_ORDER, PME_ORDER, PME_ORDER)
    gx = (ph * (dx[:, :, :, None, None] * wy[:, :, None, :, None] * wz[:, :, None, None, :])).sum(axis=(2, 3, 4)) * (K[0] / box_len[:, 0:1])
    gy = (ph * (wx[:, :, :, None, None] * dy[:, :, None, :, None] * wz[:, :, None, None, :])).sum(axis=(2, 3, 4)) * (K[1] / box_len[:, 1:2])
    gz = (ph * (wx[:, :, :, None, None] * wy[:, :, None, :, None] * dz[:, :, None, None, :])).sum(axis=(2, 3, 4)) * (K[2] / box_len[:, 2:3])
    ng = K[0] * K[1] * K[2]
    pref = (ONE_4PI_EPS0 * ng / (np.pi * vol))[:, None]
    forces = -mx.stack([gx, gy, gz], axis=-1) * (pref * t.q[None, :])[..., None]
    return energy, forces


def total_energy(x: mx.array, box_len: mx.array, pairs: mx.array, t: Topology, pme: PME) -> dict:
    v = box_len[:, 0] * box_len[:, 1] * box_len[:, 2]
    return {
        "bonded": bonded_energy(x, t),
        "direct": direct_energy(x, box_len, pairs, t),
        "exceptions": exception_energy(x, t),
        "self": mx.full((x.shape[0],), self_energy(t)),
        "reciprocal": pme.energy(x, box_len),
        "dispersion": t.e_disp_v0 * t.v0 / v,
    }


def pme_energy_forces_fused(pme: "PME", x: mx.array, box_len: mx.array, qb: mx.array | None = None) -> tuple[mx.array, mx.array]:
    """Same as pme_energy_forces, with Metal kernels for the charge spread (float atomics)
    and the force gather; FFTs and the influence function stay in MLX."""
    from mlxmolkit.md import kernels

    t = pme.t
    K = t.grid
    q = t.q if qb is None else qb
    Q = kernels.pme_spread(x, box_len, q, K)
    F = mx.fft.rfftn(Q, axes=(1, 2, 3))
    m1 = mx.array(pme.mi[0], dtype=mx.float32)[:, None, None]
    m2 = mx.array(pme.mi[1], dtype=mx.float32)[None, :, None]
    m3 = mx.array(pme.mi[2], dtype=mx.float32)[None, None, :]
    Lb = box_len[:, :, None, None, None]
    m2sum = (m1 / Lb[:, 0]) ** 2 + (m2 / Lb[:, 1]) ** 2 + (m3 / Lb[:, 2]) ** 2
    denom = pme.mod[0][:, None, None] * pme.mod[1][None, :, None] * pme.mod[2][None, None, : K[2] // 2 + 1]
    G = mx.where(m2sum > 0, mx.exp(-(np.pi**2) * m2sum / (t.alpha**2)) / (mx.maximum(m2sum, 1e-12) * denom), 0.0)
    nyq = (m3 == K[2] // 2) if K[2] % 2 == 0 else mx.zeros(m3.shape, dtype=mx.bool_)
    mult = mx.where((m3 == 0) | nyq, 1.0, 2.0)
    vol = box_len[:, 0] * box_len[:, 1] * box_len[:, 2]
    energy = 0.5 * ONE_4PI_EPS0 / (np.pi * vol) * (mult * G * (mx.real(F) ** 2 + mx.imag(F) ** 2)).sum(axis=(1, 2, 3))
    phi = mx.fft.irfftn(F * G, s=K, axes=(1, 2, 3))
    pref = ONE_4PI_EPS0 * (K[0] * K[1] * K[2]) / (np.pi * vol)
    return energy, kernels.pme_gather(x, box_len, q, phi, pref.astype(mx.float32), K)
