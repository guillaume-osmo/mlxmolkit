"""`coulomb_multipole_gxtb::get_multipole_matrix` + `tblite_coulomb_multipole::
get_potential`, both recovered, as numpy.

Arrays are stored in the FORTRAN index order -- amat_sd(k, jat, iat) is
a_sd[k, j, i] -- so each contraction below transcribes the wrap_dgemv it
replaces instead of some convention of this file's own.

Tier B: the kernel is written in fma triples and its distances come from
BLAS's overflow-safe `scaled_norm2`; numpy reproduces neither bit for bit.
The target is the MODEL.
"""
import numpy as np
from scipy.special import erf

_QIDX = ((0, 0), (0, 1), (1, 1), (0, 2), (1, 2), (2, 2))   # xx xy yy xz yz zz
_QMUL = np.array([1.0, 2.0, 1.0, 2.0, 2.0, 1.0])


def multipole_amat(coords_bohr, cn, rad_at, mrad_at, dmp, kdmp):
    xyz = np.asarray(coords_bohr, dtype=np.float64)
    nat = xyz.shape[0]
    dl = np.eye(3)

    # the per-atom sqrt(CN) scale; the port's gab builder has no trace of it
    s = np.asarray(rad_at, dtype=np.float64) * (
        np.sqrt(np.asarray(cn, dtype=np.float64) + 1.0e-12) - 1.0e-6) + 1.0

    # everything indexed [j, i], the Fortran pair order: vec = r_i - r_j
    vji = xyz[None, :, :] - xyz[:, None, :]          # [j, i, k]
    vv = np.moveaxis(vji, 2, 0)                      # [k, j, i]
    r1 = np.sqrt(np.sum(vji * vji, axis=-1))
    eye = np.eye(nat, dtype=bool)
    rr = 1.0 / np.where(eye, 1.0, r1)
    rr2 = rr * rr
    rr3, rr5 = rr * rr2, rr * rr2 * rr2
    rr7, rr9 = rr5 * rr2, rr5 * rr2 * rr2

    mrad = np.asarray(mrad_at, dtype=np.float64)
    xarg = r1 - mrad
    dmp3, dmp5, dmpq3, dmpq5 = (
        dmp[k] * 0.5 * (erf(kdmp[k] * xarg) + 1.0) for k in range(4))

    si = s[None, :]        # atom i -- the SECOND (column) index
    sj = s[:, None]        # atom j -- the FIRST  (row) index
    off = (~eye).astype(float)

    w3 = rr3 * dmp3 * sj                              # sj ALONE: asymmetric
    a_sd = vv * (w3 * off)[None]

    w5 = rr5 * dmp5
    w3d = rr3 * dmp5 * sj
    g = sj * w5 * si * 3.0
    d0 = si * w3d
    dd = (dl[:, :, None, None] * d0[None, None]
          - g[None, None] * vv[:, None] * vv[None, :])          # [a, b, j, i]
    a_dd = dd * off[None, None]

    sq = np.stack([vv[a] * vv[b] * (w5 * m if m == 1.0 else w5 * 2.0)
                   for (a, b), m in zip(_QIDX, _QMUL)])          # no si/sj
    a_sq = sq * off[None]

    p5 = rr5 * dmpq3 * sj
    p7 = rr7 * dmpq3 * sj
    dq = np.empty((3, 6, nat, nat))
    for B, (a, b) in enumerate(_QIDX):
        for k in range(3):
            dq[k, B] = _QMUL[B] * (
                p5 * (vv[a] * dl[k, b] + vv[b] * dl[k, a] + vv[k] * dl[a, b])
                - 5.0 * p7 * vv[a] * vv[b] * vv[k])
    a_dq = -dq * off[None, None]                                 # SUBTRACTED

    q5, q7, q9 = rr5 * dmpq5, rr7 * dmpq5, rr9 * dmpq5           # no si/sj
    qq = np.empty((6, 6, nat, nat))
    for A, (a, b) in enumerate(_QIDX):
        for B, (c, e) in enumerate(_QIDX):
            s6 = (dl[a, b] * vv[c] * vv[e] + dl[a, c] * vv[b] * vv[e]
                  + dl[a, e] * vv[b] * vv[c] + dl[b, c] * vv[a] * vv[e]
                  + dl[b, e] * vv[a] * vv[c] + dl[c, e] * vv[a] * vv[b])
            s3 = dl[a, b] * dl[c, e] + dl[a, c] * dl[b, e] + dl[a, e] * dl[b, c]
            qq[A, B] = _QMUL[A] * _QMUL[B] * (
                35.0 * q9 * vv[a] * vv[b] * vv[c] * vv[e]
                - 5.0 * q7 * s6 + 1.5 * q5 * s3)
    a_qq = qq * off[None, None]
    return a_sd, a_dd, a_sq, a_dq, a_qq


def multipole_energy(a_sd, a_dd, a_sq, a_dq, a_qq, qat, dpat, qpat):
    """`tblite_coulomb_multipole::get_energy`, per atom, in Hartree.

        vd  = A_sd.q  +  0.5 * A_dd.dp  +  A_dq.qp
        vq  = A_sq.q  +  (1/6) * A_qq.qp
        E_i = sum_k vd(k,i)*dp(k,i)  +  sum_k vq(k,i)*qp(k,i)

    ⚠️ The ALPHAS ARE NOT THE POTENTIAL'S.  `get_potential` uses 1 on A_dd and
    1/3 on A_qq; the energy uses 1/2 and 1/6 -- the half that makes a quadratic
    form out of a linear response.  Reusing the potential's contraction here
    overshoots the dipole-dipole and quadrupole-quadrupole halves and was worth
    4.7e-4 Ha on H2O.  The `qat` blocks carry NO factor, and `qat` itself never
    appears in the final contraction: the energy is purely dp.vd + qp.vq.

    Index conventions are `multipole_potential`'s, unchanged.
    """
    vd = np.einsum("kji,i->kj", a_sd, qat)
    vd += 0.5 * np.einsum("abji,bi->aj", a_dd, dpat)
    vq = np.einsum("kji,i->kj", a_sq, qat)
    vd += np.einsum("kBji,Bi->kj", a_dq, qpat)
    vq += (1.0 / 6.0) * np.einsum("ABji,Aj->Bi", a_qq, qpat)
    return (vd * dpat).sum(axis=0) + (vq * qpat).sum(axis=0)


def multipole_potential(a_sd, a_dd, a_sq, a_dq, a_qq, qat, dpat, qpat):
    """The eight wrap_dgemv calls of `tblite_coulomb_multipole::get_potential`.

    312: vdp(k,i)   = sum_j  A(k,j,i)   q(j)
    321: vat(i)    += sum_kj A(k,j,i)   dp(k,j)        (trans)
    422: vdp(b,i)  += sum_aj A(a,j,b,i) dp(a,j)
         vqp(b,i)  += sum_aj A(a,j,b,i) dp(a,j)  etc.
    """
    vat = np.zeros(len(qat)); vdp = np.zeros_like(dpat); vqp = np.zeros_like(qpat)
    # 🔑 An untransposed wrap_dgemv writes y over the matrix's ROW index, which
    # for amat(k, jat, iat) is (k, JAT) -- the output atom index is jat and the
    # summed one is iat.  Getting that backwards is invisible on amat_sq and
    # amat_qq, which are EVEN in v and so symmetric in (j, i), and flips the
    # SIGN on amat_sd and amat_dq, which are odd.  That is exactly the shape
    # the residual had: vqp exact, vdp wrong.
    vdp += np.einsum("kji,i->kj", a_sd, qat)
    vat += np.einsum("kji,kj->i", a_sd, dpat)          # trans="T": out over i
    vdp += np.einsum("abji,bi->aj", a_dd, dpat)
    vqp += np.einsum("kji,i->kj", a_sq, qat)
    vat += np.einsum("kji,kj->i", a_sq, qpat)          # trans="T"
    vdp += np.einsum("kBji,Bi->kj", a_dq, qpat)
    vqp += np.einsum("aBji,aj->Bi", a_dq, dpat)        # trans="T"
    vqp += (1.0 / 3.0) * np.einsum("ABji,Aj->Bi", a_qq, qpat)
    return vat, vdp, vqp
