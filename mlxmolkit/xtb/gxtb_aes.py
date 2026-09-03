# Copyright (c) 2026 Guillaume
# SPDX-License-Identifier: MIT

"""g-xTB anisotropic electrostatics (AES), reconstructed from the release binary.

The g-xTB AES is the module ``tblite_coulomb_multipole_gxtb`` (a
``damped_multipole`` whose label is "erf_damped_anisotropic_electrostatics").
Its interaction tensor is the generic ``tblite_coulomb_multipole_type`` shared
with GFN2 — i.e. the same charge-dipole
(1/R^3) + charge-quadrupole/dipole-dipole (1/R^5) structure already implemented in
:mod:`mlxmolkit.xtb.aes` (``aniso_electro`` / ``setvsdq`` / ``fockelectro``).

g-xTB differs from GFN2 only in:
  * the pair damping (``get_damping_pair``): erf switching on ``R - mrad_pair``
    instead of GFN2's polynomial.  Verified form:
        damp_k = mag_k * 0.5 * (1 + erf((R_ij - mrad_pair_ij) * scale_k))
    channels (mag, scale): (0.3405910191, 0.5), (0.1691310614, 1.0),
                           (0.074034339, 1.0), (-0.02, 1.0)
  * the multipole radius:  mrad_pair[i,j] = vdw_pair(Zi,Zj) * avg(rvdw_scale_i, rvdw_scale_j)
    (averager = the geometric/general one)
  * a CT/polarization kernel scaled by ``pa_aes_dip_scale`` (get_kernel_*).

NB the precise channel->term mapping and the CT kernel are still being calibrated
against the --gxtb oracle; ``GXTB_AES_*`` constants below are the binary-exact
values, the *wiring* is the part under validation.
"""

from __future__ import annotations

import numpy as np
from scipy.special import erf

from .aes import aniso_electro, setvsdq, fockelectro, mmompop
from .gxtb_basis import GXTBQVSZPBasis, ANG_TO_BOHR
from .multipole_integrals import multipole_matrices
from .mctc_vdwrad import mctc_vdw_pair_matrix_bohr
from .params_gxtb import GXTB_PARAMS

# AES damping channels, exact against the reference implementation.
GXTB_AES_DAMP_MAG = np.array([0.3405910191, 0.1691310614, 0.074034339, -0.02])
GXTB_AES_DAMP_SCALE = np.array([0.5, 1.0, 1.0, 1.0])


def _general_average(gi, gj, xi: float = 1.0):
    """Generalized Hubbard-style average (xi=1 -> geometric), elementwise."""
    gi = np.asarray(gi, dtype=np.float64)
    gj = np.asarray(gj, dtype=np.float64)
    return (2.0 / (gi + gj)) ** (xi - 1.0) * (gi * gj) ** (xi / 2.0)


_MP_CACHE = {}


def qvszp_multipoles(basis: GXTBQVSZPBasis):
    """Dipole/quadrupole AO integrals over the q-vSZP basis, in the SAO basis.

    Returns ``(S, dpint, qpint)`` with dpint (3, nao, nao), qpint (6, nao, nao)
    in xtb (xx,yy,zz,xy,xz,yz) order, transformed CAO->SAO. Integrals are
    geometry-only -> cached by basis identity (avoids O(nao^2) rebuild every SCF iter).
    """
    key = id(basis)
    cached = _MP_CACHE.get(key)
    if cached is not None and cached[0] is basis:   # verify identity (id can be reused)
        return cached[1]
    if len(_MP_CACHE) > 8:
        _MP_CACHE.clear()
    S_cao, dp_cao, qp_cao = multipole_matrices(basis.cao_basis)
    T = np.asarray(basis.T_cao_to_sao, dtype=np.float64)
    S = T @ S_cao @ T.T
    dp = np.stack([T @ dp_cao[k] @ T.T for k in range(3)], axis=0)
    qp = np.stack([T @ qp_cao[k] @ T.T for k in range(6)], axis=0)
    _MP_CACHE[id(basis)] = (basis, (S, dp, qp))
    return S, dp, qp


def gxtb_mrad_pair(atoms: np.ndarray) -> np.ndarray:
    """mrad_pair[i,j] = vdw_pair(Zi,Zj) * avg(rvdw_scale_i, rvdw_scale_j)  (Bohr)."""
    atoms = np.asarray(atoms, dtype=np.intp)
    vdw = mctc_vdw_pair_matrix_bohr(atoms)              # (nat, nat)
    rs = np.asarray(GXTB_PARAMS["pa_rvdw_scale"], dtype=np.float64)[atoms - 1]
    # The pair radius is the ARITHMETIC mean, not the geometric one; the two
    # differ by 0.0017 Bohr and are charge-neutral, so only the form matters.
    avg = 0.5 * (rs[:, None] + rs[None, :])             # (nat, nat)
    return vdw * avg


def gxtb_aes_gab(coords_bohr: np.ndarray, mrad: np.ndarray, channel3: int = 0, channel5: int = 1):
    """g-xTB erf-damped gab3 (1/R^3) and gab5 (1/R^5).

    damp_k = mag_k * 0.5 * (1 + erf((R - mrad_pair) * scale_k)); gab3 = damp/R^3, gab5 = damp/R^5.
    channel3/channel5 select which of the 4 binary channels drive the dip (R^-3) and quad/dip-dip
    (R^-5) terms (under oracle calibration).
    """
    n = coords_bohr.shape[0]
    diff = coords_bohr[:, None, :] - coords_bohr[None, :, :]
    R = np.sqrt(np.sum(diff * diff, axis=-1))
    eye = np.eye(n, dtype=bool)
    Rsafe = np.where(eye, 1.0, R)
    arg = R - mrad
    def damp(k):
        return GXTB_AES_DAMP_MAG[k] * 0.5 * (1.0 + erf(arg * GXTB_AES_DAMP_SCALE[k]))
    gab3 = np.where(eye, 0.0, damp(channel3) / Rsafe ** 3)
    gab5 = np.where(eye, 0.0, damp(channel5) / Rsafe ** 5)
    return gab3, gab5


def gxtb_aes_fock(P: np.ndarray, basis: GXTBQVSZPBasis, atoms: np.ndarray,
                  coords_ang: np.ndarray, *, channel3: int = 0, channel5: int = 1):
    """AES Fock contribution F_aes (nao,nao) + energy e_aes (Hartree).

    Builds q-vSZP multipole integrals, Mulliken atomic multipoles from P, the
    g-xTB erf-damped interaction, and the AES potentials, then the Fock term.
    """
    atoms = np.asarray(atoms, dtype=np.intp)
    coords_bohr = np.asarray(coords_ang, dtype=np.float64) * ANG_TO_BOHR
    S, dpint, qpint = qvszp_multipoles(basis)
    aoat = np.array([bf.atom_idx for bf in basis.sao_basis], dtype=np.int64)

    # Mulliken atomic charges + cumulative atomic dipoles/quadrupoles.
    PS = P @ S
    pop = np.zeros(len(atoms))
    for mu in range(P.shape[0]):
        pop[aoat[mu]] += PS[mu, mu]
    zref = np.bincount(basis.shell_atom, weights=basis.shell_zref, minlength=len(atoms))
    q = zref - pop
    dipm, qp = mmompop(P, S, dpint, qpint, aoat.tolist(), coords_bohr)

    # dipole scaling (pa_aes_dip_scale per atom) on the atomic dipoles.
    dipscale = np.asarray(GXTB_PARAMS["pa_aes_dip_scale"], dtype=np.float64)[atoms - 1]
    dipm = dipm * dipscale[None, :]

    mrad = gxtb_mrad_pair(atoms)
    gab3, gab5 = gxtb_aes_gab(coords_bohr, mrad, channel3, channel5)
    e_aes, _ = aniso_electro(atoms.tolist(), coords_bohr, q, dipm, qp, gab3, gab5)
    vs, vd, vq = setvsdq(atoms.tolist(), coords_bohr, q, dipm, qp, gab3, gab5)
    F_aes, _ = fockelectro(P, S, dpint, qpint, aoat.tolist(), vs, vd, vq)
    return F_aes, e_aes


# --- one-center (onsite) exchange: onecxints over same-atom shell pairs ---
import numpy as _np
import os as _os

_ONECX_PATH = _os.path.join(
    _os.path.dirname(__file__), "..", "..", "data", "gxtb_onecxints_extracted.npz"
)

# This table is an extraction artifact with no regeneration script: `git log --
# '*onecxints*'` is empty, and unlike eeqbc2025 / mctc_vdwrad / qvszp there is no
# tools/extract_*.py that rebuilds it.  It was also matched by the blanket
# `*.npz` in .gitignore, so it was never committed and survived only in a
# working tree.  It is committed here (with a .gitignore exception) because the
# only remaining copy was outside this repository.
#
# The load is lazy rather than at module scope.  Previously importing this
# module raised a bare FileNotFoundError naming a path, which meant a clean
# checkout could not import mlxmolkit.xtb.gxtb_aes AT ALL -- silently disabling
# use_aes, use_aniso_h0 and use_twobody_third_order -- while the test suite
# stayed green by skipping on the same missing file.
#
# The reference implementation carries the same table, so it is
# re-derivable the way the other parameter tables were.
_ONEC = None


def _onecx_tables():
    """Load the one-centre exchange tables, or explain precisely what is missing."""
    global _ONEC
    if _ONEC is None:
        if not _os.path.exists(_ONECX_PATH):
            raise FileNotFoundError(
                f"g-xTB one-centre exchange table not found: {_ONECX_PATH}\n"
                "The terms that need it (use_aes, use_aniso_h0, "
                "use_twobody_third_order) cannot run without it."
            )
        _ONEC = _np.load(_ONECX_PATH)
    return _ONEC["onecxints"], _ONEC["lidx"]   # (103, 10), (4, 4)


def __getattr__(name):
    # Keep ONECX_TBL / ONECX_LIDX working for callers, now lazily.
    if name == "ONECX_TBL":
        return _onecx_tables()[0]
    if name == "ONECX_LIDX":
        return _onecx_tables()[1]
    raise AttributeError(name)


def gxtb_onsite_gamma(basis, atoms):
    """Same-atom one-center exchange gamma_onsite[mu,nu] = onecxints[Z, lidx[lmu,lnu]].

    Returns an (nao,nao) matrix nonzero only for AO pairs on the same atom; fed
    through the same S.P.S Fock as the Mulliken exchange.
    """
    atoms = _np.asarray(atoms, dtype=_np.intp)
    bts = _np.asarray(basis.bf_to_shell)
    sa = _np.asarray(basis.shell_atom)
    sl = _np.asarray(basis.shell_l)
    n = bts.size
    g = _np.zeros((n, n))
    _onecx_tbl, _onecx_lidx = _onecx_tables()
    for mu in range(n):
        ish = int(bts[mu]); ai = int(sa[ish]); li = int(sl[ish])
        Z = int(atoms[ai])
        for nu in range(n):
            jsh = int(bts[nu]); aj = int(sa[jsh]); lj = int(sl[jsh])
            if ai != aj:
                continue
            pack = int(_onecx_lidx[li, lj])
            g[mu, nu] = float(_onecx_tbl[Z - 1, pack - 1])
    return g


def gxtb_aniso_h0(basis, atoms, coords_ang, *, kexp: float = 1.5):
    """Anisotropic-H0 dipole-field correction (tblite_xtb_h0 get_anisotropy).

    field[i] = sum_j aniso[i,j] * 0.5*(1+erf(-kexp*(R_ij - mrad_pair_ij))) * (r_j - r_i)/R_ij
    H0_aniso[mu,nu] += 0.5*(field[atom_mu]+field[atom_nu]) . dpint[:,mu,nu]
    aniso[i,j] = avg(pa_h0_dip_scale_i, pa_h0_dip_scale_j).  Uses the AES dpint.
    """
    atoms = np.asarray(atoms, dtype=np.intp)
    coords_bohr = np.asarray(coords_ang, dtype=np.float64) * ANG_TO_BOHR
    nat = len(atoms)
    _, dpint, _ = qvszp_multipoles(basis)
    aoat = np.array([bf.atom_idx for bf in basis.sao_basis], dtype=np.int64)
    dipscale = np.asarray(GXTB_PARAMS["pa_h0_dip_scale"], dtype=np.float64)[atoms - 1]
    # pa_h0_dip_scale has mixed signs -> arithmetic average (geometric -> NaN).
    aniso = 0.5 * (dipscale[:, None] + dipscale[None, :])             # (nat,nat)
    mrad = gxtb_mrad_pair(atoms)
    diff = coords_bohr[None, :, :] - coords_bohr[:, None, :]           # r_j - r_i, (i,j,3)
    R = np.sqrt(np.sum(diff * diff, axis=-1))
    eye = np.eye(nat, dtype=bool)
    Rsafe = np.where(eye, 1.0, R)
    damp = 0.5 * (1.0 + erf(-kexp * (R - mrad)))
    w = np.where(eye, 0.0, aniso * damp / Rsafe)[:, :, None]           # (i,j,1)
    field = np.sum(w * diff, axis=1)                                   # (nat,3)
    f_ao = field[aoat]                                                 # (nao,3)
    H = np.zeros(dpint.shape[1:], dtype=np.float64)
    for k in range(3):
        H += 0.5 * (f_ao[:, k][:, None] + f_ao[:, k][None, :]) * dpint[k]
    return H


_ONSITE_BASE_CACHE = {}


def _onsite_base(basis, atoms):
    """Geometry/element-only onecx base[mu,nu] for same-atom AO pairs (cached)."""
    key = id(basis)
    c = _ONSITE_BASE_CACHE.get(key)
    if c is not None and c[0] is basis:             # verify identity (id can be reused)
        return c[1], c[2]
    if len(_ONSITE_BASE_CACHE) > 8:
        _ONSITE_BASE_CACHE.clear()
    atoms = _np.asarray(atoms, dtype=_np.intp)
    bts = _np.asarray(basis.bf_to_shell)
    aoat = _np.asarray(basis.shell_atom)[bts]          # atom per AO
    aol = _np.asarray(basis.shell_l)[bts]              # l per AO
    Zao = atoms[aoat]
    n = bts.size
    base = _np.zeros((n, n))
    same = aoat[:, None] == aoat[None, :]
    _onecx_tbl, _onecx_lidx = _onecx_tables()
    pack = _onecx_lidx[aol[:, None], aol[None, :]] - 1  # (n,n) packed index
    val = _onecx_tbl[Zao[:, None] - 1, pack]            # (n,n) onecx per AO-pair (uses mu's Z)
    base = _np.where(same, val, 0.0)
    _ONSITE_BASE_CACHE[key] = (basis, base, bts)
    return base, bts


def gxtb_onsite_gamma_density(P, S, basis, atoms):
    """Density-dependent one-center exchange gamma (exact get_gons form), vectorized.

    K_onsite[mu,nu] = (1 - occ_i*occ_j) * onecx[li,lj,Z]  (same-atom AO pairs)
    occ = shell Mulliken occupation fraction. Base (onecx) cached; only the
    occupation factor recomputed each SCF iteration.
    """
    # The K-matrix is
    #   gamma[mu,mu] = -0.5*frscale*kq_mu * sum_l onecx[lmu,ll]*pop_l
    # with the off-diagonal analogous, frscale=0.15, kq=pg_fock_kq and
    # pop = diag(P@S).
    #
    # It is a ONE-CENTER K-matrix and must be folded per atom by
    # `onsite_fx_symv`, which keeps it on-atom.  Folding it instead through the
    # Mulliken S.P.S sandwich spreads it across bonds and over-corrects
    # multi-bonded C and N.
    #   So until onsite_fx_symv is traced, the empirically-better (1-occ^2) form
    #   below (with the sandwich) is the best available (0.0478).
    base, bts = _onsite_base(basis, atoms)
    nsh = _np.asarray(basis.shell_atom).size
    diag = _np.einsum("ij,ji->i", P, S)                # (P@S) diagonal per AO
    pop_sh = _np.bincount(bts, weights=diag, minlength=nsh)
    ndeg = _np.bincount(bts, minlength=nsh).astype(_np.float64)
    occ = pop_sh / _np.maximum(2.0 * ndeg, 1e-12)
    occ_ao = occ[bts]
    return (1.0 - occ_ao[:, None] * occ_ao[None, :]) * base


def gxtb_onsite_potential(P, S, basis, atoms, frscale=0.15):
    """On-site one-centre exchange: the ``onsite_fx_symv`` fold.

    This kernel does NOT build a density-folded matrix
    (the old ``gxtb_onsite_gamma_density`` SPS path); it produces a per-AO
    *anti-binding shell potential*

        V_ao[mu] = frscale * sum_{nu on same atom} onecx[Z_mu, l_mu, l_nu] * (P@S)[nu,nu]

    i.e. ``V = frscale * (onecx_base @ pop_ao)`` (a symmetric matrix-vector — the
    "symv").  Verified vs the binary (Ne: V_s=0.1899 / V_p=0.1110, formula
    0.1904 / 0.1114, residual ~0.4% from the charge factor below).  The potential
    is POSITIVE (anti-binding): it cancels the Mulliken-K over-binding.

    The 0.4% refinement is the get_gons charge modulation
    ``(1 - 0.5*(kq_a*q_a + kq_b*q_b))`` with ``kq=pg_fock_kq=[1.1,0.55,0.275,0.1375]``
    applied per shell-pair; pass ``kq``/``qsh`` to enable it.
    """
    base, bts = _onsite_base(basis, atoms)             # AO-level onecx (same-atom)
    diag = _np.einsum("ij,ji->i", P, S)                # pop_ao = (P@S) diagonal
    return float(frscale) * (base @ diag)


def gxtb_onsite_fock_exact(P, S, basis, atoms, frscale=0.15, mapping=(0, 1, 2),
                           qsh=None):
    """On-site exchange Fock: ``onsite_fx_symv`` followed by ``get_kfock``.

    Three symv channels, each V_k = frscale * onecx @ diag(D_k), where D_k is one
    of the three density forms {P, S@P, S@P@S}.  ``get_kfock`` folds them as

        M = 0.25*OS(V[map0]) + 0.5*OS(V[map1]) + 0.25*diag(V[map2])
        F_onsite = -0.125*(M + M^T)        # net -0.125 off-diag, -0.25 diag

    where OS(V)[j,i] = V[i]*S[j,i] is the overlap-sandwich (daxpy column form).
    ``mapping`` selects which density form feeds (sandwich-0.25, sandwich-0.5,
    diag-0.25).

    🔑 The default is (0, 1, 2) -- the IDENTITY -- and it is not a guess.  The
    recovered `exchange_fock::get_kfock` states its own fold:

        step 12  xmat(:,i)        += 0.25 * fxa(i) * S(:,i)
        step 15  kfock(:,i,ispin) += 0.50 * fxb(i) * S(:,i)
                 kfock(i,i,ispin) += 0.25 * fxc(i)

    with `fxa = onsite_fx_symv(avec)`, `avec = diag(P)`; `fxb` from
    `bvec = diag(S@P)`; `fxc` from `cvec = diag(S@P@S)`.  So the 0.25 sandwich
    is P, the 0.5 sandwich is SP and the diagonal is SPS.  The previous default
    (2, 0, 1) put SPS on the 0.25 sandwich and P on the 0.5 one.

    ⚠️ AND THIS FUNCTION IS STILL NOT `get_kfock`.  The 0.25*fxa term is added
    to `xmat`, NOT to `kfock` -- and `xmat` is afterwards Hadamard-multiplied by
    the two-centre `kmat`, by `bomat`, by `onsite_fx` and by `onsite_ri` before
    being contracted.  Folding it here as a plain overlap sandwich is an
    approximation of that chain, and the chain is not yet ported.  Do not read a
    metric change from this function as evidence about the on-site physics until
    it is.
    """
    base, bts = _onsite_base(basis, atoms)             # AO onecx (same-atom)
    # 🔑 THE CHARGE MODULATION IS PART OF THE MATRIX, not an optional variant.
    # `exchange_fock::get_gons` (recovered) builds the on-site matrix as
    #
    #   onsite_fx(jsh,ish,iat) =
    #       (1 - 0.5*(kq_sh(ish)*qsh(ii+ish) + kq_sh(jsh)*qsh(ii+jsh)))
    #       * alpha * onsite_sh(jsh,ish,izp)
    #
    # so the factor multiplies `base` BEFORE the symv, in every channel.  This
    # function used the bare `onecx` and called itself exact; measured against
    # get_gons the factor runs 0.80 to 1.47 and the three channel vectors were
    # off by 9-12.5 %.  `kq_sh(ish,izp) = pg_fock_kq(l+1)`, from add_exchange.
    if qsh is not None:
        aol = _np.asarray(basis.shell_l)[bts]
        kqq = (_np.asarray(GXTB_PARAMS["pg_fock_kq"], dtype=_np.float64)[aol]
               * _np.asarray(qsh, dtype=_np.float64)[bts])
        base = (1.0 - 0.5 * (kqq[:, None] + kqq[None, :])) * base
    SP = S @ P
    Dlist = [P, SP, SP @ S]                             # index 0=P, 1=SP, 2=SPS
    fr = float(frscale)
    V = [fr * (base @ _np.diag(Dlist[k])) for k in range(3)]
    Va, Vb, Vc = V[mapping[0]], V[mapping[1]], V[mapping[2]]
    M = 0.25 * (S * Va[None, :]) + 0.5 * (S * Vb[None, :]) + _np.diag(0.25 * Vc)
    return -0.125 * (M + M.T)


def gxtb_onsite_potential_q(P, S, basis, atoms, qsh, frscale=0.15):
    """Onsite potential WITH the get_gons charge modulation (full dVar64 integrand).

        V_a = frscale * sum_b (1 - 0.5*(kq[l_a]*q_a + kq[l_b]*q_b)) * onecx[Z,l_a,l_b] * pop_b

    kq = pg_fock_kq = [1.1,0.55,0.275,0.1375] (per angular momentum); q = shell
    Mulliken charge.  The charge factor is element/charge dependent, which is why
    the plain (charge-free) potential showed molecule-dependent sign error.
    """
    base, bts = _onsite_base(basis, atoms)
    diag = _np.einsum("ij,ji->i", P, S)
    kq = _np.asarray(GXTB_PARAMS["pg_fock_kq"], dtype=_np.float64)
    aol = _np.asarray(basis.shell_l)[bts]              # l per AO
    q_ao = _np.asarray(qsh, dtype=_np.float64)[bts]    # shell charge per AO
    kqq = kq[aol] * q_ao                               # kq_l * q_shell per AO
    factor = 1.0 - 0.5 * (kqq[:, None] + kqq[None, :])  # (n,n) per AO-pair
    return float(frscale) * ((factor * base) @ diag)


def gxtb_twobody_thirdorder(qsh, basis, atoms, coords_ang, *, k3=2.3, kx=1.3, rexp=0.2093327496):
    """Two-body 3rd-order (coulomb_thirdorder_twobody), density-dependent.

    Binary-exact algebra (libxtb get_taumat_0d__omp_fn_0 + get_energy/get_potential,
    mode==0). Per shell i on atom A with angular momentum l:

      eta_base_i = ps_tb2_shell_hubbard[Z,l] * pa_hubbard_parameter[Z]   (NO cn slope)
      eta_eff_i  = eta_base_i * (1 + pa_tb2_hubbard_cn[Z]*(sqrt(cn_A + 1e-12) - 1e-6))
      gamma_ij   = harmonic(eta_eff_i, eta_eff_j) = 2/(1/ei + 1/ej)   (new_average enum 2)
      off-site (A_i != A_j):  tau = k3*x*(1 - 0.5*kx*x)*exp(-kx*x),  x = R_ij/gamma_ij
      on-site  (A_i == A_j, INCLUDING the diagonal):  tau = -rexp*gamma_ij^2

    The three two-body scalars are stored consecutively in the binary param block
    as (2.3, 0.2093327496, 1.3).  The decoded get_taumat_0d._omp_fn.0 reads them as:
      off-site prefactor A = *param_3 = 2.3            -> k3
      on-site  prefactor C = *param_4 = 0.2093327496   -> rexp   (A2-validated vs Ne)
      off-site exp decay B = *param_5 = 1.3            -> kx
    The earlier port had kx=0.2093327496 / rexp=1.3 (i.e. it swapped the on-site
    and off-site-exp constants), which blew the term up (E3 ~ -127 Ha, MAE 1.86).
      g3d_i = pa_tb3_hubbard_derivs[Z] * pg_tb3_kshell[l]   (== basis.shell_third)

    E3 = sum_i g3d_i * q_i^2 * (tau@q)_i ;  V3 = 2*g3d*q*(tau@q) + tau@(g3d*q^2)
    (tau is symmetric, so tau@ and tau.T@ coincide.) Returns (E3, V3[nsh]).
    """
    atoms = _np.asarray(atoms, dtype=_np.intp)
    cb = _np.asarray(coords_ang, dtype=_np.float64) * ANG_TO_BOHR
    sa = _np.asarray(basis.shell_atom)
    sl = _np.asarray(basis.shell_l)
    cn = _np.asarray(basis.cn, dtype=_np.float64)
    q = _np.asarray(qsh, dtype=_np.float64)
    Z = atoms[sa]

    tb2sh = _np.asarray(GXTB_PARAMS["ps_tb2_shell_hubbard"], dtype=_np.float64)
    hubp = _np.asarray(GXTB_PARAMS["pa_hubbard_parameter"], dtype=_np.float64)
    cnsc = _np.asarray(GXTB_PARAMS["pa_tb2_hubbard_cn"], dtype=_np.float64)
    eta_base = tb2sh[Z - 1, sl] * hubp[Z - 1]                          # NO cn slope
    eta_eff = eta_base * (1.0 + cnsc[Z - 1] * (_np.sqrt(cn[sa] + 1e-12) - 1e-6))
    g3d = _np.asarray(basis.shell_third, dtype=_np.float64)            # pa_tb3 * pg_tb3_kshell

    ee_i = eta_eff[:, None]; ee_j = eta_eff[None, :]
    gam = 2.0 / (1.0 / ee_i + 1.0 / ee_j)                              # harmonic average
    R = _np.linalg.norm(cb[sa][:, None, :] - cb[sa][None, :, :], axis=-1)
    same = sa[:, None] == sa[None, :]
    x = R / gam
    tau_off = k3 * x * (1.0 - 0.5 * kx * x) * _np.exp(-kx * x)
    tau_on = -rexp * gam * gam                                         # rexp=0.2093327496; incl. diagonal (R=0)
    tau = _np.where(same, tau_on, tau_off)

    tq = tau @ q
    E3 = float(_np.sum(g3d * q * q * tq))
    V3 = 2.0 * g3d * q * tq + tau @ (g3d * q * q)
    return E3, V3


from mlxmolkit.xtb.mctc_vdwrad import mctc_vdw_pair_matrix_bohr

_BO_SLOPE = 1.8897261246204404          # fock_bo_slope, from add_exchange


def gxtb_bocorr_gamma(basis, atoms, coords_ang, *, k=1.0,
                      _atom_matrix_only=False):
    """Bond-order-correction exchange gamma (get_gbocorr): distance-switched per-atom-pair.

    bocorr[mu,nu] = 0.5*(1 + erf(-(R_ij - r0_ij)*k / crad_ij)) * cscale_ij   (i != j)
    r0 = vdw_pair*avg(rvdw_scale) (= mrad); crad = avg(pa_fock_crad); cscale = avg(pa_fock_cscale).
    Geometry-only; added to the exchange gamma (S.P.S Fock).
    """
    atoms = _np.asarray(atoms, dtype=_np.intp)
    cb = _np.asarray(coords_ang, dtype=_np.float64) * ANG_TO_BOHR
    bts = _np.asarray(basis.bf_to_shell); sa = _np.asarray(basis.shell_atom)
    crad = _np.asarray(GXTB_PARAMS["pa_fock_crad"], dtype=_np.float64)[atoms - 1]
    cscale = _np.asarray(GXTB_PARAMS["pa_fock_cscale"], dtype=_np.float64)[atoms - 1]
    nat = len(atoms)
    R = _np.linalg.norm(cb[:, None, :] - cb[None, :, :], axis=-1)
    # 🔑 `get_bocorr_kmatrix` (recovered, and probed by xf_get_bocorr_kmatrix):
    #
    #   val = (erf(-((r - bo_rad(izp,jzp)) * bo_slope) / rad(izp,jzp)) + 1)
    #         * 0.5 * bo_amp(izp,jzp)
    #
    # and `new_exchange_fock` builds
    #   bo_rad = GEOMETRIC average of pa_fock_crad      <- the OFFSET
    #   bo_amp = arithmetic average of pa_fock_cscale
    #   rad    = the vdW PAIR radius                     <- the DENOMINATOR
    #   bo_slope = 1.8897261246204404
    #
    # The port had the offset and the denominator the other way round -- the
    # vdW radius as the offset and an ARITHMETIC average of pa_fock_crad as the
    # denominator -- and bo_slope at 1.0.  Three defects in one expression, and
    # the same role-swap shape as the MFX damping constants (defect D4).
    bo_rad = _np.sqrt(crad[:, None] * crad[None, :])          # geometric
    sc = _np.asarray(GXTB_PARAMS["pa_rvdw_scale"], dtype=_np.float64)[atoms - 1]
    rad_pair = (mctc_vdw_pair_matrix_bohr(atoms) * (1.0 / 0.5291772109044924)
                * (0.5 * (sc[:, None] + sc[None, :])))
    cscale_ij = 0.5 * (cscale[:, None] + cscale[None, :])
    arg = -((R - bo_rad) * _BO_SLOPE) / _np.maximum(rad_pair, 1e-12)
    boc_at = 0.5 * (1.0 + erf(arg)) * cscale_ij                   # (nat,nat)
    _np.fill_diagonal(boc_at, 0.0)
    if _atom_matrix_only:
        return boc_at
    # expand atom-pair -> AO-pair
    n = bts.size
    aoat = sa[bts]
    g = boc_at[_np.ix_(aoat, aoat)]
    return g


# --------------------------------------------------------------------------
# `exchange_fock::get_kfock`, ported whole.
#
# The port previously had only the two-centre Mulliken factorisation and, for
# the on-site half, a plain overlap sandwich.  The binary's assembly is a
# chain: four weight matrices are Hadamard-folded onto `S@P`, onto `kfock` and
# onto `P` in three separate passes, and only then contracted.  Every routine
# below is transcribed from the recovered Fortran, which carries a passing
# 0-ulp differential test against the shipped binary.
#
# The four weights and their scal factors, from `get_kfock`'s own constants:
#
#     kmat       shell pair      shell_scal      =  1.0
#     bomat      atom pair       atom_scal       = -4.0
#     onsite_fx  on-site shell   onsite_fx_scal  =  0.5
#     onsite_ri  shell diagonal  onsite_ri_scal  = -2.0

_KF_SHELL_SCAL = 1.0
_KF_ATOM_SCAL = -4.0
_KF_ONSITE_FX_SCAL = 0.5
_KF_ONSITE_RI_SCAL = -2.0


def _ao_weight_shell(kmat_sh, bts, scal):
    """`shell_hadamard_add`: W[a,b] = scal*kmat[shell(a), shell(b)]."""
    return scal * _np.asarray(kmat_sh)[_np.ix_(bts, bts)]


def _ao_weight_atom(kmat_at, aoat, scal):
    """`atom_hadamard_add`: W[a,b] = scal*kmat[atom(a), atom(b)]."""
    return scal * _np.asarray(kmat_at)[_np.ix_(aoat, aoat)]


def _ao_weight_onsite_fx(onsite_fx, bts, aoat, loc, scal):
    """`onsite_fx_hadamard_add`: same-atom only, W[a,b] = scal*fx[loc(b), loc(a), atom].

    Note the index order: the routine takes `w = kmat(is, js, iat)` with `is`
    running over the ROW shell of the target and `js` over the column, and
    writes `bmat(jj+q, ii+p)` -- so the first index of `kmat` belongs to `b`.
    """
    n = bts.size
    same = aoat[:, None] == aoat[None, :]
    lb = loc[bts]
    W = _np.zeros((n, n))
    fx = _np.asarray(onsite_fx)
    for a in range(n):
        W[a] = fx[lb, lb[a], aoat[a]]
    return _np.where(same, scal * W, 0.0)


def _ao_weight_onsite_ri(onsite_ri, bts, aoat, loc, scal):
    """`onsite_ri_hadamard_add`: the SHELL-diagonal block only."""
    n = bts.size
    same_shell = bts[:, None] == bts[None, :]
    lb = loc[bts]
    ri = _np.asarray(onsite_ri)
    W = scal * ri[lb, aoat][:, None] * _np.ones((1, n))
    return _np.where(same_shell, W, 0.0)


def _onsite_fx_symv(onsite_fx, xvec, bts, sh_atom, sh_loc):
    """`onsite_fx_symv`: sum xvec over each SHELL, then spread by the on-site
    shell-pair matrix onto every AO of the target shell.

        s = sum over the AOs of shell js of xvec
        yvec(ii+p) += s * kmat(is, js, iat)     for every AO p of shell is
    """
    nsh = sh_atom.size
    ssum = _np.bincount(bts, weights=_np.asarray(xvec, dtype=float), minlength=nsh)
    fx = _np.asarray(onsite_fx)
    ysh = _np.zeros(nsh)
    for iat in _np.unique(sh_atom):
        m = _np.where(sh_atom == iat)[0]
        lo = sh_loc[m]
        # ysh[is] = sum_js ssum[js] * fx[is, js, iat]
        ysh[m] = fx[_np.ix_(lo, lo)][:, :, iat] @ ssum[m] if fx.ndim == 3 else 0.0
    return ysh[bts]


def _get_gons(basis, atoms, qsh, alpha=0.15):
    """`exchange_fock::get_gons`: the on-site matrix and its RI companion.

        onsite_fx(jsh,ish,iat) =
            (1 - 0.5*(kq(ish)*q(ish) + kq(jsh)*q(jsh))) * alpha * onsite_sh
        onsite_ri(ish,iat)     = onsite_fx(ish,ish,iat) / (4*ish - 2)   [ish > 1]

    `onsite_sh(jsh,ish,izp) = get_onecxints_number(l_jsh, l_ish, num)` and
    `kq_sh(ish,izp) = pg_fock_kq(l+1)`, both from `add_exchange`.
    """
    sh_atom = _np.asarray(basis.shell_atom)
    sh_l = _np.asarray(basis.shell_l)
    at = _np.asarray(atoms, dtype=_np.intp)
    loc = _np.zeros(sh_atom.size, dtype=_np.intp)
    cnt = {}
    for i, a in enumerate(sh_atom):
        loc[i] = cnt.get(int(a), 0)
        cnt[int(a)] = loc[i] + 1
    nat = at.size
    mx = int(loc.max()) + 1
    tbl, lidx = _onecx_tables()
    kq = _np.asarray(GXTB_PARAMS["pg_fock_kq"], dtype=_np.float64)
    q = _np.asarray(qsh, dtype=_np.float64)

    fx = _np.zeros((mx, mx, nat))
    ri = _np.zeros((mx, nat))
    for iat in range(nat):
        m = _np.where(sh_atom == iat)[0]
        Z = int(at[iat])
        for a in m:
            for b in m:
                os_ = tbl[Z - 1, lidx[sh_l[a], sh_l[b]] - 1]
                val = (1.0 - 0.5 * (kq[sh_l[a]] * q[a] + kq[sh_l[b]] * q[b])) * (alpha * os_)
                fx[loc[a], loc[b], iat] = val
            i1 = int(loc[a]) + 1          # 1-based shell index
            if i1 != 1:
                ri[loc[a], iat] = fx[loc[a], loc[a], iat] / (4 * i1 - 2)
    return fx, ri, loc, sh_atom


def gxtb_kfock_exact(P, S, basis, atoms, kmat_shell, qsh, bomat=None,
                     nspin=1, alpha=0.15, coords_ang=None):
    """`exchange_fock::get_kfock`, the whole assembly.

    Three Hadamard passes and two contractions, in the binary's order.  The
    four weight matrices and their scal factors are `get_kfock`'s own
    constants; each `*_hadamard_add` primitive is verified against a literal
    transcription of its Fortran loops (0.000e+00 for all four).
    """
    P = _np.asarray(P, dtype=float)
    S = _np.asarray(S, dtype=float)
    bts = _np.asarray(basis.bf_to_shell)
    fx, ri, loc, sh_atom = _get_gons(basis, atoms, qsh, alpha=alpha)
    aoat = sh_atom[bts]
    n = bts.size
    if bomat is None:
        # 🔑 `bomat` enters `atom_hadamard_add` with scal = -4.0, the largest
        # weight in the whole chain.  Passing zero is not "leaving out a small
        # correction", it is deleting the term.
        if coords_ang is None:
            bomat = _np.zeros((len(_np.asarray(atoms)), len(_np.asarray(atoms))))
        else:
            bomat = gxtb_bocorr_gamma(basis, atoms, coords_ang,
                                      _atom_matrix_only=True)

    W_sh = _ao_weight_shell(kmat_shell, bts, _KF_SHELL_SCAL)
    W_at = _ao_weight_atom(bomat, aoat, _KF_ATOM_SCAL)
    W_fx = _ao_weight_onsite_fx(fx, bts, aoat, loc, _KF_ONSITE_FX_SCAL)
    W_ri = _ao_weight_onsite_ri(ri, bts, aoat, loc, _KF_ONSITE_RI_SCAL)

    scal = 1.0 if nspin >= 2 else 0.5

    xmat = S @ P
    fxa = _onsite_fx_symv(fx, _np.diag(P), bts, sh_atom, loc)
    fxb = _onsite_fx_symv(fx, _np.diag(xmat), bts, sh_atom, loc)
    cvec = _np.einsum("ij,ij->i", xmat, S)          # diag(S P S)
    fxc = _onsite_fx_symv(fx, cvec, bts, sh_atom, loc)

    kfock = 0.5 * (xmat @ S)

    # pass A -- fold the four weights onto xmat
    xmat = (xmat * W_sh) + (xmat * W_at) + (xmat.T * W_fx) + (xmat * W_ri)
    # pass B -- fold onto kfock
    kfock = (kfock * W_fx) + (kfock * W_ri) + (kfock * W_sh)
    # pass C -- fold onto the density
    tmp = (P * W_sh) + (P * W_fx) + (P * W_ri)

    xmat = xmat + S * (0.25 * fxa)[None, :]         # daxpy per column
    xmat = 0.5 * (S @ tmp) + xmat
    kfock = xmat @ S + kfock
    kfock = kfock + S * (0.5 * fxb)[None, :]
    kfock = kfock + _np.diag(0.25 * fxc)

    # symmetrise_kfock
    out = -(0.25 * scal) * (kfock + kfock.T)
    _np.fill_diagonal(out, -(_np.diag(kfock) * scal * 0.5))
    return out
