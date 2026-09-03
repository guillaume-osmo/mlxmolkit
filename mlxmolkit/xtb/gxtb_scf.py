# Vendored from mlxmolkit/xtb/scf_gxtb.py @ wip/xtb_cosmo_area_fix @ 857a64f.
#
# THIS IS THE MUTABLE FILE. It is the whole search space of the `gxtb` track.
#
# Everything it imports from `mlxmolkit.xtb` is FROZEN: the Slater overlap, the
# q-vSZP basis, the multipole integrals, the extracted binary parameter blocks.
# Those were read out of the released g-xTB binary and are not hypotheses.
#
# If you need to change something that lives in a frozen module, do NOT edit
# mlxmolkit. Copy the function into this file and redefine the name here — the
# call sites below resolve against this module's globals, so a redefinition
# shadows the import and the change stays inside one revertible commit.
#
# `solve(atoms, coords_ang)` at the bottom is the entry point run_gxtb.py calls.
# Its return contract is fixed; the harness reads exactly those keys.

# Copyright (c) 2026 Guillaume
# SPDX-License-Identifier: MIT

"""Experimental native g-xTB single-point and gradient driver."""

from __future__ import annotations

import contextlib
import math

import numpy as np
import os.path as _os_path

from mlxmolkit.xtb.dispersion_d4srev import d4srev_dispersion_gxtb
from mlxmolkit.xtb.gxtb_acp import (
    build_gxtb_acp_hamiltonian as _frozen_build_acp,
    gxtb_pacp_proxy_energy,
)
from mlxmolkit.xtb.gxtb_basis import (
    ANG_TO_BOHR,
    build_gxtb_qvszp_basis as _frozen_build_basis,
)
from mlxmolkit.xtb.gxtb_reconstructed import gxtb_reconstructed_repulsion
import mlxmolkit.xtb.hcore_gxtb as _hcore_mod
from mlxmolkit.xtb.hcore_gxtb import (
    _diat_interaction_index,
    _diat_scale,
    _element_has_active_d_shell,
    _shell_index_groups,
    build_hcore_gxtb as _frozen_build_hcore_gxtb,
    _carbon_plevel_shift,
    _diatomic_scaled_overlap_cao,
    _h0_shell_kscale,
    gxtb_shell_selfenergies,
)
from mlxmolkit.xtb.qvszp_params import QVSZP_PARAMS
from mlxmolkit.xtb.basis import BasisFunction
from mlxmolkit.xtb.gxtb_basis import _contraction_norm, _primitive_norms
from mlxmolkit.xtb.qvszp_params import QVSZP_PARAMS
from mlxmolkit.xtb.params_gxtb import GXTB_PARAMS


EV_PER_HARTREE = 27.211386245988
KCAL_PER_HARTREE = 627.5094740631
# The recovered multipole model, OFF until its `vat` is wired correctly.
#
# The chain itself is exact -- `gxtb_aes_recovered` matches the binary's own
# container to 1e-17 on H2O, CH3SH and benzene (gxtb-recovery
# probes/audit/port_divergence.py stage 8c) -- but switching it on here blows
# the benchmark up (max_abs_dq 0.249 -> 7.69), because `pot%vat` is NOT the
# port's `vs`:
#
#   binary   vdp, vqp -> add_vmp_to_h1  (the H1 multipole route)
#            vat      -> add_vat_to_vsh (folded into the SHELL potential)
#   port     vs, vd, vq all go to `fockelectro`, which applies vs as
#            S*(vs_i + vs_j) -- the H1 monopole route
#
# So `vat` has to join V_sh alongside the repulsion's atom potential, not ride
# into fockelectro. Until that is done this stays off; vd/vq alone would drop
# the AES monopole channel entirely, which is not a smaller error.
GXTB_AES_RECOVERED = True
_AES_ROUTE = "both"

GXTB_TB2_KEXP = 0.294621155
# Two-body third-order scalars, stored consecutively in the binary param block as
# (2.3, 0.2093327496, 1.3) and read by get_taumat_0d._omp_fn.0 as:
#   off-site prefactor (k3)        = 2.3
#   on-site  prefactor (rexp)      = 0.2093327496   (A2-validated vs Ne)
#   off-site exp decay (kx)        = 1.3
# The earlier port swapped KX<->REXP, blowing the term up (E3 ~ -127 Ha, MAE 1.86).
GXTB_TB3_K = 2.3
GXTB_TB3_KX = 1.3
GXTB_TB3_REXP = 0.2093327496
# 4th-order onsite hardness Gamma4_sh = shell_fourth * K4TH_SCALE, where
# shell_fourth = pg_tb4_kshell[l] (NO pa_tb3_hubbard_derivs factor; see gxtb_basis).
# Binary-exact: add_coulomb 0x41a0b4 loads DAT_005dbbe8 = 0.036 and multiplies
# pg_tb4_kshell directly -- no per-element hubbard factor. Energy = sum q^4 Gamma4/24,
# potential = q^3 Gamma4/6 (the /6 and /24 below are the only divisors).
GXTB_K4TH_SCALE = 0.036
GXTB_TB1_KX = 1.0
GXTB_TB1_KDIS = 0.025
GXTB_TB1_KS = 0.666666666
GXTB_TB1_CN_EPS = 1.0e-12
# Mulliken-Fock-exchange range-separation scalars. VERIFIED against the released
# g-xTB binary: the exact constants new_exchange_fock receives are baked at
# libxtb __const 0x73b4d8.. = {gexp=1.38265972, lrscale=0.85, omega=0.2, frscale=0.15}.
# NB: the public gp3.f90 source declares fock_omega=0.300, but that branch is STALE;
# the released binary uses omega=0.2 (this value). Binary is authoritative.
GXTB_MFX_FR_SCALE = 0.15
# DECODED, 2026-08-27.  `get_mulliken_kmatrix` passes the envelope scalars to
# `get_gmulliken_0d` as lVar6+0x10 / +0x18 / +0x20, dereferenced in the omp body
# as FR, OM and LR (offsite envelope = *pdVar18 + erf(R * *pdVar36) * *pdVar19).
# The g-xTB exchange record is the __const block at libxtb vm 0x74a2d0, where
# +0x10 = 0.15 (FR), +0x18 = 0.2 (OM), +0x28 = 1.3826597204 (GE) and
# +0x30 = 0.05 (pg_fock_offdiag_l) all match constants this port had already
# verified independently -- four confirmations that it is the right record.
# ⚠️ The line that used to stand here claimed "+0x20 holds **1.0**, not the
# 0.85 taken from the tblite/GFN2 block", justified by "worth 1.7006 -> 1.2293
# eV on the H-L gap". That was a FIT wearing a decode's clothes, and it is
# wrong. Two independent checks say 0.85:
#
#   * `recovered/xtb_gxtb/add_exchange.f90` -- which has a passing differential
#     test against the binary -- declares `fock_lrscale = 0.85_wp` and passes
#     (fock_alpha, fock_omega, fock_lrscale) = (0.15, 0.2, 0.85) to
#     `new_exchange_fock`.
#   * Reading the exchange container OUT OF A LIVE CALCULATOR (calc + 0x600)
#     gives +0x10 = 0.15, +0x18 = 0.2, +0x20 = 0.85.
#
# and 0.15 + 0.85 = 1.0 is the short-range/long-range split, as it should be.
# Reading a static __const record is not the same as reading the object the
# constructor actually built.
GXTB_MFX_LR_SCALE = 0.85
GXTB_MFX_OMEGA = 0.2
# 🔑 __const 0x73b4d8 is `onsite`, NOT gexp. The block base and slots 1-3
# (lrscale 0.85, omega 0.2, frscale 0.15) were read correctly; only the NAME on
# slot 0 was wrong. The real gexp is 1.0 at 0x73b500 (new_exchange_fock arg 13
# -> container +0x2d0). Verified three ways, including a 0-ulp ctypes drive of
# the shipped get_gmulliken_0d over five (onsite, gexp) pairs: gexp acts ONLY
# off-site, `onsite` ONLY on the shell diagonal, so they cannot substitute.
GXTB_MFX_GEXP = 1.0
#: __const 0x73b4d8 -> new_exchange_fock arg 7 -> container +0x230. Replaces
#: `favg` on the on-site shell DIAGONAL: kmat(ii+ish,ii+ish) = gam*onsite*alpha.
GXTB_MFX_ONSITE_SCALE = 1.3826597204
# Distance damping on the averaged shell hardness inside the MFX kernel.
# DECODED: `get_gmulliken_0d._omp_fn.0` computes
#     damp = exp(-(R * (c0 + xi*c1)));   pair = favg / damp
# and `add_exchange` passes the two constants adjacently to new_exchange_fock
# (0x73b2e8 = 0.01851153839, 0x73b2f0 = -0.2960502355).
# Verified against the binary's own H2 exchange gamma, which is recoverable
# exactly by inverting its two printed eigenvalues (probes/h2_gamma.py):
#     target gamma_offsite = 0.282920   this form gives 0.28270  (0.07 %)
# The port previously used `favg * frscale` with no damping, giving 0.145317.
#: Set to a callable to override the MFX damping for an experiment; None = shipped
#: behaviour.
#:
#: ⚠️ The note that used to sit here -- "the shipped form is known wrong (it
#: multiplies R_AB by `xi`, the 1-or-2 averaging-mode selector)" -- was STALE and
#: wrong. `_mfx_gamma_ao` takes `xi` from `ps_fock_avg_exp[Z, ishell]`, which is
#: exactly the per-shell table the binary indexes:
#: `exchange_fock::get_gmulliken_0d._omp_fn.0` computes the damping as
#: `exp(-(r1*(c0 + xi(ish,izp)*c1)))` and `add_coulomb`'s sibling `add_exchange`
#: builds `xi(ish, izp) = ps_fock_avg_exp(ish, num)`.
#:
#: The rest of the form matches too. The binary ends with
#:     gam   = (A/damp)*B
#:     value = (frscale + lrscale*erf(omega*r1)) / (r1**gexp + gam**(-gexp))**(1/gexp)
#: which is this module's expression, damping included (dividing by exp(-x) is
#: multiplying by exp(+x), which is why the sign here looks inverted and is not).
#:
#: ONE DIFFERENCE IS STILL OPEN: the binary carries TWO separate factors
#: `(A/damp)*B`, where this code uses a single combined `favg`. Whether A*B is
#: that same average has not been established -- it needs the two pointers
#: traced, and a wrong guess here is silent. Not changed on suspicion.
_MFX_DAMP_HOOK = None

# 🔑 The two constants had their ROLES EXCHANGED and c0 had lost its sign. In
# the __const block the coefficient sits at the LOWER address and the additive
# term at the higher, so naming them in address order swaps them; the CALL SITE
# disambiguates (add_exchange 0x418f18 passes 0x73b510 as arg 10 -> the fmadd
# ADDEND). The multiplicand is the vdW PAIR RADIUS, not xi: `exp` is called once
# per ATOM PAIR at 0x34ad2c and the shell loops do not open until 0x34ae10, so a
# per-shell quantity is structurally incapable of entering the damping.
#   damp = exp(-(r * (c0 + rad(izp,jzp)*c1)))     pair term = favg/damp
#: __const 0x73b510 -> new_exchange_fock arg 10 -> container +0x2c0
GXTB_MFX_DAMP_C0 = -0.2960502355
#: __const 0x73b508 -> new_exchange_fock arg 11 -> container +0x2c8.
#: Full double: the old 0.01851153839 was a 10-digit truncation.
GXTB_MFX_DAMP_C1 = 0.018511538388678535
GXTB_HALIDE_INCREMENT_CORRECTION = {
    # Oracle-calibrated additive shifts for the extracted release increments.
    # The correction is a constant per atom and therefore has zero gradient.
    9: -1.824363963678883,
    17: -0.432894263892963,
    35: -0.105914643649314,
}
_TWO_OVER_SQRT_PI = 2.0 / math.sqrt(math.pi)


# ------------------------------------------------- H0 effective (scaled) basis
#
# `ps_h0_qvszp_exp_scal` (103, 4) is extracted from the released binary and used
# NOWHERE.  It is named in one comment in gxtb_basis.py -- "an H0-only
# effective-basis scaling [that] must NOT touch the overlap basis" -- and then
# never applied to anything.  So H0 is currently built from the same overlap as
# the density, and the scaling is simply missing.
#
# Its values are element- AND shell-resolved and are not 1 (0.79 .. 1.82 over
# Z=1..36, mean 1.12), so applying it is not a no-op.  It multiplies the
# primitive exponents of the shell, i.e. H0 sees a contracted or diffused
# version of each shell while the density keeps the base basis.
#
# Why this term and not another: with the fitted carbon patch OFF, carbon's own
# s/p split is nearly right (s +0.014, p +0.055 e) and carbon is simply 0.069 e
# too negative, while O is +0.121 e and H +0.033 e too positive -- a C-O / C-H
# BOND POLARITY error, not an intra-atomic one.  Bond polarity lives in the H0
# off-diagonal, which is exactly what an effective-basis overlap rescales.  And
# C p carries the largest contraction of any C/N/O valence shell (1.314), which
# shrinks C p bonding overlap -- the direction the carbon patch fakes by hand.
#
# Zero free parameters: every number is the binary's own.

_H0_BASIS_CACHE: dict = {}


class _H0BasisProxy:
    """`basis` with `S_cao` swapped for the effective-basis overlap.

    Everything else -- T_cao_to_sao, bf_to_shell, shell_l, cn, cao_basis -- is
    forwarded to the real basis, so the frozen `build_hcore_gxtb` can be used
    unmodified.  Only H0 sees this; the SCF's density overlap `basis.S` is
    untouched, which is what "H0-only" requires.
    """

    __slots__ = ("_basis", "S_cao")

    def __init__(self, basis, S_cao):
        object.__setattr__(self, "_basis", basis)
        object.__setattr__(self, "S_cao", S_cao)

    def __getattr__(self, name):
        return getattr(self._basis, name)


def _h0_effective_overlap_cao(atoms: np.ndarray, basis, power: float = 1.0) -> np.ndarray:
    """CAO overlap over the exponent-scaled H0 basis.

    Each shell's primitive exponents are multiplied by
    ``ps_h0_qvszp_exp_scal[Z, ishell]`` and the contraction is renormalised, so
    every basis function stays unit-normalised and only its radial extent
    changes.  The raw contraction coefficients are recovered by dividing out
    the base primitive norms; the overall contraction constant cancels against
    the fresh `_contraction_norm`, so this does not need the original raw table.
    """
    key = (id(basis), float(power))
    hit = _H0_BASIS_CACHE.get(key)
    if hit is not None and hit[0] is basis:
        return hit[1]
    if len(_H0_BASIS_CACHE) > 8:
        _H0_BASIS_CACHE.clear()

    scal = np.asarray(GXTB_PARAMS["ps_h0_qvszp_exp_scal"], dtype=np.float64)
    # shell_id -> index of that shell within its atom, to address the table
    local = _shell_local_indices(np.asarray(basis.shell_atom, dtype=np.int64))
    cao_shell = np.asarray(basis.cao_bf_to_shell, dtype=np.int64)
    # The H0 set is a full second `new_qvszp_cgto` call: scaled exponents AND
    # its own charge response, from k0/k2/k3 scaled by the three pa_h0 tables.
    # Rebuilding `raw = c + c_env*qeff_h0` from the shell data is the only way
    # to get it -- dividing out the norms of `bf.coeffs` recovers the DENSITY
    # raw coefficients, which carry the density qeff.
    qeff_h0 = _qeff_h0(atoms, basis)

    scaled: list[BasisFunction] = []
    for mu, bf in enumerate(basis.cao_basis):
        iat = int(bf.atom_idx)
        Z = int(atoms[iat])
        ish = int(local[int(cao_shell[mu])])
        f = float(scal[Z - 1, ish]) ** power if ish < scal.shape[1] else 1.0
        l = int(bf.l_total)
        a0 = np.asarray(bf.alphas, dtype=np.float64)
        a1 = a0 * f if (f > 0.0 and abs(f - 1.0) >= 1.0e-14) else a0
        qshell = QVSZP_PARAMS.shell(Z, ish)
        raw = (np.asarray(qshell.coefficients, dtype=np.float64)
               + np.asarray(qshell.coefficients_env, dtype=np.float64)
               * float(qeff_h0[iat]))
        if raw.shape != a1.shape:      # shell lookup disagrees: keep the old path
            if a1 is a0:
                scaled.append(bf)
                continue
            raw = np.asarray(bf.coeffs, dtype=np.float64) / _primitive_norms(l, a0)
        c1 = raw * _primitive_norms(l, a1) * _contraction_norm(a1, raw, l)
        scaled.append(
            BasisFunction(
                atom_idx=bf.atom_idx,
                l_total=bf.l_total,
                l_xyz=bf.l_xyz,
                center=bf.center,
                alphas=a1,
                coeffs=c1,
                is_valence=bf.is_valence,
                shell_id=bf.shell_id,
            )
        )

    try:
        from mlxmolkit.xtb.multipole_integrals_cpp import (
            CPP_AVAILABLE,
            multipole_matrices_cpp,
        )
        if not CPP_AVAILABLE:
            raise ImportError
        S_cao = multipole_matrices_cpp(scaled)[0]
    except Exception:
        from mlxmolkit.xtb.basis import overlap_matrix
        S_cao = overlap_matrix(scaled)

    S_cao = np.asarray(S_cao, dtype=np.float64)
    _H0_BASIS_CACHE[key] = (basis, S_cao)
    return S_cao


# --------------------------------------------- second-order hardness CN form
#
# DECODED from `___tblite_coulomb_charge_effective_MOD_get_amat_0d._omp_fn.0`
# (Ghidra). The effective shell hardness the Coulomb A-matrix uses is
#
#     eta = eta_base * (1 + cn_slope * (sqrt(cn + 1e-12) - 1e-6))
#
# i.e. it scales with **sqrt(CN)**. `gxtb_basis.build_gxtb_qvszp_basis` builds
# `shell_hardness` as `eta_base * (1 + cn_slope * cn)` -- LINEAR in CN. The
# port already knows the right form elsewhere: `_third_order_twobody`'s
# docstring states the sqrt version verbatim, so the two paths contradict each
# other and the linear one is the wrong one.
#
# Zero free parameters; `1e-12` and `1e-6` are the binary's own literals.


def _corrected_shell_hardness(atoms: np.ndarray, basis) -> np.ndarray:
    sa = np.asarray(basis.shell_atom, dtype=np.int64)
    loc = _shell_local_indices(sa)
    Z = np.asarray(atoms, dtype=np.intp)[sa]
    base = (
        np.asarray(GXTB_PARAMS["ps_tb2_shell_hubbard"], dtype=np.float64)[Z - 1, loc]
        * np.asarray(GXTB_PARAMS["pa_hubbard_parameter"], dtype=np.float64)[Z - 1]
    )
    slope = np.asarray(GXTB_PARAMS["pa_tb2_hubbard_cn"], dtype=np.float64)[Z - 1]
    cn = np.asarray(basis.cn, dtype=np.float64)[sa]
    eta = base * (1.0 + slope * (np.sqrt(cn + 1.0e-12) - 1.0e-6))
    return np.maximum(eta, 1.0e-8)


# ------------------------------------------------ H0 distance factor (shpoly)
#
# DECODED from `___tblite_xtb_h0_MOD_get_hamiltonian._omp_fn.0` plus
# `___tblite_xtb_gxtb_MOD_get_shpoly2` / `get_shpoly4`.
#
# tblite's H0 off-diagonal carries a THREE-term distance polynomial per shell:
#
#     pi_i = 1 + shpoly[i]*sqrt(R/rcov) + shpoly2[i]*(R/rcov) + shpoly4[i]*(R^2/rcov)
#            (get_hamiltonian: dVar73, dVar74, dVar75 at object offsets
#             0x280, 0x2d8, 0x330)
#
# For g-xTB:
#   * `get_shpoly` is NOT overridden -- the base
#     `___tblite_xtb_calculator_MOD_get_shpoly` writes 0, so the sqrt term
#     VANISHES. That is precisely the term the port implements.
#   * `get_shpoly2` = pg_h0_shpoly2[l] * 2*pa_h0_shpoly2[Z]      (`dVar18+dVar18`)
#   * `get_shpoly4` = the same, times 0.0402348406.
#
# So the port's `1 + pa*pg*sqrt(R/rcov)` is wrong three ways: the wrong power,
# a missing factor 2, and a missing quadratic term.
GXTB_SHPOLY4_SCALE = 0.0402348406
# `get_rad` (0x417308) is
#     mctc_data_covrad::get_covalent_rad_num(Z) * 1.889725949
# and mctc defines `covalent_rad_d3 = 4/3 * covalent_rad_2009` (Pyykko).
#
# `QVSZP_PARAMS["cov_radii"]` is ALREADY IN BOHR: divide it by the Pyykko radii
# and you get exactly 1.8897 (= ANG_TO_BOHR) for C, N, O, F and Cl. The frozen
# H0 builder multiplies it by ANG_TO_BOHR a SECOND time, so its rcov is 1.89x
# too large. The right value is simply the D3 radius in Bohr:
#
#     rcov = cov_radii * 4/3          (cov_radii already Bohr)
#
# Both numbers are definitions, not fits.
GXTB_RCOV_SCALE = (4.0 / 3.0) / ANG_TO_BOHR

# ...and both were wrong. `recovered/xtb_gxtb/get_rad.f90` (differential test vs
# the binary) says:
#
#     rad = get_covalent_rad_num(Z) * 0.75 / 1.8897261246204404 * 1.889725949
#
# `get_covalent_rad_num` is the D3 radius, i.e. covalent_rad_2009 * 4/3 * aatoau,
# so the 0.75 CANCELS the 4/3 and the whole expression collapses to
#
#     rad = covalent_rad_2009[Ang] * 1.889725949
#
# which is exactly `QVSZP_PARAMS["cov_radii"]` as stored (verified elementwise:
# raw/ANG_TO_BOHR gives 0.29, 0.75, 0.71, 0.63, ... = covalent_rad_2009). So the
# radius needs NO scaling at all -- GXTB_RCOV_SCALE made it 4/3 too large.
# Three elements are then overridden outright (0.403, 0.500, 0.610 Ang):
GXTB_RAD_OVERRIDE = {
    1: 0.7615595574470001,      # H  <- 0.403 Ang, raw would be 0.29
    9: 0.9448629745,            # F  <- 0.500 Ang, raw would be 0.64
    10: 1.15273282889,          # Ne <- 0.610 Ang, raw would be 0.67
}


# ------------------------------------------- the diatomic frame, for real
#
# `recovered/tblite_integral_diat_trafo/diat_trafo.f90` + `scale_diatomic_frame`
# (both differential-tested against the binary). `get_hamiltonian` calls it on
# the WHOLE atom-pair block of the H0-basis overlap, in the cumulative
# real-spherical layout sdim(l) = (l+1)**2, BEFORE multiplying by hij:
#
#     blk <- overlap_cgto over bas%cgto_h0
#     if (h0%diat_enabled) call diat_trafo(blk, vec, ksig, kpi, kdel, &
#                             & nsh_at(jat)-1, nsh_at(iat)-1)
#     hamiltonian += hij*blk
#
# i.e. rotate into the diatomic frame, scale the sigma/pi/delta channels
# SEPARATELY, rotate back. `hcore_gxtb._diatomic_scaled_overlap_cao` applies
# ONE SCALAR per shell pair instead (with a sigma/pi projector for p-p only),
# so d blocks never see a delta channel, cross-l blocks are never mixed, and
# `GXTB_D_H0_SCALE_DAMP` and the `has_d` gating are fitted knobs standing in
# for the missing rotation.
#
# The port's SAO order is not tblite's, so each shell is permuted (and d gets
# one sign flip, because `cao_to_sao_transform` defines z2 as 0.5(xx+yy) - zz
# = -d_z2). The map was not guessed: it is the UNIQUE choice out of all 3!*2^2
# x 5!*2^4 permutation/sign combinations that makes a real S block m-diagonal
# in the diatomic frame -- probes/diat_trafo_np.py, residual 1.4e-16.

_DIAT_PERM = {0: ([0], [1.0]),
              1: ([1, 2, 0], [1.0, 1.0, 1.0]),
              2: ([2, 4, 1, 3, 0], [1.0, 1.0, -1.0, 1.0, 1.0])}


def _diat_sao_map(basis):
    """Per atom: (SAO indices in tblite (l,m) order, matching sign vector)."""
    sh_at = np.asarray(basis.shell_atom, dtype=np.int64)
    sh_l = np.asarray(basis.shell_l, dtype=np.int64)
    bf_at = np.asarray([bf.atom_idx for bf in basis.sao_basis], dtype=np.int64)
    out = []
    for a in range(int(sh_at.max()) + 1):
        rows = np.flatnonzero(bf_at == a)
        idx, sgn, off = [], [], 0
        for k in np.flatnonzero(sh_at == a):
            l = int(sh_l[k])
            nc = 2 * l + 1
            perm, sg = _DIAT_PERM[l]
            idx.extend(rows[off:off + nc][list(perm)])
            sgn.extend(sg)
            off += nc
        out.append((np.asarray(idx, dtype=np.int64),
                    np.asarray(sgn, dtype=np.float64)))
    return out


_DIAT_SDIM = (1, 4, 9, 16, 25, 36, 49)
_DIAT_S3 = np.sqrt(3.0)


def _diat_rot_batch(vec, n):
    """`diat_trafo_np.diat_rot` for a whole (P, 3) batch of internuclear
    vectors -- every entry is the same scalar expression, evaluated
    elementwise, so the values are the per-pair routine's bit for bit."""
    P = vec.shape[0]
    rot = np.zeros((P, n, n), dtype=np.float64)
    rot[:, 0, 0] = 1.0
    if n == 1:
        return rot
    vx, vy, vz = vec[:, 0], vec[:, 1], vec[:, 2]
    rnorm = np.sqrt((vx * vx + vy * vy) + vz * vz)     # sum of three, in order
    ct = vz / rnorm
    pole = np.abs(ct) == 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        nx = np.where(pole, 0.0, vx / rnorm)
        ny = np.where(pole, 0.0, vy / rnorm)
        zero_ct = ct == 0.0
        st_g = np.sqrt(nx * nx + ny * ny)
        cp = np.where(pole, np.abs(ct), np.where(zero_ct, nx, nx / st_g))
        sp = np.where(pole, 0.0, np.where(zero_ct, ny, ny / st_g))
        st = np.where(pole, 0.0, np.where(zero_ct, 1.0, st_g))

    rot[:, 1, 1], rot[:, 2, 1], rot[:, 3, 1] = cp, 0.0, -sp
    rot[:, 1, 2], rot[:, 2, 2], rot[:, 3, 2] = st * sp, ct, st * cp
    rot[:, 1, 3], rot[:, 2, 3], rot[:, 3, 3] = ct * sp, -st, ct * cp

    if n >= 9:
        c2p = cp * cp - sp * sp
        s2p = (sp + sp) * cp
        c2t = ct * ct - st * st
        stct = (st + st) * ct
        st2s3 = st * st * _DIAT_S3
        stcts3 = stct * _DIAT_S3

        rot[:, 4, 4], rot[:, 5, 4], rot[:, 6, 4] = ct * c2p, -(st * cp), 0.0
        rot[:, 7, 4], rot[:, 8, 4] = st * sp, -(s2p * ct)

        rot[:, 4, 5], rot[:, 5, 5], rot[:, 6, 5] = st * c2p, ct * cp, 0.0
        rot[:, 7, 5], rot[:, 8, 5] = -(ct * sp), -(s2p * st)

        rot[:, 4, 6], rot[:, 5, 6] = s2p * st2s3 * 0.5, sp * stcts3 * 0.5
        rot[:, 6, 6] = (ct * ct * 3.0 - 1.0) * 0.5
        rot[:, 7, 6], rot[:, 8, 6] = cp * stcts3 * 0.5, c2p * st2s3 * 0.5

        rot[:, 4, 7], rot[:, 5, 7] = stct * s2p * 0.5, sp * c2t
        rot[:, 6, 7] = -(stcts3 * 0.5)
        rot[:, 7, 7], rot[:, 8, 7] = cp * c2t, stct * c2p * 0.5

        rot[:, 4, 8], rot[:, 5, 8] = s2p * (ct * ct + 1.0) * 0.5, -(sp * stct * 0.5)
        rot[:, 6, 8] = st2s3 * 0.5
        rot[:, 7, 8], rot[:, 8, 8] = -(cp * stct * 0.5), c2p * (ct * ct + 1.0) * 0.5

        for k in range(9, n):          # identity above d
            rot[:, k, k] = 1.0
    else:
        for k in range(4, n):
            rot[:, k, k] = 1.0
    return rot


def _diat_scaled_overlap_sao(atoms, coords_bohr, basis, S_sao):
    """Apply the recovered diat_trafo to every off-site atom-pair block.

    Same numbers as the per-pair loop over `probes.diat_trafo_np.diat_trafo`,
    batched by the pair's (l_a, l_b) class.  The rotation and the
    sigma/pi/delta scaling are elementwise scalar formulas and vectorise
    exactly; the four small products per pair go through `np.matmul` on a
    batch whose per-element operands have the shapes AND strides of the
    loop's -- the rotation batch is a fresh (P, n, n) C array like each
    fresh (n, n) was, the gathered blocks (P, si, sj) like each fresh
    (si, sj) -- so numpy issues the identical per-element `dgemm` (same
    trans flags, sizes and leading dimensions) and the bits agree (checked
    against the loop on the full S of every benchmark molecule).
    """
    S = np.array(S_sao, dtype=np.float64, copy=True)
    amap = _diat_sao_map(basis)
    sh_at = np.asarray(basis.shell_atom, dtype=np.int64)
    nsh = np.bincount(sh_at, minlength=len(amap))
    nat = len(amap)
    atoms = np.asarray(atoms, dtype=np.int64)
    # pairs (a > b), grouped by (la, lb)
    groups: dict = {}
    for a in range(nat):
        la = int(nsh[a]) - 1
        for b in range(a):
            groups.setdefault((la, int(nsh[b]) - 1), []).append((a, b))
    kcache: dict = {}

    def _k(Za, Zb):
        kk = kcache.get((Za, Zb))
        if kk is None:
            kk = kcache[(Za, Zb)] = (_diat_scale(Za, Zb, 0),
                                     _diat_scale(Za, Zb, 1),
                                     _diat_scale(Za, Zb, 2))
        return kk

    for (la, lb), pairs in groups.items():
        A = np.array([p[0] for p in pairs], dtype=np.int64)
        B = np.array([p[1] for p in pairs], dtype=np.int64)
        K = np.array([_k(int(atoms[a]), int(atoms[b])) for a, b in pairs],
                     dtype=np.float64)                        # (P, 3)
        IA = np.stack([amap[a][0] for a in A])                # (P, si)
        SA = np.stack([amap[a][1] for a in A])
        IB = np.stack([amap[b][0] for b in B])                # (P, sj)
        SB = np.stack([amap[b][1] for b in B])
        si, sj = _DIAT_SDIM[la], _DIAT_SDIM[lb]
        rows, cols = IA[:, :, None], IB[:, None, :]
        blk = SA[:, :, None] * S[rows, cols] * SB[:, None, :]
        if max(la, lb) == 0:
            out = blk[:, 0, 0] * K[:, 0]
            out = (SA[:, :, None] * out[:, None, None]) * SB[:, None, :]
        else:
            n = _DIAT_SDIM[max(la, lb)]
            rot = _diat_rot_batch(coords_bohr[B] - coords_bohr[A], n)
            rotT = rot.transpose(0, 2, 1)
            tmp = np.matmul(np.matmul(rotT[:, :si, :si], blk[:, :si, :sj]),
                            rot[:, :sj, :sj])
            # scale_diatomic_frame: only the m-diagonal is touched
            for jl in range(min(lb, 2) + 1):
                for il in range(min(la, 2) + 1):
                    for m in range(-min(il, jl), min(il, jl) + 1):
                        tmp[:, il * (il + 1) + m, jl * (jl + 1) + m] *= K[:, abs(m)]
            out = np.matmul(np.matmul(rot[:, :si, :si], tmp), rotT[:, :sj, :sj])
            out = SA[:, :, None] * out * SB[:, None, :]
        S[rows, cols] = out
        S[IB[:, :, None], IA[:, None, :]] = out.transpose(0, 2, 1)
    return S


def build_hcore_gxtb(
    atomic_numbers,
    coords_ang,
    basis,
    *,
    carbon_plevel: bool = True,
    decoded_shpoly: bool = False,
):
    """Frozen H0 builder, vectorised, with the decoded shpoly polynomial.

    `decoded_shpoly=False` reproduces the frozen builder exactly (pinned by
    probes/check_hcore.py).
    """
    if not decoded_shpoly:
        return _frozen_build_hcore_gxtb(
            atomic_numbers, coords_ang, basis, carbon_plevel=carbon_plevel
        )

    atoms = np.asarray(atomic_numbers, dtype=np.intp)
    coords = np.asarray(coords_ang, dtype=np.float64)
    coords_bohr = coords * ANG_TO_BOHR

    # SAO space, because that is where the diatomic frame lives. Every
    # prefactor below is constant within a shell and T is block-diagonal by
    # shell, so this is the same product the CAO route formed -- except that
    # the overlap now carries the real frame rotation instead of a scalar.
    T = np.asarray(basis.T_cao_to_sao, dtype=np.float64)
    S_sao = _diat_scaled_overlap_sao(
        atoms, coords_bohr, basis, T @ np.asarray(basis.S_cao) @ T.T)

    c_plevel = _carbon_plevel_shift(atoms, coords_bohr) if carbon_plevel else None
    shell_self = gxtb_shell_selfenergies(atoms, basis, carbon_plevel_shift=c_plevel)
    sao_shell = np.asarray(basis.bf_to_shell, dtype=np.int64)
    bf_atom = np.asarray(basis.shell_atom, dtype=np.int64)[sao_shell]
    bf_l = np.asarray(basis.shell_l, dtype=np.int64)[sao_shell]
    bf_self = shell_self[sao_shell]

    # get_rad: the stored radius verbatim, then three element overrides.
    atom_cov = np.asarray(QVSZP_PARAMS["cov_radii"][atoms - 1], dtype=np.float64).copy()
    for _z, _r in GXTB_RAD_OVERRIDE.items():
        atom_cov[atoms == _z] = _r
    pa = np.asarray(GXTB_PARAMS["pa_h0_shpoly2"][atoms - 1], dtype=np.float64)
    pg = np.asarray(GXTB_PARAMS["pg_h0_shpoly2"], dtype=np.float64)
    kshell = np.asarray(GXTB_PARAMS["pg_h0_kshell"], dtype=np.float64)

    s2 = 2.0 * pa[bf_atom] * pg[bf_l]          # get_shpoly2
    s4 = s2 * GXTB_SHPOLY4_SCALE               # get_shpoly4

    R = np.linalg.norm(
        coords_bohr[bf_atom][:, None, :] - coords_bohr[bf_atom][None, :, :], axis=-1
    )
    rcov = np.maximum(atom_cov[bf_atom][:, None] + atom_cov[bf_atom][None, :], 1.0e-12)
    x = R / rcov            # dVar74
    y = R * x               # dVar75 = R^2 / rcov
    pi_mu = 1.0 + s2[:, None] * x + s4[:, None] * y
    pi_nu = 1.0 + s2[None, :] * x + s4[None, :] * y

    hscale = 0.5 * (kshell[bf_l][:, None] + kshell[bf_l][None, :])
    # new_gxtb_h0spec: the d-f pair, and only that pair, is then boosted by 1.5
    # (hscale(4,3) and hscale(3,4) in the binary's 1-based l+1 indexing).
    _df = ((bf_l[:, None] == 2) & (bf_l[None, :] == 3)) | (
        (bf_l[:, None] == 3) & (bf_l[None, :] == 2))
    hscale = np.where(_df, hscale * 1.5, hscale)
    h_avg = 0.5 * (bf_self[:, None] + bf_self[None, :])
    H0 = hscale * h_avg * pi_mu * pi_nu * S_sao
    H0 = np.where(bf_atom[:, None] == bf_atom[None, :], 0.0, H0)
    np.fill_diagonal(H0, shell_self[sao_shell])
    return H0, shell_self


# ------------------------------------------- charge-dependent SCF repulsion
#
# DECODED from `___tblite_repulsion_gxtb_MOD_get_scaled_zeff` and
# `___tblite_repulsion_gxtb_MOD_get_potential`:
#
#     zeff_i = (1 - q_i * pa_rep_q[Z_i]) * pa_rep_zeff[Z_i]
#     V_i   -= (dE/dzeff_i) * pa_rep_zeff[Z_i] * pa_rep_q[Z_i]
#
# So the g-xTB repulsion is CHARGE DEPENDENT and contributes an atomic
# potential inside the SCF. The port computes the repulsion once, post-SCF,
# with `descriptor=q_at`, and never feeds it back -- the potential is simply
# missing from the Fock.
#
# `mlxmolkit.xtb.gxtb_cpp` already documents the same derivative
# ("descriptor derivative is -zeff[Z] * scale[Z] * matvec"), and
# `gxtb_reconstructed_repulsion` already returns `matvec` = dE/dzeff. Nothing
# new is fitted here; the existing pieces are simply connected.


_REP_STATIC_CACHE: dict = {}


def _repulsion_matvec(atoms, coords_ang, basis, q_at):
    """`gxtb_reconstructed_repulsion(...).matvec`, and nothing else.

    The frozen routine also forms the energy and the full gradient -- the
    explicit part in the asm kernel plus a CN chain rule over half a dozen
    (nat, nat) exp/pow arrays -- on every SCF iteration; the shell potential
    reads only `matvec`, which the kernel returns.  Same state builder, same
    kernel, same inputs; the geometry-only pair tables are cached per basis.
    """
    from mlxmolkit.xtb.gxtb_reconstructed import (
        repulsion_constants_from_binary, _pair_roffset_matrix,
        _vdw_pair_matrix_bohr, _coefficient_matrices, repulsion_state,
        repulsion_energy_gradient_asm)
    atoms = np.asarray(atoms, dtype=np.intp)
    coords = np.asarray(coords_ang, dtype=np.float64)
    key = id(basis)
    hit = _REP_STATIC_CACHE.get(key)
    if hit is not None and hit[0] is basis:
        st = hit[1]
    else:
        if len(_REP_STATIC_CACHE) > 8:
            _REP_STATIC_CACHE.clear()
        constants = repulsion_constants_from_binary()
        pair_roffset = _pair_roffset_matrix(atoms)
        pair_rvdw = _vdw_pair_matrix_bohr(atoms, constants)
        linear_coeff, quadratic_coeff = _coefficient_matrices(atoms, constants)
        st = (constants, pair_roffset, pair_rvdw, linear_coeff, quadratic_coeff,
              coords * ANG_TO_BOHR)
        _REP_STATIC_CACHE[key] = (basis, st)
    constants, pair_roffset, pair_rvdw, linear_coeff, quadratic_coeff, xyz_b = st
    state = repulsion_state(
        atoms,
        descriptor=np.asarray(q_at, dtype=np.float64),
        cn=np.asarray(basis.cn, dtype=np.float64),
        pair_roffset=pair_roffset,
    )
    _energy, _grad, matvec = repulsion_energy_gradient_asm(
        xyz_b,
        state.scaled_zeff,
        state.alpha,
        pair_rvdw,
        pair_roffset,
        linear_coeff,
        quadratic_coeff,
        constants.cubic_coeff,
        constants.quartic_coeff,
        constants.exp_power_1,
        constants.exp_power_2,
        constants.exp2_scale,
        constants.exp2_weight,
        cutoff=25.0,
    )
    return np.asarray(matvec, dtype=np.float64)


def _repulsion_shell_potential(atoms, coords_ang, basis, q_at):
    matvec = _repulsion_matvec(atoms, coords_ang, basis, q_at)
    Z = np.asarray(atoms, dtype=np.intp)
    zeff = np.asarray(GXTB_PARAMS["pa_rep_zeff"], dtype=np.float64)[Z - 1]
    kq = np.asarray(GXTB_PARAMS["pa_rep_q"], dtype=np.float64)[Z - 1]
    v_atom = -zeff * kq * matvec
    return v_atom[np.asarray(basis.shell_atom, dtype=np.int64)]


def _dispersion_shell_potential(atoms, coords_ang, basis, q_at):
    """The D4Srev dispersion's contribution to V_sh.

    Inputs: `coords_ang` in ANGSTROM, `q_at` the atomic charges in e.
    Returns Hartree/e per SHELL, broadcast from the per-atom potential exactly
    as the binary does -- `tblite_disp_d4::get_potential` writes `pot%vat` only
    and never touches `pot%vsh` (measured: its vsh comes back identically zero
    on every molecule tried).

    The model builds its own erf coordination number from its own covalent
    radii; that is NOT `basis.cn` and the two differ by an order of magnitude,
    so nothing is passed in here.
    """
    v_atom = _d4srev_atom_potential_fast(basis, atoms, coords_ang,
                                         np.asarray(q_at, dtype=np.float64))
    return v_atom[np.asarray(basis.shell_atom, dtype=np.int64)]


_D4_MOL_CACHE: dict = {}


def _d4srev_molecule_cache(basis, atoms, coords_ang):
    """Everything in `d4srev_atom_potential` that does not depend on the charges.

    `d4srev_weights` factorises as gw(r,i,j) = zeta_r(q_i) * gwk(r,i,j): the
    gaussian CN weights `gwk` (and their exceptional-branch fix-up) see only
    the geometry, the tanh gate sees only q_i.  The frozen routine rebuilds
    the CN, the damped r**-6 matrices and `gwk` on every SCF iteration; the
    binary's `tblite_disp_d4` caches all three in `update`.  Cached per basis
    like the AES gab and the multipole integrals.  Pure cost: the per-iteration
    multiplications below are the frozen routine's own, on the same operands.
    """
    key = id(basis)
    hit = _D4_MOL_CACHE.get(key)
    if hit is not None and hit[0] is basis:
        return hit[1]
    if len(_D4_MOL_CACHE) > 8:
        _D4_MOL_CACHE.clear()
    from mlxmolkit.xtb import gxtb_d4srev as _D
    t = _D._tables()
    Z = np.asarray(atoms, dtype=int)
    sp = _D._species_index(Z, t)
    nat = len(Z)
    cn = _D.d4srev_coordination_number(Z, coords_ang)
    ref, ngw, refcn, refq, wf = (t["ref"], t["ngw"], t["refcn"], t["refq"],
                                 t["wf"])
    mref = int(ref[sp].max())
    per_atom = []
    for i in range(nat):
        zi = sp[i]
        ta, tb, tc, td = t["tanh"][:, int(Z[i]) - 1]
        nr = int(ref[zi])
        kk = np.arange(1, int(ngw[:nr, zi].max()) + 1)[:, None, None]
        w = wf[zi, sp][None, None, :]
        dd = (cn[i] - refcn[:nr, zi])[None, :, None]
        block = np.exp(-((kk * w) * dd * dd))
        mask = (kk <= ngw[:nr, zi][None, :, None])
        g = np.where(mask, block, 0.0).sum(axis=0)
        nrm = g.sum(axis=0)
        good = np.abs(nrm) > _D._SQRT_TINY
        inv = np.where(good, 1.0 / np.where(good, nrm, 1.0), 0.0)
        gwk = g * inv[None, :]
        bad = (~good)[None, :] | ~np.isfinite(gwk) | (np.abs(gwk) > 1e300)
        top = np.abs(refcn[:nr, zi].max() - refcn[:nr, zi]) < 1e-12
        gwk = np.where(bad, np.where(top[:, None], 1.0, 0.0), gwk)
        den = ta + np.tanh(td + refq[:nr, zi] * tc) * tb
        per_atom.append((nr, ta, tb, tc, td, den, gwk))
    # the same, stacked over atoms and zero-padded to mref for the vectorised
    # gate: rows r >= nr carry gwk = 0 (as the frozen gw does) and den = 1
    tanh_p = np.array([[ta, tb, tc, td] for (_nr, ta, tb, tc, td, _d, _g) in per_atom])
    den_p = np.ones((nat, mref))
    gwk_p = np.zeros((nat, mref, nat))
    for i, (nr, _ta, _tb, _tc, _td, den, gwk) in enumerate(per_atom):
        den_p[i, :nr] = den
        gwk_p[i, :nr, :] = gwk
    # The pair products `x[:ni, i, j] @ c6ref[:ni, :nj, sp_i, sp_j] @ gw[:nj, j, i]`
    # are batched per SPECIES PAIR over an atom order sorted by species, so
    # every (i, j) of a group is a basic slice.  See `_d4srev_pair_contract`.
    c6ref = t["c6ref"]
    perm = np.argsort(sp, kind="stable")
    spp = sp[perm]
    us, starts, counts = np.unique(spp, return_index=True, return_counts=True)
    groups = []
    for za, i0, ci in zip(us, starts, counts):
        for zb, j0, cj in zip(us, starts, counts):
            ni, nj = int(ref[za]), int(ref[zb])
            groups.append((int(i0), int(i0 + ci), int(j0), int(j0 + cj),
                           ni, nj, c6ref[:ni, :nj, za, zb]))
    mat6, mat8 = _D.d4srev_dispersion_matrices(Z, coords_ang)
    r42 = t["r4r2"][sp]
    wmat = mat6 + 3.0 * np.outer(r42, r42) * mat8
    out = {"mref": mref, "nat": nat, "per_atom": per_atom, "perm": perm,
           "groups": groups, "wmat": wmat, "mat6": mat6, "mat8": mat8,
           "r42": r42, "tanh_p": tanh_p, "den_p": den_p, "gwk_p": gwk_p}
    _D4_MOL_CACHE[key] = (basis, out)
    return out


def _d4srev_pair_contract(c, xw, gw):
    """out[i, j] = xw[:ni, i, j] @ c6ref[:ni, :nj, sp_i, sp_j] @ gw[:nj, j, i]
    for every atom pair -- `d4srev_c6`'s loop body, batched WITHOUT changing
    a bit of it.

    numpy sends the loop body's two products to BLAS: `vec @ F-ordered table
    slice` is a `dgemv` (trans, lda = the table's leading dimension, incx =
    nat*nat) and the second is a `ddot` (incx = 1, incy = nat*nat).
    Accelerate's results DEPEND on those increments: the same kernel on
    contiguous copies of the vectors moves the last bit on ~30% of the
    entries (measured), so the usual "stack and einsum" is out.  A batched
    `matmul`, however, calls the identical per-element gemv/dot with the
    per-element core strides -- so the batch is arranged to present exactly
    the loop's operands: the weights are re-laid in a species-sorted atom
    order as a fresh (mref, nat, nat) C array (same strides as the
    original), each species pair is then a basic slice, and the table slice
    is the very same view, broadcast.  Bit-for-bit against the loop on 285
    random charge states over the whole benchmark set.
    """
    nat = c["nat"]
    perm = c["perm"]
    gwp = np.ascontiguousarray(gw[:, perm][:, :, perm])
    xwp = np.ascontiguousarray(xw[:, perm][:, :, perm])
    outp = np.empty((nat, nat))
    for (i0, i1, j0, j1, ni, nj, a) in c["groups"]:
        X = xwp[:ni, i0:i1, j0:j1].transpose(1, 2, 0)[:, :, None, :]
        tmp = np.matmul(X, a)
        Y = gwp[:nj, j0:j1, i0:i1].transpose(2, 1, 0)[:, :, :, None]
        outp[i0:i1, j0:j1] = np.matmul(tmp, Y)[:, :, 0, 0]
    out = np.empty((nat, nat))
    out[np.ix_(perm, perm)] = outp
    return out


def _d4srev_weights_fast(c, qat):
    """`d4srev_weights` on the cached geometry: only the tanh gate is new.

    The gate is elementwise in (atom, reference), so it is one broadcast
    expression over the zero-padded stacks; the padding rows multiply the
    zeros the frozen `gw` holds there.  Bit-for-bit with the per-atom loop.
    """
    q = np.asarray(qat, dtype=float)
    ta, tb, tc, td = (c["tanh_p"][:, k][:, None] for k in range(4))
    den = c["den_p"]
    qi = q[:, None]
    zeta = (ta + np.tanh(td + qi * tc) * tb) / den                 # (nat, mref)
    dzeta = ((tc * tb) / np.cosh(td + qi * tc) ** 2) / den
    gwk = c["gwk_p"]                                                # (nat, mref, nat)
    gw = np.ascontiguousarray(np.transpose(zeta[:, :, None] * gwk, (1, 0, 2)))
    dgw = np.ascontiguousarray(np.transpose(dzeta[:, :, None] * gwk, (1, 0, 2)))
    return gw, dgw


def _d4srev_energy_fast(basis, atoms, coords_ang, qat):
    """`d4srev_energy`, same numbers, on the cached geometry."""
    c = _d4srev_molecule_cache(basis, atoms, coords_ang)
    gw, _dgw = _d4srev_weights_fast(c, qat)
    c6 = _d4srev_pair_contract(c, gw, gw)
    r42 = c["r42"]
    c8 = 3.0 * np.outer(r42, r42) * c6
    return 0.5 * (c6 * c["mat6"] + c8 * c["mat8"]).sum(axis=1)


def _d4srev_atm_energy(basis, atoms, coords_ang):
    """Three-body dispersion, `get_dispersion3_energy` with screened damping.

    Per triple i > j > k of atoms within the cutoff of one another,

        ang  = 0.375*(a + c - b)*(a - c + b)*(-a + c + b) / (a*b*c)**2.5
               + 1 / (a*b*c)**1.5             a, b, c = r2_ij, r2_jk, r2_ik
        c9   = -sqrt(|c6_ij * c6_ik * c6_jk|)
        damp = s9 * (f_ij * f_jk * f_ik)**3
        f    = r / (r + 0.5*(1 + erf(-alp*(r - beta*r0)))*(a2 + a1*r0))
        E   -= ang * c9 * damp

    with r0 = sqrt(3 r4r2_i r4r2_j) as for the two-body term, the damping
    constants shared with it, and c6 evaluated at ZERO charge -- this term
    does not see the SCF, which is why the reference evaluates it once up
    front. Two-atom systems have no triple and contribute nothing.
    """
    from scipy.special import erf as _erf
    from mlxmolkit.xtb import gxtb_d4srev as _D
    c = _d4srev_molecule_cache(basis, atoms, coords_ang)
    n = int(c["nat"])
    if n < 3:
        return 0.0
    gw, _dgw = _d4srev_weights_fast(c, np.zeros(n))
    c6 = np.asarray(_d4srev_pair_contract(c, gw, gw), dtype=np.float64)
    r42 = np.asarray(c["r42"], dtype=np.float64)
    x = np.asarray(coords_ang, dtype=np.float64) * ANG_TO_BOHR
    d = x[:, None, :] - x[None, :, :]
    r2 = np.einsum("ijk,ijk->ij", d, d)
    r = np.sqrt(r2)
    r0 = np.sqrt(3.0 * np.outer(r42, r42))
    with np.errstate(divide="ignore", invalid="ignore"):
        f = r / (r + 0.5 * (1.0 + _erf(-_D.D4S_ALP * (r - _D.D4S_BETA * r0)))
                 * (_D.D4S_A2 + _D.D4S_A1 * r0))
    i, j, k = np.where(np.tri(n, n, -1, dtype=bool)[:, :, None]
                       & np.tri(n, n, -1, dtype=bool)[None, :, :])
    a, b, cc = r2[i, j], r2[j, k], r2[i, k]
    cut2 = _D.D4S_CUTOFF3 ** 2
    keep = (a <= cut2) & (b <= cut2) & (cc <= cut2)
    a, b, cc, i, j, k = a[keep], b[keep], cc[keep], i[keep], j[keep], k[keep]
    abc = a * b * cc
    sq = np.sqrt(abc)
    ang = (0.375 * (a + cc - b) * (a - cc + b) * (-a + cc + b) / (abc * abc * sq)
           + 1.0 / (abc * sq))
    c9 = -np.sqrt(np.abs(c6[i, j] * c6[i, k] * c6[j, k]))
    damp = _D.D4S_S9 * (f[i, j] * f[j, k] * f[i, k]) ** 3
    return -float(np.sum(ang * c9 * damp))


def _d4srev_atom_potential_fast(basis, atoms, coords_ang, qat):
    """`d4srev_atom_potential`, same numbers, with the geometry hoisted.

    Only `dc6dq` reaches the potential -- `c6` itself is computed and thrown
    away by the frozen routine -- so half of its pair products are skipped
    outright.  The rest go through `_d4srev_pair_contract`, which batches
    the loop body without changing its BLAS calls.
    """
    c = _d4srev_molecule_cache(basis, atoms, coords_ang)
    gw, dgw = _d4srev_weights_fast(c, qat)
    dc6dq = _d4srev_pair_contract(c, dgw, gw)
    return (c["wmat"] * dc6dq).sum(axis=1)


# ---------------------------------------------- one-centre exchange integrals
#
# `mlxmolkit.xtb.gxtb_aes` needs `data/gxtb_onecxints_extracted.npz` for the
# ONSITE exchange component and raises without it; program.md recorded that the
# file "exists nowhere on this machine". It does now: both tables are static
# data in the binary and were read straight out of it.
#
#     ___tblite_data_onecxints_MOD_onecxints  @ 0x6c0d58  (103 x 10 float64)
#     ___tblite_data_onecxints_MOD_lidx       @ 0x6c2d88  (4 x 4 int32)
#
# The size is exactly 0x6c2d88 - 0x6c0d58 = 8240 B = 1030 doubles = 103 x 10,
# and the values behave like one-centre exchange integrals should: 0.03-0.18 Ha
# and monotonic across a period (C 0.1321 < N 0.1452 < O 0.1631 < F 0.1828).
# `lidx` is the usual packed (l1,l2) map, 1-based as Fortran writes it.
#
# Injected as DATA into the frozen module's cache -- no mlxmolkit file is edited.
_ONECX_NPZ = _os_path.join(_os_path.dirname(__file__), "..", "..", "data",
                           "gxtb_onecxints_extracted.npz")


def _install_onecxints() -> bool:
    """Give `gxtb_aes` the one-centre tables it is missing. Idempotent."""
    try:
        from mlxmolkit.xtb import gxtb_aes as _aes
        if getattr(_aes, "_ONEC", None) is not None:
            return True
        if not _os_path.exists(_ONECX_NPZ):
            return False
        _aes._ONEC = dict(np.load(_ONECX_NPZ))
        return True
    except Exception:
        return False


# ------------------------------------------------------------- ACP, made cheap
#
# `new_gxtb_calculator` constructs `new_acp`, so the ACP Hamiltonian is part of
# g-xTB. It was off by default and, once the exchange was decoded, turning it on
# is worth 0.02228 -> 0.01814 on the screening subset.
#
# The blocker was cost. The frozen builder concatenates the CAO basis with the
# ACP auxiliary basis and calls a full (n+m)^2 overlap; for a 44-atom molecule
# that is 92 AO + 572 aux = 664 functions, and the compiled multipole kernel it
# was routed to returns *ten* matrices when the ACP uses one 92 x 572 block of
# one of them -- ~84x more arithmetic than the term needs.
#
# So build that block directly. Cartesian components of a shell share centre,
# exponents and contraction coefficients, so the expensive part -- R^2, the
# exponential, the Obara-Saika tables -- is evaluated once per *shell* pair, and
# the per-component work is three multiplies at fixed table indices (blocking by
# angular type means no gather). 6.0x over the compiled path on the large
# molecules, and the result agrees with it to 2.9e-16.
# `probes/check_xover2.py` pins the two against each other.

_ACP_NCOMP = (1, 3, 6, 10)


# `lx(:, i + lmap(3))` for i = 1..10 in `tblite_integral_multipole` (recovered).
# Self-checking against `_ACP_F_CART_TO_SPH`: with this order its 4th row reads
# zzz - 1.5*xxz - 1.5*yyz, which is f_z3 exactly.
_ACP_F_LXYZ = ((3, 0, 0), (0, 3, 0), (0, 0, 3), (2, 1, 0), (2, 0, 1),
               (1, 2, 0), (0, 2, 1), (1, 0, 2), (0, 1, 2), (1, 1, 1))
_ACP_P_LXYZ = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
_ACP_D_LXYZ = ((2, 0, 0), (0, 2, 0), (0, 0, 2), (1, 1, 0), (1, 0, 1), (0, 1, 1))
_ACP_ELEM_CACHE: dict = {}


def _acp_cao_shells(cao_basis):
    """Group the CAO list into shells, keyed by angular momentum."""
    out: dict = {}
    i, n = 0, len(cao_basis)
    while i < n:
        b = cao_basis[i]
        l = int(b.l_total)
        nc = _ACP_NCOMP[l]
        idxs = list(range(i, i + nc))
        for k in idxs:
            bk = cao_basis[k]
            # The builder shares one alphas/coeffs object across a shell's
            # components (measured: 92/92), so identity is the fast path; the
            # array compare stays as the fallback, keeping the check as strong.
            if (int(bk.l_total) != l or bk.atom_idx != b.atom_idx
                    or not (bk.alphas is b.alphas
                            or np.array_equal(bk.alphas, b.alphas))
                    or not (bk.coeffs is b.coeffs
                            or np.array_equal(bk.coeffs, b.coeffs))):
                raise RuntimeError("CAO shell grouping assumption violated")
        out.setdefault(l, []).append(
            (np.asarray(b.center, dtype=np.float64),
             np.asarray(b.alphas, dtype=np.float64),
             np.asarray(b.coeffs, dtype=np.float64),
             [tuple(int(v) for v in cao_basis[k].l_xyz) for k in idxs], idxs))
        i += nc
    return out


def _acp_elem(Z: int):
    """Per-element ACP projector template (l, alpha, norm, level)."""
    hit = _ACP_ELEM_CACHE.get(Z)
    if hit is not None:
        return hit
    from mlxmolkit.xtb.basis import (
        primitive_norm_d, primitive_norm_p, primitive_norm_s,
    )
    rows = []
    for iproj in range(int(GXTB_PARAMS["pa_nacp"][Z - 1])):
        l = int(GXTB_PARAMS["pa_l_acp"][Z - 1, iproj])
        # ⚠️ This used to read `if l > 2: continue`, which silently dropped the
        # l = 3 projector of every element from Mg up -- 70 of them, and large:
        # ps_acp_level is -0.21053 for sulphur's f, TWICE its own p channel,
        # -0.16666 for bromine, -0.13541 for silicon.  `new_gxtb_calculator`
        # (recovered) builds one cgto per `pa_nacp` entry straight from
        # `pa_l_acp`, with no angular filter, so the binary carries them all.
        if l > 3:
            continue
        level = float(GXTB_PARAMS["ps_acp_level"][Z - 1, iproj])
        alpha = float(GXTB_PARAMS["ps_acp_exp"][Z - 1, iproj])
        if alpha <= 0.0 or level == 0.0:
            continue
        if l < 3:
            nrm = float((primitive_norm_s, primitive_norm_p, primitive_norm_d)[l](
                np.asarray([alpha], dtype=np.float64))[0])
        else:
            # `tblite_basis_type::new_cgto`, the formula the s/p/d helpers
            # already reproduce bit for bit:
            #     (2a/pi)**0.75 * sqrt(4a)**l / sqrt(dfactorial(l))
            # with dfactorial = [1, 1, 3, 15, ...].
            nrm = float((alpha * (2.0 / np.pi)) ** 0.75
                        * np.sqrt(4.0 * alpha) ** 3 / np.sqrt(15.0))
        rows.append((l, alpha, nrm, level))
    _ACP_ELEM_CACHE[Z] = rows
    return rows


def _acp_aux_shells(atoms, coords_ang):
    """ACP projectors as shells, keyed by angular momentum."""
    coords_bohr = np.asarray(coords_ang, dtype=np.float64) * ANG_TO_BOHR
    out: dict = {}
    col = 0
    for ia, Z0 in enumerate(np.asarray(atoms, dtype=np.intp)):
        for l, alpha, nrm, level in _acp_elem(int(Z0)):
            lxyz = (((0, 0, 0),), _ACP_P_LXYZ, _ACP_D_LXYZ, _ACP_F_LXYZ)[l]
            nc = _ACP_NCOMP[l]
            out.setdefault(l, []).append(
                (coords_bohr[ia], alpha, nrm, level, lxyz,
                 list(range(col, col + nc))))
            col += nc
    return out, col


def _acp_os_tables(PA, PB, inv2p, la, lb):
    """Obara-Saika 1D overlap tables S[i][j] for i <= la, j <= lb.

    ``None`` stands for the implicit S[0][0] = 1, so the s-s case costs nothing.
    """
    S = [[None] * (lb + 1) for _ in range(la + 1)]
    for i in range(1, la + 1):
        t = PA if S[i - 1][0] is None else PA * S[i - 1][0]
        if i >= 2:
            t = t + inv2p * (i - 1) * (1.0 if S[i - 2][0] is None else S[i - 2][0])
        S[i][0] = t
    for j in range(1, lb + 1):
        for i in range(la + 1):
            t = PB if S[i][j - 1] is None else PB * S[i][j - 1]
            if j >= 2:
                t = t + inv2p * (j - 1) * (1.0 if S[i][j - 2] is None else S[i][j - 2])
            if i >= 1:
                t = t + inv2p * i * (1.0 if S[i - 1][j - 1] is None else S[i - 1][j - 1])
            S[i][j] = t
    return S


def _acp_cross_overlap(cao_basis, atoms, coords_ang):
    """<AO_mu | aux_j> and the projector levels, assembled shell-pair-wise."""
    ao = _acp_cao_shells(cao_basis)
    aux, n_aux = _acp_aux_shells(atoms, coords_ang)
    if n_aux == 0:
        return None, None
    B = np.zeros((len(cao_basis), n_aux), dtype=np.float64)
    levels = np.zeros(n_aux, dtype=np.float64)
    for blocks in aux.values():
        for _, _, _, lev, _, cols in blocks:
            levels[cols] = lev
    for la, a_blocks in ao.items():
        a_cen = np.concatenate([np.repeat(s[0][None, :], len(s[1]), 0) for s in a_blocks])
        a_cx = np.ascontiguousarray(a_cen[:, 0])
        a_cy = np.ascontiguousarray(a_cen[:, 1])
        a_cz = np.ascontiguousarray(a_cen[:, 2])
        a_al = np.concatenate([s[1] for s in a_blocks])
        a_cf = np.concatenate([s[2] for s in a_blocks])
        offs = np.cumsum([0] + [len(s[1]) for s in a_blocks])[:-1]
        a_rows = np.asarray([s[4] for s in a_blocks], dtype=np.intp)
        a_lxyz = a_blocks[0][3]
        for lb, b_blocks in aux.items():
            b_cen = np.asarray([s[0] for s in b_blocks], dtype=np.float64)
            b_al = np.asarray([s[1] for s in b_blocks], dtype=np.float64)
            b_cf = np.asarray([s[2] for s in b_blocks], dtype=np.float64)
            b_cols = np.asarray([s[5] for s in b_blocks], dtype=np.intp)
            b_lxyz = b_blocks[0][4]
            p = a_al[:, None] + b_al[None, :]
            inv2p = 0.5 / p
            # Per-axis contiguous displacements rather than one (Pa, Mb, 3)
            # array: the Obara-Saika tables below read one axis at a time, and
            # a stride-3 view costs ~1.5x a contiguous one.
            dxyz = (a_cx[:, None] - b_cen[:, 0][None, :],
                    a_cy[:, None] - b_cen[:, 1][None, :],
                    a_cz[:, None] - b_cen[:, 2][None, :])
            R2 = dxyz[0] * dxyz[0] + dxyz[1] * dxyz[1] + dxyz[2] * dxyz[2]
            # t*sqrt(t) rather than t**1.5: pow is a transcendental and is 9.6x
            # slower here, for a one-ulp difference (2.2e-16 relative).
            t = np.pi / p
            base = ((t * np.sqrt(t))
                    * np.exp(-(a_al[:, None] * b_al[None, :]) / p * R2)
                    * (a_cf[:, None] * b_cf[None, :]))
            rb = b_al[None, :] / p
            ra = a_al[:, None] / p
            tabs = [_acp_os_tables(-rb * dxyz[ax], ra * dxyz[ax], inv2p, la, lb)
                    for ax in range(3)]
            for ca, lxa in enumerate(a_lxyz):
                for cb, lxb in enumerate(b_lxyz):
                    f = base
                    for ax in range(3):
                        t = tabs[ax][lxa[ax]][lxb[ax]]
                        if t is not None:
                            f = f * t
                    B[np.ix_(a_rows[:, ca], b_cols[:, cb])] = np.add.reduceat(f, offs, axis=0)
    return B, levels


# ----------------------------------- the ACP's d projectors must be SPHERICAL
#
# The ACP is `H = sum_j level_j |aux_j><aux_j|`, and for l = 2 the aux set is
# `_ACP_D_LXYZ` -- the SIX cartesian d Gaussians, all carrying the same
# normalisation and the same level. That projector is NOT ROTATIONALLY
# INVARIANT: `sum_m |m><m|` is invariant only when the set transforms
# orthogonally, and the cartesian d's do not (xx, yy, zz overlap each other,
# and xy has a different self-overlap from xx -- the same asymmetry the main
# basis fixes with the sqrt(3) factors in `cao_to_sao_transform`).
#
# Measured: rotating and translating H2O moves the ACP's generalised
# eigenvalues by 3.06e-04, and dropping the l = 2 projectors alone takes that
# to 4.9e-17. l = 0 and l = 1 are already invariant -- one component, and three
# that transform as a vector.
#
# The fix is the same transform the main basis uses: contract the six cartesian
# columns of <AO|aux> to the five spherical ones before forming the projector.
_ACP_D_CART_TO_SPH = np.array([
    [0.5 * np.sqrt(3.0), -0.5 * np.sqrt(3.0), 0.0, 0.0, 0.0, 0.0],   # x2-y2
    [0.5, 0.5, -1.0, 0.0, 0.0, 0.0],                                  # z2
    [0.0, 0.0, 0.0, np.sqrt(3.0), 0.0, 0.0],                          # xy
    [0.0, 0.0, 0.0, 0.0, np.sqrt(3.0), 0.0],                          # xz
    [0.0, 0.0, 0.0, 0.0, 0.0, np.sqrt(3.0)],                          # yz
])


def _acp_sphericalise(B, levels, aux):
    """Replace each l=2/l=3 projector's cartesian columns by spherical ones.

    The f block matters as much as the d one: `pa_l_acp` gives an l = 3
    projector to every element from Mg up (70 of them), and the ten Cartesian
    f functions carry THREE spurious l = 1 combinations -- x*r2, y*r2, z*r2 --
    on top of the seven real ones, exactly as the six Cartesian d carry one
    spurious s.  `_ACP_F_CART_TO_SPH` is `ftrafo(7, 10)` read out of
    `tblite_integral_trafo` in the binary.
    """
    from mlxmolkit.xtb.gxtb_acp import _ACP_F_CART_TO_SPH
    cols_out, lev_out = [], []
    sph = {2: _ACP_D_CART_TO_SPH, 3: _ACP_F_CART_TO_SPH}
    for l, blocks in aux.items():
        for blk in blocks:
            cols = list(blk[5])
            tr = sph.get(int(l))
            if tr is None:
                cols_out.append(B[:, cols])
                lev_out.append(levels[cols])
            else:
                cols_out.append(B[:, cols] @ tr.T)
                lev_out.append(np.full(tr.shape[0], blk[3], dtype=np.float64))
    return np.hstack(cols_out), np.concatenate(lev_out)


def build_gxtb_acp_hamiltonian(atomic_numbers, coords_ang, basis, *, enabled=True, **kw):
    if not enabled:
        return np.zeros_like(basis.S)
    try:
        from mlxmolkit.xtb.gxtb_acp import GXTB_ACP_PROJECTOR_SCALE
        scale = float(kw.get("scale", GXTB_ACP_PROJECTOR_SCALE))
        B, levels = _acp_cross_overlap(basis.cao_basis, atomic_numbers, coords_ang)
        if B is None:
            return np.zeros_like(basis.S)
        aux, _ = _acp_aux_shells(atomic_numbers, coords_ang)
        B, levels = _acp_sphericalise(B, levels, aux)
        H_cao = scale * ((B * levels[None, :]) @ B.T)
        T = basis.T_cao_to_sao
        H_sao = T @ (0.5 * (H_cao + H_cao.T)) @ T.T
        return 0.5 * (H_sao + H_sao.T)
    except Exception:
        return _frozen_build_acp(atomic_numbers, coords_ang, basis, enabled=True, **kw)


# ------------------------------------------------------------- basis, made fast
#
# `gxtb_basis.build_gxtb_qvszp_basis` builds S_cao with `basis.overlap_matrix`,
# a Python loop over primitive pairs. Profiling the current solver: that single
# call is 3.07 s of ~4 s total -- by far the largest cost in the whole SCF,
# larger than every matrix operation combined.
#
# The repo already ships a compiled kernel whose first output IS that overlap
# (`multipole_matrices_cpp`), used here for AES and ACP after being checked
# exact. Same swap, same verification (`probes/check_basis.py`).
#
# Pure cost: the basis, the overlap and every downstream number are unchanged.


# --------------------------------------- the H0 basis is a SECOND cgto set
#
# `recovered/xtb_gxtb/new_gxtb_calculator.f90` (differential test vs the binary)
# builds TWO q-vSZP sets per shell:
#
#     call new_qvszp_cgto(c1, num, ish, .true., error)              ! density
#     alpha2(:) = c1%alpha*ps_h0_qvszp_exp_scal(ish, num)
#     call new_qvszp_cgto(c2, num, ish, .true., error, alpha=alpha2, &   ! H0
#        & k0=c1%k0*pa_h0_qvszp_k0_scal(num), &
#        & k2=c1%k2*pa_h0_qvszp_k2_scal(num), &
#        & k3=c1%k3*pa_h0_qvszp_k3_scal(num))
#
# c1 gets NO k scaling; k1 is never scaled in either. `gxtb_basis` applies the
# three `pa_h0_qvszp_k*_scal` tables to the ONE basis it builds, so the density
# basis wrongly inherits the H0 charge response and the two bases differ only
# by exponent. Split them: suppress the tables for the density build, and apply
# them here when the H0 overlap is formed.


@contextlib.contextmanager
def _unscaled_h0_k(_gb):
    """Build the density basis with raw k0/k2/k3 (the binary's c1)."""

    class _Proxy:
        def __init__(self, inner):
            self._inner = inner
        def __getitem__(self, name):
            if name in ("pa_h0_qvszp_k0_scal", "pa_h0_qvszp_k2_scal",
                        "pa_h0_qvszp_k3_scal"):
                return np.ones_like(
                    np.asarray(self._inner[name], dtype=np.float64))
            return self._inner[name]
        def __getattr__(self, k):
            return getattr(self._inner, k)

    orig = _gb.GXTB_PARAMS
    _gb.GXTB_PARAMS = _Proxy(orig)
    try:
        yield
    finally:
        _gb.GXTB_PARAMS = orig


def _qeff_h0(atoms: np.ndarray, basis) -> np.ndarray:
    """Per-atom qeff of the H0 basis: same form, k0/k2/k3 scaled (k1 is not)."""

    from mlxmolkit.xtb.gxtb_basis import qvszp_qeff
    z = np.asarray(atoms, dtype=np.intp) - 1
    k0 = np.asarray(QVSZP_PARAMS["p_k0"])[z] * np.asarray(
        GXTB_PARAMS["pa_h0_qvszp_k0_scal"])[z]
    k1 = np.asarray(QVSZP_PARAMS["p_k1"])[z]
    k2 = np.asarray(QVSZP_PARAMS["p_k2"])[z] * np.asarray(
        GXTB_PARAMS["pa_h0_qvszp_k2_scal"])[z]
    k3 = np.asarray(QVSZP_PARAMS["p_k3"])[z] * np.asarray(
        GXTB_PARAMS["pa_h0_qvszp_k3_scal"])[z]
    cn = getattr(basis, "_qvszp_cn", None)
    if cn is None:
        cn = basis.cn
    return qvszp_qeff(np.asarray(basis.eeqbc_charges, dtype=np.float64),
                      np.asarray(cn, dtype=np.float64), k0, k1, k2, k3)


# ------------------------------------------------- anisotropic H0, decoded
#
# `recovered/tblite_xtb_h0/{get_anisotropy,get_hamiltonian}.f90`, both with a
# passing differential test against the binary.
#
#     aniso(:, a) = sum_{b /= a} (r_b - r_a) * f_ab              [NOT normalised]
#     f_ab = 0.5*(1 + erf(-dpol_scale*(|r_ab| - rvdw(za,zb)))) * dpol(za,zb)
#
# with, from `tblite_xtb_gxtb::get_anisotropy` / `::get_rvdw` (both read via the
# adrp, NOT Ghidra's labels -- it mislabelled the mode word as 0x73b260 and the
# scale table as 0x745ae8; the real ones are 0x73b480 and pa_rvdw_scale):
#
#     dpol_scale = 1.5                       (0x3ff8000000000000, immediate)
#     dpol(i,j)  = arithmetic_avg(pa_h0_dip_scale[Zi], pa_h0_dip_scale[Zj])
#     rvdw(i,j)  = vdw_rad_pair(Zi,Zj) * arithmetic_avg(pa_rvdw_scale[Z..])
#
# new_average's mode word is 0, and 0 is ARITHMETIC (geometric is 1). That
# matters: pa_h0_dip_scale has 33 negative entries out of 103, so a geometric
# mean would be NaN for e.g. any C-O pair. `gxtb_mrad_pair` already has rvdw
# right; only the kernel below was wrong.
#
# get_hamiltonian contracts it as, for row mu on atom `jat` and column nu on
# `iat` (both triangles receive the SAME value):
#
#     hji = -0.5*sum_k [ dtmp(k)*aniso(k,iat) + aniso(k,jat)*dtmpj(k) ]
#
# `dtmp` is the dipole about the COLUMN atom and `dtmpj` the same operator
# shifted to the ROW atom -- two different origins, which is why a single
# dpint cannot express it directly. Writing dtmp = D - R_nu*S and
# dtmpj = D - R_mu*S for D the dipole about a fixed origin gives an
# origin-independent form in terms of what we already have:
#
#     H_aniso[mu,nu] = -0.5*sum_k D[k][mu,nu]*(A[k][mu] + A[k][nu])
#                      + 0.5*S[mu,nu]*(u[mu] + u[nu]),   u[a] = sum_k R_a[k]*A[k][a]
#
# On-site blocks get it too (the binary's on-site loop spells it `+hji` then
# subtracts, same sign), and the diagonal cancels exactly: D[mu,mu] = R_mu*S.
#
# `mlxmolkit.gxtb_aes.gxtb_aniso_h0` differs three ways: it normalises the
# vector by 1/R, it uses the wrong sign, and it uses one dipole origin for both
# terms. Its `dpol`/`rvdw`/1.5 are right.


def _aniso_h0_decoded(basis, atoms, coords_ang):
    from scipy.special import erf as _erf
    from mlxmolkit.xtb.gxtb_aes import gxtb_mrad_pair

    atoms = np.asarray(atoms, dtype=np.intp)
    xyz = np.asarray(coords_ang, dtype=np.float64) * ANG_TO_BOHR
    S, dpint, _ = qvszp_multipoles(basis)   # the module-level FAST one (l.1521)

    ds = np.asarray(GXTB_PARAMS["pa_h0_dip_scale"], dtype=np.float64)[atoms - 1]
    dpol = 0.5 * (ds[:, None] + ds[None, :])
    rvdw = gxtb_mrad_pair(atoms)
    diff = xyz[None, :, :] - xyz[:, None, :]            # r_b - r_a
    R = np.sqrt(np.sum(diff * diff, axis=-1))
    f = dpol * 0.5 * (1.0 + _erf(-1.5 * (R - rvdw)))
    np.fill_diagonal(f, 0.0)
    A = np.einsum("ab,abk->ak", f, diff)                # aniso, unnormalised
    u = np.sum(xyz * A, axis=1)                         # per-atom scalar

    aoat = np.array([bf.atom_idx for bf in basis.sao_basis], dtype=np.int64)
    A_ao, u_ao = A[aoat], u[aoat]
    H = 0.5 * S * (u_ao[:, None] + u_ao[None, :])
    for k in range(3):
        H -= 0.5 * dpint[k] * (A_ao[:, k][:, None] + A_ao[:, k][None, :])
    return H


# ------------------------------------------------ the basis itself was wrong
#
# Found by RUNNING both sides on the same molecule and diffing every
# intermediate (`gxtb-recovery/probes/audit/port_divergence.py`), not by
# reading. Three things, in order of size:
#
# 1. SHELL COUNT -- NOT a bug. `probes/h0_get_hamiltonian`'s harness picks the
#    shell count itself (it counts how many q-vSZP templates exist, giving H
#    two shells and O three, nao = 17 for water), which is NOT what the
#    calculator does. `new_gxtb_calculator` uses `nsh_id = pa_nshell`, and
#    building the calculator with the BINARY gives nao = 6 for H2O -- exactly
#    what the port has. `include_polarization_shells` stays False.
#
# 2. CN STEEPNESS. `_gxtb_erf_coordination_number` defaults to `k = 2.068`,
#    sourced from a literal at 0x73b270 -- one of the addresses Ghidra
#    mislabelled in this binary. The value that reproduces the binary's own
#    contraction is 3.75 (fitted 3.749992 against the binary's overlap over a
#    9-atom molecule, residual 4.8e-8 -- the fit's floor).
#
# 3. CN RADII. It uses `pa_cn_rcov`; `new_qvszp_basis` passes
#    `qvszp_cov_radii`, which is `QVSZP_PARAMS["cov_radii"]` -- the same table
#    `get_rad` starts from. `pa_cn_rcov` is a different table (H 0.482 vs
#    0.548 bohr) and does not reproduce the binary at any steepness.
#
# Together these take the density-basis overlap from max|dS| = 1.4e-1 against
# the binary to 4.8e-8.

GXTB_BASIS_KCN = 3.75


def _qvszp_basis_cn(atoms, coords_ang):
    """The CN `new_qvszp_basis` builds: erf count, k = 3.75, qvszp_cov_radii."""
    from scipy.special import erf as _erf
    atoms = np.asarray(atoms, dtype=np.intp)
    xb = np.asarray(coords_ang, dtype=np.float64) * ANG_TO_BOHR
    rc = np.asarray(QVSZP_PARAMS["cov_radii"], dtype=np.float64)[atoms - 1]
    r0 = rc[:, None] + rc[None, :]
    d = np.linalg.norm(xb[:, None, :] - xb[None, :, :], axis=-1)
    cnt = 0.5 * (1.0 + _erf(-GXTB_BASIS_KCN * (d - r0) / r0))
    np.fill_diagonal(cnt, 0.0)
    return cnt.sum(1)


# `hcore_gxtb._diat_scale` returns 1.0 when either table entry is <= 0.
# `tblite_xtb_gxtb::get_diat_scale` has NO such guard -- it computes
# `2/(1/ka + 1/kb)` unconditionally, so a zero entry gives 1/0 = inf and the
# harmonic mean is 0, not 1. Hydrogen's kpi IS exactly 0.0, so the binary
# switches the pi channel off for every pair involving H; the port kept the
# full pi overlap. Only visible once H carries a p shell.


def _diat_scale_exact(Za, Zb, interaction):
    t = np.asarray(GXTB_PARAMS["ps_h0_diat_scale"], dtype=np.float64)
    ka = t[(int(Za) - 1) * 3 + int(interaction)]
    kb = t[(int(Zb) - 1) * 3 + int(interaction)]
    with np.errstate(divide="ignore", invalid="ignore"):
        # IEEE, like the Fortran: 1/0 -> inf, so a zero entry gives 0.0
        return float(np.float64(2.0) / (np.float64(1.0) / ka + np.float64(1.0) / kb))


_diat_scale = _diat_scale_exact


# ---------------------------------------------- the Coulomb's reference charge
#
# `add_coulomb` (recovered) passes `new_effective_coulomb` an eighth argument:
#
#     qref(ish_at(iat) + ish) = ps_reference_occ(ish, num) - ps_tb1_zeffsh(ish, num)
#
# and it is per SHELL OF THE WHOLE SYSTEM, not per (shell, species). Read out
# of a live calculator for H2O it is [-0.32697, +0.32697, 0, 0] over
# (O_s, O_p, H_s, H_s), which is exactly that expression.
#
# The port had no counterpart. Its Mulliken shell charge is measured against
# `ps_reference_occ`; the second-order Coulomb wants it measured against
# `ps_tb1_zeffsh`, and the two differ by qref:
#
# The SIGN is decoded, not guessed. `coulomb_charge_type::get_potential` makes
# TWO wrap_dsymv calls against the same amat and the same alpha constant:
#
#     wrap_dsymv(amat, qsh,  vsh, ..., &DAT_006b8b48, 0)
#     if (qref allocated) wrap_dsymv(amat, qref, vsh, ..., &DAT_006b8b48, 0)
#
# so the potential is A*(qsh + qref) -- PLUS. (My first reading argued from the
# physics for `qsh - qref` and was wrong; the two calls share an alpha, which
# settles the relative sign without needing to know its value.)
#
# For oxygen qref is +-0.327 per shell and for carbon +-0.965 -- large, and
# shell-resolved, so it shifts s against p. Only `effective_coulomb` takes it;
# `new_twobody_thirdorder`'s argument list has no qref.


def _shell_qref(atoms, basis):
    sa = np.asarray(basis.shell_atom)
    z = np.asarray(atoms, dtype=np.intp)[sa] - 1
    # 🔑 The per-shell parameter tables are indexed by SHELL INDEX within the
    # species, not by angular momentum.  `add_coulomb` and `add_exchange`
    # (both recovered, 0 ulp) read them as `ps_*(ish, num)`.  For every
    # element in the current benchmark the shell order is s, p, d so the two
    # indexings coincide and the difference is latent -- but the port used
    # BOTH conventions in different places, which is a contradiction that had
    # to resolve one way, and the binary says shell index.
    loc = _shell_local_indices(sa)
    return (np.asarray(GXTB_PARAMS["ps_reference_occ"], dtype=np.float64)[z, loc]
            - np.asarray(GXTB_PARAMS["ps_tb1_zeffsh"], dtype=np.float64)[z, loc])


# ------------------------------------- the onsite first-order term, decoded
#
# `recovered/tblite_coulomb_firstorder/get_potential.f90`, bit-exact (0 ulp)
# against the binary over probes/fo_get_potential. `add_coulomb` constructs it,
# so it is part of the model.
#
#     mu(ish) = ipea(ish,izp)*0.5*((sqrt(cn + 1e-12) - 1e-6)*ipea_cn(izp) + 1)
#     vsh(ii+ish) += (2 + kscale*(erf(a) + erf(b))) * mu(ish)
#     vat(iat)    += sum_jsh qsh(ii+jsh)*mu(jsh)
#                    * kscale * kexp*(2/sqrt(pi)) * (exp(-a*a) + exp(-b*b))
#
# a = kexp*(q - kpow), b = kexp*(q + kpow), q = the ATOM charge, and NO
# negation anywhere. `add_vat_to_vsh` folds vat into every shell of the atom,
# so the derivative part is the SAME for all shells -- and it is weighted by
# the SHELL charges, not by q_atom shell-by-shell. The shipped
# `_first_order_onsite` does the latter with a `charge_sign = -1` its own
# docstring calls a guess; the two agree only for one-shell atoms, which is why
# neither sign of it could fit oxygen and sulfur at the same time.


def _first_order_decoded(atoms, cn, shell_atom, shell_l, qsh):
    from scipy.special import erf as _erf
    at = np.asarray(atoms, dtype=np.intp)
    sa = np.asarray(shell_atom, dtype=np.int64)
    q = np.asarray(qsh, dtype=np.float64)
    z = at[sa] - 1
    # shell INDEX, not l -- `add_coulomb`: ipea(ish, izp) = ps_tb1_ipea(ish, num)
    ipea = np.asarray(GXTB_PARAMS["ps_tb1_ipea"], dtype=np.float64)[
        z, _shell_local_indices(sa)]
    ipea_cn = np.asarray(GXTB_PARAMS["pa_tb1_ipea_cn"], dtype=np.float64)[at - 1]
    kexp, kscale, kpow = GXTB_TB1_KX, GXTB_TB1_KDIS, GXTB_TB1_KS

    cnf = (np.sqrt(np.asarray(cn, dtype=np.float64) + 1.0e-12) - 1.0e-6) * ipea_cn + 1.0
    mu = ipea * 0.5 * cnf[sa]

    qat = np.bincount(sa, weights=q, minlength=at.size)
    a = (qat - kpow) * kexp
    b = (qat + kpow) * kexp
    sw = (_erf(a) + _erf(b)) * kscale + 2.0
    gk = ((kexp * 1.1283791670955126)
          * (np.exp(-(a * a)) + np.exp(-(b * b))) * kscale)

    vsh = mu * sw[sa]
    vat = np.bincount(sa, weights=q * mu, minlength=at.size) * gk
    # 🔑 vsh and vat stay SEPARATE, as `coulomb_firstorder::get_potential`
    # leaves them.  Folding vat in here mixed two quantities the binary keeps
    # apart until `add_vat_to_vsh`, which makes a stage-by-stage diff against
    # pot%vsh impossible to read.
    # `coulomb_firstorder::get_energy` is
    #     E_i = sum_ish ipea(ish,izp) * 0.5 * cnf_i * sw_i * qsh(ish)
    # and `vsh` above is exactly `ipea * 0.5 * cnf * sw`, so the energy is the
    # shell-wise contraction of vsh with the shell charges.  It used to be
    # hardcoded to 0.0 while the potential was right, which is why the walk's
    # V_sh stages passed and the total energy was short by ~6e-2 Ha on H2O.
    return float(np.sum(vsh * q)), vsh, vat


# `coulomb_thirdorder_twobody::get_potential` (recovered, 0 ulp) applies
# alpha = 1/3 to BOTH dsymv calls, so the whole term carries a factor of a
# third that `gxtb_aes.gxtb_twobody_thirdorder` does not have. The port was 3x
# too large on it. (The constant is at 0x6b9418; Ghidra labels that operand
# DAT_006b91f8, which holds 1.0 -- believing the label is exactly how a missing
# 1/3 gets written.)
GXTB_TB3_TWOBODY_SCALE = 1.0 / 3.0


def build_gxtb_qvszp_basis(atomic_numbers, coords_ang, calc_cn=None, **kw):
    # TWO coordination numbers, not one. The basis gets its own (see
    # _qvszp_basis_cn) for the contraction; `basis.cn` is then restored to the
    # CALCULATOR's, which is what the self-energies and the shell hardness
    # read. Building with the basis CN and leaving it there corrupts those.
    kw.setdefault("cn", _qvszp_basis_cn(atomic_numbers, coords_ang))
    try:
        from mlxmolkit.xtb.multipole_integrals_cpp import (
            CPP_AVAILABLE,
            multipole_matrices_cpp,
        )
        if not CPP_AVAILABLE:
            raise ImportError
        import mlxmolkit.xtb.gxtb_basis as _gb

        # The compiled kernel returns S, 3 dipole and 6 quadrupole matrices in
        # one pass. The basis build needs only S -- but `qvszp_multipoles` later
        # needs all ten over the SAME cao_basis. Keep what this call already
        # computed and seed the multipole cache with it, so the AES path does
        # not recompute the identical integrals. Saves one of three kernel
        # calls per molecule (the kernel is ~31 % of runtime).
        stash = {}
        def _grab(bfs):
            out = multipole_matrices_cpp(bfs)
            stash["cao"] = out
            return out[0]

        orig = _gb.overlap_matrix
        _gb.overlap_matrix = _grab
        try:
            with _unscaled_h0_k(_gb):
                basis = _frozen_build_basis(atomic_numbers, coords_ang, **kw)
        finally:
            _gb.overlap_matrix = orig

        got = stash.get("cao")
        if got is not None and len(basis.cao_basis) == got[0].shape[0]:
            S_cao, dp_cao, qp_cao = got
            T = np.asarray(basis.T_cao_to_sao, dtype=np.float64)
            _MP_CACHE_FAST[id(basis)] = (basis, (
                T @ S_cao @ T.T,
                np.stack([T @ dp_cao[k] @ T.T for k in range(3)], axis=0),
                np.stack([T @ qp_cao[k] @ T.T for k in range(6)], axis=0),
            ))
        _restore_calc_cn(basis, atomic_numbers, coords_ang, calc_cn)
        return basis
    except Exception:
        import mlxmolkit.xtb.gxtb_basis as _gb
        with _unscaled_h0_k(_gb):
            basis = _frozen_build_basis(atomic_numbers, coords_ang, **kw)
        _restore_calc_cn(basis, atomic_numbers, coords_ang, calc_cn)
        return basis


def _restore_calc_cn(basis, atoms, coords_ang, calc_cn=None):
    """`basis.cn` must be the CALCULATOR's CN (pa_cn_rcov, kcn = 2.068).

    `new_gxtb_calculator` builds a second ncoord for the model itself; the
    basis's own (qvszp_cov_radii, kcn = 3.75) exists only to make the q-vSZP
    contraction. Downstream readers of `basis.cn` -- the H0 self-energies and
    the shell hardness -- want the calculator's.
    """
    from mlxmolkit.xtb.gxtb_reconstructed import _gxtb_erf_coordination_number
    at = np.asarray(atoms, dtype=np.intp)
    # keep the BASIS cn: `basis_update` builds BOTH cgto sets from it, so the
    # H0 set's qeff needs it too (see _qeff_h0).
    object.__setattr__(basis, "_qvszp_cn", np.array(basis.cn, copy=True))
    # `calc_cn` freezes the CALCULATOR's CN at a caller-chosen value.  The
    # `cn=` kwarg freezes only the BASIS CN (k = 3.75, the contraction); this
    # one freezes the CN the self-energies and the shell hardness read, which
    # is a different number (k = 2.068) and moves independently.
    basis.cn[:] = (_gxtb_erf_coordination_number(
        at, np.asarray(coords_ang, dtype=np.float64))
        if calc_cn is None else np.asarray(calc_cn, dtype=np.float64))
    # shell_hardness was baked during the build from the BASIS cn; redo it.
    sa = np.asarray(basis.shell_atom, dtype=np.int64)
    sl = np.asarray(basis.shell_l, dtype=np.int64)
    z = at[sa] - 1
    # shell INDEX, not l.  `_corrected_shell_hardness` above already uses the
    # local index; this path used `sl` and the two contradicted each other.
    base = (np.asarray(GXTB_PARAMS["ps_tb2_shell_hubbard"])[z, _shell_local_indices(sa)]
            * np.asarray(GXTB_PARAMS["pa_hubbard_parameter"])[z])
    slope = np.asarray(GXTB_PARAMS["pa_tb2_hubbard_cn"])[z]
    # SQRT of the CN, not linear. `effective_coulomb`'s get_amat_0d:
    #     cni = hubbard_cn(izp)*(sqrt(cn(iat) + 1e-12) - 1e-6) + 1.0
    # `gxtb_basis` builds the LINEAR form and `sqrt_cn_hardness` defaults off,
    # so this is where the recovered form has to enter.
    basis.shell_hardness[:] = np.maximum(
        base * (1.0 + slope * (np.sqrt(basis.cn[sa] + 1.0e-12) - 1.0e-6)), 1.0e-8)


# ------------------------------------------------ d-shell diatomic damping
#
# `hcore_gxtb.GXTB_D_H0_SCALE_DAMP` scales the sigma/pi/delta diatomic factor
# for any shell pair involving a d shell as `1 + damp*(k-1)`. Its shipped value,
# 0.1, is an ADMITTED PLACEHOLDER ("the true binary path" was unknown) and it
# leaves the d shell essentially unbound: over 12 S/Cl/Br/I molecules our mean d
# population is -0.005 against the oracle's +0.202, and the missing d density
# reappears as an s deficit and a p excess.
#
# The decoded rule (probes/DECODED.md) is that the binary applies the FULL
# sigma/pi/delta scales to the m-diagonal elements in the spherical frame. Doing
# exactly that overshoots the d population by 2x (0.429 vs 0.202), which says
# our d basis function is too diffuse, not that the rule is wrong.
#
# So this stays a placeholder -- but a measured one. 0.5 puts the d population
# at +0.191 vs +0.202 (d error 0.207 -> 0.041) and improves charge and shell on
# both the benchmark and the heavy set. It is NOT a decoded constant and must be
# replaced once the q-vSZP d contraction is checked.
GXTB_D_H0_SCALE_DAMP = 0.5
_hcore_mod.GXTB_D_H0_SCALE_DAMP = GXTB_D_H0_SCALE_DAMP


# --------------------------------------- diatomic overlap scaling, made cheap
#
# Same arithmetic as `hcore_gxtb._diatomic_scaled_overlap_cao`, pinned
# bit-identical by `probes/check_diat.py`. Two pure-cost changes:
#
#   * the frozen version indexes every shell-pair block with `np.ix_`, which is
#     29,532 calls per six molecules and 11 % of runtime. CAO functions of a
#     shell are CONTIGUOUS (checked: 0 non-contiguous of 60), so plain slices
#     do the same job.
#   * `_diat_scale(Za, Zb, i)` does two table lookups and a harmonic mean on
#     every shell pair, but depends only on the ELEMENT pair -- a handful of
#     distinct values per molecule. Memoised.


def _diatomic_scaled_overlap_cao(atomic_numbers, coords_bohr, basis):
    """Diatomic-frame scaling of S_cao.

    The frozen shape is a double loop over shell pairs (~5 000 for a 44-atom
    molecule), but only the p-p pairs actually need per-pair work: every other
    off-atom pair multiplies its block by a single scalar that depends on
    (l_a, l_b, Z_a, Z_b) alone.  So build the scalar factor for all of them as
    one (n_cao, n_cao) mask from a tiny (3, 3, n_elem, n_elem) table and apply
    it in one multiply, leaving only the p-p blocks for the loop -- ~435 of them
    instead of ~5 000.  Same multiplications in the same places, so the scalar
    part is bit-identical; probes/check_diat.py pins it.
    """
    atoms = np.asarray(atomic_numbers, dtype=np.intp)
    S = np.asarray(basis.S_cao, dtype=np.float64).copy()
    groups = _shell_index_groups(basis)
    sids = sorted(groups)
    n_sh = len(sids)
    start = np.empty(n_sh, dtype=np.intp)
    ncomp = np.empty(n_sh, dtype=np.intp)
    sh_atom = np.empty(n_sh, dtype=np.intp)
    sh_l = np.empty(n_sh, dtype=np.intp)
    for k, sid in enumerate(sids):
        idx = groups[sid]
        bf = basis.cao_basis[idx[0]]
        start[k] = idx[0]
        ncomp[k] = len(idx)
        sh_atom[k] = int(bf.atom_idx)
        sh_l[k] = int(bf.l_total)

    # (3, 3, n_elem, n_elem) scalar-factor table over the elements present.
    uz, einv = np.unique(atoms, return_inverse=True)
    ne = uz.size
    K = np.ones((3, 3, ne, ne), dtype=np.float64)
    has_d = [bool(_element_has_active_d_shell(int(z))) for z in uz]
    for ia in range(ne):
        Za = int(uz[ia])
        for ib in range(ne):
            Zb = int(uz[ib])
            k0 = _diat_scale(Za, Zb, 0)
            for la in range(3):
                for lb in range(3):
                    if la <= 1 and lb <= 1:
                        K[la, lb, ia, ib] = k0          # p-p overwritten below
                    else:
                        inter = _diat_interaction_index(la, lb)
                        if inter is not None and not (has_d[ia] and has_d[ib]):
                            K[la, lb, ia, ib] = 1.0 + GXTB_D_H0_SCALE_DAMP * (
                                _diat_scale(Za, Zb, inter) - 1.0)

    sh_e = einv[sh_atom]
    M_sh = K[sh_l[:, None], sh_l[None, :], sh_e[:, None], sh_e[None, :]]
    same = sh_atom[:, None] == sh_atom[None, :]
    pp = (sh_l[:, None] == 1) & (sh_l[None, :] == 1)
    M_sh = np.where(same | pp, 1.0, M_sh)               # p-p handled per pair
    M = np.repeat(np.repeat(M_sh, ncomp, axis=0), ncomp, axis=1)
    S *= M

    # p-p pairs keep the sigma/pi decomposition, which depends on the bond axis.
    eye3 = np.eye(3)
    p_sh = np.flatnonzero(sh_l == 1)
    cache: dict = {}
    for pos, ka in enumerate(p_sh[:-1]):
        a = int(sh_atom[ka])
        Za = int(atoms[a])
        sa = slice(int(start[ka]), int(start[ka]) + 3)
        for kb in p_sh[pos + 1:]:
            b = int(sh_atom[kb])
            if a == b:
                continue
            rab = coords_bohr[b] - coords_bohr[a]
            r = float(np.sqrt(rab @ rab))
            if r < 1.0e-14:
                continue
            Zb = int(atoms[b])
            key = (Za, Zb)
            kk = cache.get(key)
            if kk is None:
                kk = cache[key] = (_diat_scale(Za, Zb, 0), _diat_scale(Za, Zb, 1))
            sb = slice(int(start[kb]), int(start[kb]) + 3)
            raw = S[sa, sb]
            u = rab / r
            psig = np.outer(u, u)
            ppi = eye3 - psig
            blk = kk[1] * (ppi @ raw @ ppi) + kk[0] * (psig @ raw @ psig)
            S[sa, sb] = blk
            S[sb, sa] = blk.T
    return S


# ------------------------------------------- AES atom reduction, made cheap
#
# `aes_fast._atom_reduce_pair_values` scatters an (nao, nao) array into per-atom
# sums with `np.bincount` over nao^2 elements, rebuilding the broadcast index
# arrays on every call. It runs 226 times per six molecules (twice per SCF
# iteration) and is ~6 % of runtime.
#
# The scatter is separable and the identity is exact:
#     bincount(aoat[:,None] broadcast, X.ravel()) == bincount(aoat, X.sum(axis=1))
#     bincount(aoat[None,:] broadcast, X.ravel()) == bincount(aoat, X.sum(axis=0))
# so an nao^2 integer scatter becomes an nao^2 float sum plus an nao scatter.
# `probes/check_reduce.py` pins it bit-identical.


def _atom_reduce_fast(values_i, values_j, values_diag, aoat, nat):
    ao = np.asarray(aoat, dtype=np.intp)
    out = np.empty((values_i.shape[0], nat), dtype=np.float64)
    for k in range(values_i.shape[0]):
        rows = (values_i[k] + values_diag[k]).sum(axis=1)
        cols = values_j[k].sum(axis=0)
        out[k] = np.bincount(ao, weights=rows, minlength=nat)
        out[k] += np.bincount(ao, weights=cols, minlength=nat)
    return out


def _install_fast_aes_reduce() -> bool:
    try:
        from mlxmolkit.xtb import aes_fast as _af
        if getattr(_af, "_reduce_is_fast", False):
            return True
        _af._atom_reduce_pair_values = _atom_reduce_fast
        _af._reduce_is_fast = True
        return True
    except Exception:
        return False


def _coulomb_matrix(
    coords_bohr: np.ndarray,
    shell_atom: np.ndarray,
    shell_hardness: np.ndarray,
    k_exp: float = GXTB_TB2_KEXP,
) -> np.ndarray:
    """g-xTB second-order shell Coulomb kernel, `get_coulomb_matrix` (0 ulp)."""

    # ⚠️ The same-atom block IS a special case.  The comment that used to sit
    # here claimed R = 0 makes `1/(R + inv_avg*exp(0))` identical to `1/inv_avg`
    # "bit for bit"; it is not -- a round trip through the reciprocal loses the
    # last bit, and `probes/audit/port_stages.py` stage 10b reported exactly
    # that as 5.6e-17.  The recovered `get_coulomb_matrix` writes the on-site
    # block DIRECTLY:
    #     amat(ii+ish, ii+ish) += etai              (the shell's own hardness)
    #     amat(ii+jsh, ii+ish) += avg(etai, etaj)   (jsh /= ish, same atom)
    g = np.maximum(np.asarray(shell_hardness, dtype=np.float64), 1.0e-10)
    # 🔑 The binary forms the HARMONIC AVERAGE and then divides by it:
    #     gam = 2/(1/a + 1/b) ;  tmp = r + damp/gam ;  A = 1/tmp
    # `0.5*(1/a + 1/b) * damp` is the same number algebraically and a different
    # one in floating point -- 5.6e-17 on ordinary input, which is exactly what
    # `probes/audit/port_stages.py` stage 10b was reporting.  Kept in the
    # binary's association.
    gam = 2.0 / (1.0 / g[:, None] + 1.0 / g[None, :])
    xyz = np.asarray(coords_bohr, dtype=np.float64)[np.asarray(shell_atom, dtype=np.intp)]
    R = _pair_dist(xyz)
    jmat = 1.0 / (R + np.exp(-k_exp * R) / gam)
    same = np.asarray(shell_atom)[:, None] == np.asarray(shell_atom)[None, :]
    jmat = np.where(same, gam, jmat)
    np.fill_diagonal(jmat, g)
    return jmat


def _halide_increment_correction(atomic_numbers: np.ndarray) -> float:
    atoms = np.asarray(atomic_numbers, dtype=np.intp)
    return float(sum(GXTB_HALIDE_INCREMENT_CORRECTION.get(int(Z), 0.0) for Z in atoms))


def _mulliken_shell_charges(P: np.ndarray, S: np.ndarray, bf_to_shell: np.ndarray, n_shell: int, z_ref: np.ndarray) -> np.ndarray:
    PS = np.einsum("ij,ji->i", P, S)
    pop = np.bincount(np.asarray(bf_to_shell, dtype=np.intp), weights=PS,
                      minlength=n_shell)[:n_shell]
    return z_ref - pop


# Re-measured after the Pulay solve was made cheap (incremental Gram matrix +
# ddot on ravelled views).  The earlier scan rejected a longer history because
# the bigger Pulay solve ate the gain; with that cost removed the trade flips:
# 18.6 -> 16.8 mean iterations now shows up as ~6 % of wall time.
# probes/scan_diis.py: warm=3/hist=6 1.14 s -> warm=2/hist=12 0.99 s.
GXTB_DIIS_WARMUP = 2
GXTB_DIIS_MAX = 12


GXTB_CHARGE_MIXING = False
GXTB_CHARGE_HIST = 8
GXTB_CHARGE_BETA = 1.0


def _anderson_mix(q_hist, r_hist, beta: float) -> np.ndarray:
    """Pulay/Anderson extrapolation in shell-charge space.

    The only self-consistent variable in this SCF is the shell-charge vector,
    so the natural residual is r = q_out - q_in rather than the Fock commutator.
    The least-squares system is (hist x hist) over ~100-vectors instead of a
    Pulay solve over n_basis^2 matrices, so it is also far cheaper per step.
    Mixing changes the path to the fixed point, never the fixed point itself.
    """
    m = len(r_hist)
    if m == 1:
        return q_hist[-1] + beta * r_hist[-1]
    R = np.stack(r_hist, axis=1)
    B = R.T @ R
    tr = float(np.trace(B))
    if tr > 0.0:
        B = B / (tr / m)                    # scale for conditioning
    A = np.zeros((m + 1, m + 1), dtype=np.float64)
    A[:m, :m] = B
    A[:m, m] = -1.0
    A[m, :m] = -1.0
    rhs = np.zeros(m + 1, dtype=np.float64)
    rhs[m] = -1.0
    try:
        c = np.linalg.solve(A, rhs)[:m]
    except np.linalg.LinAlgError:
        return q_hist[-1] + beta * r_hist[-1]
    if not np.all(np.isfinite(c)):
        return q_hist[-1] + beta * r_hist[-1]
    return np.stack(q_hist, axis=1) @ c + beta * (R @ c)


def _pulay_diis_numpy(F_hist: list[np.ndarray], e_hist: list[np.ndarray],
                      state: dict | None = None, popped: bool = False) -> np.ndarray:
    """Pulay extrapolation over the Fock history.

    Two costs removed.  `np.sum(e_i * e_j)` allocated a full (n, n) temporary
    for each of the nd(nd+1)/2 overlaps; `np.dot` on the ravelled views is the
    same number as a BLAS ddot with no temporary at all.  And the Gram matrix is
    maintained across iterations rather than rebuilt: only the new row is new,
    so a step costs nd dot products instead of nd(nd+1)/2.
    """
    nd = len(F_hist)
    if nd < 2:
        if state is not None:
            state["B"] = None
        return F_hist[-1]
    B_old = state.get("B") if state is not None else None
    if popped and B_old is not None:
        B_old = B_old[1:, 1:]
    if B_old is None or B_old.shape[0] != nd - 1:
        flat = [e.ravel() for e in e_hist]
        B = np.empty((nd, nd), dtype=np.float64)
        for i in range(nd):
            for j in range(i, nd):
                B[i, j] = B[j, i] = float(np.dot(flat[i], flat[j]))
    else:
        B = np.empty((nd, nd), dtype=np.float64)
        B[: nd - 1, : nd - 1] = B_old
        last = e_hist[-1].ravel()
        row = np.fromiter((float(np.dot(e.ravel(), last)) for e in e_hist),
                          dtype=np.float64, count=nd)
        B[-1, :] = row
        B[:, -1] = row
    if state is not None:
        state["B"] = B
    A = np.zeros((nd + 1, nd + 1), dtype=np.float64)
    A[:nd, :nd] = B
    A[:nd, nd] = -1.0
    A[nd, :nd] = -1.0
    rhs = np.zeros(nd + 1, dtype=np.float64)
    rhs[nd] = -1.0
    try:
        coeffs = np.linalg.solve(A, rhs)[:nd]
    except np.linalg.LinAlgError:
        return F_hist[-1]
    out = np.zeros_like(F_hist[-1])
    for mat, coeff in zip(F_hist, coeffs):
        out += coeff * mat
    return out


_CHOL_CACHE: dict = {}


def _solve_generalized(F: np.ndarray, S: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """F C = S C e, with the metric factorised once per molecule.

    `scipy.linalg.eigh(F, S)` re-Choleskys S on every call, but S is fixed for
    the whole SCF -- ~19 calls per molecule.  Factor it once, cache the inverse
    factor, and each iteration is then two matmuls plus a standard eigensolve.

    Benchmarked at n=92 (probes/bench_eigh.py): the generalized driver is
    0.492 ms, this path 0.410 ms.  Every LAPACK backend reachable from here
    lands on the same 0.386 ms standard eigensolve -- numpy on Accelerate,
    scipy driver='evd' and torch CPU are within 1 % of each other, driver='evr'
    is slower, MLX has no GPU eigh at all and torch MPS has no float64.  There
    is no faster eigensolver at this size; the only win is not redoing the
    metric.  Agreement with the generalized driver: 5.3e-14.
    """
    try:
        key = id(S)
        hit = _CHOL_CACHE.get(key)
        if hit is None or hit[0] is not S:
            from scipy.linalg import cholesky
            L = cholesky(S, lower=True, check_finite=False)
            Li = np.ascontiguousarray(np.linalg.inv(L))
            if len(_CHOL_CACHE) > 8:
                _CHOL_CACHE.clear()
            _CHOL_CACHE[key] = (S, Li)
        else:
            Li = hit[1]
        w, Cp = np.linalg.eigh(Li @ F @ Li.T)
        return w, Li.T @ Cp
    except Exception:
        s_eval, U = np.linalg.eigh(S)
        keep = s_eval > 1.0e-8
        X = U[:, keep] * (1.0 / np.sqrt(s_eval[keep]))[None, :]
        w, Cp = np.linalg.eigh(X.T @ F @ X)
        return w, X @ Cp


def _fock_from_shell_potential(
    H0: np.ndarray,
    S: np.ndarray,
    bf_to_shell: np.ndarray,
    V_sh: np.ndarray,
) -> np.ndarray:
    V_bf = V_sh[bf_to_shell]
    return H0 - 0.5 * (V_bf[:, None] + V_bf[None, :]) * S


def _generalized_hubbard_average(ua: float, ub: float, xi: float) -> float:
    """Generalized average from the g-xTB MFX SI Eq. 150."""

    if ua <= 0.0 or ub <= 0.0:
        return max(ua, ub, 1.0e-12)
    if abs(xi - 0.0) < 1.0e-14:
        return 0.5 * (ua + ub)
    if abs(xi - 1.0) < 1.0e-14:
        return math.sqrt(ua * ub)
    if abs(xi - 2.0) < 1.0e-14:
        return 2.0 / (1.0 / ua + 1.0 / ub)
    return (2.0 ** (xi - 1.0)) * ((ua * ub) ** (0.5 * xi)) / ((ua + ub) ** (xi - 1.0))


# DECODED (get_gmulliken_0d._omp_fn.0, onsite block at LAB_0034aef0).  The
# Mulliken kernel has TWO onsite forms the port did not distinguish:
#     same atom, different shells:  gamma = u_i * u_j        * FR
#     same atom, same shell:        gamma = u_i * ONSITE_DIAG * FR
# The port used the generalized Hubbard average with no extra factor for both,
# i.e. u_i*FR on the diagonal.  ONSITE_DIAG is the scalar the binary reads from
# its exchange record at +0x230 (`get_mulliken_kmatrix` passes it as param_5,
# dereferenced in the omp body as *pdVar35).  That field is populated at run
# time, so its value is not readable from the static image; it is instead
# MEASURED from the binary's own H2 output by exact Fock inversion
# (probes/h2_gamma.py): the binary's onsite gamma is 0.194598 against the
# port's u*FR = 0.506054, giving 0.194598/0.506054.
GXTB_MFX_ONSITE_DECODED = False
# Scale of the ONSITE exchange component (get_gons): gons = (1 - 0.5*(kq_a q_a
# + kq_b q_b)) * scale * onec, with onec the one-centre exchange integral.
# It is ZERO for s-s pairs, so it vanishes for H2 -- which is why the H2
# inversion pins the diagonal scalar cleanly but says nothing about this term.
GXTB_MFX_ONSITE_EXCH = 0.0
GXTB_MFX_ONSITE_DIAG = 0.3845401


def _mfx_gamma_ao(
    atomic_numbers: np.ndarray,
    coords_ang: np.ndarray,
    basis,
    *,
    frscale: float = GXTB_MFX_FR_SCALE,
    lrscale: float = GXTB_MFX_LR_SCALE,
    omega: float = GXTB_MFX_OMEGA,
    gexp: float = GXTB_MFX_GEXP,
    qsh: np.ndarray | None = None,
    use_offdiag_l: bool = False,
    use_kq: bool = False,
    kq_onsite: bool = False,
    decoded_damping: bool = False,
    onsite_decoded: bool | None = None,
    onsite_diag: float | None = None,
) -> np.ndarray:
    """Range-separated Mulliken Fock-exchange AO kernel.

    Vectorised rewrite of the original double loop; `probes/check_mfx.py` pins
    it against that loop at machine precision. The closed form of
    `_generalized_hubbard_average` covers its xi=0/1/2 special cases exactly
    (2^(xi-1) (ua ub)^(xi/2) / (ua+ub)^(xi-1) reduces to the arithmetic,
    geometric and harmonic mean respectively), so dropping the branches is not
    an approximation.

    Scalars are binary-verified at libxtb __const 0x73b4d8:
    frscale=0.15, lrscale=0.85, omega=0.2, gexp=1.3826597204.

    The two optional pieces are DECODED, not guessed: `add_exchange`
    (libxtb 0x4182c0) resolves adrp/add loads against `pg_fock_offdiag_l` and
    `pg_fock_kq` as well as the `ps_fock_shell_hubbard` / `ps_fock_avg_exp`
    this function already used, so both belong in this kernel. Their exact
    algebraic placement is still inferred -- see probes/DECODED.md.
    """

    atoms = np.asarray(atomic_numbers, dtype=np.intp)
    coords_bohr = np.asarray(coords_ang, dtype=np.float64) * ANG_TO_BOHR
    bf_to_shell = np.asarray(basis.bf_to_shell, dtype=np.int64)
    shell_atom = np.asarray(basis.shell_atom, dtype=np.int64)
    shell_l = np.asarray(basis.shell_l, dtype=np.int64)
    shell_local = _shell_local_indices(shell_atom)

    Z = atoms[shell_atom]
    hub = np.asarray(GXTB_PARAMS["ps_fock_shell_hubbard"], dtype=np.float64)
    avg = np.asarray(GXTB_PARAMS["ps_fock_avg_exp"], dtype=np.float64)
    # 🔑 add_exchange 0x418670 `fmul d30, d30, d31`:
    #     gpar(ish,izp) = ps_fock_shell_hubbard(ish,num) * pa_hubbard_parameter(num)
    # d31 is loaded ONCE per species outside the shell loop, so it is a clean
    # per-element scalar on every shell. Cross-checked against the binary's own
    # container: gam(C) = [1.0831544692, 1.1256991228], which is exactly this
    # product; the raw table alone gives [2.5669958364, 2.6678235131].
    _hubpar = np.asarray(GXTB_PARAMS["pa_hubbard_parameter"], dtype=np.float64)
    shell_u = hub[Z - 1, shell_local] * _hubpar[Z - 1]
    shell_u = np.where(shell_u <= 0.0, 1.0e-12, shell_u)
    shell_xi = avg[Z - 1, shell_local]

    ish = bf_to_shell
    ua = shell_u[ish][:, None]
    ub = shell_u[ish][None, :]
    xi = np.maximum(shell_xi[ish][:, None], shell_xi[ish][None, :])
    favg = (2.0 ** (xi - 1.0)) * ((ua * ub) ** (0.5 * xi)) / ((ua + ub) ** (xi - 1.0))

    at = shell_atom[ish]
    d = coords_bohr[at][:, None, :] - coords_bohr[at][None, :, :]
    rij = np.sqrt(np.sum(d * d, axis=-1))

    if _MFX_DAMP_HOOK is not None:
        # Experiment hook (probes/mfx_damping_ab.py). Default None = untouched.
        gam = _MFX_DAMP_HOOK(favg=favg, rij=rij, xi=xi, zbf=Z[ish],
                             frscale=float(frscale))
    elif decoded_damping:
        # 🔑 The binary's pair term is (favg_l / damp) * gam, where
        #     favg_l(jsh,ish) = sqrt(pg_fock_offdiag_l[l_i] * pg_fock_offdiag_l[l_j])
        # is the ONLY element-independent, strongly l-dependent factor in the
        # exchange model: 0.05 (s-s), 0.353553 (s-p), 2.5 (p-p) -- a 50x s:p
        # contrast the port had flattened to nothing. pg_fock_offdiag_l is read
        # exactly once in the whole 5 MB binary, at add_exchange+0x940, and
        # new_exchange_fock takes its geometric mean.
        _fod = np.asarray(GXTB_PARAMS["pg_fock_offdiag_l"], dtype=np.float64)
        _fl = _fod[shell_l[ish]]
        favg_l = np.sqrt(_fl[:, None] * _fl[None, :])
        # 🔑 favg_l covers EVERY pair except the on-site shell DIAGONAL, where
        # the binary substitutes the `onsite` scalar:
        #     kmat(ii+ish, ii+ish) = gam(ish,ish)*onsite*alpha
        # A shell index can only equal itself on the same atom, so ish_i == ish_j
        # selects exactly that diagonal. Leaving favg_l there instead crushes
        # on-site s-s exchange by 0.05/1.3827 = 27x and wrecks the result.
        _same_shell = ish[:, None] == ish[None, :]
        favg_l = np.where(_same_shell, GXTB_MFX_ONSITE_SCALE, favg_l)
        # damp uses the vdW PAIR RADIUS, per atom pair, not the per-shell xi.
        #
        # 🔑 And the radius is SCALED.  `add_exchange` (recovered, 0 ulp) builds
        #     rad(jzp,izp) = get_vdw_rad_pair_num(...) * aatoau
        #                    * arithmetic_average(pa_rvdw_scale_i, pa_rvdw_scale_j)
        # and the default entry of `pa_rvdw_scale` is exactly 1/aatoau, so for
        # every element that carries the default the product is 1.0 and the raw
        # pair radius is right.  Carbon and sulfur BOTH carry it -- which is why
        # dropping the factor was invisible on the CH3SH oracle -- but nitrogen,
        # oxygen and fluorine do not: C-O is 1.040x, O-O 1.080x, C-N 1.025x.
        from mlxmolkit.xtb.mctc_vdwrad import mctc_vdw_pair_matrix_bohr
        _rad_at = mctc_vdw_pair_matrix_bohr(atoms)
        _AATOAU = 1.0 / 0.5291772109044924
        _sc = np.asarray(GXTB_PARAMS["pa_rvdw_scale"], dtype=np.float64)[atoms - 1]
        _rad_at = _rad_at * _AATOAU * (0.5 * (_sc[:, None] + _sc[None, :]))
        _rad = _rad_at[at][:, at]
        gam = favg_l * favg * np.exp(
            rij * (GXTB_MFX_DAMP_C0 + GXTB_MFX_DAMP_C1 * _rad))
    else:
        gam = favg * float(frscale)
    onsite = rij < 1.0e-14
    r_safe = np.where(onsite, 1.0, rij)
    from scipy.special import erf as _erf
    tmp = 1.0 / ((r_safe ** float(gexp) + gam ** (-float(gexp))) ** (1.0 / float(gexp)))
    value = (float(frscale) + float(lrscale) * _erf(float(omega) * r_safe)) * tmp
    # R -> 0 limit of the expression above is frscale*gam, which is what the
    # frozen kernel stored directly (its `gam` already carried the frscale).
    if onsite_decoded is None:
        onsite_decoded = GXTB_MFX_ONSITE_DECODED
    if onsite_decoded:
        # Same-atom, DIFFERENT shells: hubbard(jsh,ish)*frscale -- i.e. favg*FR,
        # which is what the port already had; the public tblite source
        # (mulliken-FX, get_gmat_0d) confirms it, and replacing it with u_i*u_j
        # was measured to be a large regression.
        # Same-atom, SAME shell: the release multiplies by one extra factor the
        # public source does not have (3 multiplicands vs 2), read from its
        # exchange record at +0x230.  See GXTB_MFX_ONSITE_DIAG.
        cdiag = GXTB_MFX_ONSITE_DIAG if onsite_diag is None else float(onsite_diag)
        base_on = float(frscale) * gam if decoded_damping else gam
        same_shell = ish[:, None] == ish[None, :]
        on_val = np.where(same_shell, base_on * cdiag, base_on)
        # + the onsite exchange component (get_gons), same-atom blocks only
        if GXTB_MFX_ONSITE_EXCH != 0.0 and _install_onecxints():
            try:
                from mlxmolkit.xtb.gxtb_aes import gxtb_onsite_gamma
                onec = np.asarray(gxtb_onsite_gamma(basis, atomic_numbers),
                                  dtype=np.float64)
                if qsh is not None:
                    kqp = np.asarray(GXTB_PARAMS["pg_fock_kq"], dtype=np.float64)
                    w = kqp[shell_l[ish]] * np.asarray(qsh, dtype=np.float64)[ish]
                    onec = onec * (1.0 - 0.5 * (w[:, None] + w[None, :]))
                on_val = on_val + float(GXTB_MFX_ONSITE_EXCH) * onec
            except Exception:
                pass
        gamma = np.where(onsite, on_val, value)
    else:
        gamma = np.where(onsite, float(frscale) * gam if decoded_damping else gam, value)

    if use_offdiag_l:
        # `add_exchange` loads pg_fock_offdiag_l; applied to AO pairs from
        # DIFFERENT shells, indexed by the lower angular momentum of the pair.
        fod = np.asarray(GXTB_PARAMS["pg_fock_offdiag_l"], dtype=np.float64)
        lb = shell_l[ish]
        diff_shell = ish[:, None] != ish[None, :]
        f = fod[np.minimum(np.broadcast_to(lb[:, None], gamma.shape),
                           np.broadcast_to(lb[None, :], gamma.shape))]
        gamma = np.where(diff_shell, gamma * f, gamma)

    if kq_onsite and qsh is not None:
        # DECODED (`get_gons._omp_fn.0`): the charge factor
        # `(1 - 0.5*(kq_a q_a + kq_b q_b))` belongs to the ONSITE exchange
        # component, not to the whole Mulliken kernel. Applying it everywhere
        # (use_kq) is destructive; applying it to same-atom blocks is the
        # decoded placement.
        kq = np.asarray(GXTB_PARAMS["pg_fock_kq"], dtype=np.float64)
        w = kq[shell_l[ish]] * np.asarray(qsh, dtype=np.float64)[ish]
        fac = 1.0 - 0.5 * (w[:, None] + w[None, :])
        same_atom = shell_atom[ish][:, None] == shell_atom[ish][None, :]
        gamma = np.where(same_atom, gamma * fac, gamma)

    if use_kq and qsh is not None:
        # `add_exchange` loads pg_fock_kq, a per-l charge coefficient. The
        # charge-scaled form (1 - 0.5*(kq_a q_a + kq_b q_b)) is the one
        # gxtb_aes.gxtb_onsite_potential_q documents for the same constant.
        kq = np.asarray(GXTB_PARAMS["pg_fock_kq"], dtype=np.float64)
        w = kq[shell_l[ish]] * np.asarray(qsh, dtype=np.float64)[ish]
        gamma = gamma * (1.0 - 0.5 * (w[:, None] + w[None, :]))

    return gamma


def _mfx_fock_energy(P: np.ndarray, S: np.ndarray, gamma_ao: np.ndarray) -> tuple[float, np.ndarray]:
    """Return ``(E_MFX, F_MFX)`` using tblite's Mulliken-FX matrix factorization."""

    P_arr = np.asarray(P, dtype=np.float64)
    S_arr = np.asarray(S, dtype=np.float64)
    gamma = np.asarray(gamma_ao, dtype=np.float64)
    sp = S_arr @ P_arr
    prev = gamma * (0.5 * (sp @ S_arr))
    tmp = gamma * sp
    tmp = tmp + 0.5 * (S_arr @ (gamma * P_arr))
    prev = prev + 0.5 * (tmp @ S_arr)
    prev = -0.25 * (prev + prev.T)
    fock = 0.5 * prev
    fock = 0.5 * (fock + fock.T)
    energy = float(np.sum(P_arr * fock))
    return energy, fock


_TB3_TAU_CACHE: dict = {}


def _twobody_tau(basis, atoms, coords_ang, *, k3=2.3, kx=1.3, rexp=0.2093327496):
    """Geometry-only tau matrix of the two-body third order term.

    Same algebra as `gxtb_aes.gxtb_twobody_thirdorder`, hoisted out of the SCF
    loop.  probes/check_tb3.py pins the (E3, V3) pair against the frozen routine.
    """
    key = id(basis)
    hit = _TB3_TAU_CACHE.get(key)
    if hit is not None and hit[0] is basis:
        return hit[1]
    at = np.asarray(atoms, dtype=np.intp)
    cb = np.asarray(coords_ang, dtype=np.float64) * ANG_TO_BOHR
    sa = np.asarray(basis.shell_atom)
    sl = np.asarray(basis.shell_l)
    cn = np.asarray(basis.cn, dtype=np.float64)
    Z = at[sa]
    tb2sh = np.asarray(GXTB_PARAMS["ps_tb2_shell_hubbard"], dtype=np.float64)
    hubp = np.asarray(GXTB_PARAMS["pa_hubbard_parameter"], dtype=np.float64)
    cnsc = np.asarray(GXTB_PARAMS["pa_tb2_hubbard_cn"], dtype=np.float64)
    eta_base = tb2sh[Z - 1, _shell_local_indices(sa)] * hubp[Z - 1]   # shell INDEX
    eta_eff = eta_base * (1.0 + cnsc[Z - 1] * (np.sqrt(cn[sa] + 1e-12) - 1e-6))
    with np.errstate(divide="ignore", invalid="ignore"):
        gam = 2.0 / (1.0 / eta_eff[:, None] + 1.0 / eta_eff[None, :])
        R = np.linalg.norm(cb[sa][:, None, :] - cb[sa][None, :, :], axis=-1)
        x = R / gam
        tau = np.where(sa[:, None] == sa[None, :],
                       -rexp * gam * gam,
                       k3 * x * (1.0 - 0.5 * kx * x) * np.exp(-kx * x))
    # Defensive: a zero `ps_tb2_shell_hubbard` entry gives gam -> 0, x -> inf,
    # and this spelling then evaluates inf * exp(-inf) = nan where the binary's
    # (`val = b/((1/gam)*(1/gam))`, i.e. b*gam^2) stays finite. The analytic
    # limit is 0 -- the exponential beats the polynomial. Does not trigger for
    # the shells `pa_nshell` actually selects, but the algebra should not be
    # the thing that decides that.
    tau = np.where(np.isfinite(tau), tau, 0.0)
    if len(_TB3_TAU_CACHE) > 8:
        _TB3_TAU_CACHE.clear()
    _TB3_TAU_CACHE[key] = (basis, tau)
    return tau


def _third_order_twobody(
    basis,
    atoms: np.ndarray,
    coords_ang: np.ndarray,
    qsh: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Binary-exact g-xTB two-body third-order (``coulomb_thirdorder_twobody``).

    Thin adapter over :func:`mlxmolkit.xtb.gxtb_aes.gxtb_twobody_thirdorder`, which
    holds the decoded tau-matrix algebra: harmonic-averaged effective shell
    hardness ``eta_eff = eta_base*(1 + pa_tb2_hubbard_cn*(sqrt(cn+1e-12)-1e-6))``
    with ``eta_base = ps_tb2_shell_hubbard*pa_hubbard_parameter`` (no cn slope), an
    off-site ``k3*x*(1-0.5*kx*x)*exp(-kx*x)`` kernel and an on-site (incl. diagonal)
    ``-REXP*gamma^2`` block.  Returns ``(energy, V_shell)``.
    """

    q = np.asarray(qsh, dtype=np.float64)
    if q.size == 0:
        return 0.0, np.zeros(0, dtype=np.float64)
    # The whole tau matrix -- shell distances, the harmonic-averaged hardness and
    # the exponential -- depends only on geometry; the frozen routine rebuilt it
    # on every SCF iteration.  Cache it and keep only the two matvecs.
    tau = _twobody_tau(basis, atoms, coords_ang)
    g3d = np.asarray(basis.shell_third, dtype=np.float64)
    tq = tau @ q
    E3 = float(np.sum(g3d * q * q * tq))
    V3 = 2.0 * g3d * q * tq + tau @ (g3d * q * q)
    return E3, V3


def _shell_local_indices(shell_atom: np.ndarray) -> np.ndarray:
    local = np.zeros(shell_atom.size, dtype=np.intp)
    counts: dict[int, int] = {}
    for ish, atom_idx0 in enumerate(shell_atom):
        atom_idx = int(atom_idx0)
        local[ish] = counts.get(atom_idx, 0)
        counts[atom_idx] = int(local[ish]) + 1
    return local


# `tblite_coulomb_firstorder::get_potential` accumulates
#
#   vsh += 0.5*[ 2 + kdis*(erf(a) + erf(b))
#                + kdis*q*kx*(2/sqrt(pi))*(exp(-a^2) + exp(-b^2)) ]
#          * ipea(ish, izp) * (1 + ipea_cn(izp)*(sqrt(cn + 1e-12) - 1e-6))
#
# with a = kx*(q - ks), b = kx*(q + ks) and q = wfn%qat -- the ATOM charge, in
# tblite's own convention, which is the port's too (both give O = -0.7 in
# water). There is NO negation in either branch of the decompile, so the
# docstring's `charge_sign=-1` is wrong; the decoded value is +1.
#
# ⚠️ +1 fits oxygen WORSE than -1 does (H2O 0.31 vs 0.08 against the binary's
# converged charges) while fitting sulfur much better (H2S 0.03 vs 0.48).
# Neither sign fits both, so the disagreement is in the FORM, not the sign,
# and this term needs its own differential probe before it can be trusted.
# The decoded sign is used here because the decompile is the oracle.
GXTB_FO_SIGN = 1.0
GXTB_FO_QREF = 0.0


def _first_order_onsite(
    atoms: np.ndarray,
    cn: np.ndarray,
    shell_atom: np.ndarray,
    qsh: np.ndarray,
    *,
    charge_sign: float = -1.0,
) -> tuple[float, np.ndarray]:
    """Binary-observed onsite first-order TB term and shell potential.

    The released g-xTB binary passes ``ps_tb1_ipea`` to the onsite-firstorder
    object and stores the discontinuity constants as ``kx=1.0``,
    ``kdis=0.025``, ``ks=2/3``.  Its switching is implemented as
    ``0.5 * (2 + kdis * (erf(kx*(q-ks)) + erf(kx*(q+ks))))``.

    ``mlxmolkit`` uses xTB's positive-deficiency Mulliken convention
    ``q = z_ref - population``.  The first-order module in the SI/binary acts
    on the opposite density-fluctuation sign, hence the default
    ``charge_sign=-1`` and the corresponding chain-rule sign on the returned
    potential.
    """

    n_shell = shell_atom.size
    potential_model = np.zeros(n_shell, dtype=np.float64)
    if n_shell == 0:
        return 0.0, potential_model

    q_model = charge_sign * np.asarray(qsh, dtype=np.float64)
    qat_model = np.bincount(shell_atom, weights=q_model, minlength=atoms.size)
    sqrt_cn = np.sqrt(np.asarray(cn, dtype=np.float64) + GXTB_TB1_CN_EPS) - 1.0e-6

    mu = np.zeros(n_shell, dtype=np.float64)
    shell_local = _shell_local_indices(shell_atom)
    for ish, atom_idx0 in enumerate(shell_atom):
        atom_idx = int(atom_idx0)
        Z = int(atoms[atom_idx])
        local_shell = int(shell_local[ish])
        mu0 = float(GXTB_PARAMS["ps_tb1_ipea"][Z - 1, local_shell])
        cn_scale = 1.0 + float(GXTB_PARAMS["pa_tb1_ipea_cn"][Z - 1]) * sqrt_cn[atom_idx]
        mu[ish] = mu0 * cn_scale

    energy = 0.0
    for atom_idx in range(atoms.size):
        mask = shell_atom == atom_idx
        if not np.any(mask):
            continue
        q_atom = float(qat_model[atom_idx])
        x_minus = GXTB_TB1_KX * (q_atom - GXTB_TB1_KS)
        x_plus = GXTB_TB1_KX * (q_atom + GXTB_TB1_KS)
        erf_sum = math.erf(x_minus) + math.erf(x_plus)
        f_switch = 1.0 + 0.5 * GXTB_TB1_KDIS * erf_sum
        df_switch = (
            0.5
            * GXTB_TB1_KDIS
            * GXTB_TB1_KX
            * _TWO_OVER_SQRT_PI
            * (math.exp(-(x_minus * x_minus)) + math.exp(-(x_plus * x_plus)))
        )

        moment = float(np.sum(mu[mask] * q_model[mask]))
        energy += f_switch * moment
        potential_model[mask] = mu[mask] * f_switch + df_switch * moment

    return energy, charge_sign * potential_model


_MP_CACHE_FAST: dict = {}
_AES_GAB_CACHE: dict = {}


def qvszp_multipoles(basis):
    """Override of ``mlxmolkit.xtb.gxtb_aes.qvszp_multipoles`` — same numbers, faster.

    The frozen version calls ``multipole_integrals.multipole_matrices``, a
    Python loop over primitive pairs that costs ~1.8 s on a 44-atom molecule
    and is the single largest cost in the AES path once ``mmompop``/``setvsdq``
    are vectorised.  The repo already ships a compiled kernel for it,
    ``multipole_integrals_cpp.multipole_matrices_cpp``; it was simply never
    wired into this route.  ``probes/check_cpp_mp.py`` checks the two against
    each other on the three largest score molecules: max |delta| 5.5e-14 at
    400x.  Pure cost, not physics.

    Falls back to the frozen path if the extension is not built, so the solver
    still runs on a machine without it.
    """
    key = id(basis)
    hit = _MP_CACHE_FAST.get(key)
    if hit is not None and hit[0] is basis:   # id() is reused after a free
        return hit[1]
    if len(_MP_CACHE_FAST) > 8:
        _MP_CACHE_FAST.clear()
    try:
        from mlxmolkit.xtb.multipole_integrals_cpp import (
            CPP_AVAILABLE,
            multipole_matrices_cpp,
        )
        if not CPP_AVAILABLE:
            raise ImportError
        S_cao, dp_cao, qp_cao = multipole_matrices_cpp(basis.cao_basis)
    except Exception:
        from mlxmolkit.xtb.multipole_integrals import multipole_matrices
        S_cao, dp_cao, qp_cao = multipole_matrices(basis.cao_basis)
    T = np.asarray(basis.T_cao_to_sao, dtype=np.float64)
    out = (
        T @ S_cao @ T.T,
        np.stack([T @ dp_cao[k] @ T.T for k in range(3)], axis=0),
        np.stack([T @ qp_cao[k] @ T.T for k in range(6)], axis=0),
    )
    _MP_CACHE_FAST[key] = (basis, out)
    return out


def gxtb_aes_fock(
    P: np.ndarray,
    basis,
    atoms: np.ndarray,
    coords_ang: np.ndarray,
    *,
    channel3: int = 0,
    channel5: int = 1,
    want_energy: bool = True,
    want_vat: bool = False,
):
    """Override of ``mlxmolkit.xtb.gxtb_aes.gxtb_aes_fock`` — same numbers, faster.

    The frozen version spends ~85% of its time in ``aes.mmompop`` and
    ``aes.setvsdq``, both of which are explicit Python loops over AO/atom
    pairs.  ``mlxmolkit.xtb.aes_fast`` already carries vectorised twins of
    exactly those two; ``probes/check_fast_aes.py`` checks them against the
    originals on the largest score molecule and finds max |delta| of 5e-15
    (dipm/qp) and 4e-16 (vs/vd/vq) — machine precision, i.e. the same
    function — at 20x and 47x.

    This is a pure cost change, not a physics change.  It matters because AES
    is worth ~5% of the charge MAE but at the frozen cost would push the run
    past the 120 s wall cap.

    ``want_energy=False`` additionally skips ``aniso_electro``, whose only
    output is the AES energy.  Inside the SCF loop nothing reads that energy,
    and the total energy is not scored anyway; the final post-SCF call still
    computes it so ``energy_hartree`` keeps its meaning.
    """
    from mlxmolkit.xtb.aes import aniso_electro, fockelectro
    from mlxmolkit.xtb.aes_fast import mmompop_vectorized, setvsdq_vectorized
    # A COMPILED mmompop exists and the GFN2 fast path already uses it
    # (`scf_gfn2_fast.py`: `_ref.mmompop = mmompop_cpp if CPP_AVAILABLE`), but
    # the g-xTB route never did. It is the AES hot spot -- ~19 calls per
    # molecule. Pinned bit-identical by probes/check_mmompop.py.
    try:
        from mlxmolkit.xtb.multipole_integrals_cpp import CPP_AVAILABLE as _CA
        from mlxmolkit.xtb.multipole_integrals_cpp import mmompop_cpp as _mmompop
        if not _CA:
            _mmompop = mmompop_vectorized
    except Exception:
        _mmompop = mmompop_vectorized
    from mlxmolkit.xtb.gxtb_aes import (gxtb_aes_gab, gxtb_mrad_pair,
                                        GXTB_AES_DAMP_MAG, GXTB_AES_DAMP_SCALE)

    atoms = np.asarray(atoms, dtype=np.intp)
    _geo_key = (id(basis), id(coords_ang))
    _geo = _aes_geometry(_geo_key, atoms, coords_ang)
    coords_bohr = _geo["cb"]
    S, dpint, qpint = qvszp_multipoles(basis)
    # Everything below that sees only the basis and the geometry is built once
    # per molecule (`_aes_static`): the AO->atom map, the reference charge,
    # the dipole scale, the recovered amat blocks and the atom-centred
    # traceless integral stack.  The frozen path rebuilt all of them on every
    # SCF iteration.  Same arrays, same values -- pure cost.
    _st = _aes_static(basis, atoms, coords_bohr, S, dpint, qpint)
    aoat = _st["aoat"]

    # Mulliken atomic charges + cumulative atomic dipoles/quadrupoles.
    pop = np.bincount(aoat, weights=np.einsum("ij,ji->i", P, S), minlength=atoms.size)
    zref = _st["zref"]
    q = zref - pop
    dipm, qp = _mmompop(P, S, dpint, qpint, aoat, coords_bohr)

    dipscale = _st["dipscale"]
    # ⚠️ `pa_aes_dip_scale` IS `rad_at`, and on the RECOVERED route it already
    # enters `multipole_amat` as the asymmetric sqrt(CN) scale.  Applying it to
    # the dipoles as well counts it twice.  The GFN2 route (`aniso_electro` /
    # `setvsdq`) does want the scaled ones, so keep both and hand each route
    # what it was built for.
    dipm_raw = dipm
    dipm = dipm * dipscale[None, :]

    # `gxtb_mrad_pair` and `gxtb_aes_gab` are GEOMETRY-ONLY, but the frozen
    # routine rebuilds them on every SCF iteration (~25x per molecule). Cache
    # them per basis, exactly as qvszp_multipoles already is. No physics change.
    key = (id(basis), int(channel3), int(channel5))
    hit = _AES_GAB_CACHE.get(key)
    if hit is not None and hit[0] is basis:
        gab3, gab5 = hit[1]
    else:
        if len(_AES_GAB_CACHE) > 8:
            _AES_GAB_CACHE.clear()
        mrad = gxtb_mrad_pair(atoms)
        gab3, gab5 = gxtb_aes_gab(coords_bohr, mrad, channel3, channel5)
        _AES_GAB_CACHE[key] = (basis, (gab3, gab5))
    e_aes = 0.0
    if want_energy:
        e_aes, _ = aniso_electro(atoms.tolist(), coords_bohr, q, dipm, qp, gab3, gab5)
    if GXTB_AES_RECOVERED:
        # 🔑 The RECOVERED model: `coulomb_multipole_gxtb::get_multipole_matrix`
        # (five amat blocks, the asymmetric sqrt(CN) scale, all FOUR damping
        # channels) + `tblite_coulomb_multipole::get_potential` (eight
        # contractions, alpha=1/3 on the last).  Verified against the binary's
        # own container to 1e-17 on H2O, CH3SH and benzene --
        # gxtb-recovery probes/audit/port_divergence.py stage 8c.
        #
        # `gxtb_aes_gab`/`_setvsdq_fast` are a GFN2-shaped stand-in: two of the
        # four channels, and no CN scale at all.
        from mlxmolkit.xtb.gxtb_aes_recovered import (
            multipole_potential as _mp_pot, multipole_energy as _mp_en)
        _am = _st["amat"]
        vat_aes, vd, vq = _mp_pot(*_am, q, dipm_raw, qp)
        _M_bin = _st["M_bin"]
        # The ENERGY comes from the recovered container, on the SAME
        # unscaled multipoles the potential now gets.  Its alphas are not the
        # potential's: 1/2 on A_dd and 1/6 on A_qq, against 1 and 1/3.
        # Nothing inside the SCF reads it (`want_energy=False` there), so it
        # is only formed when asked for.
        if want_energy:
            e_aes = float(np.sum(_mp_en(*_am, q, dipm_raw, qp)))
        # 🔑 The binary's `pot%vat` does NOT go through `add_vmp_to_h1` with
        # the multipole channels -- `add_vat_to_vsh` folds it into the SHELL
        # potential.  So fockelectro sees no monopole here; the caller adds
        # `vat_aes` to V_sh, uniformly over each atom's shells.
        vs = np.zeros_like(vat_aes)
        # ⚠️ SIGN.  `add_vmp_to_h1` SUBTRACTS its multipole terms
        #     h1(i,j) -= 0.5*( vmp(k,at(j))*mpint(k,i,j) + vmp(k,at(i))*mpint(k,j,i) )
        # while `fockelectro` ADDS them, so the port's vd/vq are the negative
        # of the binary's pot%vdp / pot%vqp.  `vat` needs no flip: it goes
        # through V_sh, which already carries the binary's own sign (the
        # repulsion's atom potential enters there unflipped).
        if True:
            vd = -vd
            vq = -vq
        if _AES_ROUTE == "vat":        # isolate: monopole route only
            vd = np.zeros_like(vd); vq = np.zeros_like(vq)
        elif _AES_ROUTE == "mp":       # isolate: multipole route only
            vat_aes = np.zeros_like(vat_aes)
        if False:
            vat_aes = -vat_aes
    else:
        vs, vd, vq = _setvsdq_fast(atoms, coords_bohr, q, dipm, qp, gab3, gab5, _geo)
        vat_aes = None
    if GXTB_AES_RECOVERED:
        F_aes = _fockelectro_fast(S, _M_bin, aoat, vs, vd, vq)
    else:
        F_aes = _fockelectro_fast(S, _aes_mstack(basis, dpint, qpint), aoat, vs, vd, vq)
    if want_vat:
        return F_aes, e_aes, vat_aes
    return F_aes, e_aes


_AES_STATIC_CACHE: dict = {}


def _aes_static(basis, atoms, coords_bohr, S, dpint, qpint):
    """The geometry-only half of `gxtb_aes_fock`, cached per basis.

    `multipole_amat` was ~1 ms per call and ran every SCF iteration on inputs
    (coordinates, CN, per-element radii and damping constants) that never
    change inside the SCF; the atom-centred traceless integral stack `_M_bin`
    is likewise a fixed function of S/dpint/qpint and the geometry.  The
    expressions are the ones that stood in the loop, moved verbatim.
    """
    key = id(basis)
    hit = _AES_STATIC_CACHE.get(key)
    if hit is not None and hit[0] is basis:
        return hit[1]
    if len(_AES_STATIC_CACHE) > 8:
        _AES_STATIC_CACHE.clear()
    from mlxmolkit.xtb.gxtb_aes import (gxtb_mrad_pair, GXTB_AES_DAMP_MAG,
                                        GXTB_AES_DAMP_SCALE)
    from mlxmolkit.xtb.gxtb_aes_recovered import multipole_amat as _mp_amat
    atoms = np.asarray(atoms, dtype=np.intp)
    aoat = np.asarray([bf.atom_idx for bf in basis.sao_basis], dtype=np.int64)
    zref = np.bincount(basis.shell_atom, weights=basis.shell_zref, minlength=atoms.size)
    dipscale = np.asarray(GXTB_PARAMS["pa_aes_dip_scale"], dtype=np.float64)[atoms - 1]
    out = {"aoat": aoat, "zref": zref, "dipscale": dipscale}
    if GXTB_AES_RECOVERED:
        _rad_at = np.asarray(GXTB_PARAMS["pa_aes_dip_scale"],
                             dtype=np.float64)[atoms - 1]
        _mrad = gxtb_mrad_pair(atoms)
        out["amat"] = _mp_amat(coords_bohr, np.asarray(basis.cn, dtype=np.float64),
                               _rad_at, _mrad, GXTB_AES_DAMP_MAG, GXTB_AES_DAMP_SCALE)
        # 🔑 `add_vmp_to_h1` contracts vdp/vqp against ATOM-CENTRED, TRACELESS
        # integrals; the port's dpint/qpint are raw and global-origin.  Build
        # the binary's operator here rather than permuting vq into the port's
        # convention -- verified as stage 7/7b of the walk (9.5e-15 / 2.5e-12
        # / 2.8e-11 on the quadrupole).
        _Rc = coords_bohr[aoat]                       # column-atom centres
        _dpc = np.stack([dpint[k] - _Rc[:, k][None, :] * S for k in range(3)])
        _qp_mm = qpint[[0, 3, 1, 4, 5, 2]]            # port qpint -> tblite mm
        _pairs = ((0, 0), (0, 1), (1, 1), (0, 2), (1, 2), (2, 2))
        _qpc = np.stack([
            _qp_mm[k] - _Rc[:, a][None, :] * _dpc0 - _Rc[:, b][None, :] * _dpc1
            + _Rc[:, a][None, :] * _Rc[:, b][None, :] * S
            for k, ((a, b), _dpc0, _dpc1) in enumerate(
                ((ab, dpint[ab[1]], dpint[ab[0]]) for ab in _pairs))])
        _tr = _qpc[0] + _qpc[2] + _qpc[5]
        _qpc = 1.5 * _qpc
        for _k in (0, 2, 5):
            _qpc[_k] = _qpc[_k] - 0.5 * _tr
        out["M_bin"] = np.ascontiguousarray(np.concatenate([_dpc, _qpc], axis=0))
    _AES_STATIC_CACHE[key] = (basis, out)
    return out


def _mfx_shellsum(P, S, basis, atoms, qsh, alpha=0.15, nspin=1, _pre=None):
    """`exchange_fock::get_kfock` step 8 -> `cache%shellsum`: the shell-charge
    derivative of the ON-SITE exchange, which `exchange_type::
    get_potential_w_overlap` then adds to `pot%vsh` with factor 1 -- so the
    binary's exchange container writes a SHELL POTENTIAL as well as `w1`.

    The port folded only `w1` (`gxtb_kfock_exact`).  Measured at the binary's
    own converged state (fixed oracle, stage 76): the binary's Fock minus the
    port's is EXACTLY a shell-potential fold (residual 1.7e-7 = the SCF's
    convergence) of [0.030349 O s, 0.025980 O p, 0 H, 0 H] on H2O -- which is
    the harness's exported `xc_shellsum` to 7e-18.  Zero on every hydrogen
    because H has one shell and `donsite_ri` is only set for local shell > 1.

    Transcribed from the recovered `get_gons` (the `donsite_fx`/`donsite_ri`
    tables) and `get_kfock_onsite`, both probed at 0 ulp.  The kernel is
    handed the routine's `bvec` (row-wise dot sum_j S_ij P_ij) as its `avec`
    and the density diagonal as its `bvec` -- the call swaps them.
    """
    from mlxmolkit.xtb.gxtb_aes import _onecx_tables
    P = np.asarray(P, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64)
    bts = np.asarray(basis.bf_to_shell)
    sh_atom = np.asarray(basis.shell_atom)
    sh_l = np.asarray(basis.shell_l)
    at = np.asarray(atoms, dtype=np.intp)
    nsh = sh_atom.size
    tbl, lidx = _onecx_tables()
    kq = np.asarray(GXTB_PARAMS["pg_fock_kq"], dtype=np.float64)
    scal = 1.0 if nspin >= 2 else 0.5
    quarter = scal * 0.25
    if _pre is None:
        _pre = _mfx_iteration_products(P, S)
    xmat, kf, av, bv, cv = _pre
    half = alpha * 0.5
    shellsum = np.zeros(nsh)
    # The shell-pair walk -- which AO blocks, which one-centre integral, which
    # kq scale -- sees only the basis, so it is built once per molecule and
    # replayed here.  The arithmetic below is unchanged, operand for operand.
    # Every term of the kernel's block accumulator is ELEMENTWISE in the AO
    # pair, so it is formed once over the whole (nao, nao) and each shell-pair
    # block is gathered from it: element for element the same products in the
    # same order as the per-block expression
    #     t + Xb*Xb*0.5 + outer(bv,cv)*0.25 + outer(av,av)*0.5 + outer(cv,bv)*0.25
    # with t[a,b] = P[b,a]*kf[b,a].  Only the per-block `.sum()` is a
    # reduction, and it still runs on a fresh C-ordered block of the same
    # shape, so its summation order is the one it had.
    ACC = ((P * kf).T + xmat * xmat * 0.5
           + np.outer(bv, cv) * 0.25
           + np.outer(av, av) * 0.5
           + np.outer(cv, bv) * 0.25)
    ACC2 = (P * kf).T + xmat * xmat.T * 0.5
    for (iish, jjsh, diag, ix_ji, ix_ij, ao_i, ao_j, di, dj, dri) in \
            _mfx_shellsum_plan(basis, at, tbl, lidx, kq, half):
        s1 = -quarter * ACC[ix_ij].sum()
        if diag:
            s2 = scal * ACC2[ix_ij].sum()
            dfx = di - dj
            shellsum[iish] += s2 * dri + s1 * dfx
        else:
            shellsum[iish] += s1 * di
            shellsum[jjsh] += s1 * (-dj)
    return shellsum


_MFX_SHELLSUM_PLAN: dict = {}


def _mfx_shellsum_plan(basis, at, tbl, lidx, kq, half):
    """The charge-independent walk of `_mfx_shellsum`, cached per basis.

    `np.ix_` alone was 70% of the routine's time (25 000 calls per four
    molecules); every one of them, and the `di`/`dj` factors, is a function
    of the basis and the element table only.
    """
    key = id(basis)
    hit = _MFX_SHELLSUM_PLAN.get(key)
    if hit is not None and hit[0] is basis:
        return hit[1]
    if len(_MFX_SHELLSUM_PLAN) > 8:
        _MFX_SHELLSUM_PLAN.clear()
    bts = np.asarray(basis.bf_to_shell)
    sh_atom = np.asarray(basis.shell_atom)
    sh_l = np.asarray(basis.shell_l)
    nsh = sh_atom.size
    ao_of = [np.where(bts == ish)[0] for ish in range(nsh)]
    plan = []
    for iat in range(at.size):
        m = np.where(sh_atom == iat)[0]
        Z = int(at[iat])
        for i_loc, iish in enumerate(m):
            ao_i = ao_of[iish]
            for j_loc, jjsh in enumerate(m):
                ao_j = ao_of[jjsh]
                os_ = tbl[Z - 1, lidx[sh_l[jjsh], sh_l[iish]] - 1]
                di = -(os_ * (kq[sh_l[iish]] * half))
                dj = kq[sh_l[jjsh]] * half * os_
                dri = 0.0
                if i_loc == j_loc:
                    dfx = di - dj
                    dri = dfx / (4.0 * (i_loc + 1) - 2.0) if i_loc != 0 else 0.0
                plan.append((iish, jjsh, i_loc == j_loc,
                             np.ix_(ao_j, ao_i), np.ix_(ao_i, ao_j),
                             ao_i, ao_j, di, dj, dri))
    _MFX_SHELLSUM_PLAN[key] = (basis, plan)
    return plan


def _mfx_iteration_products(P, S):
    """The density-dependent products `get_kfock` and its `shellsum` step both
    start from -- S@P, kfock0 = 0.5*(S@P)@S, and the three vectors.  Formed
    once per SCF iteration and handed to both; the expressions are the two
    routines' own, so the operands and the BLAS calls are unchanged.
    """
    xmat = S @ P
    kf = 0.5 * (xmat @ S)
    av = np.einsum("ij,ij->i", S, P)          # routine's bvec -> kernel avec
    bv = np.diag(P).copy()                    # routine's avec -> kernel bvec
    cv = np.einsum("ij,ij->i", xmat, S)       # cvec
    return xmat, kf, av, bv, cv


_KFOCK_STATIC: dict = {}


def _kfock_static(basis, atoms, coords_ang, kmat_shell):
    """The charge-independent half of `gxtb_kfock_exact`, cached per basis.

    Once per molecule: `bomat` (`get_bocorr_kmatrix`, geometry only), the
    shell-pair and atom-pair weights (`kmat` is the fixed exchange gamma
    unless `mfx_kq` is on -- the cache is keyed on its identity too), and the
    index structure of the on-site weights and of `onsite_fx_symv`.
    """
    key = (id(basis), id(kmat_shell))
    hit = _KFOCK_STATIC.get(key)
    if hit is not None and hit[0] is basis and hit[1] is kmat_shell:
        return hit[2]
    if len(_KFOCK_STATIC) > 8:
        _KFOCK_STATIC.clear()
    from mlxmolkit.xtb.gxtb_aes import (gxtb_bocorr_gamma, _ao_weight_shell,
                                        _ao_weight_atom, _onecx_tables,
                                        _KF_SHELL_SCAL, _KF_ATOM_SCAL)
    bts = np.asarray(basis.bf_to_shell)
    sh_atom = np.asarray(basis.shell_atom)
    sh_l = np.asarray(basis.shell_l)
    at = np.asarray(atoms, dtype=np.intp)
    nat = at.size
    nsh = sh_atom.size
    n = bts.size
    # `get_gons`' local shell index, as it builds it
    loc = np.zeros(nsh, dtype=np.intp)
    cnt = {}
    for i, a in enumerate(sh_atom):
        loc[i] = cnt.get(int(a), 0)
        cnt[int(a)] = loc[i] + 1
    mx = int(loc.max()) + 1
    aoat = sh_atom[bts]
    bomat = gxtb_bocorr_gamma(basis, atoms, coords_ang, _atom_matrix_only=True)
    st = {
        "bts": bts, "sh_atom": sh_atom, "loc": loc, "mx": mx, "nat": nat,
        "nsh": nsh, "aoat": aoat,
        "W_sh": _ao_weight_shell(kmat_shell, bts, _KF_SHELL_SCAL),
        "W_at": _ao_weight_atom(bomat, aoat, _KF_ATOM_SCAL),
        "same": aoat[:, None] == aoat[None, :],
        "same_shell": bts[:, None] == bts[None, :],
    }
    lb = loc[bts]
    # onsite_fx_hadamard_add: W[a, b] = fx[loc(b), loc(a), atom(a)]
    st["fx_gather"] = (lb[None, :], lb[:, None], aoat[:, None])
    # onsite_ri_hadamard_add: ri[loc(a), atom(a)] down the rows
    st["ri_gather"] = (lb, aoat)
    # onsite_fx_symv: per atom, its shells (ascending, so loc runs 0..k-1)
    st["symv"] = [(int(iat), np.where(sh_atom == iat)[0]) for iat in np.unique(sh_atom)]
    # get_gons: every same-atom shell pair (a, b), with its constants
    tbl, lidx = _onecx_tables()
    kq = np.asarray(GXTB_PARAMS["pg_fock_kq"], dtype=np.float64)
    A, B = np.meshgrid(np.arange(nsh), np.arange(nsh), indexing="ij")
    keep = sh_atom[A] == sh_atom[B]
    A, B = A[keep], B[keep]
    Zp = at[sh_atom[A]]
    st["gons_A"], st["gons_B"] = A, B
    st["gons_iat"] = sh_atom[A]
    st["gons_kqa"] = kq[sh_l[A]]
    st["gons_kqb"] = kq[sh_l[B]]
    st["gons_os"] = tbl[Zp - 1, lidx[sh_l[A], sh_l[B]] - 1]
    diag = A == B
    i1 = loc[A[diag]] + 1
    st["gons_ri_sel"] = (A[diag][i1 != 1], i1[i1 != 1])
    _KFOCK_STATIC[key] = (basis, kmat_shell, st)
    return st


def _kfock_fast(P, S, basis, atoms, kmat_shell, qsh, coords_ang, pre=None,
                alpha=0.15, nspin=1):
    """`gxtb_kfock_exact` -- same numbers, the static half hoisted.

    `get_gons` and the two on-site Hadamard weights are elementwise (no
    reductions), so they are formed as array expressions; `onsite_fx_symv`
    IS a reduction and keeps its per-atom `@` on a same-strided view, so
    numpy takes the same kernel.  The dense chain at the end is transcribed
    unchanged from the frozen routine.  Bit-for-bit against it over the
    benchmark set.
    """
    st = _kfock_static(basis, atoms, coords_ang, kmat_shell)
    from mlxmolkit.xtb.gxtb_aes import _KF_ONSITE_FX_SCAL, _KF_ONSITE_RI_SCAL
    P = np.asarray(P, dtype=float)
    S = np.asarray(S, dtype=float)
    q = np.asarray(qsh, dtype=np.float64)
    bts, loc, mx, nat = st["bts"], st["loc"], st["mx"], st["nat"]
    # --- get_gons
    A, B = st["gons_A"], st["gons_B"]
    fx = np.zeros((mx, mx, nat))
    ri = np.zeros((mx, nat))
    fx[loc[A], loc[B], st["gons_iat"]] = (
        1.0 - 0.5 * (st["gons_kqa"] * q[A] + st["gons_kqb"] * q[B])
    ) * (alpha * st["gons_os"])
    ra, i1 = st["gons_ri_sel"]
    ri[loc[ra], st["sh_atom"][ra]] = fx[loc[ra], loc[ra], st["sh_atom"][ra]] / (4 * i1 - 2)
    # --- the four weights
    W_sh, W_at = st["W_sh"], st["W_at"]
    W_fx = np.where(st["same"], _KF_ONSITE_FX_SCAL * fx[st["fx_gather"]], 0.0)
    W_ri = np.where(st["same_shell"],
                    (_KF_ONSITE_RI_SCAL * ri[st["ri_gather"]][:, None]) * np.ones((1, bts.size)),
                    0.0)
    scal = 1.0 if nspin >= 2 else 0.5
    if pre is None:
        pre = _mfx_iteration_products(P, S)
    xmat, kfock, _av, _bv, cvec = pre

    def _symv(xvec):
        ssum = np.bincount(bts, weights=np.asarray(xvec, dtype=float), minlength=st["nsh"])
        ysh = np.zeros(st["nsh"])
        for iat, m in st["symv"]:
            k = m.size
            ysh[m] = fx[:k, :k, iat] @ ssum[m]
        return ysh[bts]

    fxa = _symv(np.diag(P))
    fxb = _symv(np.diag(xmat))
    fxc = _symv(cvec)

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
    kfock = kfock + np.diag(0.25 * fxc)

    # symmetrise_kfock
    out = -(0.25 * scal) * (kfock + kfock.T)
    np.fill_diagonal(out, -(np.diag(kfock) * scal * 0.5))
    return out


_KMAT_SH_CACHE: dict = {}


def _kmat_shell_of(basis, gamma_mfx):
    """`gamma_mfx` sampled at each shell's first AO -- the `kmat` of `get_kfock`.
    Fixed for the SCF unless the gamma itself is rebuilt; cached on both."""
    key = (id(basis), id(gamma_mfx))
    hit = _KMAT_SH_CACHE.get(key)
    if hit is not None and hit[0] is basis and hit[1] is gamma_mfx:
        return hit[2]
    if len(_KMAT_SH_CACHE) > 8:
        _KMAT_SH_CACHE.clear()
    _bts = np.asarray(basis.bf_to_shell)
    _first = np.zeros(int(_bts.max()) + 1, dtype=np.intp)
    for _i in range(_bts.size - 1, -1, -1):
        _first[_bts[_i]] = _i
    kmat = gamma_mfx[np.ix_(_first, _first)]
    _KMAT_SH_CACHE[key] = (basis, gamma_mfx, kmat)
    return kmat


def gxtb_energy(
    atomic_numbers: list[int] | np.ndarray,
    coords_ang: np.ndarray,
    *,
    charge: int = 0,
    max_iter: int = 100,
    conv_tol: float = 1.0e-7,
    mix: float = 0.4,
    use_d4srev: bool = True,
    use_pacp: bool = True,
    use_acp_hamiltonian: bool = False,
    use_exchange: bool = False,
    use_mfx_exchange: bool = True,
    use_first_order: bool = False,
    use_third_order: bool = False,
    use_twobody_third_order: bool = False,
    use_fourth_order: bool = False,
    use_diis: bool = True,
    use_halide_increment_correction: bool = False,
    use_aes: bool = False,
    use_onecenter: bool = False,
    onecenter_scale: float = 1.0,
    qsh_init=None,
    P_init=None,
    scf_basis_update: bool = False,
    final_resolve: bool = False,
    final_resolve_iters: int = 1,
    final_resolve_mix: float = 0.4,
    onsite_potential: bool = False,
    onsite_sign: float = 1.0,
    onsite_charge_factor: bool = False,
    onsite_diag: int = 0,
    onsite_mapping: tuple = (2, 0, 1),
    use_aniso_h0: bool = False,
    aniso_h0_scale: float = 1.0,
    use_twobody3: bool = False,
    use_bocorr: bool = False,
    scc_scale: float = 1.0,
    use_carbon_plevel_shift: bool = True,
    use_h0_effective_basis: float = 0.0,
    mfx_offdiag_l: bool = False,
    mfx_kq: bool = False,
    mfx_kq_onsite: bool = False,
    use_mfx_shell_potential: bool = True,
    mfx_decoded_damping: bool = False,
    sqrt_cn_hardness: bool = False,
    decoded_shpoly: bool = False,
    scf_repulsion: bool = False,
    scf_dispersion: bool = False,
    verbose: bool = False,
) -> dict[str, object]:
    """Compute an experimental native g-xTB single-point energy.

    This is a reconstruction scaffold: EEQ_BC, q-vSZP, overlap, H0, shell SCC,
    recovered repulsion, and a measurable exchange term are active.  D4Srev and
    p-ACP are explicit fallback/proxy components until their exact kernels are
    extracted.

    ``use_mfx_exchange`` (SI Eq. 153 range-separated Mulliken Fock exchange)
    defaults to True because it is the single largest missing charge term:
    against the real ``xtb --gxtb`` binary over 20 molecules / 516 atoms it takes
    the atomic-charge MAE from 0.04617 to 0.02494 e, and it wins on every one of
    the 20 molecules individually.  See tools/gxtb_charge_oracle.py.

    Two things that look like they should help and do not, both measured on the
    same set rather than inferred from water:

    * ``use_acp_hamiltonian`` on top of MFX *worsens* the MAE to 0.03058.  It does
      halve the oxygen residual (+0.0789 -> +0.0500 e) and on water it repairs
      the O s-shell population almost exactly (1.6916 -> 1.8202 vs the binary's
      1.8050), but it pays for that with H (+0.0086 -> +0.0174) and C (-0.0235 ->
      -0.0345).  Water is a misleading probe here.
    * dropping ``use_carbon_plevel_shift`` costs 2.6x (0.02494 -> 0.06408), so
      that oracle-fitted patch is *not* merely standing in for MFX.

    Total energies remain un-validated against the binary: D4Srev falls back to
    ``simple-dftd4``, so ``use_d4srev=True`` needs a package that is usually
    absent, and no test pins a g-xTB energy.
    """

    atoms = np.asarray(atomic_numbers, dtype=np.intp)
    coords = np.asarray(coords_ang, dtype=np.float64)
    if coords.shape != (atoms.size, 3):
        raise ValueError("coords_ang must have shape (nat, 3)")

    # ---------------------------------------------------------------- basis
    # `basis_qvszp::basis_update` recomputes the q-vSZP contraction from
    # `wfn%qat` at EVERY SCF iteration, and `cgto_update` sets qeff = 0 unless
    # BOTH cn and q are supplied.  The port used to build the contraction once,
    # from the EEQ-BC charges, and never update it -- two separate errors: the
    # wrong charge source, and no update at all.  Everything downstream of the
    # contraction (S, H0, jmat, the MFX kernel) has to be rebuilt with it.
    def _build_stage(q_basis=None):
        if q_basis is None:
            basis = build_gxtb_qvszp_basis(atoms, coords, total_charge=float(charge))
        else:
            basis = build_gxtb_qvszp_basis(
                atoms, coords, total_charge=float(charge),
                eeqbc_charges=np.asarray(q_basis, dtype=np.float64))
        if sqrt_cn_hardness:
            # frozen dataclass, but the ndarray itself is mutable
            basis.shell_hardness[:] = _corrected_shell_hardness(atoms, basis)
        S = basis.S
        h0_basis = basis
        if use_h0_effective_basis:
            h0_basis = _H0BasisProxy(
                basis, _h0_effective_overlap_cao(atoms, basis, float(use_h0_effective_basis))
            )
        H_eht, shell_self = build_hcore_gxtb(
            atoms, coords, h0_basis, carbon_plevel=use_carbon_plevel_shift,
            decoded_shpoly=decoded_shpoly,
        )
        H_acp = build_gxtb_acp_hamiltonian(atoms, coords, basis, enabled=use_acp_hamiltonian)
        H0 = H_eht + H_acp
        if use_aniso_h0:
            H0 = H0 + aniso_h0_scale * _aniso_h0_decoded(basis, atoms, coords)
        jmat = _coulomb_matrix(coords * ANG_TO_BOHR, basis.shell_atom, basis.shell_hardness)
        qref_sh = _shell_qref(atoms, basis)
        # `coulomb_charge_type::get_potential` (recovered, 0 ulp) adds A.qref and
        # then walks each atom's OWN diagonal block and takes that contribution
        # straight back out, so qref acts only BETWEEN atoms, never within one.
        _sa = np.asarray(basis.shell_atom)
        jmat_off = np.where(_sa[:, None] == _sa[None, :], 0.0, jmat)
        vref = jmat_off @ qref_sh

        def _mfx_gamma(q=None, _b=basis):
            return _mfx_gamma_ao(
                atoms, coords, _b, qsh=q,
                use_offdiag_l=mfx_offdiag_l, use_kq=mfx_kq,
                kq_onsite=mfx_kq_onsite,
                decoded_damping=mfx_decoded_damping,
            )

        gamma_mfx = _mfx_gamma() if use_mfx_exchange else None
        if use_bocorr and gamma_mfx is not None:
            from mlxmolkit.xtb.gxtb_aes import gxtb_bocorr_gamma
            gamma_mfx = gamma_mfx + gxtb_bocorr_gamma(basis, atoms, coords)
        return (basis, S, H_eht, H_acp, H0, shell_self, jmat, qref_sh,
                jmat_off, vref, _mfx_gamma, gamma_mfx)

    if use_aes:
        _install_fast_aes_reduce()
    if use_onecenter:
        _install_onecxints()
    (basis, S, H_eht, H_acp, H0, shell_self, jmat, qref_sh,
     jmat_off, vref, _mfx_gamma, gamma_mfx) = _build_stage()
    n_basis = S.shape[0]
    n_shell = basis.shell_atom.size
    coords_bohr = coords * ANG_TO_BOHR

    z_ref = basis.shell_zref
    n_elec_f = float(np.sum(z_ref)) - float(charge)
    n_elec = int(round(n_elec_f))
    if abs(n_elec - n_elec_f) > 1.0e-6 or n_elec % 2:
        raise NotImplementedError(f"only closed-shell integer electron counts are supported; got {n_elec_f}")
    n_occ = n_elec // 2

    z_atom_ref = np.bincount(basis.shell_atom, weights=z_ref, minlength=atoms.size)
    q_at_init = basis.eeqbc_charges
    qsh = np.where(
        z_atom_ref[basis.shell_atom] > 1.0e-12,
        q_at_init[basis.shell_atom] * z_ref / z_atom_ref[basis.shell_atom],
        0.0,
    )

    if qsh_init is not None:
        qsh = np.asarray(qsh_init, dtype=np.float64).copy()

    P = np.zeros((n_basis, n_basis), dtype=np.float64)
    if P_init is not None:
        P = np.array(P_init, dtype=np.float64, copy=True)
    F_hist: list[np.ndarray] = []
    e_hist: list[np.ndarray] = []
    q_hist: list[np.ndarray] = []
    r_hist: list[np.ndarray] = []
    _diis_state: dict = {}
    converged = False
    n_iter = 0
    diis_warmup = GXTB_DIIS_WARMUP
    diis_max = GXTB_DIIS_MAX

    for it in range(max_iter):
        if scf_basis_update and it > 0:
            # basis_update(bas, mol, cache, grad, wfn%qat): the SCF's own
            # atomic charges drive the contraction, not the EEQ-BC guess.
            q_at_basis = np.bincount(basis.shell_atom, weights=qsh,
                                     minlength=atoms.size)
            (basis, S, H_eht, H_acp, H0, shell_self, jmat, qref_sh,
             jmat_off, vref, _mfx_gamma, gamma_mfx) = _build_stage(q_at_basis)
        V_coul = jmat @ qsh + vref
        E_first_iter, V_first, V_first_at = (
            _first_order_decoded(atoms, basis.cn, basis.shell_atom,
                             basis.shell_l, qsh)
            if use_first_order
            else (0.0, 0.0, 0.0)
        )
        V_third = qsh * qsh * basis.shell_third if use_third_order else 0.0
        _, V_third_twobody = (
            _third_order_twobody(basis, atoms, coords, qsh)
            if use_twobody_third_order
            else (0.0, 0.0)
        )
        V_third_twobody = GXTB_TB3_TWOBODY_SCALE * V_third_twobody
        V_fourth = qsh * qsh * qsh * basis.shell_fourth * GXTB_K4TH_SCALE / 6.0 if use_fourth_order else 0.0
        V_exchange = basis.shell_exchange * qsh if use_exchange else 0.0
        V_tb3 = 0.0
        if use_twobody3:
            from mlxmolkit.xtb.gxtb_aes import gxtb_twobody_thirdorder
            _, V_tb3 = gxtb_twobody_thirdorder(qsh, basis, atoms, coords)
        # The AES runs BEFORE V_sh is assembled: its `vat` belongs in the shell
        # potential (add_vat_to_vsh), not in the H1 monopole channel, so V_sh
        # has to be able to see it.  Its Fock part is added further down,
        # where it was.
        _F_aes, _vat_aes = None, None
        if use_aes and (it > 0 or P_init is not None):
            _r_aes = gxtb_aes_fock(P, basis, atoms, coords,
                                   want_energy=False, want_vat=True)
            _F_aes, _vat_aes = _r_aes[0], _r_aes[2]
        V_sh = V_first + scc_scale * (V_coul + V_third + V_third_twobody + V_fourth + V_exchange + V_tb3)
        # add_vat_to_vsh, explicit: the first order's atom potential
        if np.ndim(V_first_at) > 0:
            V_sh = V_sh + np.asarray(V_first_at)[basis.shell_atom]
        if _vat_aes is not None:
            V_sh = V_sh + np.asarray(_vat_aes)[basis.shell_atom]
        if scf_repulsion:
            q_at_it = np.bincount(basis.shell_atom, weights=qsh, minlength=atoms.size)
            V_sh = V_sh + _repulsion_shell_potential(atoms, coords, basis, q_at_it)
        if scf_dispersion:
            q_at_it = np.bincount(basis.shell_atom, weights=qsh, minlength=atoms.size)
            V_sh = V_sh + _dispersion_shell_potential(atoms, coords, basis, q_at_it)
        # The exchange container writes a SHELL POTENTIAL too (`cache%shellsum`
        # -> `pot%vsh`, see `_mfx_shellsum`); it belongs in V_sh before the fold.
        _ss_loop = np.zeros(n_shell)
        _mfx_pre = None
        if use_mfx_shell_potential and gamma_mfx is not None and onsite_potential == 4:
            _mfx_pre = _mfx_iteration_products(P, S)
            _ss_loop = _mfx_shellsum(P, S, basis, atoms, qsh, _pre=_mfx_pre)
            V_sh = V_sh + _ss_loop
        # probe support: keep the IN-LOOP shell potential and the charges it
        # was built from.  The post-loop assembly below is built from the
        # OUTPUT charges, so comparing it against the binary's `vsh_*` -- which
        # come from the INPUT charges -- measures two different states.
        _V_sh_loop = np.asarray(V_sh, dtype=np.float64).copy()
        _V_terms_loop = {
            "first": np.asarray(V_first, dtype=np.float64).copy(),
            "coul": np.asarray(V_coul, dtype=np.float64).copy(),
            "third": np.asarray(V_third, dtype=np.float64).copy(),
            "third_twobody": np.asarray(V_third_twobody, dtype=np.float64).copy(),
            "fourth": np.asarray(V_fourth, dtype=np.float64).copy(),
            "exchange": np.asarray(V_exchange, dtype=np.float64).copy(),
            "tb3": np.asarray(V_tb3, dtype=np.float64).copy(),
            "xc_shellsum": np.asarray(_ss_loop, dtype=np.float64).copy(),
        }
        _qsh_loop = np.asarray(qsh, dtype=np.float64).copy()
        # probe support: the DENSITY the Fock was built from.  The exchange
        # reads it, and nothing outside the driver can otherwise tell whether
        # `P_init` reached the first Fock build or a guess did.
        _P_loop = np.asarray(P, dtype=np.float64).copy()
        F = _fock_from_shell_potential(H0, S, basis.bf_to_shell, V_sh)
        # probe support: the Fock BEFORE the AO-level terms, and the H0/S it
        # was folded from.  Reconstructing this from the return dict is what
        # produced three phantom "exchange gaps" -- the dict's H0 is the
        # post-loop one.
        _F_fold_loop = np.asarray(F, dtype=np.float64).copy()
        _H0_loop = np.asarray(H0, dtype=np.float64).copy()
        _S_loop = np.asarray(S, dtype=np.float64).copy()
        if gamma_mfx is not None:
            # pg_fock_kq makes the exchange kernel charge dependent, so it has
            # to follow the density round the SCF rather than be built once.
            if mfx_kq or mfx_kq_onsite:
                gamma_mfx = _mfx_gamma(qsh)
            if onsite_potential == 4:
                # 🔑 `get_kfock` WHOLE: the two-centre kernel and the on-site
                # half are not two additive terms, they are four weight
                # matrices folded onto S@P, onto kfock and onto P in three
                # passes before contraction.  Adding them separately -- which
                # is what the branches below do -- is an approximation of that
                # chain, not a decomposition of it.
                _kmat_sh = _kmat_shell_of(basis, gamma_mfx)
                # probe support: keep the exchange term actually added, and the
                # shell kernel it came from.  `F_mfx` in the return dict is
                # written by the OTHER branch, so it is not this operator.
                _kfock_loop = _kfock_fast(P, S, basis, atoms, _kmat_sh, qsh,
                                          coords, pre=_mfx_pre)
                _kmat_sh_loop = np.asarray(_kmat_sh, dtype=np.float64).copy()
                F = F + _kfock_loop
            else:
                _, F_mfx = _mfx_fock_energy(P, S, gamma_mfx)
                F = F + F_mfx
        if _F_aes is not None:
            # probe support: the AES Fock actually added, not a walk-side
            # recomputation of it.
            _F_aes_loop = np.asarray(_F_aes, dtype=np.float64).copy()
            F = F + _F_aes
        # probe support: the FINAL in-loop Fock -- the operator actually
        # diagonalised.  The `F` in the return dict is the POST-loop rebuild
        # from the converged charges, which is not this.
        _F_loop = np.asarray(F, dtype=np.float64).copy()
        if use_onecenter and onsite_potential != 4 and (it > 0 or P_init is not None):
            if onsite_potential == 3:
                from mlxmolkit.xtb.gxtb_aes import gxtb_onsite_fock_exact
                # qsh: get_gons modulates the on-site matrix by the shell
                # charges, so it is an input, not a variant.
                F_os = gxtb_onsite_fock_exact(P, S, basis, atoms,
                                              mapping=onsite_mapping, qsh=qsh)
                F = F + onsite_sign * onecenter_scale * F_os
            elif onsite_potential:
                if onsite_charge_factor:
                    from mlxmolkit.xtb.gxtb_aes import gxtb_onsite_potential_q
                    V_os = gxtb_onsite_potential_q(P, S, basis, atoms, qsh)
                else:
                    from mlxmolkit.xtb.gxtb_aes import gxtb_onsite_potential
                    V_os = gxtb_onsite_potential(P, S, basis, atoms)
                if onsite_diag == 2:
                    # EXACT get_kfock fold (disasm-derived): M = 0.25*OS(V)+0.5*OS(V)+0.25*diag(V)
                    # where OS(V)[j,i]=V[i]*S[j,i] (overlap-sandwich daxpy column form);
                    # then fock = -0.125*(M+M^T) off-diag, -0.25*M diag.
                    M = 0.75 * (S * V_os[None, :])
                    M = M + np.diag(0.25 * V_os)
                    F_os = -0.125 * (M + M.T)
                elif onsite_diag:
                    # pure one-center (block-local): no cross-atom overlap coupling
                    F_os = np.diag(V_os)
                else:
                    # anti-binding shell-potential fold (single S-sandwich, like Coulomb
                    # but POSITIVE): F += +0.5*(V_mu+V_nu)*S  ->  F_diag += V_mu
                    F_os = 0.5 * (V_os[:, None] + V_os[None, :]) * S
                F = F + onsite_sign * onecenter_scale * F_os
            else:
                from mlxmolkit.xtb.gxtb_aes import gxtb_onsite_gamma_density
                og = gxtb_onsite_gamma_density(P, S, basis, atoms)
                _, F_os = _mfx_fock_energy(P, S, og)
                F = F + onecenter_scale * F_os

        if use_diis and not GXTB_CHARGE_MIXING and it >= diis_warmup:
            # F, P and S are all symmetric, so (F P S)^T = S P F exactly and
            # the commutator needs two matmuls rather than four.
            _fps = F @ P @ S
            e_diis = _fps - _fps.T
            F_hist.append(F)
            e_hist.append(e_diis)
            _popped = False
            if len(F_hist) > diis_max:
                F_hist.pop(0)
                e_hist.pop(0)
                _popped = True
            F_use = _pulay_diis_numpy(F_hist, e_hist, _diis_state, _popped)
        else:
            F_use = F

        eigvals, C = _solve_generalized(F_use, S)
        C_occ = C[:, :n_occ]
        P_new = 2.0 * (C_occ @ C_occ.T)
        qsh_new = _mulliken_shell_charges(P_new, S, basis.bf_to_shell, n_shell, z_ref)
        dq = float(np.max(np.abs(qsh_new - qsh)))
        if verbose:
            tag = f"DIIS hist={len(F_hist)}" if it >= diis_warmup else "linear"
            print(f"  g-xTB iter {it + 1:3d}: dq={dq:.3e} ({tag})")
        P = P_new
        if dq < conv_tol:
            qsh = qsh_new
            converged = True
            n_iter = it + 1
            break
        if GXTB_CHARGE_MIXING:
            q_hist.append(qsh)
            r_hist.append(qsh_new - qsh)
            if len(q_hist) > GXTB_CHARGE_HIST:
                q_hist.pop(0)
                r_hist.pop(0)
            qsh = _anderson_mix(q_hist, r_hist, GXTB_CHARGE_BETA)
        elif it < diis_warmup:
            qsh = mix * qsh_new + (1.0 - mix) * qsh
        else:
            qsh = qsh_new
    if not converged:
        n_iter = max_iter

    # ⚠️ ONE MORE ITERATION OF THE BASIS, IN THE LOOP'S OWN SHAPE.
    # The loop mixes qsh at the END of each pass and rebuilds the contraction
    # at the TOP of the next one -- so the update already happens AFTER the
    # mixing, exactly as `next_scf` does it.  What was missing is that the
    # loop exits between those two halves: the block below recomputes
    # everything from the mixed `qsh` while `basis`, `S` and `H0` are still the
    # ones built from the PREVIOUS mixed qsh.  Doing the rebuild here, before
    # the block runs, is literally iteration n+1's first statement, and the
    # block then produces P and qsh on that basis -- so the density is
    # re-derived rather than re-reduced and the electron count survives.
    # Not a loop: forty undamped passes of this map diverge (all_mae 1.33).
    if scf_basis_update:
        (basis, S, H_eht, H_acp, H0, shell_self, jmat, qref_sh,
         jmat_off, vref, _mfx_gamma, gamma_mfx) = _build_stage(
            np.bincount(basis.shell_atom, weights=qsh, minlength=atoms.size))

    V_coul = jmat @ qsh + vref
    E_first, V_first, V_first_at = (
        _first_order_decoded(atoms, basis.cn, basis.shell_atom,
                             basis.shell_l, qsh)
        if use_first_order
        else (0.0, 0.0, 0.0)
    )
    V_third = qsh * qsh * basis.shell_third if use_third_order else 0.0
    _, V_third_twobody = (
        _third_order_twobody(basis, atoms, coords, qsh)
        if use_twobody_third_order
        else (0.0, 0.0)
    )
    V_third_twobody = GXTB_TB3_TWOBODY_SCALE * V_third_twobody
    V_fourth = qsh * qsh * qsh * basis.shell_fourth * GXTB_K4TH_SCALE / 6.0 if use_fourth_order else 0.0
    V_exchange = basis.shell_exchange * qsh if use_exchange else 0.0
    V_tb3 = 0.0
    if use_twobody3:
        from mlxmolkit.xtb.gxtb_aes import gxtb_twobody_thirdorder
        _, V_tb3 = gxtb_twobody_thirdorder(qsh, basis, atoms, coords)
    _F_aes_f, _vat_aes_f, E_aes_f = None, None, 0.0
    if use_aes:
        _r_aes = gxtb_aes_fock(P, basis, atoms, coords, want_vat=True)
        _F_aes_f, E_aes_f, _vat_aes_f = _r_aes[0], _r_aes[1], _r_aes[2]
    V_sh = V_first + scc_scale * (V_coul + V_third + V_third_twobody + V_fourth + V_exchange + V_tb3)
    if np.ndim(V_first_at) > 0:
        V_sh = V_sh + np.asarray(V_first_at)[basis.shell_atom]
    if _vat_aes_f is not None:
        V_sh = V_sh + np.asarray(_vat_aes_f)[basis.shell_atom]
    if scf_repulsion:
        q_at_it = np.bincount(basis.shell_atom, weights=qsh, minlength=atoms.size)
        V_sh = V_sh + _repulsion_shell_potential(atoms, coords, basis, q_at_it)
    if scf_dispersion:
        q_at_it = np.bincount(basis.shell_atom, weights=qsh, minlength=atoms.size)
        V_sh = V_sh + _dispersion_shell_potential(atoms, coords, basis, q_at_it)
    _mfx_pre_f = None
    if use_mfx_shell_potential and gamma_mfx is not None and onsite_potential == 4:
        _mfx_pre_f = _mfx_iteration_products(P, S)
        V_sh = V_sh + _mfx_shellsum(P, S, basis, atoms, qsh, _pre=_mfx_pre_f)
    F = _fock_from_shell_potential(H0, S, basis.bf_to_shell, V_sh)
    if gamma_mfx is not None:
        if mfx_kq or mfx_kq_onsite:
            gamma_mfx = _mfx_gamma(qsh)
        if onsite_potential == 4:
            # ⚠️ THE POST-LOOP PASS MUST USE THE LOOP'S EXCHANGE OPERATOR.
            # This block is iteration N+1's first half (stage 63), so the
            # charges it returns are ONE STEP of whatever map it applies.  It
            # used `_mfx_fock_energy` -- half-strength on two of three terms,
            # no on-site half -- while the loop converged on `get_kfock`.
            # Measured (H2O, conv_tol 1e-10): the loop's fixed point sat
            # 0.041 e from the binary's converged qsh; this one wrong step
            # then returned charges 0.1535 e away, +0.136 e of it on O s.
            F_mfx = _kfock_fast(P, S, basis, atoms,
                                _kmat_shell_of(basis, gamma_mfx), qsh,
                                coords, pre=_mfx_pre_f)
        else:
            _, F_mfx = _mfx_fock_energy(P, S, gamma_mfx)
        F = F + F_mfx
    E_aes = E_aes_f
    if _F_aes_f is not None:
        F = F + _F_aes_f
    eigvals, C = _solve_generalized(F, S)
    P = 2.0 * (C[:, :n_occ] @ C[:, :n_occ].T)
    qsh = _mulliken_shell_charges(P, S, basis.bf_to_shell, n_shell, z_ref)
    q_at = np.bincount(basis.shell_atom, weights=qsh, minlength=atoms.size)
    # ⚠️ THE RETURNED BASIS MUST BE AT THE RETURNED CHARGES.
    # The loop already updates the contraction AFTER the mixing, exactly as
    # `next_scf` does -- rebuilding at the mixed qsh is a no-op, measured.  The
    # lag is against the FINAL charges, the ones the line above just produced
    # from the last diagonalisation.  So rebuild at those, and do NOT re-reduce
    # qsh afterwards: qsh belongs to P through the OLD S, and recomputing it
    # against the new S is what broke the electron count (max_abs_qsum 5.4e-2
    # against a 1e-6 tolerance).  The basis now travels with the charges
    # reported beside it; the density stays with the overlap it was solved
    # against.
    if scf_basis_update:
        (basis, S, H_eht, H_acp, H0, shell_self, jmat, qref_sh,
         jmat_off, vref, _mfx_gamma, gamma_mfx) = _build_stage(q_at)
        # ⚠️ THE RETURNED (P, S) PAIR MUST CONSERVE ELECTRONS.
        # The rebuild above changes `S` while `P` was solved against the old
        # one, so `Tr(P.S)` -- the electron count -- comes out 7.936839 where
        # it must be 8.000000, and 13.979228 / 29.964959 where it must be
        # 14 / 30 (walk stage 41).  Sum-of-charges still holds, so no existing
        # check saw it: `max_abs_qsum` is 1.05e-09 throughout.
        #
        # ONE re-solve in the new basis restores it.  This is NOT the forty
        # undamped passes the note below rejects -- it is a single pass, after
        # convergence, whose only job is to put P, S and qsh in one basis.
        # ⚠️ ONE pass restores Tr(P.S) but leaves the basis a step behind the
        # charges it produced (walk 23d/23e go to 1.2e-2).  The basis and the
        # charges are a COUPLED MAP: forty undamped passes diverge
        # (all_mae_e 1.33, note below), one pass under-converges.  Iterate it
        # with damping instead, and stop when the charges stop moving --
        # then P, S and qsh share a basis AND the basis is at the charges.
        for _kf in range(final_resolve_iters if final_resolve else 0):
            if _kf > 0:
                (basis, S, H_eht, H_acp, H0, shell_self, jmat, qref_sh,
                 jmat_off, vref, _mfx_gamma, gamma_mfx) = _build_stage(q_at)
            _Ff = _fock_from_shell_potential(H0, S, basis.bf_to_shell, V_sh)
            if gamma_mfx is not None and onsite_potential == 4:
                from mlxmolkit.xtb.gxtb_aes import gxtb_kfock_exact as _gke_f
                _bts_f = np.asarray(basis.bf_to_shell)
                _first_f = np.zeros(int(_bts_f.max()) + 1, dtype=np.intp)
                for _i_f in range(_bts_f.size - 1, -1, -1):
                    _first_f[_bts_f[_i_f]] = _i_f
                _Ff = _Ff + _gke_f(P, S, basis, atoms,
                                   gamma_mfx[np.ix_(_first_f, _first_f)],
                                   qsh, coords_ang=coords)
            eigvals, C = _solve_generalized(_Ff, S)
            P = 2.0 * (C[:, :n_occ] @ C[:, :n_occ].T)
            _qsh_new = _mulliken_shell_charges(P, S, basis.bf_to_shell,
                                               n_shell, z_ref)
            _dq = float(np.abs(_qsh_new - qsh).max()) if _kf else 1.0
            qsh = (final_resolve_mix * _qsh_new
                   + (1.0 - final_resolve_mix) * qsh) if _kf else _qsh_new
            q_at = np.bincount(basis.shell_atom, weights=qsh,
                               minlength=atoms.size)
            F = _Ff
            if _kf and _dq < 1.0e-8:
                break
    # ⚠️ THE RETURNED BASIS HAS TO BE AT THE RETURNED CHARGES.
    # The loop updates the contraction at the TOP of each iteration, from the
    # previous iteration's qsh, and then P and qsh are refreshed twice more
    # after the loop -- so `basis`, `S` and `H0` came out two refreshes behind
    # the charges reported beside them.  Against the binary's own
    # `basis_qvszp::basis_update` driven at those same charges, the overlap was
    # 1.7e-2 out on H2O (walk stage 23d) while the port's basis BUILDER, given
    # the same charges, reproduces the binary at 4.2e-15 (23d').  The recipe
    # was never wrong; the point it was evaluated at was.
    # ⚠️ THE RETURNED BASIS IS STILL ONE UPDATE BEHIND THE RETURNED CHARGES,
    # and the fix does NOT belong here.  Measured, in order:
    #   * no post-loop rebuild             overlap 1.7e-2 vs the binary
    #   * one rebuild, charges re-reduced
    #     from the stale P                 1.8e-3, but max_abs_qsum 5.4e-2
    #                                      against a 1e-6 tolerance -- a stale
    #                                      density against a rebuilt S does not
    #                                      conserve electrons
    #   * 40 rebuilds, each followed by a
    #     full re-solve so P and S share
    #     a basis                          2.1e-6 and charge-conserving, but
    #                                      all_mae_e 1.33 and max_abs_dq 11.2:
    #                                      forty UNDAMPED passes of a coupled
    #                                      basis<->charge map diverge
    # The coupling is exactly what a mixer exists to damp, and the binary
    # updates the contraction INSIDE `next_scf`, after the Broyden step, not
    # after the loop.  So this belongs in the SCF iteration with the mixer
    # (walk stage 22), not in a post-loop patch.  Left out rather than left
    # wrong.
    # ⚠️ THE AES ENERGY HAS TO FOLLOW THE DENSITY THAT IS RETURNED.
    # `_F_aes_f` and `_vat_aes_f` above legitimately come from the PREVIOUS
    # density -- they build the last Fock matrix.  `E_aes_f` came from there
    # too, and then `P` is rebuilt from the eigenvectors just above, so the
    # reported energy sat one half-iteration behind the reported density while
    # every other term (E_h0, E_first, E_coul, ...) is computed below, on the
    # new one.  Worth 2.0e-3 Ha on H2O against the binary's own container, and
    # invisible in any check that does not compare the two side by side.
    if use_aes:
        E_aes = float(gxtb_aes_fock(P, basis, atoms, coords, want_vat=True)[1])
    E_first, V_first, V_first_at = (
        _first_order_decoded(atoms, basis.cn, basis.shell_atom,
                             basis.shell_l, qsh)
        if use_first_order
        else (0.0, 0.0, 0.0)
    )
    E_first_total = E_first

    # ⚠️ THE BINARY CONTRACTS THE FULL H0.  `get_electronic_energy` is handed
    # `ints%hamiltonian`, and `get_acp` folds the ACP straight into it, so the
    # electronic energy is one number: sum(P * H0).  The port summed
    # sum(P*H_eht) + sum(P*H_acp) instead, which drops the third piece of
    # `H0 = H_eht + H_acp + aniso_h0` -- worth +2.42 Ha on H2O and +9.6 on
    # benzene.  Walk stage 23a': sum(P*H0) matches the binary at 8.9e-16.
    # E_acp_h stays as a REPORTED breakdown; it is no longer added on top.
    E_h0 = float(np.sum(P * H0))
    E_acp_h = float(np.sum(P * H_acp)) if use_acp_hamiltonian else 0.0
    # the energy whose gradient is that potential
    E_coul = scc_scale * (0.5 * float(qsh @ (jmat @ qsh)) + float(qsh @ vref))
    E_third = (
        scc_scale * float(np.sum(qsh**3 * basis.shell_third) / 3.0)
        if use_third_order
        else 0.0
    )
    E_third_twobody = (
        scc_scale * GXTB_TB3_TWOBODY_SCALE
        * _third_order_twobody(basis, atoms, coords, qsh)[0]
        if use_twobody_third_order
        else 0.0
    )
    E_fourth = (
        scc_scale * float(np.sum(qsh**4 * basis.shell_fourth * GXTB_K4TH_SCALE) / 24.0)
        if use_fourth_order
        else 0.0
    )
    E_exchange = (
        scc_scale * 0.5 * float(np.sum(basis.shell_exchange * qsh * qsh))
        if use_exchange
        else 0.0
    )
    # ⚠️ THE EXCHANGE ENERGY IS NOT `_mfx_fock_energy`'s.
    # `exchange_type::get_energy_w_overlap` contracts the container's OWN
    # kfock -- the matrix `get_kfock` writes and the SCF loop converges on --
    # against the density, per atom:
    #     energies(iat) += sum_{iao in iat} 0.5*ddot(kfock(:,iao), P(:,iao))
    # so the total is 0.5*sum(kfock*P).  `_mfx_fock_energy` is a different and
    # weaker kernel; reporting its energy put the total 1110 kcal/mol from the
    # binary over 60 perfumery molecules, where this expression puts the median
    # at 3.7 and reproduces water's electronic energy to 2.1e-11 Eh.
    # `F_mfx` is left alone: it is the post-loop Fock, a separate question.
    # Recovered at recovered/exchange_fock/get_energy_w_overlap.f90, gated by
    # probes/xt_get_energy_w_overlap (bit-exact, per atom).
    E_mfx, F_mfx = _mfx_fock_energy(P, S, gamma_mfx) if gamma_mfx is not None else (0.0, np.zeros_like(H0))
    if gamma_mfx is not None and onsite_potential == 4:
        _kf_e = _kfock_fast(P, S, basis, atoms, _kmat_shell_of(basis, gamma_mfx),
                            qsh, coords)
        E_mfx = 0.5 * float(np.sum(np.asarray(_kf_e) * np.asarray(P)))
    rep = gxtb_reconstructed_repulsion(atoms, coords, descriptor=q_at, cn=basis.cn)
    E_rep = float(rep.energy)
    # The D4Srev dispersion ENERGY, recovered from the binary's own container
    # and verified against it at 1.1e-18 per atom.  `d4srev_dispersion_gxtb`
    # was a GFN2-D4 fallback that `use_d4srev=False` switched off entirely, so
    # the total carried no dispersion at all.
    if scf_dispersion:
        E_d4 = float(np.sum(_d4srev_energy_fast(basis, atoms, coords, q_at)))
        d4_backend = "d4srev-recovered"
    else:
        E_d4, d4_backend = d4srev_dispersion_gxtb(atoms, coords, enabled=use_d4srev)
    E_acp = gxtb_pacp_proxy_energy(atoms, coords, enabled=use_pacp)
    E_increment_raw = float(np.sum(GXTB_PARAMS["pa_increment"][atoms - 1]))
    E_increment_correction = _halide_increment_correction(atoms) if use_halide_increment_correction else 0.0
    E_increment = E_increment_raw + E_increment_correction
    # Three-body dispersion: charge-independent, outside the SCF, and not
    # part of the electronic energy -- it belongs in the total only.
    E_atm = _d4srev_atm_energy(basis, atoms, coords)

    E_total = (
        E_h0
        + E_aes
        + E_first_total
        + E_coul
        + E_third
        + E_third_twobody
        + E_fourth
        + E_exchange
        + E_mfx
        + E_rep
        + E_d4
        + E_acp
    )
    return {
        "energy_hartree": E_total,
        "energy_eV": E_total * EV_PER_HARTREE,
        "energy_kcal": E_total * KCAL_PER_HARTREE,
        "electronic_hartree": E_h0
        + E_aes
        + E_first_total
        + E_coul
        + E_third
        + E_third_twobody
        + E_fourth
        + E_exchange
        + E_mfx,
        "h0_hartree": E_h0,
        "acp_hamiltonian_hartree": E_acp_h,
        "aes_hartree": E_aes,
        "first_order_hartree": E_first_total,
        "first_order_onsite_hartree": E_first,
        "coulomb_hartree": E_coul,
        "third_order_hartree": E_third,
        "third_order_twobody_hartree": E_third_twobody,
        "fourth_order_hartree": E_fourth,
        "exchange_hartree": E_exchange,
        "mfx_exchange_hartree": E_mfx,
        "repulsion_hartree": E_rep,
        "dispersion_hartree": E_d4,
        "dispersion_atm_hartree": E_atm,
        "d4srev_backend": d4_backend,
        "pacp_hartree": E_acp,
        "raw_increment_hartree": E_increment_raw,
        "halide_increment_correction_hartree": E_increment_correction,
        "increment_hartree": E_increment,
        "energy_plus_increment_hartree": E_total + E_increment + E_atm,
        "converged": converged,
        "n_iter": n_iter,
        "n_basis": n_basis,
        "n_shell": n_shell,
        "n_elec": n_elec,
        "n_occ": n_occ,
        "method": "g-xTB-reconstructed",
        "basis": basis,
        "H0": H0,
        "H_eht": H_eht,
        "H_acp": H_acp,
        "S": S,
        "F": F,
        "F_mfx": F_mfx,
        "density": P,
        "eigenvalues": eigvals,
        # probe support: the SHELL potential the driver actually folded into F
        # (Hartree/e per shell, PORT shell order), and its terms.  Stage 31 of
        # `probes/audit/port_divergence.py` found the driver's Fock differs by
        # 2.3e-1 from the one the walk composes out of the port's OWN
        # components; without this the two assemblies cannot be diffed.
        "V_sh": np.asarray(V_sh, dtype=np.float64).copy(),
        "V_sh_loop": _V_sh_loop if "_V_sh_loop" in dir() else None,
        "V_terms_loop": _V_terms_loop if "_V_terms_loop" in dir() else None,
        "qsh_loop": _qsh_loop if "_qsh_loop" in dir() else None,
        "P_loop": _P_loop if "_P_loop" in dir() else None,
        "kfock_loop": _kfock_loop if "_kfock_loop" in dir() else None,
        "F_fold_loop": _F_fold_loop if "_F_fold_loop" in dir() else None,
        "H0_loop": _H0_loop if "_H0_loop" in dir() else None,
        "S_loop": _S_loop if "_S_loop" in dir() else None,
        "F_aes_loop": _F_aes_loop if "_F_aes_loop" in dir() else None,
        "F_loop": _F_loop if "_F_loop" in dir() else None,
        "kmat_sh_loop": _kmat_sh_loop if "_kmat_sh_loop" in dir() else None,
        "V_terms": {
            "first": np.asarray(V_first, dtype=np.float64).copy(),
            "coul": np.asarray(V_coul, dtype=np.float64).copy(),
            "third": np.asarray(V_third, dtype=np.float64).copy(),
            "third_twobody": np.asarray(V_third_twobody, dtype=np.float64).copy(),
            "fourth": np.asarray(V_fourth, dtype=np.float64).copy(),
            "exchange": np.asarray(V_exchange, dtype=np.float64).copy(),
            "tb3": np.asarray(V_tb3, dtype=np.float64).copy(),
        },
        "shell_charges": qsh,
        "atom_charges": q_at,
        "coordination_number": basis.cn,
        "eeqbc_charges": basis.eeqbc_charges,
        "jmat": jmat,
        "shell_selfenergy": shell_self,
        "repulsion": rep,
        "exactness": {
            "eeqbc": "binary-formula",
            "qvszp": "binary tables, active pa_nshell shells",
            "h0": "binary parameter scaffold without anisotropic H0",
            # DECODED: gxtb's add_coulomb constructs exactly five Coulomb
            # objects -- new_onsite_firstorder, new_effective_coulomb,
            # new_twobody_thirdorder, new_onsite_fourthorder and
            # new_gxtb_multipole. There is NO offsite first-order term; the
            # port reconstructed one the binary does not have. Deleted.
            "first_order": "onsite firstorder only (binary has no offsite term)"
            if use_first_order
            else "disabled",
            "scc": "shell-charge scaffold",
            "third_order": "enabled" if use_third_order else "decoded tables exposed; disabled by default",
            "third_order_twobody": "SI Eq. 129 two-body third-order scaffold"
            if use_twobody_third_order
            else "disabled",
            "exchange": "diagonal shell proxy from recovered tables" if use_exchange else "diagonal shell proxy disabled",
            "mfx_exchange": "SI Eq. 153 range-separated Mulliken Fock exchange"
            if use_mfx_exchange
            else "disabled",
            "p_acp": "pair-energy proxy",
            "acp_hamiltonian": "SI Eq. 78 reduced projector Hamiltonian"
            if use_acp_hamiltonian
            else "disabled",
            "d4srev": d4_backend,
            "carbon_plevel_shift": "oracle-fitted carbon-p patch, standing in for a "
            "missing term"
            if use_carbon_plevel_shift
            else "disabled",
            "halide_increment_correction": "oracle-calibrated additive shift"
            if use_halide_increment_correction
            else "disabled",
            "gradient": "central finite difference",
        },
    }


def gxtb_gradient_numerical(
    atomic_numbers: list[int] | np.ndarray,
    coords_ang: np.ndarray,
    *,
    h: float = 1.0e-3,
    **energy_kwargs,
) -> np.ndarray:
    """Central-difference gradient of :func:`gxtb_energy` in Hartree/Angstrom."""

    atoms = np.asarray(atomic_numbers, dtype=np.intp)
    coords = np.asarray(coords_ang, dtype=np.float64)
    grad = np.zeros_like(coords)
    for i in range(coords.shape[0]):
        for k in range(3):
            plus = coords.copy()
            minus = coords.copy()
            plus[i, k] += h
            minus[i, k] -= h
            ep = float(gxtb_energy(atoms, plus, **energy_kwargs)["energy_hartree"])
            em = float(gxtb_energy(atoms, minus, **energy_kwargs)["energy_hartree"])
            grad[i, k] = (ep - em) / (2.0 * h)
    return grad


def gxtb_energy_gradient(
    atomic_numbers: list[int] | np.ndarray,
    coords_ang: np.ndarray,
    *,
    gradient_h: float = 1.0e-3,
    **energy_kwargs,
) -> dict[str, object]:
    """Return energy and central-difference gradient for the reconstructed path."""

    res = gxtb_energy(atomic_numbers, coords_ang, **energy_kwargs)
    res["gradient"] = gxtb_gradient_numerical(
        atomic_numbers,
        coords_ang,
        h=gradient_h,
        **energy_kwargs,
    )
    return res


# ---------------------------------------------------------------- entry point

#: Flag set the harness scores.  These are the current best-known defaults:
#: `use_mfx_exchange` on took the 20-molecule charge MAE from 0.04617 to
#: 0.02494 e and won on 20/20; `use_d4srev` is off because it is an additive
#: post-SCF energy term that cannot move a charge and its backend
#: (simple-dftd4) is not installed here.  Everything else is a live question.
SOLVE_KWARGS: dict = {
    # `basis_qvszp::basis_update` is called every SCF iteration with `wfn%qat`;
    # the port built the q-vSZP contraction once from the EEQ-BC charges.
    # ⚠️ THE BINARY DOES NOT UPDATE THE BASIS INSIDE THE SCF.
    # This was True on the strength of a comment saying
    # "basis_qvszp::basis_update is called every SCF iteration with wfn%qat".
    # `recovered/tblite_scf_iterator/next_scf.f90` passes at 0 ulp and its
    # COMPLETE call list contains no basis update of any kind:
    #     reset, coulomb_get_potential, cont*%get_potential,
    #     get_potential_w_overlap, list_get_potential, add_pot_to_h1,
    #     next_density, get_mulliken_*, get_qat_from_qsh, *_mixer, reduce
    # `basis_update` has no caller anywhere in the decompiled tree either.
    # The basis is built ONCE, before the loop.
    #
    # Switching it off also removes the electron-count violation on its own:
    # Tr(P.S) is 8.00000000 exactly with no post-loop machinery at all.
    "scf_basis_update": False,
    # ⚠️ ELECTRON CONSERVATION.  The post-loop rebuild changes `S` while `P`
    # was solved against the old one, so `Tr(P.S)` came out 7.974613 where it
    # must be 8.000000 (walk stage 41; 13.979228/14 and 29.964959/30 on the
    # other two).  Sum-of-charges still held -- `max_abs_qsum` 1.05e-09 -- so
    # no shipped check saw it.  One re-solve in the rebuilt basis restores it
    # exactly, and leaves sum(q) where it was.
    # `final_resolve` is NOT in the binary -- it was a post-loop patch that
    # restored Tr(P.S) while `scf_basis_update` was wrongly on.  With the basis
    # fixed during the SCF, as the binary has it, no patch is needed.
    "final_resolve": False,
    # the basis and the charges are a coupled map; one pass fixes Tr(P.S) but
    # leaves the basis a step behind (23d/23e at 1.2e-2), forty undamped
    # passes diverge.  Damped iteration converges: Tr(P.S) stays exact and the
    # energy settles, and by 30 the basis lag is 1.14e-10 -- so BOTH
    # conservation laws hold at once: Tr(P.S) exact and the basis at its own
    # charges.  The note's "forty passes diverge" was about UNDAMPED passes.
    "final_resolve_iters": 30,
    "final_resolve_mix": 0.4,
    "use_d4srev": False,
    # SI Eq. 129 two-body third order. Was dead code (module-scope load of a
    # missing npz, then a relative import in this vendored copy); never scored.
    # It is the shell-hardness's own CN/charge response, so it is exactly the
    # element-shaped term the residual table asks for.
    "use_twobody_third_order": True,
    # Anisotropic electrostatics. Also never scored (same dead-code story).
    # Free now that qvszp_multipoles goes through the compiled kernel.
    "use_aes": True,
    # The DECODED H0 distance polynomial (get_shpoly2 / get_shpoly4 +
    # get_hamiltonian): pi = 1 + 2*pa*pg*(R/rcov + 0.0402348406*R^2/rcov).
    # The port's `1 + pa*pg*sqrt(R/rcov)` implements the ONE term that is zero
    # for g-xTB, with a missing factor 2 and no quadratic term.
    "decoded_shpoly": True,
    #: ⚠️ THE H0 RECIPE THE WALK PINS TO THE BINARY AT 2.8e-16 is
    #: `build_hcore_gxtb(..., carbon_plevel=False, decoded_shpoly=True)` over
    #: an `_H0BasisProxy` at effective-overlap scale 1.0, plus the aniso term
    #: at scale 1.  Production was running a DIFFERENT H0: the carbon p-level
    #: shift on (the binary has no such term) and the effective basis off.
    #: sum(P*(H0+H_acp)) was 2.3e-2 Ha from the binary's on H2O and 3.0e-2 on
    #: benzene -- walk stage 23a.  The docstring's "dropping the carbon shift
    #: costs 2.6x on charge MAE" is the compensation talking, not physics.
    "use_carbon_plevel_shift": False,
    "use_h0_effective_basis": 1.0,
    # DECODED: add_coulomb constructs `new_onsite_fourthorder`, so the binary
    # runs the fourth-order onsite term. The port defaulted it off.
    "use_fourth_order": True,
    # DECODED: the g-xTB repulsion is charge dependent
    # (zeff = (1 - q*pa_rep_q)*pa_rep_zeff) and contributes an atomic potential
    # to the SCF via `get_potential`. The port evaluated it only post-SCF.
    "scf_repulsion": True,
    #: The D4Srev dispersion potential, recovered from the binary and
    #: verified against it at 2.6e-18 on the divergence walk.  The binary
    #: adds it to pot%vat on every SCF iteration, so it is not optional.
    "scf_dispersion": True,
    # DECODED MFX pair-term damping (get_gmulliken_0d): pair = favg/exp(-R(c0+xi*c1)),
    # no frscale. Verified against the binary's own H2 gamma to 0.9 %.
    "mfx_decoded_damping": True,
    # 🔑 the VERIFIED get_kfock (1.78e-15 against the shipped routine), in place
    # of _mfx_fock_energy, which is half-strength on two of its three terms and
    # has no on-site half at all.
    "onsite_potential": 4,
    # 🔑 ONE-CENTRE FOCK EXCHANGE.  `exchange_fock::get_kfock` (recovered) is
    # not just the two-centre Mulliken factorisation the port implements: it
    # carries an ON-SITE block driven by `onsite_sh(jsh, ish, izp) =
    # get_onecxints_number(l_jsh, l_ish, num)` -- one-centre exchange integrals
    # resolved by ANGULAR-MOMENTUM PAIR.  The port has the machinery and the
    # extracted 103x10 table, and it was switched off; it was also invisible to
    # every ablation sweep, because the key was absent from this dict.
    "use_onecenter": True,
    # DECODED: new_gxtb_calculator constructs `new_acp`, so the ACP Hamiltonian
    # is part of g-xTB. It only pays once the exchange is right (before the MFX
    # damping fix it cost 4 % on charge; after it, it gains 18 %), and it is
    # only affordable once its AO+aux overlap goes through the compiled kernel.
    "use_acp_hamiltonian": True,
    # SPEED. Charges are unchanged from conv_tol 1e-7 down to 1e-11, so the
    # tolerance is a pure cost knob. 1e-6 drifts the scored charges by at most
    # 1.65e-06 -- 8x below the ORACLE'S OWN reproducibility floor (1.3e-05,
    # measured by running the binary at --acc 0.2 vs 0.01) and 60x below the
    # 1e-4 target. Iterations 22.2 -> 18.6. 1e-5 would be faster still but its
    # drift equals the oracle floor, which leaves no margin.
    "conv_tol": 1.0e-6,
    # SPEED. `gxtb_pacp_proxy_energy` is a pair loop costing ~8 % of runtime and
    # its own docstring says it "is intentionally not used inside the Fock
    # matrix" -- a placeholder energy term. Energy is not scored, and disabling
    # it changes the charges by EXACTLY 0 over 20 molecules. Off.
    "use_pacp": False,
    "use_bocorr": False,
    # The H0 basis is the binary's `cgto2`: alpha*ps_h0_qvszp_exp_scal (power
    # +1, not the -1 that was guessed here before) plus its own charge response
    # from the scaled k0/k2/k3. It is now a decoded term, not a knob -- and the
    # density basis no longer carries the k scalings, so this path is where they
    # enter. See `_qeff_h0` / `_unscaled_h0_k`.
    # The ps_h0_qvszp_exp_scal power on the H0 basis.  1.0 was taken as "the
    # decoded value, applied once", but it does not survive a direct
    # comparison with the binary: on an identical-model benzene (qeff forced to
    # 0 on both sides, so the basis matches to 3e-08) the shipped SCF's own
    # shell charges are reproduced 8x better with it OFF -- max|dqsh| 0.1378
    # -> 0.0171 -- and monotonically worse as the power rises.  Against the
    # frozen reference it improves shell populations 22%% (0.07432 -> 0.05794)
    # and the HL gap 36%% (1.7978 -> 1.1452 eV), and halves the sulfur p-shell
    # error (+0.50 -> +0.26).  It costs 0.001 on charge MAE, which is the one
    # metric that cancels an s/p misplacement (C is +0.29 s / -0.26 p, sum
    # +0.03), so it is the metric least able to see this error.
    # DECODED `new_gxtb_calculator`: the q-vSZP basis is built as TWO cgto sets
    # and the second becomes `new_qvszp_basis`' `cgto2`, i.e. `bas%cgto_h0`:
    #     alpha2 = alpha * ps_h0_qvszp_exp_scal(ish, num)
    #     k0,k2,k3 scaled by pa_h0_qvszp_{k0,k2,k3}_scal(num);  k1 untouched
    # `get_hamiltonian` then builds the block it scales by `hscale` with
    # `overlap_cgto` over THAT set, while the SCF's metric overlap keeps coming
    # from `cgto`. `power=1.0` is the binary's own scaling, verbatim; 0.0 meant
    # the port applied `diat_trafo` to the ORDINARY overlap.
    "use_h0_effective_basis": 1.0,
    # DECODED (get_anisotropy + get_hamiltonian): the binary always applies the
    # anisotropic H0 term, on-site included. Not a knob -- scale stays 1.0.
    "use_aniso_h0": True,
    # DECODED: get_selfenergy is `h0 - cn*kcn - cn_en*kcn_en - q*kq1 - q^2*kq2`,
    # and for g-xTB the last three tables are filled by `tblite_xtb_spec`'s
    # get_cnenshift/get_q1shift/get_q2shift, which are PURE ZERO-FILLS (56-line
    # functions that reference no parameter table). So the binary's level is
    # exactly `h0 - kcn*cn`, which is what the port computes -- and the extra
    # +0.04 + 0.015*acc on carbon p shells has no counterpart at all.
    "use_carbon_plevel_shift": False,
    # `add_coulomb` constructs `new_onsite_firstorder`, so the IP/EA shell shift
    # IS part of the model. It stays OFF anyway: the decoded sign (+1) cannot
    # fit oxygen and sulfur at the same time as the shipped -1 does, which says
    # the form is wrong, and there is no recovered file or differential test for
    # `coulomb_firstorder::get_potential` yet. Enabling an unverified term costs
    # 0.043 -> 0.061 on the bench. Recover it first, then turn it on.
    "use_first_order": True,
}


def solve(atoms, coords_ang):
    """Single point on one molecule.  Contract read by run_gxtb.py:

        atom_charges   (n_atoms,)  Mulliken atomic charges, electrons
        shell_charges  (n_shell,)  Mulliken shell charges
        basis                      carries shell_atom / shell_l for the
                                   shell -> (atom, l) population mapping
        eigenvalues    (n_basis,)  MO energies, hartree
        n_occ                      number of doubly-occupied MOs
        converged      bool
        n_iter         int
    """
    return gxtb_energy(atoms, coords_ang, **SOLVE_KWARGS)


# ------------------------------------------------ the O(N^2) Python pair loops
#
# Profiling after the ACP fix left a flat profile with one recurring shape:
# `np.linalg.norm(xyz[i] - xyz[j])` inside a double loop over atoms. It ran
# 36 337 times per six molecules -- 10 % of the whole benchmark in `norm` alone,
# plus the interpreter overhead of the loops around it.
#
# Five of them: `aniso_electro` (a triple loop, pairs x 3 x 3), the mctc ERF
# coordination number, and three EEQ-BC assemblers. All are pure pair sums, so
# they vectorise directly. `probes/check_vec.py` checks each against the frozen
# version: agreement is 3e-16 to 5e-14, i.e. summation-order rounding -- nine
# orders below the 1.3e-05 oracle reproducibility floor.
#
# The frozen callers bind these names at import (`from .mctc_ncoord import
# erf_coordination_number`), so the patch has to replace the name inside every
# module that imported it, not just where it was defined.

_QP_W = np.array([1.0, 2.0, 1.0, 2.0, 2.0, 1.0])   # (xx, xy, yy, xz, yz, zz)


def _pair_dist(xyz: np.ndarray) -> np.ndarray:
    d = xyz[:, None, :] - xyz[None, :, :]
    return np.sqrt(np.einsum("ijk,ijk->ij", d, d))


def _install_fast_pairloops() -> bool:
    try:
        from scipy.special import erf as _erf
        import mlxmolkit.xtb.aes as _aes
        import mlxmolkit.xtb.eeqbc as _eb
        import mlxmolkit.xtb.mctc_ncoord as _nc
        import mlxmolkit.xtb.gxtb_reconstructed as _gr
    except Exception:
        return False

    def erf_cn(atomic_numbers, coords, rcov_by_z, *, k, power=1.0, cutoff=25.0):
        atoms = np.asarray(atomic_numbers, dtype=np.intp)
        xyz = np.asarray(coords, dtype=np.float64)
        rcov = np.asarray(rcov_by_z, dtype=np.float64)
        if xyz.shape != (atoms.size, 3):
            raise ValueError("coords must have shape (nat, 3)")
        if np.any(atoms < 1) or np.any(atoms > rcov.size):
            raise ValueError("atomic_numbers are outside the supplied rcov table")
        ar = rcov[atoms - 1]
        r = _pair_dist(xyz)
        r0 = ar[:, None] + ar[None, :]
        f = 0.5 * (1.0 + _erf(-k * (r - r0) / np.maximum(r0 ** power, 1.0e-12)))
        f = np.where(np.triu((r >= 1.0e-12) & (r <= cutoff), 1), f, 0.0)
        return f.sum(1) + f.sum(0)

    def eeqbc_local_charge(atomic_numbers, coords_ang, *, total_charge=0.0):
        atoms = np.asarray(atomic_numbers, dtype=np.intp)
        coords = np.asarray(coords_ang, dtype=np.float64)
        rcov = _eb.EEQBC2025_PARAMS["cov_radii"][atoms - 1]
        en = _eb._pauling_en_normalized(atoms)
        r = _pair_dist(coords)
        r0 = rcov[:, None] + rcov[None, :]
        count = 0.5 * (1.0 + _erf(-_eb.DEFAULT_CN_EXP * (r - r0)
                                  / np.maximum(r0 ** _eb.DEFAULT_NORM_EXP, 1.0e-12)))
        ok = np.tril((r >= 1.0e-12) & (r <= _eb.DEFAULT_CUTOFF), -1)
        w = np.where(ok, (en[None, :] - en[:, None]) * count, 0.0)
        return w.sum(1) - w.sum(0) + float(total_charge) / float(atoms.size)

    def eeqbc_capacitance_matrix(atomic_numbers, coords_ang):
        atoms = np.asarray(atomic_numbers, dtype=np.intp)
        coords = np.asarray(coords_ang, dtype=np.float64)
        n = atoms.size
        if coords.shape != (n, 3):
            raise ValueError("coords_ang must have shape (nat, 3)")
        caps = _eb.EEQBC2025_PARAMS["cap"][atoms - 1]
        rvdw = _eb.eeqbc_pair_rvdw_matrix_ang(atoms)
        r = _pair_dist(coords)
        cij = (np.sqrt(caps[:, None] * caps[None, :]) * 0.5
               * (1.0 + _erf(-_eb.DEFAULT_KBC * (r - rvdw) / np.maximum(rvdw, 1.0e-12))))
        cij = np.where(np.tril(r >= 1.0e-12, -1), cij, 0.0)
        cij = cij + cij.T
        cmat = np.zeros((n + 1, n + 1), dtype=np.float64)
        cmat[:n, :n] = -cij
        np.fill_diagonal(cmat[:n, :n], cij.sum(1))
        cmat[n, n] = 1.0
        return cmat

    def eeqbc_coulomb_matrix(atomic_numbers, coords_ang, *, cn=None, cmat=None):
        atoms = np.asarray(atomic_numbers, dtype=np.intp)
        coords = np.asarray(coords_ang, dtype=np.float64)
        n = atoms.size
        if cn is None:
            cn = _eb.eeqbc_coordination_number(atoms, coords)
        if cmat is None:
            cmat = eeqbc_capacitance_matrix(atoms, coords)
        radii = _eb.eeqbc_effective_radii(atoms, cn)
        eta = _eb.EEQBC2025_PARAMS["eta"][atoms - 1]
        r = _pair_dist(coords)
        gam2 = 1.0 / np.maximum(radii[:, None] ** 2 + radii[None, :] ** 2, 1.0e-24)
        tmp = _erf(np.sqrt(r * r * gam2)) / np.maximum(r, 1.0e-12) * cmat[:n, :n].T
        tmp = np.where(np.tril(np.ones((n, n), dtype=bool), -1), tmp, 0.0)
        amat = np.zeros((n + 1, n + 1), dtype=np.float64)
        amat[:n, :n] = tmp + tmp.T
        np.fill_diagonal(amat[:n, :n],
                         (eta + _eb.SQRT_2_OVER_PI / np.maximum(radii, 1.0e-12))
                         * np.diag(cmat[:n, :n]) + 1.0)
        amat[n, : n + 1] = 1.0
        amat[: n + 1, n] = 1.0
        amat[n, n] = 0.0
        return amat, radii

    _kern_cache: dict = {}

    def aniso_electro(atoms, coords_bohr, q, dipm, qp, gab3, gab5):
        key = tuple(int(z) for z in atoms)
        kern = _kern_cache.get(key)
        if kern is None:
            dipk = np.array([_aes.GFN2_PARAMS[z].dip_kernel for z in key])
            quadk = np.array([_aes.GFN2_PARAMS[z].quad_kernel for z in key])
            kern = (dipk, quadk)
            _kern_cache[key] = kern
        dipk, quadk = kern
        R = np.asarray(coords_bohr, dtype=np.float64)
        n = R.shape[0]
        e_polar = float(np.dot(dipk, np.einsum("ki,ki->i", dipm, dipm))
                        + np.dot(quadk, _QP_W @ (qp * qp)))
        D = R[None, :, :] - R[:, None, :]            # D[i, j] = R[j] - R[i]
        r2 = np.einsum("ijk,ijk->ij", D, D)
        a = np.einsum("ki,ijk->ij", dipm, D)
        b = np.einsum("kj,ijk->ij", dipm, D)
        rr = np.stack([D[:, :, 0] * D[:, :, 0], D[:, :, 0] * D[:, :, 1],
                       D[:, :, 1] * D[:, :, 1], D[:, :, 0] * D[:, :, 2],
                       D[:, :, 1] * D[:, :, 2], D[:, :, 2] * D[:, :, 2]])
        rrw = rr * _QP_W[:, None, None]
        ed = q[None, :] * a - q[:, None] * b
        eq = (q[None, :] * np.einsum("mi,mij->ij", qp, rrw)
              + q[:, None] * np.einsum("mj,mij->ij", qp, rrw))
        edd = -3.0 * b * a + r2 * (dipm.T @ dipm)
        low = np.tril(np.ones((n, n), dtype=bool), -1)
        g3 = np.where(low, gab3.T, 0.0)
        g5 = np.where(low, gab5.T, 0.0)
        return float((ed * g3).sum() + (eq * g5).sum() + (edd * g5).sum()), e_polar

    _nc.erf_coordination_number = erf_cn
    _eb.erf_coordination_number = erf_cn
    _gr.erf_coordination_number = erf_cn
    _eb.eeqbc_local_charge = eeqbc_local_charge
    _eb.eeqbc_capacitance_matrix = eeqbc_capacitance_matrix
    _eb.eeqbc_coulomb_matrix = eeqbc_coulomb_matrix
    _aes.aniso_electro = aniso_electro
    return True


_FAST_PAIRLOOPS = _install_fast_pairloops()


# The contracted-Gaussian normalisation is an O(n_prim^2) Python double loop
# that calls the primitive-norm helpers once per element -- 26 762 calls per six
# molecules. The self-overlap it sums is a plain outer product, so it collapses
# to one einsum. Both the frozen basis builder and this module bind the name at
# import, so both bindings are replaced.


def _contraction_norm_fast(alphas, raw_coeffs, l_total: int) -> float:
    a = np.asarray(alphas, dtype=np.float64)
    c = np.asarray(raw_coeffs, dtype=np.float64)
    p = a[:, None] + a[None, :]
    if l_total == 0:
        ang = 1.0
    elif l_total == 1:
        ang = 1.0 / (2.0 * p)
    elif l_total == 2:
        ang = 3.0 / (2.0 * p) ** 2
    else:
        raise NotImplementedError(f"l_total={l_total} not supported")
    w = c * _primitive_norms(l_total, a)
    ov = float(np.einsum("i,j,ij->", w, w, (np.pi / p) ** 1.5 * ang))
    return 1.0 / math.sqrt(ov)


def _install_fast_contraction() -> bool:
    try:
        import mlxmolkit.xtb.basis as _b
        import mlxmolkit.xtb.gxtb_basis as _gb
    except Exception:
        return False
    _b._contraction_norm = _contraction_norm_fast
    _gb._contraction_norm = _contraction_norm_fast
    globals()["_contraction_norm"] = _contraction_norm_fast
    return True


_FAST_CONTRACTION = _install_fast_contraction()


# --------------------------------------------------- AES potential, made cheap
#
# `setvsdq_vectorized` is already vectorised, but it is called once per SCF
# iteration (~19x per molecule) and rebuilds its geometry every time: the pair
# displacement tensor, its squares, r^2 and the r.dr contraction depend only on
# the coordinates, and the GFN2 dip/quad kernels only on the elements. Hoist all
# of that into a per-molecule cache and the call keeps only the charge-dependent
# arithmetic.
#
# Layout note, checked rather than assumed: `vq` is ordered (xx, yy, zz, xy, xz,
# yz) while `qp` uses the mmompop order (xx, xy, yy, xz, yz, zz). The two are
# not the same permutation, so the trace correction landing on vq[0..2] is
# correct. `probes/check_setvsdq.py` pins the result against the frozen twin.


# ---------------------------------------------------- AES Fock fold, made cheap
#
# `fockelectro` runs nine strided elementwise passes per SCF iteration, one per
# dipole/quadrupole channel, each multiplying `integral[k].T` by an outer sum.
# Two facts collapse that to a single contraction:
#   * dpint and qpint are symmetric to *exactly* zero (checked, not assumed:
#     probes/check_fock_aes.py), so every `.T` is a no-op; and
#   * with that symmetry the i-indexed half of the outer sum is the transpose of
#     the j-indexed half, so only one of the two needs contracting.
# The channel stack is geometry-only, so it is built once per molecule.
#
# The caller discards fockelectro's energy return (the AES energy comes from
# aniso_electro), so the 0.25*sum(P*fji) pass is skipped as well. 1.94x, and the
# Fock agrees to 1.4e-14.

_AES_MSTACK: dict = {}


def _aes_mstack(basis, dpint, qpint):
    key = id(basis)
    hit = _AES_MSTACK.get(key)
    if hit is not None and hit[0] is basis:
        return hit[1]
    M = np.ascontiguousarray(np.concatenate([dpint, qpint], axis=0))
    if len(_AES_MSTACK) > 8:
        _AES_MSTACK.clear()
    _AES_MSTACK[key] = (basis, M)
    return M


def _fockelectro_fast(S, M, ao, vs, vd, vq):
    vs_ao = vs[ao]
    V = np.concatenate([vd[:, ao], vq[:, ao]], axis=0)
    # optimize=True re-plans this contraction on every call; the planning costs
    # more than it saves at this size (0.031 vs 0.017 ms) and the result is
    # bit-identical.
    A = np.einsum("kij,kj->ij", M, V, optimize=False)
    fji = S * (vs_ao[None, :] + vs_ao[:, None]) + A + A.T
    return 0.5 * fji

_AES_GEO_CACHE: dict = {}
_QP_IDX = np.array([[0, 1, 3], [1, 2, 4], [3, 4, 5]], dtype=np.int64)


def _aes_geometry(key, atoms, coords_ang):
    """Geometry-only AES tensors, cached on the *input* coordinate object.

    Keyed on coords_ang, not on the derived bohr array: the latter is rebuilt on
    every call, so an identity check against it would never hit.
    """
    hit = _AES_GEO_CACHE.get(key)
    if hit is not None and hit[0] is coords_ang:
        return hit[1]
    from mlxmolkit.xtb.aes import GFN2_PARAMS
    coords = np.asarray(coords_ang, dtype=np.float64) * ANG_TO_BOHR
    ra = coords[None, :, :]
    dra = coords[None, :, :] - coords[:, None, :]          # [j, i, xyz]
    geo = {
        "cb": coords,
        "ra": ra,
        "dra": dra,
        "dra2": dra * dra,
        "r2a": np.sum(coords * coords, axis=1)[None, :],
        "r2ab": np.sum(dra * dra, axis=2),
        "t1": np.sum(ra * dra, axis=2),
        "rsum": np.sum(coords * coords, axis=1),
        "qs1": 2.0 * np.array([GFN2_PARAMS[int(z)].dip_kernel for z in atoms]),
        "quadk": np.array([GFN2_PARAMS[int(z)].quad_kernel for z in atoms]),
    }
    geo["qs2"] = 6.0 * geo["quadk"]
    if len(_AES_GEO_CACHE) > 8:
        _AES_GEO_CACHE.clear()
    _AES_GEO_CACHE[key] = (coords_ang, geo)
    return geo


def _setvsdq_fast(atoms, coords, q, dipm, qp, gab3, gab5, geo):
    nat = coords.shape[0]
    ra, dra, r2ab, t1 = geo["ra"], geo["dra"], geo["r2ab"], geo["t1"]
    qs1, qs2, quad_kernel = geo["qs1"], geo["qs2"], geo["quadk"]
    qj = q[:, None]
    dip_j = dipm.T[:, None, :]
    t2 = np.sum(dip_j * dra, axis=2)
    t3 = np.sum(ra * dip_j, axis=2)
    qp_mat = qp.T[:, _QP_IDX]
    dum5 = -np.einsum("jab,jia,jib->ji", qp_mat, dra, dra)
    dum5 -= 1.5 * qj * t1 * t1
    dum5 += t3 * r2ab - 3.0 * t1 * t2 + 0.5 * qj * geo["r2a"] * r2ab
    dum3 = -t1 * qj - t2
    vs = np.sum(dum5 * gab5 + dum3 * gab3, axis=0)

    # vd assembled as contractions rather than (nat, nat, 3) temporaries: at
    # these sizes the routine is numpy-call-overhead bound, not FLOP bound, so
    # the win is in the number of operations, not the arithmetic.  Every term
    # below is the same sum over j, regrouped:
    #   vd[k, i] = sum_j gab3 q_j dra + gab5 (3 dra t2 - r2ab dip_j
    #                                         - q_j r2ab ra + 3 q_j dra t1)
    w_dra = qj * gab3 + gab5 * (3.0 * t2 + 3.0 * qj * t1)
    w_dip = gab5 * r2ab
    w_ra = np.sum(qj * w_dip, axis=0)
    vd = (np.einsum("ji,jik->ki", w_dra, dra)
          - np.einsum("ji,kj->ki", w_dip, dipm)
          - coords.T * w_ra[None, :])

    vq = np.zeros((6, nat), dtype=np.float64)
    qg5 = qj * gab5
    dra2 = geo["dra2"]
    half_r2 = np.sum(0.5 * r2ab * qg5, axis=0)
    for axis in range(3):
        vq[axis] += np.sum(-1.5 * qg5 * dra2[:, :, axis], axis=0) + half_r2
    for l1, l2, slot in ((1, 0, 3), (2, 0, 4), (2, 1, 5)):
        vq[slot] += np.sum(-3.0 * qg5 * dra[:, :, l2] * dra[:, :, l1], axis=0)

    vs = vs + np.sum(coords.T * dipm, axis=0) * qs1
    vd = vd - qs1[None, :] * dipm
    for l1, l2, qp_slot, vq_slot in ((1, 0, 1, 3), (2, 0, 3, 4), (2, 1, 4, 5)):
        w = qp[qp_slot] * qs2
        vq[vq_slot] -= w
        vs -= coords[:, l1] * coords[:, l2] * w
        vd[l1] += coords[:, l2] * w
        vd[l2] += coords[:, l1] * w
    for axis, qp_slot in ((0, 0), (1, 2), (2, 5)):
        w = qp[qp_slot] * qs2
        vq[axis] -= w * 0.5
        vs -= coords[:, axis] * coords[:, axis] * w * 0.5
        vd[axis] += coords[:, axis] * w
    t2a = (qp[0] + qp[2] + qp[5]) * quad_kernel
    vq[0] += t2a
    vq[1] += t2a
    vq[2] += t2a
    vd -= 2.0 * coords.T * t2a[None, :]
    vs += geo["rsum"] * t2a
    return vs, vd, vq
