# Copyright (c) 2026 Guillaume
# SPDX-License-Identifier: MIT

"""Clean-room, binary-guided g-xTB reconstruction pieces.

This module is intentionally narrower than a full g-xTB calculator.  It turns
the recovered repulsion constants and native pair-loop microkernel into a
callable component so we can run molecule-scale numerical probes while the
Hamiltonian/SCC pieces are still being reconstructed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .gxtb_cpp import (
    GXTBRepulsionState,
    repulsion_energy_gradient_asm,
    repulsion_state,
)
from .mctc_vdwrad import mctc_vdw_pair_matrix_bohr
from .mctc_ncoord import erf_coordination_number
from .params_gxtb import GXTB_PARAMS, GXTB_REPULSION_LITERALS


ANG_TO_BOHR = 1.8897261246204404
#: the erf coordination number's steepness
GXTB_CN_K = 2.068


@dataclass(frozen=True)
class GXTBReconstructedRepulsionConstants:
    """Scalar constants mapped from the recovered repulsion constructor calls."""

    stored_scalar_1p5: float
    exp_power_1: float
    exp_power_2: float
    exp2_scale: float
    exp2_weight: float
    cubic_coeff: float
    quartic_coeff: float
    light_pair_coeff: float
    heavy_pair_coeff: float


@dataclass(frozen=True)
class GXTBReconstructedRepulsion:
    """Result of the current reconstructed g-xTB repulsion block."""

    energy: float
    #: dE/dR in HARTREE PER ANGSTROM (the callers' convention)
    gradient: np.ndarray
    #: dE/dR in HARTREE PER BOHR (note the units: the rest of the API is per Angstrom)
    gradient_bohr: np.ndarray
    matvec: np.ndarray
    state: GXTBRepulsionState
    cn: np.ndarray
    pair_rvdw: np.ndarray
    pair_roffset: np.ndarray
    linear_coeff: np.ndarray
    quadratic_coeff: np.ndarray
    constants: GXTBReconstructedRepulsionConstants
    metadata: dict[str, object]


def repulsion_constants_from_binary() -> GXTBReconstructedRepulsionConstants:
    """Return the current scalar mapping used by the g-xTB model.

    The mapping below is the direct working hypothesis from ``add_repulsion``
    and ``new_repulsion_gxtb``:

    * 1.5 is stored in the repulsion object at offset 0x170 -- and 0x170 IS
      ``pexp1``, the FIRST exponential power.  It was assigned to a spare
      ``stored_scalar_1p5`` field while 2.068 (the erf-CN steepness ``k``,
      a different constant entirely) was passed as ``exp_power_1``.  Reading
      the live container's 0x170 out of a running calculator settles it: the
      binary damps with ``exp(-(r+roff)**1.5 * zz)``.
    * 2.0 is the second exponential power.
    * 0.73 and 0.0046511298 scale/weight the second exponential.
    * 0.0110955395 and 0.0116077951 are the global rvdw/R cubic/quartic
      coefficients loaded in the pair matrix loop.
    * 0.0120981314 and 0.0085442527 are the light/heavy atom-class constants
      selected by ``Z < 3`` in the constructor loop.
    * The two recovered average objects are geometric (ID 1, for rvdw scale)
      and arithmetic (ID 0, for pair coefficient matrices).
    """

    lit = GXTB_REPULSION_LITERALS
    return GXTBReconstructedRepulsionConstants(
        stored_scalar_1p5=lit["exp_power_1"],
        exp_power_1=lit["exp_power_1"],
        exp_power_2=lit["exp_power_2"],
        exp2_scale=lit["exp2_scale"],
        exp2_weight=lit["exp2_weight"],
        cubic_coeff=lit["cubic_coeff"],
        quartic_coeff=lit["quartic_coeff"],
        light_pair_coeff=lit["light_pair_coeff"],
        heavy_pair_coeff=lit["heavy_pair_coeff"],
    )


def _arithmetic_pair_average(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    return 0.5 * (arr[:, None] + arr[None, :])


def _geometric_pair_average(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if np.any(arr < 0.0):
        raise ValueError("geometric average requires non-negative values")
    return np.sqrt(arr[:, None] * arr[None, :])


def _coefficient_matrices(
    atomic_numbers: np.ndarray,
    constants: GXTBReconstructedRepulsionConstants,
) -> tuple[np.ndarray, np.ndarray]:
    atoms = np.asarray(atomic_numbers, dtype=np.intp)
    atom_k1 = np.asarray(GXTB_PARAMS["pa_rep_k1"][atoms - 1], dtype=np.float64)
    atom_coeff = np.where(
        atoms < 3,
        constants.light_pair_coeff,
        constants.heavy_pair_coeff,
    ).astype(np.float64)
    linear = _arithmetic_pair_average(atom_k1)
    quadratic = _arithmetic_pair_average(atom_coeff)
    np.fill_diagonal(linear, 0.0)
    np.fill_diagonal(quadratic, 0.0)
    return linear, quadratic


def _vdw_pair_matrix_bohr(
    atomic_numbers: np.ndarray,
    constants: GXTBReconstructedRepulsionConstants,
) -> np.ndarray:
    atoms = np.asarray(atomic_numbers, dtype=np.intp)
    scale = np.asarray(GXTB_PARAMS["pa_rvdw_scale"][atoms - 1], dtype=np.float64)
    # `add_repulsion` scales the MCTC vdW pair radius with the ARITHMETIC
    # average of pa_rvdw_scale, not the geometric one:
    #     rad(jzp,izp) = get_vdw_rad_pair_num(...)
    #                    * arithmetic_average(pa_rvdw_scale(i), pa_rvdw_scale(j))
    pair_scale = _arithmetic_pair_average(scale)
    pair = mctc_vdw_pair_matrix_bohr(atoms) * pair_scale
    np.fill_diagonal(pair, 0.0)
    return pair


def _pair_roffset_matrix(atomic_numbers: np.ndarray) -> np.ndarray:
    atoms = np.asarray(atomic_numbers, dtype=np.intp)
    atom_roffset = np.asarray(GXTB_PARAMS["pa_rep_roffset"][atoms - 1], dtype=np.float64)
    # `new_repulsion_gxtb` builds roff with GEOMETRIC average, not arithmetic:
    #     self%roff(jzp, izp) = geometric_average(roffset(izp), roffset(jzp))
    # and tblite_utils_average::geometric_average is sqrt(a*b).
    pair = _geometric_pair_average(atom_roffset)
    np.fill_diagonal(pair, 0.0)
    return pair


def _gxtb_erf_coordination_number(
    atomic_numbers: np.ndarray,
    coords_ang: np.ndarray,
    *,
    k: float = 2.068,
    power: float = 1.0,
    cutoff: float = 25.0,
) -> np.ndarray:
    """Binary-guided g-xTB ``mctc_ncoord`` ERF coordination number.

    ``new_gxtb_calculator`` selects the erf coordination number, whose
    steepness is ``GXTB_CN_K``. The recovered pair formula is:

    ``0.5 * (1 + erf(-k * (r - r0) / r0**power))``.
    """

    # ⚠️ BOHR, not angstrom.  `erf_coordination_number`'s own docstring says
    # "coordinates and radii must be in the same units", the count enters as
    # erf(-k (r - r0)/r0), and `pa_cn_rcov` is in bohr -- so passing angstrom
    # made r 1.8897x smaller relative to r0 and inflated every count by about
    # ten.  Verified against the binary: `tb_coulomb`'s own `cache%cn` for
    # CH3SH is [0.30946, 0.20788, 0.08615, 0.08865, 0.08865, 0.16185] and this
    # reproduces it to 4.8e-06, where angstrom gave [2.94, 1.88, 0.887, ...].
    return erf_coordination_number(
        atomic_numbers,
        np.asarray(coords_ang, dtype=np.float64) * ANG_TO_BOHR,
        GXTB_PARAMS["pa_cn_rcov"],
        k=k,
        power=power,
        cutoff=cutoff,
    )


def gxtb_reconstructed_repulsion(
    atomic_numbers: np.ndarray | list[int],
    coords_ang: np.ndarray,
    *,
    descriptor: np.ndarray | None = None,
    cn: np.ndarray | None = None,
    cutoff_bohr: float = 25.0,
) -> GXTBReconstructedRepulsion:
    """Compute the current reconstructed g-xTB repulsion energy/gradient.

    Coordinates are supplied in Angstrom. The native pair loop runs in Bohr and
    returns dE/dBohr, which is converted to Hartree/Angstrom for the public
    gradient field.
    """

    atoms = np.asarray(atomic_numbers, dtype=np.intp)
    coords = np.asarray(coords_ang, dtype=np.float64)
    if coords.shape != (atoms.size, 3):
        raise ValueError("coords_ang must have shape (nat, 3)")

    constants = repulsion_constants_from_binary()
    cn_arr = _gxtb_erf_coordination_number(atoms, coords) if cn is None else np.asarray(cn, dtype=np.float64)
    if descriptor is None:
        descriptor_arr = np.zeros(atoms.size, dtype=np.float64)
    else:
        descriptor_arr = np.asarray(descriptor, dtype=np.float64)
    if cn_arr.shape != atoms.shape:
        raise ValueError("cn must have one value per atom")
    if descriptor_arr.shape != atoms.shape:
        raise ValueError("descriptor must have one value per atom")

    pair_roffset = _pair_roffset_matrix(atoms)
    state = repulsion_state(
        atoms,
        descriptor=descriptor_arr,
        cn=cn_arr,
        pair_roffset=pair_roffset,
    )
    pair_rvdw = _vdw_pair_matrix_bohr(atoms, constants)
    linear_coeff, quadratic_coeff = _coefficient_matrices(atoms, constants)

    energy, gradient_bohr, matvec = repulsion_energy_gradient_asm(
        coords * ANG_TO_BOHR,
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
        cutoff=cutoff_bohr,
    )

    # ⚠️ THE COORDINATION-NUMBER CHAIN RULE.
    # `gxtb_repulsion::get_gradient` ends with
    #     call self%ncoord%get_dcn(mol, trans, dEdcn, gradient, sigma)
    # because zeta = (kcn*(sqrt(cn)-..)+1)*alpha depends on the geometry
    # THROUGH cn, and `get_repulsion_derivs` fills dEdcn for exactly that
    # contraction.  The reference kernel differentiates the explicit
    # r-dependence only, so this term was absent -- not wrong, absent -- and
    # the gradient was 1.9e-2 to 3.1e-2 Ha/Bohr out.  Adding it closes the
    # deviation to 1.1e-16 / 3.4e-16 / 5.6e-17.
    #
    # Everything below is in BOHR.  E = sum_{i<j} mat_ij z_i z_j, so
    #     dE/dcn_i = 0.5 * sum_{j!=i} z_i z_j poly_ij (ddamp/dzz)
    #                      * (zeta_j/(zeta_i+zeta_j))**2 * dzeta_i/dcn_i
    # and the CN itself is the erf form 0.5(1+erf(-k(r-r0)/r0)).
    xyz_b = coords * ANG_TO_BOHR
    zeta = np.asarray(state.alpha, dtype=np.float64)
    szeff = np.asarray(state.scaled_zeff, dtype=np.float64)
    dzeta_dcn = np.asarray(state.dalpha_dcn, dtype=np.float64)
    nat = atoms.size
    d = xyz_b[:, None, :] - xyz_b[None, :, :]
    r = np.linalg.norm(d, axis=2)
    np.fill_diagonal(r, 1.0)
    inv_r = 1.0 / r
    zz = zeta[:, None] * zeta[None, :] / (zeta[:, None] + zeta[None, :])
    x = pair_rvdw * inv_r
    x2 = x * x
    rr = r + pair_roffset
    e1 = np.exp(-(rr ** constants.exp_power_1 * zz))
    e2 = np.exp(-(rr ** constants.exp_power_2 * (zz * constants.exp2_scale)))
    poly = (1.0 + linear_coeff * inv_r + quadratic_coeff * x2
            + constants.cubic_coeff * x2 * x + constants.quartic_coeff * x2 * x2)
    ddamp = (-(rr ** constants.exp_power_1) * e1
             - constants.exp2_weight * constants.exp2_scale
             * (rr ** constants.exp_power_2) * e2)
    dzz_i = (zeta[None, :] / (zeta[:, None] + zeta[None, :])) ** 2
    w = 0.5 * szeff[:, None] * szeff[None, :] * poly * ddamp * dzz_i
    np.fill_diagonal(w, 0.0)
    dEdcn = w.sum(axis=1) * dzeta_dcn

    rc = np.asarray(GXTB_PARAMS["pa_cn_rcov"], dtype=np.float64)[atoms - 1]
    r0 = rc[:, None] + rc[None, :]
    u = -GXTB_CN_K * (r - r0) / r0
    dcn = (1.0 / np.sqrt(np.pi)) * np.exp(-u * u) * (-GXTB_CN_K / r0)
    np.fill_diagonal(dcn, 0.0)
    pref = (dEdcn[:, None] + dEdcn[None, :]) * dcn * inv_r
    grad_cn = np.einsum("ij,ijk->ik", pref, d) - np.einsum("ij,ijk->jk", pref, d)
    gradient_bohr = np.asarray(gradient_bohr, dtype=np.float64) + grad_cn

    return GXTBReconstructedRepulsion(
        energy=float(energy),
        # ⚠️ TWO UNITS, and the field name is the only thing that says which.
        # `gradient` is HARTREE PER ANGSTROM, because that is what the existing
        # callers and the finite-difference test expect.  `gradient_bohr` is
        # the binary's own convention, Hartree per Bohr, and is what any
        # comparison against it must use.
        gradient=np.asarray(gradient_bohr, dtype=np.float64) * ANG_TO_BOHR,
        gradient_bohr=np.asarray(gradient_bohr, dtype=np.float64),
        matvec=np.asarray(matvec, dtype=np.float64),
        state=state,
        cn=cn_arr,
        pair_rvdw=pair_rvdw,
        pair_roffset=pair_roffset,
        linear_coeff=linear_coeff,
        quadratic_coeff=quadratic_coeff,
        constants=constants,
        metadata={
            "component": "gxtb_repulsion",
            "reconstruction": "binary-guided",
            "complete_gxtb": False,
            "missing_for_full_energy": (
                "EEQ_BC",
                "q-vSZP basis",
                "overlap/H0",
                "shell-charge SCC",
                "exchange",
                "anisotropic H0",
                "p-ACP",
                "D4Srev",
            ),
            "pair_builder_status": "candidate; pair-loop kernel/constants, average IDs, exact MCTC vdW pair table, and ERF CN type/k recovered; H0/SCC still absent",
        },
    )
