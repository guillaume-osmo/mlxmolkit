"""mlxmolkit.nddo — NDDO semi-empirical SCF on Apple Silicon.

Public API — only the entry points covered by the test suite.
Lower-level internals are reachable via submodule imports but are not
part of the stable surface.

Tested entry points
-------------------

SCF
    nddo_energy(atoms, coords, method='RM1') -> dict
        Single-molecule SCF. Returns electronic + nuclear + heat-of-formation
        energies plus the converged density. Methods include MNDO, RM1, AM1,
        PM3, PM6 and PM7. PM6_D aliases PM6. PM7 covers 11 main-group elements
        and includes geometry corrections in its energy and gradient.
        Tests: tests/test_rm1_scf.py, tests/test_pm6_d_native.py

    nddo_energy_batch(molecules, method='RM1') -> list[dict]
        Batched version. Tests: tests/test_rm1_scf.py

PM6-D3H4 corrections (post-SCF)
    pm6_d3h4_correction(atoms, coords) -> float
        Sum of Grimme D3 dispersion + Rezáč-Hobza H4 H-bond
        + HH-repulsion (eV). Tests: tests/test_pm6_d3h4.py

    d3_energy(atoms, coords) -> float
        D3 dispersion, zero damping (the PM6-D3H4 variant).

    d3bj_energy(atoms, coords) -> dict
        D3 dispersion, Becke-Johnson rational damping. Validated against
        Grimme's simple-dftd3 to ~1e-9 relative. This is the variant PM6-ML
        uses; it is NOT interchangeable with the zero-damping form.

    x_energy(atoms, coords) -> float
        Halogen-bond correction for Cl/Br/I with N/O/S.

    pm6_d3h4x_correction(atoms, coords) -> dict
        PM6-D3H4 plus the halogen-bond term.
    h4_energy(atoms, coords) -> float
    hh_repulsion(atoms, coords) -> float
        Individual components.

Parameters
    METHOD_PARAMS : dict[str, dict[int, ElementParams]]
        Method-name -> Z -> parameters lookup.
    get_params(method) -> dict[int, ElementParams]
    ElementParams
        Dataclass holding one element's NDDO parameters.

Integral primitives (vendored from PYSEQM, BSD-3, LANL)
    These use current MOPAC CODATA constants. Port regression snapshots and
    independent MOPAC comparisons are tested separately.

    from mlxmolkit.nddo._pyseqm_port import (
        diatom_overlap_matrixD,     # qn=1..6 diatomic overlap (incl. d)
        two_elec_two_center_int,    # full TETCI per pair
        qn_int, qnD_int,            # periodic-table tables
    )
"""

# --- public API ---
from .scf import (nddo_energy, nddo_energy_batch, nddo_energy_many,
                  shutdown_worker_pool)
from .gradient import nddo_gradient, nddo_optimize, nddo_optimize_batch
from .pm6_d3h4 import (
    pm6_d3h4_correction,
    pm6_d3h4x_correction,
    d3bj_correction,
    d3_energy,
    d3bj_energy,
    x_energy,
    h4_energy,
    hh_repulsion,
)
from .methods import METHOD_PARAMS, get_params
from .params import ElementParams, ANG_TO_BOHR, principal_qn, RM1_PARAMS

__all__ = [
    # SCF
    "nddo_energy",
    "nddo_energy_batch",
    "nddo_energy_many",
    "shutdown_worker_pool",
    # Gradient + geometry optimization
    "nddo_gradient",
    "nddo_optimize",
    "nddo_optimize_batch",
    # PM6-D3H4 corrections
    "pm6_d3h4_correction",
    "pm6_d3h4x_correction",
    "d3bj_correction",
    "d3_energy",
    "d3bj_energy",
    "x_energy",
    "h4_energy",
    "hh_repulsion",
    # Parameters
    "METHOD_PARAMS",
    "get_params",
    "ElementParams",
    "RM1_PARAMS",
    # Constants
    "ANG_TO_BOHR",
    "principal_qn",
]
