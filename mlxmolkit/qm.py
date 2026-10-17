"""Session facade over the NDDO SCF and its charge models.

``QMSession`` fixes a method, backend and charge model once, so callers stop
re-spelling them on every call:

    qm = QMSession(method="PM6_D3H4", backend="mlx", charge_model="resp")
    result = qm.energy(atoms, coords, molecular_charge=0)
    charges = qm.charges(atoms, coords, molecular_charge=0)

It adds no physics. ``energy`` is :func:`mlxmolkit.nddo.nddo_energy` and the
ESP/RESP models are :func:`mlxmolkit.esp_resp.pm6_esp_resp_charge_labels`, so
the low-level functions remain the reference the session is tested against.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

BACKENDS = ("mlx",)
CHARGE_MODELS = ("mulliken", "esp", "resp")


class QMSession:
    """A semi-empirical method, backend and charge model bound together.

    Args:
        method: any registered NDDO method (``get_params`` spelling; ``-`` and
            ``_`` are interchangeable). The dispersion variants ``PM6_D3``,
            ``PM6_D3H4`` and ``PM6_D3H4X`` share PM6's SCF, density and charges
            and differ only in a post-SCF term added to the heat of formation.
        backend: ``"mlx"``, the in-process NDDO stack. External QM backends
            can still feed ``fit_esp_charges_mlx`` / ``fit_resp_charges_mlx``
            directly.
        charge_model: ``"mulliken"`` (the SCF's own charges), ``"esp"`` or
            ``"resp"`` (fits to the NDDO multipole ESP on a Connolly grid).
    """

    def __init__(
        self,
        method: str = "PM6_D3H4",
        backend: str = "mlx",
        charge_model: str = "resp",
    ) -> None:
        from mlxmolkit.nddo import get_params
        from mlxmolkit.nddo.pwcct import normalize_method

        if backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
        if charge_model not in CHARGE_MODELS:
            raise ValueError(
                f"charge_model must be one of {CHARGE_MODELS}, got {charge_model!r}"
            )
        get_params(method)  # unknown methods fail here, not on the first call
        self.method = normalize_method(method)
        self.backend = backend
        self.charge_model = charge_model

    def __repr__(self) -> str:
        return (
            f"QMSession(method={self.method!r}, backend={self.backend!r}, "
            f"charge_model={self.charge_model!r})"
        )

    def energy(
        self,
        atoms: Sequence[int],
        coords: np.ndarray,
        molecular_charge: float = 0.0,
    ) -> dict:
        """Single-point SCF; the ``nddo_energy`` result dict, unchanged.

        Units are in the key names. Post-SCF corrections (the D3H4 dispersion
        and H-bond terms, PM7's corrections) follow MOPAC and are included in
        ``heat_of_formation_kcal`` / ``heat_of_formation_eV`` only;
        ``energy_eV`` is the SCF total. Check ``result["converged"]``.
        """
        from mlxmolkit.nddo import nddo_energy

        return nddo_energy(
            list(map(int, atoms)),
            np.asarray(coords, dtype=np.float64),
            method=self.method,
            molecular_charge=float(molecular_charge),
        )

    def charges(
        self,
        atoms: Sequence[int],
        coords: np.ndarray,
        molecular_charge: float = 0.0,
    ) -> np.ndarray:
        """Partial charges ``(n_atoms,)`` from the session's charge model.

        Raises RuntimeError if the SCF does not converge: charges from an
        unconverged density are not worth returning.
        """
        if self.charge_model == "mulliken":
            result = self.energy(atoms, coords, molecular_charge)
            if not result["converged"]:
                raise RuntimeError(f"{self.method} SCF did not converge")
            return np.asarray(result["charges"], dtype=np.float64)

        from mlxmolkit.esp_resp import pm6_esp_resp_charge_labels

        esp, resp = pm6_esp_resp_charge_labels(
            list(map(int, atoms)),
            np.asarray(coords, dtype=np.float64),
            total_charge=float(molecular_charge),
            method=self.method,
        )
        fit = esp if self.charge_model == "esp" else resp
        return np.asarray(fit.charges, dtype=np.float64)
