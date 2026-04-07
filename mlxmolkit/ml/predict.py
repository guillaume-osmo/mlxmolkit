"""
Inference API for charge prediction models.

ChargePredictor: load a trained SchNet model and predict charges
from atomic numbers + coordinates.
"""
from __future__ import annotations

import numpy as np
import mlx.core as mx

from .schnet_charge import SchNetCharge
from .graph_builder import build_graph, MAX_ELEMENT_IDX


class ChargePredictor:
    """Predict per-atom partial charges from molecular geometry.

    Usage:
        predictor = ChargePredictor.from_checkpoint("best_model.npz")
        charges = predictor.predict([6, 1, 1, 1, 1], coords)
    """

    def __init__(
        self,
        model: SchNetCharge,
        cutoff: float = 5.0,
        n_rbf: int = 64,
        rbf_type: str = "expnorm",
    ):
        self.model = model
        self.cutoff = cutoff
        self.n_rbf = n_rbf
        self.rbf_type = rbf_type

    @classmethod
    def from_checkpoint(
        cls,
        path: str,
        hidden: int = 128,
        n_layers: int = 4,
        n_rbf: int = 64,
        delta_ml: bool = False,
        cutoff: float = 5.0,
    ) -> "ChargePredictor":
        """Load from saved weights."""
        model = SchNetCharge(
            hidden=hidden,
            n_layers=n_layers,
            n_rbf=n_rbf,
            delta_ml=delta_ml,
            enforce_neutrality=True,
        )
        model.load_weights(path)
        model.eval()
        return cls(model, cutoff=cutoff, n_rbf=n_rbf)

    def predict(
        self,
        atomic_numbers: list[int] | np.ndarray,
        coords: np.ndarray,
        formal_charge: float = 0.0,
    ) -> np.ndarray:
        """Predict charges for a single molecule.

        Args:
            atomic_numbers: (N,) atomic numbers
            coords: (N, 3) coordinates in Angstrom
            formal_charge: total molecular charge

        Returns:
            (N,) predicted partial charges in electron units
        """
        graph = build_graph(
            atomic_numbers, coords,
            cutoff=self.cutoff,
            n_rbf=self.n_rbf,
            rbf_type=self.rbf_type,
            formal_charge=formal_charge,
        )
        q = self.model(graph)
        mx.eval(q)
        return np.array(q)

    def predict_delta(
        self,
        atomic_numbers: list[int] | np.ndarray,
        coords: np.ndarray,
        pm6_charges: np.ndarray,
        formal_charge: float = 0.0,
    ) -> np.ndarray:
        """Δ-ML prediction: PM6 charges + learned correction.

        Args:
            atomic_numbers: (N,) atomic numbers
            coords: (N, 3) coordinates in Angstrom
            pm6_charges: (N,) PM6 Mulliken charges (baseline)
            formal_charge: total molecular charge

        Returns:
            (N,) corrected charges = PM6 + Δq_predicted
        """
        graph = build_graph(
            atomic_numbers, coords,
            cutoff=self.cutoff,
            n_rbf=self.n_rbf,
            rbf_type=self.rbf_type,
            pm6_charges=pm6_charges,
            formal_charge=formal_charge,
        )
        delta_q = self.model(graph)
        mx.eval(delta_q)
        return pm6_charges + np.array(delta_q)

    def predict_batch(
        self,
        molecules: list[tuple],
        formal_charges: list[float] | None = None,
    ) -> list[np.ndarray]:
        """Predict charges for multiple molecules.

        Args:
            molecules: list of (atomic_numbers, coords) tuples
            formal_charges: list of formal charges (default: all 0)

        Returns:
            list of (N_i,) charge arrays
        """
        from mlx_graphs.loaders import Dataloader

        if formal_charges is None:
            formal_charges = [0.0] * len(molecules)

        graphs = []
        for (atoms, coords), fc in zip(molecules, formal_charges):
            g = build_graph(
                atoms, coords,
                cutoff=self.cutoff,
                n_rbf=self.n_rbf,
                rbf_type=self.rbf_type,
                formal_charge=fc,
            )
            graphs.append(g)

        loader = Dataloader(graphs, batch_size=32, shuffle=False)
        all_charges = []
        offset = 0

        for batch in loader:
            q = self.model(batch)
            mx.eval(q)
            q_np = np.array(q)

            # Split back into per-molecule arrays
            if hasattr(batch, '_batch_indices') and batch._batch_indices is not None:
                bi = np.array(batch._batch_indices)
                n_graphs = int(bi.max()) + 1
                for g_idx in range(n_graphs):
                    mask = bi == g_idx
                    all_charges.append(q_np[mask])
            else:
                all_charges.append(q_np)

        return all_charges
