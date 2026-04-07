"""
SchNet-like GNN for per-atom charge prediction on MLX Metal.

Architecture:
  Embedding(Z → hidden) → CFConv × n_layers → Output MLP → charges (N,)

Uses mlx-graphs MessagePassing for scatter aggregation.
Charges are scalar (rotation-invariant), so no E(3) equivariance needed.

Supports:
  - Direct mode: coordinates → DFT charges
  - Δ-ML mode:  coordinates + PM6 charges → charge corrections
"""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
from mlx_graphs.nn import MessagePassing
from mlx_graphs.data import GraphData, GraphDataBatch

from .graph_builder import MAX_ELEMENT_IDX


def _silu(x: mx.array) -> mx.array:
    return x * mx.sigmoid(x)


class FilterNet(nn.Module):
    """2-layer MLP for continuous-filter convolution.

    Maps RBF edge features → per-edge filter weights.
    f(d) = Linear(SiLU(Linear(rbf)))
    """

    def __init__(self, n_rbf: int, hidden: int):
        super().__init__()
        self.lin1 = nn.Linear(n_rbf, hidden)
        self.lin2 = nn.Linear(hidden, hidden)

    def __call__(self, edge_features: mx.array) -> mx.array:
        return self.lin2(_silu(self.lin1(edge_features)))


class CFConvBlock(MessagePassing):
    """Continuous-filter convolution block (SchNet-style).

    message(i←j) = x_j ⊙ W(rbf(d_ij))
    aggregate → scatter_sum
    update → MLP + residual + LayerNorm
    """

    def __init__(self, hidden: int, n_rbf: int):
        super().__init__(aggr="add")
        self.filter_net = FilterNet(n_rbf, hidden)
        self.lin1 = nn.Linear(hidden, hidden)
        self.lin2 = nn.Linear(hidden, hidden)
        self.norm = nn.LayerNorm(hidden)

    def __call__(
        self,
        node_features: mx.array,
        edge_index: mx.array,
        edge_features: mx.array,
    ) -> mx.array:
        """Forward pass.

        Args:
            node_features: (N, hidden)
            edge_index: (2, E)
            edge_features: (E, n_rbf)

        Returns:
            updated node features (N, hidden)
        """
        # Compute filter weights from edge features (RBF distances)
        W = self.filter_net(edge_features)  # (E, hidden)

        # Message passing: gather source features, multiply by filter, scatter to dst
        out = self.propagate(
            edge_index=edge_index,
            node_features=node_features,
            message_kwargs={"W": W},
        )

        # Post-aggregation update MLP
        out = self.lin2(_silu(self.lin1(out)))  # (N, hidden)

        # Residual + LayerNorm
        return self.norm(node_features + out)

    def message(self, src_features: mx.array, dst_features: mx.array, **kwargs):
        """Compute messages: source features ⊙ filter weights."""
        W = kwargs["W"]  # (E, hidden)
        return src_features * W


class SchNetCharge(nn.Module):
    """SchNet-like model for per-atom charge prediction.

    Args:
        hidden: hidden dimension (default 128)
        n_layers: number of CFConv blocks (default 4)
        n_rbf: number of RBF basis functions (default 64)
        max_z: maximum element index for embedding
        delta_ml: if True, expects PM6 charges as extra node feature
        enforce_neutrality: if True, applies hard charge neutrality constraint
    """

    def __init__(
        self,
        hidden: int = 128,
        n_layers: int = 4,
        n_rbf: int = 64,
        max_z: int = MAX_ELEMENT_IDX,
        delta_ml: bool = False,
        enforce_neutrality: bool = True,
    ):
        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.delta_ml = delta_ml
        self.enforce_neutrality = enforce_neutrality

        # Input projection: one-hot Z [+ PM6 charge] → hidden
        in_dim = max_z + (1 if delta_ml else 0)
        self.input_proj = nn.Linear(in_dim, hidden)

        # Interaction blocks
        self.conv_blocks = [CFConvBlock(hidden, n_rbf) for _ in range(n_layers)]

        # Output head: hidden → 1 (per-atom charge)
        self.out_lin1 = nn.Linear(hidden, hidden // 2)
        self.out_lin2 = nn.Linear(hidden // 2, 1)

    def __call__(self, graph: GraphData | GraphDataBatch) -> mx.array:
        """Forward pass: graph → per-atom charges.

        Args:
            graph: GraphData or GraphDataBatch with:
                - node_features: (N_total, n_feat)
                - edge_index: (2, E_total)
                - edge_features: (E_total, n_rbf)
                - graph_labels: (n_graphs, 1) formal charges [optional]

        Returns:
            charges: (N_total,) per-atom predicted charges
        """
        x = self.input_proj(graph.node_features)  # (N, hidden)
        ei = graph.edge_index                       # (2, E)
        ef = graph.edge_features                    # (E, n_rbf)

        # Interaction blocks
        for conv in self.conv_blocks:
            x = conv(x, ei, ef)

        # Output MLP → per-atom raw charge
        q_raw = self.out_lin2(_silu(self.out_lin1(x))).squeeze(-1)  # (N,)

        # Hard charge neutrality constraint
        if self.enforce_neutrality:
            q_raw = self._neutrality(q_raw, graph)

        return q_raw

    def _neutrality(self, q_raw: mx.array, graph) -> mx.array:
        """Apply hard charge neutrality: shift per-molecule to match formal charge.

        q_i = q_raw_i - mean(q_raw per mol) + Q_formal / N_atoms_per_mol
        """
        if hasattr(graph, '_batch_indices') and graph._batch_indices is not None:
            # Batched: multiple molecules
            batch_idx = graph._batch_indices  # (N_total,)
            n_graphs = graph.num_graphs if hasattr(graph, 'num_graphs') else int(mx.max(batch_idx).item()) + 1

            # Per-molecule sum of raw charges
            q_sum = mx.zeros(n_graphs)
            # One-hot scatter for sum
            oh = (batch_idx[:, None] == mx.arange(n_graphs)[None, :]).astype(q_raw.dtype)
            q_sum = oh.T @ q_raw  # (n_graphs,)

            # Atoms per molecule
            n_per_mol = oh.T @ mx.ones_like(q_raw)  # (n_graphs,)

            # Target per-molecule charge
            if graph.graph_labels is not None:
                Q_formal = graph.graph_labels.squeeze()  # (n_graphs,)
            else:
                Q_formal = mx.zeros(n_graphs)

            # Mean shift per molecule
            shift = (Q_formal - q_sum) / (n_per_mol + 1e-10)  # (n_graphs,)

            # Broadcast back to atoms
            q_raw = q_raw + shift[batch_idx]
        else:
            # Single molecule
            N = q_raw.shape[0]
            Q_formal = 0.0
            if graph.graph_labels is not None:
                Q_formal = float(graph.graph_labels[0])
            q_mean = mx.mean(q_raw)
            q_raw = q_raw - q_mean + Q_formal / N

        return q_raw
