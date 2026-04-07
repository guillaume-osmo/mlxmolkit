"""
Graph construction: coordinates → edge_index + RBF features → GraphData.

Converts molecular geometry into graph representation for GNN training/inference.
Reuses patterns from pm6_ml_mlx.py (edge building, RBF expansion, cosine cutoff).
"""
from __future__ import annotations

import numpy as np
import mlx.core as mx
from mlx_graphs.data import GraphData


# Supported elements → index (0-based, for one-hot encoding)
# Covers CHONPS + halogens + metals commonly in SPICE/drug-like molecules
ELEMENT_TO_IDX = {
    1: 0, 5: 1, 6: 2, 7: 3, 8: 4, 9: 5, 11: 6, 12: 7, 14: 8, 15: 9,
    16: 10, 17: 11, 19: 12, 20: 13, 26: 14, 29: 15, 30: 16, 35: 17, 53: 18,
}
MAX_ELEMENT_IDX = len(ELEMENT_TO_IDX)  # 19 classes


def build_edges_np(
    coords: np.ndarray,
    cutoff: float = 5.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build undirected edge list from coordinates with distance cutoff.

    Args:
        coords: (N, 3) atomic coordinates in Angstrom
        cutoff: distance cutoff in Angstrom

    Returns:
        src: (E,) source node indices (int32)
        dst: (E,) destination node indices (int32)
        distances: (E,) edge distances (float32)
    """
    N = len(coords)
    # Pairwise distances
    diff = coords[:, None, :] - coords[None, :, :]       # (N, N, 3)
    dist_sq = np.sum(diff * diff, axis=2)                  # (N, N)
    dist = np.sqrt(dist_sq + 1e-20).astype(np.float32)

    # Find pairs within cutoff (excluding self-loops)
    mask = (dist < cutoff) & (np.eye(N, dtype=bool) == False)
    src_np, dst_np = np.where(mask)

    return (
        src_np.astype(np.int32),
        dst_np.astype(np.int32),
        dist[src_np, dst_np].astype(np.float32),
    )


def gaussian_rbf(
    distances: np.ndarray,
    n_rbf: int = 64,
    cutoff: float = 5.0,
) -> np.ndarray:
    """Gaussian radial basis functions.

    μ_k evenly spaced in [0, cutoff], σ = (μ_1 - μ_0)
    f_k(d) = exp(-0.5 * ((d - μ_k) / σ)²)

    Args:
        distances: (E,) edge distances
        n_rbf: number of basis functions
        cutoff: upper cutoff

    Returns:
        (E, n_rbf) basis function values
    """
    mu = np.linspace(0, cutoff, n_rbf, dtype=np.float32)
    sigma = mu[1] - mu[0]
    d = distances[:, None]  # (E, 1)
    return np.exp(-0.5 * ((d - mu[None, :]) / sigma) ** 2).astype(np.float32)


def expnorm_rbf(
    distances: np.ndarray,
    n_rbf: int = 64,
    cutoff: float = 5.0,
) -> np.ndarray:
    """ExpNormal radial basis functions (same as TorchMD-NET / PM6-ML).

    f_k(d) = C(d) * exp(-β * (exp(-αd) - μ_k)²)
    where α = 5/cutoff, C(d) = cosine_cutoff

    Args:
        distances: (E,) edge distances
        n_rbf: number of basis functions
        cutoff: upper cutoff

    Returns:
        (E, n_rbf) basis function values
    """
    alpha = 5.0 / cutoff
    start = np.exp(-cutoff * alpha)
    mu = np.linspace(start, 1.0, n_rbf, dtype=np.float32)
    beta = np.full(n_rbf, (2.0 / n_rbf * (1.0 - start)) ** -2, dtype=np.float32)

    d = distances[:, None]  # (E, 1)
    # Cosine cutoff envelope
    C = np.where(d < cutoff, 0.5 * (np.cos(d * np.pi / cutoff) + 1.0), 0.0)
    return (C * np.exp(-beta * (np.exp(-alpha * d) - mu) ** 2)).astype(np.float32)


def _one_hot_z(atomic_numbers: list[int] | np.ndarray) -> np.ndarray:
    """One-hot encode atomic numbers → (N, MAX_ELEMENT_IDX) float32."""
    N = len(atomic_numbers)
    oh = np.zeros((N, MAX_ELEMENT_IDX), dtype=np.float32)
    for i, z in enumerate(atomic_numbers):
        idx = ELEMENT_TO_IDX.get(int(z), 0)  # default to H if unknown
        oh[i, idx] = 1.0
    return oh


def build_graph(
    atomic_numbers: list[int] | np.ndarray,
    coords: np.ndarray,
    charges: np.ndarray | None = None,
    cutoff: float = 5.0,
    n_rbf: int = 64,
    rbf_type: str = "expnorm",
    pm6_charges: np.ndarray | None = None,
    formal_charge: float = 0.0,
) -> GraphData:
    """Build a molecular graph for GNN charge prediction.

    Args:
        atomic_numbers: (N,) atomic numbers
        coords: (N, 3) coordinates in Angstrom
        charges: (N,) target charges (for training). None for inference.
        cutoff: distance cutoff for edges
        n_rbf: number of RBF basis functions
        rbf_type: "gaussian" or "expnorm"
        pm6_charges: (N,) PM6 Mulliken charges for Δ-ML mode. None for direct mode.
        formal_charge: total molecular charge (for neutrality constraint)

    Returns:
        GraphData with:
          - node_features: (N, n_feat) one-hot Z [+ PM6 charges if provided]
          - edge_index: (2, E) source/destination pairs
          - edge_features: (E, n_rbf) RBF-expanded distances
          - node_labels: (N, 1) target charges [or None]
          - graph_labels: (1,) formal charge
    """
    coords = np.asarray(coords, dtype=np.float32)
    N = len(atomic_numbers)

    # Node features: one-hot atomic number
    nf = _one_hot_z(atomic_numbers)

    # Optional: append PM6 charges as extra feature (Δ-ML mode)
    if pm6_charges is not None:
        pm6 = np.asarray(pm6_charges, dtype=np.float32).reshape(N, 1)
        nf = np.concatenate([nf, pm6], axis=1)

    # Edges + RBF features
    src, dst, dists = build_edges_np(coords, cutoff)
    if len(src) == 0:
        # Isolated atoms — create a minimal graph
        src = np.array([0], dtype=np.int32)
        dst = np.array([0], dtype=np.int32)
        dists = np.array([0.0], dtype=np.float32)

    if rbf_type == "gaussian":
        ef = gaussian_rbf(dists, n_rbf, cutoff)
    else:
        ef = expnorm_rbf(dists, n_rbf, cutoff)

    # Edge index: (2, E)
    ei = np.stack([src, dst], axis=0)

    # Labels
    nl = None
    if charges is not None:
        nl = mx.array(np.asarray(charges, dtype=np.float32).reshape(N, 1))

    gl = mx.array(np.array([formal_charge], dtype=np.float32))

    return GraphData(
        edge_index=mx.array(ei),
        node_features=mx.array(nf),
        edge_features=mx.array(ef),
        node_labels=nl,
        graph_labels=gl,
    )
