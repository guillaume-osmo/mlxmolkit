"""
PM6-ML on MLX Metal GPU — exact port of TorchMD-NET EquivariantTransformer.

No PyTorch dependency at inference time (use from_npz for torch-free loading).

Architecture: 6-layer EquivariantTransformer
  - 128-dim embeddings, 8 heads, 64 RBF
  - SiLU activation, SiLU attention activation
  - Cutoff: 5.0 Å, distance_influence="both", vector_cutoff=True
  - ~1.4M parameters
  - EquivariantScalar output (2× GatedEquivariantBlock)

Exact reimplementation of:
  TorchMD_ET  (representation)  → torchmd_et.py
  EquivariantScalar (output)    → output_modules.py
  TorchMD_Net (wrapper)         → model.py

Port of TorchMD-NET model for Apple Silicon Metal acceleration.
Reference: Nováček & Řezáč, JCTC 2025, 21(2), 678-690.
"""
from __future__ import annotations

import os
import numpy as np
import mlx.core as mx
from typing import Dict

Z_TO_ATYPE = {
    35: 1, 6: 3, 20: 5, 17: 7, 9: 9, 1: 10, 53: 12, 19: 13,
    3: 14, 12: 15, 7: 17, 11: 19, 8: 21, 15: 23, 16: 26,
}

# Model hyperparameters (from PM6-ML checkpoint)
HIDDEN = 128
N_HEADS = 8
N_LAYERS = 6
N_RBF = 64
CUTOFF = 5.0
HEAD_DIM = HIDDEN // N_HEADS  # 16


# ─── helper functions ─────────────────────────────────────────────────

def _scatter_sum(src: mx.array, index: mx.array, dim_size: int) -> mx.array:
    """Scatter-add along dim 0: src (E, ...) → (dim_size, ...).

    Uses one-hot matmul — efficient for small molecules on Metal.
    """
    oh = (index[:, None] == mx.arange(dim_size)[None, :]).astype(src.dtype)  # (E, N)
    flat = src.reshape(src.shape[0], -1)                                      # (E, D)
    out = oh.T @ flat                                                          # (N, D)
    return out.reshape(dim_size, *src.shape[1:])


def _cosine_cutoff(d: mx.array) -> mx.array:
    """CosineCutoff(0, 5 Å) — matches TorchMD-NET exactly."""
    return mx.where(d < CUTOFF,
                    0.5 * (mx.cos(d * (np.pi / CUTOFF)) + 1.0),
                    mx.zeros_like(d))


def _silu(x: mx.array) -> mx.array:
    """SiLU (Swish) activation."""
    return x * mx.sigmoid(x)


def _layer_norm(x: mx.array, w: mx.array, b: mx.array) -> mx.array:
    """LayerNorm on last dim (eps=1e-5, biased variance = ddof 0)."""
    mu = mx.mean(x, axis=-1, keepdims=True)
    var = mx.var(x, axis=-1, keepdims=True)  # ddof=0 matches PyTorch LayerNorm
    return (x - mu) * mx.rsqrt(var + 1e-5) * w + b


# ─── model ────────────────────────────────────────────────────────────

class PM6ML_MLX:
    """PM6-ML EquivariantTransformer on MLX Metal — exact TorchMD-NET port."""

    def __init__(self, weights: Dict[str, mx.array]):
        self.w = weights

    # ── constructors ──────────────────────────────────────────────────

    @classmethod
    def from_checkpoint(cls, ckpt_path: str) -> "PM6ML_MLX":
        """Load from TorchMD-NET .ckpt (requires torch for conversion)."""
        import torch
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
        w = {k: mx.array(v.detach().numpy().astype(np.float32)) for k, v in sd.items()}
        return cls(w)

    @classmethod
    def from_npz(cls, path: str) -> "PM6ML_MLX":
        """Load from pre-converted .npz (no torch dependency)."""
        d = np.load(path, allow_pickle=True)
        return cls({k: mx.array(d[k]) for k in d.files})

    def save_npz(self, path: str):
        """Save weights as .npz for torch-free inference."""
        np.savez_compressed(path, **{k: np.array(v) for k, v in self.w.items()})

    # ── edge construction ─────────────────────────────────────────────

    @staticmethod
    def _build_edges(pos_np: np.ndarray, N: int):
        """Build edge list matching OptimizedDistance(loop=True, include_transpose=True).

        Returns all pairs within cutoff plus self-loops.
        Convention: edge_index[0]=src, edge_index[1]=dst
                    edge_vec = pos[src] - pos[dst]  (TorchMD-NET convention)
        """
        src_l, dst_l = [], []
        for i in range(N):
            for j in range(N):
                if i == j:
                    src_l.append(i)
                    dst_l.append(j)
                else:
                    d = np.linalg.norm(pos_np[i] - pos_np[j])
                    if d < CUTOFF:
                        src_l.append(i)
                        dst_l.append(j)

        src_np = np.array(src_l, dtype=np.int32)
        dst_np = np.array(dst_l, dtype=np.int32)
        is_self = src_np == dst_np

        # edge vectors: pos[src] - pos[dst] (TorchMD-NET convention)
        ev = pos_np[src_np] - pos_np[dst_np]               # (E, 3)
        ew = np.linalg.norm(ev, axis=1).astype(np.float32)  # (E,)

        # Normalize non-self edges; self-loop vectors stay zero
        safe = np.where(is_self, 1.0, ew)
        ev_norm = (ev / safe[:, None]).astype(np.float32)
        ev_norm[is_self] = 0.0
        ew[is_self] = 0.0

        return (mx.array(src_np), mx.array(dst_np),
                mx.array(ew), mx.array(ev_norm),
                mx.array(is_self))

    # ── RBF expansion ─────────────────────────────────────────────────

    def _rbf(self, ew: mx.array) -> mx.array:
        """ExpNormalSmearing(0, 5 Å) with cosine envelope.

        f(d) = CosineCutoff(d) · exp(-β · (exp(-αd) - μ)²)
        α = 5.0 / cutoff = 1.0
        """
        means = self.w["model.representation_model.distance_expansion.means"]
        betas = self.w["model.representation_model.distance_expansion.betas"]
        d = ew[:, None]       # (E, 1)
        C = _cosine_cutoff(d)  # (E, 1)
        return C * mx.exp(-betas * (mx.exp(-d) - means) ** 2)  # (E, N_RBF)

    # ── neighbor embedding ────────────────────────────────────────────

    def _neighbor_embed(self, types, x, src, dst, ew, ea, is_self, N):
        """NeighborEmbedding: aggregate neighbor type info weighted by distance.

        Matches TorchMD-NET: msg from edge_index[1] → aggregate at edge_index[0].
        Self-loops are zeroed out (TorchMD-NET removes them explicitly).
        """
        W = self.w
        # Zero out self-loop contributions
        C = _cosine_cutoff(ew) * (~is_self).astype(mx.float32)  # (E,)

        # Distance projection: (E, 64) → (E, 128), scaled by cutoff
        dp = (ea @ W["model.representation_model.neighbor_embedding.distance_proj.weight"].T
              + W["model.representation_model.neighbor_embedding.distance_proj.bias"])
        dp = dp * C[:, None]  # (E, 128)

        # Neighbor type embedding
        ne = W["model.representation_model.neighbor_embedding.embedding.weight"][types]  # (N, 128)

        # Message: dp * embedding[dst], scatter to src
        # (TorchMD-NET uses reversed convention for NE but equivalent due to bidirectional edges)
        msg = dp * ne[dst]  # (E, 128)
        agg = _scatter_sum(msg, src, N)  # (N, 128)

        # Combine: concat(x, agg) → Linear → (N, 128)
        return (mx.concatenate([x, agg], axis=1)
                @ W["model.representation_model.neighbor_embedding.combine.weight"].T
                + W["model.representation_model.neighbor_embedding.combine.bias"])

    # ── attention layer ───────────────────────────────────────────────

    def _attn_layer(self, L, x, vec, src, dst, ew, ea, ev, N):
        """EquivariantMultiHeadAttention — full equivariant vector message passing.

        Args:
            L: layer index
            x: (N, 128) scalar features
            vec: (N, 3, 128) equivariant vector features
            src, dst: (E,) edge indices [src=edge_index[0], dst=edge_index[1]]
            ew: (E,) distances
            ea: (E, 64) RBF features
            ev: (E, 3) normalized edge vectors
            N: number of atoms

        Returns:
            dx: (N, 128) scalar update
            dvec: (N, 3, 128) vector update
        """
        W = self.w
        p = f"model.representation_model.attention_layers.{L}"
        E = src.shape[0]

        # 1. LayerNorm
        xn = _layer_norm(x, W[f"{p}.layernorm.weight"], W[f"{p}.layernorm.bias"])

        # 2. Q, K, V projections — reshape for multi-head
        q = (xn @ W[f"{p}.q_proj.weight"].T + W[f"{p}.q_proj.bias"]).reshape(N, N_HEADS, HEAD_DIM)
        k = (xn @ W[f"{p}.k_proj.weight"].T + W[f"{p}.k_proj.bias"]).reshape(N, N_HEADS, HEAD_DIM)
        v = (xn @ W[f"{p}.v_proj.weight"].T + W[f"{p}.v_proj.bias"]).reshape(N, N_HEADS, HEAD_DIM * 3)

        # 3. Vec projection: (N, 3, 128) → (N, 3, 384) → split into vec1, vec2, vec3
        vp = vec @ W[f"{p}.vec_proj.weight"].T                   # (N, 3, 384)
        v1, v2, v3 = mx.split(vp, 3, axis=-1)                    # each (N, 3, 128)

        # 4. Reshape vec for multi-head attention
        vec_mh = vec.reshape(N, 3, N_HEADS, HEAD_DIM)             # (N, 3, 8, 16)

        # 5. Vec dot product: scalar coupling from vectors
        vec_dot = mx.sum(v1 * v2, axis=1)                         # (N, 128)

        # 6. Distance-dependent K and V (distance_influence="both")
        dk = _silu(ea @ W[f"{p}.dk_proj.weight"].T + W[f"{p}.dk_proj.bias"]).reshape(E, N_HEADS, HEAD_DIM)
        dv = _silu(ea @ W[f"{p}.dv_proj.weight"].T + W[f"{p}.dv_proj.bias"]).reshape(E, N_HEADS, HEAD_DIM * 3)

        # 7. Gather for edges: Q from target, K/V/vec from source
        qi = q[dst]           # (E, 8, 16)
        kj = k[src]           # (E, 8, 16)
        vj = v[src]           # (E, 8, 48)
        vecj = vec_mh[src]    # (E, 3, 8, 16)

        # 8. Attention scores: dot(Q, K*dk) with SiLU activation
        attn = mx.sum(qi * kj * dk, axis=-1)                      # (E, 8)
        attn = _silu(attn)                                         # (E, 8)
        # NOTE: vector_cutoff=True in PM6-ML → cutoff on v_j, NOT on attn

        # 9. Value pathway: apply cutoff + distance-dependent scaling
        cutoff = _cosine_cutoff(ew)[:, None, None]                 # (E, 1, 1)
        vj = vj * cutoff                                           # (E, 8, 48) — cutoff on values
        vj = vj * dv                                               # (E, 8, 48)
        xm, vm1, vm2 = mx.split(vj, 3, axis=-1)                  # each (E, 8, 16)

        # 10. Scalar message: x * attention weight
        xm = xm * attn[:, :, None]                                # (E, 8, 16)

        # 11. Equivariant vector message:
        #     vec_j * vec1_value + vec2_value * edge_direction
        #     vecj: (E, 3, 8, 16), vm1: (E, 8, 16) → (E, 1, 8, 16)
        #     vm2: (E, 8, 16) → (E, 1, 8, 16), ev: (E, 3) → (E, 3, 1, 1)
        vec_m = vecj * vm1[:, None, :, :] + vm2[:, None, :, :] * ev[:, :, None, None]

        # 12. Aggregate at target nodes
        xa = _scatter_sum(xm, dst, N).reshape(N, HIDDEN)           # (N, 128)
        va = _scatter_sum(vec_m, dst, N).reshape(N, 3, HIDDEN)     # (N, 3, 128)

        # 13. Output projection: (N, 128) → (N, 384) → split into o1, o2, o3
        o = xa @ W[f"{p}.o_proj.weight"].T + W[f"{p}.o_proj.bias"]  # (N, 384)
        o1, o2, o3 = mx.split(o, 3, axis=-1)                       # each (N, 128)

        # 14. Final: scalar dx = vec_dot * o2 + o3, vector dvec = vec3 * o1 + aggregated_vec
        dx = vec_dot * o2 + o3                                      # (N, 128)
        dvec = v3 * o1[:, None, :] + va                             # (N, 3, 128)

        return dx, dvec

    # ── output model ──────────────────────────────────────────────────

    def _output(self, x, vec, N):
        """EquivariantScalar output: two GatedEquivariantBlocks.

        Block 0: (128 → 64), scalar_activation=True
        Block 1: (64 → 1), scalar_activation=False
        """
        W = self.w

        # ── Block 0: GatedEquivariantBlock(128, 64, scalar_activation=True) ──
        # vec1_proj: project vec for norm computation
        vb0 = vec @ W["model.output_model.output_network.0.vec1_proj.weight"].T  # (N, 3, 128)
        v1_norm = mx.sqrt(mx.sum(vb0 * vb0, axis=1))  # (N, 128) — L2 norm over spatial dim

        # vec2_proj: project vec for gated output
        v2_0 = vec @ W["model.output_model.output_network.0.vec2_proj.weight"].T  # (N, 3, 64)

        # update_net MLP: cat(x, vec_norm) → Linear(256,128) → SiLU → Linear(128,128)
        h = mx.concatenate([x, v1_norm], axis=-1)  # (N, 256)
        h = _silu(h @ W["model.output_model.output_network.0.update_net.0.weight"].T
                  + W["model.output_model.output_network.0.update_net.0.bias"])  # (N, 128)
        h = (h @ W["model.output_model.output_network.0.update_net.2.weight"].T
             + W["model.output_model.output_network.0.update_net.2.bias"])  # (N, 128)

        # Split into scalar and vector gate, apply gating
        xs, vg = mx.split(h, 2, axis=-1)   # each (N, 64)
        vec = vg[:, None, :] * v2_0          # (N, 3, 64) — gated vector output
        x = _silu(xs)                        # (N, 64) — scalar activation

        # ── Block 1: GatedEquivariantBlock(64, 1, scalar_activation=False) ──
        vb1 = vec @ W["model.output_model.output_network.1.vec1_proj.weight"].T  # (N, 3, 64)
        v1_norm2 = mx.sqrt(mx.sum(vb1 * vb1, axis=1))  # (N, 64)

        v2_1 = vec @ W["model.output_model.output_network.1.vec2_proj.weight"].T  # (N, 3, 1)

        h2 = mx.concatenate([x, v1_norm2], axis=-1)  # (N, 128)
        h2 = _silu(h2 @ W["model.output_model.output_network.1.update_net.0.weight"].T
                   + W["model.output_model.output_network.1.update_net.0.bias"])  # (N, 64)
        h2 = (h2 @ W["model.output_model.output_network.1.update_net.2.weight"].T
              + W["model.output_model.output_network.1.update_net.2.bias"])  # (N, 2)

        xs2, _ = mx.split(h2, 2, axis=-1)  # xs2: (N, 1), _: (N, 1) [v_gate, unused]
        # No scalar activation for block 1
        # pre_reduce returns x + v.sum()*0 — the v term is zero for inference
        return xs2  # (N, 1)

    # ── forward pass ──────────────────────────────────────────────────

    def __call__(self, atomic_numbers: list[int], coords: np.ndarray) -> float:
        """Compute PM6-ML energy correction.

        Args:
            atomic_numbers: list of Z values
            coords: (N, 3) in Angstrom

        Returns:
            energy correction in kJ/mol
        """
        W = self.w
        types = mx.array([Z_TO_ATYPE.get(z, 0) for z in atomic_numbers], dtype=mx.int32)
        pos_np = np.asarray(coords, dtype=np.float32)
        N = len(atomic_numbers)

        # 1. Build edge list (all pairs within cutoff + self-loops)
        src, dst, ew, ev, is_self = self._build_edges(pos_np, N)

        # 2. RBF expansion with cosine envelope
        ea = self._rbf(ew)  # (E, 64)

        # 3. Atom embedding
        x = W["model.representation_model.embedding.weight"][types]  # (N, 128)

        # 4. Neighbor embedding
        x = self._neighbor_embed(types, x, src, dst, ew, ea, is_self, N)

        # 5. Initialize equivariant vector features
        vec = mx.zeros((N, 3, HIDDEN))

        # 6. Equivariant attention layers (6 layers)
        for L in range(N_LAYERS):
            dx, dv = self._attn_layer(L, x, vec, src, dst, ew, ea, ev, N)
            x = x + dx     # scalar residual
            vec = vec + dv  # vector residual

        # 7. Output LayerNorm
        x = _layer_norm(x, W["model.representation_model.out_norm.weight"],
                        W["model.representation_model.out_norm.bias"])

        # 8. Output model (EquivariantScalar: GatedEquivariantBlocks)
        per_atom = self._output(x, vec, N)  # (N, 1)

        # 9. Scale by data std
        per_atom = per_atom * W["model.std"]

        # 10. Sum over atoms
        energy = mx.sum(per_atom)

        # 11. Shift by data mean
        energy = energy + W["model.mean"]

        mx.eval(energy)
        return float(energy)


# ─── convenience function ─────────────────────────────────────────────

_mlx_model = None


def pm6_ml_correction_mlx(atoms: list[int], coords: np.ndarray) -> float:
    """Compute PM6-ML correction on MLX Metal GPU (eV).

    Loads model on first call (from .npz if available, else converts .ckpt).
    """
    global _mlx_model
    if _mlx_model is None:
        ckpt = "/Users/tgg/Github/mopac-ml/models/PM6-ML_correction_seed8_best.ckpt"
        npz = ckpt.replace(".ckpt", "_mlx.npz")
        if os.path.exists(npz):
            _mlx_model = PM6ML_MLX.from_npz(npz)
        else:
            _mlx_model = PM6ML_MLX.from_checkpoint(ckpt)
            _mlx_model.save_npz(npz)
    for z in atoms:
        if z not in Z_TO_ATYPE:
            return 0.0
    energy_kj = _mlx_model(atoms, coords)
    return energy_kj / 96.485  # kJ/mol → eV
