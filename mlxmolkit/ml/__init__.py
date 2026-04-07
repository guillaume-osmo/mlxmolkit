"""
mlxmolkit.ml — Machine learning models on MLX Metal GPU.

SchNet-like charge prediction GNN trained directly on Apple Silicon.

Modules:
  graph_builder  — molecular geometry → graph representation
  schnet_charge  — SchNet model for per-atom charge prediction
  data_spice     — SPICE HDF5 dataset loader
  train          — MLX Metal training loop
  predict        — inference API
"""
from .graph_builder import build_graph, build_edges_np, gaussian_rbf, expnorm_rbf
from .schnet_charge import SchNetCharge, CFConvBlock
from .predict import ChargePredictor
