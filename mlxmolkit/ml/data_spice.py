"""
SPICE dataset loader for charge prediction training.

Loads SPICE HDF5 file → list of GraphData objects with charge labels.

SPICE format (HDF5):
  /molecule_name/
    atomic_numbers: (n_atoms,)
    conformations: (n_conformations, n_atoms, 3) in Bohr
    formation_energy: (n_conformations,)
    dft_total_energy: (n_conformations,)
    mbis_charges: (n_conformations, n_atoms)    ← our target
    ...

Download: pip install openff-qcsubmit
Or manually from: https://github.com/openmm/spice-dataset
"""
from __future__ import annotations

import os
import numpy as np
from typing import Optional
from .graph_builder import build_graph, ELEMENT_TO_IDX
from mlx_graphs.data import GraphData


BOHR_TO_ANG = 0.529177249

# Elements we support (organic focus)
SUPPORTED_ELEMENTS = set(ELEMENT_TO_IDX.keys())


def load_spice(
    hdf5_path: str,
    max_molecules: int | None = None,
    max_conformations: int = 5,
    cutoff: float = 5.0,
    n_rbf: int = 64,
    rbf_type: str = "expnorm",
    elements_filter: set[int] | None = None,
    verbose: bool = True,
) -> list[GraphData]:
    """Load SPICE HDF5 file and convert to GraphData list.

    Args:
        hdf5_path: path to SPICE .hdf5 file
        max_molecules: limit number of molecules (None = all)
        max_conformations: max conformations per molecule
        cutoff: edge distance cutoff in Angstrom
        n_rbf: number of RBF basis functions
        rbf_type: "gaussian" or "expnorm"
        elements_filter: only include molecules with these elements
        verbose: print progress

    Returns:
        list of GraphData objects with node_labels = MBIS charges
    """
    import h5py

    if elements_filter is None:
        elements_filter = SUPPORTED_ELEMENTS

    graphs = []
    n_skipped = 0
    n_mol = 0

    with h5py.File(hdf5_path, "r") as f:
        mol_names = list(f.keys())
        if max_molecules:
            mol_names = mol_names[:max_molecules]

        for mol_name in mol_names:
            grp = f[mol_name]

            # Get atomic numbers
            if "atomic_numbers" not in grp:
                n_skipped += 1
                continue
            z = np.array(grp["atomic_numbers"], dtype=np.int32)

            # Check element support
            z_set = set(z.tolist())
            if not z_set.issubset(elements_filter):
                n_skipped += 1
                continue

            # Get conformations (coordinates in Bohr → Angstrom)
            if "conformations" in grp:
                coords_all = np.array(grp["conformations"]) * BOHR_TO_ANG
            elif "geometry" in grp:
                coords_all = np.array(grp["geometry"]) * BOHR_TO_ANG
            else:
                n_skipped += 1
                continue

            # Get charges (MBIS preferred, fallback to mulliken)
            charges_all = None
            for charge_key in ["mbis_charges", "mulliken_charges", "charges"]:
                if charge_key in grp:
                    charges_all = np.array(grp[charge_key])
                    break
            if charges_all is None:
                n_skipped += 1
                continue

            # Handle single conformation case
            if coords_all.ndim == 2:
                coords_all = coords_all[np.newaxis, :, :]
                charges_all = charges_all[np.newaxis, :]

            # Determine formal charge (sum of MBIS charges ≈ formal charge)
            formal_charge = round(float(charges_all[0].sum()))

            # Sample conformations
            n_conf = min(max_conformations, coords_all.shape[0])
            if n_conf < coords_all.shape[0]:
                # Evenly sample
                indices = np.linspace(0, coords_all.shape[0] - 1, n_conf, dtype=int)
            else:
                indices = range(n_conf)

            for ci in indices:
                coords = coords_all[ci].astype(np.float32)
                charges = charges_all[ci].astype(np.float32)

                g = build_graph(
                    z.tolist(), coords,
                    charges=charges,
                    cutoff=cutoff,
                    n_rbf=n_rbf,
                    rbf_type=rbf_type,
                    formal_charge=formal_charge,
                )
                graphs.append(g)

            n_mol += 1

            if verbose and n_mol % 500 == 0:
                print(f"  Loaded {n_mol} molecules ({len(graphs)} graphs) ...")

    if verbose:
        print(f"Loaded {n_mol} molecules → {len(graphs)} graphs "
              f"(skipped {n_skipped})")

    return graphs


def split_by_molecule(
    graphs: list[GraphData],
    n_conformations: int,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = 42,
) -> tuple[list[GraphData], list[GraphData], list[GraphData]]:
    """Split graphs by molecule identity (not conformation).

    Ensures all conformations of a molecule stay in the same split.

    Args:
        graphs: list of all GraphData
        n_conformations: max conformations per molecule (for grouping)
        train_frac, val_frac: split fractions (test = 1 - train - val)
        seed: random seed

    Returns:
        (train_graphs, val_graphs, test_graphs)
    """
    rng = np.random.RandomState(seed)

    # Group by molecule (every n_conformations consecutive graphs)
    n_groups = len(graphs) // max(1, n_conformations)
    if n_groups == 0:
        n_groups = len(graphs)
        n_conformations = 1

    indices = np.arange(n_groups)
    rng.shuffle(indices)

    n_train = int(n_groups * train_frac)
    n_val = int(n_groups * val_frac)

    train_idx = set(indices[:n_train].tolist())
    val_idx = set(indices[n_train:n_train + n_val].tolist())

    train, val, test = [], [], []
    for i, g in enumerate(graphs):
        mol_idx = i // n_conformations
        if mol_idx in train_idx:
            train.append(g)
        elif mol_idx in val_idx:
            val.append(g)
        else:
            test.append(g)

    return train, val, test


def save_preprocessed(graphs: list[GraphData], path: str):
    """Save preprocessed GraphData list for fast reload (numpy arrays)."""
    import pickle
    data = []
    for g in graphs:
        d = {}
        for k, v in vars(g).items():
            if v is not None and hasattr(v, 'shape'):
                d[k] = np.array(v)
            elif v is not None:
                d[k] = v
        data.append(d)
    with open(path, "wb") as f:
        pickle.dump(data, f, protocol=4)
    print(f"Saved {len(graphs)} graphs to {path}")


def load_preprocessed(path: str) -> list[GraphData]:
    """Load preprocessed GraphData list."""
    import pickle
    import mlx.core as mx
    with open(path, "rb") as f:
        data = pickle.load(f)

    graphs = []
    for d in data:
        kwargs = {}
        for k, v in d.items():
            if isinstance(v, np.ndarray):
                kwargs[k] = mx.array(v)
            else:
                kwargs[k] = v
        graphs.append(GraphData(**kwargs))
    print(f"Loaded {len(graphs)} graphs from {path}")
    return graphs
