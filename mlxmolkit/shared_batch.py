"""
Shared-constraint batch for N molecules × k conformers.

Constraints are stored ONCE per molecule. Each conformer references
its parent molecule via ``conf_to_mol``, and accesses positions via
``conf_atom_starts``.  Constraint atom indices are LOCAL [0, n_atoms_mol).

Memory: O(N × constraints + C × atoms) where C = Σ k_i.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import mlx.core as mx

from .dg_extract import DGParams, TetrahedralCheckData, extract_tetrahedral_data
from .etk_extract import ETKParams


@dataclass
class SharedConstraintBatch:
    """N molecules × k conformers with shared constraints."""

    n_mols: int
    n_confs_total: int                     # C = Σ k_i
    n_confs_per_mol: List[int]             # [k_0, ..., k_{N-1}]
    dim: int                               # 4 (DG) or 3 (ETK)

    # ---- Per-conformer (positions differ) ----
    conf_atom_starts: np.ndarray           # (C+1,) int32 — cumulative atoms
    conf_to_mol: np.ndarray                # (C,) int32

    # ---- Per-molecule (shared constraints) ----
    mol_n_atoms: np.ndarray                # (N,) int32

    # DG distance terms — LOCAL indices [0, n_atoms_mol)
    dist_idx1: np.ndarray
    dist_idx2: np.ndarray
    dist_lb2: np.ndarray
    dist_ub2: np.ndarray
    dist_weight: np.ndarray
    dist_term_starts: np.ndarray           # (N+1,) int32

    # DG chiral terms — LOCAL indices
    chiral_idx1: np.ndarray
    chiral_idx2: np.ndarray
    chiral_idx3: np.ndarray
    chiral_idx4: np.ndarray
    chiral_vol_lower: np.ndarray
    chiral_vol_upper: np.ndarray
    chiral_term_starts: np.ndarray         # (N+1,) int32

    # Fourth dimension indices — LOCAL
    fourth_idx: np.ndarray
    fourth_term_starts: np.ndarray         # (N+1,) int32

    # ETK terms (optional, filled for 3D stage)
    etk_torsion_idx: Optional[np.ndarray] = None      # (n_tors, 4) LOCAL
    etk_torsion_V: Optional[np.ndarray] = None         # (n_tors, 6)
    etk_torsion_signs: Optional[np.ndarray] = None     # (n_tors, 6)
    etk_torsion_term_starts: Optional[np.ndarray] = None  # (N+1,)

    etk_improper_idx: Optional[np.ndarray] = None      # (n_imp, 4) LOCAL
    etk_improper_weight: Optional[np.ndarray] = None
    etk_improper_term_starts: Optional[np.ndarray] = None

    etk_dist12_idx1: Optional[np.ndarray] = None
    etk_dist12_idx2: Optional[np.ndarray] = None
    etk_dist12_lb: Optional[np.ndarray] = None
    etk_dist12_ub: Optional[np.ndarray] = None
    etk_dist12_weight: Optional[np.ndarray] = None
    etk_dist12_term_starts: Optional[np.ndarray] = None

    etk_dist13_idx1: Optional[np.ndarray] = None
    etk_dist13_idx2: Optional[np.ndarray] = None
    etk_dist13_lb: Optional[np.ndarray] = None
    etk_dist13_ub: Optional[np.ndarray] = None
    etk_dist13_weight: Optional[np.ndarray] = None
    etk_dist13_term_starts: Optional[np.ndarray] = None

    etk_dist14_idx1: Optional[np.ndarray] = None
    etk_dist14_idx2: Optional[np.ndarray] = None
    etk_dist14_lb: Optional[np.ndarray] = None
    etk_dist14_ub: Optional[np.ndarray] = None
    etk_dist14_weight: Optional[np.ndarray] = None
    etk_dist14_term_starts: Optional[np.ndarray] = None

    @property
    def n_atoms_total(self) -> int:
        return int(self.conf_atom_starts[-1])

    @property
    def total_coords(self) -> int:
        return self.n_atoms_total * self.dim


def _concat_or_empty(parts: list, dtype) -> np.ndarray:
    return np.concatenate(parts).astype(dtype) if parts else np.zeros(0, dtype=dtype)


def pack_shared_dg_batch(
    dg_params_list: List[DGParams],
    n_confs_per_mol: List[int],
    dim: int = 4,
) -> SharedConstraintBatch:
    """Pack N molecules × k conformers into a shared-constraint batch.

    DG constraints are stored once per molecule with LOCAL atom indices.
    Position arrays have space for C = Σ k_i conformers.

    Parameters
    ----------
    dg_params_list : list of DGParams
        One per molecule (from ``extract_dg_params``).
    n_confs_per_mol : list of int
        k_i conformers for each molecule.
    dim : int
        Coordinate dimension (4 for DG, 3 for ETK).
    """
    n_mols = len(dg_params_list)
    assert len(n_confs_per_mol) == n_mols
    n_confs_total = sum(n_confs_per_mol)

    mol_n_atoms = np.array([p.n_atoms for p in dg_params_list], dtype=np.int32)

    # Build conf_to_mol and conf_atom_starts
    conf_to_mol = np.empty(n_confs_total, dtype=np.int32)
    conf_atom_starts = np.zeros(n_confs_total + 1, dtype=np.int32)

    c = 0
    for m, k in enumerate(n_confs_per_mol):
        for _ in range(k):
            conf_to_mol[c] = m
            conf_atom_starts[c + 1] = conf_atom_starts[c] + mol_n_atoms[m]
            c += 1

    # Distance terms — keep LOCAL indices, concatenate across molecules
    dist_parts_i1, dist_parts_i2 = [], []
    dist_parts_lb2, dist_parts_ub2, dist_parts_w = [], [], []
    dist_term_starts = np.zeros(n_mols + 1, dtype=np.int32)

    for m, p in enumerate(dg_params_list):
        n = len(p.dist_idx1)
        dist_term_starts[m + 1] = dist_term_starts[m] + n
        if n > 0:
            dist_parts_i1.append(p.dist_idx1)     # LOCAL — no offset!
            dist_parts_i2.append(p.dist_idx2)
            dist_parts_lb2.append(p.dist_lb2)
            dist_parts_ub2.append(p.dist_ub2)
            dist_parts_w.append(p.dist_weight)

    # Chiral terms — LOCAL indices
    ch1, ch2, ch3, ch4, ch_lo, ch_hi = [], [], [], [], [], []
    chiral_term_starts = np.zeros(n_mols + 1, dtype=np.int32)

    for m, p in enumerate(dg_params_list):
        n = len(p.chiral_idx1)
        chiral_term_starts[m + 1] = chiral_term_starts[m] + n
        if n > 0:
            ch1.append(p.chiral_idx1)
            ch2.append(p.chiral_idx2)
            ch3.append(p.chiral_idx3)
            ch4.append(p.chiral_idx4)
            ch_lo.append(p.chiral_vol_lower)
            ch_hi.append(p.chiral_vol_upper)

    # Fourth dim — LOCAL indices
    f_parts = []
    fourth_term_starts = np.zeros(n_mols + 1, dtype=np.int32)
    for m, p in enumerate(dg_params_list):
        n = len(p.fourth_idx)
        fourth_term_starts[m + 1] = fourth_term_starts[m] + n
        if n > 0:
            f_parts.append(p.fourth_idx)   # LOCAL

    return SharedConstraintBatch(
        n_mols=n_mols,
        n_confs_total=n_confs_total,
        n_confs_per_mol=list(n_confs_per_mol),
        dim=dim,
        conf_atom_starts=conf_atom_starts,
        conf_to_mol=conf_to_mol,
        mol_n_atoms=mol_n_atoms,
        dist_idx1=_concat_or_empty(dist_parts_i1, np.int32),
        dist_idx2=_concat_or_empty(dist_parts_i2, np.int32),
        dist_lb2=_concat_or_empty(dist_parts_lb2, np.float32),
        dist_ub2=_concat_or_empty(dist_parts_ub2, np.float32),
        dist_weight=_concat_or_empty(dist_parts_w, np.float32),
        dist_term_starts=dist_term_starts,
        chiral_idx1=_concat_or_empty(ch1, np.int32),
        chiral_idx2=_concat_or_empty(ch2, np.int32),
        chiral_idx3=_concat_or_empty(ch3, np.int32),
        chiral_idx4=_concat_or_empty(ch4, np.int32),
        chiral_vol_lower=_concat_or_empty(ch_lo, np.float32),
        chiral_vol_upper=_concat_or_empty(ch_hi, np.float32),
        chiral_term_starts=chiral_term_starts,
        fourth_idx=_concat_or_empty(f_parts, np.int32),
        fourth_term_starts=fourth_term_starts,
    )


def add_etk_to_batch(
    batch: SharedConstraintBatch,
    etk_params_list: List[ETKParams],
) -> None:
    """Add ETK (3D torsion) terms to an existing shared batch.

    Mutates *batch* in place, setting the etk_* fields.
    """
    n_mols = batch.n_mols
    assert len(etk_params_list) == n_mols

    tor_idx, tor_V, tor_signs = [], [], []
    tor_starts = np.zeros(n_mols + 1, dtype=np.int32)

    imp_idx, imp_w = [], []
    imp_starts = np.zeros(n_mols + 1, dtype=np.int32)

    d12_i1, d12_i2, d12_lb, d12_ub, d12_w = [], [], [], [], []
    d12_starts = np.zeros(n_mols + 1, dtype=np.int32)

    d13_i1, d13_i2, d13_lb, d13_ub, d13_w = [], [], [], [], []
    d13_starts = np.zeros(n_mols + 1, dtype=np.int32)

    d14_i1, d14_i2, d14_lb, d14_ub, d14_w = [], [], [], [], []
    d14_starts = np.zeros(n_mols + 1, dtype=np.int32)

    for m, p in enumerate(etk_params_list):
        # Torsions
        n_t = len(p.torsion_idx) if p.torsion_idx is not None else 0
        tor_starts[m + 1] = tor_starts[m] + n_t
        if n_t > 0:
            tor_idx.append(p.torsion_idx)
            tor_V.append(p.torsion_V)
            tor_signs.append(p.torsion_signs)

        # Improper torsions
        n_i = len(p.improper_idx) if p.improper_idx is not None else 0
        imp_starts[m + 1] = imp_starts[m] + n_i
        if n_i > 0:
            imp_idx.append(p.improper_idx)
            imp_w.append(p.improper_weight)

        # 1-2 bonds
        n_d12 = len(p.dist12_idx1) if hasattr(p, 'dist12_idx1') and p.dist12_idx1 is not None else 0
        d12_starts[m + 1] = d12_starts[m] + n_d12
        if n_d12 > 0:
            d12_i1.append(p.dist12_idx1); d12_i2.append(p.dist12_idx2)
            d12_lb.append(p.dist12_lb); d12_ub.append(p.dist12_ub); d12_w.append(p.dist12_weight)

        # 1-3 angles
        n_d13 = len(p.dist13_idx1) if hasattr(p, 'dist13_idx1') and p.dist13_idx1 is not None else 0
        d13_starts[m + 1] = d13_starts[m] + n_d13
        if n_d13 > 0:
            d13_i1.append(p.dist13_idx1); d13_i2.append(p.dist13_idx2)
            d13_lb.append(p.dist13_lb); d13_ub.append(p.dist13_ub); d13_w.append(p.dist13_weight)

        # 1-4 distance constraints
        n_d = len(p.dist14_idx1) if p.dist14_idx1 is not None else 0
        d14_starts[m + 1] = d14_starts[m] + n_d
        if n_d > 0:
            d14_i1.append(p.dist14_idx1); d14_i2.append(p.dist14_idx2)
            d14_lb.append(p.dist14_lb); d14_ub.append(p.dist14_ub); d14_w.append(p.dist14_weight)

    batch.etk_torsion_idx = np.concatenate(tor_idx).astype(np.int32) if tor_idx else np.zeros((0, 4), dtype=np.int32)
    batch.etk_torsion_V = np.concatenate(tor_V).astype(np.float32) if tor_V else np.zeros((0, 6), dtype=np.float32)
    batch.etk_torsion_signs = np.concatenate(tor_signs).astype(np.float32) if tor_signs else np.zeros((0, 6), dtype=np.float32)
    batch.etk_torsion_term_starts = tor_starts

    batch.etk_improper_idx = np.concatenate(imp_idx).astype(np.int32) if imp_idx else np.zeros((0, 4), dtype=np.int32)
    batch.etk_improper_weight = np.concatenate(imp_w).astype(np.float32) if imp_w else np.zeros(0, dtype=np.float32)
    batch.etk_improper_term_starts = imp_starts

    batch.etk_dist12_idx1 = _concat_or_empty(d12_i1, np.int32)
    batch.etk_dist12_idx2 = _concat_or_empty(d12_i2, np.int32)
    batch.etk_dist12_lb = _concat_or_empty(d12_lb, np.float32)
    batch.etk_dist12_ub = _concat_or_empty(d12_ub, np.float32)
    batch.etk_dist12_weight = _concat_or_empty(d12_w, np.float32)
    batch.etk_dist12_term_starts = d12_starts

    batch.etk_dist13_idx1 = _concat_or_empty(d13_i1, np.int32)
    batch.etk_dist13_idx2 = _concat_or_empty(d13_i2, np.int32)
    batch.etk_dist13_lb = _concat_or_empty(d13_lb, np.float32)
    batch.etk_dist13_ub = _concat_or_empty(d13_ub, np.float32)
    batch.etk_dist13_weight = _concat_or_empty(d13_w, np.float32)
    batch.etk_dist13_term_starts = d13_starts

    batch.etk_dist14_idx1 = _concat_or_empty(d14_i1, np.int32)
    batch.etk_dist14_idx2 = _concat_or_empty(d14_i2, np.int32)
    batch.etk_dist14_lb = _concat_or_empty(d14_lb, np.float32)
    batch.etk_dist14_ub = _concat_or_empty(d14_ub, np.float32)
    batch.etk_dist14_weight = _concat_or_empty(d14_w, np.float32)
    batch.etk_dist14_term_starts = d14_starts


_ETK_TERM_FIELDS = {
    # name: (params attribute names, batch attribute names)
    "torsion": (("torsion_idx", "torsion_V", "torsion_signs"),
                ("etk_torsion_idx", "etk_torsion_V", "etk_torsion_signs")),
    "improper": (("improper_idx", "improper_weight"),
                 ("etk_improper_idx", "etk_improper_weight")),
    "dist12": (("dist12_idx1", "dist12_idx2", "dist12_lb", "dist12_ub", "dist12_weight"),
               ("etk_dist12_idx1", "etk_dist12_idx2", "etk_dist12_lb", "etk_dist12_ub",
                "etk_dist12_weight")),
    "dist13": (("dist13_idx1", "dist13_idx2", "dist13_lb", "dist13_ub", "dist13_weight"),
               ("etk_dist13_idx1", "etk_dist13_idx2", "etk_dist13_lb", "etk_dist13_ub",
                "etk_dist13_weight")),
    "dist14": (("dist14_idx1", "dist14_idx2", "dist14_lb", "dist14_ub", "dist14_weight"),
               ("etk_dist14_idx1", "etk_dist14_idx2", "etk_dist14_lb", "etk_dist14_ub",
                "etk_dist14_weight")),
}
_ETK_EMPTY_SHAPES = {
    "torsion_idx": (0, 4), "torsion_V": (0, 6), "torsion_signs": (0, 6),
    "improper_idx": (0, 4),
}


def expand_terms_per_conformer(
    term_starts: np.ndarray, conf_mol: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gather map that replicates per-molecule term blocks once per conformer.

    ``term_starts`` is the (n_mols+1,) CSR offset of a per-molecule term table;
    ``conf_mol`` gives each conformer's molecule. Returns ``(conf_of_term,
    src_term, conf_term_starts)``: for every replicated term, the conformer it
    belongs to and the row of the per-molecule table it copies, plus the new
    (C+1,) CSR offsets with one block per conformer.
    """
    conf_mol = np.asarray(conf_mol, dtype=np.int64)
    counts = (term_starts[conf_mol + 1] - term_starts[conf_mol]).astype(np.int64)
    conf_term_starts = np.zeros(len(conf_mol) + 1, dtype=np.int64)
    np.cumsum(counts, out=conf_term_starts[1:])
    total = int(conf_term_starts[-1])
    conf_of_term = np.repeat(np.arange(len(conf_mol), dtype=np.int64), counts)
    src_term = (np.arange(total, dtype=np.int64)
                - np.repeat(conf_term_starts[:-1], counts)
                + np.repeat(term_starts[conf_mol].astype(np.int64), counts))
    return conf_of_term, src_term, conf_term_starts


def concat_etk_params(etk_params_list: List[ETKParams]) -> dict:
    """Concatenate per-molecule ETK tables once (LOCAL atom indices).

    Returns ``{term: (starts, {param_attr: array})}`` for every ETK term type,
    the input of :func:`pack_per_conformer_etk_batch`.
    """
    out = {}
    for term, (attrs, _) in _ETK_TERM_FIELDS.items():
        starts = np.zeros(len(etk_params_list) + 1, dtype=np.int64)
        parts = {a: [] for a in attrs}
        for m, p in enumerate(etk_params_list):
            first = getattr(p, attrs[0], None)
            n = 0 if first is None else len(first)
            starts[m + 1] = starts[m] + n
            if n:
                for a in attrs:
                    parts[a].append(np.asarray(getattr(p, a)))
        arrays = {}
        for a in attrs:
            if parts[a]:
                arrays[a] = np.concatenate(parts[a])
            else:
                arrays[a] = np.zeros(_ETK_EMPTY_SHAPES.get(a, (0,)))
        out[term] = (starts, arrays)
    return out


def pack_per_conformer_etk_batch(
    etk_concat: dict,
    mol_n_atoms: np.ndarray,
    conf_mol: np.ndarray,
    positions_3d: np.ndarray,
) -> SharedConstraintBatch:
    """ETK batch in which every conformer is its own constraint set.

    RDKit's ETK stage (``construct3DForceField``) restrains each 1-2 and 1-3
    distance to the value it has in THAT conformer's DG output, +/- 0.01 A.
    Those reference values therefore differ per conformer, which a table shared
    by all conformers of a molecule cannot express. This batch holds one
    "molecule" per conformer (``conf_to_mol = arange(C)``), with the 1-2 and
    1-3 windows re-centred on the conformer's own distances and every other ETK
    term copied from its molecule. The ETK kernel reads constraints only
    through ``conf_to_mol``, so it runs unchanged on this layout.

    Args:
        etk_concat: output of :func:`concat_etk_params` over all molecules.
        mol_n_atoms: (N,) atoms per molecule (with hydrogens).
        conf_mol: (C,) molecule index of each conformer, contiguous per molecule
            or not, in the order the conformers appear in ``positions_3d``.
        positions_3d: flat (sum of conformer atoms * 3,) float32 coordinates.

    Returns:
        A SharedConstraintBatch with dim 3, empty DG terms and per-conformer
        ETK terms.
    """
    conf_mol = np.asarray(conf_mol, dtype=np.int64)
    n_confs = len(conf_mol)
    conf_n_atoms = np.asarray(mol_n_atoms, dtype=np.int32)[conf_mol]
    conf_atom_starts = np.zeros(n_confs + 1, dtype=np.int32)
    np.cumsum(conf_n_atoms, out=conf_atom_starts[1:])
    pos = np.asarray(positions_3d, dtype=np.float32).reshape(-1, 3)
    zero_starts = np.zeros(n_confs + 1, dtype=np.int32)
    empty_i = np.zeros(0, dtype=np.int32)
    empty_f = np.zeros(0, dtype=np.float32)
    batch = SharedConstraintBatch(
        n_mols=n_confs, n_confs_total=n_confs, n_confs_per_mol=[1] * n_confs, dim=3,
        conf_atom_starts=conf_atom_starts,
        conf_to_mol=np.arange(n_confs, dtype=np.int32),
        mol_n_atoms=conf_n_atoms,
        dist_idx1=empty_i, dist_idx2=empty_i, dist_lb2=empty_f, dist_ub2=empty_f,
        dist_weight=empty_f, dist_term_starts=zero_starts,
        chiral_idx1=empty_i, chiral_idx2=empty_i, chiral_idx3=empty_i, chiral_idx4=empty_i,
        chiral_vol_lower=empty_f, chiral_vol_upper=empty_f, chiral_term_starts=zero_starts,
        fourth_idx=empty_i, fourth_term_starts=zero_starts,
    )
    for term, (attrs, batch_attrs) in _ETK_TERM_FIELDS.items():
        starts, arrays = etk_concat[term]
        conf_of_term, src, conf_starts = expand_terms_per_conformer(starts, conf_mol)
        gathered = {a: arrays[a][src] for a in attrs}
        if term in ("dist12", "dist13"):
            # Re-centre the window on this conformer's own distance, keeping
            # the molecule's half-width (RDKit: d - 0.01 .. d + 0.01).
            off = conf_atom_starts[conf_of_term]
            i1 = gathered[attrs[0]].astype(np.int64) + off
            i2 = gathered[attrs[1]].astype(np.int64) + off
            measured = np.linalg.norm(pos[i1] - pos[i2], axis=1)
            half = (gathered[attrs[3]] - gathered[attrs[2]]) / 2.0
            gathered[attrs[2]] = measured - half
            gathered[attrs[3]] = measured + half
        for a, ba in zip(attrs, batch_attrs):
            arr = gathered[a]
            if a.endswith("idx") or a.endswith("idx1") or a.endswith("idx2"):
                arr = arr.astype(np.int32)
            else:
                arr = arr.astype(np.float32)
            setattr(batch, ba, arr)
        setattr(batch, f"etk_{term}_term_starts", conf_starts.astype(np.int32))
    return batch


def init_random_positions(
    batch: SharedConstraintBatch,
    seed: int = 42,
) -> np.ndarray:
    """Generate random initial positions for all conformers.

    Each conformer gets independent Gaussian random coordinates.
    Returns flat array of shape (n_atoms_total × dim,).
    """
    rng = np.random.default_rng(seed)
    n_total = int(batch.conf_atom_starts[-1])
    return rng.standard_normal(n_total * batch.dim).astype(np.float32)
