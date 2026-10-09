"""
Extract ETKDG torsion parameters from RDKit molecules.

Extracts:
  - CSD experimental torsion preferences (6-term Fourier)
  - Improper torsion terms (planarity at sp2 centers)
  - 1-2 and 1-3 distance constraints (per-conformer reference lengths) and
    long-range distance constraints (bounds matrix)

These parameters are used in stage 5 of the ETKDG pipeline, where 3D
coordinates are refined after 4D→3D collapse to match torsional
preferences from the Cambridge Structural Database (CSD).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDistGeom


ETKDG_VARIANTS = {
    # Mirrors the flags returned by RDKit's rdDistGeom factories.
    # tuple order:
    # use_exp_torsion, use_basic_knowledge, use_small_ring_torsions,
    # use_macrocycle_torsions, use_macrocycle14config, et_version
    "DG":        (False, False, False, False, False, 1),
    "KDG":       (False, True,  False, False, False, 1),
    "ETDG":      (True,  False, False, False, False, 1),
    "ETDGv2":    (True,  False, False, False, False, 2),
    "ETKDG":     (True,  True,  False, False, False, 1),
    "ETKDGv2":   (True,  True,  False, False, False, 2),
    "ETKDGv3":   (True,  True,  False, True,  True,  2),
    "srETKDGv3": (True,  True,  True,  False, False, 2),
    # Not an RDKit variant. RDKit's table makes small-ring and macrocycle
    # knowledge mutually exclusive — srETKDGv3 turns the macrocycle terms off,
    # ETKDGv3 turns the small-ring terms off — although the two act on
    # different molecules and never compete. This is their union.
    #
    # Measured over 400 molecules (100 each acyclic / single-ring / fused /
    # macrocyclic) from the 12k ePOM subset, scored with RDKit's own MMFF94:
    # it has the fewest catastrophic failures of any setting tried, 8 molecules
    # above 5 kcal/mol against 12 for ETKDGv3 and 15 for RDKit's own ETKDGv3.
    # Typical-case differences are within noise — see
    # tools/bench_etkdg_variants.py.
    "ETKDGv3sr": (True,  True,  True,  True,  True,  2),
}


@dataclass
class ETKParams:
    """ETK torsion parameters for a single molecule."""
    n_atoms: int

    # CSD torsion terms: E = Σ V_k * (1 + sign_k * cos(k * φ)) / 2
    torsion_idx: np.ndarray      # (n_torsions, 4) int32 — i,j,k,l atoms
    torsion_V: np.ndarray        # (n_torsions, 6) float32 — Fourier coefficients
    torsion_signs: np.ndarray    # (n_torsions, 6) int32 — sign multipliers

    # Impropers, UFF inversion: E = w * (1 - sin Y), Y the angle of J->L to plane (I, J, K)
    improper_idx: np.ndarray     # (n_improper, 4) int32 — I, J (sp2 centre), K, L
    improper_weight: np.ndarray  # (n_improper,) float32

    # 1-2 distance constraints (bonds): flat-bottom harmonic
    dist12_idx1: np.ndarray      # (n_dist12,) int32
    dist12_idx2: np.ndarray      # (n_dist12,) int32
    dist12_lb: np.ndarray        # (n_dist12,) float32
    dist12_ub: np.ndarray        # (n_dist12,) float32
    dist12_weight: np.ndarray    # (n_dist12,) float32

    # 1-3 distance constraints (angles): flat-bottom harmonic
    dist13_idx1: np.ndarray      # (n_dist13,) int32
    dist13_idx2: np.ndarray      # (n_dist13,) int32
    dist13_lb: np.ndarray        # (n_dist13,) float32
    dist13_ub: np.ndarray        # (n_dist13,) float32
    dist13_weight: np.ndarray    # (n_dist13,) float32

    # Fixed-window distance restraints, E = 0.5 * w * (d - bound)² outside [lb, ub]: every pair
    # that is not bonded, not a bond angle's ends and not a torsion's ends (RDKit's long-range
    # terms). The field name is historical.
    dist14_idx1: np.ndarray      # (n_dist14,) int32
    dist14_idx2: np.ndarray      # (n_dist14,) int32
    dist14_lb: np.ndarray        # (n_dist14,) float32 — lower bound distance
    dist14_ub: np.ndarray        # (n_dist14,) float32 — upper bound distance
    dist14_weight: np.ndarray    # (n_dist14,) float32

    # Angle constraints, E = w * (theta_deg - bound)² outside [min, max]: RDKit's 179-180 deg
    # term in place of the 1-3 distance at a linear centre (a triple bond, or two double bonds
    # at a degree-2 atom), with basic knowledge only.
    angle_idx: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.int32))
    angle_min: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    angle_max: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    angle_weight: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))


@dataclass
class BatchedETKSystem:
    """Batched ETK parameters for N molecules, ready for Metal kernel."""
    n_mols: int
    n_atoms_total: int

    atom_starts: np.ndarray  # (n_mols+1,) int32

    # CSD torsion terms (global atom indices)
    torsion_idx: np.ndarray      # (n_torsions_total, 4) int32
    torsion_V: np.ndarray        # (n_torsions_total, 6) float32
    torsion_signs: np.ndarray    # (n_torsions_total, 6) int32
    torsion_term_starts: np.ndarray  # (n_mols+1,) int32

    # Improper torsion terms (global atom indices)
    improper_idx: np.ndarray         # (n_improper_total, 4) int32
    improper_weight: np.ndarray      # (n_improper_total,) float32
    improper_term_starts: np.ndarray # (n_mols+1,) int32

    # 1-2 distance constraints (global atom indices)
    dist12_idx1: np.ndarray
    dist12_idx2: np.ndarray
    dist12_lb: np.ndarray
    dist12_ub: np.ndarray
    dist12_weight: np.ndarray
    dist12_term_starts: np.ndarray   # (n_mols+1,) int32

    # 1-3 distance constraints (global atom indices)
    dist13_idx1: np.ndarray
    dist13_idx2: np.ndarray
    dist13_lb: np.ndarray
    dist13_ub: np.ndarray
    dist13_weight: np.ndarray
    dist13_term_starts: np.ndarray   # (n_mols+1,) int32

    # Fixed-window distance restraints (global atom indices)
    dist14_idx1: np.ndarray
    dist14_idx2: np.ndarray
    dist14_lb: np.ndarray
    dist14_ub: np.ndarray
    dist14_weight: np.ndarray
    dist14_term_starts: np.ndarray   # (n_mols+1,) int32


def extract_etk_params(
    mol: Chem.Mol,
    bounds_mat: np.ndarray,
    improper_weight: float = 10.0,
    *,
    use_exp_torsion: bool = True,
    use_basic_knowledge: bool = True,
    long_range_weight: float = 10.0,
    ring_planarity_fc: float = 100.0,
    use_small_ring_torsions: bool = False,
    use_macrocycle_torsions: bool = True,
    use_macrocycle14config: bool = False,
    et_version: int = 2,
    variant: str | None = None,
) -> ETKParams:
    """
    Extract ETKDG parameters from an RDKit molecule.

    Supports all ETKDG variants via the ``variant`` shortcut or individual flags:

    ========== ============ ================ ========== ========== ============ ==========
    variant    exp_torsion  basic_knowledge  small_ring macrocycle macrocycle14 et_version
    ========== ============ ================ ========== ========== ============ ==========
    DG         False        False            False      False      False        —
    KDG        False        True             False      False      False        1
    ETDG       True         False            False      False      False        1
    ETDGv2     True         False            False      False      False        2
    ETKDG      True         True             False      False      False        1
    ETKDGv2    True         True             False      False      False        2
    ETKDGv3    True         True             False      True       True         2
    srETKDGv3  True         True             True       False      False        2
    ETKDGv3sr  True         True             True       True       True         2
    ========== ============ ================ ========== ========== ============ ==========

    ``ETKDGv3sr`` is not one of RDKit's — it is the union of ETKDGv3 and
    srETKDGv3, which RDKit offers only as alternatives.
    """
    # Variant shortcut
    torsion_embed_params = None
    if variant is not None:
        if variant not in ETKDG_VARIANTS:
            raise ValueError(f"Unknown variant '{variant}'. Choose from: {list(ETKDG_VARIANTS)}")
        (
            use_exp_torsion,
            use_basic_knowledge,
            use_small_ring_torsions,
            use_macrocycle_torsions,
            use_macrocycle14config,
            et_version,
        ) = ETKDG_VARIANTS[variant]
        if variant != "DG":
            factory = getattr(rdDistGeom, variant, None)
            if factory is not None:
                torsion_embed_params = factory()

    n_atoms = mol.GetNumAtoms()

    # --- CSD experimental torsion preferences ---
    torsion_idx_list = []
    torsion_V_list = []
    torsion_signs_list = []

    if use_exp_torsion or use_basic_knowledge:
        try:
            if torsion_embed_params is not None:
                exp_torsions = rdDistGeom.GetExperimentalTorsions(mol, torsion_embed_params)
            else:
                try:
                    exp_torsions = rdDistGeom.GetExperimentalTorsions(
                        mol,
                        useExpTorsionAnglePrefs=use_exp_torsion,
                        useSmallRingTorsions=use_small_ring_torsions,
                        useMacrocycleTorsions=use_macrocycle_torsions,
                        useBasicKnowledge=use_basic_knowledge,
                        ETversion=et_version,
                    )
                except TypeError:
                    # Older RDKit: no variant params
                    exp_torsions = rdDistGeom.GetExperimentalTorsions(mol)

            for t in exp_torsions:
                atoms = list(t["atomIndices"])
                V = list(t["V"])
                signs = list(t["signs"])
                if len(atoms) == 4 and len(V) == 6 and len(signs) == 6:
                    torsion_idx_list.append(atoms)
                    torsion_V_list.append(V)
                    torsion_signs_list.append(signs)
        except Exception:
            pass

    # Flat-ring planarity torsions. RDKit adds these in getExperimentalTorsions (to the
    # CrystalFFDetails, but not to the list Python's GetExperimentalTorsions returns): for 4
    # consecutive SP2 atoms a-b-c-d of a 4- to 6-membered ring, V*(1 - cos2φ) drives the
    # endocyclic dihedral to planar (φ=0), once per central bond (b,c), and not on a bond a CSD
    # torsion already uses (RDKit's doneBonds). Reuses the torsion energy/gradient (V at index 1,
    # sign -1).
    if use_basic_knowledge:
        ring_fc = ring_planarity_fc
        ri = mol.GetRingInfo()
        seen_ring_bonds = {(min(q[1], q[2]), max(q[1], q[2])) for q in torsion_idx_list}
        for ring in ri.AtomRings():
            nring = len(ring)
            if nring < 4 or nring > 6:
                continue
            for k in range(nring):
                a, b, c, d = ring[k], ring[(k + 1) % nring], ring[(k + 2) % nring], ring[(k + 3) % nring]
                if any(mol.GetAtomWithIdx(x).GetHybridization() != Chem.HybridizationType.SP2
                       for x in (a, b, c, d)):
                    continue
                key = (min(b, c), max(b, c))
                if key in seen_ring_bonds:
                    continue
                seen_ring_bonds.add(key)
                torsion_idx_list.append([a, b, c, d])
                torsion_V_list.append([0.0, ring_fc, 0.0, 0.0, 0.0, 0.0])
                torsion_signs_list.append([0, -1, 0, 0, 0, 0])

    n_torsions = len(torsion_idx_list)
    if n_torsions > 0:
        torsion_idx = np.array(torsion_idx_list, dtype=np.int32)
        torsion_V = np.array(torsion_V_list, dtype=np.float32)
        torsion_signs = np.array(torsion_signs_list, dtype=np.int32)
    else:
        torsion_idx = np.zeros((0, 4), dtype=np.int32)
        torsion_V = np.zeros((0, 6), dtype=np.float32)
        torsion_signs = np.zeros((0, 6), dtype=np.int32)

    # --- Impropers: RDKit's UFF inversion terms at sp2 centres ---
    # Only with basic knowledge (ETKDG/KDG, not ETDG/DG). TorsionPreferences.cpp takes C/N/O
    # atoms that are SP2 with exactly three neighbours; addImproperTorsionTerms gives each
    # three UFF::InversionContrib terms (I, J = centre, K, L) over the neighbours n0, n1, n2 in
    # RDKit's order -- (n0, n1, n2), (n0, n2, n1), (n1, n2, n0) -- each with force constant
    # oobForceScalingFactor (improper_weight, 10) * K_UFF / 3, K_UFF = 50 for a carbon bound
    # to an SP2 oxygen (isBoundToSP2O, any degree), else 6.
    improper_idx_list = []
    improper_w_list = []
    if use_basic_knowledge:
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() not in (6, 7, 8):
                continue
            if atom.GetHybridization() != Chem.HybridizationType.SP2:
                continue
            neighbors = [n.GetIdx() for n in atom.GetNeighbors()]
            if len(neighbors) != 3:
                continue
            bound_to_sp2_o = atom.GetAtomicNum() == 6 and any(
                nb.GetAtomicNum() == 8 and nb.GetHybridization() == Chem.HybridizationType.SP2
                for nb in atom.GetNeighbors())
            w = improper_weight * (50.0 if bound_to_sp2_o else 6.0) / 3.0
            center = atom.GetIdx()
            n0, n1, n2 = neighbors
            for (i, k, l) in ((n0, n1, n2), (n0, n2, n1), (n1, n2, n0)):
                improper_idx_list.append([i, center, k, l])
                improper_w_list.append(w)

    n_improper = len(improper_idx_list)
    if n_improper > 0:
        imp_idx = np.array(improper_idx_list, dtype=np.int32)
        imp_w = np.array(improper_w_list, dtype=np.float32)
    else:
        imp_idx = np.zeros((0, 4), dtype=np.int32)
        imp_w = np.zeros(0, dtype=np.float32)

    # --- Distance restraints, RDKit's construct3DForceField / constructPlain3DForceField ---
    # RDKit tracks the pairs it restrains (atomPairs): the end atoms of every torsion term,
    # every bond, every bond angle. Each of those has its own term (or, for a torsion's end
    # atoms, none); every other pair is held to its bounds-matrix window. No ETK stage runs
    # without experimental torsions or basic knowledge (plain DG).
    run_field = use_exp_torsion or use_basic_knowledge
    restrained = {(min(q[0], q[3]), max(q[0], q[3])) for q in torsion_idx_list}

    # 1-2 (every bond) and 1-3 (every bond angle): RDKit restrains them to the length they
    # have in the conformer entering ETK, +/- KNOWN_DIST_TOL, with KNOWN_DIST_FORCE_CONSTANT --
    # except an angle whose centre carries an improper, held to its bounds-matrix window.
    # The window stored here is centred on the bounds-matrix midpoint and re-centred on each
    # conformer's own distance by pack_per_conformer_etk_batch.
    KNOWN_DIST_TOL = 0.01  # A
    KNOWN_DIST_FC = 100.0
    d12_i1, d12_i2, d12_lb, d12_ub, d12_w = [], [], [], [], []
    d13_i1, d13_i2, d13_lb, d13_ub, d13_w = [], [], [], [], []
    # Fixed-window terms (the dist14 family): long-range pairs, and the 1-3 pairs below.
    unique_i1, unique_i2, unique_lb, unique_ub, unique_w = [], [], [], [], []
    ang_idx, ang_min, ang_max, ang_w = [], [], [], []
    ANGLE_FC = 1.0  # RDKit: angleContribs->addContrib(i, j, k, 179.0, 180.0, 1)
    improper_centres = {q[1] for q in improper_idx_list}
    if run_field:
        for bond in mol.GetBonds():
            a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            lo, hi = min(a, b), max(a, b)
            mid = (bounds_mat[hi, lo] + bounds_mat[lo, hi]) / 2.0
            d12_i1.append(a); d12_i2.append(b)
            d12_lb.append(mid - KNOWN_DIST_TOL); d12_ub.append(mid + KNOWN_DIST_TOL)
            d12_w.append(KNOWN_DIST_FC)
            restrained.add((lo, hi))
        for atom in mol.GetAtoms():
            neighbors = sorted([n.GetIdx() for n in atom.GetNeighbors()])
            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    a, b = neighbors[i], neighbors[j]
                    restrained.add((a, b))
                    if use_basic_knowledge and _is_linear_angle(mol, a, atom.GetIdx(), b):
                        # A linear centre: an angle constraint, not a distance (add13Terms).
                        ang_idx.append([a, atom.GetIdx(), b])
                        ang_min.append(179.0); ang_max.append(180.0); ang_w.append(ANGLE_FC)
                        continue
                    if atom.GetIdx() in improper_centres:
                        # An angle at an improper centre keeps its bounds-matrix window (add13Terms).
                        unique_i1.append(a); unique_i2.append(b)
                        unique_lb.append(bounds_mat[b, a]); unique_ub.append(bounds_mat[a, b])
                        unique_w.append(KNOWN_DIST_FC)
                        continue
                    mid = (bounds_mat[b, a] + bounds_mat[a, b]) / 2.0
                    d13_i1.append(a); d13_i2.append(b)
                    d13_lb.append(mid - KNOWN_DIST_TOL); d13_ub.append(mid + KNOWN_DIST_TOL)
                    d13_w.append(KNOWN_DIST_FC)
    n_d12 = len(d12_i1)
    n_d13 = len(d13_i1)

    # Every other pair -- 1-4 pairs that are not a torsion's end atoms included -- at its fixed
    # bounds-matrix window, force constant 10 * boundsMatForceScaling
    # (addLongRangeDistanceConstraints).
    if run_field:
        na = mol.GetNumAtoms()
        for a in range(na):
            for d in range(a + 1, na):
                if (a, d) in restrained:
                    continue
                unique_i1.append(a); unique_i2.append(d)
                unique_lb.append(bounds_mat[d, a]); unique_ub.append(bounds_mat[a, d])
                unique_w.append(long_range_weight)
    n_d14 = len(unique_i1)

    def _a(lst, dt=np.int32):
        return np.array(lst, dtype=dt) if lst else np.zeros(0, dtype=dt)

    return ETKParams(
        n_atoms=n_atoms,
        torsion_idx=torsion_idx,
        torsion_V=torsion_V,
        torsion_signs=torsion_signs,
        improper_idx=imp_idx,
        improper_weight=imp_w,
        dist12_idx1=_a(d12_i1), dist12_idx2=_a(d12_i2),
        dist12_lb=_a(d12_lb, np.float32), dist12_ub=_a(d12_ub, np.float32),
        dist12_weight=_a(d12_w, np.float32),
        dist13_idx1=_a(d13_i1), dist13_idx2=_a(d13_i2),
        dist13_lb=_a(d13_lb, np.float32), dist13_ub=_a(d13_ub, np.float32),
        dist13_weight=_a(d13_w, np.float32),
        dist14_idx1=_a(unique_i1), dist14_idx2=_a(unique_i2),
        dist14_lb=_a(unique_lb, np.float32), dist14_ub=_a(unique_ub, np.float32),
        dist14_weight=_a(unique_w, np.float32),
        angle_idx=(np.array(ang_idx, dtype=np.int32) if ang_idx
                   else np.zeros((0, 3), dtype=np.int32)),
        angle_min=_a(ang_min, np.float32), angle_max=_a(ang_max, np.float32),
        angle_weight=_a(ang_w, np.float32),
    )


def _is_linear_angle(mol: Chem.Mol, a: int, centre: int, b: int) -> bool:
    """RDKit's collectBondsAndAngles flag: either bond triple, or both double at a degree-2 atom."""
    t1 = mol.GetBondBetweenAtoms(a, centre).GetBondType()
    t2 = mol.GetBondBetweenAtoms(centre, b).GetBondType()
    if t1 == Chem.BondType.TRIPLE or t2 == Chem.BondType.TRIPLE:
        return True
    return (t1 == Chem.BondType.DOUBLE and t2 == Chem.BondType.DOUBLE
            and mol.GetAtomWithIdx(centre).GetDegree() == 2)


def batch_etk_params(
    params_list: list[ETKParams],
    atom_starts: np.ndarray,
) -> BatchedETKSystem:
    """Batch per-molecule ETK params into CSR arrays with global atom indices."""
    n_mols = len(params_list)
    n_atoms_total = int(atom_starts[-1])

    def _concat_or_empty(parts, dtype, shape_suffix=None):
        if parts:
            return np.concatenate(parts).astype(dtype)
        if shape_suffix:
            return np.zeros((0,) + shape_suffix, dtype=dtype)
        return np.zeros(0, dtype=dtype)

    # --- CSD torsion terms ---
    tor_idx_parts, tor_V_parts, tor_signs_parts = [], [], []
    torsion_term_starts = np.zeros(n_mols + 1, dtype=np.int32)

    for i, p in enumerate(params_list):
        offset = int(atom_starts[i])
        n = len(p.torsion_idx)
        torsion_term_starts[i + 1] = torsion_term_starts[i] + n
        if n > 0:
            idx_shifted = p.torsion_idx.copy()
            idx_shifted += offset
            tor_idx_parts.append(idx_shifted)
            tor_V_parts.append(p.torsion_V)
            tor_signs_parts.append(p.torsion_signs)

    # --- Improper torsion terms ---
    imp_idx_parts, imp_w_parts = [], []
    improper_term_starts = np.zeros(n_mols + 1, dtype=np.int32)

    for i, p in enumerate(params_list):
        offset = int(atom_starts[i])
        n = len(p.improper_idx)
        improper_term_starts[i + 1] = improper_term_starts[i] + n
        if n > 0:
            idx_shifted = p.improper_idx.copy()
            idx_shifted += offset
            imp_idx_parts.append(idx_shifted)
            imp_w_parts.append(p.improper_weight)

    # --- 1-2 distance constraints ---
    d12_i1_parts, d12_i2_parts = [], []
    d12_lb_parts, d12_ub_parts, d12_w_parts = [], [], []
    dist12_term_starts = np.zeros(n_mols + 1, dtype=np.int32)

    for i, p in enumerate(params_list):
        offset = int(atom_starts[i])
        n = len(p.dist12_idx1)
        dist12_term_starts[i + 1] = dist12_term_starts[i] + n
        if n > 0:
            d12_i1_parts.append(p.dist12_idx1 + offset)
            d12_i2_parts.append(p.dist12_idx2 + offset)
            d12_lb_parts.append(p.dist12_lb)
            d12_ub_parts.append(p.dist12_ub)
            d12_w_parts.append(p.dist12_weight)

    # --- 1-3 distance constraints ---
    d13_i1_parts, d13_i2_parts = [], []
    d13_lb_parts, d13_ub_parts, d13_w_parts = [], [], []
    dist13_term_starts = np.zeros(n_mols + 1, dtype=np.int32)

    for i, p in enumerate(params_list):
        offset = int(atom_starts[i])
        n = len(p.dist13_idx1)
        dist13_term_starts[i + 1] = dist13_term_starts[i] + n
        if n > 0:
            d13_i1_parts.append(p.dist13_idx1 + offset)
            d13_i2_parts.append(p.dist13_idx2 + offset)
            d13_lb_parts.append(p.dist13_lb)
            d13_ub_parts.append(p.dist13_ub)
            d13_w_parts.append(p.dist13_weight)

    # --- 1-4 distance constraints ---
    d14_i1_parts, d14_i2_parts = [], []
    d14_lb_parts, d14_ub_parts, d14_w_parts = [], [], []
    dist14_term_starts = np.zeros(n_mols + 1, dtype=np.int32)

    for i, p in enumerate(params_list):
        offset = int(atom_starts[i])
        n = len(p.dist14_idx1)
        dist14_term_starts[i + 1] = dist14_term_starts[i] + n
        if n > 0:
            d14_i1_parts.append(p.dist14_idx1 + offset)
            d14_i2_parts.append(p.dist14_idx2 + offset)
            d14_lb_parts.append(p.dist14_lb)
            d14_ub_parts.append(p.dist14_ub)
            d14_w_parts.append(p.dist14_weight)

    return BatchedETKSystem(
        n_mols=n_mols,
        n_atoms_total=n_atoms_total,
        atom_starts=atom_starts,
        torsion_idx=_concat_or_empty(tor_idx_parts, np.int32, (4,)),
        torsion_V=_concat_or_empty(tor_V_parts, np.float32, (6,)),
        torsion_signs=_concat_or_empty(tor_signs_parts, np.int32, (6,)),
        torsion_term_starts=torsion_term_starts,
        improper_idx=_concat_or_empty(imp_idx_parts, np.int32, (4,)),
        improper_weight=_concat_or_empty(imp_w_parts, np.float32),
        improper_term_starts=improper_term_starts,
        dist12_idx1=_concat_or_empty(d12_i1_parts, np.int32),
        dist12_idx2=_concat_or_empty(d12_i2_parts, np.int32),
        dist12_lb=_concat_or_empty(d12_lb_parts, np.float32),
        dist12_ub=_concat_or_empty(d12_ub_parts, np.float32),
        dist12_weight=_concat_or_empty(d12_w_parts, np.float32),
        dist12_term_starts=dist12_term_starts,
        dist13_idx1=_concat_or_empty(d13_i1_parts, np.int32),
        dist13_idx2=_concat_or_empty(d13_i2_parts, np.int32),
        dist13_lb=_concat_or_empty(d13_lb_parts, np.float32),
        dist13_ub=_concat_or_empty(d13_ub_parts, np.float32),
        dist13_weight=_concat_or_empty(d13_w_parts, np.float32),
        dist13_term_starts=dist13_term_starts,
        dist14_idx1=_concat_or_empty(d14_i1_parts, np.int32),
        dist14_idx2=_concat_or_empty(d14_i2_parts, np.int32),
        dist14_lb=_concat_or_empty(d14_lb_parts, np.float32),
        dist14_ub=_concat_or_empty(d14_ub_parts, np.float32),
        dist14_weight=_concat_or_empty(d14_w_parts, np.float32),
        dist14_term_starts=dist14_term_starts,
    )
