"""Acceptance checks for embedded conformers, vectorised over a whole batch.

RDKit's ETKDG (``Code/GraphMol/DistGeomHelpers/Embedder.cpp``) accepts an
embedding attempt only if it passes a fixed sequence of checks, and otherwise
starts the next attempt from new random coordinates. This module reproduces
those checks with the same selections, formulas and tolerances, evaluated with
numpy gathers over every conformer of a batch at once, plus three checks RDKit
does not need but a GPU pipeline that keeps optimising after the final RDKit
check does:

Distance-geometry stage (on the first three coordinates of the 4D DG output):

* ``FIRST_MINIMIZATION``: DG energy per atom >= 0.05 (``MAX_MINIMIZED_E_PER_ATOM``).
* ``CHECK_TETRAHEDRAL_CENTERS``: for RDKit's ``tetrahedralCenters`` (untagged
  C/N of degree 4 in two or more rings, none of them a 3-ring), the
  normalised-volume test (``_volumeTest``, 0.5, 0.125 when the atom is in two
  or more rings smaller than 5) and the centre-in-volume test with tolerance
  0.3 (``_centerInVolume``).
* ``CHECK_CHIRAL_CENTERS``: every tagged centre's signed chiral volume against
  its bounds ([5, 100], or [2, 100] for a 3-coordinate centre whose own
  position is the 4th vertex), failing when ``vol/lb < 0.8`` or the sign is
  opposite (``checkChiralCenters``).

Final stage (on 3D coordinates, after ETK and again after MMFF; the checks
marked "ETK only" are skipped after MMFF, see below):

* ``NON_FINITE`` (mlxmolkit): any non-finite coordinate.
* ``LINEAR_DOUBLE_BOND``: an ``a-b=c`` angle at a double-bond atom with
  ``cos + 1 < 1e-3`` (about 177.4 deg), RDKit's ``doubleBondEnds`` selection.
* ``CHECK_CHIRAL_CENTERS2``: the chiral-volume test again, and (mlxmolkit) the
  configuration RDKit would perceive from these coordinates
  (``assignChiralTypesFrom3D``: the first three neighbours seen from the
  centre, |volume| > 0.1) must be the tagged one. For a 4-coordinate centre
  the DG volume is spanned by the neighbours alone and keeps its sign when the
  centre is pushed out of their tetrahedron; the perceived configuration does
  not.
* ``FINAL_CHIRAL_BOUNDS`` (ETK only): every pair of atoms of the 4-coordinate
  chiral sets within its bounds-matrix window, with RDKit's ``0.1 * ub`` slack
  (``_boundsFulfilled``).
* ``FINAL_CENTER_IN_VOLUME``: each 4-coordinate tagged centre inside the
  tetrahedron of its neighbours, tolerance 0.1.
* ``BAD_DOUBLE_BOND_STEREO``: each specified double bond's reference-atom
  dihedral on the correct side of 90 deg.
* ``CHECK_TETRAHEDRAL_CENTERS``: the centre-in-volume half of the DG-stage
  tetrahedral test, again (RDKit runs it on the DG output only; ETK and MMFF
  can still flatten or turn an untagged bridgehead inside out after that).
  The normalised-volume half is not repeated: a strained but correct centre
  can sit below its 0.5 threshold in a finished geometry -- the quinuclidine
  CH2 of CHEMBL107360 does (0.465) in 2 of 3 of RDKit's own conformers.
* ``LINEAR_ANGLE`` (mlxmolkit): any bond angle above 175 deg at an sp2 or sp3
  atom. RDKit's 1-3 bounds make this unreachable for an RDKit embedding, but a
  conformer can arrive at it in ETK, and MMFF cannot leave an exact 180 deg
  angle (its bending gradient vanishes there).
* ``BOND_LENGTH`` (mlxmolkit): any bonded distance more than
  ``BOND_LENGTH_TOL`` (0.10 A after ETK, 0.25 A after MMFF) outside its
  bounds-matrix window.

After MMFF, ``FINAL_CHIRAL_BOUNDS`` is skipped and the bond tolerance widened:
the bounds matrix encodes UFF-like lengths, and an MMFF minimum legitimately
departs from them -- RDKit's own MMFF-optimised conformers of the 500-compound
ChEMBL probe put bonds up to 0.165 A outside the 1-2 window (RDKit-aromatic
tropolone C-C at 1.55 A) and chiral-set pairs up to 0.23 * ub outside theirs.
RDKit itself never re-checks after a force-field optimisation.

Not reproduced: RDKit's ETK planarity test (``ETK_MINIMIZATION``, improper
energy above 0.7 per sp2 centre) and ``INITIAL_COORDS`` (the metric-matrix
start never fails here: non-positive eigenvalues get random coordinates).
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Dict, Sequence

import numpy as np
from rdkit import Chem


class EmbedFailureCause(enum.IntEnum):
    """Why an embedding attempt was rejected.

    Values 0-11 are RDKit's ``rdDistGeom.EmbedFailureCauses``; values from 100
    are checks RDKit does not have.
    """
    INITIAL_COORDS = 0
    FIRST_MINIMIZATION = 1
    CHECK_TETRAHEDRAL_CENTERS = 2
    CHECK_CHIRAL_CENTERS = 3
    MINIMIZE_FOURTH_DIMENSION = 4
    ETK_MINIMIZATION = 5
    FINAL_CHIRAL_BOUNDS = 6
    FINAL_CENTER_IN_VOLUME = 7
    LINEAR_DOUBLE_BOND = 8
    BAD_DOUBLE_BOND_STEREO = 9
    CHECK_CHIRAL_CENTERS2 = 10
    EXCEEDED_TIMEOUT = 11
    NON_FINITE = 100
    LINEAR_ANGLE = 101
    BOND_LENGTH = 102


PASS = -1  # cause code of an accepted conformer

# RDKit Embedder.cpp constants.
MAX_MINIMIZED_E_PER_ATOM = 0.05
MIN_TETRAHEDRAL_CHIRAL_VOL = 0.50
TETRAHEDRAL_CENTERINVOLUME_TOL = 0.30
FINAL_CENTERINVOLUME_TOL = 0.1
LINEAR_DOUBLE_BOND_TOL = 1e-3
CHIRAL_VOLUME_FRACTION = 0.8
PERCEPTION_ZERO_VOLUME_TOL = 0.1   # RDKit assignChiralTypesFrom3D
# mlxmolkit-only thresholds.
LINEAR_ANGLE_DEG = 175.0
BOND_LENGTH_TOL = 0.10        # A outside the bounds-matrix 1-2 window, after ETK
BOND_LENGTH_TOL_MMFF = 0.25   # after MMFF (see the module docstring)

_TET = (Chem.ChiralType.CHI_TETRAHEDRAL_CW, Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
_STEREO_DB = {
    Chem.BondStereo.STEREOZ: -1, Chem.BondStereo.STEREOCIS: -1,
    Chem.BondStereo.STEREOE: 1, Chem.BondStereo.STEREOTRANS: 1,
}


@dataclass
class GateParams:
    """Everything the checks need for one molecule (atom indices are local)."""
    n_atoms: int
    chiral_idx: np.ndarray        # (n, 5) centre, then the 4 volume vertices
    chiral_lb: np.ndarray         # (n,)
    chiral_ub: np.ndarray         # (n,)
    tetra_idx: np.ndarray         # (n, 5) untagged C/N centres RDKit tests
    tetra_scale: np.ndarray       # (n,) 1.0, or 0.25 in fused small rings
    cbound_idx: np.ndarray        # (n, 2) pairs among 4-coordinate chiral sets
    cbound_lb: np.ndarray
    cbound_ub: np.ndarray
    dbend_idx: np.ndarray         # (n, 3) neighbour, double-bond atom, partner
    dbstereo_idx: np.ndarray      # (n, 4) reference atom, begin, end, reference atom
    dbstereo_sign: np.ndarray     # (n,) +1 trans, -1 cis
    angle_idx: np.ndarray         # (n, 3) i, sp2/sp3 centre, k
    bond_idx: np.ndarray          # (n, 2)
    bond_lb: np.ndarray
    bond_ub: np.ndarray

    @property
    def n_stereo_elements(self) -> int:
        return len(self.chiral_idx) + len(self.dbstereo_idx)


def _i(rows, width: int) -> np.ndarray:
    return np.asarray(rows, dtype=np.int64).reshape(-1, width)


def _f(vals) -> np.ndarray:
    return np.asarray(vals, dtype=np.float64).reshape(-1)


def build_gate_params(mol: Chem.Mol, bounds_mat: np.ndarray) -> GateParams:
    """Select the atoms every check looks at, exactly as RDKit's embedder does.

    Args:
        mol: the hydrogen-complete molecule being embedded, in the state the
            DG parameters were extracted from (chiral tags and bond stereo).
        bounds_mat: its distance-bounds matrix, upper bounds in the upper
            triangle and lower bounds in the lower triangle.
    """
    bm = np.asarray(bounds_mat, dtype=np.float64)
    ri = mol.GetRingInfo()

    def lb_ub(a, b):
        lo, hi = (a, b) if a < b else (b, a)
        return bm[hi, lo], bm[lo, hi]

    chiral, c_lb, c_ub = [], [], []
    tetra, t_scale = [], []
    four_coord_atoms = set()
    # findChiralSets: heavy atoms that are tagged, or untagged C/N of degree 4.
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1:
            continue
        tag = atom.GetChiralTag()
        tagged = tag in _TET
        if not tagged and not (atom.GetAtomicNum() in (6, 7) and atom.GetDegree() == 4):
            continue
        idx = atom.GetIdx()
        # Neighbours in bond order: the order a chiral tag refers to.
        nbrs = [b.GetOtherAtomIdx(idx) for b in atom.GetBonds()]
        if len(nbrs) < 3:
            continue
        vol_lb = 5.0
        if len(nbrs) < 4:
            vol_lb = 2.0  # github #5883: three neighbours give smaller volumes
            nbrs.append(idx)
        n_small = sum(1 for ring in ri.AtomRings() if idx in ring and len(ring) < 5)
        scale = 0.25 if n_small > 1 else 1.0
        if tagged:
            chiral.append([idx] + nbrs[:4])
            if tag == Chem.ChiralType.CHI_TETRAHEDRAL_CCW:
                c_lb.append(vol_lb), c_ub.append(100.0)
            else:
                c_lb.append(-100.0), c_ub.append(-vol_lb)
            if nbrs[3] != idx:
                four_coord_atoms.update([idx] + nbrs[:4])
        elif ri.NumAtomRings(idx) >= 2 and not ri.IsAtomInRingOfSize(idx, 3):
            tetra.append([idx] + nbrs[:4])
            t_scale.append(scale)

    cb = sorted(four_coord_atoms)
    cb_pairs = [(cb[i], cb[j]) for i in range(len(cb)) for j in range(i + 1, len(cb))]
    cb_bounds = [lb_ub(a, b) for a, b in cb_pairs]

    dbend, dbst, dbsign = [], [], []
    for bond in mol.GetBonds():
        if bond.GetBondType() != Chem.BondType.DOUBLE:
            continue
        for atm in (bond.GetBeginAtom(), bond.GetEndAtom()):
            if atm.GetDegree() < 2:
                continue
            oidx = bond.GetOtherAtomIdx(atm.GetIdx())
            for nbr in atm.GetNeighbors():
                if nbr.GetIdx() == oidx:
                    continue
                obnd = mol.GetBondBetweenAtoms(atm.GetIdx(), nbr.GetIdx())
                if obnd.GetBondType() != Chem.BondType.SINGLE and atm.GetDegree() == 2:
                    continue
                dbend.append([nbr.GetIdx(), atm.GetIdx(), oidx])
        sign = _STEREO_DB.get(bond.GetStereo())
        sa = list(bond.GetStereoAtoms())
        if sign is not None and len(sa) == 2:
            dbst.append([sa[0], bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), sa[1]])
            dbsign.append(sign)

    angles = []
    for atom in mol.GetAtoms():
        if atom.GetHybridization() not in (Chem.HybridizationType.SP2,
                                           Chem.HybridizationType.SP3):
            continue
        nb = [n.GetIdx() for n in atom.GetNeighbors()]
        for a in range(len(nb)):
            for b in range(a + 1, len(nb)):
                angles.append([nb[a], atom.GetIdx(), nb[b]])

    bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]
    b_bounds = [lb_ub(a, b) for a, b in bonds]

    return GateParams(
        n_atoms=mol.GetNumAtoms(),
        chiral_idx=_i(chiral, 5), chiral_lb=_f(c_lb), chiral_ub=_f(c_ub),
        tetra_idx=_i(tetra, 5), tetra_scale=_f(t_scale),
        cbound_idx=_i(cb_pairs, 2), cbound_lb=_f([x[0] for x in cb_bounds]),
        cbound_ub=_f([x[1] for x in cb_bounds]),
        dbend_idx=_i(dbend, 3), dbstereo_idx=_i(dbst, 4), dbstereo_sign=_f(dbsign),
        angle_idx=_i(angles, 3),
        bond_idx=_i(bonds, 2), bond_lb=_f([x[0] for x in b_bounds]),
        bond_ub=_f([x[1] for x in b_bounds]),
    )


# Term tables: name -> (index field, value fields)
_TERMS = {
    "chiral": ("chiral_idx", ("chiral_lb", "chiral_ub")),
    "tetra": ("tetra_idx", ("tetra_scale",)),
    "cbound": ("cbound_idx", ("cbound_lb", "cbound_ub")),
    "dbend": ("dbend_idx", ()),
    "dbstereo": ("dbstereo_idx", ("dbstereo_sign",)),
    "angle": ("angle_idx", ()),
    "bond": ("bond_idx", ("bond_lb", "bond_ub")),
}


_IDX_WIDTH = {"chiral_idx": 5, "tetra_idx": 5, "cbound_idx": 2, "dbend_idx": 3,
              "dbstereo_idx": 4, "angle_idx": 3, "bond_idx": 2}


@dataclass
class PackedGate:
    """GateParams of N molecules concatenated, with CSR offsets per term table."""
    n_atoms: np.ndarray                         # (N,)
    starts: Dict[str, np.ndarray] = field(default_factory=dict)   # term -> (N+1,)
    arrays: Dict[str, np.ndarray] = field(default_factory=dict)   # field -> stacked


def pack_gate_params(params: Sequence[GateParams]) -> PackedGate:
    packed = PackedGate(n_atoms=np.array([p.n_atoms for p in params], dtype=np.int64))
    for term, (idx_field, val_fields) in _TERMS.items():
        counts = np.array([len(getattr(p, idx_field)) for p in params], dtype=np.int64)
        starts = np.zeros(len(params) + 1, dtype=np.int64)
        np.cumsum(counts, out=starts[1:])
        packed.starts[term] = starts
        width = _IDX_WIDTH[idx_field]
        packed.arrays[idx_field] = (np.concatenate([getattr(p, idx_field) for p in params])
                                    if params else np.zeros((0, width), dtype=np.int64))
        for f in val_fields:
            packed.arrays[f] = (np.concatenate([getattr(p, f) for p in params])
                                if params else np.zeros(0))
    return packed


def _gather(packed: PackedGate, term: str, conf_mol: np.ndarray,
            conf_atom_starts: np.ndarray):
    """Replicate a term table once per conformer, indices made global."""
    idx_field, val_fields = _TERMS[term]
    starts = packed.starts[term]
    counts = starts[conf_mol + 1] - starts[conf_mol]
    total = int(counts.sum())
    conf_of_term = np.repeat(np.arange(len(conf_mol), dtype=np.int64), counts)
    if total == 0:
        return (conf_of_term, np.zeros((0, _IDX_WIDTH[idx_field]), dtype=np.int64),
                [np.zeros(0) for _ in val_fields])
    first = np.cumsum(counts) - counts
    src = (np.arange(total, dtype=np.int64) - np.repeat(first, counts)
           + np.repeat(starts[conf_mol], counts))
    idx = packed.arrays[idx_field][src] + np.asarray(conf_atom_starts, np.int64)[conf_of_term, None]
    return conf_of_term, idx, [packed.arrays[f][src] for f in val_fields]


def _any_per_conf(conf_of_term: np.ndarray, bad: np.ndarray, n_confs: int) -> np.ndarray:
    if len(bad) == 0:
        return np.zeros(n_confs, dtype=bool)
    return np.bincount(conf_of_term[bad], minlength=n_confs) > 0


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n > 0, n, 1.0)


def _chiral_volume(x: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """RDKit calcChiralVolume(idx1..idx4): (p1-p4) . ((p2-p4) x (p3-p4))."""
    p4 = x[idx[:, 4]]
    return np.einsum("ij,ij->i", x[idx[:, 1]] - p4,
                     np.cross(x[idx[:, 2]] - p4, x[idx[:, 3]] - p4))


def _chiral_bad(vol: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    """RDKit checkChiralCenters: too small (vol/lb < .8) or of the wrong sign."""
    with np.errstate(divide="ignore", invalid="ignore"):
        low = (lb > 0) & (vol < lb) & ((vol / lb < CHIRAL_VOLUME_FRACTION)
                                       | (np.signbit(vol) != np.signbit(lb)))
        high = (ub < 0) & (vol > ub) & ((vol / ub < CHIRAL_VOLUME_FRACTION)
                                        | (np.signbit(vol) != np.signbit(ub)))
    return low | high | ~np.isfinite(vol)


def _perceived_as_tagged(x: np.ndarray, idx: np.ndarray, lb: np.ndarray) -> np.ndarray:
    """True where RDKit would read the tagged configuration back from 3D.

    assignChiralTypesFrom3D: (n1-c) . ((n2-c) x (n3-c)) over the first three
    neighbours in bond order; CCW above +0.1, CW below -0.1, unassigned in
    between. CCW sets have positive volume bounds (lb > 0).
    """
    c = x[idx[:, 0]]
    v = np.einsum("ij,ij->i", x[idx[:, 1]] - c,
                  np.cross(x[idx[:, 2]] - c, x[idx[:, 3]] - c))
    return np.where(lb > 0, v > PERCEPTION_ZERO_VOLUME_TOL, v < -PERCEPTION_ZERO_VOLUME_TOL)


def _same_side(v1, v2, v3, v4, p0, tol):
    normal = np.cross(v2 - v1, v3 - v1)
    d1 = np.einsum("ij,ij->i", normal, v4 - v1)
    d2 = np.einsum("ij,ij->i", normal, p0 - v1)
    ok = (np.abs(d1) >= tol) & (np.abs(d2) >= tol)
    return ok & ((d1 < 0) == (d2 < 0))


def _center_in_volume(x: np.ndarray, idx: np.ndarray, tol: float) -> np.ndarray:
    """RDKit _centerInVolume (True = inside); 3-coordinate sets always pass."""
    p0, p1, p2, p3, p4 = (x[idx[:, i]] for i in range(5))
    inside = (_same_side(p1, p2, p3, p4, p0, tol) & _same_side(p2, p3, p4, p1, p0, tol)
              & _same_side(p3, p4, p1, p2, p0, tol) & _same_side(p4, p1, p2, p3, p0, tol))
    return inside | (idx[:, 0] == idx[:, 4])


def _volume_test(x: np.ndarray, idx: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """RDKit _volumeTest (True = enough volume) on normalised centre->neighbour vectors."""
    p0 = x[idx[:, 0]]
    v1, v2, v3, v4 = (_unit(p0 - x[idx[:, i]]) for i in range(1, 5))
    thr = scale * MIN_TETRAHEDRAL_CHIRAL_VOL
    ok = np.ones(len(idx), dtype=bool)
    for a, b, c in ((v1, v2, v3), (v1, v2, v4), (v1, v3, v4), (v2, v3, v4)):
        ok &= np.abs(np.einsum("ij,ij->i", np.cross(a, b), c)) >= thr
    return ok


def _dihedral_unsigned(p0, p1, p2, p3) -> np.ndarray:
    """RDKit computeDihedralAngle: angle in [0, pi] between the two bond planes."""
    beg_end = p2 - p1
    crs1 = np.cross(p0 - p1, beg_end)
    crs2 = np.cross(p3 - p2, beg_end)
    den = np.linalg.norm(crs1, axis=1) * np.linalg.norm(crs2, axis=1)
    cosv = np.einsum("ij,ij->i", crs1, crs2) / np.where(den > 0, den, 1.0)
    return np.arccos(np.clip(cosv, -1.0, 1.0))


@dataclass
class GateResult:
    """Per-conformer outcome. ``cause`` is PASS (-1) or an EmbedFailureCause value."""
    cause: np.ndarray        # (C,) int16
    # (C,) bool: every specified stereo element is perceived as specified --
    # tetrahedral centres as RDKit's assignChiralTypesFrom3D would read them,
    # double bonds by the side of 90 deg of their reference-atom dihedral.
    stereo_ok: np.ndarray

    @property
    def passed(self) -> np.ndarray:
        return self.cause == PASS


def _assign(cause: np.ndarray, fails: np.ndarray, code: EmbedFailureCause) -> None:
    cause[(cause == PASS) & fails] = int(code)


def _coords3(coords: np.ndarray, n_atoms_total: int) -> np.ndarray:
    x = np.asarray(coords, dtype=np.float64).reshape(n_atoms_total, -1)
    return x[:, :3]


def check_dg_stage(
    coords: np.ndarray, conf_atom_starts: np.ndarray, conf_mol: np.ndarray,
    packed: PackedGate, dg_energy: np.ndarray,
) -> np.ndarray:
    """RDKit's checks after the first DG minimisation.

    Args:
        coords: flat DG output (atoms * dim, dim 3 or 4); only xyz is used.
        conf_atom_starts: (C+1,) atom offset of each conformer in ``coords``.
        conf_mol: (C,) row of ``packed`` describing each conformer.
        packed: :func:`pack_gate_params` output.
        dg_energy: (C,) DG energy of each conformer (distance + chiral + 4th
            dimension terms, RDKit's weights).

    Returns:
        (C,) int16 cause codes, PASS for accepted conformers.
    """
    conf_mol = np.asarray(conf_mol, dtype=np.int64)
    conf_atom_starts = np.asarray(conf_atom_starts, dtype=np.int64)
    n_confs = len(conf_mol)
    x = _coords3(coords, int(conf_atom_starts[-1]))
    cause = np.full(n_confs, PASS, dtype=np.int16)

    e = np.asarray(dg_energy, dtype=np.float64)
    e_per_atom = e / packed.n_atoms[conf_mol]
    _assign(cause, ~np.isfinite(e_per_atom) | (e_per_atom >= MAX_MINIMIZED_E_PER_ATOM),
            EmbedFailureCause.FIRST_MINIMIZATION)

    ct, idx, (scale,) = _gather(packed, "tetra", conf_mol, conf_atom_starts)
    if len(idx):
        bad = ~(_volume_test(x, idx, scale)
                & _center_in_volume(x, idx, TETRAHEDRAL_CENTERINVOLUME_TOL))
        _assign(cause, _any_per_conf(ct, bad, n_confs),
                EmbedFailureCause.CHECK_TETRAHEDRAL_CENTERS)

    ct, idx, (lb, ub) = _gather(packed, "chiral", conf_mol, conf_atom_starts)
    if len(idx):
        bad = _chiral_bad(_chiral_volume(x, idx), lb, ub)
        _assign(cause, _any_per_conf(ct, bad, n_confs), EmbedFailureCause.CHECK_CHIRAL_CENTERS)
    return cause


def check_final(
    coords: np.ndarray, conf_atom_starts: np.ndarray, conf_mol: np.ndarray,
    packed: PackedGate, *, after_mmff: bool = False,
) -> GateResult:
    """Final acceptance checks on 3D coordinates (see the module docstring).

    Args:
        coords: flat 3D coordinates of C conformers (atoms * 3).
        conf_atom_starts: (C+1,) atom offset of each conformer in ``coords``.
        conf_mol: (C,) row of ``packed`` describing each conformer.
        packed: :func:`pack_gate_params` output.
        after_mmff: the coordinates are an MMFF minimum, not an ETK output:
            skip FINAL_CHIRAL_BOUNDS and use BOND_LENGTH_TOL_MMFF.
    """
    conf_mol = np.asarray(conf_mol, dtype=np.int64)
    conf_atom_starts = np.asarray(conf_atom_starts, dtype=np.int64)
    n_confs = len(conf_mol)
    x = _coords3(coords, int(conf_atom_starts[-1]))
    cause = np.full(n_confs, PASS, dtype=np.int16)
    stereo_ok = np.ones(n_confs, dtype=bool)

    atom_conf = np.repeat(np.arange(n_confs), np.diff(conf_atom_starts))
    finite = ~(np.bincount(atom_conf[~np.isfinite(x).all(axis=1)], minlength=n_confs) > 0)
    _assign(cause, ~finite, EmbedFailureCause.NON_FINITE)
    # Later checks on a non-finite conformer would only add warnings.
    x = np.where(np.isfinite(x), x, 0.0)

    ct, idx, _ = _gather(packed, "dbend", conf_mol, conf_atom_starts)
    if len(idx):
        v1 = _unit(x[idx[:, 1]] - x[idx[:, 0]])
        v2 = _unit(x[idx[:, 1]] - x[idx[:, 2]])
        bad = np.einsum("ij,ij->i", v1, v2) + 1.0 < LINEAR_DOUBLE_BOND_TOL
        _assign(cause, _any_per_conf(ct, bad, n_confs), EmbedFailureCause.LINEAR_DOUBLE_BOND)

    ct, idx, (lb, ub) = _gather(packed, "chiral", conf_mol, conf_atom_starts)
    if len(idx):
        vol = _chiral_volume(x, idx)
        perceived_wrong = ~_perceived_as_tagged(x, idx, lb)
        stereo_ok &= ~_any_per_conf(ct, perceived_wrong, n_confs)
        _assign(cause, _any_per_conf(ct, _chiral_bad(vol, lb, ub) | perceived_wrong, n_confs),
                EmbedFailureCause.CHECK_CHIRAL_CENTERS2)

    ct2, idx2, (blb, bub) = _gather(packed, "cbound", conf_mol, conf_atom_starts)
    if len(idx2) and not after_mmff:
        d = np.linalg.norm(x[idx2[:, 0]] - x[idx2[:, 1]], axis=1)
        bad = (((d < blb) & (np.abs(d - blb) > 0.1 * bub))
               | ((d > bub) & (np.abs(d - bub) > 0.1 * bub)))
        _assign(cause, _any_per_conf(ct2, bad, n_confs), EmbedFailureCause.FINAL_CHIRAL_BOUNDS)

    if len(idx):
        bad = ~_center_in_volume(x, idx, FINAL_CENTERINVOLUME_TOL)
        _assign(cause, _any_per_conf(ct, bad, n_confs), EmbedFailureCause.FINAL_CENTER_IN_VOLUME)

    ct, idx, (sign,) = _gather(packed, "dbstereo", conf_mol, conf_atom_starts)
    if len(idx):
        dih = _dihedral_unsigned(x[idx[:, 0]], x[idx[:, 1]], x[idx[:, 2]], x[idx[:, 3]])
        bad = (dih - np.pi / 2) * sign < 0
        fails = _any_per_conf(ct, bad, n_confs)
        _assign(cause, fails, EmbedFailureCause.BAD_DOUBLE_BOND_STEREO)
        stereo_ok &= ~fails

    ct, idx, _ = _gather(packed, "tetra", conf_mol, conf_atom_starts)
    if len(idx):
        bad = ~_center_in_volume(x, idx, TETRAHEDRAL_CENTERINVOLUME_TOL)
        _assign(cause, _any_per_conf(ct, bad, n_confs),
                EmbedFailureCause.CHECK_TETRAHEDRAL_CENTERS)

    ct, idx, _ = _gather(packed, "angle", conf_mol, conf_atom_starts)
    if len(idx):
        u = _unit(x[idx[:, 0]] - x[idx[:, 1]])
        v = _unit(x[idx[:, 2]] - x[idx[:, 1]])
        bad = np.einsum("ij,ij->i", u, v) < np.cos(np.radians(LINEAR_ANGLE_DEG))
        _assign(cause, _any_per_conf(ct, bad, n_confs), EmbedFailureCause.LINEAR_ANGLE)

    ct, idx, (blb, bub) = _gather(packed, "bond", conf_mol, conf_atom_starts)
    if len(idx):
        d = np.linalg.norm(x[idx[:, 0]] - x[idx[:, 1]], axis=1)
        tol = BOND_LENGTH_TOL_MMFF if after_mmff else BOND_LENGTH_TOL
        bad = (d < blb - tol) | (d > bub + tol)
        _assign(cause, _any_per_conf(ct, bad, n_confs), EmbedFailureCause.BOND_LENGTH)

    return GateResult(cause=cause, stereo_ok=stereo_ok)


def cause_name(code: int) -> str:
    return "PASS" if int(code) == PASS else EmbedFailureCause(int(code)).name
