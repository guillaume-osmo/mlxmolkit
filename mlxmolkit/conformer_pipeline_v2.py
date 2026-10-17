"""
N×k conformer generation pipeline with divide-and-conquer memory management.

Full pipeline: SMILES → DG (4D) → 4D→3D → ETK (3D) → MMFF94 (optional)

Every conformer is one embedding attempt from its own random start. Like
RDKit's EmbedMultipleConfs, an attempt is accepted only if it passes RDKit's
embedding checks (see :mod:`mlxmolkit.conformer_gate`): after the first DG
minimisation, after ETK, and -- because MMFF runs after RDKit's last check
and can still invert a centre -- again after MMFF. Rejected attempts are
dropped, never returned as successes, and molecules still short of their k
conformers get new attempts from fresh random starts, in bounded rounds,
until each has k accepted conformers or has used its attempt budget.

Attempts are packed into GPU-sized chunks; each chunk runs DG → 3D → ETK →
(optional MMFF) and is gathered back on the CPU. A conformer's result
depends only on (seed, molecule, slot, round), not on the chunking.

Constraints are stored ONCE per molecule (SharedConstraintBatch) for the DG
stages; the ETK stage carries one set per conformer because its 1-2 and 1-3
references are that conformer's own DG distances.
"""
from __future__ import annotations

import math
import subprocess
import time
import warnings
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from . import conformer_gate as cg
from .conformer_gate import EmbedFailureCause
from .dg_extract import DGParams, extract_dg_params, get_bounds_matrix, metric_matrix_positions
from .etk_extract import ETKDG_VARIANTS, extract_etk_params
from .shared_batch import (
    SharedConstraintBatch,
    concat_etk_params,
    pack_per_conformer_etk_batch,
    pack_shared_dg_batch,
)
from .conformer_metal import dg_minimize_shared
from .etk_metal import etk_minimize_shared
from .mmff_params import extract_mmff_params
from .mmff_minimize import mmff_minimize_nk


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ConformerResult:
    """Result for one molecule.

    The per-conformer lists (positions_3d, energies, converged, stereo_ok,
    fail_cause, fail_stage) are parallel. By default they hold only accepted
    conformers -- at most the k requested, fewer (possibly none) if the
    molecule ran out of attempts -- so fail_cause is None and stereo_ok True
    throughout. With ``return_failed=True`` the rejected attempts are listed
    too, in attempt order, with their cause.
    """
    n_atoms: int
    positions_3d: List[np.ndarray]   # list of (n_atoms, 3) arrays
    energies: List[float]
    # Convergence of the last optimiser applied to the conformer: MMFF when it
    # ran, else ETK, else the DG minimisation. It says nothing about the
    # geometry checks -- see fail_cause.
    converged: List[bool]
    # True when MMFF94 optimised this molecule's conformers (their energies are
    # then MMFF energies); False when MMFF was not requested or could not type
    # the molecule, in which case mmff_error says why.
    mmff_applied: bool = False
    mmff_error: Optional[str] = None
    # Every specified tetrahedral centre and double bond perceived as specified.
    stereo_ok: List[bool] = field(default_factory=list)
    # None for an accepted conformer, else the first check it failed.
    fail_cause: List[Optional[EmbedFailureCause]] = field(default_factory=list)
    # None, or the stage whose output failed: "dg", "etk" or "mmff".
    fail_stage: List[Optional[str]] = field(default_factory=list)
    n_requested: int = 0
    n_attempted: int = 0
    # Rejected attempts by EmbedFailureCause name, over all attempts.
    n_failed_by_cause: Dict[str, int] = field(default_factory=dict)


@dataclass
class PipelineResult:
    """Result for the full N×k pipeline."""
    molecules: List[ConformerResult]
    total_conformers: int      # conformers returned (accepted ones, plus failed if asked)
    total_time: float
    n_batches: int             # GPU chunks run, over all rounds
    total_attempted: int = 0   # embedding attempts made
    n_rounds: int = 0          # attempt rounds run


# ---------------------------------------------------------------------------
# Memory helpers
# ---------------------------------------------------------------------------

def _get_free_memory_bytes() -> int:
    try:
        result = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5)
        page_size = 16384
        free_pages = 0
        for line in result.stdout.splitlines():
            if "Pages free" in line or "Pages speculative" in line:
                free_pages += int(line.split(":")[1].strip().rstrip("."))
        if free_pages > 0:
            return free_pages * page_size
    except Exception:
        pass
    return 4 * 1024 ** 3


def _compute_max_confs_per_batch(
    mol_n_atoms: List[int], dim: int = 4,
    max_memory_bytes: Optional[int] = None, lbfgs_m: int = 8,
) -> int:
    if max_memory_bytes is None:
        max_memory_bytes = _get_free_memory_bytes() // 2
    avg_atoms = int(np.mean(mol_n_atoms)) if mol_n_atoms else 30
    n_vars = avg_atoms * dim
    # pos + grad + dir + 3×scratch + 2×m×lbfgs + rho + alpha + outputs
    mem_per_conf = n_vars * 4 * 7 + 2 * lbfgs_m * n_vars * 4 + lbfgs_m * 8 + n_vars * 4 + 8
    return max(1, max_memory_bytes // max(mem_per_conf, 1))


# ---------------------------------------------------------------------------
# Iteration caps, seeds, DG continuation, MMFF typing
# ---------------------------------------------------------------------------

def _auto_iters(max_atoms: int, base: int, scale: float) -> int:
    """Scale iterations by largest molecule size. Small molecules converge
    early via in-kernel TOLX/grad checks — no wasted compute."""
    return max(base, int(base + scale * max_atoms))


def _resolve_iteration_caps(
    dg_params_list: List[DGParams], dg_max_iters: int, etk_max_iters: int,
    mmff_max_iters: int,
) -> tuple[int, int, int]:
    """Iteration caps for the whole call, from the most complex molecule.

    The caps are fixed once per call, not per chunk: a cap that followed the
    largest molecule of each chunk made a conformer's result depend on which
    other molecules happened to share its chunk. A cap is a ceiling, not a
    cost -- every conformer leaves its loop on its own convergence test.
    """
    if not dg_params_list:
        return max(dg_max_iters, 1), max(etk_max_iters, 1), max(mmff_max_iters, 1)
    max_atoms = max(p.n_atoms for p in dg_params_list)
    max_constraints = max(len(p.dist_idx1) for p in dg_params_list)
    complexity = max(max_atoms, int(max_constraints ** 0.5))
    if dg_max_iters <= 0:
        dg_max_iters = _auto_iters(complexity, base=300, scale=20.0)
    if etk_max_iters <= 0:
        etk_max_iters = _auto_iters(complexity, base=150, scale=10.0)
    if mmff_max_iters <= 0:
        mmff_max_iters = _auto_iters(complexity, base=200, scale=15.0)
    return dg_max_iters, etk_max_iters, mmff_max_iters


def _conformer_seed(base_entropy: int, mol_idx: int, conf_idx: int, attempt: int) -> tuple:
    """Seed of one embedding attempt: (seed, molecule, conformer slot, attempt).

    Every attempt draws its random start from its own stream, so the output
    does not depend on chunking, batch size or the free memory that sets it.
    """
    return (int(base_entropy), int(mol_idx), int(conf_idx), int(attempt))


def _dg_continue_unconverged(
    batch4: SharedConstraintBatch, chunk_dg: List[DGParams], dg_out: np.ndarray,
    dg_e: np.ndarray, dg_s: np.ndarray, max_iters: int,
    fourth_dim_weight: float, chiral_weight: float,
) -> None:
    """Continue the DG minimisation of the conformers that hit the iteration cap.

    RDKit's firstMinimization keeps calling minimize() until it converges;
    the kernel stops at max_iters, so the unconverged conformers -- and only
    those, whatever their number -- are restarted from where they stopped
    with twice the cap. The improved result is kept per conformer.
    Updates dg_out / dg_e / dg_s in place.
    """
    todo = np.flatnonzero(dg_s != 0)
    if len(todo) == 0:
        return
    sub_mol = batch4.conf_to_mol[todo]
    uniq, counts = np.unique(sub_mol, return_counts=True)  # todo is mol-sorted
    sub_batch = pack_shared_dg_batch([chunk_dg[m] for m in uniq], counts.tolist(), dim=4)
    spans = [(int(batch4.conf_atom_starts[c]) * 4, int(batch4.conf_atom_starts[c + 1]) * 4)
             for c in todo]
    sub_pos = np.concatenate([dg_out[a:b] for a, b in spans])
    out2, e2, s2 = dg_minimize_shared(
        sub_batch, sub_pos, max_iters=max_iters,
        fourth_dim_weight=fourth_dim_weight, chiral_weight=chiral_weight,
    )
    for j, c in enumerate(todo):
        if s2[j] == 0 or e2[j] < dg_e[c]:
            a, b = spans[j]
            sa = int(sub_batch.conf_atom_starts[j]) * 4
            dg_out[a:b] = out2[sa:sa + (b - a)]
            dg_e[c] = e2[j]
            dg_s[c] = s2[j]


def _mmff_params_or_error(mol, mmff_variant: str):
    """MMFF parameters of one molecule, or ``(None, reason)`` if it cannot be typed.

    The parameters do not depend on the coordinates except through the 100 A
    non-bonded cutoff, which an all-zero conformer satisfies for every pair, so
    each molecule is typed once instead of once per chunk.
    """
    from rdkit import Chem
    m = Chem.Mol(mol)
    m.RemoveAllConformers()
    m.AddConformer(Chem.Conformer(m.GetNumAtoms()), assignId=True)
    try:
        return extract_mmff_params(m, mmff_variant=mmff_variant), None
    except Exception as exc:  # untypable atom, missing parameters, ...
        return None, f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# Attempt processing
# ---------------------------------------------------------------------------

_STAGE_NONE, _STAGE_DG, _STAGE_ETK, _STAGE_MMFF = 0, 1, 2, 3
_STAGE_NAMES = {_STAGE_DG: "dg", _STAGE_ETK: "etk", _STAGE_MMFF: "mmff"}


# RDKit before 2026.03 gives both 1-4 pairs across a stereo double bond in a
# ring the trans window (BoundsMatrixBuilder's _getAtomStereo ignores the
# direction a 1-4 path is walked): on humulene-type terpenes the cis pairs get
# [3.76, 3.88] A instead of [2.76, 2.88]. ETK then twists the bond to the wrong
# isomer and the E/Z check rejects most attempts -- RDKit 2025.09.4's own
# srETKDGv3 rejects 40-1048 per 8 conformers of them, 2026.03.6 none.
_RING_DB_BOUNDS_FIXED = (2026, 3)


def _rdkit_version() -> tuple:
    from rdkit import __version__
    parts = []
    for p in __version__.split(".")[:2]:
        digits = "".join(ch for ch in p if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def _warn_ring_stereo_double_bonds(mols) -> None:
    """Warn once if this RDKit has the ring double-bond 1-4 bounds bug and it applies."""
    if _rdkit_version() >= _RING_DB_BOUNDS_FIXED:
        return
    from rdkit import Chem
    for mol in mols:
        for bond in mol.GetBonds():
            if (bond.GetBondType() == Chem.BondType.DOUBLE and bond.IsInRing()
                    and bond.GetStereo() > Chem.BondStereo.STEREOANY):
                from rdkit import __version__
                warnings.warn(
                    f"RDKit {__version__} gives the cis 1-4 pairs across a stereo double bond "
                    f"in a ring a trans bounds window (fixed in RDKit 2026.03): such "
                    f"molecules (e.g. {Chem.MolToSmiles(Chem.RemoveHs(mol))}) will mostly embed "
                    f"with the wrong E/Z and be rejected and resampled. Use RDKit >= 2026.03.",
                    RuntimeWarning, stacklevel=3)
                return


# Floats of BFGS inverse Hessian one ETK call may hold (512 MB).
_ETK_HESSIAN_BUDGET = 128 * 1024 * 1024


def _etk_slices(conf_n_atoms: np.ndarray, budget: int = _ETK_HESSIAN_BUDGET):
    """Contiguous [lo, hi) slices of conformers whose (3n)^2 Hessians fit in *budget*."""
    sizes = (3 * np.asarray(conf_n_atoms, dtype=np.int64)) ** 2
    out, lo, acc = [], 0, 0
    for i, sz in enumerate(sizes):
        if i > lo and acc + sz > budget:
            out.append((lo, i)); lo, acc = i, 0
        acc += int(sz)
    if len(sizes):
        out.append((lo, len(sizes)))
    return out


def _runs(conf_mol: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Run-length encoding of consecutive equal molecule indices."""
    conf_mol = np.asarray(conf_mol, dtype=np.int64)
    if len(conf_mol) == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    starts = np.concatenate([[0], np.flatnonzero(np.diff(conf_mol)) + 1])
    counts = np.diff(np.concatenate([starts, [len(conf_mol)]]))
    return conf_mol[starts], counts


def _pack_dg(dg_params_list: List[DGParams], conf_mol: np.ndarray, dim: int):
    vals, counts = _runs(conf_mol)
    batch = pack_shared_dg_batch([dg_params_list[m] for m in vals], counts.tolist(), dim=dim)
    return batch, vals


def _take_confs(flat: np.ndarray, conf_atom_starts: np.ndarray, idx: np.ndarray,
                dim: int) -> np.ndarray:
    """Concatenate the coordinate blocks of conformers ``idx``."""
    if len(idx) == 0:
        return np.zeros(0, dtype=flat.dtype)
    return np.concatenate([flat[int(conf_atom_starts[c]) * dim:int(conf_atom_starts[c + 1]) * dim]
                           for c in idx])


@dataclass
class _Context:
    dg_params_list: List[DGParams]
    etk_concat: Optional[dict]
    mol_n_atoms: np.ndarray
    gate: cg.PackedGate
    mmff_params: Optional[list]       # per molecule: MMFFParams or None (None list = no MMFF)
    dg_max_iters: int
    etk_max_iters: int
    mmff_max_iters: int
    mmff_use_lbfgs: bool
    fourth_dim_weight: float
    chiral_weight: float
    return_failed: bool


def _run_attempts(att_mol: np.ndarray, seeds: List[tuple], ctx: _Context) -> dict:
    """Run one chunk of embedding attempts through every stage and check.

    ``att_mol`` holds the molecule of each attempt, grouped by molecule.
    Returns per-attempt arrays (cause, stage, stereo_ok, energy, converged,
    mmff_applied) and the final coordinates of every accepted attempt (of
    every attempt with ``return_failed``), None elsewhere.
    """
    att_mol = np.asarray(att_mol, dtype=np.int64)
    C = len(att_mol)
    dgp = ctx.dg_params_list

    # ---- Stage 1: DG minimize (4D) from metric-matrix starts ----
    # Initial coords via RDKit's metric-matrix distance-geometry embedding (random distance
    # matrix within bounds -> double-centred Gram -> top-4 eigenvectors), NOT Gaussian noise:
    # the Gaussian start lands in a different DG basin and does not reproduce RDKit's conformers.
    batch4, run_mols = _pack_dg(dgp, att_mol, 4)
    pos4 = metric_matrix_positions(
        batch4, [dgp[m].bounds_mat for m in run_mols], dim=4, conf_seeds=seeds)
    dg_out, dg_e, dg_s = dg_minimize_shared(
        batch4, pos4, max_iters=ctx.dg_max_iters,
        fourth_dim_weight=ctx.fourth_dim_weight, chiral_weight=ctx.chiral_weight,
    )
    # ---- Stage 1b: continue the conformers that hit the cap (warm start) ----
    _dg_continue_unconverged(
        batch4, [dgp[m] for m in run_mols], dg_out, dg_e, dg_s, ctx.dg_max_iters * 2,
        ctx.fourth_dim_weight, ctx.chiral_weight)

    # ---- Stage 1c: RDKit's checks on the DG output ----
    cause = cg.check_dg_stage(dg_out, batch4.conf_atom_starts, att_mol, ctx.gate, dg_e)
    stage = np.where(cause != cg.PASS, _STAGE_DG, _STAGE_NONE).astype(np.int8)
    stereo_ok = np.zeros(C, dtype=bool)
    energy = dg_e.astype(np.float64)
    converged = dg_s == 0
    mmff_applied = np.zeros(C, dtype=bool)
    positions: List[Optional[np.ndarray]] = [None] * C
    out = dict(cause=cause, stage=stage, stereo_ok=stereo_ok, energy=energy,
               converged=converged, mmff_applied=mmff_applied, positions=positions)

    keep = np.flatnonzero((cause == cg.PASS) | ctx.return_failed)
    if len(keep) == 0:
        return out
    kmol = att_mol[keep]

    # ---- Stage 2: 4D→3D collapse (re-minimize with heavy 4th-dim penalty) ----
    sub4, _ = _pack_dg(dgp, kmol, 4)
    collapsed, _, _ = dg_minimize_shared(
        sub4, _take_confs(dg_out, batch4.conf_atom_starts, keep, 4), max_iters=200,
        fourth_dim_weight=1.0, chiral_weight=0.2,
    )
    pos3 = np.ascontiguousarray(collapsed.reshape(-1, 4)[:, :3]).ravel()
    cas3 = sub4.conf_atom_starts

    # ---- Stage 3: ETK minimize (3D), each conformer restrained to its own ----
    # 1-2 / 1-3 distances (RDKit setReferenceValues: d +/- 0.01 A).
    # The ETK optimiser (RDKit's BFGS) keeps a dense (3n)^2 inverse Hessian per
    # conformer, so the conformers go through it in slices under a fixed budget.
    if ctx.etk_concat is not None:
        conf_n = ctx.mol_n_atoms[kmol].astype(np.int64)
        out3 = pos3.copy()
        for lo, hi in _etk_slices(conf_n):
            p_in = pos3[int(cas3[lo]) * 3:int(cas3[hi]) * 3]
            etk_batch = pack_per_conformer_etk_batch(ctx.etk_concat, ctx.mol_n_atoms, kmol[lo:hi], p_in)
            has_etk = any(int(getattr(etk_batch, f"etk_{t}_term_starts")[-1]) > 0
                          for t in ("torsion", "improper", "dist12", "dist13", "dist14", "angle"))
            if has_etk:
                p_out, etk_e, etk_s = etk_minimize_shared(etk_batch, p_in, max_iters=ctx.etk_max_iters)
                out3[int(cas3[lo]) * 3:int(cas3[hi]) * 3] = p_out
                energy[keep[lo:hi]] += etk_e
                converged[keep[lo:hi]] = etk_s == 0
        pos3 = out3

    # ---- Stage 3b: final checks on the ETK output ----
    res = cg.check_final(pos3, cas3, kmol, ctx.gate)
    stereo_ok[keep] = res.stereo_ok
    fresh = stage[keep] == _STAGE_NONE
    failed_now = fresh & ~res.passed
    cause[keep[failed_now]] = res.cause[failed_now]
    stage[keep[failed_now]] = _STAGE_ETK
    final3 = pos3.reshape(-1, 3)

    # ---- Stage 4: MMFF94 on the accepted conformers of typable molecules ----
    if ctx.mmff_params is not None:
        local = np.flatnonzero((cause[keep] == cg.PASS)
                               & np.array([ctx.mmff_params[m] is not None for m in kmol], bool))
        if len(local):
            mmol = kmol[local]
            vals, counts = _runs(mmol)
            mm_pos = _take_confs(pos3, cas3, local, 3)
            mm_out, mm_e, mm_conv = mmff_minimize_nk(
                [ctx.mmff_params[m] for m in vals], counts.tolist(), mm_pos,
                max_iters=ctx.mmff_max_iters, use_lbfgs=ctx.mmff_use_lbfgs,
            )
            mm_cas = np.concatenate([[0], np.cumsum(ctx.mol_n_atoms[mmol])])
            # ---- Stage 4b: final checks again -- MMFF runs after RDKit's last
            # check and can still invert a centre or keep a 180 deg angle.
            res2 = cg.check_final(mm_out, mm_cas, mmol, ctx.gate, after_mmff=True)
            g = keep[local]
            mmff_applied[g] = True
            energy[g] = mm_e
            converged[g] = mm_conv
            stereo_ok[g] = res2.stereo_ok
            cause[g[~res2.passed]] = res2.cause[~res2.passed]
            stage[g[~res2.passed]] = _STAGE_MMFF
            final3 = final3.copy()
            mm3 = mm_out.reshape(-1, 3)
            for j, c in enumerate(local):
                final3[cas3[c]:cas3[c + 1]] = mm3[mm_cas[j]:mm_cas[j + 1]]

    for j, c in enumerate(keep):
        if cause[c] == cg.PASS or ctx.return_failed:
            positions[c] = final3[cas3[j]:cas3[j + 1]].astype(np.float32, copy=True)
    return out


def _attempts_this_round(k: int, n_passed: int, n_attempted: int, per_conf_budget: int,
                         first_round: bool, oversample: float) -> int:
    """How many new attempts a molecule gets in the next round.

    The budget is ``k * per_conf_budget`` attempts, RDKit's maxIterations per
    conformer -- except that a molecule with no accepted attempt after
    ``per_conf_budget`` attempts is given up: that is RDKit failing its first
    conformer, after which RDKit would spend (k - 1) more budgets reaching the
    same verdict. Round 0 asks for k * (1 + oversample). Later rounds size the
    request from the molecule's own acceptance rate so far (Laplace-smoothed)
    with a threefold margin: they run few conformers, so their wall time is
    the latency of the slowest one, not the count, and a generous request
    saves a whole further round. A hopeless molecule reaches its budget in
    three rounds.
    """
    need = k - n_passed
    budget = per_conf_budget if n_passed == 0 else k * per_conf_budget
    left = budget - n_attempted
    if need <= 0 or left <= 0:
        return 0
    if first_round:
        n = math.ceil(k * (1.0 + oversample))
    else:
        rate = (n_passed + 1.0) / (n_attempted + 2.0)
        n = max(2 * need, math.ceil(3.0 * need / rate))
    return int(min(n, left))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_conformers_nk(
    smiles_list: Sequence[str],
    n_confs_per_mol: int | List[int] = 10,
    *,
    max_confs_per_batch: Optional[int] = None,
    max_memory_bytes: Optional[int] = None,
    dg_max_iters: int = 0,
    etk_max_iters: int = 0,
    fourth_dim_weight: float = 0.1,
    chiral_weight: float = 1.0,
    variant: str = "ETKDGv2",
    run_mmff: bool = False,
    mmff_max_iters: int = 0,
    mmff_use_lbfgs: bool | None = None,
    mmff_variant: str = "MMFF94",
    seed: int = 42,
    max_attempts: Optional[int] = None,
    max_rounds: int = 8,
    oversample: float = 0.25,
    return_failed: bool = False,
) -> PipelineResult:
    """Generate 3D conformers for N molecules x k conformers each.

    Full pipeline: SMILES → DG (4D) → 3D → ETK (3D) → MMFF94 (optional)

    Supports all ETKDG variants: DG, KDG, ETDG, ETDGv2, ETKDG, ETKDGv2, ETKDGv3,
    srETKDGv3.  The variant controls which ETK terms are active.

    Every returned conformer passed RDKit's embedding checks (chirality,
    centre-in-volume, tetrahedral centres, double-bond geometry and E/Z) plus
    three of mlxmolkit's (perceived stereo, no bond angle above 175 deg at an
    sp2/sp3 atom, no bond far outside its bounds), on the coordinates that are
    returned -- after MMFF when MMFF ran. See :mod:`mlxmolkit.conformer_gate`.
    Rejected attempts are replaced by attempts from new random starts, like
    RDKit's EmbedMultipleConfs, so a molecule gets up to k conformers; one
    whose stereo cannot be realised gets none, as with RDKit.

    Parameters
    ----------
    smiles_list : list of str
        N SMILES strings.
    n_confs_per_mol : int or list of int
        k conformers per molecule.
    max_confs_per_batch : int, optional
        Max conformers per GPU batch (auto-computed from free memory). Does
        not change the result, only how the work is split.
    dg_max_iters : int
        L-BFGS iterations for DG stage. 0 (default) = auto-scale by the most
        complex molecule of the call:
        ``300 + 20 * max(n_atoms, sqrt(n_constraints))``.
        Small molecules converge early via in-kernel checks — no wasted compute.
    etk_max_iters : int
        L-BFGS iterations for ETK stage. 0 = auto-scale.
    variant : str
        ETKDG variant: DG, KDG, ETDG, ETDGv2, ETKDG, ETKDGv2, ETKDGv3, srETKDGv3.
    run_mmff : bool
        Whether to run MMFF94 force field optimization (default False).
        Molecules MMFF cannot type keep their ETK geometry (mmff_applied False,
        mmff_error set); the others are optimised.
    mmff_max_iters : int
        L-BFGS iterations for MMFF stage.
    seed : int
        Random seed. Attempt j of round r for molecule i starts from its own
        random stream seeded by (seed, i, j, r), so the output is identical
        whatever the chunking, ``max_confs_per_batch`` or free memory. A
        negative seed draws fresh entropy (non-reproducible), like RDKit's -1.
    max_attempts : int, optional
        Embedding attempts allowed per requested conformer (RDKit's
        maxIterations); default 10 x the number of atoms with hydrogens, as in
        RDKit. A molecule stops after ``k * max_attempts`` attempts, or after
        ``max_attempts`` if none of them was accepted -- where RDKit, having
        failed its first conformer, would go on to fail the other k - 1 at
        k times the cost.
    max_rounds : int
        Upper bound on the rounds of new attempts (each round is one pass of
        the pipeline over the molecules still short of k). Bounds the wall
        time spent on molecules whose stereo cannot be realised.
    oversample : float
        First-round attempts per molecule are ``ceil(k * (1 + oversample))``;
        later rounds size themselves from each molecule's acceptance rate.
    return_failed : bool
        Debugging: also return the rejected attempts (their coordinates at the
        stage that failed, or after ETK for DG-stage failures), with
        fail_cause / fail_stage set. Accepted conformers are unchanged.

    Returns
    -------
    PipelineResult
        ``molecules[i]`` holds molecule i's accepted conformers (at most k)
        and how many attempts it took, by failure cause.
    """
    from rdkit import Chem

    t_start = time.time()
    N = len(smiles_list)
    k_list = [n_confs_per_mol] * N if isinstance(n_confs_per_mol, int) else list(n_confs_per_mol)
    if len(k_list) != N:
        raise ValueError(f"n_confs_per_mol has {len(k_list)} entries for {N} molecules")

    # Determine which ETK stages to run based on variant
    run_etk = variant != "DG"

    # The macrocycle 1-4 bounds are what makes ETKDGv3 v3 for large rings, and
    # they enter through the bounds matrix, not through the torsion terms. The
    # variant table is the only place that knows whether they are wanted.
    macro14 = ETKDG_VARIANTS.get(variant, (False,) * 6)[4]

    # ---- Extract per-molecule params (CPU, once) ----
    mols, dg_params_list, etk_params_list, gate_params, mol_n_atoms = [], [], [], [], []
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError(f"Invalid SMILES at index {i}: {smi}")
        mol = Chem.AddHs(mol)
        mols.append(mol)
        bmat = get_bounds_matrix(mol, use_macrocycle14config=macro14)
        dg_params_list.append(extract_dg_params(mol, bmat, dim=4))
        if run_etk:
            etk_params_list.append(extract_etk_params(mol, bmat, variant=variant))
        # After extract_dg_params: same stereo perception as the DG constraints.
        gate_params.append(cg.build_gate_params(mol, bmat))
        mol_n_atoms.append(dg_params_list[-1].n_atoms)

    _warn_ring_stereo_double_bonds(mols)

    result = PipelineResult(molecules=[
        ConformerResult(n_atoms=dg_params_list[i].n_atoms, positions_3d=[], energies=[],
                        converged=[], n_requested=int(k_list[i]))
        for i in range(N)
    ], total_conformers=0, total_time=0.0, n_batches=0)
    if N == 0:
        return result

    dg_max_iters, etk_max_iters, mmff_max_iters = _resolve_iteration_caps(
        dg_params_list, dg_max_iters, etk_max_iters, mmff_max_iters)
    base_entropy = seed if seed >= 0 else int(np.random.SeedSequence().entropy)

    # Auto-select BFGS vs L-BFGS once for the call: BFGS is faster for <150
    # atoms (with H). A per-chunk choice would make results chunk-dependent.
    _LBFGS_ATOM_THRESHOLD = 150
    if mmff_use_lbfgs is None:
        mmff_use_lbfgs = max(mol_n_atoms) >= _LBFGS_ATOM_THRESHOLD

    mmff_params = None
    if run_mmff:
        mmff_params = []
        for i, mol in enumerate(mols):
            params, err = _mmff_params_or_error(mol, mmff_variant)
            mmff_params.append(params)
            result.molecules[i].mmff_applied = params is not None
            result.molecules[i].mmff_error = err

    ctx = _Context(
        dg_params_list=dg_params_list,
        etk_concat=concat_etk_params(etk_params_list) if run_etk else None,
        mol_n_atoms=np.asarray(mol_n_atoms, dtype=np.int64),
        gate=cg.pack_gate_params(gate_params),
        mmff_params=mmff_params,
        dg_max_iters=dg_max_iters, etk_max_iters=etk_max_iters,
        mmff_max_iters=mmff_max_iters, mmff_use_lbfgs=bool(mmff_use_lbfgs),
        fourth_dim_weight=fourth_dim_weight, chiral_weight=chiral_weight,
        return_failed=return_failed,
    )

    # ---- Compute batch size ----
    if max_confs_per_batch is None:
        max_confs_per_batch = _compute_max_confs_per_batch(
            mol_n_atoms, dim=4, max_memory_bytes=max_memory_bytes,
        )
    max_confs_per_batch = max(1, int(max_confs_per_batch))

    per_conf_budget = ([max(1, int(max_attempts))] * N if max_attempts is not None
                       else [10 * n for n in mol_n_atoms])
    attempted = [0] * N
    accepted: List[list] = [[] for _ in range(N)]   # (round, slot, record)
    rejected: List[list] = [[] for _ in range(N)]
    failed_by_cause: List[Counter] = [Counter() for _ in range(N)]

    # ---- Rounds of attempts (divide-and-conquer chunks within a round) ----
    for rnd in range(max(0, int(max_rounds))):
        req = [_attempts_this_round(int(k_list[m]), len(accepted[m]), attempted[m],
                                    per_conf_budget[m], rnd == 0, oversample) for m in range(N)]
        if not any(req):
            break
        result.n_rounds += 1
        att_mol = np.repeat(np.arange(N, dtype=np.int64), req)
        att_slot = np.concatenate([np.arange(n, dtype=np.int64) for n in req])
        for a in range(0, len(att_mol), max_confs_per_batch):
            b = min(a + max_confs_per_batch, len(att_mol))
            seeds = [_conformer_seed(base_entropy, m, j, rnd)
                     for m, j in zip(att_mol[a:b], att_slot[a:b])]
            out = _run_attempts(att_mol[a:b], seeds, ctx)
            result.n_batches += 1
            for j in range(b - a):
                m = int(att_mol[a + j])
                code = int(out["cause"][j])
                rec = (rnd, int(att_slot[a + j]), out["positions"][j], float(out["energy"][j]),
                       bool(out["converged"][j]), bool(out["stereo_ok"][j]), code,
                       _STAGE_NAMES.get(int(out["stage"][j])))
                if code == cg.PASS:
                    accepted[m].append(rec)
                else:
                    failed_by_cause[m][cg.cause_name(code)] += 1
                    if return_failed:
                        rejected[m].append(rec)
        for m in range(N):
            attempted[m] += req[m]

    # ---- Assemble: the first k accepted attempts, in (round, slot) order ----
    for m, mres in enumerate(result.molecules):
        kept = sorted(accepted[m], key=lambda r: (r[0], r[1]))[:int(k_list[m])]
        rows = sorted(kept + rejected[m], key=lambda r: (r[0], r[1]))
        for _, _, pos, e, conv, ok, code, stage_name in rows:
            mres.positions_3d.append(pos)
            mres.energies.append(e)
            mres.converged.append(conv)
            mres.stereo_ok.append(ok)
            mres.fail_cause.append(None if code == cg.PASS else EmbedFailureCause(code))
            mres.fail_stage.append(stage_name)
        mres.n_attempted = attempted[m]
        mres.n_failed_by_cause = dict(failed_by_cause[m])
        result.total_conformers += len(rows)
        result.total_attempted += attempted[m]

    result.total_time = time.time() - t_start
    return result
