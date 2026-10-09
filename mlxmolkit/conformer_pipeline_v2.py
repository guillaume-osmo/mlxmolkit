"""
N×k conformer generation pipeline with divide-and-conquer memory management.

Full pipeline: SMILES → DG (4D) → 4D→3D → ETK (3D) → MMFF94 (optional)

The divide-and-conquer queue splits conformers into GPU-sized batches.
Each batch: DG → extract 3D → ETK → (optional MMFF) → accumulate on CPU.
GPU memory is released between batches.

Constraints are stored ONCE per molecule (SharedConstraintBatch).
"""
from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import mlx.core as mx

from .dg_extract import DGParams, extract_dg_params, get_bounds_matrix, metric_matrix_positions
from .etk_extract import ETKDG_VARIANTS, ETKParams, extract_etk_params
from .shared_batch import (
    SharedConstraintBatch,
    concat_etk_params,
    pack_per_conformer_etk_batch,
    pack_shared_dg_batch,
)
from .conformer_metal import dg_minimize_shared
from .etk_metal import etk_minimize_shared
from .mmff_params import MMFFParams, extract_mmff_params
from .mmff_minimize import mmff_minimize_nk
from .stereo_checks_metal import run_stereo_checks


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ConformerResult:
    """Result for one molecule."""
    n_atoms: int
    positions_3d: List[np.ndarray]   # list of (n_atoms, 3) arrays
    energies: List[float]
    converged: List[bool]
    # True when MMFF94 optimised this molecule's conformers (their energies are
    # then MMFF energies); False when MMFF was not requested or could not type
    # the molecule, in which case mmff_error says why.
    mmff_applied: bool = False
    mmff_error: Optional[str] = None


@dataclass
class PipelineResult:
    """Result for the full N×k pipeline."""
    molecules: List[ConformerResult]
    total_conformers: int
    total_time: float
    n_batches: int


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
# Chunk scheduler
# ---------------------------------------------------------------------------

def _build_chunk_schedule(
    n_confs_per_mol: List[int], max_confs_per_batch: int,
) -> List[List[tuple]]:
    chunks: List[List[tuple]] = []
    current_chunk: List[tuple] = []
    current_count = 0
    for mol_idx, k in enumerate(n_confs_per_mol):
        remaining, offset = k, 0
        while remaining > 0:
            space = max_confs_per_batch - current_count
            take = min(remaining, space)
            current_chunk.append((mol_idx, offset, offset + take))
            current_count += take
            offset += take
            remaining -= take
            if current_count >= max_confs_per_batch:
                chunks.append(current_chunk)
                current_chunk = []
                current_count = 0
    if current_chunk:
        chunks.append(current_chunk)
    return chunks


# ---------------------------------------------------------------------------
# Per-chunk processing
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


def _process_chunk(
    chunk: List[tuple],
    dg_params_list: List[DGParams],
    etk_params_list: Optional[List[ETKParams]],
    mols_list: Optional[list],
    run_mmff: bool,
    conf_seeds: List[tuple],
    dg_max_iters: int,
    etk_max_iters: int,
    mmff_max_iters: int,
    mmff_use_lbfgs: bool,
    mmff_variant: str,
    fourth_dim_weight: float,
    chiral_weight: float,
    mmff_cache: Optional[dict] = None,
) -> List[tuple]:
    """Run DG → 3D → ETK → MMFF on one chunk. Returns per-conformer results.

    ``conf_seeds`` holds one seed per conformer of the chunk, in chunk order.
    The iteration caps must already be resolved (> 0), see
    :func:`_resolve_iteration_caps`.
    """
    if mmff_cache is None:
        mmff_cache = {}

    # Identify molecules in this chunk
    mol_k: dict[int, int] = {}
    mol_order: List[int] = []
    for mol_idx, c_start, c_end in chunk:
        if mol_idx not in mol_k:
            mol_k[mol_idx] = 0
            mol_order.append(mol_idx)
        mol_k[mol_idx] += c_end - c_start

    chunk_dg = [dg_params_list[m] for m in mol_order]
    chunk_k = [mol_k[m] for m in mol_order]
    C = sum(chunk_k)

    # ---- Stage 1: DG minimize (4D) ----
    # Initial coords via RDKit's metric-matrix distance-geometry embedding (random distance
    # matrix within bounds -> double-centred Gram -> top-4 eigenvectors), NOT Gaussian noise:
    # the Gaussian start lands in a different DG basin and does not reproduce RDKit's conformers.
    # A molecule without a bounds matrix falls back to a Gaussian start, drawn
    # from the same per-conformer stream.
    batch4 = pack_shared_dg_batch(chunk_dg, chunk_k, dim=4)
    pos4 = metric_matrix_positions(
        batch4, [dg.bounds_mat for dg in chunk_dg], dim=4, conf_seeds=conf_seeds)
    dg_out, dg_e, dg_s = dg_minimize_shared(
        batch4, pos4, max_iters=dg_max_iters,
        fourth_dim_weight=fourth_dim_weight, chiral_weight=chiral_weight,
    )

    # ---- Stage 1b: continue the conformers that hit the cap (warm start) ----
    _dg_continue_unconverged(
        batch4, chunk_dg, dg_out, dg_e, dg_s, dg_max_iters * 2,
        fourth_dim_weight, chiral_weight)

    # ---- Stage 1c: Stereo checks (reject bad chirality) ----
    if mols_list is not None:
        stereo_passed = run_stereo_checks(
            dg_out, dg_params_list, mols_list,
            batch4.conf_atom_starts, batch4.conf_to_mol,
            mol_order, C, dim=4,
        )
        # Mark failed conformers as not converged
        for c in range(C):
            if not stereo_passed[c]:
                dg_s[c] = 2  # stereo failure

    # ---- Stage 2: 4D→3D collapse (re-minimize with heavy 4th-dim penalty) ----
    dg_collapse, _, _ = dg_minimize_shared(
        batch4, dg_out, max_iters=200,
        fourth_dim_weight=1.0, chiral_weight=0.2,
    )

    # ---- Stage 2b: Extract 3D from collapsed 4D ----
    batch3 = pack_shared_dg_batch(chunk_dg, chunk_k, dim=3)
    pos3 = np.zeros(int(batch3.conf_atom_starts[-1]) * 3, dtype=np.float32)
    for c in range(C):
        n_a = batch3.mol_n_atoms[batch3.conf_to_mol[c]]
        s4 = int(batch4.conf_atom_starts[c]) * 4
        p4 = dg_collapse[s4:s4 + n_a * 4].reshape(n_a, 4)
        s3 = int(batch3.conf_atom_starts[c]) * 3
        pos3[s3:s3 + n_a * 3] = p4[:, :3].flatten()

    # ---- Stage 2c: per-conformer reference values (RDKit setReferenceValues) ----
    # RDKit's ETK force field restrains every 1-2 and 1-3 distance to the value
    # it has in THAT conformer's DG output (+/- 0.01 A). The reference therefore
    # differs per conformer, so the ETK batch carries one constraint set per
    # conformer instead of one per molecule.
    # ---- Stage 3: ETK minimize (3D) ----
    etk_e = np.zeros(C, dtype=np.float32)
    etk_s = np.zeros(C, dtype=np.int32)
    has_etk = False
    if etk_params_list is not None:
        etk_batch = pack_per_conformer_etk_batch(
            concat_etk_params([etk_params_list[m] for m in mol_order]),
            batch3.mol_n_atoms, batch3.conf_to_mol, pos3,
        )
        has_etk = any(
            int(getattr(etk_batch, f"etk_{t}_term_starts")[-1]) > 0
            for t in ("torsion", "improper", "dist12", "dist13", "dist14")
        )
        if has_etk:
            etk_out, etk_e, etk_s = etk_minimize_shared(
                etk_batch, pos3, max_iters=etk_max_iters,
            )
            pos3 = etk_out

    # ---- Stage 4: MMFF94 optimization (3D) ----
    # Each molecule is typed once (mmff_cache); a molecule MMFF cannot type is
    # skipped on its own and keeps its ETK geometry and energy, while the rest
    # of the chunk is still optimised.
    mmff_e = np.zeros(C, dtype=np.float32)
    mmff_converged = np.zeros(C, dtype=bool)
    mmff_applied = np.zeros(C, dtype=bool)
    if run_mmff and mols_list is not None:
        ok_mols, ok_k = [], []
        seg = []  # (start, end) flat coordinate ranges of the optimised conformers
        c0 = 0
        for mol_idx in mol_order:
            k = mol_k[mol_idx]
            if mol_idx not in mmff_cache:
                mmff_cache[mol_idx] = _mmff_params_or_error(mols_list[mol_idx], mmff_variant)
            params, _ = mmff_cache[mol_idx]
            if params is not None:
                ok_mols.append(params)
                ok_k.append(k)
                mmff_applied[c0:c0 + k] = True
                seg.append((int(batch3.conf_atom_starts[c0]) * 3,
                            int(batch3.conf_atom_starts[c0 + k]) * 3))
            c0 += k
        if ok_mols:
            sub_pos = np.concatenate([pos3[a:b] for a, b in seg])
            sub_out, sub_e, sub_conv = mmff_minimize_nk(
                ok_mols, ok_k, sub_pos,
                max_iters=mmff_max_iters,
                use_lbfgs=mmff_use_lbfgs,
            )
            pos3 = pos3.copy()
            cur = 0
            for a, b in seg:
                pos3[a:b] = sub_out[cur:cur + (b - a)]
                cur += b - a
            mmff_e[mmff_applied] = sub_e
            mmff_converged[mmff_applied] = sub_conv

    # ---- Collect results per conformer ----
    results = []
    c = 0
    for mol_idx in mol_order:
        k = mol_k[mol_idx]
        for lk in range(k):
            n_a = dg_params_list[mol_idx].n_atoms
            s3 = int(batch3.conf_atom_starts[c]) * 3
            p3 = pos3[s3:s3 + n_a * 3].reshape(n_a, 3).copy()
            applied = bool(mmff_applied[c])
            energy = float(mmff_e[c]) if applied else float(dg_e[c]) + float(etk_e[c])
            converged = (
                bool(dg_s[c] == 0)
                and (not has_etk or bool(etk_s[c] == 0))
                and (not applied or bool(mmff_converged[c]))
            )
            results.append((
                mol_idx, p3,
                energy,
                converged,
                applied,
            ))
            c += 1

    return results


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
) -> PipelineResult:
    """Generate 3D conformers for N molecules x k conformers each.

    Full pipeline: SMILES → DG (4D) → 3D → ETK (3D) → MMFF94 (optional)

    Supports all ETKDG variants: DG, KDG, ETDG, ETDGv2, ETKDG, ETKDGv2, ETKDGv3,
    srETKDGv3.  The variant controls which ETK terms are active.

    Parameters
    ----------
    smiles_list : list of str
        N SMILES strings.
    n_confs_per_mol : int or list of int
        k conformers per molecule.
    max_confs_per_batch : int, optional
        Max conformers per GPU batch (auto-computed from free memory).
    dg_max_iters : int
        L-BFGS iterations for DG stage. 0 (default) = auto-scale by
        molecule complexity: ``300 + 20 * max(n_atoms, sqrt(n_constraints))``.
        Small molecules converge early via in-kernel checks — no wasted compute.
    etk_max_iters : int
        L-BFGS iterations for ETK stage. 0 = auto-scale.
    variant : str
        ETKDG variant: DG, KDG, ETDG, ETDGv2, ETKDG, ETKDGv2, ETKDGv3, srETKDGv3.
    run_mmff : bool
        Whether to run MMFF94 force field optimization (default False).
    mmff_max_iters : int
        L-BFGS iterations for MMFF stage.
    seed : int
        Random seed. Conformer j of molecule i starts from its own random
        stream seeded by (seed, i, j, attempt), so the output is identical
        whatever the chunking, ``max_confs_per_batch`` or free memory. A
        negative seed draws fresh entropy (non-reproducible), like RDKit's -1.
    """
    from rdkit import Chem

    t_start = time.time()
    N = len(smiles_list)
    k_list = [n_confs_per_mol] * N if isinstance(n_confs_per_mol, int) else list(n_confs_per_mol)

    # Determine which ETK stages to run based on variant
    run_etk = variant != "DG"

    # The macrocycle 1-4 bounds are what makes ETKDGv3 v3 for large rings, and
    # they enter through the bounds matrix, not through the torsion terms. The
    # variant table is the only place that knows whether they are wanted.
    macro14 = ETKDG_VARIANTS.get(variant, (False,) * 6)[4]

    # ---- Extract per-molecule params (CPU, once) ----
    mols, dg_params_list, etk_params_list, mmff_params_list_all, mol_n_atoms = [], [], [], [], []
    bmats = []
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError(f"Invalid SMILES at index {i}: {smi}")
        mol = Chem.AddHs(mol)
        mols.append(mol)
        bmat = get_bounds_matrix(mol, use_macrocycle14config=macro14)
        bmats.append(bmat)
        dg_params_list.append(extract_dg_params(mol, bmat, dim=4))
        if run_etk:
            etk_params_list.append(extract_etk_params(mol, bmat, variant=variant))
        mol_n_atoms.append(dg_params_list[-1].n_atoms)

    # MMFF extraction deferred — needs a conformer, we'll embed after DG+ETK
    # to avoid wasting time on a separate RDKit EmbedMolecule call

    dg_max_iters, etk_max_iters, mmff_max_iters = _resolve_iteration_caps(
        dg_params_list, dg_max_iters, etk_max_iters, mmff_max_iters)
    base_entropy = seed if seed >= 0 else int(np.random.SeedSequence().entropy)

    # ---- Compute batch size ----
    if max_confs_per_batch is None:
        max_confs_per_batch = _compute_max_confs_per_batch(
            mol_n_atoms, dim=4, max_memory_bytes=max_memory_bytes,
        )

    # ---- Build chunk schedule ----
    chunks = _build_chunk_schedule(k_list, max_confs_per_batch)

    # ---- Process chunks (divide-and-conquer) ----
    mol_results = [
        ConformerResult(n_atoms=dg_params_list[i].n_atoms, positions_3d=[], energies=[], converged=[])
        for i in range(N)
    ]
    # Auto-select BFGS vs L-BFGS: BFGS is faster for <150 atoms (with H)
    _LBFGS_ATOM_THRESHOLD = 150
    if mmff_use_lbfgs is None:
        max_atoms_all = max(p.n_atoms for p in dg_params_list)
        mmff_use_lbfgs = max_atoms_all >= _LBFGS_ATOM_THRESHOLD

    total_confs = 0
    mmff_cache: dict = {}
    for chunk_idx, chunk in enumerate(chunks):
        chunk_results = _process_chunk(
            chunk, dg_params_list,
            etk_params_list if run_etk else None,
            mols,  # needed for stereo checks + MMFF
            run_mmff,
            conf_seeds=[_conformer_seed(base_entropy, m, j, 0)
                        for m, c_start, c_end in chunk for j in range(c_start, c_end)],
            dg_max_iters=dg_max_iters,
            etk_max_iters=etk_max_iters,
            mmff_max_iters=mmff_max_iters,
            mmff_use_lbfgs=mmff_use_lbfgs,
            mmff_variant=mmff_variant,
            fourth_dim_weight=fourth_dim_weight,
            chiral_weight=chiral_weight,
            mmff_cache=mmff_cache,
        )
        for mol_idx, pos_3d, energy, converged, applied in chunk_results:
            mol_results[mol_idx].positions_3d.append(pos_3d)
            mol_results[mol_idx].energies.append(energy)
            mol_results[mol_idx].converged.append(converged)
            mol_results[mol_idx].mmff_applied = applied
        total_confs += sum(c_end - c_start for _, c_start, c_end in chunk)

    for mol_idx, (_, err) in mmff_cache.items():
        mol_results[mol_idx].mmff_error = err

    return PipelineResult(
        molecules=mol_results,
        total_conformers=total_confs,
        total_time=time.time() - t_start,
        n_batches=len(chunks),
    )
