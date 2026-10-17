"""
Batched MMFF L-BFGS optimizer using the fused Metal kernel.

Modes:
  1. Single molecule, many conformers: mmff_optimize_metal_batch()
  2. Many molecules, Python L-BFGS:    mmff_optimize_metal_multi_mol()
     (lockstep, capped steps, no line search)
  2b. Many molecules, per-conformer L-BFGS with line search:
                                       mmff_optimize_metal_lbfgs()
  3. Many molecules, fused GPU opt:    mmff_optimize_metal_fused_multi_mol()

Mode 3 runs the entire optimization in a single Metal kernel launch —
each GPU thread runs all iterations of steepest descent internally.
Zero Python loop overhead.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import numpy as np
from rdkit import Chem

from mlxmolkit.mmff_metal_kernel import (
    mmff_energy_grad_metal,
    mmff_energy_grad_metal_mega,
    mmff_optimize_metal_fused,
    pack_multi_mol_for_metal,
    pack_params_for_metal,
)
from mlxmolkit.mmff_params import extract_mmff_params

EPS_CURV = 1e-10


@dataclass
class MetalBatchResult:
    """Result of fused Metal MMFF optimization."""

    energies: np.ndarray
    positions: np.ndarray
    grad_norms: np.ndarray
    n_iters: int
    n_conformers: int
    per_conf_iters: Optional[np.ndarray] = None
    converged: Optional[np.ndarray] = None


def _lbfgs_direction_batch(
    grad: mx.array,
    s_hist: list[mx.array],
    y_hist: list[mx.array],
    rho_hist: list[mx.array],
) -> mx.array:
    """Vectorized L-BFGS two-loop recursion for C conformers."""
    m = len(s_hist)
    q = grad
    alphas = []
    for i in range(m - 1, -1, -1):
        a_i = rho_hist[i] * mx.sum(s_hist[i] * q, axis=1, keepdims=True)
        q = q - a_i * y_hist[i]
        alphas.append(a_i)

    if m > 0:
        sy = mx.sum(s_hist[-1] * y_hist[-1], axis=1, keepdims=True)
        yy = mx.sum(y_hist[-1] * y_hist[-1], axis=1, keepdims=True) + 1e-30
        gamma = sy / yy
    else:
        gamma = mx.array(1.0)

    r = gamma * q
    for i in range(m):
        beta = rho_hist[i] * mx.sum(y_hist[i] * r, axis=1, keepdims=True)
        r = r + s_hist[i] * (alphas[m - 1 - i] - beta)

    return -r


def mmff_optimize_metal_batch(
    mol: Chem.Mol,
    conf_ids: Optional[list[int]] = None,
    max_iters: int = 200,
    eval_freq: int = 10,
    max_step_per_atom: float = 0.3,
    lbfgs_m: int = 10,
    grad_scale: float = 0.1,
) -> MetalBatchResult:
    """
    Batched L-BFGS MMFF optimization using the fused Metal kernel.

    Args:
        mol: RDKit molecule with embedded conformers.
        conf_ids: conformer IDs to optimize (default: all).
        max_iters: maximum L-BFGS iterations.
        eval_freq: GPU sync frequency (10 = sync every 10 iters).
        max_step_per_atom: max displacement per atom per step (Angstrom).
        lbfgs_m: L-BFGS history pairs.
        grad_scale: gradient scaling factor.

    Returns:
        MetalBatchResult with final energies, positions, grad norms.
    """
    n_atoms = mol.GetNumAtoms()
    dim = n_atoms * 3

    if conf_ids is None:
        conf_ids = [c.GetId() for c in mol.GetConformers()]
    C = len(conf_ids)

    # --- Extract and pack params for Metal ---
    np_params = extract_mmff_params(mol, conf_id=conf_ids[0])
    idx_buf, param_buf, meta = pack_params_for_metal(np_params)

    pos_np = np.zeros((C, n_atoms, 3), dtype=np.float32)
    for k, cid in enumerate(conf_ids):
        conf = mol.GetConformer(cid)
        for i in range(n_atoms):
            p = conf.GetAtomPosition(i)
            pos_np[k, i] = [p.x, p.y, p.z]

    # --- Flat positions for optimizer ---
    x = mx.array(pos_np.reshape(C, dim))
    mx.eval(x)

    # Initial energy + gradient via Metal kernel
    pos3d = mx.reshape(x, (C, n_atoms, 3))
    e_init, g = mmff_energy_grad_metal(idx_buf, param_buf, meta, pos3d)
    mx.eval(e_init, g)
    g_sc = g * grad_scale

    # Track best-known energy/positions (monotone guard)
    best_e = e_init  # (C,)
    best_x = x  # (C, dim)

    # L-BFGS state
    s_hist: list[mx.array] = []
    y_hist: list[mx.array] = []
    rho_hist: list[mx.array] = []
    max_step = mx.array(max_step_per_atom, dtype=mx.float32)
    gs = mx.array(grad_scale, dtype=mx.float32)
    one = mx.array(1.0, dtype=mx.float32)

    # --- Optimization loop with energy guard ---
    for it in range(max_iters):
        d = _lbfgs_direction_batch(g_sc, s_hist, y_hist, rho_hist)

        d3 = mx.reshape(d, (C, n_atoms, 3))
        atom_disp = mx.sqrt(mx.sum(d3 * d3, axis=-1, keepdims=True) + 1e-30)
        d = mx.reshape(d3 * mx.minimum(max_step / atom_disp, one), (C, dim))

        x_new = x + d

        pos3d_new = mx.reshape(x_new, (C, n_atoms, 3))
        e_new, g_new = mmff_energy_grad_metal(idx_buf, param_buf, meta, pos3d_new)
        g_new_sc = g_new * gs

        s_k = x_new - x
        y_k = g_new_sc - g_sc
        sy = mx.sum(s_k * y_k, axis=1, keepdims=True)
        rho_k = mx.where(sy > EPS_CURV, 1.0 / (sy + 1e-30), mx.zeros_like(sy))

        if len(s_hist) >= lbfgs_m:
            s_hist.pop(0)
            y_hist.pop(0)
            rho_hist.pop(0)
        s_hist.append(s_k)
        y_hist.append(y_k)
        rho_hist.append(rho_k)

        x = x_new
        g_sc = g_new_sc

        if (it + 1) % eval_freq == 0:
            mx.eval(x, g_sc, e_new, *s_hist, *y_hist, *rho_hist)
            # Monotone guard: remember best positions per conformer
            improved = mx.expand_dims(e_new < best_e, axis=1)
            best_x = mx.where(improved, x, best_x)
            best_e = mx.minimum(e_new, best_e)
            mx.eval(best_x, best_e)

    # --- Final: use best-seen positions ---
    mx.eval(best_x)
    pos3d_final = mx.reshape(best_x, (C, n_atoms, 3))
    final_e, _ = mmff_energy_grad_metal(idx_buf, param_buf, meta, pos3d_final)
    g_norms = mx.sqrt(mx.sum(g_sc * g_sc, axis=1))
    mx.eval(final_e, g_norms)

    final_pos = np.array(best_x).reshape(C, n_atoms, 3)
    final_energies = np.array(final_e)
    final_gnorms = np.array(g_norms)

    # Write back to RDKit
    for k, cid in enumerate(conf_ids):
        conf = mol.GetConformer(cid)
        for i in range(n_atoms):
            conf.SetAtomPosition(i, final_pos[k, i].astype(float).tolist())

    return MetalBatchResult(
        energies=final_energies,
        positions=final_pos,
        grad_norms=final_gnorms,
        n_iters=max_iters,
        n_conformers=C,
    )


# ---------------------------------------------------------------------------
# Multi-molecule parallel optimisation via mega-kernel
# ---------------------------------------------------------------------------


@dataclass
class MultiMolResult:
    """Result of multi-molecule parallel MMFF optimisation."""

    mol_results: list[MetalBatchResult]


@dataclass
class _PaddedBatch:
    """Conformers of many molecules on one padded (total_confs, max_dim) grid."""

    mols: list
    conf_ids: list
    conf_counts: list
    total_confs: int
    max_dim: int
    kernel_args: tuple  # idx_buf, param_buf, all_meta, dispatch, pos_offsets
    pos: np.ndarray  # (total_confs, max_dim) float32, zero padding


def _padded_batch(mol_specs) -> _PaddedBatch:
    mols, all_conf_ids, params_list, conf_counts = [], [], [], []
    for mol, conf_ids in mol_specs:
        if conf_ids is None:
            conf_ids = [c.GetId() for c in mol.GetConformers()]
        mols.append(mol)
        all_conf_ids.append(conf_ids)
        params_list.append(extract_mmff_params(mol, conf_id=conf_ids[0]))
        conf_counts.append(len(conf_ids))

    total_confs = sum(conf_counts)
    max_dim = max(p.n_atoms * 3 for p in params_list)
    idx_buf, param_buf, all_meta, dispatch, _, _ = pack_multi_mol_for_metal(
        params_list, conf_counts
    )
    # Padded stride cid * max_dim instead of the packed offsets.
    pos_offsets = mx.array(np.arange(total_confs, dtype=np.uint32) * max_dim)

    pos = np.zeros((total_confs, max_dim), dtype=np.float32)
    row = 0
    for mol, cids in zip(mols, all_conf_ids):
        for cid in cids:
            xyz = mol.GetConformer(cid).GetPositions().reshape(-1)
            pos[row, : xyz.size] = xyz
            row += 1
    return _PaddedBatch(
        mols, all_conf_ids, conf_counts, total_confs, max_dim,
        (idx_buf, param_buf, all_meta, dispatch, pos_offsets), pos,
    )


def _write_back(pb: _PaddedBatch, x_np, e_np, gn_np, n_iters, per_conf_iters, converged) -> MultiMolResult:
    """Split per-conformer results by molecule and write positions to RDKit."""
    results: list[MetalBatchResult] = []
    conf_cursor = 0
    for m_idx, mol in enumerate(pb.mols):
        C = pb.conf_counts[m_idx]
        n_atoms = mol.GetNumAtoms()
        sl = slice(conf_cursor, conf_cursor + C)
        mol_pos = x_np[sl, : n_atoms * 3].reshape(C, n_atoms, 3)
        for k, cid in enumerate(pb.conf_ids[m_idx]):
            conf = mol.GetConformer(cid)
            for a in range(n_atoms):
                conf.SetAtomPosition(a, mol_pos[k, a].astype(float).tolist())
        results.append(
            MetalBatchResult(
                energies=e_np[sl],
                positions=mol_pos,
                grad_norms=gn_np[sl],
                n_iters=n_iters,
                n_conformers=C,
                per_conf_iters=per_conf_iters[sl].copy(),
                converged=converged[sl].copy(),
            )
        )
        conf_cursor += C
    return MultiMolResult(mol_results=results)


def mmff_optimize_metal_multi_mol(
    mol_specs: list[tuple[Chem.Mol, Optional[list[int]]]],
    max_iters: int = 200,
    eval_freq: int = 10,
    max_step_per_atom: float = 0.3,
    lbfgs_m: int = 10,
    grad_scale: float = 0.1,
    grad_tol: float = 1e-4,
) -> MultiMolResult:
    """
    Optimise many molecules simultaneously in a **single Metal kernel launch**
    with padded positions for vectorised L-BFGS.

    Architecture:
      - Positions padded to max_dim so all conformers form a clean
        (total_confs, max_dim) matrix.
      - One mega-kernel call per iteration.
      - Batched L-BFGS direction via _lbfgs_direction_batch() on the full
        (total_confs, max_dim) matrix — zero Python loops per iteration.
      - Padded elements carry zero gradient (the mega-kernel zero-fills its
        output) so they don't influence L-BFGS.

    Args:
        mol_specs: list of (mol, conf_ids) tuples.  If conf_ids is None
                   every embedded conformer is optimised.
        max_iters, eval_freq, max_step_per_atom, lbfgs_m, grad_scale:
            same semantics as mmff_optimize_metal_batch().
        grad_tol: convergence tolerance on the largest per-atom gradient
            norm.  Applied to the same scaled gradient the result reports as
            ``grad_norms``, so the two always agree; note that means
            ``grad_scale`` shifts it.  Checked at the existing eval_freq
            sync points, so it costs no extra GPU->CPU round trips; the
            loop exits as soon as every conformer is under it.

    Returns:
        MultiMolResult containing per-molecule MetalBatchResult objects.
        A conformer that meets ``grad_tol`` is frozen there: its returned
        positions are the ones that passed the check. The others return
        their lowest-energy positions. ``grad_norms`` and energies are
        evaluated at the returned positions, so ``converged`` is exactly
        ``grad_norms <= grad_tol``. ``per_conf_iters`` is the iteration at
        which a conformer converged, counting the polishing pass after the
        main loop (so it can exceed ``max_iters``), or ``max_iters`` if it
        never did.

    Note:
        Because the batch advances in lockstep, the loop can only stop when
        the *slowest* conformer converges.  Iteration counts vary ~65x over
        drug-like molecules and track flexibility (rotatable bonds r=0.81,
        rings *negatively*), so one floppy chain sets the cost for the whole
        batch.  ``per_conf_iters`` is what lets a caller see that happening.
    """
    gs_val = grad_scale

    # ---- 1-2. Padded positions + mega-kernel parameters ----------------
    pb = _padded_batch(mol_specs)
    total_confs, max_dim = pb.total_confs, pb.max_dim
    max_atoms = max_dim // 3
    total_padded = total_confs * max_dim
    idx_buf, param_buf, all_meta, dispatch, pos_offsets = pb.kernel_args
    pos_np = pb.pos

    # ---- 3. Initial energy + gradient ----------------------------------
    x = mx.array(pos_np.ravel())
    e_init, g = mmff_energy_grad_metal_mega(
        idx_buf, param_buf, all_meta, dispatch, pos_offsets,
        x, total_confs, total_padded,
    )
    x = mx.reshape(x, (total_confs, max_dim))
    g_sc = mx.reshape(g, (total_confs, max_dim)) * gs_val
    mx.eval(x, g_sc, e_init)

    # Track best-known energy and positions for monotone guarantee
    best_e = e_init  # (total_confs,)
    best_x = x  # (total_confs, max_dim)

    max_step_mx = mx.array(max_step_per_atom, dtype=mx.float32)
    one_f = mx.array(1.0, dtype=mx.float32)

    s_hist: list[mx.array] = []
    y_hist: list[mx.array] = []
    rho_hist: list[mx.array] = []

    # Iteration at which each conformer first met grad_tol; max_iters means
    # "never did".  Filled in at the sync points below.
    per_conf_iters = np.full(total_confs, max_iters, dtype=np.int32)
    conv_mask = np.zeros(total_confs, dtype=bool)
    iters_used = max_iters

    def _max_grad_norm(g_scaled: mx.array) -> np.ndarray:
        """Largest per-atom gradient norm per conformer.

        Takes the *scaled* gradient, so this is the same quantity the result
        reports as ``grad_norms`` — otherwise a conformer could come back
        with grad_norms below grad_tol and converged=False.  Padded atoms
        carry zero gradient and so never set the maximum.
        """
        gn = mx.max(
            mx.sqrt(
                mx.sum(
                    mx.reshape(g_scaled, (total_confs, max_atoms, 3)) ** 2,
                    axis=-1,
                )
            ),
            axis=1,
        )
        mx.eval(gn)
        return np.array(gn)

    def _sync(x, g_scaled, e, best_x, best_e, iteration):
        """Freeze newly converged conformers, then apply the energy guard.

        A conformer that meets grad_tol is frozen at *these* positions — the
        ones that just passed the check — even if an earlier snapshot had a
        lower energy: near a minimum float32 energies differ by noise, and
        the lowest-energy snapshot could carry a gradient several times
        grad_tol. Frozen conformers take no further steps and are never
        replaced by the guard, so converged <=> returned grad_norms <=
        grad_tol. Every best_x snapshot passes through here, which is what
        makes that hold for the unconverged ones too.
        """
        newly = (~conv_mask) & (_max_grad_norm(g_scaled) <= grad_tol)
        was_frozen = mx.array(conv_mask)[:, None]
        per_conf_iters[newly] = iteration
        conv_mask[newly] = True
        take = mx.array(newly)[:, None] | (
            ~was_frozen & mx.expand_dims(e < best_e, axis=1)
        )
        best_x = mx.where(take, x, best_x)
        best_e = mx.where(take[:, 0], e, best_e)
        mx.eval(best_x, best_e)
        return best_x, best_e, mx.array(conv_mask)[:, None]

    def _capped_step(g_scaled, s_hist, y_hist, rho_hist, frozen):
        d = _lbfgs_direction_batch(g_scaled, s_hist, y_hist, rho_hist)
        d3 = mx.reshape(d, (total_confs, max_atoms, 3))
        atom_disp = mx.sqrt(
            mx.sum(d3 * d3, axis=-1, keepdims=True) + 1e-30
        )
        d = mx.reshape(
            d3 * mx.minimum(max_step_mx / atom_disp, one_f),
            (total_confs, max_dim),
        )
        return mx.where(frozen, 0.0, d)

    # A start geometry already under grad_tol is returned untouched.
    best_x, best_e, frozen = _sync(x, g_sc, e_init, best_x, best_e, 0)
    if conv_mask.all():
        iters_used = 0

    # ---- 4. Optimisation loop with energy guard ------------------------
    for it in range(iters_used):
        d = _capped_step(g_sc, s_hist, y_hist, rho_hist, frozen)

        x_new = x + d

        # Single mega-kernel call: energy + gradient
        x_flat = mx.reshape(x_new, (total_padded,))
        e_new, g_new = mmff_energy_grad_metal_mega(
            idx_buf, param_buf, all_meta, dispatch, pos_offsets,
            x_flat, total_confs, total_padded,
        )
        g_new_sc = mx.reshape(g_new, (total_confs, max_dim)) * gs_val

        # L-BFGS history update
        s_k = x_new - x
        y_k = g_new_sc - g_sc
        sy = mx.sum(s_k * y_k, axis=1, keepdims=True)
        rho_k = mx.where(
            sy > EPS_CURV, 1.0 / (sy + 1e-30), mx.zeros_like(sy)
        )

        if len(s_hist) >= lbfgs_m:
            s_hist.pop(0)
            y_hist.pop(0)
            rho_hist.pop(0)
        s_hist.append(s_k)
        y_hist.append(y_k)
        rho_hist.append(rho_k)

        x = x_new
        g_sc = g_new_sc

        # Periodic sync: convergence freeze + energy guard
        if (it + 1) % eval_freq == 0:
            mx.eval(x, g_sc, e_new, *s_hist, *y_hist, *rho_hist)
            best_x, best_e, frozen = _sync(x, g_sc, e_new, best_x, best_e, it + 1)
            if conv_mask.all():
                iters_used = it + 1
                break

    # Unconverged conformers restart from their best-seen positions for a
    # short "polishing" pass (fresh L-BFGS); frozen ones stay where they are.
    # Their iteration counts carry on from the main loop.
    polish_iters = 0 if conv_mask.all() else max_iters // 4
    if polish_iters:
        x = best_x
        x_flat = mx.reshape(x, (total_padded,))
        _, g_at_best = mmff_energy_grad_metal_mega(
            idx_buf, param_buf, all_meta, dispatch, pos_offsets,
            x_flat, total_confs, total_padded,
        )
        g_sc = mx.reshape(g_at_best, (total_confs, max_dim)) * gs_val
        mx.eval(x, g_sc)
    s_hist = []
    y_hist = []
    rho_hist = []
    for it in range(polish_iters):
        d = _capped_step(g_sc, s_hist, y_hist, rho_hist, frozen)
        x_new = x + d
        x_flat = mx.reshape(x_new, (total_padded,))
        e_new, g_new = mmff_energy_grad_metal_mega(
            idx_buf, param_buf, all_meta, dispatch, pos_offsets,
            x_flat, total_confs, total_padded,
        )
        g_new_sc = mx.reshape(g_new, (total_confs, max_dim)) * gs_val

        s_k = x_new - x
        y_k = g_new_sc - g_sc
        sy = mx.sum(s_k * y_k, axis=1, keepdims=True)
        rho_k = mx.where(
            sy > EPS_CURV, 1.0 / (sy + 1e-30), mx.zeros_like(sy)
        )
        if len(s_hist) >= lbfgs_m:
            s_hist.pop(0)
            y_hist.pop(0)
            rho_hist.pop(0)
        s_hist.append(s_k)
        y_hist.append(y_k)
        rho_hist.append(rho_k)

        x = x_new
        g_sc = g_new_sc

        if (it + 1) % eval_freq == 0:
            mx.eval(x, g_sc, e_new, *s_hist, *y_hist, *rho_hist)
            best_x, best_e, frozen = _sync(
                x, g_sc, e_new, best_x, best_e, iters_used + it + 1
            )
            if conv_mask.all():
                break

    # Final: frozen positions for converged conformers, best-seen otherwise
    x = best_x

    # ---- 5. Final energies + write back --------------------------------
    # Energy and gradient both at the returned positions, so grad_norms
    # describes the geometry the caller gets back.
    mx.eval(x)
    x_flat = mx.reshape(x, (total_padded,))
    final_e, final_g = mmff_energy_grad_metal_mega(
        idx_buf, param_buf, all_meta, dispatch, pos_offsets,
        x_flat, total_confs, total_padded,
    )
    g_norms_3d = mx.reshape(final_g * gs_val, (total_confs, max_atoms, 3))
    g_norms_per_atom = mx.sqrt(mx.sum(g_norms_3d ** 2, axis=-1))
    mx.eval(final_e, g_norms_per_atom)

    return _write_back(
        pb,
        np.array(x),
        np.array(final_e),
        np.max(np.array(g_norms_per_atom), axis=1),  # padded atoms are 0
        iters_used,
        per_conf_iters,
        conv_mask,
    )



def mmff_optimize_metal_lbfgs(
    mol_specs: list[tuple[Chem.Mol, Optional[list[int]]]],
    max_iters: int = 1000,
    grad_tol: float = 1e-4,
    max_step_per_atom: float = 0.3,
    lbfgs_m: int = 10,
    grad_scale: float = 0.1,
) -> MultiMolResult:
    """Optimise many molecules with an independent L-BFGS per conformer.

    Same padded layout and one mega-kernel call per iteration as
    :func:`mmff_optimize_metal_multi_mol`, but each conformer runs its own
    L-BFGS through :func:`mlxmolkit.batch_lbfgs.lbfgs_batch`: its own
    history, a line search that can reject a bad step, and its own stopping
    point. Nothing waits for the slowest conformer except the shared
    evaluation call, and nothing is accepted without a sufficient-decrease
    test (on the energy, or on the gradient once float32 energy differences
    are below noise — see that module).

    Args:
        mol_specs: list of (mol, conf_ids); conf_ids None means all.
        max_iters: accepted L-BFGS steps per conformer.
        grad_tol: on the largest per-atom gradient norm times
            ``grad_scale`` — the same quantity and default as
            :func:`mmff_optimize_metal_multi_mol`, so results compare.
        max_step_per_atom: cap on any atom's displacement per step (A); it
            scales the whole step, so the step stays a descent direction.
        lbfgs_m: history pairs per conformer.

    Returns:
        MultiMolResult. Per conformer: ``per_conf_iters`` is its accepted
        steps, ``converged`` is exactly ``grad_norms <= grad_tol``, and the
        energies and gradient norms are at the returned positions.
        ``n_iters`` is the number of batched evaluations, which every
        conformer shared.
    """
    from mlxmolkit.batch_lbfgs import lbfgs_batch

    pb = _padded_batch(mol_specs)
    C, max_dim = pb.total_confs, pb.max_dim
    total = C * max_dim

    def energy_grad(X):
        e, g = mmff_energy_grad_metal_mega(
            *pb.kernel_args, mx.reshape(X, (total,)), C, total
        )
        return e, mx.reshape(g, (C, max_dim))

    def scaled_max_atom_grad(G):
        per_atom = mx.sqrt(mx.sum(mx.reshape(G, (C, max_dim // 3, 3)) ** 2, axis=-1))
        return grad_scale * mx.max(per_atom, axis=-1)

    mask = np.zeros((C, max_dim), dtype=bool)
    row = 0
    for mol, k in zip(pb.mols, pb.conf_counts):
        mask[row : row + k, : 3 * mol.GetNumAtoms()] = True
        row += k

    res = lbfgs_batch(
        energy_grad,
        pb.pos,
        mask=mask,
        m=lbfgs_m,
        max_iters=max_iters,
        grad_tol=grad_tol,
        grad_norm=scaled_max_atom_grad,
        max_step=max_step_per_atom,
        atom_dim=3,
        # float32 MMFF energies carry up to 6e-5 kcal/mol of noise against
        # float64; |E| can be near 0, so the relative floor alone is not enough.
        f_abs_noise=1e-4,
    )
    return _write_back(
        pb, res.x, res.energy, res.grad_norm, res.n_evals, res.n_iters, res.converged
    )

# ---------------------------------------------------------------------------
# Mode 3: Fully fused GPU optimiser (zero Python loop overhead)
# ---------------------------------------------------------------------------


def mmff_optimize_metal_fused_multi_mol(
    mol_specs: list[tuple[Chem.Mol, Optional[list[int]]]],
    max_iters: int = 500,
    max_step: float = 0.3,
    grad_lr: float = 0.1,
) -> MultiMolResult:
    """
    Optimise many molecules in a **single Metal kernel launch**.

    The entire optimisation loop (energy → gradient → SD step × max_iters)
    runs inside the GPU kernel.  Python only does parameter extraction
    and packing (once), then a single kernel call, then reads back results.

    Steepest descent converges slower than L-BFGS so default max_iters
    is higher (500 vs 200).

    Args:
        mol_specs: list of (mol, conf_ids) tuples.
        max_iters: gradient descent iterations (default 500).
        max_step: max per-atom displacement per step (Angstrom).
        grad_lr: gradient learning rate (step = min(max_step/gnorm, grad_lr)).

    Returns:
        MultiMolResult containing per-molecule MetalBatchResult objects.
    """
    mols: list[Chem.Mol] = []
    all_conf_ids: list[list[int]] = []
    params_list: list = []
    conf_counts: list[int] = []

    for mol, conf_ids in mol_specs:
        if conf_ids is None:
            conf_ids = [c.GetId() for c in mol.GetConformers()]
        mols.append(mol)
        all_conf_ids.append(conf_ids)
        params_list.append(extract_mmff_params(mol, conf_id=conf_ids[0]))
        conf_counts.append(len(conf_ids))

    total_confs = sum(conf_counts)
    dims_per_mol = [p.n_atoms * 3 for p in params_list]
    max_dim = max(dims_per_mol)
    total_padded = total_confs * max_dim

    # Pack parameters (mega-kernel dispatch table)
    (
        idx_buf, param_buf, all_meta, dispatch, _, _,
    ) = pack_multi_mol_for_metal(params_list, conf_counts)

    # Padded position offsets
    pos_offsets = mx.array(
        np.arange(total_confs, dtype=np.uint32) * max_dim
    )

    # Fill padded position buffer
    pos_np = np.zeros((total_confs, max_dim), dtype=np.float32)
    ci = 0
    for m_idx, (mol, cids) in enumerate(zip(mols, all_conf_ids)):
        n_atoms = mol.GetNumAtoms()
        for cid in cids:
            conf = mol.GetConformer(cid)
            for a in range(n_atoms):
                p = conf.GetAtomPosition(a)
                pos_np[ci, a * 3] = p.x
                pos_np[ci, a * 3 + 1] = p.y
                pos_np[ci, a * 3 + 2] = p.z
            ci += 1

    pos_flat = mx.array(pos_np.ravel())

    # ONE kernel call — entire optimization
    pos_out, energy_out, _ = mmff_optimize_metal_fused(
        idx_buf, param_buf, all_meta, dispatch, pos_offsets,
        pos_flat, total_confs, total_padded,
        max_iters=max_iters, max_step=max_step, grad_lr=grad_lr,
    )
    mx.eval(pos_out, energy_out)

    pos_out_np = np.array(pos_out).reshape(total_confs, max_dim)
    e_np = np.array(energy_out)

    # Unpack results per molecule and write back to RDKit
    results: list[MetalBatchResult] = []
    conf_cursor = 0
    for m_idx, mol in enumerate(mols):
        C = conf_counts[m_idx]
        n_atoms = mol.GetNumAtoms()

        mol_pos = pos_out_np[
            conf_cursor : conf_cursor + C, : n_atoms * 3
        ].reshape(C, n_atoms, 3)
        mol_e = e_np[conf_cursor : conf_cursor + C]

        for k, cid in enumerate(all_conf_ids[m_idx]):
            conf = mol.GetConformer(cid)
            for a in range(n_atoms):
                conf.SetAtomPosition(
                    a, mol_pos[k, a].astype(float).tolist()
                )

        results.append(
            MetalBatchResult(
                energies=mol_e,
                positions=mol_pos,
                grad_norms=mol_e * 0,  # not tracked in fused kernel
                n_iters=max_iters,
                n_conformers=C,
            )
        )
        conf_cursor += C

    return MultiMolResult(mol_results=results)
