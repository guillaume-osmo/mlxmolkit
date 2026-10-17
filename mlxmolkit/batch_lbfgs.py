"""Batched L-BFGS: B independent minimisations, one batched evaluation per step.

Each system in the batch runs its own L-BFGS — its own history, its own line
search, its own step acceptance and its own stopping point — while every loop
iteration makes exactly one call to the user's batched ``fun(X) -> (E, G)``.
A system that converges is frozen where it passed the gradient test; one
whose line search is still backtracking does not hold back the others, and
one that accepted a step does not wait for them. The only thing shared is
the evaluation call, which is what makes batching pay on a GPU.

This is the piece the older multi-molecule optimiser lacked: it advanced all
conformers in lockstep with a capped step and no line search, so it could
only stop when the slowest conformer did and could not reject a bad step.

Precision. Everything is float32, where an energy near a minimum carries
noise from cancelling terms (MMFF: 2e-6 to 6e-5 kcal/mol absolute, up to
1e-5 relative, measured against float64). That is larger than the decrease
Armijo asks for close to convergence, so a pure Armijo line search stalls on
noise. A step is therefore also accepted under Hager & Zhang's approximate
Wolfe condition (SIAM J. Optim. 16, 170 (2005)): energy within the noise
floor ``f_rel_noise * |E| + f_abs_noise`` and the directional derivative,
which float32 resolves much better than an energy difference, no longer
strongly positive. Convergence is tested on the gradient only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import mlx.core as mx
import numpy as np

CONVERGED, MAX_ITERS, LINE_SEARCH_FAILED = 0, 1, 2
_TINY = 1e-30


@dataclass
class BatchLBFGSResult:
    """Per-system outcome of :func:`lbfgs_batch`.

    ``x``/``energy``/``grad`` are each system's last accepted iterate — for a
    converged system, the one that passed the gradient test — so
    ``converged`` is exactly ``grad_norm <= grad_tol``. ``status`` is
    CONVERGED, MAX_ITERS (``max_iters`` accepted steps, or the evaluation
    budget ran out first) or LINE_SEARCH_FAILED (no acceptable step even
    along steepest descent: the precision floor). ``n_iters`` counts accepted
    steps per system; ``n_evals`` the batched evaluations, shared by all.
    """

    x: np.ndarray
    energy: np.ndarray
    grad: np.ndarray
    grad_norm: np.ndarray
    converged: np.ndarray
    status: np.ndarray
    n_iters: np.ndarray
    n_evals: int


def _rowdot(a: mx.array, b: mx.array) -> mx.array:
    return mx.sum(a * b, axis=-1)


def lbfgs_batch(
    fun: Callable[[mx.array], tuple[mx.array, mx.array]],
    x0,
    *,
    mask=None,
    m: int = 10,
    max_iters: int = 500,
    grad_tol: float = 1e-5,
    grad_norm: Optional[Callable[[mx.array], mx.array]] = None,
    max_step: Optional[float] = None,
    atom_dim: Optional[int] = None,
    c1: float = 1e-4,
    wolfe_delta: float = 0.1,
    f_rel_noise: float = 1e-4,
    f_abs_noise: float = 0.0,
    curvature_eps: float = 1e-10,
    max_linesearch: int = 30,
    max_evals: Optional[int] = None,
    check_every: int = 5,
) -> BatchLBFGSResult:
    """Minimise B independent functions with one batched call per iteration.

    Args:
        fun: ``X (B, D) -> (E (B,), G (B, D))``, float32 MLX arrays.
        x0: ``(B, D)`` starting points.
        mask: optional ``(B, D)``, nonzero on real coordinates. Padded
            coordinates get zero gradient whatever ``fun`` returns there
            (``where``, not a multiply, so NaN in padding cannot leak) and so
            never move.
        m: history pairs per system.
        max_iters: accepted steps per system.
        grad_tol: convergence threshold on ``grad_norm(G)``.
        grad_norm: ``G (B, D) -> (B,)``; default the largest absolute
            component. Use the per-atom norm for molecules.
        max_step: cap on a step's size — the largest per-atom displacement
            if ``atom_dim`` is given, else the step's Euclidean norm. The cap
            scales the whole step, so it stays a descent direction.
        atom_dim: coordinates per atom (3) for the per-atom cap.
        c1: Armijo constant.
        wolfe_delta: approximate-Wolfe delta; accepts a step whose
            directional derivative is at most ``(1 - 2*delta) * |g.d|``.
        f_rel_noise, f_abs_noise: energy noise floor used by the
            approximate-Wolfe test (see module docstring).
        curvature_eps: a pair is stored only if ``s.y > eps * y.y``.
        max_linesearch: backtracks before the history is dropped and the
            system restarts along steepest descent; a second exhaustion with
            no history stops it with LINE_SEARCH_FAILED.
        max_evals: batched evaluations allowed (default ``4 * max_iters``).
        check_every: iterations between host checks for "all finished".
    """
    X = mx.array(x0, dtype=mx.float32)
    B, D = X.shape
    mask_b = None if mask is None else mx.array(mask) != 0
    max_evals = 4 * max_iters if max_evals is None else max_evals
    if grad_norm is None:
        grad_norm = lambda g: mx.max(mx.abs(g), axis=-1)  # noqa: E731

    def evaluate(x):
        e, g = fun(x)
        e = mx.reshape(e, (B,)).astype(mx.float32)
        g = mx.reshape(g, (B, D)).astype(mx.float32)
        if mask_b is not None:
            g = mx.where(mask_b, g, 0.0)
        return e, g

    def max_disp(d):
        if atom_dim:
            return mx.max(mx.sqrt(mx.sum(mx.reshape(d, (B, -1, atom_dim)) ** 2, -1)), axis=-1)
        return mx.sqrt(_rowdot(d, d))

    def direction(G, S, Y, rho, n_hist):
        """Two-loop recursion, each row using only its own n_hist pairs."""
        q = G
        alphas = []
        for j in range(m):  # newest first
            a = mx.where(j < n_hist, rho[:, j] * _rowdot(S[:, j], q), 0.0)
            q = q - a[:, None] * Y[:, j]
            alphas.append(a)
        sy, yy = _rowdot(S[:, 0], Y[:, 0]), _rowdot(Y[:, 0], Y[:, 0])
        gamma = mx.where(n_hist > 0, sy / mx.maximum(yy, _TINY), 1.0)
        r = gamma[:, None] * q
        for j in reversed(range(m)):
            b = rho[:, j] * _rowdot(Y[:, j], r)
            r = r + mx.where((j < n_hist)[:, None], S[:, j] * (alphas[j] - b)[:, None], 0.0)
        d = -r
        gd = _rowdot(G, d)
        # Not a descent direction (or not finite): drop the history.
        bad = ~(gd < 0)
        d = mx.where(bad[:, None], -G, d)
        n_hist = mx.where(bad, 0, n_hist)
        gd = _rowdot(G, d)
        # Steepest descent starts at a unit-gradient step; a quasi-Newton
        # step at 1. Either way the cap scales the whole step.
        alpha = mx.where(n_hist > 0, 1.0, 1.0 / mx.maximum(mx.sqrt(_rowdot(G, G)), 1.0))
        if max_step is not None:
            alpha = mx.minimum(alpha, max_step / mx.maximum(max_disp(d), _TINY))
        return d, gd, alpha, n_hist

    def update(X, E, G, S, Y, rho, n_hist, d, gd, alpha, ls, done, status, iters, x_t, E_t, G_t):
        active = ~done
        finite = mx.isfinite(E_t) & mx.all(mx.isfinite(G_t), axis=-1)
        noise = f_rel_noise * mx.abs(E) + f_abs_noise
        armijo = E_t <= E + c1 * alpha * gd
        approx_wolfe = (E_t <= E + noise) & (_rowdot(G_t, d) <= (2.0 * wolfe_delta - 1.0) * gd)
        accept = active & finite & (armijo | approx_wolfe)
        reject = active & ~accept

        # History: newest pair at slot 0, only for rows that accepted with
        # positive curvature.
        s, y = x_t - X, G_t - G
        sy = _rowdot(s, y)
        store = accept & (sy > curvature_eps * _rowdot(y, y))
        roll = lambda h, new: mx.concatenate([new[:, None], h[:, :-1]], axis=1)  # noqa: E731
        S = mx.where(store[:, None, None], roll(S, s), S)
        Y = mx.where(store[:, None, None], roll(Y, y), Y)
        rho = mx.where(store[:, None], roll(rho, 1.0 / mx.where(store, sy, 1.0)), rho)
        n_hist = mx.where(store, mx.minimum(n_hist + 1, m), n_hist)

        X = mx.where(accept[:, None], x_t, X)
        E = mx.where(accept, E_t, E)
        G = mx.where(accept[:, None], G_t, G)
        iters = iters + accept.astype(mx.int32)

        converged = accept & (grad_norm(G) <= grad_tol)
        out_of_iters = accept & ~converged & (iters >= max_iters)

        # Backtrack: safeguarded quadratic interpolation, else halve.
        denom = 2.0 * (E_t - E - gd * alpha)
        a_q = mx.where(finite & (denom > 0), -gd * alpha * alpha / mx.maximum(denom, _TINY), 0.1 * alpha)
        alpha_bt = mx.clip(a_q, 0.1 * alpha, 0.5 * alpha)
        ls = mx.where(reject, ls + 1, mx.where(accept, 0, ls))
        exhausted = reject & (ls >= max_linesearch)
        failed = exhausted & (n_hist == 0)
        restart = exhausted & (n_hist > 0)
        n_hist = mx.where(restart, 0, n_hist)
        ls = mx.where(restart, 0, ls)

        status = mx.where(converged, CONVERGED, mx.where(out_of_iters, MAX_ITERS, mx.where(failed, LINE_SEARCH_FAILED, status)))
        done = done | converged | out_of_iters | failed

        need_dir = ~done & (accept | restart)
        d_new, gd_new, alpha_new, n_hist_new = direction(G, S, Y, rho, n_hist)
        d = mx.where(need_dir[:, None], d_new, d)
        gd = mx.where(need_dir, gd_new, gd)
        n_hist = mx.where(need_dir, n_hist_new, n_hist)
        alpha = mx.where(done, 0.0, mx.where(need_dir, alpha_new, mx.where(reject, alpha_bt, alpha)))
        return X, E, G, S, Y, rho, n_hist, d, gd, alpha, ls, done, status, iters

    update = mx.compile(update)

    # ---- initial state -------------------------------------------------
    E, G = evaluate(X)
    S = mx.zeros((B, m, D), dtype=mx.float32)
    Y = mx.zeros((B, m, D), dtype=mx.float32)
    rho = mx.zeros((B, m), dtype=mx.float32)
    n_hist = mx.zeros((B,), dtype=mx.int32)
    ls = mx.zeros((B,), dtype=mx.int32)
    iters = mx.zeros((B,), dtype=mx.int32)
    converged0 = grad_norm(G) <= grad_tol
    done = converged0 | ~(mx.isfinite(E) & mx.all(mx.isfinite(G), axis=-1))
    status = mx.where(converged0, CONVERGED, mx.full((B,), LINE_SEARCH_FAILED, dtype=mx.int32))
    status = mx.where(done, status, MAX_ITERS)  # placeholder for rows still running
    d, gd, alpha, n_hist = direction(G, S, Y, rho, n_hist)
    alpha = mx.where(done, 0.0, alpha)
    mx.eval(X, E, G, d, gd, alpha, done, status)

    n_evals = 1
    while n_evals < max_evals:
        x_t = X + alpha[:, None] * d
        E_t, G_t = evaluate(x_t)
        n_evals += 1
        (X, E, G, S, Y, rho, n_hist, d, gd, alpha, ls, done, status, iters) = update(
            X, E, G, S, Y, rho, n_hist, d, gd, alpha, ls, done, status, iters, x_t, E_t, G_t
        )
        # Queue this iteration without waiting for it, so the CPU builds the
        # next one while the GPU runs; block only to check for completion.
        mx.async_eval(X, E, G, S, Y, rho, n_hist, d, gd, alpha, ls, done, status, iters)
        if n_evals % check_every == 0 and bool(mx.all(done)):
            break

    gn = grad_norm(G)
    mx.eval(gn)
    # Rows still running when the budget ran out keep the MAX_ITERS placeholder.
    return BatchLBFGSResult(
        x=np.array(X),
        energy=np.array(E),
        grad=np.array(G),
        grad_norm=np.array(gn),
        converged=np.array(status) == CONVERGED,
        status=np.array(status),
        n_iters=np.array(iters),
        n_evals=n_evals,
    )
