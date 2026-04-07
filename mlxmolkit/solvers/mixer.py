"""
Portable fixed-point mixer: Anderson / Pulay-DIIS / periodic Pulay.

Works with numpy or MLX arrays. Treats any fixed-point problem:

    x_{k+1} = G(x_k),   r_k = G(x_k) - x_k

The mixer accelerates convergence by combining recent history of
(x, r) pairs via least-squares extrapolation.

Three methods:
  - "linear":         x_{k+1} = x_k + β·P⁻¹·r_k
  - "anderson":       Type-II Anderson in delta form (nonlinear Krylov)
  - "periodic_pulay": Pulay/DIIS every p steps, damped mixing in between

All methods support:
  - Ridge-regularized least-squares
  - Step-norm clipping
  - Residual-growth restart
  - Optional preconditioner P⁻¹
  - History size control

Usage:
    mixer = SCFMixer(MixerConfig(method="periodic_pulay", beta=0.3))
    for k in range(max_iter):
        Gx = scf_map(x)
        x, info = mixer.step(x, Gx)
        if info["rnorm"] < tol:
            break

References:
    Walker & Ni, SIAM J. Numer. Anal. 49(4), 2011 — Anderson acceleration
    Pulay, Chem. Phys. Lett. 73(2), 1980 — DIIS
    Banerjee et al., Chem. Phys. Lett. 985, 1985 — Periodic Pulay
    Pratapa et al., Chem. Phys. Lett. 635, 2015 — Periodic Pulay for DFT
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class MixerConfig:
    """Configuration for the fixed-point mixer.

    Args:
        method: "linear", "anderson", or "periodic_pulay"
        beta: base mixing/damping parameter (0 < β ≤ 1)
        history_size: max number of (x, r) pairs to keep
        pulay_period: apply Pulay every p steps (only for periodic_pulay)
        start_after: begin acceleration after this many iterations
        ridge: Tikhonov regularization λ for LS solve
        max_step_factor: clip step to max_step_factor × ‖r‖
        restart_threshold: restart history if ‖r_{k+1}‖ > threshold × ‖r_k‖
        symmetrize: enforce (x + xᵀ)/2 after update (for density matrices)
    """
    method: str = "periodic_pulay"
    beta: float = 0.3
    history_size: int = 8
    pulay_period: int = 2
    start_after: int = 2
    ridge: float = 1e-10
    max_step_factor: float = 3.0
    restart_threshold: float = 2.0
    symmetrize: bool = False


class SCFMixer:
    """Portable fixed-point mixer for SCF and other iterative problems.

    Accepts numpy arrays of any shape. Internally flattens to 1D for
    the linear algebra, then reshapes back.
    """

    def __init__(
        self,
        config: MixerConfig | None = None,
        preconditioner: Callable[[np.ndarray], np.ndarray] | None = None,
    ):
        self.cfg = config or MixerConfig()
        self.precond = preconditioner  # P⁻¹(r) → preconditioned residual
        self.reset()

    def reset(self):
        """Clear all history. Call between independent problems."""
        self.iteration = 0
        self._x_hist: list[np.ndarray] = []  # flattened x vectors
        self._z_hist: list[np.ndarray] = []  # flattened preconditioned residuals
        self._rnorms: list[float] = []

    def step(
        self,
        x_k: np.ndarray,
        Gx_k: np.ndarray,
    ) -> tuple[np.ndarray, dict]:
        """Propose next iterate from current state and fixed-point map output.

        Args:
            x_k: current state (any shape)
            Gx_k: G(x_k), the unmixed next state from the fixed-point map

        Returns:
            (x_{k+1}, info_dict) where info contains:
                rnorm: residual norm
                method_used: "linear", "anderson", or "pulay"
                restarted: whether history was cleared
                history_len: current history size
                iteration: mixer iteration count
        """
        shape = x_k.shape
        x = x_k.ravel()
        Gx = Gx_k.ravel()

        # Residual: r = G(x) - x
        r = Gx - x
        rnorm = float(np.sqrt(np.mean(r * r)))

        # Precondition
        z = self.precond(r) if self.precond is not None else r

        info = {
            "rnorm": rnorm,
            "iteration": self.iteration,
            "history_len": len(self._x_hist),
            "method_used": "linear",
            "restarted": False,
        }

        # Residual-growth restart
        if (self._rnorms and rnorm > self.cfg.restart_threshold * self._rnorms[-1]
                and len(self._x_hist) > 2):
            self._x_hist.clear()
            self._z_hist.clear()
            info["restarted"] = True

        # Store in history
        self._x_hist.append(x.copy())
        self._z_hist.append(z.copy())
        if len(self._x_hist) > self.cfg.history_size + 1:
            self._x_hist.pop(0)
            self._z_hist.pop(0)

        # Decide method
        can_accel = (self.iteration >= self.cfg.start_after and len(self._x_hist) >= 2)

        if not can_accel or self.cfg.method == "linear":
            x_new = self._linear(x, z)
            info["method_used"] = "linear"

        elif self.cfg.method == "anderson":
            x_new = self._anderson(x, z)
            info["method_used"] = "anderson"

        elif self.cfg.method == "periodic_pulay":
            if self.iteration % self.cfg.pulay_period == 0:
                x_new = self._pulay(x, z)
                info["method_used"] = "pulay"
            else:
                x_new = self._linear(x, z)
                info["method_used"] = "linear"
        else:
            raise ValueError(f"Unknown method: {self.cfg.method}")

        self._rnorms.append(rnorm)
        self.iteration += 1

        # Reshape
        result = x_new.reshape(shape)

        # Symmetrize if requested (density matrices, etc.)
        if self.cfg.symmetrize and result.ndim == 2:
            result = 0.5 * (result + result.T)

        return result, info

    # ── Methods ───────────────────────────────────────────────────────

    def _linear(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Damped linear mixing: x_{k+1} = x_k + β·z_k."""
        step = self.cfg.beta * z
        step = self._clip(step, z)
        return x + step

    def _anderson(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Anderson Type-II in delta form.

        γ = argmin ‖z_k - ΔZ·γ‖² + λ‖γ‖²
        x_{k+1} = (x_k + β·z_k) − (ΔX + β·ΔZ)·γ
        """
        m = len(self._x_hist)
        if m < 2:
            return self._linear(x, z)

        n_cols = min(m - 1, self.cfg.history_size)
        xs = self._x_hist[-(n_cols + 1):]
        zs = self._z_hist[-(n_cols + 1):]

        # Build difference matrices
        dX = np.array([xs[i + 1] - xs[i] for i in range(n_cols)])  # (n_cols, n)
        dZ = np.array([zs[i + 1] - zs[i] for i in range(n_cols)])  # (n_cols, n)

        # Solve regularized normal equations: (dZ·dZᵀ + λI)γ = dZ·z
        G = dZ @ dZ.T  # (n_cols, n_cols)
        G += self.cfg.ridge * np.eye(n_cols)
        b = dZ @ z

        try:
            gamma = np.linalg.solve(G, b)
        except np.linalg.LinAlgError:
            return self._linear(x, z)

        # Anderson update
        beta = self.cfg.beta
        correction = (dX + beta * dZ).T @ gamma
        step = beta * z - correction
        step = self._clip(step, z)
        return x + step

    def _pulay(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Pulay DIIS: minimize ‖Σ cᵢ·zᵢ‖² subject to Σ cᵢ = 1.

        Extrapolate: x_{k+1} = Σ cᵢ·(xᵢ + β·zᵢ)

        Falls back to linear if system is ill-conditioned.
        """
        m = len(self._x_hist)
        if m < 2:
            return self._linear(x, z)

        n = min(m, self.cfg.history_size)
        xs = self._x_hist[-n:]
        zs = self._z_hist[-n:]

        # Build DIIS B matrix: B[i,j] = <z_i, z_j>
        B = np.zeros((n + 1, n + 1))
        for i in range(n):
            for j in range(n):
                B[i, j] = np.dot(zs[i], zs[j])
        # Add ridge to diagonal
        for i in range(n):
            B[i, i] += self.cfg.ridge

        # Constraint row/column: Σ c_i = 1
        B[n, :n] = -1.0
        B[:n, n] = -1.0
        B[n, n] = 0.0

        rhs = np.zeros(n + 1)
        rhs[n] = -1.0

        try:
            sol = np.linalg.solve(B, rhs)
            c = sol[:n]
        except np.linalg.LinAlgError:
            return self._linear(x, z)

        # Extrapolate: x_{k+1} = Σ c_i · (x_i + β·z_i)
        beta = self.cfg.beta
        x_new = np.zeros_like(x)
        for i in range(n):
            x_new += c[i] * (xs[i] + beta * zs[i])

        # Guarded blend with linear step for safety
        x_lin = self._linear(x, z)
        x_new = 0.85 * x_new + 0.15 * x_lin

        step = x_new - x
        step = self._clip(step, z)
        return x + step

    # ── Utilities ─────────────────────────────────────────────────────

    def _clip(self, step: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Clip step norm to max_step_factor × ‖z‖."""
        s_norm = np.linalg.norm(step)
        z_norm = np.linalg.norm(z)
        max_norm = self.cfg.max_step_factor * z_norm
        if max_norm > 0 and s_norm > max_norm:
            step = step * (max_norm / s_norm)
        return step


# ─── Convenience: generic fixed-point loop ────────────────────────────

def run_fixed_point(
    x0: np.ndarray,
    G: Callable[[np.ndarray], np.ndarray],
    mixer: SCFMixer | None = None,
    max_iter: int = 100,
    tol: float = 1e-8,
    verbose: bool = False,
) -> tuple[np.ndarray, list[dict]]:
    """Run a fixed-point iteration x_{k+1} = G(x_k) with mixing.

    Args:
        x0: initial state
        G: fixed-point map
        mixer: SCFMixer instance (default: periodic_pulay)
        max_iter: maximum iterations
        tol: convergence tolerance on ‖r‖_rms
        verbose: print convergence info

    Returns:
        (x_converged, history)
    """
    if mixer is None:
        mixer = SCFMixer()

    x = x0.copy()
    history = []

    for k in range(max_iter):
        Gx = G(x)
        x_new, info = mixer.step(x, Gx)
        info["converged"] = info["rnorm"] < tol
        history.append(info)

        if verbose:
            tag = f" [{info['method_used']}]" if info['method_used'] != 'linear' else ''
            restart = ' RESTART' if info.get('restarted') else ''
            print(f"  iter {k:3d}: ‖r‖={info['rnorm']:.2e}{tag}{restart}")

        if info["converged"]:
            return x_new, history

        x = x_new

    return x, history
