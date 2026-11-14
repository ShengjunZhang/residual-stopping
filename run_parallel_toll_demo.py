"""Parallel-route toll design demo for the VI-constrained manuscript.

The lower-level equilibrium is a strongly monotone affine VI over the
traffic-flow simplex:

    find z in K such that <a + B z + u, y - z> >= 0 for all y in K,

where K = {z >= 0, sum_i z_i = demand}. The outer planner chooses tolls u
and minimizes social travel cost plus a quadratic toll regularizer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import csv
import math

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
MANUSCRIPT_FIGURES = ROOT.parent / "manuscript" / "figures"


def project_simplex(v: np.ndarray, total: float) -> np.ndarray:
    """Euclidean projection onto {x >= 0, sum x = total}."""
    if total <= 0:
        raise ValueError("simplex total must be positive")
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - total
    ind = np.arange(1, len(v) + 1)
    cond = u - cssv / ind > 0
    if not np.any(cond):
        return np.full_like(v, total / len(v))
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / rho
    return np.maximum(v - theta, 0.0)


@dataclass(frozen=True)
class ParallelTollModel:
    free_flow: np.ndarray
    slope: np.ndarray
    demand: float
    toll_reg: float
    toll_max: float
    lower_step: float
    outer_step: float

    @property
    def n_routes(self) -> int:
        return len(self.free_flow)

    def cost_map(self, z: np.ndarray, u: np.ndarray) -> np.ndarray:
        return self.free_flow + self.slope * z + u

    def social_objective(self, u: np.ndarray, z: np.ndarray) -> float:
        social_cost = np.sum(z * (self.free_flow + self.slope * z))
        toll_penalty = 0.5 * self.toll_reg * np.dot(u, u)
        return float(social_cost + toll_penalty)

    def exact_equilibrium(self, u: np.ndarray) -> np.ndarray:
        """Solve the affine VI exactly by the KKT multiplier equation."""
        lo, hi = -1.0, 1.0

        def flow_sum(nu: float) -> float:
            z = np.maximum(0.0, -(self.free_flow + u + nu) / self.slope)
            return float(np.sum(z))

        while flow_sum(lo) < self.demand:
            lo *= 2.0
        while flow_sum(hi) > self.demand:
            hi *= 2.0

        for _ in range(100):
            mid = 0.5 * (lo + hi)
            if flow_sum(mid) > self.demand:
                lo = mid
            else:
                hi = mid

        nu = 0.5 * (lo + hi)
        z = np.maximum(0.0, -(self.free_flow + u + nu) / self.slope)
        return z * (self.demand / np.sum(z))

    def lower_fixed_point(self, u: np.ndarray, z0: np.ndarray, steps: int) -> np.ndarray:
        z = project_simplex(z0, self.demand)
        for _ in range(steps):
            z = project_simplex(z - self.lower_step * self.cost_map(z, u), self.demand)
        return z

    def natural_residual(self, u: np.ndarray, z: np.ndarray) -> float:
        next_z = project_simplex(z - self.lower_step * self.cost_map(z, u), self.demand)
        return float(np.linalg.norm(z - next_z))

    def contraction_factor(self) -> float:
        """Contraction modulus for the projected fixed-point bound."""
        mu = float(np.min(self.slope))
        lipschitz = float(np.max(self.slope))
        value = 1.0 - 2.0 * self.lower_step * mu + (self.lower_step * lipschitz) ** 2
        return float(math.sqrt(max(value, 0.0)))

    def hypergradient(self, u: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Hypergradient assuming the active route set is locally fixed."""
        active = z > 1e-8
        grad_z = self.free_flow + 2.0 * self.slope * z
        grad = self.toll_reg * u.copy()
        if np.any(active):
            inv_d = 1.0 / self.slope[active]
            denom = np.sum(inv_d)
            jac = -np.diag(inv_d) + np.outer(inv_d, inv_d) / denom
            grad[active] += jac.T @ grad_z[active]
        return grad

    def projected_gradient_mapping_norm(self, u: np.ndarray) -> float:
        z = self.exact_equilibrium(u)
        grad = self.hypergradient(u, z)
        projected = np.clip(u - self.outer_step * grad, 0.0, self.toll_max)
        return float(np.linalg.norm((u - projected) / self.outer_step))


def run_case(
    model: ParallelTollModel,
    name: str,
    lower_steps,
    outer_iters: int,
    warm_start: bool,
):
    u = np.zeros(model.n_routes)
    z_warm = np.full(model.n_routes, model.demand / model.n_routes)
    rows = []
    total_lower_steps = 0

    for k in range(outer_iters):
        if lower_steps == "exact":
            z_oracle = model.exact_equilibrium(u)
            steps = 0
        else:
            steps = int(lower_steps(k) if callable(lower_steps) else lower_steps)
            z0 = z_warm if warm_start else np.full(model.n_routes, model.demand / model.n_routes)
            z_oracle = model.lower_fixed_point(u, z0, steps)
            if warm_start:
                z_warm = z_oracle
            total_lower_steps += steps

        lower_residual = model.natural_residual(u, z_oracle)
        grad = model.hypergradient(u, z_oracle)
        u = np.clip(u - model.outer_step * grad, 0.0, model.toll_max)

        z_exact = model.exact_equilibrium(u)
        rows.append(
            {
                "method": name,
                "iter": k + 1,
                "objective": model.social_objective(u, z_exact),
                "gradmap_norm": model.projected_gradient_mapping_norm(u),
                "lower_residual": lower_residual,
                "lower_steps_used": steps,
                "total_lower_steps": total_lower_steps,
                "mean_toll": float(np.mean(u)),
                "max_toll": float(np.max(u)),
            }
        )
    return rows


def run_residual_balanced_case(
    model: ParallelTollModel,
    name: str,
    outer_iters: int,
    batch_size: int = 2,
    theta: float = 0.20,
    max_lower_steps: int = 120,
    residual_scale: float | None = None,
    stationarity_floor: float = 1e-8,
):
    """Warm-started lower oracle with the residual-balanced acceptance test."""
    if residual_scale is None:
        q = model.contraction_factor()
        residual_scale = 1.0 / max(1.0 - q, 1e-6)

    u = np.zeros(model.n_routes)
    z_warm = np.full(model.n_routes, model.demand / model.n_routes)
    rows = []
    total_lower_steps = 0

    for k in range(outer_iters):
        z_current = z_warm
        steps_used = 0
        accepted = False
        while not accepted:
            z_oracle = model.lower_fixed_point(u, z_current, batch_size)
            steps_used += batch_size
            lower_residual = model.natural_residual(u, z_oracle)
            grad = model.hypergradient(u, z_oracle)
            u_next = np.clip(u - model.outer_step * grad, 0.0, model.toll_max)
            inexact_step = float(np.linalg.norm((u - u_next) / model.outer_step))
            threshold = theta * max(inexact_step, stationarity_floor)
            accepted = residual_scale * lower_residual <= threshold or steps_used >= max_lower_steps
            z_current = z_oracle

        z_warm = z_oracle
        total_lower_steps += steps_used
        u = u_next
        z_exact = model.exact_equilibrium(u)
        rows.append(
            {
                "method": name,
                "iter": k + 1,
                "objective": model.social_objective(u, z_exact),
                "gradmap_norm": model.projected_gradient_mapping_norm(u),
                "lower_residual": lower_residual,
                "lower_steps_used": steps_used,
                "total_lower_steps": total_lower_steps,
                "mean_toll": float(np.mean(u)),
                "max_toll": float(np.max(u)),
            }
        )
    return rows


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    MANUSCRIPT_FIGURES.mkdir(parents=True, exist_ok=True)
    model = ParallelTollModel(
        free_flow=np.array([1.00, 1.03, 1.08, 1.12, 1.18, 1.24]),
        slope=np.array([1.90, 1.65, 1.45, 1.25, 1.10, 0.95]),
        demand=3.0,
        toll_reg=0.08,
        toll_max=1.4,
        lower_step=0.36,
        outer_step=0.55,
    )

    cases = [
        ("exact lower solve", "exact", True),
        ("1 cold lower step", 1, False),
        ("4 cold lower steps", 4, False),
        ("scheduled warm", lambda k: min(1 + k // 4, 40), True),
    ]

    all_rows = []
    for name, lower_steps, warm_start in cases:
        all_rows.extend(run_case(model, name, lower_steps, outer_iters=120, warm_start=warm_start))
    all_rows.extend(
        run_residual_balanced_case(
            model,
            "residual-balanced warm",
            outer_iters=120,
            batch_size=2,
            theta=0.20,
            max_lower_steps=80,
        )
    )
    case_names = [name for name, _, _ in cases] + ["residual-balanced warm"]

    csv_path = RESULTS / "parallel_toll_history.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8), constrained_layout=True)
    labels = {
        "exact lower solve": "exact lower solve",
        "1 cold lower step": "1 cold lower step",
        "4 cold lower steps": "4 cold lower steps",
        "scheduled warm": "scheduled warm",
        "residual-balanced warm": "residual-balanced (proposed)",
    }
    for name in case_names:
        rows = [r for r in all_rows if r["method"] == name]
        x = np.array([r["iter"] for r in rows])
        axes[0].plot(x, [r["objective"] for r in rows], label=labels[name], linewidth=2)
        axes[1].semilogy(x, [r["gradmap_norm"] for r in rows], label=labels[name], linewidth=2)
        axes[2].semilogy(x, [max(r["lower_residual"], 1e-14) for r in rows], label=labels[name], linewidth=2)

    axes[0].set_title("(a) Objective")
    axes[0].set_xlabel("outer iteration")
    axes[0].set_ylabel("exact induced objective")
    axes[1].set_title("(b) Stationarity")
    axes[1].set_xlabel("outer iteration")
    axes[1].set_ylabel(r"$\|\mathcal{G}_\alpha(u^k)\|$")
    axes[2].set_title("(c) Lower residual")
    axes[2].set_xlabel("outer iteration")
    axes[2].set_ylabel(r"$\mathcal{R}_\gamma(\tilde z^k,u^k)$")

    for ax in axes:
        ax.grid(True, which="both", color="0.85", linewidth=0.7)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.8)
        ax.tick_params(direction="out", width=0.8)
    axes[1].legend(loc="upper right", fontsize=7, framealpha=1.0, edgecolor="0.8")

    png_path = RESULTS / "parallel_toll_convergence.png"
    eps_path = RESULTS / "parallel_toll_convergence.eps"
    manuscript_eps_path = MANUSCRIPT_FIGURES / "parallel_toll_convergence.eps"
    fig.savefig(png_path, dpi=220)
    fig.savefig(eps_path, format="eps")
    fig.savefig(manuscript_eps_path, format="eps")

    print(f"Wrote {csv_path}")
    print(f"Wrote {png_path}")
    print(f"Wrote {eps_path}")
    print(f"Wrote {manuscript_eps_path}")
    print()
    print("Final metrics:")
    for name in case_names:
        row = [r for r in all_rows if r["method"] == name][-1]
        print(
            f"{name:24s}  objective={row['objective']:.6f}  "
            f"gradmap={row['gradmap_norm']:.3e}  "
            f"lower_residual={row['lower_residual']:.3e}  "
            f"total_lower_steps={int(row['total_lower_steps']):4d}  "
            f"mean_toll={row['mean_toll']:.3f}"
        )


if __name__ == "__main__":
    main()
