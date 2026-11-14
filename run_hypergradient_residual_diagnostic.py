"""Residual-to-hypergradient diagnostic for the SIOPT manuscript."""

from __future__ import annotations

from pathlib import Path

import csv

import matplotlib.pyplot as plt
import numpy as np

from run_parallel_toll_demo import ParallelTollModel, project_simplex


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
MANUSCRIPT_FIGURES = ROOT.parent / "manuscript" / "figures"


def build_model() -> ParallelTollModel:
    return ParallelTollModel(
        free_flow=np.array([1.00, 1.03, 1.08, 1.12, 1.18, 1.24]),
        slope=np.array([1.90, 1.65, 1.45, 1.25, 1.10, 0.95]),
        demand=3.0,
        toll_reg=0.08,
        toll_max=1.4,
        lower_step=0.36,
        outer_step=0.55,
    )


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    MANUSCRIPT_FIGURES.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(20260601)
    model = build_model()
    budgets = np.array([1, 2, 3, 4, 6, 8, 12, 16, 24, 32])
    rows: list[dict] = []

    for sample in range(80):
        u = rng.uniform(0.0, 0.55, size=model.n_routes)
        z_exact = model.exact_equilibrium(u)
        g_exact = model.hypergradient(u, z_exact)
        z0 = project_simplex(rng.gamma(shape=1.0, scale=1.0, size=model.n_routes), model.demand)
        for steps in budgets:
            z_tilde = model.lower_fixed_point(u, z0, int(steps))
            residual = model.natural_residual(u, z_tilde)
            state_error = float(np.linalg.norm(z_tilde - z_exact))
            gradient_bias = float(np.linalg.norm(model.hypergradient(u, z_tilde) - g_exact))
            rows.append(
                {
                    "sample": sample,
                    "lower_steps": int(steps),
                    "lower_residual": residual,
                    "state_error": state_error,
                    "gradient_bias": gradient_bias,
                }
            )

    csv_path = RESULTS / "hypergradient_residual_diagnostic.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    residual = np.array([r["lower_residual"] for r in rows])
    bias = np.array([r["gradient_bias"] for r in rows])
    state_error = np.array([r["state_error"] for r in rows])
    lower_steps = np.array([r["lower_steps"] for r in rows])

    positive = (residual > 0) & (bias > 0) & (state_error > 0)
    residual = residual[positive]
    bias = bias[positive]
    state_error = state_error[positive]
    lower_steps = lower_steps[positive]

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.2), constrained_layout=True)

    for ax, y, ylabel in [
        (axes[0], bias, r"$\|g(\tilde z,u)-\nabla f(u)\|$"),
        (axes[1], state_error, r"$\|\tilde z-z(u)\|$"),
    ]:
        scatter = ax.scatter(
            residual,
            y,
            c=lower_steps,
            cmap="viridis_r",
            s=18,
            alpha=1.0,
            linewidths=0.0,
        )
        x_line = np.array([residual.min(), residual.max()])
        guide_scale = np.median(y / residual)
        ax.plot(x_line, guide_scale * x_line, "--", color="0.35", linewidth=1.2, label="linear guide")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"lower VI residual")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", color="0.86", linewidth=0.7)
        ax.legend(loc="upper left", frameon=True, framealpha=1.0)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.9)
            spine.set_edgecolor("0.15")

    axes[0].set_title("(a) Hypergradient bias")
    axes[1].set_title("(b) State error")
    cbar = fig.colorbar(scatter, ax=axes, shrink=0.88, pad=0.02)
    cbar.set_label("lower fixed-point steps")

    png_path = RESULTS / "hypergradient_residual_diagnostic.png"
    eps_path = MANUSCRIPT_FIGURES / "hypergradient_residual_diagnostic.eps"
    fig.savefig(png_path, dpi=220)
    fig.savefig(eps_path, format="eps")
    plt.close(fig)

    print(f"Wrote {csv_path}")
    print(f"Wrote {png_path}")
    print(f"Wrote {eps_path}")


if __name__ == "__main__":
    main()
