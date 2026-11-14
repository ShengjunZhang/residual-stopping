"""Residual-balanced lower-oracle ablations for the VI manuscript.

The first panel compares fixed, scheduled, and residual-balanced lower-level
work against total projection calls. The second panel stresses the same methods
as the lower VI becomes more ill conditioned.
"""

from __future__ import annotations

from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

from figure_export import save_png_backed_eps

from run_parallel_toll_demo import (
    ParallelTollModel,
    run_case,
    run_residual_balanced_case,
)


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
MANUSCRIPT_FIGURES = ROOT.parent / "manuscript_setp_v1" / "figures"


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "SimHei",
            "axes.unicode_minus": False,
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "ps.fonttype": 42,
            "pdf.fonttype": 42,
        }
    )


def _ascii_log_formatter(x: float, _pos: int) -> str:
    if x <= 0:
        return ""
    exponent = np.log10(x)
    if abs(exponent - round(exponent)) < 1e-8:
        return f"1e{int(round(exponent))}"
    return ""


def apply_ascii_log_ticks(*axes) -> None:
    major = FuncFormatter(_ascii_log_formatter)
    blank = FuncFormatter(lambda _x, _pos: "")
    for ax in axes:
        if ax.get_xscale() == "log":
            ax.xaxis.set_major_formatter(major)
            ax.xaxis.set_minor_formatter(blank)
        if ax.get_yscale() == "log":
            ax.yaxis.set_major_formatter(major)
            ax.yaxis.set_minor_formatter(blank)


def base_model() -> ParallelTollModel:
    return ParallelTollModel(
        free_flow=np.array([1.00, 1.03, 1.08, 1.12, 1.18, 1.24]),
        slope=np.array([1.90, 1.65, 1.45, 1.25, 1.10, 0.95]),
        demand=3.0,
        toll_reg=0.08,
        toll_max=1.4,
        lower_step=0.36,
        outer_step=0.55,
    )


def conditioned_model(kappa: float, seed: int) -> ParallelTollModel:
    rng = np.random.default_rng(seed)
    n_routes = 8
    free_flow = np.sort(1.0 + 0.18 * rng.random(n_routes))
    base_slopes = np.linspace(kappa, 1.0, n_routes)
    jitter = 1.0 + 0.04 * rng.standard_normal(n_routes)
    slope = np.clip(base_slopes * jitter, 1.0, kappa)
    slope[0] = kappa
    slope[-1] = 1.0
    slope = np.sort(slope)[::-1]
    lower_step = 0.95 / (kappa * kappa)
    return ParallelTollModel(
        free_flow=free_flow,
        slope=slope,
        demand=4.0,
        toll_reg=0.08,
        toll_max=1.4,
        lower_step=lower_step,
        outer_step=0.45,
    )


def run_fixed_residual_case(
    model: ParallelTollModel,
    name: str,
    tolerance: float,
    outer_iters: int,
    batch_size: int = 2,
    max_lower_steps: int = 160,
) -> list[dict]:
    u = np.zeros(model.n_routes)
    z_warm = np.full(model.n_routes, model.demand / model.n_routes)
    rows = []
    total_lower_steps = 0

    for k in range(outer_iters):
        z_current = z_warm
        steps_used = 0
        while True:
            z_oracle = model.lower_fixed_point(u, z_current, batch_size)
            steps_used += batch_size
            lower_residual = model.natural_residual(u, z_oracle)
            z_current = z_oracle
            if lower_residual <= tolerance or steps_used >= max_lower_steps:
                break

        z_warm = z_oracle
        total_lower_steps += steps_used
        grad = model.hypergradient(u, z_oracle)
        u = np.clip(u - model.outer_step * grad, 0.0, model.toll_max)
        rows.append(
            {
                "method": name,
                "iter": k + 1,
                "objective": model.social_objective(u, model.exact_equilibrium(u)),
                "gradmap_norm": model.projected_gradient_mapping_norm(u),
                "lower_residual": lower_residual,
                "lower_steps_used": steps_used,
                "total_lower_steps": total_lower_steps,
                "mean_toll": float(np.mean(u)),
                "max_toll": float(np.max(u)),
            }
        )
    return rows


def median_quantiles(rows: list[dict], method: str, kappa: float) -> dict:
    sub = [r for r in rows if r["method"] == method and abs(r["condition_ratio"] - kappa) < 1e-12]
    grad = np.array([r["final_gradmap_norm"] for r in sub], dtype=float)
    work = np.array([r["total_lower_steps"] for r in sub], dtype=float)
    return {
        "method": method,
        "condition_ratio": kappa,
        "grad_med": float(np.median(grad)),
        "grad_q1": float(np.quantile(grad, 0.25)),
        "grad_q3": float(np.quantile(grad, 0.75)),
        "work_med": float(np.median(work)),
        "work_q1": float(np.quantile(work, 0.25)),
        "work_q3": float(np.quantile(work, 0.75)),
    }


def main() -> None:
    configure_matplotlib()
    RESULTS.mkdir(parents=True, exist_ok=True)
    MANUSCRIPT_FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "ps.fonttype": 42,
            "pdf.fonttype": 42,
            "axes.spines.top": True,
            "axes.spines.right": True,
        }
    )

    outer_iters = 100
    model = base_model()
    work_histories: dict[str, list[dict]] = {
        "4 cold lower steps": run_case(model, "4 cold lower steps", 4, outer_iters, warm_start=False),
        "fixed residual 1e-3": run_fixed_residual_case(
            model,
            "fixed residual 1e-3",
            tolerance=1e-3,
            outer_iters=outer_iters,
        ),
        "scheduled warm": run_case(model, "scheduled warm", lambda k: min(1 + k // 4, 40), outer_iters, warm_start=True),
        "residual-balanced warm": run_residual_balanced_case(
            model,
            "residual-balanced warm",
            outer_iters,
            batch_size=2,
            theta=0.20,
            max_lower_steps=80,
        ),
    }

    kappa_values = [1.25, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]
    seeds = list(range(8))
    condition_rows = []
    for kappa in kappa_values:
        for seed in seeds:
            cm = conditioned_model(kappa, seed)
            cases = [
                ("4 cold lower steps", run_case(cm, "4 cold lower steps", 4, 80, warm_start=False)),
                (
                    "fixed residual 1e-3",
                    run_fixed_residual_case(
                        cm,
                        "fixed residual 1e-3",
                        tolerance=1e-3,
                        outer_iters=80,
                    ),
                ),
                ("scheduled warm", run_case(cm, "scheduled warm", lambda k: min(1 + k // 4, 40), 80, warm_start=True)),
                (
                    "residual-balanced warm",
                    run_residual_balanced_case(
                        cm,
                        "residual-balanced warm",
                        80,
                        batch_size=2,
                        theta=0.20,
                        max_lower_steps=160,
                    ),
                ),
            ]
            for method, hist in cases:
                final = hist[-1]
                condition_rows.append(
                    {
                        "condition_ratio": kappa,
                        "seed": seed,
                        "method": method,
                        "final_gradmap_norm": final["gradmap_norm"],
                        "final_lower_residual": final["lower_residual"],
                        "total_lower_steps": final["total_lower_steps"],
                    }
                )

    csv_path = RESULTS / "residual_balanced_ablation.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(condition_rows[0].keys()))
        writer.writeheader()
        writer.writerows(condition_rows)

    summary = [
        median_quantiles(condition_rows, method, kappa)
        for method in work_histories
        for kappa in kappa_values
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.65), constrained_layout=True)
    method_order = [
        "4 cold lower steps",
        "fixed residual 1e-3",
        "scheduled warm",
        "residual-balanced warm",
    ]
    labels = {
        "4 cold lower steps": "固定4步",
        "fixed residual 1e-3": r"固定残差 $10^{-3}$",
        "scheduled warm": "计划式",
        "residual-balanced warm": "残差平衡",
    }
    line_colors = {
        "4 cold lower steps": "#4c78a8",
        "fixed residual 1e-3": "#b279a2",
        "scheduled warm": "#f58518",
        "residual-balanced warm": "#54a24b",
    }
    linestyles = {
        "4 cold lower steps": "-",
        "fixed residual 1e-3": ":",
        "scheduled warm": "--",
        "residual-balanced warm": "-.",
    }
    markers = {
        "4 cold lower steps": "o",
        "fixed residual 1e-3": "D",
        "scheduled warm": "s",
        "residual-balanced warm": "^",
    }
    for method in method_order:
        hist = work_histories[method]
        x = np.array([max(r["total_lower_steps"], 1) for r in hist])
        y = np.array([max(r["gradmap_norm"], 1e-14) for r in hist])
        axes[0].semilogy(
            x,
            y,
            linewidth=2,
            linestyle=linestyles[method],
            color=line_colors[method],
            label=labels[method],
        )
    axes[0].set_title("(a) 平稳性与下层工作量")
    axes[0].set_xlabel("累计下层投影次数")
    axes[0].set_ylabel("投影梯度范数")

    for method in method_order:
        sub = [s for s in summary if s["method"] == method]
        x = np.array([s["condition_ratio"] for s in sub])
        med = np.array([s["grad_med"] for s in sub])
        q1 = np.array([s["grad_q1"] for s in sub])
        q3 = np.array([s["grad_q3"] for s in sub])
        axes[1].errorbar(
            x,
            med,
            yerr=np.vstack([med - q1, q3 - med]),
            marker=markers[method],
            linestyle=linestyles[method],
            linewidth=2,
            capsize=2.5,
            color=line_colors[method],
            zorder=3,
            label=labels[method],
        )
    axes[1].set_title("(b) 对 VI 条件数的敏感性")
    axes[1].set_xlabel(r"条件比 $L_z/\mu$")
    axes[1].set_ylabel("投影梯度范数中位数")
    axes[1].set_yscale("log")

    for method in method_order:
        sub = [s for s in summary if s["method"] == method]
        x = np.array([s["condition_ratio"] for s in sub])
        med = np.array([s["work_med"] for s in sub])
        q1 = np.array([s["work_q1"] for s in sub])
        q3 = np.array([s["work_q3"] for s in sub])
        axes[2].errorbar(
            x,
            med,
            yerr=np.vstack([med - q1, q3 - med]),
            marker=markers[method],
            linestyle=linestyles[method],
            linewidth=2,
            capsize=2.5,
            color=line_colors[method],
            label=labels[method],
        )
    axes[2].set_title("(c) 条件数与下层工作量")
    axes[2].set_xlabel(r"条件比 $L_z/\mu$")
    axes[2].set_ylabel("累计下层投影次数")

    for ax in axes:
        ax.grid(True, which="both", color="0.85", linewidth=0.7)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.8)
        ax.tick_params(direction="out", width=0.8)
    apply_ascii_log_ticks(*axes)
    axes[0].legend(frameon=True, framealpha=1.0, edgecolor="0.8")

    png_path = RESULTS / "residual_balanced_ablation.png"
    eps_path = RESULTS / "residual_balanced_ablation.eps"
    manuscript_eps_path = MANUSCRIPT_FIGURES / "residual_balanced_ablation.eps"
    fig.savefig(png_path, dpi=220)
    fig.savefig(eps_path, format="eps")
    save_png_backed_eps(fig, manuscript_eps_path, RESULTS / "residual_balanced_ablation_manuscript.png")

    print(f"Wrote {csv_path}")
    print(f"Wrote {png_path}")
    print(f"Wrote {eps_path}")
    print(f"Wrote {manuscript_eps_path}")
    print()
    print("Conditioning summary:")
    for s in summary:
        print(
            f"kappa={s['condition_ratio']:4.2f}  {s['method']:24s}  "
            f"gradmap={s['grad_med']:.3e}  work={s['work_med']:.0f}"
        )


if __name__ == "__main__":
    main()
