"""Policy benchmark for TNTP lower-oracle control rules.

The table produced by this script compares residual-balanced stopping against
common lower-oracle baselines on the same networks, seeds, and solver.
"""

from __future__ import annotations

from pathlib import Path
import csv
from time import perf_counter

import numpy as np

from run_tntp_eta_seed_robustness import (
    final_metrics,
    initial_toll,
    run_exact,
    run_residual_balanced,
    run_scheduled,
)
from run_tntp_multi_network_benchmark import NETWORKS, attach_fast_equilibrium, build_tntp_model


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def run_fixed_work(model, outer_iters: int, u0: np.ndarray, steps_per_iter: int) -> list[dict]:
    u = np.clip(u0.copy(), 0.0, model.toll_max)
    h_warm = model.initial_path_flow()
    exact_warm = None
    rows: list[dict] = []
    total_lower_steps = 0
    for k in range(outer_iters):
        h_oracle = model.lower_fixed_point(u, h_warm, steps_per_iter)
        h_warm = h_oracle
        total_lower_steps += steps_per_iter
        lower_residual = model.natural_residual(u, h_oracle)
        grad = model.hypergradient(u, h_oracle)
        u = np.clip(u - model.outer_step * grad, 0.0, model.toll_max)
        h_exact = model.exact_equilibrium(u, exact_warm)
        exact_warm = h_exact
        rows.append(
            {
                "iter": k + 1,
                "objective": model.social_objective(u, h_exact),
                "gradmap_norm": model.projected_gradient_mapping_norm(u),
                "lower_residual": lower_residual,
                "total_lower_steps": total_lower_steps,
            }
        )
    return rows


def run_fixed_residual(
    model,
    outer_iters: int,
    u0: np.ndarray,
    tolerance: float,
    batch_size: int = 4,
    max_lower_steps: int = 120,
) -> list[dict]:
    u = np.clip(u0.copy(), 0.0, model.toll_max)
    h_warm = model.initial_path_flow()
    exact_warm = None
    rows: list[dict] = []
    total_lower_steps = 0
    for k in range(outer_iters):
        h_current = h_warm
        steps_used = 0
        while True:
            h_oracle = model.lower_fixed_point(u, h_current, batch_size)
            steps_used += batch_size
            lower_residual = model.natural_residual(u, h_oracle)
            h_current = h_oracle
            if lower_residual <= tolerance or steps_used >= max_lower_steps:
                break
        h_warm = h_oracle
        total_lower_steps += steps_used
        grad = model.hypergradient(u, h_oracle)
        u = np.clip(u - model.outer_step * grad, 0.0, model.toll_max)
        h_exact = model.exact_equilibrium(u, exact_warm)
        exact_warm = h_exact
        rows.append(
            {
                "iter": k + 1,
                "objective": model.social_objective(u, h_exact),
                "gradmap_norm": model.projected_gradient_mapping_norm(u),
                "lower_residual": lower_residual,
                "total_lower_steps": total_lower_steps,
            }
        )
    return rows


def timed_run(fn) -> tuple[list[dict], float]:
    tic = perf_counter()
    rows = fn()
    return rows, perf_counter() - tic


def aggregate(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str, str], list[dict]] = {}
    for row in rows:
        groups.setdefault((row["network"], row["policy"], row["parameter"]), []).append(row)

    def mean(items: list[dict], name: str) -> float:
        return float(np.mean([float(item[name]) for item in items]))

    def sd(items: list[dict], name: str) -> float:
        return float(np.std([float(item[name]) for item in items], ddof=1)) if len(items) > 1 else 0.0

    order = {
        "fixed-work warm": 0,
        "fixed residual": 1,
        "scheduled warm": 2,
        "residual-balanced": 3,
    }
    out: list[dict] = []
    for (network, policy, parameter), items in groups.items():
        first = items[0]
        out.append(
            {
                "network": network,
                "nodes": first["nodes"],
                "links": first["links"],
                "ods": first["ods"],
                "paths": first["paths"],
                "policy": policy,
                "parameter": parameter,
                "seeds": len(items),
                "steps_mean": mean(items, "total_lower_steps"),
                "steps_sd": sd(items, "total_lower_steps"),
                "time_mean": mean(items, "wall_clock_sec"),
                "time_sd": sd(items, "wall_clock_sec"),
                "gap_mean": mean(items, "objective_gap"),
                "gap_sd": sd(items, "objective_gap"),
                "stationarity_mean": mean(items, "best_stationarity"),
                "stationarity_sd": sd(items, "best_stationarity"),
                "residual_mean": mean(items, "avg_lower_residual"),
                "residual_sd": sd(items, "avg_lower_residual"),
            }
        )
    return sorted(out, key=lambda r: (r["network"], order.get(r["policy"], 9), r["parameter"]))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def append_run(
    detailed: list[dict],
    model,
    network: str,
    seed: int,
    policy: str,
    parameter: str,
    rows: list[dict],
    wall_clock: float,
    base_obj: float,
) -> None:
    metrics = final_metrics(rows)
    detailed.append(
        {
            "network": network,
            "nodes": model.n_nodes,
            "links": model.n_edges,
            "ods": model.n_selected_ods,
            "paths": model.n_paths,
            "seed": seed,
            "policy": policy,
            "parameter": parameter,
            "objective": metrics["objective"],
            "objective_gap": abs(metrics["objective"] - base_obj),
            "best_stationarity": metrics["best_stationarity"],
            "final_stationarity": metrics["final_stationarity"],
            "avg_lower_residual": metrics["avg_lower_residual"],
            "total_lower_steps": metrics["total_lower_steps"],
            "wall_clock_sec": wall_clock,
        }
    )


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    seeds = list(range(5))
    fixed_residual_tolerances = [1e-2, 1e-3]
    residual_balanced_etas = [0.10, 0.20]
    detailed: list[dict] = []

    for network, config in NETWORKS.items():
        model = build_tntp_model(config)
        attach_fast_equilibrium(model)
        outer_iters = int(config["outer_iters"])
        for seed in seeds:
            u0 = initial_toll(model, seed)
            exact_rows, _ = timed_run(lambda: run_exact(model, outer_iters, u0))
            base_obj = final_metrics(exact_rows)["objective"]

            rows, seconds = timed_run(lambda: run_fixed_work(model, outer_iters, u0, 10))
            append_run(detailed, model, network, seed, "fixed-work warm", "10 steps", rows, seconds, base_obj)

            for tol in fixed_residual_tolerances:
                rows, seconds = timed_run(lambda tol=tol: run_fixed_residual(model, outer_iters, u0, tol))
                append_run(detailed, model, network, seed, "fixed residual", f"{tol:.0e}", rows, seconds, base_obj)

            rows, seconds = timed_run(lambda: run_scheduled(model, outer_iters, u0))
            append_run(detailed, model, network, seed, "scheduled warm", "-", rows, seconds, base_obj)

            for eta in residual_balanced_etas:
                rows, seconds = timed_run(lambda eta=eta: run_residual_balanced(model, outer_iters, u0, eta))
                append_run(
                    detailed,
                    model,
                    network,
                    seed,
                    "residual-balanced",
                    f"eta={eta:.2f}",
                    rows,
                    seconds,
                    base_obj,
                )

        print(f"finished {network}", flush=True)

    summary = aggregate(detailed)
    write_csv(RESULTS / "tntp_oracle_policy_benchmark_detail.csv", detailed)
    write_csv(RESULTS / "tntp_oracle_policy_benchmark_summary.csv", summary)
    print(f"wrote {RESULTS / 'tntp_oracle_policy_benchmark_summary.csv'}")


if __name__ == "__main__":
    main()
