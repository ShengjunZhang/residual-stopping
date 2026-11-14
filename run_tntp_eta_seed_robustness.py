"""TNTP robustness table over networks, seeds, and acceptance parameters."""

from __future__ import annotations

from pathlib import Path
import csv
from time import perf_counter

import numpy as np

from run_tntp_multi_network_benchmark import NETWORKS, attach_fast_equilibrium, build_tntp_model


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def initial_toll(model, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # Small random initial tolls give distinct outer trajectories without
    # changing the TNTP instance or the lower equilibrium model.
    return rng.uniform(0.0, 0.15 * model.toll_max, size=model.n_edges)


def final_metrics(rows: list[dict]) -> dict:
    final = rows[-1]
    return {
        "objective": float(final["objective"]),
        "best_stationarity": min(float(r["gradmap_norm"]) for r in rows),
        "final_stationarity": float(final["gradmap_norm"]),
        "avg_lower_residual": float(np.mean([float(r["lower_residual"]) for r in rows])),
        "total_lower_steps": int(final["total_lower_steps"]),
    }


def run_scheduled(model, outer_iters: int, u0: np.ndarray) -> list[dict]:
    u = np.clip(u0.copy(), 0.0, model.toll_max)
    h_warm = model.initial_path_flow()
    exact_warm = None
    rows: list[dict] = []
    total_lower_steps = 0
    for k in range(outer_iters):
        steps = min(10 + k, 100)
        h_oracle = model.lower_fixed_point(u, h_warm, steps)
        h_warm = h_oracle
        total_lower_steps += steps
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


def run_exact(model, outer_iters: int, u0: np.ndarray) -> list[dict]:
    u = np.clip(u0.copy(), 0.0, model.toll_max)
    exact_warm = None
    rows: list[dict] = []
    for k in range(outer_iters):
        h_oracle = model.exact_equilibrium(u, exact_warm)
        exact_warm = h_oracle
        grad = model.hypergradient(u, h_oracle)
        u = np.clip(u - model.outer_step * grad, 0.0, model.toll_max)
        h_exact = model.exact_equilibrium(u, exact_warm)
        exact_warm = h_exact
        rows.append(
            {
                "iter": k + 1,
                "objective": model.social_objective(u, h_exact),
                "gradmap_norm": model.projected_gradient_mapping_norm(u),
                "lower_residual": 0.0,
                "total_lower_steps": 0,
            }
        )
    return rows


def run_residual_balanced(model, outer_iters: int, u0: np.ndarray, theta: float) -> list[dict]:
    q = model.contraction_factor()
    residual_scale = 1.0 / max(1.0 - q, 1e-6)
    u = np.clip(u0.copy(), 0.0, model.toll_max)
    h_warm = model.initial_path_flow()
    exact_warm = None
    rows: list[dict] = []
    total_lower_steps = 0
    for k in range(outer_iters):
        h_current = h_warm
        steps_used = 0
        accepted = False
        while not accepted:
            h_oracle = model.lower_fixed_point(u, h_current, 4)
            steps_used += 4
            lower_residual = model.natural_residual(u, h_oracle)
            grad = model.hypergradient(u, h_oracle)
            u_next = np.clip(u - model.outer_step * grad, 0.0, model.toll_max)
            step_scale = float(np.linalg.norm((u - u_next) / model.outer_step))
            accepted = residual_scale * lower_residual <= theta * max(step_scale, 1e-8) or steps_used >= 120
            h_current = h_oracle
        h_warm = h_oracle
        total_lower_steps += steps_used
        u = u_next
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
        groups.setdefault((row["network"], row["policy"], row["eta"]), []).append(row)
    out: list[dict] = []
    for (network, policy, eta), items in groups.items():
        def mean(name: str) -> float:
            return float(np.mean([float(item[name]) for item in items]))

        def sd(name: str) -> float:
            return float(np.std([float(item[name]) for item in items], ddof=1)) if len(items) > 1 else 0.0

        first = items[0]
        out.append(
            {
                "network": network,
                "nodes": first["nodes"],
                "links": first["links"],
                "ods": first["ods"],
                "paths": first["paths"],
                "policy": policy,
                "eta": eta,
                "seeds": len(items),
                "steps_mean": mean("total_lower_steps"),
                "steps_sd": sd("total_lower_steps"),
                "time_mean": mean("wall_clock_sec"),
                "time_sd": sd("wall_clock_sec"),
                "gap_mean": mean("objective_gap"),
                "gap_sd": sd("objective_gap"),
                "stationarity_mean": mean("best_stationarity"),
                "stationarity_sd": sd("best_stationarity"),
            }
        )
    order = {"scheduled warm": 0, "residual-balanced": 1}
    return sorted(out, key=lambda r: (r["network"], order.get(r["policy"], 9), r["eta"]))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    seeds = [0, 1, 2]
    etas = [0.10, 0.20, 0.40]
    detailed: list[dict] = []

    for network, config in NETWORKS.items():
        model = build_tntp_model(config)
        attach_fast_equilibrium(model)
        outer_iters = int(config["outer_iters"])
        for seed in seeds:
            u0 = initial_toll(model, seed)
            exact_rows, exact_time = timed_run(lambda: run_exact(model, outer_iters, u0))
            exact_metrics = final_metrics(exact_rows)
            base_obj = exact_metrics["objective"]

            scheduled_rows, scheduled_time = timed_run(lambda: run_scheduled(model, outer_iters, u0))
            metrics = final_metrics(scheduled_rows)
            detailed.append(
                {
                    "network": network,
                    "nodes": model.n_nodes,
                    "links": model.n_edges,
                    "ods": model.n_selected_ods,
                    "paths": model.n_paths,
                    "seed": seed,
                    "policy": "scheduled warm",
                    "eta": "-",
                    "objective": metrics["objective"],
                    "objective_gap": abs(metrics["objective"] - base_obj),
                    "best_stationarity": metrics["best_stationarity"],
                    "avg_lower_residual": metrics["avg_lower_residual"],
                    "total_lower_steps": metrics["total_lower_steps"],
                    "wall_clock_sec": scheduled_time,
                    "exact_wall_clock_sec": exact_time,
                }
            )

            for eta in etas:
                rb_rows, rb_time = timed_run(lambda eta=eta: run_residual_balanced(model, outer_iters, u0, eta))
                metrics = final_metrics(rb_rows)
                detailed.append(
                    {
                        "network": network,
                        "nodes": model.n_nodes,
                        "links": model.n_edges,
                        "ods": model.n_selected_ods,
                        "paths": model.n_paths,
                        "seed": seed,
                        "policy": "residual-balanced",
                        "eta": f"{eta:.2f}",
                        "objective": metrics["objective"],
                        "objective_gap": abs(metrics["objective"] - base_obj),
                        "best_stationarity": metrics["best_stationarity"],
                        "avg_lower_residual": metrics["avg_lower_residual"],
                        "total_lower_steps": metrics["total_lower_steps"],
                        "wall_clock_sec": rb_time,
                        "exact_wall_clock_sec": exact_time,
                    }
                )
        print(f"finished {network}", flush=True)

    summary = aggregate(detailed)
    write_csv(RESULTS / "tntp_eta_seed_robustness_detail.csv", detailed)
    write_csv(RESULTS / "tntp_eta_seed_robustness_summary.csv", summary)
    print(f"wrote {RESULTS / 'tntp_eta_seed_robustness_summary.csv'}")


if __name__ == "__main__":
    main()
