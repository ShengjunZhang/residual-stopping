"""Multi-network TNTP benchmark for residual-balanced VI oracles."""

from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve
from types import MethodType

import csv
import math

import numpy as np

from run_tntp_sioux_falls_toll import (
    MANUSCRIPT_FIGURES,
    RESULTS,
    SiouxFallsTollModel,
    k_shortest_simple_paths,
    parse_network,
    parse_trips,
    run_case,
    run_residual_balanced_case,
)


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"


NETWORKS = {
    "Sioux Falls": {
        "repo": "SiouxFalls",
        "prefix": "SiouxFalls",
        "top_od_pairs": 24,
        "k_paths": 4,
        "outer_iters": 40,
        "flow_scale": 1000.0,
    },
    "Eastern Massachusetts": {
        "repo": "Eastern-Massachusetts",
        "prefix": "EMA",
        "top_od_pairs": 20,
        "k_paths": 3,
        "outer_iters": 30,
        "flow_scale": 1000.0,
    },
    "Anaheim": {
        "repo": "Anaheim",
        "prefix": "Anaheim",
        "top_od_pairs": 12,
        "k_paths": 2,
        "outer_iters": 20,
        "flow_scale": 1000.0,
    },
}


def ensure_tntp_data(repo: str, prefix: str) -> tuple[Path, Path]:
    out_dir = DATA / repo
    out_dir.mkdir(parents=True, exist_ok=True)
    net_path = out_dir / f"{prefix}_net.tntp"
    trips_path = out_dir / f"{prefix}_trips.tntp"
    base = f"https://raw.githubusercontent.com/bstabler/TransportationNetworks/master/{repo}"
    if not net_path.exists():
        urlretrieve(f"{base}/{prefix}_net.tntp", net_path)
    if not trips_path.exists():
        urlretrieve(f"{base}/{prefix}_trips.tntp", trips_path)
    return net_path, trips_path


def build_tntp_model(config: dict) -> SiouxFallsTollModel:
    net_path, trips_path = ensure_tntp_data(config["repo"], config["prefix"])
    links = parse_network(net_path)
    trips = parse_trips(trips_path)
    n_nodes = 1 + max(max(link["tail"], link["head"]) for link in links)

    adjacency: list[list[tuple[int, int, float]]] = [[] for _ in range(n_nodes)]
    for edge_id, link in enumerate(links):
        adjacency[link["tail"]].append((link["head"], edge_id, link["free_flow"]))

    selected = sorted(trips, key=lambda item: item[2], reverse=True)[: config["top_od_pairs"]]
    path_edges: list[list[int]] = []
    od_slices: list[slice] = []
    demands: list[float] = []
    max_hops = 18 if n_nodes <= 100 else 28
    for origin, dest, demand in selected:
        paths = k_shortest_simple_paths(
            n_nodes,
            adjacency,
            origin,
            dest,
            k_paths=config["k_paths"],
            max_hops=max_hops,
            max_expansions=120000,
        )
        if not paths:
            continue
        start = len(path_edges)
        path_edges.extend(paths)
        od_slices.append(slice(start, len(path_edges)))
        demands.append(demand / config["flow_scale"])

    incidence = np.zeros((len(links), len(path_edges)))
    for path_id, edges in enumerate(path_edges):
        incidence[edges, path_id] = 1.0

    free_flow = np.array([link["free_flow"] for link in links], dtype=float)
    capacity = np.array([link["capacity"] / config["flow_scale"] for link in links], dtype=float)
    bpr_b = np.array([link["b"] for link in links], dtype=float)
    slope = 0.35 * bpr_b * free_flow / np.maximum(capacity, 1.0)
    path_reg = 0.35

    trial_q = incidence.T @ (slope[:, None] * incidence) + path_reg * np.eye(incidence.shape[1])
    lipschitz = float(np.max(np.linalg.eigvalsh(trial_q)))
    lower_step = 0.25 / lipschitz

    return SiouxFallsTollModel(
        incidence=incidence,
        od_slices=od_slices,
        demands=np.array(demands, dtype=float),
        free_flow=free_flow,
        slope=slope,
        path_reg=path_reg,
        toll_reg=0.025,
        toll_max=4.0,
        lower_step=lower_step,
        outer_step=0.18,
        n_nodes=n_nodes,
        n_selected_ods=len(od_slices),
    )


def attach_fast_equilibrium(model: SiouxFallsTollModel, max_iter: int = 2500, tol: float = 1e-7) -> None:
    def fast_exact_equilibrium(self: SiouxFallsTollModel, u: np.ndarray, warm_start: np.ndarray | None = None) -> np.ndarray:
        h = self.initial_path_flow() if warm_start is None else self.lower_fixed_point(u, warm_start, 0)
        for _ in range(max_iter):
            h_next = self.lower_fixed_point(u, h, 1)
            if self.natural_residual(u, h_next) < tol:
                return h_next
            h = h_next
        return h

    object.__setattr__(model, "exact_equilibrium", MethodType(fast_exact_equilibrium, model))


def summarize_network(network: str, model: SiouxFallsTollModel, rows: list[dict]) -> list[dict]:
    out = []
    for method in [
        "exact lower solve",
        "10 cold lower steps",
        "scheduled warm",
        "residual-balanced warm",
    ]:
        hist = [r for r in rows if r["method"] == method]
        final = hist[-1]
        out.append(
            {
                "network": network,
                "nodes": model.n_nodes,
                "links": model.n_edges,
                "ods": model.n_selected_ods,
                "paths": model.n_paths,
                "method": method,
                "objective": final["objective"],
                "best_stationarity": min(float(r["gradmap_norm"]) for r in hist),
                "avg_lower_residual": float(np.mean([float(r["lower_residual"]) for r in hist])),
                "total_lower_steps": final["total_lower_steps"],
            }
        )
    return out


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    MANUSCRIPT_FIGURES.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    summary_rows: list[dict] = []

    for network, config in NETWORKS.items():
        model = build_tntp_model(config)
        attach_fast_equilibrium(model)
        outer_iters = int(config["outer_iters"])
        cases = [
            ("exact lower solve", "exact", True),
            ("10 cold lower steps", 10, False),
            ("scheduled warm", lambda k: min(10 + k, 100), True),
        ]
        rows: list[dict] = []
        for name, lower_steps, warm_start in cases:
            rows.extend(run_case(model, name, lower_steps, outer_iters, warm_start))
        rows.extend(
            run_residual_balanced_case(
                model,
                "residual-balanced warm",
                outer_iters,
                batch_size=4,
                theta=0.20,
                max_lower_steps=120,
                residual_scale=None,
            )
        )
        for row in rows:
            row["network"] = network
        all_rows.extend(rows)
        summary_rows.extend(summarize_network(network, model, rows))
        print(
            f"{network}: nodes={model.n_nodes}, links={model.n_edges}, "
            f"ods={model.n_selected_ods}, paths={model.n_paths}, q={model.contraction_factor():.3f}"
        , flush=True)

    history_path = RESULTS / "tntp_multi_network_history.csv"
    summary_path = RESULTS / "tntp_multi_network_summary.csv"
    with history_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
