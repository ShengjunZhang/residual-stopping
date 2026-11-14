"""Full OD candidate-path elastic-demand VIs with computed face enclosures."""

from pathlib import Path
import argparse
import json
import re
import sys
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import dijkstra
from scipy.sparse.linalg import splu
from threadpoolctl import threadpool_limits, threadpool_info
from run_sparse_enclosure_benchmark import run_policy, reference_outer, write_csv, summarize

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "experiments" / "data"
sys.path.insert(0, str(HERE.parent / "experiments"))
from run_tntp_sioux_falls_toll import parse_network, parse_trips

NETWORKS = [("Sioux Falls", "SiouxFalls", "SiouxFalls"),
            ("Eastern Mass.", "Eastern-Massachusetts", "EMA"),
            ("Anaheim", "Anaheim", "Anaheim")]


def build_network(name, directory, stem):
    net = DATA / directory / (stem + "_net.tntp")
    trip_file = DATA / directory / (stem + "_trips.tntp")
    links, trips = parse_network(net), parse_trips(trip_file)
    metadata = net.read_text(encoding="utf-8")
    first_through = int(re.search(r"<FIRST THRU NODE>\s*(\d+)", metadata).group(1)) - 1
    tails = np.array([r["tail"] for r in links])
    heads = np.array([r["head"] for r in links])
    n_nodes = int(max(tails.max(), heads.max())) + 1
    t0 = np.array([r["free_flow"] for r in links])
    pair_edge = {(r["tail"], r["head"]): i for i, r in enumerate(links)}
    by_origin = {}
    for o, d, demand in trips:
        by_origin.setdefault(o, []).append((d, demand))
    rng = np.random.default_rng(20261014)
    weights = [t0, t0 * (1 + rng.uniform(0, .1, len(links))),
               t0 * (1 + rng.uniform(0, .2, len(links)))]
    paths, demands, od_indices, skipped = [], [], [], 0
    for origin, destinations in sorted(by_origin.items()):
        allowed = (tails >= first_through) | (tails == origin)
        predecessors = []
        for cost in weights:
            graph = sparse.csr_matrix((np.maximum(cost[allowed], 1e-6),
                                       (tails[allowed], heads[allowed])), shape=(n_nodes, n_nodes))
            _, pred = dijkstra(graph, directed=True, indices=origin, return_predecessors=True)
            predecessors.append(pred)
        for destination, demand in sorted(destinations):
            candidates = []
            for pred in predecessors:
                route, node = [], destination
                while node != origin and pred[node] >= 0 and len(route) <= n_nodes:
                    parent = int(pred[node])
                    route.append(pair_edge[(parent, node)])
                    node = parent
                if node == origin:
                    route = tuple(reversed(route))
                    if route not in candidates:
                        candidates.append(route)
            if not candidates:
                skipped += 1
                continue
            od = len(demands)
            demands.append(demand / 1000)
            paths.extend(candidates)
            od_indices.extend([od] * len(candidates))
    n = len(paths)
    row, col = [], []
    for j, route in enumerate(paths):
        row.extend(route)
        col.extend([j] * len(route))
    R = sparse.csc_matrix((np.ones(len(row)), (row, col)), shape=(len(links), n))
    S = sparse.csc_matrix((np.ones(n), (od_indices, np.arange(n))), shape=(len(demands), n))
    path_cost = np.asarray(R.T @ t0).ravel()
    cost_unit = float(np.median(path_cost))
    path_cost /= cost_unit
    demand = np.array(demands)
    capacity = np.array([max(r["capacity"] / 1000, .01) for r in links])
    slope = .35 * np.array([r["b"] for r in links]) * t0 / capacity / cost_unit
    beta = .2 / np.maximum(demand, .02)
    raw = R.T @ sparse.diags(slope) @ R + S.T @ sparse.diags(beta) @ S
    scale = 4 / float(np.max(np.asarray(abs(raw).sum(axis=1))))
    M = sparse.eye(n, format="csc") + scale * raw
    best_cost = np.array([np.min(path_cost[np.array(od_indices) == i]) for i in range(len(demand))])
    willingness = best_cost + .15 + scale * beta * demand
    c = path_cost - np.asarray(S.T @ willingness).ravel()
    frequencies = np.asarray(R.sum(axis=1)).ravel()
    selected_links = np.argsort(frequencies)[-8:]
    B = R[selected_links].T.toarray()
    B /= np.maximum(np.linalg.norm(B, axis=0), 1.)
    row_norms = np.sqrt(np.asarray(M.power(2).sum(axis=1)).ravel())
    norm_B = float(np.linalg.norm(B, 2))
    rho, lam, mu = .5, .4, 1.
    Q_bound = np.linalg.norm(c) + norm_B * .5 * np.sqrt(8)
    L_f = lam + (1 + rho * Q_bound ** 2 / (2 * n)) * norm_B ** 2 + rho * (norm_B * Q_bound) ** 2 / n
    nominal = splu(M.tocsc()).solve(-c)
    nominal_value = -.5 * c @ nominal
    bias = (1 + rho * nominal_value / n) * B.T @ nominal
    info = dict(network=name, nodes=n_nodes, links=len(links), positive_ODs=len(trips),
                retained_ODs=len(demands), skipped_ODs=skipped, paths=n, nnz=M.nnz,
                cost_unit=cost_unit, coupling_scale=scale, selected_links=selected_links.tolist(),
                paths_per_OD_min=int(np.min(np.asarray(S.sum(axis=1)))),
                paths_per_OD_max=int(np.max(np.asarray(S.sum(axis=1)))))
    print("Network setup:", json.dumps(info), flush=True)
    return dict(M=M.tocsc(), B=B, c=c, row_norms=row_norms,
                L_M=float(np.max(np.asarray(abs(M).sum(axis=1)))), mu=mu,
                rho=rho, lam=lam, bias=bias, alpha=.5 / L_f,
                B_norm=norm_B, n=n, coupling=name), info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    out = HERE / "numerical_results"
    out.mkdir(exist_ok=True)
    detail, history, setup = [], [], []
    with threadpool_limits(limits=1):
        for name, directory, stem in (NETWORKS[:1] if args.quick else NETWORKS):
            base, info = build_network(name, directory, stem)
            setup.append(info)
            for seed in (range(1) if args.quick else range(5)):
                rng = np.random.default_rng(11000 + base["n"] + seed)
                model = dict(base, seed=seed, target=rng.uniform(-.3, .3, 8), u0=rng.uniform(-.3, .3, 8))
                T = 10 if args.quick else 30
                reference = reference_outer(model, T)
                for method in ["FR1e-4", "FR1e-6", "Scheduled", "RG", "EI-A", "EI-L"]:
                    result, traces = run_policy(model, method, T=T)
                    result["gap_to_reference"] = abs(result["objective"] - reference)
                    result["network"] = name
                    detail.append(result)
                    for row in traces:
                        row.update(n=model["n"], coupling=name, network=name, seed=seed, eta=.2)
                    history.extend(traces)
                    print(name, seed, method, result["total_projections"], f"{result['total_sec']:.3f}s",
                          f"gap={result['gap_to_reference']:.2e}", flush=True)
        prefix = "elastic_quick" if args.quick else "elastic_network"
        write_csv(prefix + "_detail.csv", detail)
        write_csv(prefix + "_summary.csv", summarize(detail))
        write_csv(prefix + "_history.csv", history)
        (out / (prefix + "_setup.json")).write_text(json.dumps(setup, indent=2), encoding="utf-8")
        (out / (prefix + "_environment.json")).write_text(json.dumps(dict(
            python=sys.version, numpy=np.__version__, threadpools=threadpool_info(),
            outer_iterations=10 if args.quick else 30, seeds=1 if args.quick else 5,
            rho=.5, lam=.4, eta=.2, floor=1e-5, diagnostic_time_included=False), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
