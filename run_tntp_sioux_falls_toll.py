"""TNTP Sioux Falls path-flow toll experiment.

The script downloads the Sioux Falls network and trip table from the
TransportationNetworks repository if the files are not already cached.  It then
builds a reduced path-flow benchmark from the largest OD pairs and runs the
same residual-certified lower-oracle comparison used in the manuscript.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve

import csv
import heapq
import math
import re

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

from figure_export import save_png_backed_eps


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "SiouxFalls"
RESULTS = ROOT / "results"
MANUSCRIPT_FIGURES = ROOT.parent / "manuscript_setp_v1" / "figures"

BASE_URL = "https://raw.githubusercontent.com/bstabler/TransportationNetworks/master/SiouxFalls"
NET_URL = f"{BASE_URL}/SiouxFalls_net.tntp"
TRIPS_URL = f"{BASE_URL}/SiouxFalls_trips.tntp"


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "SimHei",
            "axes.unicode_minus": False,
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
            "legend.frameon": False,
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


def ensure_data() -> tuple[Path, Path]:
    DATA.mkdir(parents=True, exist_ok=True)
    net_path = DATA / "SiouxFalls_net.tntp"
    trips_path = DATA / "SiouxFalls_trips.tntp"
    if not net_path.exists():
        urlretrieve(NET_URL, net_path)
    if not trips_path.exists():
        urlretrieve(TRIPS_URL, trips_path)
    return net_path, trips_path


def parse_network(path: Path) -> list[dict]:
    links: list[dict] = []
    in_data = False
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("<END OF METADATA>"):
            in_data = True
            continue
        if not in_data or line.startswith("~"):
            continue
        line = line.replace(";", "")
        parts = line.split()
        if len(parts) < 10:
            continue
        links.append(
            {
                "tail": int(parts[0]) - 1,
                "head": int(parts[1]) - 1,
                "capacity": float(parts[2]),
                "length": float(parts[3]),
                "free_flow": float(parts[4]),
                "b": float(parts[5]),
                "power": float(parts[6]),
            }
        )
    return links


def parse_trips(path: Path) -> list[tuple[int, int, float]]:
    trips: list[tuple[int, int, float]] = []
    origin: int | None = None
    entry_pattern = re.compile(r"(\d+)\s*:\s*([0-9.]+)")
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("<"):
            continue
        if line.startswith("Origin"):
            origin = int(line.split()[1]) - 1
            continue
        if origin is None:
            continue
        for dest_text, demand_text in entry_pattern.findall(line):
            dest = int(dest_text) - 1
            demand = float(demand_text)
            if dest != origin and demand > 0.0:
                trips.append((origin, dest, demand))
    return trips


def k_shortest_simple_paths(
    n_nodes: int,
    adjacency: list[list[tuple[int, int, float]]],
    source: int,
    target: int,
    k_paths: int,
    max_hops: int = 16,
    max_expansions: int = 50000,
) -> list[list[int]]:
    heap: list[tuple[float, int, tuple[int, ...], tuple[int, ...]]] = [(0.0, source, (source,), ())]
    out: list[list[int]] = []
    expansions = 0
    while heap and len(out) < k_paths and expansions < max_expansions:
        cost, node, nodes, edges = heapq.heappop(heap)
        expansions += 1
        if node == target:
            out.append(list(edges))
            continue
        if len(edges) >= max_hops:
            continue
        visited = set(nodes)
        for nxt, edge_id, weight in adjacency[node]:
            if nxt in visited:
                continue
            heapq.heappush(heap, (cost + weight, nxt, nodes + (nxt,), edges + (edge_id,)))
    return out


def project_simplex(v: np.ndarray, total: float) -> np.ndarray:
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


def project_product_simplex(h: np.ndarray, od_slices: list[slice], demands: np.ndarray) -> np.ndarray:
    out = np.zeros_like(h)
    for slc, demand in zip(od_slices, demands):
        out[slc] = project_simplex(h[slc], float(demand))
    return out


@dataclass(frozen=True)
class SiouxFallsTollModel:
    incidence: np.ndarray
    od_slices: list[slice]
    demands: np.ndarray
    free_flow: np.ndarray
    slope: np.ndarray
    path_reg: float
    toll_reg: float
    toll_max: float
    lower_step: float
    outer_step: float
    n_nodes: int
    n_selected_ods: int

    @property
    def n_edges(self) -> int:
        return self.incidence.shape[0]

    @property
    def n_paths(self) -> int:
        return self.incidence.shape[1]

    @property
    def q_matrix(self) -> np.ndarray:
        weighted = self.slope[:, None] * self.incidence
        return self.incidence.T @ weighted + self.path_reg * np.eye(self.n_paths)

    @property
    def p_matrix(self) -> np.ndarray:
        return self.incidence.T

    def initial_path_flow(self) -> np.ndarray:
        h = np.zeros(self.n_paths)
        for slc, demand in zip(self.od_slices, self.demands):
            h[slc] = demand / (slc.stop - slc.start)
        return h

    def edge_flow(self, h: np.ndarray) -> np.ndarray:
        return self.incidence @ h

    def cost_map(self, h: np.ndarray, u: np.ndarray) -> np.ndarray:
        x = self.edge_flow(h)
        edge_cost = self.free_flow + self.slope * x + u
        return self.incidence.T @ edge_cost + self.path_reg * h

    def social_objective(self, u: np.ndarray, h: np.ndarray) -> float:
        x = self.edge_flow(h)
        social_cost = np.sum(x * (self.free_flow + self.slope * x))
        toll_penalty = 0.5 * self.toll_reg * np.dot(u, u)
        return float(social_cost + toll_penalty)

    def lower_fixed_point(self, u: np.ndarray, h0: np.ndarray, steps: int) -> np.ndarray:
        h = project_product_simplex(h0, self.od_slices, self.demands)
        for _ in range(steps):
            h = project_product_simplex(h - self.lower_step * self.cost_map(h, u), self.od_slices, self.demands)
        return h

    def natural_residual(self, u: np.ndarray, h: np.ndarray) -> float:
        next_h = project_product_simplex(h - self.lower_step * self.cost_map(h, u), self.od_slices, self.demands)
        return float(np.linalg.norm(h - next_h))

    def exact_equilibrium(self, u: np.ndarray, warm_start: np.ndarray | None = None) -> np.ndarray:
        h = self.initial_path_flow() if warm_start is None else project_product_simplex(warm_start, self.od_slices, self.demands)
        for _ in range(12000):
            h_next = self.lower_fixed_point(u, h, 1)
            if self.natural_residual(u, h_next) < 1e-9:
                return h_next
            h = h_next
        return h

    def contraction_factor(self) -> float:
        eigs = np.linalg.eigvalsh(self.q_matrix)
        mu = float(np.min(eigs))
        lipschitz = float(np.max(eigs))
        return float(max(abs(1.0 - self.lower_step * mu), abs(1.0 - self.lower_step * lipschitz)))

    def hypergradient(self, u: np.ndarray, h: np.ndarray) -> np.ndarray:
        active = h > 1e-8
        for slc in self.od_slices:
            if not np.any(active[slc]):
                active[slc.start + int(np.argmax(h[slc]))] = True

        active_idx = np.flatnonzero(active)
        q_active = self.q_matrix[np.ix_(active_idx, active_idx)]
        p_active = self.p_matrix[active_idx, :]

        rows = []
        for slc in self.od_slices:
            row = np.zeros(active_idx.size)
            mask = (active_idx >= slc.start) & (active_idx < slc.stop)
            row[mask] = 1.0
            rows.append(row)
        dmat = np.vstack(rows)

        kkt = np.block(
            [
                [q_active, dmat.T],
                [dmat, np.zeros((dmat.shape[0], dmat.shape[0]))],
            ]
        )
        rhs = np.vstack([-p_active, np.zeros((dmat.shape[0], self.n_edges))])
        sol = np.linalg.solve(kkt, rhs)
        dh_du_active = sol[: active_idx.size, :]

        x = self.edge_flow(h)
        grad_h = self.incidence.T @ (self.free_flow + 2.0 * self.slope * x)
        grad = self.toll_reg * u.copy()
        grad += dh_du_active.T @ grad_h[active_idx]
        return grad

    def projected_gradient_mapping_norm(self, u: np.ndarray) -> float:
        h_exact = self.exact_equilibrium(u)
        grad = self.hypergradient(u, h_exact)
        projected = np.clip(u - self.outer_step * grad, 0.0, self.toll_max)
        return float(np.linalg.norm((u - projected) / self.outer_step))


def build_model(top_od_pairs: int = 30, k_paths: int = 4, flow_scale: float = 1000.0) -> SiouxFallsTollModel:
    net_path, trips_path = ensure_data()
    links = parse_network(net_path)
    trips = parse_trips(trips_path)
    n_nodes = 1 + max(max(link["tail"], link["head"]) for link in links)

    adjacency: list[list[tuple[int, int, float]]] = [[] for _ in range(n_nodes)]
    for edge_id, link in enumerate(links):
        adjacency[link["tail"]].append((link["head"], edge_id, link["free_flow"]))

    selected = sorted(trips, key=lambda item: item[2], reverse=True)[:top_od_pairs]
    path_edges: list[list[int]] = []
    od_slices: list[slice] = []
    demands: list[float] = []
    for origin, dest, demand in selected:
        paths = k_shortest_simple_paths(n_nodes, adjacency, origin, dest, k_paths=k_paths)
        if not paths:
            continue
        start = len(path_edges)
        path_edges.extend(paths)
        od_slices.append(slice(start, len(path_edges)))
        demands.append(demand / flow_scale)

    incidence = np.zeros((len(links), len(path_edges)))
    for path_id, edges in enumerate(path_edges):
        incidence[edges, path_id] = 1.0

    free_flow = np.array([link["free_flow"] for link in links], dtype=float)
    capacity = np.array([link["capacity"] / flow_scale for link in links], dtype=float)
    bpr_b = np.array([link["b"] for link in links], dtype=float)
    # Affine delay slope in scaled-flow units.  This preserves TNTP free-flow
    # times/capacities while keeping the lower VI strongly monotone after the
    # route-preference regularization.
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


def run_case(
    model: SiouxFallsTollModel,
    name: str,
    lower_steps,
    outer_iters: int,
    warm_start: bool,
) -> list[dict]:
    u = np.zeros(model.n_edges)
    h_warm = model.initial_path_flow()
    exact_warm = None
    rows = []
    total_lower_steps = 0

    for k in range(outer_iters):
        if lower_steps == "exact":
            h_oracle = model.exact_equilibrium(u, exact_warm)
            exact_warm = h_oracle
            steps = 0
        else:
            steps = int(lower_steps(k) if callable(lower_steps) else lower_steps)
            h0 = h_warm if warm_start else model.initial_path_flow()
            h_oracle = model.lower_fixed_point(u, h0, steps)
            if warm_start:
                h_warm = h_oracle
            total_lower_steps += steps

        lower_residual = model.natural_residual(u, h_oracle)
        grad = model.hypergradient(u, h_oracle)
        u = np.clip(u - model.outer_step * grad, 0.0, model.toll_max)

        h_exact = model.exact_equilibrium(u, exact_warm)
        exact_warm = h_exact
        rows.append(
            {
                "method": name,
                "iter": k + 1,
                "objective": model.social_objective(u, h_exact),
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
    model: SiouxFallsTollModel,
    name: str,
    outer_iters: int,
    batch_size: int = 4,
    theta: float = 0.20,
    max_lower_steps: int = 120,
    residual_scale: float | None = None,
    stationarity_floor: float = 1e-8,
) -> list[dict]:
    if residual_scale is None:
        q = model.contraction_factor()
        residual_scale = 1.0 / max(1.0 - q, 1e-6)

    u = np.zeros(model.n_edges)
    h_warm = model.initial_path_flow()
    exact_warm = None
    rows = []
    total_lower_steps = 0

    for k in range(outer_iters):
        h_current = h_warm
        steps_used = 0
        accepted = False
        while not accepted:
            h_oracle = model.lower_fixed_point(u, h_current, batch_size)
            steps_used += batch_size
            lower_residual = model.natural_residual(u, h_oracle)
            grad = model.hypergradient(u, h_oracle)
            u_next = np.clip(u - model.outer_step * grad, 0.0, model.toll_max)
            inexact_step = float(np.linalg.norm((u - u_next) / model.outer_step))
            threshold = theta * max(inexact_step, stationarity_floor)
            accepted = residual_scale * lower_residual <= threshold or steps_used >= max_lower_steps
            h_current = h_oracle

        h_warm = h_oracle
        total_lower_steps += steps_used
        u = u_next
        h_exact = model.exact_equilibrium(u, exact_warm)
        exact_warm = h_exact
        rows.append(
            {
                "method": name,
                "iter": k + 1,
                "objective": model.social_objective(u, h_exact),
                "gradmap_norm": model.projected_gradient_mapping_norm(u),
                "lower_residual": lower_residual,
                "lower_steps_used": steps_used,
                "total_lower_steps": total_lower_steps,
                "mean_toll": float(np.mean(u)),
                "max_toll": float(np.max(u)),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, rows: list[dict], method_order: list[str]) -> list[dict]:
    summary = []
    for method in method_order:
        hist = [r for r in rows if r["method"] == method]
        final = hist[-1]
        best_stationarity = min(float(r["gradmap_norm"]) for r in hist)
        best_row = min(hist, key=lambda r: float(r["gradmap_norm"]))
        avg_residual = float(np.mean([float(r["lower_residual"]) for r in hist]))
        summary.append(
            {
                "method": method,
                "objective": final["objective"],
                "final_stationarity": final["gradmap_norm"],
                "best_stationarity": best_stationarity,
                "avg_lower_residual": avg_residual,
                "lower_residual": final["lower_residual"],
                "total_lower_steps": final["total_lower_steps"],
                "best_iter": best_row["iter"],
            }
        )
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    return summary


def plot_results(rows: list[dict], method_order: list[str]) -> None:
    configure_matplotlib()
    colors = {
        "exact lower solve": "0.05",
        "3 cold lower steps": "#d55e00",
        "10 cold lower steps": "#0072b2",
        "scheduled warm": "#cc79a7",
        "residual-balanced warm": "#009e73",
    }
    labels = {
        "exact lower solve": "精确下层",
        "3 cold lower steps": "固定3步",
        "10 cold lower steps": "固定10步",
        "scheduled warm": "计划式",
        "residual-balanced warm": "残差平衡",
    }
    linestyles = {
        "exact lower solve": "-",
        "3 cold lower steps": "--",
        "10 cold lower steps": "-.",
        "scheduled warm": ":",
        "residual-balanced warm": "-",
    }
    markers = {
        "exact lower solve": None,
        "3 cold lower steps": "o",
        "10 cold lower steps": "s",
        "scheduled warm": "^",
        "residual-balanced warm": "D",
    }
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.25))
    exact_final = [r for r in rows if r["method"] == "exact lower solve"][-1]["objective"]

    for method in method_order:
        sub = [r for r in rows if r["method"] == method]
        x_iter = np.array([r["iter"] for r in sub])
        objective = np.array([r["objective"] for r in sub])
        objective_gap = np.maximum(objective - exact_final, 1e-12)
        grad = np.array([max(r["gradmap_norm"], 1e-14) for r in sub])
        residual = np.array([max(r["lower_residual"], 1e-14) for r in sub])
        work = np.array([max(r["total_lower_steps"], 1) for r in sub])
        best_grad = np.minimum.accumulate(grad)
        markevery = max(1, len(x_iter) // 8)

        axes[0].semilogy(
            x_iter,
            objective_gap,
            linewidth=2.3,
            linestyle=linestyles[method],
            marker=markers[method],
            markevery=markevery,
            markersize=3.5,
            color=colors[method],
            label=labels[method],
        )
        axes[1].semilogy(
            x_iter,
            best_grad,
            linewidth=2.3,
            linestyle=linestyles[method],
            marker=markers[method],
            markevery=markevery,
            markersize=3.5,
            color=colors[method],
        )
        if method != "exact lower solve":
            axes[2].loglog(
                work,
                best_grad,
                linewidth=2.3,
                linestyle=linestyles[method],
                marker=markers[method],
                markevery=markevery,
                markersize=3.5,
                color=colors[method],
                label=labels[method],
            )
        axes[1].semilogy(x_iter, residual, linestyle=":", linewidth=1.8, color=colors[method])

    axes[0].set_xlabel("外层迭代")
    axes[0].set_ylabel("相对精确下层目标差距")
    axes[0].set_title("(a) 目标差距")
    axes[1].set_xlabel("外层迭代")
    axes[1].set_ylabel("平稳性/残差")
    axes[1].set_title("(b) 残差与平稳性")
    axes[2].set_xlabel("累计下层投影次数")
    axes[2].set_ylabel("最佳精确平稳性")
    axes[2].set_title("(c) 平稳性与工作量")
    for ax in axes:
        ax.grid(True, which="both", color="0.88", linewidth=0.7)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.0)
        ax.tick_params(direction="out", width=1.0)
    apply_ascii_log_ticks(*axes)
    axes[0].legend(loc="best")
    axes[2].legend(loc="best")
    fig.tight_layout()

    fig.savefig(RESULTS / "tntp_sioux_falls_toll.eps", format="eps", bbox_inches="tight")
    save_png_backed_eps(
        fig,
        MANUSCRIPT_FIGURES / "tntp_sioux_falls_toll.eps",
        RESULTS / "tntp_sioux_falls_toll_manuscript.png",
        bbox_inches="tight",
    )
    fig.savefig(RESULTS / "tntp_sioux_falls_toll.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    MANUSCRIPT_FIGURES.mkdir(parents=True, exist_ok=True)

    model = build_model(top_od_pairs=30, k_paths=4)
    outer_iters = 80
    cases = [
        ("exact lower solve", "exact", True),
        ("3 cold lower steps", 3, False),
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

    method_order = [name for name, _, _ in cases] + ["residual-balanced warm"]
    write_csv(RESULTS / "tntp_sioux_falls_history.csv", rows)
    summary = write_summary(RESULTS / "tntp_sioux_falls_summary.csv", rows, method_order)
    plot_results(rows, method_order)

    print("TNTP Sioux Falls toll experiment")
    print(
        f"nodes={model.n_nodes}, edges={model.n_edges}, selected_ods={model.n_selected_ods}, "
        f"paths={model.n_paths}, lower_step={model.lower_step:.3e}, contraction={model.contraction_factor():.3f}"
    )
    for row in summary:
        print(
            f"{row['method']:24s}  obj={row['objective']:.4f}  "
            f"final_grad={row['final_stationarity']:.3e}  best_grad={row['best_stationarity']:.3e}  "
            f"avg_residual={row['avg_lower_residual']:.3e}  work={int(row['total_lower_steps'])}"
        )


if __name__ == "__main__":
    main()
