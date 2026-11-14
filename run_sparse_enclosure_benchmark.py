"""Sparse nonlinear-value benchmarks with two-system compatible-face bounds."""

from pathlib import Path
from time import perf_counter
import argparse
import csv
import json
import platform
import sys

import numpy as np
from scipy import sparse
from scipy.optimize import minimize
from scipy.sparse.linalg import splu
from threadpoolctl import threadpool_limits, threadpool_info
from ellipsoid_enclosure import PrincipalCache, quadratic_statistics, value_enclosure, stationarity_error, build_geometry, component_geometry

HERE = Path(__file__).resolve().parent
OUT = HERE / "numerical_results"
METHODS = ["FW4", "FR1e-4", "FR1e-6", "Scheduled", "RG", "W-RG", "EI-A", "EI-G", "EI-L", "EI-C", "EI-CG", "EI-CGL"]


def write_csv(name, rows):
    with (OUT / name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def generate_model(n, coupling, seed):
    rng = np.random.default_rng(96000 + n + int(coupling) * 100 + seed)
    weights = rng.uniform(.5, 1.5, n - 1) * coupling
    diagonal = np.ones(n)
    diagonal[:-1] += weights
    diagonal[1:] += weights
    M = sparse.diags([-weights, diagonal, -weights], [-1, 0, 1], format="csc")
    B = np.zeros((n, 8))
    centers = np.linspace(12, n - 13, 8).astype(int)
    for j, center in enumerate(centers):
        B[center - 2:center + 3, j] = rng.uniform(.3, .7, 5)
    c = rng.uniform(-.5, .5, n)
    for center in centers:
        c[center - 4:center + 5] -= 1.5
    target = rng.uniform(-.3, .3, 8)
    u0 = rng.uniform(-.3, .3, 8)
    row_norms = np.sqrt(np.asarray(M.power(2).sum(axis=1)).ravel())
    L_M = float(np.max(np.asarray(abs(M).sum(axis=1))))
    B_norm = float(np.linalg.norm(B, 2))
    Q_bound = np.linalg.norm(c) + B_norm * .5 * np.sqrt(8)
    rho, lam, mu = .5, .4, 1.
    unconstrained = splu(M).solve(-c)
    nominal_value = -.5 * c @ unconstrained
    bias = (1 + rho * nominal_value / n) * B.T @ unconstrained
    L_f = lam + (1 + rho * Q_bound ** 2 / (2 * n * mu)) * B_norm ** 2 / mu
    L_f += rho * (B_norm * Q_bound / mu) ** 2 / n
    return dict(M=M, B=B, c=c, target=target, u0=u0, row_norms=row_norms,
                L_M=L_M, mu=mu, rho=rho, lam=lam, bias=bias, alpha=.5 / L_f,
                B_norm=B_norm, n=n, coupling=coupling, seed=seed)


def reference_state(model, u, initial=None):
    M, B, c = model["M"], model["B"], model["c"]
    q = B @ u + c
    z0 = np.zeros(len(c)) if initial is None else initial
    def fun(z):
        Mz = M @ z
        return .5 * z @ Mz + q @ z, Mz + q
    result = minimize(fun, z0, jac=True, method="L-BFGS-B", bounds=[(0, None)] * len(c),
                      options={"gtol": 1e-11, "ftol": 1e-15, "maxiter": 10000, "maxls": 40})
    z = result.x.copy()
    for _ in range(20):
        free = z > 1e-9
        zz = np.zeros_like(z)
        if np.any(free):
            zz[free] = splu(M[free][:, free].tocsc()).solve(-q[free])
        F = M @ zz + q
        if np.min(zz) >= -1e-9 and np.min(F[~free], initial=0.) >= -1e-9:
            z = np.maximum(zz, 0.)
            break
        # Continue the convex reference solve when the proposed face is infeasible.
        result = minimize(fun, np.maximum(zz, 0), jac=True, method="L-BFGS-B",
                          bounds=[(0, None)] * len(c),
                          options={"gtol": 1e-12, "ftol": 1e-16, "maxiter": 20000, "maxls": 40})
        z = result.x.copy()
    error, _ = stationarity_error(M, q, z)
    if error > 1e-8:
        raise AssertionError(f"Reference KKT error {error}")
    return z


def value_and_gradient(model, u, z):
    M, B, c = model["M"], model["B"], model["c"]
    surplus = -.5 * z @ (M @ z) - (B @ u + c) @ z
    theta = 1 + model["rho"] * surplus / len(z)
    value = surplus + model["rho"] * surplus ** 2 / (2 * len(z))
    value += .5 * model["lam"] * np.linalg.norm(u - model["target"]) ** 2
    value += model["bias"] @ u
    gradient = model["lam"] * (u - model["target"]) + model["bias"] - theta * B.T @ z
    return float(value), gradient


def reference_outer(model, T):
    u = model["u0"].copy()
    z = None
    for _ in range(T):
        z = reference_state(model, u, z)
        _, g = value_and_gradient(model, u, z)
        u = np.clip(u - model["alpha"] * g, -.5, .5)
    z = reference_state(model, u, z)
    return value_and_gradient(model, u, z)[0]


def run_policy(model, method, T=40, eta=.2, floor=1e-5):
    M, B, c = model["M"], model["B"], model["c"]
    n, alpha, mu = len(c), model["alpha"], model["mu"]
    cache = PrincipalCache(M, B, c)
    u, z = model["u0"].copy(), np.zeros(n)
    total_time = envelope_time = 0.
    component_factor = None
    weighted_adjoint = None
    if method in ["EI-C", "EI-CG", "EI-CGL", "W-RG"]:
        timer = perf_counter()
        component_factor = splu(M.tocsc(), permc_spec="MMD_AT_PLUS_A")
        if method == "W-RG":
            weighted_adjoint = component_factor.solve(abs(B))
        total_time += perf_counter() - timer
    advances = checks = kkt_checks = 0
    rows = []
    reference_z = None
    min_action_slack = min_gradient_slack = 0.
    for k in range(T):
        timer = perf_counter()
        q = B @ u + c
        used = tested = 0
        bound = None
        eps, ehat = np.nan, np.nan
        geometry = None
        last_refresh = 0
        prescribed = 4 if method == "FW4" else min(4 + k // 2, 30)
        while True:
            batch = prescribed if method in ["FW4", "Scheduled"] else 2
            for _ in range(batch):
                z = np.maximum(z - .9 / model["L_M"] * (M @ z + q), 0.)
            used += batch
            if method.startswith("FR"):
                natural = float(np.linalg.norm(z - np.maximum(z - .9 / model["L_M"] * (M @ z + q), 0.)))
                tested += 1
                tolerance = 1e-4 if method == "FR1e-4" else 1e-6
                if natural > tolerance:
                    if used >= 20000:
                        raise RuntimeError("Fixed-residual tolerance unmet at the cap")
                    continue
            error_raw, residual_vector = np.nan, None
            error = 0.
            if method in ["RG", "W-RG"] or method.startswith("EI"):
                error_raw, residual_vector = stationarity_error(M, q, z)
                kkt_checks += 1
                error = error_raw + 1e-11 * (1 + np.linalg.norm(q) + np.linalg.norm(M @ z))
            if method.startswith("EI"):
                start = perf_counter()
                if method == "EI-L":
                    if geometry is None or error > geometry["error"] or used - last_refresh >= 16:
                        geometry = build_geometry(cache, u, z, error, mu, model["row_norms"])
                        last_refresh = used
                if method in ["EI-C", "EI-CG", "EI-CGL"]:
                    xi = abs(residual_vector) + 1e-11 * (1 + abs(q) + abs(M @ z))
                    error = float(np.linalg.norm(xi))
                    if method != "EI-CGL" or geometry is None or np.any(xi > geometry["xi_anchor"]) or used - last_refresh >= 16:
                        anchor = 2 * xi if method == "EI-CGL" else xi
                        geometry = component_geometry(cache, component_factor, u, z, anchor)
                        geometry["xi_anchor"] = anchor
                        last_refresh = used
                bound = value_enclosure(cache, u, z, error, mu, model["row_norms"],
                                        model["rho"], model["lam"], model["target"], alpha, model["bias"], geometry)
                envelope_time += perf_counter() - start
                g, action = bound["gradient"], bound["action"]
                eps = bound["action_bound"] if method in ["EI-A", "EI-L", "EI-C"] else 2 * alpha * bound["gradient_bound"]
                stop = eps <= eta * max(np.linalg.norm(action - u), alpha * floor)
            else:
                base, _ = quadratic_statistics(cache.statistics(np.flatnonzero(z > 0)), u)
                surplus = -.5 * z @ (M @ z) - q @ z
                theta = 1 + model["rho"] * surplus / n
                g = model["lam"] * (u - model["target"]) + model["bias"] + theta * base
                action = np.clip(u - alpha * g, -.5, .5)
                if method.startswith("FR"):
                    stop = True
                elif method in ["RG", "W-RG"]:
                    weight_error = model["rho"] * error ** 2 / (2 * n * mu)
                    if method == "RG":
                        ehat = 2 * abs(theta) * model["B_norm"] * error / mu
                        ehat += weight_error * model["B_norm"] * (np.linalg.norm(z) + error / mu)
                    else:
                        y = weighted_adjoint.T @ abs(residual_vector)
                        ehat = 2 * abs(theta) * np.linalg.norm(y)
                        ehat += weight_error * np.linalg.norm(abs(B).T @ z + y)
                    eps = 2 * alpha * ehat
                    stop = eps <= eta * max(np.linalg.norm(action - u), alpha * floor)
                else:
                    stop = True
            if stop:
                break
            if used >= 20000:
                raise RuntimeError(f"Unmet stopping rule: {method}, n={n}, k={k}")
        total_time += perf_counter() - timer
        advances += used
        checks += tested
        if np.isnan(error_raw):
            error_raw, _ = stationarity_error(M, q, z)
        reference_z = reference_state(model, u, reference_z)
        _, exact_g = value_and_gradient(model, u, reference_z)
        exact_action = np.clip(u - alpha * exact_g, -.5, .5)
        action_error = np.linalg.norm(action - exact_action)
        gradient_error = np.linalg.norm(g - exact_g)
        if method in ["RG", "W-RG"] and gradient_error > ehat + 1e-7:
            raise AssertionError("Relative-gradient bound does not cover the reference")
        if bound is not None:
            aslk = bound["action_bound"] - action_error
            gslk = bound["gradient_bound"] - gradient_error
            if min(aslk, gslk) < -1e-7:
                raise AssertionError((method, aslk, gslk))
            min_action_slack, min_gradient_slack = min(min_action_slack, aslk), min(min_gradient_slack, gslk)
        rows.append(dict(method=method, outer=k + 1, advances=advances, total_projections=advances + checks,
                         inner=used, kkt_error=error_raw, update_error=float(action_error),
                         gradient_error=float(gradient_error),
                         observed_error_bound=bound["gradient_bound"] if bound else ehat,
                         accepted_epsilon=eps, step_norm=float(np.linalg.norm(action - u)),
                         action_bound=bound["action_bound"] if bound else np.nan,
                         gradient_bound=bound["gradient_bound"] if bound else np.nan,
                         uncertainty=bound["uncertainty"] if bound else -1,
                         global_uncertainty=bound["global_uncertainty"] if bound else -1,
                         local_dimension=bound["local_dimension"] if bound else -1,
                         relative_test=int(np.linalg.norm(action - u) >= alpha * floor)))
        u = action
    exact_z = reference_state(model, u, reference_z)
    value, exact_g = value_and_gradient(model, u, exact_z)
    mapping = np.linalg.norm(u - np.clip(u - alpha * exact_g, -.5, .5)) / alpha
    return dict(method=method, n=n, coupling=model["coupling"], seed=model["seed"], eta=eta,
                advances=advances, total_projections=advances + checks, kkt_checks=kkt_checks, total_sec=total_time,
                envelope_sec=envelope_time, factorizations=cache.factorizations,
                comparison_factorizations=int(component_factor is not None),
                stationarity=float(mapping), objective=value,
                min_action_slack=min_action_slack, min_gradient_slack=min_gradient_slack,
                ambiguous_accepts=sum(r["uncertainty"] > 0 for r in rows),
                max_uncertainty=max(r["uncertainty"] for r in rows)), rows


def summarize(detail):
    rows = []
    for n in sorted({r["n"] for r in detail}):
        for coupling in sorted({r["coupling"] for r in detail}):
            for eta in sorted({r["eta"] for r in detail}):
                for method in METHODS:
                    group = [r for r in detail if (r["n"], r["coupling"], r["eta"], r["method"]) ==
                             (n, coupling, eta, method)]
                    if not group:
                        continue
                    row = dict(n=n, coupling=coupling, eta=eta, method=method, seeds=len(group))
                    for key in ["advances", "total_projections", "total_sec", "envelope_sec", "factorizations",
                                "stationarity", "gap_to_reference", "ambiguous_accepts", "max_uncertainty"]:
                        values = [r[key] for r in group]
                        row[key + "_mean"] = float(np.mean(values))
                        row[key + "_sd"] = float(np.std(values, ddof=1)) if len(group) > 1 else 0.
                    rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--etas", action="store_true")
    parser.add_argument("--lazy-only", action="store_true")
    parser.add_argument("--component-only", action="store_true")
    parser.add_argument("--paired-relative", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    ns = [200] if args.quick else [200, 1000, 5000]
    couplings = [3] if args.quick else [1, 5]
    seeds = range(1) if args.quick else range(5)
    T = 20 if args.quick else 40
    detail, history = [], []
    with threadpool_limits(limits=1):
        for n in ns:
            for coupling in couplings:
                for seed in seeds:
                    model = generate_model(n, coupling, seed)
                    reference = reference_outer(model, T)
                    core_methods = [m for m in METHODS if m not in ["W-RG", "EI-CG", "EI-CGL"]]
                    methods = ["RG", "W-RG", "EI-CG", "EI-CGL"] if args.paired_relative else (["EI-C"] if args.component_only else (["EI-L"] if args.lazy_only else core_methods))
                    for method in methods:
                        etas = [.1, .2, .4] if args.etas and method.startswith("EI") else [.2]
                        for eta in etas:
                            result, traces = run_policy(model, method, T=T, eta=eta)
                            result["gap_to_reference"] = abs(result["objective"] - reference)
                            detail.append(result)
                            for row in traces:
                                row.update(n=n, coupling=coupling, seed=seed, eta=eta)
                            history.extend(traces)
                            print(n, coupling, seed, method, eta, result["total_projections"],
                                  f"{result['total_sec']:.3f}s", f"stat={result['stationarity']:.2e}",
                                  f"amb={result['ambiguous_accepts']}", flush=True)
        prefix = "sparse_paired" if args.paired_relative else ("sparse_component" if args.component_only else ("sparse_lazy" if args.lazy_only else ("sparse_quick" if args.quick else "sparse_enclosure")))
        write_csv(prefix + "_detail.csv", detail)
        write_csv(prefix + "_summary.csv", summarize(detail))
        write_csv(prefix + "_history.csv", history)
        metadata = dict(python=sys.version, numpy=np.__version__, platform=platform.platform(),
                        processor=platform.processor(), threadpools=threadpool_info(),
                        ns=ns, couplings=couplings, seeds=list(seeds), T=T,
                        upper_dimension=8, rho=.5, lam=.4, floor=1e-5,
                        diagnostic_time_included=False)
        (OUT / (prefix + "_environment.json")).write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
