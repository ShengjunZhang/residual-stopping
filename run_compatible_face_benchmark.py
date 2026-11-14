"""Coupled affine VI benchmark using computed compatible-face update bounds."""

from pathlib import Path
from time import perf_counter
import argparse
import csv
import json
import platform
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from threadpoolctl import threadpool_limits, threadpool_info

HERE = Path(__file__).resolve().parent
OUT = HERE / "numerical_results"
METHODS = ["FW4", "FR1e-4", "Scheduled", "CF-A", "CF-G", "CF-E"]


def write_csv(name, rows):
    with (OUT / name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class FaceCache:
    def __init__(self, M, B, c):
        self.M, self.B, self.c = M, B, c
        self.n = len(c)
        self.items = {}

    def get(self, mask):
        if mask not in self.items:
            free = np.flatnonzero([(mask >> i) & 1 for i in range(self.n)])
            active = np.setdiff1d(np.arange(self.n), free)
            inverse = np.linalg.solve(self.M[np.ix_(free, free)], np.eye(len(free)))
            z_matrix = np.zeros((self.n, self.B.shape[1]))
            z_offset = np.zeros(self.n)
            z_radius = np.zeros(self.n)
            z_matrix[free] = -inverse @ self.B[free]
            z_offset[free] = -inverse @ self.c[free]
            z_radius[free] = np.abs(inverse).sum(axis=1)
            f_matrix = np.zeros_like(z_matrix)
            f_offset = np.zeros(self.n)
            f_radius = np.zeros(self.n)
            f_matrix[active] = self.M[active] @ z_matrix + self.B[active]
            f_offset[active] = self.M[active] @ z_offset + self.c[active]
            f_radius[active] = np.abs(self.M[np.ix_(active, free)] @ inverse).sum(axis=1)
            gradient_matrix = -self.B.T @ z_matrix
            gradient_offset = -self.B.T @ z_offset
            self.items[mask] = (free, active, z_matrix, z_offset, z_radius,
                                f_matrix, f_offset, f_radius,
                                gradient_matrix, gradient_offset)
        return self.items[mask]

    def prepare_all(self):
        items = [self.get(mask) for mask in range(1 << self.n)]
        self.free_flags = np.array([[(mask >> i) & 1 for i in range(self.n)]
                                   for mask in range(1 << self.n)], dtype=bool)
        for position, name in [(2, "z_matrix"), (3, "z_offset"), (4, "z_radius"),
                               (5, "f_matrix"), (6, "f_offset"), (7, "f_radius"),
                               (8, "gradient_matrix"), (9, "gradient_offset")]:
            setattr(self, name, np.stack([item[position] for item in items]))

    def screen(self, u, delta):
        z_center = self.z_matrix @ u + self.z_offset
        f_center = self.f_matrix @ u + self.f_offset
        free_ok = (~self.free_flags) | (z_center + delta * self.z_radius >= 0)
        active_ok = self.free_flags | (f_center + delta * (self.f_radius + 1) >= 0)
        return np.flatnonzero(np.all(free_ok & active_ok, axis=1))

    def screened_subset(self, u, z, delta, mu, max_uncertain=6):
        radius = 2 * np.sqrt(self.n) * delta / mu
        F = self.M @ z + self.B @ u + self.c
        fixed_free = z > radius
        fixed_active = F > delta + np.linalg.norm(self.M, axis=1) * radius
        if np.any(fixed_free & fixed_active):
            raise AssertionError("Conflicting certified statuses")
        uncertain = np.flatnonzero(~(fixed_free | fixed_active))
        if len(uncertain) > max_uncertain:
            return None, len(uncertain)
        base_mask = sum(1 << int(i) for i in np.flatnonzero(fixed_free))
        masks = [base_mask + sum(1 << int(j) for bit, j in enumerate(uncertain)
                                if (choice >> bit) & 1)
                 for choice in range(1 << len(uncertain))]
        retained = []
        for mask in masks:
            item = self.get(mask)
            free, active = item[:2]
            z_center = item[2] @ u + item[3]
            f_center = item[5] @ u + item[6]
            if (np.all(z_center[free] + delta * item[4][free] >= 0)
                    and np.all(f_center[active] + delta * (item[7][active] + 1) >= 0)):
                retained.append(mask)
        return np.array(retained, dtype=int), len(uncertain)

    def reference(self, u):
        z_center = self.z_matrix @ u + self.z_offset
        f_center = self.f_matrix @ u + self.f_offset
        scale = 1 + np.linalg.norm(self.B @ u + self.c, np.inf)
        feasible = np.all(((~self.free_flags) | (z_center >= -1e-10 * scale)) &
                          (self.free_flags | (f_center >= -1e-10 * scale)), axis=1)
        indices = np.flatnonzero(feasible)
        if not len(indices):
            raise RuntimeError("No reference KKT face found")
        z = np.maximum(z_center[indices[0]], 0)
        assert np.linalg.norm(z * (self.M @ z + self.B @ u + self.c), np.inf) < 1e-7 * scale
        return z, indices


def generate_model(n, condition, seed):
    rng = np.random.default_rng(73000 + 100 * n + 10 * int(condition) + seed)
    Q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    M = (Q * np.linspace(1, condition, n)) @ Q.T
    B = rng.normal(size=(n, 3)) / np.sqrt(n)
    c = rng.uniform(-0.35, 0.35, size=n)
    target = rng.uniform(-1.2, 1.2, size=3)
    u0 = rng.uniform(-0.5, 0.5, size=3)
    return M, B, c, target, u0


def support_mask(z):
    return sum(1 << int(i) for i in np.flatnonzero(z > 0))


def kkt_residual(M, B, c, u, z):
    F = M @ z + B @ u + c
    free = z > 0
    return float(np.max(np.where(free, np.abs(F), np.maximum(-F, 0))))


def objective(M, B, c, target, u, z, lam):
    return float(-0.5 * z @ M @ z - (B @ u + c) @ z + .5 * lam * np.sum((u - target) ** 2))


def run_reference(model, reference, iterations=60):
    M, B, c, target, u = model
    u = u.copy()
    lam = .4
    alpha = .9 / (lam + np.linalg.norm(B, 2) ** 2 / np.linalg.eigvalsh(M)[0])
    for _ in range(iterations):
        z, _ = reference.reference(u)
        g = lam * (u - target) - B.T @ z
        u = np.clip(u - alpha * g, -.5, .5)
    z, _ = reference.reference(u)
    return objective(M, B, c, target, u, z, lam)


def run_policy(model, reference, method, iterations=60, eta=.20):
    M, B, c, target, u0 = model
    n = len(c)
    lam = .4
    mu = np.linalg.eigvalsh(M)[0]
    gamma = .9 / np.linalg.eigvalsh(M)[-1]
    alpha = .9 / (lam + np.linalg.norm(B, 2) ** 2 / mu)
    cache = FaceCache(M, B, c)
    setup_start = perf_counter()
    if method == "CF-E":
        cache.prepare_all()
    setup_time = perf_counter() - setup_start
    u, z = u0.copy(), np.zeros(n)
    iteration_work = residual_checks = 0
    online_time = envelope_time = 0.
    rows = []
    min_slack = 0.
    acceptance_failures = 0
    for k in range(iterations):
        timer = perf_counter()
        used = checks = 0
        candidate_count = 0
        eps = np.nan
        certificate_met = False
        prescribed = 4 if method == "FW4" else min(4 + k // 3, 30)
        while True:
            batch = prescribed if method in ["FW4", "Scheduled"] else 2
            for _ in range(batch):
                z = np.maximum(z - gamma * (M @ z + B @ u + c), 0)
            used += batch
            if method not in ["FW4", "Scheduled"]:
                natural = float(np.linalg.norm(z - np.maximum(z - gamma * (M @ z + B @ u + c), 0)))
                checks += 1
            mask = support_mask(z)
            item = cache.get(mask)
            g = lam * (u - target) + item[8] @ u + item[9]
            action = np.clip(u - alpha * g, -.5, .5)
            if method.startswith("CF"):
                start = perf_counter()
                delta_raw = kkt_residual(M, B, c, u, z)
                # Inflate, rather than shrink, the class for floating-point screening.
                guard = 1e-10 * (1 + np.linalg.norm(M, np.inf) * np.linalg.norm(z, np.inf)
                                 + np.linalg.norm(B @ u + c, np.inf))
                delta = delta_raw + guard
                if method == "CF-E":
                    candidates = cache.screen(u, delta)
                else:
                    candidates, uncertain = cache.screened_subset(u, z, delta, mu)
                if candidates is None:
                    envelope_time += perf_counter() - start
                    if used >= 1200:
                        raise RuntimeError("Face construction deferred at iteration cap")
                    continue
                if mask not in candidates:
                    raise AssertionError("Observed face removed by screen")
                if method == "CF-E":
                    gradients = lam * (u - target) + cache.gradient_matrix[candidates] @ u + cache.gradient_offset[candidates]
                else:
                    gradients = np.array([lam * (u - target) + cache.get(int(face))[8] @ u
                                          + cache.get(int(face))[9] for face in candidates])
                actions = np.clip(u - alpha * gradients, -.5, .5)
                action_bound = float(np.max(np.linalg.norm(actions - action, axis=1)))
                gradient_bound = float(np.max(np.linalg.norm(gradients - g, axis=1)))
                eps = 2 * alpha * gradient_bound if method == "CF-G" else action_bound
                candidate_count = len(candidates)
                certificate_met = eps <= eta * np.linalg.norm(action - u)
                envelope_time += perf_counter() - start
                stop = certificate_met
            elif method == "FR1e-4":
                stop = natural <= 1e-4
            else:
                stop = True
            if stop or used >= 1200:
                break
        online_time += perf_counter() - timer
        if method in ["FW4", "Scheduled"]:
            natural = float(np.linalg.norm(z - np.maximum(z - gamma * (M @ z + B @ u + c), 0)))
        iteration_work += used
        residual_checks += checks

        # Reference evaluations and checks occur outside every timed online block.
        exact_z, exact_faces = reference.reference(u)
        exact_g = lam * (u - target) - B.T @ exact_z
        exact_action = np.clip(u - alpha * exact_g, -.5, .5)
        exact_mapping = np.linalg.norm((u - exact_action) / alpha)
        update_error = np.linalg.norm(exact_action - action)
        if method.startswith("CF"):
            if not set(exact_faces).intersection(candidates):
                raise AssertionError("Reference face removed by screen")
            if update_error > action_bound + 1e-8:
                raise AssertionError("Action enclosure does not cover reference")
            if np.linalg.norm(exact_g - g) > gradient_bound + 1e-8:
                raise AssertionError("Gradient enclosure does not cover reference")
            if not certificate_met:
                acceptance_failures += 1
            min_slack = min(min_slack, action_bound - update_error)
        rows.append(dict(method=method, iteration=k + 1, iteration_work=iteration_work,
                         total_projections=iteration_work + residual_checks,
                         inner_iterations=used, residual_checks=checks, natural_residual=natural,
                         candidate_faces=candidate_count, action_bound=action_bound if method.startswith("CF") else np.nan,
                         exact_update_error=float(update_error), exact_stationarity=float(exact_mapping),
                         accepted=int(certificate_met) if method.startswith("CF") else -1))
        u = action
    final_z, _ = reference.reference(u)
    final_g = lam * (u - target) - B.T @ final_z
    final_mapping = np.linalg.norm((u - np.clip(u - alpha * final_g, -.5, .5)) / alpha)
    return dict(method=method, iteration_work=iteration_work, residual_checks=residual_checks,
                total_projections=iteration_work + residual_checks, setup_sec=setup_time,
                online_sec=online_time, envelope_sec=envelope_time,
                total_sec=setup_time + online_time, stationarity=float(final_mapping),
                objective=objective(M, B, c, target, u, final_z, lam),
                cap_failures=acceptance_failures,
                multi_face_accepts=sum(r["accepted"] == 1 and r["candidate_faces"] > 1 for r in rows),
                screened_face_mean=float(np.mean([r["candidate_faces"] for r in rows])),
                computed_face_blocks=len(cache.items),
                coverage_slack=min_slack), rows


def validate_screen():
    rng = np.random.default_rng(811)
    count = 0
    for n in [3, 6, 10]:
        model = generate_model(n, 10, 2)
        M, B, c, _, _ = model
        cache = FaceCache(M, B, c)
        cache.prepare_all()
        for _ in range(100):
            u = rng.uniform(-.5, .5, 3)
            perturbation = rng.normal(scale=.04, size=n)
            shifted = FaceCache(M, B, c - perturbation)
            shifted.prepare_all()
            z, faces = shifted.reference(u)
            delta = np.linalg.norm(perturbation, np.inf) + 1e-9
            candidates = cache.screen(u, delta)
            assert set(faces).intersection(candidates)
            # No enumeration budget in this test: compare the status screen
            # against independently solved perturbed KKT systems.
            subset, _ = cache.screened_subset(u, z, delta, 1.0, max_uncertain=n)
            assert set(faces).intersection(subset)
            _, original_faces = cache.reference(u)
            assert set(original_faces).intersection(subset)
            count += 1
    print(f"Perturbed-KKT screen checks: {count} passed", flush=True)
    for n in [3, 6, 10]:
        M, B, c, target, _ = generate_model(n, 10, 2)
        cache = FaceCache(M, B, c)
        cache.prepare_all()
        for _ in range(10):
            u = rng.uniform(-.5, .5, 3)
            z, _ = cache.reference(u)
            gradient = .4 * (u - target) - B.T @ z
            h = 1e-5
            for j in range(3):
                step = np.eye(3)[j] * h
                z_plus, _ = cache.reference(u + step)
                z_minus, _ = cache.reference(u - step)
                difference = (objective(M, B, c, target, u + step, z_plus, .4)
                              - objective(M, B, c, target, u - step, z_minus, .4)) / (2 * h)
                assert abs(difference - gradient[j]) <= 1e-6
    print("Reduced-objective finite-difference checks: 90 passed", flush=True)


def summarize(rows):
    out = []
    for n in sorted({r["n"] for r in rows}):
        for condition in sorted({r["condition"] for r in rows}):
            for method in METHODS:
                group = [r for r in rows if r["n"] == n and r["condition"] == condition and r["method"] == method]
                row = dict(n=n, condition=condition, method=method, seeds=len(group))
                for metric in ["iteration_work", "total_projections", "setup_sec", "online_sec",
                               "envelope_sec", "total_sec", "stationarity", "gap_to_reference",
                               "multi_face_accepts", "screened_face_mean", "computed_face_blocks"]:
                    values = [r[metric] for r in group]
                    row[metric + "_mean"] = float(np.mean(values))
                    row[metric + "_sd"] = float(np.std(values, ddof=1)) if len(group) > 1 else 0.
                row["cap_failures"] = sum(r["cap_failures"] for r in group)
                out.append(row)
    return out


def plot(summary):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "ps.fonttype": 42, "pdf.fonttype": 42})
    colors = ["#0072B2", "#CC79A7", "#D55E00", "#009E73", "#333333", "#8C564B"]
    styles = ["-", ":", "--", "-.", "--", ":"]
    markers = ["o", "D", "s", "^", "x", "v"]
    fig, axes = plt.subplots(3, 1, figsize=(4.8, 6.0), constrained_layout=True)
    for i, method in enumerate(METHODS):
        group = [r for r in summary if r["condition"] == 10 and r["method"] == method]
        x = [r["n"] for r in group]
        for ax, metric in zip(axes, ["total_projections", "stationarity", "total_sec"]):
            mean = np.array([r[metric + "_mean"] for r in group])
            sd = np.array([r[metric + "_sd"] for r in group])
            ax.plot(x, np.maximum(mean, 1e-14), linestyle=styles[i], marker=markers[i],
                    color=colors[i], label=method, markersize=4)
            if metric != "stationarity":
                ax.errorbar(x, mean, yerr=sd, fmt="none", color=colors[i], capsize=2)
    axes[0].set(ylabel="All lower projections", title="(a) Work including residual checks")
    axes[1].set(ylabel="Final stationarity", yscale="log", title="(b) Reference-based accuracy")
    axes[2].set(ylabel="Total time (s)", yscale="log", title="(c) Time including face setup")
    axes[0].legend(ncol=3, fontsize=7, framealpha=1)
    for ax in axes:
        ax.set_xlabel("Lower dimension")
        ax.set_xticks([r["n"] for r in summary if r["condition"] == 10 and r["method"] == METHODS[0]])
        ax.grid(color="0.86", linewidth=.5)
    fig.savefig(HERE / "LegacyFig3.eps", format="eps", bbox_inches="tight")
    fig.savefig(HERE / "LegacyFig3.pdf", bbox_inches="tight")
    fig.savefig(OUT / "LegacyFig3.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    if args.plot_only:
        with (OUT / "compatible_face_summary.csv").open(newline="") as stream:
            summary = list(csv.DictReader(stream))
        for row in summary:
            for key in row:
                if key != "method":
                    row[key] = float(row[key])
        plot(summary)
        return
    with threadpool_limits(limits=1):
        if args.validate_only:
            validate_screen()
            return
        ns = [6] if args.quick else [6, 8, 10]
        seeds = range(2) if args.quick else range(5)
        detail, all_histories = [], []
        for n in ns:
            for condition in [3, 10]:
                for seed in seeds:
                    model = generate_model(n, condition, seed)
                    reference = FaceCache(*model[:3])
                    reference.prepare_all()
                    ref_objective = run_reference(model, reference)
                    for method in METHODS:
                        result, history = run_policy(model, reference, method)
                        result.update(n=n, condition=condition, seed=seed,
                                      gap_to_reference=abs(result["objective"] - ref_objective))
                        detail.append(result)
                        for r in history:
                            r.update(n=n, condition=condition, seed=seed)
                        all_histories.extend(history)
                print(f"n={n}, condition={condition} complete", flush=True)
        summary = summarize(detail)
        write_csv("compatible_face_detail.csv", detail)
        write_csv("compatible_face_summary.csv", summary)
        write_csv("compatible_face_history.csv", all_histories)
        plot(summary)
        metadata = dict(python=sys.version, platform=platform.platform(), processor=platform.processor(),
                        numpy=np.__version__, matplotlib=matplotlib.__version__,
                        threadpools=threadpool_info(), seeds=list(seeds), ns=ns,
                        upper_dimension=3, outer_iterations=60, eta=.20,
                        max_uncertain_for_lazy_enclosure=6,
                        diagnostic_time_included=False, projection_checks_counted=True)
        (OUT / "compatible_face_environment.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print("Candidate-screen enclosure violations: 0; all reference checks passed.", flush=True)
        for row in summary:
            print(row["n"], row["condition"], row["method"],
                  "work", round(row["total_projections_mean"]), "stat", f"{row['stationarity_mean']:.2e}",
                  "time", f"{row['total_sec_mean']:.3f}", "multi", row["multi_face_accepts_mean"],
                  "cap", row["cap_failures"], flush=True)


if __name__ == "__main__":
    main()
