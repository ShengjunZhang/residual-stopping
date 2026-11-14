"""Build JAMC figures from archived data and a common-warm-start VI comparison."""

from pathlib import Path
import csv
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "experiments"
sys.path.insert(0, str(SOURCE))
from run_parallel_toll_demo import run_case, run_residual_balanced_case
from run_residual_balanced_ablation import base_model, conditioned_model, run_fixed_residual_case

DATA = SOURCE / "results"
OUT = HERE / "numerical_results"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "ps.fonttype": 42, "pdf.fonttype": 42,
    "axes.spines.top": True, "axes.spines.right": True,
})
COLORS = ["#0072B2", "#CC79A7", "#D55E00", "#009E73"]
STYLES = ["-", ":", "--", "-."]
MARKERS = ["o", "D", "s", "^"]
METHODS = ["FW4", "FR1e-3", "Scheduled", "RB0.20"]


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(name, rows):
    with (OUT / name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def polish(axes):
    for ax in np.atleast_1d(axes).flat:
        ax.grid(True, which="major", color="0.87", linewidth=0.5)
        for spine in ax.spines.values():
            spine.set_visible(True)
        ax.tick_params(direction="out")
        formatter = FuncFormatter(lambda x, _: f"1e{int(round(np.log10(x)))}" if x > 0 else "")
        if ax.get_yscale() == "log":
            ax.yaxis.set_major_formatter(formatter)


def save(fig, number):
    # Opaque vector EPS avoids rasterized lettering and PS transparency changes.
    fig.savefig(HERE / f"LegacyFig{number}.eps", format="eps", bbox_inches="tight")
    fig.savefig(HERE / f"LegacyFig{number}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"LegacyFig{number}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def compatible_faces():
    u, alpha = 0.2, 0.25
    exact_gradient = 0.1 * u + u - 1
    exact_action = np.clip(u - alpha * exact_gradient, 0, 1)
    rows = []
    for r in [0.2, 0.19, 0.1, 0.02]:
        lo, hi = max(0, u - r), min(1, u + r)
        observed = lo
        observed_gradient = 0.1 * u if observed == 0 else 0.1 * u + observed - 1
        observed_action = np.clip(u - alpha * observed_gradient, 0, 1)
        # The interior-face closure contributes its endpoints; boundary faces
        # contribute their own zero state derivative.
        vertex_gradients = [0.1 * u + lo - 1, 0.1 * u + hi - 1]
        if lo == 0 or hi == 1:
            vertex_gradients.append(0.1 * u)
        enclosure = max(abs(np.clip(u - alpha * g, 0, 1) - observed_action)
                        for g in vertex_gradients)
        actual_error = abs(exact_action - observed_action)
        grid = np.linspace(lo, hi, 1001)
        gradients = np.where((grid > 0) & (grid < 1), 0.1 * u + grid - 1, 0.1 * u)
        actions = np.clip(u - alpha * gradients, 0, 1)
        assert np.max(np.abs(actions - observed_action)) <= enclosure + 1e-12
        assert actual_error <= enclosure + 1e-12
        assert abs(observed - u) <= r + 1e-12
        rows.append(dict(residual=r, observed_state=observed,
                         exact_update_error=float(actual_error),
                         union_enclosure=float(enclosure),
                         candidate_faces=2 if lo == 0 else 1,
                         accepted=int(enclosure <= .2 * abs(observed_action - u))))
    write_csv("compatible_face_check.csv", rows)
    assert rows[0]["exact_update_error"] > 0
    print("Compatible-face grid checks passed", rows, flush=True)


def calibration():
    rows = read_csv(DATA / "hypergradient_residual_diagnostic.csv")
    valid = [r for r in rows if float(r["lower_residual"]) > 1e-14
             and float(r["state_error"]) > 0 and float(r["gradient_bias"]) > 0]
    a = np.array([[int(r["sample"]), float(r["lower_residual"]),
                   float(r["state_error"]), float(r["gradient_bias"])] for r in valid])
    train = a[:, 0] < np.median(np.unique(a[:, 0]))
    c_z = np.max(a[train, 2] / a[train, 1])
    c_g = np.max(a[train, 3] / a[train, 1])
    ratios = [a[~train, 2] / (c_z * a[~train, 1]), a[~train, 3] / (c_g * a[~train, 1])]
    fig, axes = plt.subplots(2, 1, figsize=(4.8, 4.7), constrained_layout=True)
    for i, (column, coefficient, label, marker) in enumerate([
        (2, c_z, "State error", "o"), (3, c_g, "Hypergradient error", "s")
    ]):
        axes[0].loglog(a[:, 1], a[:, column], marker, color=COLORS[2 * i],
                      markersize=2, label=label)
        x = np.array([a[:, 1].min(), a[:, 1].max()])
        axes[0].loglog(x, coefficient * x, STYLES[2 * i], color=COLORS[2 * i])
        axes[1].plot(np.linspace(0, 1, len(ratios[i])), np.sort(ratios[i]),
                     STYLES[2 * i], color=COLORS[2 * i], label=label)
    axes[0].set(xlabel="Natural residual", ylabel="Error", title="(a) Residual transfer")
    axes[1].set(xlabel="Ordered held-out sample (fraction)", ylabel="Error / fitted bound",
                title="(b) Held-out coverage", ylim=(0, 1.08))
    axes[1].axhline(1, color="0.25", linestyle=":")
    for ax in axes:
        ax.legend(framealpha=1)
    polish(axes)
    save(fig, 1)
    report = dict(state_coefficient=float(c_z), gradient_coefficient=float(c_g),
                  calibration_pairs=int(train.sum()), test_pairs=int((~train).sum()),
                  state_coverage=float(np.mean(ratios[0] <= 1)),
                  gradient_coverage=float(np.mean(ratios[1] <= 1)))
    write_csv("calibration_summary.csv", [report])
    print(report, flush=True)


def histories(model, iterations):
    return {
        "FW4": run_case(model, "FW4", 4, iterations, warm_start=True),
        "FR1e-3": run_fixed_residual_case(model, "FR1e-3", 1e-3, iterations),
        "Scheduled": run_case(model, "Scheduled", lambda k: min(1 + k // 4, 40),
                              iterations, warm_start=True),
        "RB0.20": run_residual_balanced_case(model, "RB0.20", iterations,
                                             batch_size=2, theta=0.20, max_lower_steps=160),
    }


def comparison():
    base = histories(base_model(), 100)
    write_csv("vi_base_history.csv", [r for hist in base.values() for r in hist])
    rows = []
    kappas = [1.25, 1.5, 2, 3, 4, 6, 8]
    for kappa in kappas:
        for seed in range(8):
            for method, hist in histories(conditioned_model(kappa, seed), 80).items():
                rows.append(dict(condition_ratio=kappa, seed=seed, method=method,
                                 stationarity=hist[-1]["gradmap_norm"],
                                 residual=hist[-1]["lower_residual"],
                                 work=hist[-1]["total_lower_steps"]))
        print(f"completed common-warm-start condition ratio {kappa}", flush=True)
    write_csv("vi_condition_detail.csv", rows)
    summary = []
    fig, axes = plt.subplots(3, 1, figsize=(4.8, 6.5), constrained_layout=True)
    for i, method in enumerate(METHODS):
        hist = base[method]
        axes[0].semilogy([r["total_lower_steps"] for r in hist],
                         [max(r["gradmap_norm"], 1e-14) for r in hist],
                         STYLES[i], color=COLORS[i], label=method)
        groups = []
        for kappa in kappas:
            group = [r for r in rows if r["method"] == method and r["condition_ratio"] == kappa]
            item = dict(method=method, condition_ratio=kappa)
            for metric in ["stationarity", "work"]:
                values = [r[metric] for r in group]
                item.update({f"{metric}_median": float(np.median(values)),
                             f"{metric}_q1": float(np.quantile(values, .25)),
                             f"{metric}_q3": float(np.quantile(values, .75))})
            groups.append(item)
            summary.append(item)
        for ax, metric in zip(axes[1:], ["stationarity", "work"]):
            med = np.array([r[f"{metric}_median"] for r in groups])
            lo = np.array([r[f"{metric}_q1"] for r in groups])
            hi = np.array([r[f"{metric}_q3"] for r in groups])
            ax.errorbar(kappas, med, yerr=[med - lo, hi - med], color=COLORS[i],
                        linestyle=STYLES[i], marker=MARKERS[i], markersize=3,
                        capsize=2, label=method)
    axes[0].set(xlabel="Lower iteration projections", ylabel="Stationarity",
                title="(a) Base instance")
    axes[1].set(xlabel="Condition ratio", ylabel="Median stationarity",
                title="(b) Conditioning and accuracy", yscale="log")
    axes[2].set(xlabel="Condition ratio", ylabel="Median lower work",
                title="(c) Conditioning and work")
    axes[0].legend(ncol=2, framealpha=1)
    polish(axes)
    save(fig, 2)
    write_csv("vi_condition_summary.csv", summary)
    for method, hist in base.items():
        print(method, hist[-1], flush=True)


def traffic():
    rows = read_csv(DATA / "tntp_oracle_policy_benchmark_summary.csv")
    networks = ["Sioux Falls", "Eastern Massachusetts", "Anaheim"]
    keys = [("fixed-work warm", "10 steps"), ("fixed residual", "1e-02"),
            ("fixed residual", "1e-03"), ("scheduled warm", "-"),
            ("residual-balanced", "eta=0.10"), ("residual-balanced", "eta=0.20")]
    labels = ["FW10", "FR1e-2", "FR1e-3", "Scheduled", "RB0.10", "RB0.20"]
    colors = ["#0072B2", "#CC79A7", "#8C564B", "#D55E00", "#009E73", "#333333"]
    styles = ["-", ":", "--", "-.", "-", "--"]
    markers = ["o", "D", "s", "v", "^", "x"]
    fig, axes = plt.subplots(3, 1, figsize=(4.8, 6.0), constrained_layout=True)
    for i, (policy, parameter) in enumerate(keys):
        group = [next(r for r in rows if r["network"] == n and r["policy"] == policy
                      and r["parameter"] == parameter) for n in networks]
        for ax, metric in zip(axes, ["steps", "gap", "time"]):
            med = np.array([float(r[f"{metric}_mean"]) for r in group])
            sd = np.array([float(r[f"{metric}_sd"]) for r in group])
            ax.plot(np.arange(3), med, color=colors[i], linestyle=styles[i],
                    marker=markers[i], markersize=4, label=labels[i])
            if metric != "gap":
                ax.errorbar(np.arange(3), med, yerr=sd, color=colors[i],
                            fmt="none", capsize=2)
    axes[0].set(ylabel="Mean iteration projections", title="(a) Lower work")
    axes[1].set(ylabel="Mean gap to reference", title="(b) Numerical accuracy", yscale="log")
    axes[2].set(ylabel="Mean evaluation time (s)", title="(c) Diagnostic-inclusive time")
    for ax in axes:
        ax.set_xticks(np.arange(3), ["Sioux Falls", "Eastern Mass.", "Anaheim"])
    axes[0].legend(framealpha=1, fontsize=7, ncol=3)
    polish(axes)
    save(fig, 4)


if __name__ == "__main__":
    if "--traffic-only" in sys.argv:
        traffic()
    elif "--face-only" in sys.argv:
        compatible_faces()
    else:
        compatible_faces()
        calibration()
        comparison()
        traffic()
