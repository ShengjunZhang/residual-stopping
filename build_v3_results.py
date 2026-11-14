"""Render publication figures and table fragments from frozen benchmark CSVs."""

from pathlib import Path
import csv
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "numerical_results"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.labelsize": 10, "axes.titlesize": 10.5,
                     "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
                     "legend.fontsize": 9.5,
                     "ps.fonttype": 42, "pdf.fonttype": 42,
                     "axes.spines.top": True, "axes.spines.right": True})
PAIRED = ["RG", "W-RG", "EI-CG", "EI-CGL"]
COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#333333", "#8C564B"]
STYLES = ["--", ":", "-", "-.", "--", ":"]
MARKERS = ["o", "s", "^", "D", "x", "v"]


def read(name):
    with (OUT / name).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def polish(axes):
    for ax in np.atleast_1d(axes).flat:
        ax.grid(color=".86", linewidth=.5)
        for spine in ax.spines.values():
            spine.set_visible(True)


def save(fig, number):
    for ext in ["eps", "pdf"]:
        fig.savefig(HERE / f"Fig{number}.{ext}", bbox_inches="tight")
    fig.savefig(OUT / f"Fig{number}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def mean_sd(values):
    return float(np.mean(values)), float(np.std(values, ddof=1))


def cell(mean, sd, scientific=False):
    if scientific:
        def sci(x):
            a, e = f"{x:.2e}".split("e")
            return f"${a}{{\\times}}10^{{{int(e)}}}$"
        return "\\makecell{" + sci(mean) + "\\\\(" + sci(sd) + ")}"
    return f"\\makecell{{{mean:.2f}\\\\({sd:.2f})}}"


def get(summary, n, coupling, eta, method):
    return next(r for r in summary if int(r["n"]) == n and r["coupling"] == str(coupling)
                and float(r["eta"]) == eta and r["method"] == method)


def plot_paired(summary):
    fig, axes = plt.subplots(3, 1, figsize=(4.8, 6.), constrained_layout=True)
    for i, method in enumerate(PAIRED):
        rows = [get(summary, n, 5, .2, method) for n in [200, 1000, 5000]]
        for ax, metric in zip(axes, ["total_projections", "stationarity", "total_sec"]):
            vals = [float(r[metric + "_mean"]) for r in rows]
            sd = [float(r[metric + "_sd"]) for r in rows]
            ax.plot([200, 1000, 5000], vals, color=COLORS[i], linestyle=STYLES[i],
                    marker=MARKERS[i], markersize=4, label=method)
            if metric != "stationarity":
                ax.errorbar([200, 1000, 5000], vals, yerr=sd, color=COLORS[i], fmt="none", capsize=2)
    axes[0].set(ylabel="Lower projections", title="(a) Same gradient-control condition")
    axes[1].set(ylabel="Final stationarity", title="(b) Reference-based accuracy", yscale="log")
    axes[2].set(ylabel="Total time (s)", title="(c) Online time including setup")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xticks([200, 1000, 5000], ["200", "1000", "5000"])
        ax.set_xlabel("Lower dimension")
    axes[0].legend(ncol=2, framealpha=1)
    polish(axes)
    save(fig, 1)


def plot_network(summary, setup):
    methods = ["FR1e-4", "FR1e-6", "Scheduled", "RG", "EI-A", "EI-L"]
    fig, axes = plt.subplots(3, 1, figsize=(4.8, 6.), constrained_layout=True)
    for i, method in enumerate(methods):
        rows = [get(summary, s["paths"], s["network"], .2, method) for s in setup]
        for ax, metric in zip(axes, ["total_projections", "gap_to_reference", "total_sec"]):
            vals = np.array([float(r[metric + "_mean"]) for r in rows])
            sd = np.array([float(r[metric + "_sd"]) for r in rows])
            plotted = np.maximum(vals, 1e-14)
            ax.plot(range(3), plotted, color=COLORS[i], linestyle=STYLES[i], marker=MARKERS[i],
                    markersize=4, label=method)
            if metric != "gap_to_reference":
                ax.errorbar(range(3), vals, yerr=sd, color=COLORS[i], fmt="none", capsize=2)
    axes[0].set(ylabel="All lower projections", title="(a) Lower work")
    axes[1].set(ylabel="Gap to reference", title="(b) Finite upper-run comparison", yscale="log")
    axes[2].set(ylabel="Total time (s)", title="(c) Online time including setup", yscale="log")
    for ax in axes:
        ax.set_xticks(range(3), ["Sioux Falls", "Eastern Mass.", "Anaheim"])
    axes[0].legend(ncol=3, framealpha=1)
    polish(axes)
    save(fig, 2)


def plot_eta(core, paired):
    fig, axes = plt.subplots(2, 1, figsize=(4.8, 4.3), constrained_layout=True)
    for i, (method, summary) in enumerate([("EI-G", core), ("EI-CG", paired), ("EI-CGL", paired)]):
        rows = [get(summary, 5000, 5, eta, method) for eta in [.1, .2, .4]]
        for ax, metric in zip(axes, ["total_projections", "gap_to_reference"]):
            vals = np.maximum([float(r[metric + "_mean"]) for r in rows], 1e-14)
            ax.plot([.1, .2, .4], vals, color=COLORS[i], linestyle=STYLES[i], marker=MARKERS[i], label=method)
    axes[0].set(ylabel="Mean lower projections", title="(a) Accuracy allocation")
    axes[1].set(ylabel="Mean gap to reference", yscale="log", title="(b) Upper-trajectory difference")
    for ax in axes:
        ax.set_xlabel("Balance parameter eta")
        ax.set_xticks([.1, .2, .4])
    axes[0].legend(ncol=3, framealpha=1)
    polish(axes)
    save(fig, 3)


def plot_locality(history):
    group = [r for r in history if r["method"] == "EI-CG" and r["n"] == "5000"
             and r["coupling"] == "5" and r["seed"] == "0" and float(r["eta"]) == .2]
    fig, axes = plt.subplots(2, 1, figsize=(4.8, 4.3), constrained_layout=True)
    x = [int(r["outer"]) for r in group]
    axes[0].plot(x, [int(r["global_uncertainty"]) for r in group], color=COLORS[0], label="Global unresolved")
    axes[0].plot(x, [int(r["uncertainty"]) for r in group], "--", color=COLORS[1], label="Relevant unresolved")
    axes[0].set(ylabel="Unresolved indices", title="(a) Accepted status intervals")
    axes[1].plot(x, [int(r["local_dimension"]) for r in group], color=COLORS[2], label="Enclosure systems")
    axes[1].axhline(5000, linestyle=":", color=".35", label="Full lower dimension")
    axes[1].set(ylabel="System dimension", yscale="log", title="(b) Changing principal systems")
    for ax in axes:
        ax.set_xlabel("Upper iteration")
        ax.legend(framealpha=1)
    polish(axes)
    save(fig, 4)


def fragments(core, paired, network, setup, history):
    out = {}
    lines = [r"\begin{table}[htbp]", r"\caption{Paired gradient-controlled results at coupling 5}\label{tab:paired}",
             r"\centering\footnotesize\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{llrrrr}",
             r"\toprule", r"$n$ & Policy & Projections & Time (s) & Stationarity & Gap to reference\\", r"\midrule"]
    for n in [200, 1000, 5000]:
        for method in PAIRED:
            row = get(paired, n, 5, .2, method)
            cells = [cell(float(row[k + "_mean"]), float(row[k + "_sd"]), k in ["stationarity", "gap_to_reference"])
                     for k in ["total_projections", "total_sec", "stationarity", "gap_to_reference"]]
            lines.append(f"{n} & {method} & " + " & ".join(cells) + r"\\")
        if n != 5000:
            lines.append(r"\midrule")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    out["PAIRED"] = "\n".join(lines)
    cg, rg, wrg = [get(paired, 5000, 5, .2, m) for m in ["EI-CG", "RG", "W-RG"]]
    projections = float(cg["total_projections_mean"])
    speed = (1 - float(cg["total_sec_mean"]) / float(rg["total_sec_mean"])) * 100
    wspeed = (1 - float(cg["total_sec_mean"]) / float(wrg["total_sec_mean"])) * 100
    reduction = (1 - projections / float(rg["total_projections_mean"])) * 100
    out["SPARSE"] = (
        f"Table~\\ref{{tab:paired}} compares methods satisfying the same observed-gradient error condition. "
        f"At $n=5000$, coupling 5, EI-CG uses {projections:.1f} projections compared with "
        f"{float(rg['total_projections_mean']):.1f} for RG, a reduction of {reduction:.1f}\\%. "
        f"Mean online time falls from {float(rg['total_sec_mean']):.3f} to "
        f"{float(cg['total_sec_mean']):.3f} s ({speed:.1f}\\%). Against W-RG the time reduction is "
        f"{wspeed:.1f}\\%. The final stationarity norms and reference gaps are close. "
        "Figure~\\ref{fig:sparse} compares work, stationarity, and online time across all three "
        "dimensions. Full norm-based, fixed-work, scheduled, and fixed-residual comparisons "
        "accompany Online Resource 1."
    )
    lines = [r"\begin{table}[htbp]", r"\caption{Elastic-demand candidate-path models}\label{tab:elasticsetup}",
             r"\centering\small", r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Network & Nodes & Links & Positive ODs & Paths\\", r"\midrule"]
    for s in setup:
        lines.append(f"{s['network']} & {s['nodes']} & {s['links']} & {s['retained_ODs']} & {s['paths']}" + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    out["NETWORK_SETUP"] = "\n".join(lines)
    lines = [r"\begin{table}[htbp]", r"\caption{Elastic-demand results over five seeds}\label{tab:elasticresults}",
             r"\centering\footnotesize\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{llrrrr}",
             r"\toprule", r"Network & Policy & Projections & Time (s) & Stationarity & Gap to reference\\", r"\midrule"]
    for s, abbreviation in zip(setup, ["SF", "EM", "AN"]):
        for method in ["FR1e-4", "FR1e-6", "Scheduled", "RG", "EI-A", "EI-L"]:
            row = get(network, s["paths"], s["network"], .2, method)
            cells = [cell(float(row[k + "_mean"]), float(row[k + "_sd"]), k in ["stationarity", "gap_to_reference"])
                     for k in ["total_projections", "total_sec", "stationarity", "gap_to_reference"]]
            lines.append(f"{abbreviation} & {method} & " + " & ".join(cells) + r"\\")
        if abbreviation != "AN":
            lines.append(r"\midrule")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    out["NETWORK_RESULTS"] = "\n".join(lines)
    reductions = []
    for s in setup:
        a, fr = [get(network, s["paths"], s["network"], .2, m) for m in ["EI-A", "FR1e-4"]]
        reductions.append(100 * (1 - float(a["total_projections_mean"]) / float(fr["total_projections_mean"])))
    out["NETWORK"] = (
        "Table~\\ref{tab:elasticsetup} gives the realized path counts, and Table~\\ref{tab:elasticresults} "
        "gives means and standard deviations (SF: Sioux Falls; EM: Eastern Massachusetts; AN: Anaheim). "
        "Candidate routes cover every positive-demand OD pair, giving lower problems with "
        + ", ".join(str(s["paths"]) for s in setup[:-1]) + ", and " + str(setup[-1]["paths"]) + " path variables. "
        "EI-A reduces total projections relative to FR $10^{-4}$ by "
        f"{min(reductions):.1f}--{max(reductions):.1f}\\%, with final stationarity norms close "
        "to those of the fixed-residual policies. EI-L reuses the enclosure between refinement "
        "checks. Figure~\\ref{fig:elastic} and Table~\\ref{tab:elasticresults} give the corresponding "
        "online times, including principal-system factorization. All objective differences are evaluated "
        "against the finite tight-lower run. Values below $10^{-14}$ are clipped only in logarithmic plots."
    )
    low, high = [get(paired, 5000, 5, eta, "EI-CG") for eta in [.1, .4]]
    out["ETA"] = (
        f"For EI-CG at $n=5000$, coupling 5, mean work changes from "
        f"{float(low['total_projections_mean']):.1f} at $\\eta=0.10$ to "
        f"{float(high['total_projections_mean']):.1f} at $\\eta=0.40$. "
        "The gaps remain small on this group, although their variation need not be monotone in "
        "$\\eta$: changing lower accuracy changes every subsequent upper point. "
        "The iteration histories identify when the relative part or the floor in "
        "\\eqref{eq:flooracceptance} is used."
    )
    group = [r for r in history if r["method"] == "EI-CG" and r["n"] == "5000"
             and r["coupling"] == "5" and r["seed"] == "0" and float(r["eta"]) == .2]
    max_global = max(int(r["global_uncertainty"]) for r in group)
    max_local = max(int(r["local_dimension"]) for r in group)
    out["LOCALITY"] = (
        f"In the run shown in Figure~\\ref{{fig:locality}}, accepted updates retain up to "
        f"{max_global} globally unresolved statuses, while the changing enclosure systems "
        f"have at most {max_local} variables. "
        "The comparison step still uses a full-dimensional solve with a fixed factorization, "
        "whose setup and solve costs are included. The reduction applies to the changing "
        "principal systems, and therefore avoids equating uncertain-state dimension with "
        "the amount of work required to determine the upper update."
    )
    for key, content in out.items():
        (OUT / ("text_" + key.lower() + ".texfrag")).write_text(content + "\n", encoding="utf-8")
    print(out["SPARSE"], out["NETWORK"], out["LOCALITY"], sep="\n", flush=True)


def main():
    core = read("sparse_enclosure_summary.csv")
    paired = read("sparse_paired_summary.csv")
    network = read("elastic_network_summary.csv")
    setup = json.loads((OUT / "elastic_network_setup.json").read_text(encoding="utf-8"))
    history = read("sparse_paired_history.csv")
    plot_paired(paired)
    plot_network(network, setup)
    plot_eta(core, paired)
    plot_locality(history)
    fragments(core, paired, network, setup, history)


if __name__ == "__main__":
    main()
