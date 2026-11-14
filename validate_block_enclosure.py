"""Independent enumeration, support witnesses, and a structured gap diagnostic."""

from pathlib import Path
from itertools import combinations
import csv
import json
import time
import platform
import sys
import numpy as np
import scipy
from scipy import sparse
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from block_enclosure import block_geometry, centered_support, projected_bound

HERE = Path(__file__).resolve().parent
OUT = HERE / "numerical_results"


def principal(M, J):
    P = np.zeros_like(M)
    if len(J):
        P[np.ix_(J, J)] = np.linalg.solve(M[np.ix_(J, J)], np.eye(len(J)))
    return P


def subsets(indices):
    for length in range(len(indices) + 1):
        yield from combinations(indices, length)


def psd_sqrt(M):
    values, vectors = np.linalg.eigh(M)
    return (vectors * np.sqrt(np.maximum(values, 0.))) @ vectors.T


def validate():
    rng = np.random.default_rng(20261015)
    counts = dict(face_coordinates=0, support_witnesses=0,
                  projected_comparisons=0, nonlinear_coordinates=0)
    violation = 0.
    for free, possible, v in [([], [], np.ones(4)), ([0, 1], [0, 1], np.ones(4)),
                               ([], [0, 1, 2, 3], np.zeros(4))]:
        edge = block_geometry(sparse.eye(4, format="csc"), np.ones((4, 2)), v, free, possible)
        assert np.linalg.norm(edge["widths"]) == 0
        assert np.linalg.norm(edge["global_widths"]) == 0
    for trial in range(100):
        blocks = []
        for size in [2, 3, 3]:
            A = rng.normal(size=(size, size))
            blocks.append(A.T @ A + np.eye(size))
        M = sparse.block_diag([sparse.csc_matrix(b) for b in blocks]).toarray()
        B, v, p = rng.normal(size=(8, 3)), rng.normal(size=8), rng.normal(size=3)
        free = np.flatnonzero(rng.random(8) < .25)
        possible = np.arange(8)
        geo = block_geometry(M, B, v, free, possible, p)
        assert geo["factorizations"] <= 2
        candidates = []
        optional = np.setdiff1d(possible, free)
        for extra in subsets(optional):
            J = np.sort(np.r_[free, extra]).astype(int)
            g = p - B.T @ principal(M, J) @ v
            candidates.append((J, g))
            violation = max(violation, np.max(abs(g - geo["center"]) - geo["widths"]))
            assert np.linalg.norm(g - geo["center"]) <= geo["radius"] + 1e-9
            counts["face_coordinates"] += 3
        assert np.all(geo["widths"] <= geo["global_widths"] + 1e-9)
        for _ in range(20):
            d = rng.normal(size=3)
            # Construct an extremizer independently in each PSD interval.
            T = principal(M, free)
            support_value = float(d @ geo["center"])
            for b in range(geo["blocks"]):
                C = np.flatnonzero(geo["labels"] == b)
                D = principal(M, C) - principal(M, np.intersect1d(C, free))
                root = psd_sqrt(D)
                x, t = root @ v, root @ B @ d
                y = x / 2 if np.linalg.norm(t) < 1e-12 else x / 2 - np.linalg.norm(x) * t / (2 * np.linalg.norm(t))
                if np.linalg.norm(y) < 1e-10:
                    Q = np.zeros_like(D)
                else:
                    Q = np.outer(y, y) / (y @ y)
                assert np.linalg.eigvalsh(Q)[0] >= -1e-8
                assert np.linalg.eigvalsh(Q)[-1] <= 1 + 1e-7
                T += root @ Q @ root
            attained = float(d @ (p - B.T @ T @ v))
            support_value += centered_support(geo, d)
            assert abs(attained - support_value) < 1e-7
            global_support = .5 * np.sqrt(sum(geo["energy"]) * (d @ sum(geo["metrics"]) @ d))
            assert centered_support(geo, d) <= global_support + 1e-8
            counts["support_witnesses"] += 1
            u, observed = rng.uniform(-.5, .5, 3), candidates[rng.integers(len(candidates))][1]
            new = projected_bound(u, observed, geo["center"], geo["widths"], .2)
            old = projected_bound(u, observed, geo["center"], geo["global_widths"], .2)
            assert new <= old + 1e-9
            for _, g in candidates:
                actual = np.linalg.norm(np.clip(u - .2*g, -.5, .5) - np.clip(u - .2*observed, -.5, .5))
                assert actual <= new + 1e-9
            counts["projected_comparisons"] += 1

        # Use a separately enumerated exact equilibrium for value-correction checks.
        q = rng.normal(size=8)
        exact = None
        for J in subsets(possible):
            J = np.array(J, int)
            z = -principal(M, J) @ q
            F = M @ z + q
            outside = np.setdiff1d(possible, J)
            if np.min(z, initial=0.) >= -1e-10 and np.min(F[outside], initial=0.) >= -1e-10:
                exact = z
                break
        assert exact is not None
        # Empty certified-free set covers every support used below.
        gq = block_geometry(M, B, q, [], possible)
        c0 = -gq["center"]
        Sstar = -.5 * exact @ M @ exact - q @ exact
        samples = []
        for _ in range(10):
            J = rng.random(8) < .5
            z = np.where(J, rng.uniform(.01, 1, 8), 0.)
            F = M @ z + q
            residual = np.where(J, F, np.minimum(F, 0.))
            S = -.5 * z @ M @ z - q @ z
            samples.append((z, float(np.linalg.norm(residual)), S))
        dmax = max(s[1] for s in samples)
        rho = .7
        theta = 1 + rho * samples[0][2] / 8
        td = rho * dmax**2 / 16  # M >= I, n=8.
        width = abs(theta) * gq["widths"] + td * (abs(c0) + gq["widths"])
        for z, error, S in samples:
            assert 0 <= Sstar - S + 1e-9 <= error**2/2 + 2e-9
            bJ = B.T @ principal(M, np.flatnonzero(z > 0)) @ q
            g = (1 + rho * S / 8) * bJ
            violation = max(violation, np.max(abs(g - theta*c0) - width))
            counts["nonlinear_coordinates"] += 3
    assert violation < 1e-8
    return dict(seed=20261015, instances=100, degenerate_cases=3, counts=counts,
                maximum_violation=float(max(violation, 0.)),
                tolerance=1e-8, arithmetic="double precision; not an interval-arithmetic proof")


def diagnostic():
    rows = []
    for n in [2, 200, 5000]:
        for epsilon in [.1, .01, .001]:
            for seed in range(5):
                rng = np.random.default_rng(seed)
                # Independent positive scales preserve the exact scalar face hull.
                weights = rng.uniform(.8, 1.2, n//2)
                q = np.tile([-1., -epsilon], n//2) * np.repeat(weights, 2)
                B = np.tile([epsilon, 1.], n//2)[:, None]
                start = time.perf_counter()
                geo = block_geometry(sparse.eye(n, format="csc"), B, q, [], np.arange(n))
                elapsed = time.perf_counter() - start
                exact_radius = float(np.sum(abs(B[:, 0] * q)) / 2)
                assert abs(geo["widths"][0] - exact_radius) <= 1e-9 * (1 + exact_radius)
                assert np.all(B != 0)
                rows.append(dict(n=n, epsilon=epsilon, seed=seed,
                                 unstructured_radius=geo["global_widths"][0],
                                 structured_radius=geo["widths"][0],
                                 radius_ratio=geo["global_widths"][0]/geo["widths"][0],
                                 exact_hull_radius=exact_radius,
                                 seconds=elapsed, factorizations=geo["factorizations"],
                                 right_hand_sides=geo["right_hand_sides"]))
    return rows


def main():
    OUT.mkdir(exist_ok=True)
    with threadpool_limits(limits=1):
        report = validate()
        rows = diagnostic()
    with (OUT / "block_structure_detail.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
    summary = []
    for n in [2, 200, 5000]:
        for e in [.1, .01, .001]:
            group = [r for r in rows if r["n"] == n and r["epsilon"] == e]
            entry = dict(n=n, epsilon=e)
            for name in ["unstructured_radius", "structured_radius", "radius_ratio", "seconds"]:
                values = [r[name] for r in group]
                entry[name + "_mean"] = float(np.mean(values))
                entry[name + "_std"] = float(np.std(values, ddof=1))
            summary.append(entry)
    with (OUT / "block_structure_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary[0]); w.writeheader(); w.writerows(summary)
    report["diagnostic_runs"] = len(rows)
    report["runtime"] = dict(python=sys.version, platform=platform.platform(),
                             numpy=np.__version__, scipy=scipy.__version__, threads=1)
    (OUT / "block_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    table_rows = []
    for row in [r for r in summary if r["n"] == 5000]:
        exponent = round(np.log10(row["epsilon"]))
        cells = []
        for name, digits in [("unstructured_radius", 2), ("structured_radius", 3), ("seconds", 4)]:
            mean, sd = row[name + "_mean"], row[name + "_std"]
            cells.append(r"\makecell{" + f"{mean:.{digits}f}" + r"\\(" + f"{sd:.{digits}f}" + ")}")
        table_rows.append("$10^{" + str(exponent) + "}$ & " + " & ".join(cells) + r"\\")
    (HERE / "BlockRows.texfrag").write_text("\n".join(table_rows) + "\n\\bottomrule\n", encoding="ascii")
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 10,
                         "axes.spines.top": True, "axes.spines.right": True})
    fig, ax = plt.subplots(figsize=(5.6, 3.3))
    for n, color, marker in zip([2, 200, 5000], ["#0072B2", "#D55E00", "#009E73"], ["o", "s", "^"]):
        group = [r for r in summary if r["n"] == n]
        ax.loglog([r["epsilon"] for r in group], [r["radius_ratio_mean"] for r in group],
                  color=color, marker=marker, label=f"n = {n}", linewidth=1.5)
    ax.set(xlabel=r"$\epsilon$", ylabel="Unstructured / structured radius")
    ax.legend(frameon=True, framealpha=1); ax.grid(True, which="major", color=".86")
    fig.tight_layout()
    fig.savefig(HERE / "BlockFig.eps")
    fig.savefig(HERE / "BlockFig.pdf")
    plt.close(fig)
    print(json.dumps(report, indent=2))
    print(json.dumps([r for r in summary if r["n"] == 5000], indent=2))


if __name__ == "__main__":
    main()
