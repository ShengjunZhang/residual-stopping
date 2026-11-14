"""Independent small-system tests of the inverse interval and nonlinear bounds."""

from itertools import combinations
from pathlib import Path
import json
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from threadpoolctl import threadpool_limits
from ellipsoid_enclosure import PrincipalCache, value_enclosure, status_bounds, build_geometry, component_geometry


def embedded_inverse(M, indices):
    out = np.zeros_like(M)
    if len(indices):
        out[np.ix_(indices, indices)] = np.linalg.solve(M[np.ix_(indices, indices)], np.eye(len(indices)))
    return out


def solve_by_faces(M, q):
    n = len(q)
    for mask in range(1 << n):
        free = np.flatnonzero([(mask >> i) & 1 for i in range(n)])
        z = -embedded_inverse(M, free) @ q
        active = np.setdiff1d(np.arange(n), free)
        if np.min(z[free], initial=0.) >= -1e-10 and np.min((M @ z + q)[active], initial=0.) >= -1e-10:
            return np.maximum(z, 0.)
    raise AssertionError("No feasible KKT face")


def main():
    rng = np.random.default_rng(710503)
    counts = dict(principal_inverse_order=0, support_sharpness=0,
                  nonlinear_value_coverage=0, general_objective_coverage=0,
                  objective_gap_bound=0, finite_difference=0,
                  reused_geometry_coverage=0, component_coverage=0)
    max_violation = 0.
    with threadpool_limits(limits=1):
        for case in range(100):
            n, m = 6, 3
            A0 = rng.normal(size=(n, n))
            M = np.eye(n) + A0.T @ A0 / n
            mu = np.linalg.eigvalsh(M)[0]
            B, c = rng.normal(size=(n, m)), rng.normal(size=n)
            u, target = rng.uniform(-.3, .3, size=m), rng.uniform(-.3, .3, size=m)
            q = B @ u + c
            e = rng.uniform(.001, .3)
            s = rng.normal(size=n)
            s *= .8 * e / np.linalg.norm(s)
            obs = solve_by_faces(M, q - s)
            row_norms = np.linalg.norm(M, axis=1)
            F, H = status_bounds(M, q, obs, e, mu, row_norms)
            KF, KH = embedded_inverse(M, F), embedded_inverse(M, H)
            Delta = KH - KF
            S = np.setdiff1d(H, F)
            for size in range(len(S) + 1):
                for group in combinations(S, size):
                    J = np.union1d(F, group).astype(int)
                    KJ = embedded_inverse(M, J)
                    assert np.linalg.eigvalsh(KJ - KF)[0] >= -1e-10
                    assert np.linalg.eigvalsh(KH - KJ)[0] >= -1e-10
                    counts["principal_inverse_order"] += 1
            eigenvalues, vectors = np.linalg.eigh(Delta)
            root = (vectors * np.sqrt(np.maximum(eigenvalues, 0))) @ vectors.T
            a = root @ q
            center = B.T @ (KF + KH) @ q / 2
            for _ in range(5):
                direction = rng.normal(size=m)
                p = root @ B @ direction
                support = direction @ center + .5 * np.linalg.norm(a) * np.linalg.norm(p)
                if np.linalg.norm(p) < 1e-12:
                    witness = center
                else:
                    eigenvalues, eigenvectors = np.linalg.eigh((np.outer(a, p) + np.outer(p, a)) / 2)
                    positive = eigenvectors[:, eigenvalues > 0]
                    W = positive @ positive.T
                    assert np.linalg.eigvalsh(W)[0] >= -1e-8
                    assert np.linalg.eigvalsh(np.eye(n) - W)[0] >= -1e-8
                    witness = B.T @ (KF + root @ W @ root) @ q
                assert abs(direction @ witness - support) <= 1e-7
                counts["support_sharpness"] += 1
            cache = PrincipalCache(sparse.csc_matrix(M), B, c)
            bound = value_enclosure(cache, u, obs, e, mu, row_norms, .5, .4, target, .1)
            geometry = build_geometry(cache, u, obs, e, mu, row_norms)
            zstar = solve_by_faces(M, q)
            Vstar = .5 * zstar @ M @ zstar + q @ zstar
            D0, A = rng.normal(size=(n, n)), rng.normal(size=(n, m))
            D = D0.T @ D0 / n
            p0, h0 = .4 * u - A.T @ obs, D @ obs - A @ u
            gc = p0 - B.T @ (KF + KH) @ h0 / 2
            rad = .5 * np.sqrt(max(h0 @ Delta @ h0, 0) * max(np.linalg.eigvalsh(B.T @ Delta @ B)[-1], 0))
            remainder = 2 * e / mu * (np.linalg.norm(A, 2) + np.linalg.norm(B, 2) * np.linalg.norm(D, 2) / mu)
            for _ in range(10):
                perturbation = rng.normal(size=n)
                perturbation *= rng.uniform(0, e) / np.linalg.norm(perturbation)
                z = solve_by_faces(M, q - perturbation)
                J = np.flatnonzero(z > 1e-10)
                assert set(F) <= set(J) <= set(H)
                KJ = embedded_inverse(M, J)
                V = .5 * z @ M @ z + q @ z
                assert -1e-9 <= V - Vstar <= e ** 2 / (2 * mu) + 1e-9
                counts["objective_gap_bound"] += 1
                theta = 1 - .5 * V / n
                g = .4 * (u - target) + theta * B.T @ KJ @ q
                difference = np.linalg.norm(g - bound["gradient"])
                slack = difference - bound["gradient_bound"]
                max_violation = max(max_violation, slack)
                assert slack <= 1e-7
                action = np.clip(u - .1 * g, -.5, .5)
                assert np.linalg.norm(action - bound["action"]) <= bound["action_bound"] + 1e-7
                counts["nonlinear_value_coverage"] += 1
                general = .4 * u - A.T @ z - B.T @ KJ @ (D @ z - A @ u)
                assert np.linalg.norm(general - gc) <= rad + remainder + 1e-7
                counts["general_objective_coverage"] += 1
                smaller = .6 * e
                zz = solve_by_faces(M, q - perturbation * .5)
                reused = value_enclosure(cache, u, zz, smaller, mu, row_norms, .5, .4, target, .1, geometry=geometry)
                exact_theta = 1 - .5 * Vstar / n
                exact = .4 * (u - target) - exact_theta * B.T @ zstar
                assert np.linalg.norm(exact - reused["gradient"]) <= reused["gradient_bound"] + 1e-7
                counts["reused_geometry_coverage"] += 1
            for j in range(m):
                d = np.eye(m)[j] * 1e-5
                vals = []
                for uu in [u + d, u - d]:
                    zz = solve_by_faces(M, B @ uu + c)
                    ss = -.5 * zz @ M @ zz - (B @ uu + c) @ zz
                    vals.append(ss + .5 * ss ** 2 / (2 * n) + .2 * np.linalg.norm(uu - target) ** 2)
                ss = -Vstar
                exact_g = .4 * (u - target) - (1 + .5 * ss / n) * B.T @ zstar
                assert abs((vals[0] - vals[1]) / 2e-5 - exact_g[j]) < 1e-6
                counts["finite_difference"] += 1
        for case in range(100):
            n, m = 8, 2
            edges = rng.uniform(.2, 1., n - 1)
            edges[3] = 0.
            diagonal = np.ones(n)
            diagonal[:-1] += edges
            diagonal[1:] += edges
            M = sparse.diags([-edges, diagonal, -edges], [-1, 0, 1], format="csc")
            dense = M.toarray()
            B = np.zeros((n, m))
            B[:3] = rng.normal(size=(3, m))
            c, u, target = rng.normal(size=n), rng.uniform(-.3, .3, m), np.zeros(m)
            q = B @ u + c
            xi = rng.uniform(.005, .1, n)
            obs = solve_by_faces(dense, q - rng.uniform(-xi, xi))
            cache = PrincipalCache(M, B, c)
            geometry = component_geometry(cache, splu(M), u, obs, xi)
            bound = value_enclosure(cache, u, obs, np.linalg.norm(xi), 1.,
                                    np.linalg.norm(dense, axis=1), .5, .4, target, .1, geometry=geometry)
            assert set(geometry["relevant"]) <= set(range(4))
            for _ in range(5):
                z = solve_by_faces(dense, q - rng.uniform(-xi, xi))
                KJ = embedded_inverse(dense, np.flatnonzero(z > 0))
                V = .5 * z @ dense @ z + q @ z
                g = .4 * u + (1 - .5 * V / n) * B.T @ KJ @ q
                assert np.linalg.norm(g - bound["gradient"]) <= bound["gradient_bound"] + 1e-7
                counts["component_coverage"] += 1
    counts["maximum_enclosure_violation"] = max_violation
    out = Path(__file__).resolve().parent / "numerical_results"
    out.mkdir(exist_ok=True)
    (out / "ellipsoid_validation.json").write_text(json.dumps(counts, indent=2), encoding="utf-8")
    print(json.dumps(counts, indent=2), flush=True)


if __name__ == "__main__":
    main()
