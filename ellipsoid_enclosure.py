"""Two-principal-system bounds for uncertain faces of an affine orthant VI."""

from collections import OrderedDict
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.sparse.csgraph import connected_components


class PrincipalCache:
    def __init__(self, M, B, c, capacity=12):
        self.M = sparse.csc_matrix(M)
        self.B, self.c = np.asarray(B), np.asarray(c)
        self.capacity = capacity
        self.items = OrderedDict()
        self.factorizations = 0
        self.right_hand_sides = 0

    def statistics(self, indices):
        indices = np.asarray(indices, dtype=int)
        key = indices.tobytes()
        if key in self.items:
            self.items.move_to_end(key)
            return self.items[key]
        m = self.B.shape[1]
        if not len(indices):
            result = (np.zeros((m, m)), np.zeros(m), 0.)
        else:
            block = self.M[indices][:, indices].tocsc()
            factor = splu(block, permc_spec="MMD_AT_PLUS_A")
            rhs = np.column_stack((self.B[indices], self.c[indices]))
            solution = factor.solve(rhs)
            self.factorizations += 1
            self.right_hand_sides += m + 1
            result = (self.B[indices].T @ solution[:, :m],
                      self.B[indices].T @ solution[:, m],
                      float(self.c[indices] @ solution[:, m]))
        self.items[key] = result
        if len(self.items) > self.capacity:
            self.items.popitem(last=False)
        return result


def quadratic_statistics(stats, u):
    W, v, h = stats
    return W @ u + v, float(u @ W @ u + 2 * u @ v + h)


def status_bounds(M, q, z, error, mu, row_norms):
    radius = 2 * error / mu
    F = np.flatnonzero(z > radius)
    active = (M @ z + q) > error + row_norms * radius
    if np.any(active[F]):
        raise AssertionError("Inconsistent certified constraint statuses")
    H = np.flatnonzero(~active)
    return F, H


def build_geometry(cache, u, z, error, mu, row_norms):
    q = cache.B @ u + cache.c
    F, H = status_bounds(cache.M, q, z, error, mu, row_norms)
    return geometry_from_sets(cache, u, error, F, H)


def geometry_from_sets(cache, u, error, F, H, global_uncertainty=None):
    sf, sh = cache.statistics(F), cache.statistics(H)
    bf, ef = quadratic_statistics(sf, u)
    bh, eh = quadratic_statistics(sh, u)
    metric = (sh[0] - sf[0])
    metric = (metric + metric.T) / 2
    # Inflate the positive-semidefinite energies to absorb subtraction roundoff.
    guard = 5e-12 * (1 + abs(eh) + abs(ef) + np.linalg.norm(sh[0], 2))
    metric = metric + guard * np.eye(len(u))
    energy = max(eh - ef, 0.) + guard
    base_center = (bf + bh) / 2
    widths = .5 * np.sqrt(energy * np.maximum(np.diag(metric), 0.))
    radius = .5 * np.sqrt(energy * max(np.linalg.eigvalsh(metric)[-1], 0.))
    return dict(error=error, base_center=base_center, widths=widths, radius=radius,
                uncertainty=len(H) - len(F), local_dimension=len(H),
                global_uncertainty=len(H) - len(F) if global_uncertainty is None else global_uncertainty)


def component_geometry(cache, factor, u, z, xi):
    """Stieltjes comparison followed by exact graph-component localization."""
    distance = factor.solve(xi)
    if np.min(distance) < -1e-12:
        raise AssertionError("The Stieltjes comparison produced a negative radius")
    distance = np.maximum(distance, 0.)
    q = cache.B @ u + cache.c
    free = z > 2 * distance
    active = (cache.M @ z + q) > xi + 2 * (abs(cache.M) @ distance)
    if np.any(free & active):
        raise AssertionError("Inconsistent componentwise statuses")
    H = np.flatnonzero(~active)
    graph = cache.M[H][:, H].tocsr()
    _, labels = connected_components(graph, directed=False)
    touched = np.unique(labels[np.any(cache.B[H] != 0, axis=1)])
    relevant = H[np.isin(labels, touched)]
    F = relevant[free[relevant]]
    geometry = geometry_from_sets(cache, u, float(np.linalg.norm(xi)), F, relevant,
                                  global_uncertainty=int(np.sum(~(free | active))))
    geometry["relevant"] = relevant
    return geometry


def value_enclosure(cache, u, z, error, mu, row_norms, rho, lam, target, alpha, bias=None, geometry=None):
    """Reuse a geometry only on the same upper point and a smaller error class."""
    if geometry is None:
        geometry = build_geometry(cache, u, z, error, mu, row_norms)
    if error > geometry["error"] * (1 + 1e-12):
        raise ValueError("The cached geometry does not cover the current error class")
    base_center, widths, radius = (geometry[k] for k in ["base_center", "widths", "radius"])
    q = cache.B @ u + cache.c
    surplus = -.5 * z @ (cache.M @ z) - q @ z
    theta = 1 + rho * surplus / len(z)
    weight_error = rho * error ** 2 / (2 * len(z) * mu)
    component_remainder = weight_error * (np.abs(base_center) + widths)
    bias = np.zeros_like(u) if bias is None else bias
    center = lam * (u - target) + bias + theta * base_center
    gradient_width = abs(theta) * widths + component_remainder
    observed_free = np.flatnonzero(z > 0)
    if "relevant" in geometry:
        observed_free = np.intersect1d(observed_free, geometry["relevant"])
    obs, _ = quadratic_statistics(cache.statistics(observed_free), u)
    gradient = lam * (u - target) + bias + theta * obs
    action = np.clip(u - alpha * gradient, -.5, .5)
    gradient_bound = np.linalg.norm(center - gradient) + abs(theta) * radius
    gradient_bound += weight_error * (np.linalg.norm(base_center) + radius)
    lower, upper = center - gradient_width, center + gradient_width
    endpoints = np.array([np.clip(u - alpha * lower, -.5, .5),
                          np.clip(u - alpha * upper, -.5, .5)])
    action_bound = np.linalg.norm(np.max(np.abs(endpoints - action), axis=0))
    gradient_bound = min(gradient_bound,
                         np.linalg.norm(np.maximum(abs(lower - gradient), abs(upper - gradient))))
    return dict(gradient=gradient, action=action, action_bound=float(action_bound),
                gradient_bound=float(gradient_bound), uncertainty=geometry["uncertainty"],
                local_dimension=geometry["local_dimension"], global_uncertainty=geometry["global_uncertainty"],
                theta=theta, base_center=base_center)


def stationarity_error(M, q, z):
    F = M @ z + q
    s = np.where(z > 0, F, np.minimum(F, 0.))
    return float(np.linalg.norm(s)), s
