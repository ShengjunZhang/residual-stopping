"""Graph-preserving inverse-interval images, using two endpoint solves."""

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import splu


def block_geometry(M, B, v, free, possible, p=None):
    """Return frozen image bounds; free/possible must cover the face family."""
    M = sparse.csc_matrix(M)
    B, v = np.asarray(B), np.asarray(v)
    free, possible = np.asarray(free, int), np.asarray(possible, int)
    n, m = B.shape
    if not set(free).issubset(possible):
        raise ValueError("The certified free set must be a subset of possible")
    k, labels = connected_components(M[possible][:, possible], directed=False)
    full_labels = np.full(n, -1, int)
    full_labels[possible] = labels
    metrics = np.zeros((2, k, m, m))
    moments = np.zeros((2, k, m))
    energies = np.zeros((2, k))
    factorizations = 0
    for endpoint, indices in enumerate([free, possible]):
        if not len(indices):
            continue
        rhs = np.column_stack((B[indices], v[indices]))
        solution = splu(M[indices][:, indices], permc_spec="MMD_AT_PLUS_A").solve(rhs)
        factorizations += 1
        for b in range(k):
            rows = full_labels[indices] == b
            I = indices[rows]
            metrics[endpoint, b] = B[I].T @ solution[rows, :m]
            moments[endpoint, b] = B[I].T @ solution[rows, m]
            energies[endpoint, b] = v[I] @ solution[rows, m]
    W = metrics[1] - metrics[0]
    W = (W + W.transpose(0, 2, 1)) / 2
    energy = energies[1] - energies[0]
    # These are floating-point checks, not outward-rounded interval arithmetic.
    scale = 1 + np.max(np.abs(energies), initial=0.) + np.max(np.abs(metrics), initial=0.)
    if np.min(energy, initial=0.) < -1e-10 * scale:
        raise ArithmeticError("A block energy is negative beyond roundoff")
    if k and min(np.linalg.eigvalsh(w)[0] for w in W) < -1e-10 * scale:
        raise ArithmeticError("A block metric is indefinite beyond roundoff")
    energy = np.maximum(energy, 0.)
    diag = np.maximum(np.diagonal(W, axis1=1, axis2=2), 0.)
    widths = .5 * np.sum(np.sqrt(energy[:, None] * diag), axis=0)
    global_widths = .5 * np.sqrt(np.sum(energy) * np.sum(diag, axis=0))
    aggregate = np.sum(W, axis=0)
    global_radius = .5 * np.sqrt(np.sum(energy) * max(np.linalg.eigvalsh(aggregate)[-1], 0.))
    block_radius = .5 * sum(np.sqrt(e * max(np.linalg.eigvalsh(w)[-1], 0.))
                             for e, w in zip(energy, W))
    center = (np.zeros(m) if p is None else np.asarray(p)) - .5 * np.sum(moments, axis=(0, 1))
    return dict(center=center, widths=widths, global_widths=global_widths,
                radius=min(global_radius, block_radius, np.linalg.norm(widths)),
                global_radius=global_radius, energy=energy, metrics=W,
                factorizations=factorizations, right_hand_sides=factorizations * (m + 1),
                labels=full_labels, blocks=k)


def centered_support(geometry, direction):
    d = np.asarray(direction)
    return .5 * sum(np.sqrt(max(e * (d @ w @ d), 0.))
                     for e, w in zip(geometry["energy"], geometry["metrics"]))


def projected_bound(u, gradient, center, widths, alpha, lower=-.5, upper=.5):
    observed = np.clip(u - alpha * gradient, lower, upper)
    endpoints = np.array([np.clip(u - alpha * (center - widths), lower, upper),
                          np.clip(u - alpha * (center + widths), lower, upper)])
    return float(np.linalg.norm(np.max(abs(endpoints - observed), axis=0)))
