"""RankMe-style effective rank — representation collapse detector (ODEWorld App.C)."""
from __future__ import annotations

import math
from typing import Optional, Sequence


def _mean_center(rows: list[list[float]]) -> list[list[float]]:
    if not rows:
        return []
    d = len(rows[0])
    means = [0.0] * d
    for r in rows:
        for i, x in enumerate(r):
            if i < d:
                means[i] += float(x)
    n = len(rows)
    means = [m / n for m in means]
    out = []
    for r in rows:
        out.append([float(r[i]) - means[i] for i in range(min(len(r), d))])
    return out


def _singular_values_power_iter(matrix: list[list[float]], iters: int = 24) -> list[float]:
    """Approximate singular values via power iteration on Gram matrix.

    Good enough for RankMe-style collapse alarms on small ARSI feature windows.
    """
    n = len(matrix)
    if n == 0:
        return []
    d = len(matrix[0])
    k = min(n, d, 32)
    # Gram G = X X^T (n x n) or X^T X (d x d) — use smaller side
    if n <= d:
        # G_ij = <x_i, x_j>
        G = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i, n):
                s = 0.0
                for a in range(d):
                    s += matrix[i][a] * matrix[j][a]
                G[i][j] = s
                G[j][i] = s
        dim = n
    else:
        G = [[0.0] * d for _ in range(d)]
        for i in range(d):
            for j in range(i, d):
                s = 0.0
                for a in range(n):
                    s += matrix[a][i] * matrix[a][j]
                G[i][j] = s
                G[j][i] = s
        dim = d

    # deflation power iteration for top-k eigenvalues of G (= σ²)
    work = [row[:] for row in G]
    eigs: list[float] = []
    for _ in range(k):
        v = [1.0 / math.sqrt(max(1, dim)) + (i % 7) * 0.01 for i in range(dim)]
        for _p in range(iters):
            nv = [0.0] * dim
            for i in range(dim):
                acc = 0.0
                row = work[i]
                for j in range(dim):
                    acc += row[j] * v[j]
                nv[i] = acc
            norm = math.sqrt(sum(x * x for x in nv)) or 1.0
            v = [x / norm for x in nv]
        # Rayleigh
        Av = [0.0] * dim
        for i in range(dim):
            acc = 0.0
            row = work[i]
            for j in range(dim):
                acc += row[j] * v[j]
            Av[i] = acc
        lam = sum(v[i] * Av[i] for i in range(dim))
        eigs.append(max(0.0, lam))
        # deflate
        for i in range(dim):
            for j in range(dim):
                work[i][j] -= lam * v[i] * v[j]
    sigmas = [math.sqrt(e) for e in eigs if e > 1e-12]
    return sigmas


def centered_effective_rank(rows: Sequence[Sequence[float]], eps: float = 1e-12) -> dict:
    """RankMe: effective rank = exp(entropy of normalized singular values)."""
    mat = _mean_center([list(map(float, r)) for r in rows if r])
    n = len(mat)
    if n == 0 or not mat or not mat[0]:
        return {"effective_rank": 0.0, "dim": 0, "n": n, "collapse": True, "note": "empty"}
    dim = len(mat[0])
    sigmas = _singular_values_power_iter(mat)
    total = sum(sigmas) + eps
    p = [s / total for s in sigmas if s > eps]
    if not p:
        return {"effective_rank": 0.0, "dim": dim, "n": n, "collapse": True, "note": "zero_singular_values"}
    entropy = -sum(pi * math.log(pi) for pi in p if pi > 0)
    rank = math.exp(entropy)
    ratio = rank / max(1.0, float(dim))
    return {
        "effective_rank": round(rank, 4),
        "dim": dim,
        "n": n,
        "rank_over_dim": round(ratio, 4),
        # collapse if effective rank << ambient dim or all mass on one component
        "collapse": bool(rank < max(1.5, 0.15 * dim)),
        "sigma_head": [round(s, 6) for s in sigmas[:8]],
        "note": "rankme_centered_effective_rank",
    }


def feature_rows_from_dicts(dicts: Sequence[dict], keys: Sequence[str]) -> list[list[float]]:
    rows = []
    for d in dicts:
        rows.append([float(d.get(k, 0.0) or 0.0) for k in keys])
    return rows
