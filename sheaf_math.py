"""
Corrected q-based cellular sheaf restriction maps for MPSL.

Fixes a composition-axiom violation in the original construction
(`psl_utilsA.py`): the original edge-to-triangle restriction map,
rho_{(u,v)->(u,v,w)} = 0.5*(kappa(d_uw)+kappa(d_vw)), was defined
independently of the vertex-to-edge map rho_{u->(u,v)} = kappa(d_uv), so
composing restrictions through different faces of the same triangle
generally did not agree.

Mathematical construction
--------------------------
Preserve the existing exponential vertex-to-edge kernel exactly:

    K_uv = kappa(D(u,v)) = exp(-D(u,v)^2 / sigma^2)

(sigma = median of nonzero pairwise distances in the local complex, falling
back to 1.0 if none exist).

Define a positive scalar simplex weight q:

    q({u})   = 1                     for every vertex
    q({u,v}) = K_uv                  for every edge
    q(sigma) = 1                     for every simplex with dim(sigma) >= 2

For every face inclusion tau <= sigma, define

    rho_{tau<=sigma} = q(sigma) / q(tau)

Composition holds for ANY chain tau <= sigma <= omega, by construction:

    rho_{sigma<=omega} * rho_{tau<=sigma}
        = [q(omega)/q(sigma)] * [q(sigma)/q(tau)]
        = q(omega)/q(tau)
        = rho_{tau<=omega}

Special cases (this construction has no label-modulation / alpha term):

    rho_{u<={u,v}}       = K_uv        (unchanged from the original)
    rho_{{u,v}<={u,v,w}} = 1 / K_uv    (original: 0.5*(K_uw+K_vw) -- differs in general)
    rho_{u<={u,v,w}}     = 1
    rho_{sigma<=sigma}   = 1

Because the vertex-to-edge map K_uv is unchanged, the degree-0 sheaf
Laplacian L0 is numerically unchanged from the original construction. L1
(and anything using the edge-to-triangle map) will generally differ --
that is the fix, not a regression.

Numerical stability
--------------------
K_uv in (0, 1] always. The only reciprocal taken is q(sigma)/q(tau) in the
edge-to-triangle case (1/K_uv), which can blow up as K_uv -> 0 for a very
distant/dissimilar edge. K_uv is floored at K_FLOOR before any reciprocal or
log is taken; `ClipStats` optionally tracks how often the floor is hit so
this never happens silently. Ratios are computed in log-domain for
stability, then exponentiated once at the end.

Orientation
-----------
This module returns only the positive restriction *magnitude*; sign is
applied by petls's `sheaf_simplex_tree.apply_restriction_function`.

Edge-order invariance
----------------------
All simplex vertex tuples are canonicalized via `sorted(...)` before any
lookup, and D is symmetric, so q({u,v}) does not depend on vertex order.
"""
from dataclasses import dataclass, field

import numpy as np

# Numerically safe lower bound for K_uv before any reciprocal (1/K_uv) or log
# is computed.
K_FLOOR = 1e-12


@dataclass
class ClipStats:
    """Accumulates how often K_uv was floored to K_FLOOR, for transparency."""

    n_total: int = 0
    n_clipped: int = 0
    min_k_seen: float = field(default=float("inf"))

    def record(self, k_uv_raw):
        self.n_total += 1
        if k_uv_raw < self.min_k_seen:
            self.min_k_seen = float(k_uv_raw)
        if k_uv_raw < K_FLOOR:
            self.n_clipped += 1

    @property
    def clip_rate(self):
        if self.n_total == 0:
            return 0.0
        return self.n_clipped / self.n_total

    def as_dict(self):
        return {
            "n_total": self.n_total,
            "n_clipped": self.n_clipped,
            "clip_rate": self.clip_rate,
            "min_k_seen": self.min_k_seen if self.n_total > 0 else None,
            "k_floor": K_FLOOR,
        }


def kernel(d, sigma):
    """kappa(t) = exp(-t^2 / sigma^2) (same convention as psl_utilsA.py:
    literal sigma**2 denominator)."""
    d = np.asarray(d, dtype=float)
    return np.exp(-(d ** 2) / (sigma ** 2))


def compute_sigma(D_local):
    """sigma = median of nonzero pairwise distances in the local complex,
    falling back to 1.0 if none exist."""
    D_local = np.asarray(D_local, dtype=float)
    dists = D_local[D_local > 0]
    if dists.size == 0:
        return 1.0
    return float(np.median(dists))


def edge_weight(u, v, D, sigma, clip_stats=None):
    """K_uv = kappa(D(u,v)), floored at K_FLOOR before use."""
    d_uv = float(D[u, v])
    k_uv_raw = float(kernel(d_uv, sigma))
    if clip_stats is not None:
        clip_stats.record(k_uv_raw)
    return max(k_uv_raw, K_FLOOR)


def simplex_q(vertices, D, sigma, clip_stats=None):
    """q(sigma) for a simplex given as an iterable of vertex indices.

    q({u}) = 1, q({u,v}) = K_uv, q(sigma) = 1 for dim(sigma) >= 2.
    Returns (q, log_q).
    """
    verts = tuple(sorted(vertices))
    if len(verts) == 1:
        return 1.0, 0.0
    if len(verts) == 2:
        u, v = verts
        k_uv = edge_weight(u, v, D, sigma, clip_stats=clip_stats)
        return k_uv, float(np.log(k_uv))
    return 1.0, 0.0


def restriction_value(tau_vertices, sigma_vertices, D, sigma, clip_stats=None):
    """rho_{tau<=sigma} = q(sigma)/q(tau), computed in log-domain.

    tau_vertices, sigma_vertices: iterables of vertex indices (any order --
    canonicalized internally). When tau == sigma, returns exactly 1.0.
    """
    q_tau, log_q_tau = simplex_q(tau_vertices, D, sigma, clip_stats=clip_stats)
    q_sig, log_q_sig = simplex_q(sigma_vertices, D, sigma, clip_stats=clip_stats)
    if tuple(sorted(tau_vertices)) == tuple(sorted(sigma_vertices)):
        return 1.0
    log_rho = log_q_sig - log_q_tau
    return float(np.exp(log_rho))


def make_corrected_restriction(D, sigma=None, clip_stats=None):
    """Build a (simplex, coface, sst) -> float callable for
    petls.sheaf_simplex_tree, implementing the corrected q-based restriction
    map for any face inclusion.

    Parameters
    ----------
    D : array-like (m, m)
        Local distance matrix.
    sigma : float or None
        Kernel bandwidth. If None, computed via compute_sigma(D).
    clip_stats : ClipStats or None
        If given, accumulates K_uv floor-clipping statistics.

    Returns
    -------
    restriction_fn : callable
    sigma_used : float
    clip_stats : ClipStats (the one passed in, or a freshly created one)
    """
    D = np.asarray(D, dtype=float)
    if sigma is None:
        sigma = compute_sigma(D)
    if clip_stats is None:
        clip_stats = ClipStats()

    def restriction(simplex, coface, sst):
        # petls convention: `simplex` is the face (tau), `coface` is the
        # simplex containing it (sigma) -- this implements rho_{tau<=sigma}.
        return restriction_value(simplex, coface, D, sigma, clip_stats=clip_stats)

    return restriction, sigma, clip_stats


def build_simplex_tree_from_dist(D, r_max=None, max_dim=2):
    """Same Rips-construction convention as psl_utilsA.py, kept separate so
    this module has no import-time dependency on it."""
    import gudhi

    D = np.asarray(D, dtype=float)
    dists = D[D > 0]
    r_used = float(r_max) if r_max is not None else (float(dists.max()) if dists.size > 0 else 1.0)
    rips = gudhi.RipsComplex(distance_matrix=D.tolist(), max_edge_length=r_used)
    st = rips.create_simplex_tree(max_dimension=max_dim)
    return st, r_used


def compute_corrected_psl_eigs(
    X_local,
    D_local,
    a=0.0,
    b=None,
    max_dim=2,
    sigma=None,
    dims=(0, 1),
    clip_stats=None,
):
    """High-level helper mirroring psl_utilsA.compute_psl_eigs's signature
    (minus `alpha` and `center_index`, which have no role here: there is no
    label term, and the restriction map does not depend on which vertex is
    the "center" of the local patch).

    Returns (spectra_dict, sigma_used, clip_stats).
    """
    from petls import sheaf_simplex_tree, PersistentSheafLaplacian

    st, r_used = build_simplex_tree_from_dist(D_local, r_max=None, max_dim=max_dim)
    if b is None:
        b = r_used

    restriction, sigma_used, clip_stats = make_corrected_restriction(
        D_local, sigma=sigma, clip_stats=clip_stats
    )
    extra_data = {(i,): {} for i in range(np.asarray(X_local).shape[0])}
    sst = sheaf_simplex_tree(st, extra_data, restriction)
    psl = PersistentSheafLaplacian(sst)

    results = {}
    for dim in dims:
        results[int(dim)] = psl.spectra(dim=int(dim), a=float(a), b=float(b))

    return results, sigma_used, clip_stats
