"""
Drop-in corrected replacement for `psl_utilsA.py::compute_psl_eigs`.

This module does not reimplement the sheaf math -- it is a thin adapter that
preserves the exact original public signature and return structure while
delegating the restriction-map construction to the composition-axiom-correct
q-based construction in `sheaf_math.py`.

Preserved verbatim from the original interface:
  - function name, argument order/names/defaults, return type (dict keyed by
    requested homology dimension -> eigenvalue array);
  - Rips complex construction convention (gudhi.RipsComplex on the local
    distance matrix, `max_edge_length=r_used`, `r_used` = max nonzero local
    distance when the caller does not pass an explicit filtration bound);
  - default-b-from-r_used behavior when `b is None`.

Replaced:
  - only the restriction-map construction. The original averaging
    edge->triangle map `rho_{(u,v)->(u,v,w)} = 0.5*(kappa(d_uw)+kappa(d_vw))`
    is replaced by the q-based map `rho_{tau<=sigma} = q(sigma)/q(tau)` from
    `sheaf_math.py`.

`center_index` and `alpha` are accepted for call-site compatibility, but the
corrected q-based construction has no label-modulation term: there is no
"center" vertex distinction and no alpha-weighted penalty in q. Any nonzero
alpha is rejected explicitly rather than silently ignored.
"""
import hashlib
import os

import numpy as np

import sheaf_math as sm

RESTRICTION_VERSION = "q_based_composition_correct_v2_bcapped_rips"

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SHEAF_MATH_PATH = os.path.join(_THIS_DIR, "sheaf_math.py")
_THIS_FILE_PATH = os.path.abspath(__file__)


def _sha256_of(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


SHEAF_MATH_SOURCE_HASH = _sha256_of(_SHEAF_MATH_PATH)
ADAPTER_SOURCE_HASH = _sha256_of(_THIS_FILE_PATH)
SOURCE_HASH = hashlib.sha256((SHEAF_MATH_SOURCE_HASH + ADAPTER_SOURCE_HASH).encode()).hexdigest()

ACTIVE_FORMULAS = {
    "q_vertex": "q({u}) = 1",
    "q_edge": "q({u,v}) = K_uv = exp(-D(u,v)^2 / sigma^2)",
    "q_higher": "q(sigma) = 1 for dim(sigma) >= 2",
    "restriction": "rho_{tau<=sigma} = q(sigma) / q(tau)",
    "vertex_to_edge": "rho_{u<=(u,v)} = K_uv  (unchanged from original)",
    "edge_to_triangle": "rho_{(u,v)<=(u,v,w)} = 1 / K_uv  (original: 0.5*(K_uw+K_vw))",
    "vertex_to_triangle": "rho_{u<=(u,v,w)} = 1",
    "identity": "rho_{sigma<=sigma} = 1",
    "sigma_definition": "sigma = median of nonzero pairwise distances in the local complex, else 1.0",
}


def get_provenance():
    return {
        "restriction_version": RESTRICTION_VERSION,
        "sheaf_math_source_hash": SHEAF_MATH_SOURCE_HASH,
        "adapter_source_hash": ADAPTER_SOURCE_HASH,
        "source_hash": SOURCE_HASH,
        "active_formulas": dict(ACTIVE_FORMULAS),
    }


def compute_psl_eigs(
    X_local,
    D_local,
    a=0.0,
    b=None,
    max_dim=2,
    sigma=None,
    dims=(0, 1, 2),
    center_index=0,
    alpha=0.0,
):
    """
    Corrected drop-in for the original `psl_utilsA.compute_psl_eigs`. Returns
    a dict `{dim: eigenvalue_array}` for every dim in `dims`.

    `center_index` is accepted for signature compatibility only -- the
    corrected q-based restriction has no notion of a distinguished center
    vertex (unlike the original label-modulation term).
    """
    if float(alpha) != 0.0:
        raise ValueError(
            f"psl_utilsA_corrected.compute_psl_eigs: alpha={alpha!r} requested, but the "
            "corrected q-based restriction has no label-modulation term and is only "
            "validated at alpha=0.0. Refusing to silently ignore a nonzero alpha."
        )

    # Rips build is capped at `b` (not the local patch's full max distance)
    # since L_h^{a,b} depends only on X_a/X_b by definition -- simplices with
    # filtration > b are provably irrelevant to the result, and capping avoids
    # an O(k^3) dense-coboundary blowup at high local k. This changes no
    # eigenvalue in the output.
    if b is None:
        st_full, r_used = sm.build_simplex_tree_from_dist(D_local, r_max=None, max_dim=max_dim)
        b = r_used
        st = st_full
    else:
        st, r_used = sm.build_simplex_tree_from_dist(D_local, r_max=float(b), max_dim=max_dim)

    from petls import sheaf_simplex_tree, PersistentSheafLaplacian

    restriction, sigma_used, clip_stats = sm.make_corrected_restriction(D_local, sigma=sigma)
    extra_data = {(i,): {} for i in range(np.asarray(X_local).shape[0])}
    sst = sheaf_simplex_tree(st, extra_data, restriction)
    psl = PersistentSheafLaplacian(sst)

    results = {}
    for dim in dims:
        results[dim] = psl.spectra(dim=dim, a=float(a), b=float(b))

    return results
