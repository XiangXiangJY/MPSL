# main_coil0_corrected.py
#
# Corrected-restriction-map version of the PSL feature driver. Structurally
# identical to the original per-center COIL20 driver (same global-distance
# k-NN construction, same A/B quantile filtration, same output record
# schema) -- the only change is importing `compute_psl_eigs` from
# `psl_utilsA_corrected` (composition-axiom-correct restriction map, see
# that module and `sheaf_math.py`) instead of `psl_utilsA`.
#
# Reads the same global PCA/distance cache produced by `main_distance.py`
# (same tag convention: DATA_NAME_pca{D}_met{metric}_rs{rs}_{w}_{l2}).
# Writes to a separate `psl_eigs_corrected/` output root so this never
# overwrites any features computed with the original restriction map.
#
# H0 (vertex-to-edge only) is numerically unchanged from the original
# construction. H1 (uses the edge-to-triangle map) differs -- that is the
# fix, not a bug.
#
# Usage (single machine, one array-task-worth of centers):
#   python main_coil0_corrected.py
# Usage (SLURM array, reads SLURM_ARRAY_TASK_ID as the center-chunk index):
#   sbatch --array=0-71 ... main_coil0_corrected.py   # CHUNK=20 -> 1440/20=72 tasks

import os
import numpy as np

from psl_utilsA_corrected import compute_psl_eigs


DATA_NAME = "COIL20"

GLOBAL_DIST_DIR = "/mnt/gs21/scratch/wangx306/project3/results/global_dist/COIL20"
Y_PATH = "/mnt/gs21/scratch/wangx306/project3/results/coil20_vectors/y.npy"

OUT_ROOT = "/mnt/gs21/scratch/wangx306/project3/results/psl_eigs_corrected"

PCA_DIMS = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

PCA_RANDOM_STATE = 1
METRIC_TAG = "meteuc"
WHITEN_TAG = "w0"
L2_TAG = "l20"

K_LOCAL_LIST = [7]
ALPHA = 0.0

CHUNK = 20

PSL_DIMS = (0, 1)
RIPS_MAX_DIM = 2
PSL_SIGMA = None

A_MODE = "quantile_local_nz"
A_Q = 0.30
A_FIXED = None

B_MODE = "quantile_local_nz"
B_Q = 0.85
B_FIXED = None
B_SCALE = 1.0

B_FLOOR = 1e-8


def tag_float(x):
    return str(float(x)).replace(".", "p")


def build_global_tag(pca_dim):
    return f"{DATA_NAME}_pca{int(pca_dim)}_{METRIC_TAG}_rs{int(PCA_RANDOM_STATE)}_{WHITEN_TAG}_{L2_TAG}"


def load_global(pca_dim):
    tag = build_global_tag(pca_dim)
    x_path = os.path.join(GLOBAL_DIST_DIR, f"{tag}_X_pca.npy")
    d_path = os.path.join(GLOBAL_DIST_DIR, f"{tag}_D_global.npy")
    if not os.path.exists(x_path):
        raise FileNotFoundError(x_path)
    if not os.path.exists(d_path):
        raise FileNotFoundError(d_path)
    x_pca = np.load(x_path)
    d_global = np.load(d_path)
    return x_pca, d_global, tag


def upper_triangle_nonzero(d, eps=1e-12):
    iu = np.triu_indices(d.shape[0], k=1)
    vals = d[iu]
    vals = vals[np.isfinite(vals)]
    vals = vals[vals > eps]
    return vals


def clamp01(q):
    qv = float(q)
    if qv < 0.0:
        qv = 0.0
    if qv > 1.0:
        qv = 1.0
    return qv


def compute_a(d_local):
    if A_FIXED is not None:
        return float(A_FIXED)
    if A_MODE == "zero":
        return 0.0

    vals = upper_triangle_nonzero(d_local)
    if vals.size == 0:
        return 0.0

    if A_MODE == "quantile_local_nz":
        return float(np.quantile(vals, clamp01(A_Q)))

    if A_MODE == "fixed":
        raise ValueError("A_MODE fixed requires A_FIXED")

    raise ValueError("unknown A_MODE")


def compute_b(d_local):
    if B_FIXED is not None:
        return float(B_FIXED)

    vals = upper_triangle_nonzero(d_local)
    if vals.size == 0:
        return 0.0

    if B_MODE == "max":
        return float(np.max(vals))

    if B_MODE == "quantile_local_nz":
        return float(np.quantile(vals, clamp01(B_Q)))

    if B_MODE == "fixed":
        raise ValueError("B_MODE fixed requires B_FIXED")

    raise ValueError("unknown B_MODE")


def knn_from_global_distance(d_global, k_local):
    n = d_global.shape[0]
    idx_knn = np.empty((n, k_local + 1), dtype=int)
    for i in range(n):
        order = np.argsort(d_global[i], kind="mergesort")
        idx_knn[i] = order[: k_local + 1]
    return idx_knn


def make_out_dir(pca_dim, k_local, alpha):
    base_dir = os.path.join(OUT_ROOT, DATA_NAME)
    pca_dir = os.path.join(base_dir, f"pca{int(pca_dim)}")
    k_dir = os.path.join(pca_dir, f"k{int(k_local)}")
    a_dir = os.path.join(k_dir, f"alpha{tag_float(alpha)}")
    return a_dir


def safe_save_npy_atomic(out_path, obj_array):
    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)

    tmp_path = out_path + ".tmp"
    with open(tmp_path, "wb") as f:
        np.save(f, obj_array, allow_pickle=True)
        f.flush()
        os.fsync(f.fileno())

    os.replace(tmp_path, out_path)


def get_center_range(n_samples):
    task_id_str = os.environ.get("SLURM_ARRAY_TASK_ID", "0")
    task_id = int(task_id_str)

    start = task_id * int(CHUNK)
    end = start + int(CHUNK)

    if start < 0:
        start = 0
    if start > n_samples:
        start = n_samples
    if end > n_samples:
        end = n_samples

    return start, end, task_id


def main():
    y = np.load(Y_PATH)
    alpha_float = float(ALPHA)

    for pca_dim in PCA_DIMS:
        x_pca, d_global, gtag = load_global(pca_dim)
        n = int(x_pca.shape[0])

        if d_global.shape[0] != n or d_global.shape[1] != n:
            raise ValueError("d_global shape mismatch")

        center_start, center_end, task_id = get_center_range(n)

        for k_local in K_LOCAL_LIST:
            k_local_int = int(k_local)
            if k_local_int + 1 > n:
                raise ValueError("k_local too large")

            idx_knn = knn_from_global_distance(d_global, k_local_int)

            out_dir = make_out_dir(pca_dim, k_local_int, alpha_float)
            os.makedirs(out_dir, exist_ok=True)

            print("task_id", int(task_id), "pca_dim", int(pca_dim), "k", int(k_local_int), "alpha", float(alpha_float))
            print("centers", int(center_start), int(center_end), "out_dir", out_dir)
            print("filtration", "A_MODE", A_MODE, "A_Q", float(A_Q), "B_MODE", B_MODE, "B_Q", float(B_Q), "B_SCALE", float(B_SCALE))

            for center in range(center_start, center_end):
                out_path = os.path.join(out_dir, f"center{center:03d}_psl.npy")
                if os.path.exists(out_path):
                    continue

                local_idx = idx_knn[center].astype(int)
                pos = np.where(local_idx == center)[0]
                if pos.size == 0:
                    raise ValueError("center not in local_idx")
                center_local_index = int(pos[0])

                d_local = d_global[np.ix_(local_idx, local_idx)]
                x_local = x_pca[local_idx]

                a = compute_a(d_local)
                b0 = compute_b(d_local)
                b = float(b0) * float(B_SCALE)

                if not np.isfinite(a):
                    a = 0.0
                if (not np.isfinite(b)) or (b <= a):
                    b = float(a) + float(B_FLOOR)

                spectra = compute_psl_eigs(
                    X_local=x_local,
                    D_local=d_local,
                    a=float(a),
                    b=float(b),
                    max_dim=int(RIPS_MAX_DIM),
                    sigma=PSL_SIGMA,
                    dims=PSL_DIMS,
                    center_index=center_local_index,
                    alpha=alpha_float,
                )

                record = {
                    "center": int(center),
                    "y_center": int(y[center]),
                    "pca_dim": int(pca_dim),
                    "global_tag": str(gtag),
                    "idx": local_idx,
                    "interval": (float(a), float(b)),
                    "spectra": {int(d): np.array(spectra[int(d)], dtype=float) for d in PSL_DIMS},
                    "meta": {
                        "k_local": int(k_local_int),
                        "alpha": float(alpha_float),
                        "rips_max_dim": int(RIPS_MAX_DIM),
                        "psl_dims": tuple(int(d) for d in PSL_DIMS),
                        "sigma": PSL_SIGMA,
                        "a_mode": str(A_MODE),
                        "a_q": float(A_Q),
                        "a_fixed": A_FIXED,
                        "b_mode": str(B_MODE),
                        "b_q": float(B_Q),
                        "b_fixed": B_FIXED,
                        "b_scale": float(B_SCALE),
                        "chunk": int(CHUNK),
                        "task_id": int(task_id),
                        "center_start": int(center_start),
                        "center_end": int(center_end),
                        "restriction_version": "q_based_composition_correct_v2_bcapped_rips",
                    },
                }

                safe_save_npy_atomic(out_path, np.array(record, dtype=object))

            print("done pca_dim", int(pca_dim), "k", int(k_local_int))

    print("done all")


if __name__ == "__main__":
    main()
