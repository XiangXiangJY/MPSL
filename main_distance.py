import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances


DATA_NAME = "COIL20"
IN_ROOT = "/mnt/gs21/scratch/wangx306/project3/results/coil20_vectors"
X_PATH = os.path.join(IN_ROOT, "X.npy")
Y_PATH = os.path.join(IN_ROOT, "y.npy")

OUT_ROOT = "/mnt/gs21/scratch/wangx306/project3/results/global_dist"
OUT_DIR = os.path.join(OUT_ROOT, DATA_NAME)

PCA_DIMS = [1000,900, 800,700, 600,500,400,300,200]
PCA_RANDOM_STATE = 1
METRIC = "euclidean"
WHITEN = False
L2_NORMALIZE = False


def l2_normalize_rows(x, eps=1e-12):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return x / norms


def metric_tag(metric):
    m = str(metric)
    m = m.replace("euclidean", "euc")
    m = m.replace("correlation", "corr")
    return m


def build_tag(pca_dim, metric, whiten, l2n, rs):
    w_tag = "w1" if whiten else "w0"
    l2_tag = "l21" if l2n else "l20"
    m_tag = metric_tag(metric)
    return f"{DATA_NAME}_pca{int(pca_dim)}_met{m_tag}_rs{int(rs)}_{w_tag}_{l2_tag}"


def save_outputs(out_dir, tag, x_pca, d_global, y, meta):
    x_out = os.path.join(out_dir, f"{tag}_X_pca.npy")
    d_out = os.path.join(out_dir, f"{tag}_D_global.npy")
    y_out = os.path.join(out_dir, f"{tag}_y.npy")
    meta_out = os.path.join(out_dir, f"{tag}_meta.npz")

    np.save(x_out, x_pca)
    np.save(d_out, d_global)
    np.save(y_out, y)
    np.savez(meta_out, **meta)

    print("saved", x_out)
    print("saved", d_out)
    print("saved", y_out)
    print("saved", meta_out)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    x = np.load(X_PATH)
    y = np.load(Y_PATH)

    n_samples = int(x.shape[0])
    orig_dim = int(x.shape[1])

    for pca_dim in PCA_DIMS:
        pca_dim_int = int(pca_dim)
        if pca_dim_int <= 0:
            raise ValueError("invalid pca_dim")
        if pca_dim_int > orig_dim:
            raise ValueError("pca_dim larger than input dim")

        tag = build_tag(
            pca_dim=pca_dim_int,
            metric=METRIC,
            whiten=WHITEN,
            l2n=L2_NORMALIZE,
            rs=PCA_RANDOM_STATE,
        )

        d_out = os.path.join(OUT_DIR, f"{tag}_D_global.npy")
        x_out = os.path.join(OUT_DIR, f"{tag}_X_pca.npy")
        meta_out = os.path.join(OUT_DIR, f"{tag}_meta.npz")
        y_out = os.path.join(OUT_DIR, f"{tag}_y.npy")

        if os.path.exists(d_out) and os.path.exists(x_out) and os.path.exists(meta_out) and os.path.exists(y_out):
            print("skip", tag)
            continue

        pca = PCA(n_components=pca_dim_int, random_state=PCA_RANDOM_STATE, whiten=WHITEN)
        x_pca = pca.fit_transform(x)

        if L2_NORMALIZE:
            x_pca = l2_normalize_rows(x_pca)

        d_global = pairwise_distances(x_pca, metric=METRIC)

        meta = {
            "data_name": DATA_NAME,
            "x_path": X_PATH,
            "y_path": Y_PATH,
            "pca_dim": pca_dim_int,
            "pca_random_state": int(PCA_RANDOM_STATE),
            "pca_whiten": bool(WHITEN),
            "l2_normalize": bool(L2_NORMALIZE),
            "metric": str(METRIC),
            "n_samples": n_samples,
            "orig_dim": orig_dim,
            "embed_dim": int(x_pca.shape[1]),
        }

        save_outputs(OUT_DIR, tag, x_pca, d_global, y, meta)
        print("shapes", x_pca.shape, d_global.shape)

    print("done")


if __name__ == "__main__":

    main()
