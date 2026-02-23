import json
from pathlib import Path
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


DATASET = "COIL20"
METHOD_DIR = "psl_eigs"

RESULTS_ROOT = "/mnt/gs21/scratch/wangx306/project3/results"
OUT_DIR = "/mnt/gs21/scratch/wangx306/project3/results/figures/concat_eval"

ALPHA_TAG = "alpha0p0"

PCA_DIMS_TO_USE = [10,20,50,100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
K_LOCALS_TO_USE = [5, 7, 10, 12, 15, 17, 20, 25, 30, 35, 40, 45, 50, 55, 60, 70, 80, 90, 100, 110]

MERGED_FILENAME = "psl_all_centers.npy"

STAT_ID_MAP = {
    1: "max",
    2: "min",
    3: "sum",
    4: "mean",
    5: "std",
    6: "median",
    7: "q25",
    8: "q75",
    9: "iqr",
    10: "energy",
    11: "l1norm",
    12: "l2norm",
    13: "entropy",
    14: "top5sum",
    15: "top10sum",
    16: "nnz",
    17: "gt1e6",
    18: "num_zero_eigs",
    19: "min_positive_eig",
}

L0_STAT_IDS = [1, 3, 4, 5,6, 18, 19]
L1_STAT_IDS = [1,3,4,5, 6,18, 19]

N_SPLITS = 5
RANDOM_STATE = 1

KNN_K = 5


def label_path_from_dataset(dataset: str) -> Path:
    ds = dataset.lower()
    y_path = Path(RESULTS_ROOT) / f"{ds}_vectors" / "y.npy"
    if not y_path.exists():
        raise FileNotFoundError(f"Label file not found: {y_path}")
    return y_path


def stat_names_from_ids(stat_ids):
    names = []
    for sid in stat_ids:
        if sid not in STAT_ID_MAP:
            raise ValueError(f"Unknown stat id: {sid}")
        names.append(STAT_ID_MAP[sid])
    return names


def ensure_list_of_scales(eigs_obj):
    if isinstance(eigs_obj, np.ndarray) and eigs_obj.dtype == object and eigs_obj.shape == ():
        eigs_obj = eigs_obj.item()

    if isinstance(eigs_obj, np.ndarray) and eigs_obj.dtype != object:
        if eigs_obj.ndim == 1:
            return [eigs_obj.astype(float)]
        if eigs_obj.ndim == 2:
            return [eigs_obj[i, :].astype(float) for i in range(eigs_obj.shape[0])]
        return [eigs_obj.reshape(-1).astype(float)]

    if isinstance(eigs_obj, (list, tuple)):
        out = []
        for s in eigs_obj:
            out.append(np.asarray(s, dtype=float).ravel())
        return out

    raise TypeError(f"Cannot interpret eigen container type {type(eigs_obj)}")


def compute_stats(x: np.ndarray, stat_ids):
    if stat_ids is None or len(stat_ids) == 0:
        return np.zeros((0,), dtype=float)

    x = np.asarray(x, dtype=float).ravel()
    if x.size == 0:
        return np.zeros((len(stat_ids),), dtype=float)

    xs = np.sort(x)
    out = []

    eps = 1e-8

    for sid in stat_ids:
        name = STAT_ID_MAP[sid]

        if name == "max":
            out.append(float(xs[-1]))
        elif name == "min":
            pos = xs[xs > eps]
            if pos.size == 0:
                out.append(0.0)
            else:
                out.append(float(pos[0]))
        elif name == "sum":
            out.append(float(np.sum(xs)))
        elif name == "mean":
            out.append(float(np.mean(xs)))
        elif name == "std":
            out.append(float(np.std(xs, ddof=0)))
        elif name == "median":
            out.append(float(np.median(xs)))
        elif name == "q25":
            out.append(float(np.quantile(xs, 0.25)))
        elif name == "q75":
            out.append(float(np.quantile(xs, 0.75)))
        elif name == "iqr":
            q25 = float(np.quantile(xs, 0.25))
            q75 = float(np.quantile(xs, 0.75))
            out.append(q75 - q25)
        elif name == "energy":
            out.append(float(np.sum(xs * xs)))
        elif name == "l1norm":
            out.append(float(np.sum(np.abs(xs))))
        elif name == "l2norm":
            out.append(float(np.linalg.norm(xs)))
        elif name == "entropy":
            s = float(np.sum(np.maximum(xs, 0.0)))
            if s <= 0:
                out.append(0.0)
            else:
                p = np.maximum(xs, 0.0) / s
                p = np.clip(p, 1e-12, 1.0)
                out.append(float(-np.sum(p * np.log(p))))
        elif name == "top5sum":
            k = min(5, xs.size)
            out.append(float(np.sum(xs[-k:])))
        elif name == "top10sum":
            k = min(10, xs.size)
            out.append(float(np.sum(xs[-k:])))
        elif name == "nnz":
            out.append(float(np.count_nonzero(xs)))
        elif name == "gt1e6":
            out.append(float(np.sum(xs > 1e-6)))
        elif name == "num_zero_eigs":
            out.append(float(np.sum(np.abs(xs) <= eps)))
        elif name == "min_positive_eig":
            pos = xs[xs > eps]
            if pos.size == 0:
                out.append(0.0)
            else:
                out.append(float(pos[0]))
        else:
            raise ValueError(f"Unknown stat id {sid}")

    return np.asarray(out, dtype=float)


def extract_feature_from_record(d: dict):
    center_id = int(d["center"])

    spectra = d.get("spectra", None)
    if spectra is None:
        raise KeyError("Missing spectra")
    if isinstance(spectra, np.ndarray) and spectra.dtype == object and spectra.shape == ():
        spectra = spectra.item()
    if not isinstance(spectra, dict):
        raise TypeError(f"spectra must be dict, got {type(spectra)}")

    blocks = []

    if L0_STAT_IDS:
        if 0 not in spectra:
            raise KeyError("Missing dim 0")
        scales = ensure_list_of_scales(spectra[0])
        feats = [compute_stats(eigs, L0_STAT_IDS) for eigs in scales]
        blocks.append(np.concatenate(feats, axis=0))

    if L1_STAT_IDS:
        if 1 not in spectra:
            raise KeyError("Missing dim 1")
        scales = ensure_list_of_scales(spectra[1])
        feats = [compute_stats(eigs, L1_STAT_IDS) for eigs in scales]
        blocks.append(np.concatenate(feats, axis=0))

    if len(blocks) == 0:
        raise ValueError("No features selected. Check L0_STAT_IDS and L1_STAT_IDS")

    feat = np.concatenate(blocks, axis=0)
    return center_id, feat


def load_setting_matrix(alpha_dir: Path, n_samples: int):
    merged_path = alpha_dir / MERGED_FILENAME
    if not merged_path.exists():
        raise FileNotFoundError(f"Merged file not found: {merged_path}")

    records = np.load(merged_path, allow_pickle=True)
    if not isinstance(records, np.ndarray) or records.dtype != object:
        raise TypeError(f"Expected object ndarray in {merged_path}")
    if records.shape != (n_samples,):
        raise ValueError(f"Shape mismatch in {merged_path}: got {records.shape}, expected {(n_samples,)}")

    feat_dim = None
    X = None

    for cid in range(n_samples):
        d = records[cid]
        if d is None:
            raise ValueError(f"Missing record at center {cid} in {merged_path}")
        if isinstance(d, np.ndarray) and d.dtype == object and d.shape == ():
            d = d.item()
        if not isinstance(d, dict):
            raise TypeError(f"Record at center {cid} is not dict in {merged_path}")

        rcid, feat = extract_feature_from_record(d)
        if rcid != cid:
            raise ValueError(f"Center id mismatch at index {cid}: record says {rcid} in {merged_path}")

        if feat_dim is None:
            feat_dim = int(feat.shape[0])
            X = np.zeros((n_samples, feat_dim), dtype=float)
        else:
            if int(feat.shape[0]) != feat_dim:
                raise ValueError(f"Feature dim mismatch inside {merged_path}")

        X[cid, :] = feat

    return X, int(feat_dim), str(merged_path)


def evaluate_cv_knn(X: np.ndarray, y: np.ndarray, knn_k: int):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    accs = []
    baccs = []
    f1macs = []

    for tr, te in skf.split(X, y):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X[tr])
        Xte = scaler.transform(X[te])

        clf = KNeighborsClassifier(n_neighbors=int(knn_k))
        clf.fit(Xtr, y[tr])
        pred = clf.predict(Xte)

        accs.append(accuracy_score(y[te], pred))
        baccs.append(balanced_accuracy_score(y[te], pred))
        f1macs.append(f1_score(y[te], pred, average="macro"))

    return {
        "knn_k": int(knn_k),
        "acc_mean": float(np.mean(accs)),
        "acc_std": float(np.std(accs)),
        "bacc_mean": float(np.mean(baccs)),
        "bacc_std": float(np.std(baccs)),
        "f1macro_mean": float(np.mean(f1macs)),
        "f1macro_std": float(np.std(f1macs)),
    }


def main():
    results_root = Path(RESULTS_ROOT).expanduser().resolve()
    method_root = results_root / METHOD_DIR / DATASET
    if not method_root.exists():
        raise FileNotFoundError(f"Method root not found: {method_root}")

    y_path = label_path_from_dataset(DATASET)
    y = np.load(y_path, allow_pickle=True).ravel()
    n_samples = int(y.shape[0])

    l0_names = stat_names_from_ids(L0_STAT_IDS) if L0_STAT_IDS else []
    l1_names = stat_names_from_ids(L1_STAT_IDS) if L1_STAT_IDS else []

    X_blocks = []
    used_settings = []
    total_dim = 0

    for pca_dim in PCA_DIMS_TO_USE:
        for k_local in K_LOCALS_TO_USE:
            alpha_dir = method_root / f"pca{pca_dim}" / f"k{k_local}" / ALPHA_TAG
            if not alpha_dir.exists():
                continue

            X_set, d_set, merged_fp = load_setting_matrix(alpha_dir, n_samples)
            X_blocks.append(X_set)
            used_settings.append(
                {
                    "pca_dim": int(pca_dim),
                    "k_local": int(k_local),
                    "alpha_dir": str(alpha_dir),
                    "setting_feature_dim": int(d_set),
                    "merged_file": str(merged_fp),
                }
            )
            total_dim += int(d_set)

    if not X_blocks:
        raise RuntimeError("No settings loaded. Check PCA_DIMS_TO_USE K_LOCALS_TO_USE ALPHA_TAG paths")

    X_final = np.concatenate(X_blocks, axis=1)
    metrics = evaluate_cv_knn(X_final, y, knn_k=KNN_K)

    print("Using label file:", y_path)
    print("Dataset:", DATASET)
    print("Method:", METHOD_DIR, "Alpha:", ALPHA_TAG)
    print("Merged file:", MERGED_FILENAME)
    print("L0 stat ids:", L0_STAT_IDS, "names:", l0_names)
    print("L1 stat ids:", L1_STAT_IDS, "names:", l1_names)
    print("PCA dims requested:", PCA_DIMS_TO_USE)
    print("K locals requested:", K_LOCALS_TO_USE)
    print("Settings actually used:", [(s["pca_dim"], s["k_local"]) for s in used_settings])
    print("Final feature shape:", X_final.shape)
    print(
        "Final CV scores",
        "knn_k",
        f"{metrics['knn_k']}",
        "acc",
        f"{metrics['acc_mean']:.4f}",
        "bacc",
        f"{metrics['bacc_mean']:.4f}",
        "f1",
        f"{metrics['f1macro_mean']:.4f}",
    )

    out_dir = Path(OUT_DIR).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{DATASET}_{METHOD_DIR}_{ALPHA_TAG}_multimultiscale_concat_eval_knn.json"

    summary = {
        "dataset": DATASET,
        "method_dir": METHOD_DIR,
        "alpha_tag": ALPHA_TAG,
        "label_path": str(y_path),
        "pca_dims_requested": list(PCA_DIMS_TO_USE),
        "k_locals_requested": list(K_LOCALS_TO_USE),
        "settings_actually_used": used_settings,
        "stat_id_map": dict(STAT_ID_MAP),
        "l0_stat_ids": list(L0_STAT_IDS),
        "l1_stat_ids": list(L1_STAT_IDS),
        "l0_stat_names": list(l0_names),
        "l1_stat_names": list(l1_names),
        "final_feature_shape": [int(X_final.shape[0]), int(X_final.shape[1])],
        "knn": {
            "n_neighbors": int(KNN_K),
            "metrics": metrics,
        },
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Saved summary to", out_path)


if __name__ == "__main__":
    main()