import os
import json
from pathlib import Path
import numpy as np


DATASET = "ETH80"
RESULTS_ROOT = "/mnt/gs21/scratch/wangx306/project3/results"
METHOD_DIR = "psl_eigs"
ALPHA_TAG = "alpha0p0"

PCA_DIMS = [200, 300, 400, 500, 600, 700, 800, 900, 1000]
K_LOCALS = [5, 7, 10, 12, 15, 17, 20, 25, 30, 35, 40, 45, 50, 55, 60, 70, 80, 90, 100, 110]

MERGED_FILENAME = "psl_all_centers.npy"
MANIFEST_FILENAME = "psl_merge_manifest.json"

DELETE_CENTER_FILES_AFTER_MERGE = False


def load_obj_dict(fp: Path) -> dict:
    obj = np.load(fp, allow_pickle=True)
    if isinstance(obj, np.ndarray) and obj.dtype == object and obj.shape == ():
        obj = obj.item()
    if not isinstance(obj, dict):
        raise TypeError(f"expected dict in {fp}, got {type(obj)}")
    return obj


def atomic_save_npy(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        np.save(f, arr, allow_pickle=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def merge_one_setting(alpha_dir: Path, n_samples: int) -> dict:
    alpha_dir = alpha_dir.resolve()
    merged_path = alpha_dir / MERGED_FILENAME
    manifest_path = alpha_dir / MANIFEST_FILENAME

    center_files = sorted(alpha_dir.glob("center*_psl.npy"))
    if not center_files:
        raise FileNotFoundError(f"no center files in {alpha_dir}")

    records = np.empty((n_samples,), dtype=object)
    seen = np.zeros((n_samples,), dtype=bool)

    bad = 0
    for fp in center_files:
        try:
            d = load_obj_dict(fp)
            cid = int(d.get("center", -1))
            if cid < 0 or cid >= n_samples:
                bad += 1
                continue
            records[cid] = d
            seen[cid] = True
        except Exception:
            bad += 1

    missing_idx = np.where(~seen)[0]
    if missing_idx.size > 0:
        first_missing = missing_idx[:10].astype(int).tolist()
        raise RuntimeError(f"missing centers in {alpha_dir}, missing={int(missing_idx.size)}, first={first_missing}")

    atomic_save_npy(merged_path, records)

    chk = np.load(merged_path, allow_pickle=True)
    if chk.shape != (n_samples,):
        raise RuntimeError("merged file shape check failed")
    for i in range(n_samples):
        if chk[i] is None:
            raise RuntimeError(f"merged file missing record at {i}")

    meta = {
        "alpha_dir": str(alpha_dir),
        "n_samples_expected": int(n_samples),
        "n_center_files_found": int(len(center_files)),
        "n_bad_files": int(bad),
        "merged_file": str(merged_path),
        "deleted_center_files": bool(DELETE_CENTER_FILES_AFTER_MERGE),
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    if DELETE_CENTER_FILES_AFTER_MERGE:
        for fp in center_files:
            fp.unlink()

    return meta


def main():
    results_root = Path(RESULTS_ROOT).expanduser().resolve()
    method_root = results_root / METHOD_DIR / DATASET
    if not method_root.exists():
        raise FileNotFoundError(f"method root not found: {method_root}")

    y_path = results_root / f"{DATASET.lower()}_vectors" / "y.npy"
    if not y_path.exists():
        raise FileNotFoundError(f"label file not found: {y_path}")
    y = np.load(y_path, allow_pickle=True).ravel()
    n_samples = int(y.shape[0])

    merged_settings = []
    missing_settings = []

    for pca_dim in PCA_DIMS:
        for k_local in K_LOCALS:
            alpha_dir = method_root / f"pca{int(pca_dim)}" / f"k{int(k_local)}" / ALPHA_TAG
            if not alpha_dir.exists():
                missing_settings.append({"pca_dim": int(pca_dim), "k_local": int(k_local), "alpha_dir": str(alpha_dir)})
                continue

            print(f"merge pca={int(pca_dim)} k={int(k_local)} delete={DELETE_CENTER_FILES_AFTER_MERGE}")
            meta = merge_one_setting(alpha_dir, n_samples)
            merged_settings.append({"pca_dim": int(pca_dim), "k_local": int(k_local), **meta})
            print(f"done pca={int(pca_dim)} k={int(k_local)}")

    out_summary = results_root / "figures" / "concat_eval" / f"{DATASET}_{METHOD_DIR}_{ALPHA_TAG}_merge_summary.json"
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": DATASET,
                "method_dir": METHOD_DIR,
                "alpha_tag": ALPHA_TAG,
                "delete_center_files_after_merge": bool(DELETE_CENTER_FILES_AFTER_MERGE),
                "pca_dims_requested": list(PCA_DIMS),
                "k_locals_requested": list(K_LOCALS),
                "merged_settings": merged_settings,
                "missing_settings": missing_settings,
            },
            f,
            indent=2,
        )

    print("all done")
    print("summary saved to", str(out_summary))


if __name__ == "__main__":
    main()