import os
import glob
import json
import numpy as np
from PIL import Image


# =========================
# 1. Dataset config
# =========================

DATASET = "coil120"   #  

DATASET_CONFIG = {
    "coil20": {
        "img_dir": "/mnt/home/wangx306/data/coil/coil20_raw/coil-20-proc",
        "out_dir": "/mnt/gs21/scratch/wangx306/project3/results/coil20_vectors",
        "label_offset": 1,
    },
    "coil100": {
        "img_dir": "/mnt/home/wangx306/data/coil/coil100_raw/coil-100",
        "out_dir": "/mnt/gs21/scratch/wangx306/project3/results/coil100_vectors",
        "label_offset": 1,
    },
}


# =========================
# 2. IO helpers
# =========================

def list_png_files(img_dir):
    pattern = os.path.join(img_dir, "obj*__*.png")
    files = sorted(glob.glob(pattern))
    if len(files) == 0:
        raise RuntimeError(f"No png files found in {img_dir}")
    return files


def parse_label_from_filename(file_path, label_offset=1):
    """
    Expected format: obj<ID>__<angle>.png
    """
    base = os.path.basename(file_path)
    if not base.startswith("obj"):
        raise RuntimeError(f"Unexpected filename: {base}")
    obj_part = base.split("__")[0]
    label = int(obj_part.replace("obj", "")) - label_offset
    return label


# =========================
# 3. Core loader
# =========================

def load_image_dataset_as_matrix(
    img_dir,
    normalize=True,
    dtype=np.float32,
    label_offset=1,
):
    files = list_png_files(img_dir)
    n = len(files)

    first = Image.open(files[0]).convert("L")
    w, h = first.size
    d = h * w

    X = np.zeros((n, d), dtype=dtype)
    y = np.zeros((n,), dtype=np.int32)

    for i, fp in enumerate(files):
        img = Image.open(fp).convert("L")
        arr = np.asarray(img, dtype=dtype).reshape(-1)
        if normalize:
            arr = arr / dtype(255.0)
        X[i] = arr
        y[i] = parse_label_from_filename(fp, label_offset)

    meta = {
        "img_dir": img_dir,
        "n_samples": int(n),
        "n_features": int(d),
        "image_height": int(h),
        "image_width": int(w),
        "normalize": bool(normalize),
        "dtype": str(np.dtype(dtype)),
        "label_rule": f"label = int(obj_id) - {label_offset}",
        "filename_pattern": "obj<ID>__<angle>.png",
    }
    return X, y, files, meta


# =========================
# 4. Save + sanity check
# =========================

def save_outputs(out_dir, X, y, files, meta):
    os.makedirs(out_dir, exist_ok=True)

    np.save(os.path.join(out_dir, "X.npy"), X)
    np.save(os.path.join(out_dir, "y.npy"), y)

    with open(os.path.join(out_dir, "files.txt"), "w") as f:
        for fp in files:
            f.write(fp + "\n")

    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


def sanity_check(X, y):
    print("Sanity check")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("X min / max:", float(X.min()), float(X.max()))
    print("num classes:", int(np.unique(y).size))
    print("y min / max:", int(y.min()), int(y.max()))


# =========================
# 5. Main
# =========================

def main():
    if DATASET not in DATASET_CONFIG:
        raise ValueError(f"Unknown dataset: {DATASET}")

    cfg = DATASET_CONFIG[DATASET]

    print("Dataset:", DATASET)
    print("Image dir:", cfg["img_dir"])
    print("Output dir:", cfg["out_dir"])

    X, y, files, meta = load_image_dataset_as_matrix(
        img_dir=cfg["img_dir"],
        normalize=True,
        label_offset=cfg["label_offset"],
    )

    sanity_check(X, y)
    save_outputs(cfg["out_dir"], X, y, files, meta)

    print("Saved dataset matrices successfully")


if __name__ == "__main__":
    main()