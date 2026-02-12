import os
import zipfile
import urllib.request
from pathlib import Path

DATA_ROOT = Path("/mnt/home/wangx306/data/coil")

URLS = {
    "coil20": "https://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-20/coil-20-proc.zip",
    "coil100": "https://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-100/coil-100.zip",
}

EXPECTED_COUNTS = {
    "coil20": 20 * 72,
    "coil100": 100 * 72,
}

def download(url: str, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        print(f"{out_path.name} already exists, skip download")
        return
    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, out_path)

def extract(zip_path: Path, out_dir: Path):
    print(f"Extracting {zip_path.name}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)

def count_images(root: Path):
    return len(list(root.rglob("obj*__*.png")))

def main():
    DATA_ROOT.mkdir(parents=True, exist_ok=True)

    for tag in ["coil20", "coil100"]:
        zip_path = DATA_ROOT / f"{tag}.zip"
        extract_dir = DATA_ROOT / f"{tag}_raw"

        download(URLS[tag], zip_path)

        if not extract_dir.exists():
            extract_dir.mkdir()
            extract(zip_path, extract_dir)

        n_images = count_images(extract_dir)
        expected = EXPECTED_COUNTS[tag]

        if n_images != expected:
            raise RuntimeError(
                f"{tag} image count mismatch: got {n_images}, expected {expected}"
            )

        print(f"{tag} ready: {n_images} images")

    print("All COIL datasets downloaded successfully")

if __name__ == "__main__":
    main()