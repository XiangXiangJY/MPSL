# Multi-dimensional Persistent Sheaf Laplacian (MPSL)

This repository provides the implementation of the multi-dimensional persistent sheaf Laplacian (MPSL) framework for image analysis.

## 📌 Overview

The MPSL framework constructs spectral features based on persistent sheaf Laplacians defined on simplicial complexes.  
It integrates information across multiple dimensions and scales to generate stable and structured representations for image datasets.

This implementation includes scripts for:
- Data preparation
- Distance computation
- Feature aggregation
- Classification experiments

---

## 📂 Repository Structure

- `main_distance.py`  
  Computes pairwise distances between samples using the proposed method.

- `main_merge.py`  
  Aggregates multi-dimensional spectral features.

- `main_multiclassknnCOIL20M.py`  
  Performs classification experiments using k-NN.

- `psl_utilsA.py`  
  Core implementation of persistent sheaf Laplacian utilities.

- `download_coil.py`  
  Script to download and preprocess the COIL20 dataset.

- `readcoil.py`  
  Helper functions for loading dataset.

- `environment.yml`  
  Conda environment configuration.

- `sheaf_math.py`  
  Corrected restriction-map math (composition-axiom fix) — see "Restriction-Map Correction" below.

- `psl_utilsA_corrected.py`  
  Drop-in corrected replacement for `psl_utilsA.py::compute_psl_eigs`, built on `sheaf_math.py`.

- `main_coil0_corrected.py`  
  Per-center PSL feature driver using the corrected restriction map (the corrected counterpart of the original, unpublished `main_coil0.py`).

---

## 📊 Datasets

The experiments in this project use the following publicly available datasets:

- **COIL20**  
  Available at:  
  https://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-20/coil-20-proc.zip

- **ETH80**  
  Available at:  
  https://github.com/chenchkx/ETH-80

Please download the datasets manually and place them in the appropriate directory before running the code.

---

## ⚙️ Installation

We recommend using Conda:

```bash
conda env create -f environment.yml
conda activate <env_name>
```

---

## 🔧 Restriction-Map Correction

`psl_utilsA.py`'s restriction map does not satisfy the sheaf composition
axiom: the edge-to-triangle map
`rho_{(u,v)->(u,v,w)} = 0.5*(kappa(d_uw)+kappa(d_vw))` was defined
independently of the vertex-to-edge map `rho_{u->(u,v)} = kappa(d_uv)`, so
composing restrictions through different faces of the same triangle
generally does not agree.

`sheaf_math.py` + `psl_utilsA_corrected.py` fix this with a positive scalar
simplex weight `q` (`q({u})=1`, `q({u,v})=K_uv`, `q(sigma)=1` for
`dim(sigma)>=2`) and `rho_{tau<=sigma} = q(sigma)/q(tau)` for every face
inclusion, which satisfies composition by construction. The vertex-to-edge
map `K_uv` is unchanged, so degree-0 (H0) spectra are numerically unchanged
from the original construction; degree-1 (H1) spectra, which use the
edge-to-triangle map, differ — that is the fix, not a regression. See the
docstring in `sheaf_math.py` for the full derivation.

`main_coil0_corrected.py` is a corrected-restriction-map version of the PSL
feature-computation driver: structurally identical to the original per-center
workflow (same global-distance k-NN construction, same A/B quantile
filtration, same output record schema), reading the same global PCA/distance
cache produced by `main_distance.py` and writing to a separate
`psl_eigs_corrected/` output root so it never overwrites features computed
with the original restriction map. `main_merge.py` and
`main_multiclassknnCOIL20M.py` work unchanged against the corrected output —
they just need to be pointed at that root (see the step-by-step guide below).

### How to compute the corrected features, step by step

All scripts in this repo are configured by editing the constants near the
top of the file, not by command-line flags — edit, save, then run
`python <script>.py`. This is the exact sequence for COIL20:

1. **Get the raw images.**
   ```bash
   python download_coil.py
   ```
   Downloads/extracts COIL20 (and COIL100) under `DATA_ROOT` (edit that
   constant at the top of the file if you want a different location).

2. **Build `X.npy`/`y.npy` from the raw images.**
   In `readcoil.py`, set `DATASET = "coil20"` (the file currently defaults to
   `"coil120"`, which is not a valid key in `DATASET_CONFIG` — a pre-existing
   typo, unrelated to the restriction-map fix; you must correct it locally
   before this step will run). Then:
   ```bash
   python readcoil.py
   ```
   Writes `X.npy`/`y.npy` to `.../coil20_vectors/` (path from
   `DATASET_CONFIG["coil20"]["out_dir"]`).

3. **Build the global PCA embedding + pairwise-distance cache.**
   ```bash
   python main_distance.py
   ```
   Reads `X.npy`/`y.npy` from step 2, writes `{tag}_X_pca.npy` /
   `{tag}_D_global.npy` (one pair per `PCA_DIMS` entry) to
   `.../global_dist/COIL20/`.

4. **Compute corrected per-center PSL eigenvalues.**
   ```bash
   python main_coil0_corrected.py
   ```
   Reads the cache from step 3, writes one `center###_psl.npy` file per
   image per `(pca_dim, k_local)` setting under
   `.../psl_eigs_corrected/COIL20/pca{D}/k{K}/alpha0p0/`. With the default
   `CHUNK = 20`, one run covers 20 centers (`SLURM_ARRAY_TASK_ID` unset ->
   task 0 -> centers 0-19); to cover all 1440 COIL20 images either loop
   locally (e.g. `for i in $(seq 0 71); do SLURM_ARRAY_TASK_ID=$i python
   main_coil0_corrected.py; done`) or submit as a SLURM array
   (`--array=0-71`). Edit `PCA_DIMS` / `K_LOCAL_LIST` at the top of the file
   to control which settings get computed.

5. **Merge per-center files into one array per setting.**
   In `main_merge.py`, set `DATASET = "COIL20"` (it currently defaults to
   `"ETH80"`) and `METHOD_DIR = "psl_eigs_corrected"` (so it reads step 4's
   output instead of the original construction's). Then:
   ```bash
   python main_merge.py
   ```

6. **Run the k-NN classification experiment.**
   In `main_multiclassknnCOIL20M.py`, set `METHOD_DIR = "psl_eigs_corrected"`
   (same reasoning as step 5). Then:
   ```bash
   python main_multiclassknnCOIL20M.py
   ```
   Prints cross-validated accuracy/balanced-accuracy/macro-F1 and saves a
   JSON summary under `OUT_DIR`.

Steps 5-6 never touch the original `psl_eigs/` output, so you can compute
both the original and corrected features and compare them by toggling
`METHOD_DIR` between `"psl_eigs"` and `"psl_eigs_corrected"`.
