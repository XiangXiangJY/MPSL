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

- `sheaf_math.py`, `psl_utilsA_corrected.py`, `main_coil0_corrected.py`  
  Corrected restriction-map construction — see "Restriction-Map Correction" below.

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
just point them at the `psl_eigs_corrected/` root (and set
`DATASET = "COIL20"` at the top of `main_merge.py`, which currently defaults
to `"ETH80"`).
