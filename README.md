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
