#!/usr/bin/env python
# coding: utf-8

# # All-ligand distance-dependent GP tests for ATAC-derived modalities
# 
# This notebook reproduces the spatial Gaussian-process (GP) likelihood-ratio screens for **all 15 ligand perturbations** across four ATAC-derived feature spaces:
# 
# 1. called ATAC peaks;
# 2. fixed 50-kb ATAC bins;
# 3. chromVAR motif-deviation scores; and
# 4. Signac ATAC gene-activity scores.
# 
# The four pure-guide negative controls are also included for 50-kb bins, chromVAR, and gene activity. Peak-level controls are not included because the peak-preparation helper defines ligand reference cells from `target_call`; the all-ligand peak batch therefore contains the 15 biological ligand perturbations.
# 
# Each GP input contains distance-bin midpoints (`X`), pseudobulk feature means (`means`), standard errors (`sems`), and calibrated fixed-noise variances (`mean_sem_sq`). Called peaks use six 50-µm bins; the other modalities use twelve 25-µm bins. Count-based features are retained when detected in strictly more than 5% of the eligible binned cells.
# 

# ## Statistical test
# 
# For every feature within every perturbation, a zero-mean GP with a constant (Bias) covariance kernel is compared with a zero-mean GP with an RBF covariance kernel. Gaussian observation noise is fixed at the mean squared pseudobulk SEM across distance bins for that feature, $\sigma_f^2 = B^{-1}\sum_b SEM_{bf}^2$, rather than optimized.
# 
# The RBF model is optimized from three deterministic initial lengthscales—10, 50, and 200 µm—and the fit with the largest likelihood is retained. Each fit is allowed up to 1,000 L-BFGS-B iterations. The primary p-value uses the approximate boundary reference $0.5\chi_0^2 + 0.5\chi_1^2$, because the constant model is approached as the RBF lengthscale tends to infinity. Benjamini-Hochberg correction is performed **separately within each perturbation-by-modality screen** at 5% FDR. Ordinary $\chi_1^2$ and Benjamini-Yekutieli values are retained as sensitivity analyses.
# 
# The primary screens are unconstrained. Set `MIN_LENGTHSCALE_UM = 25.0` to reproduce the prespecified constrained sensitivity analysis for the 25-µm inputs; keep it separate from the primary results.
# 

# ## 1. Project paths and run controls
# 
# This version is adapted to the data currently available: called peaks and chromVAR are read from the H5Mu file, fixed genomic bins are read from the three 50-kb SnapATAC2 tile files, and Signac gene activity is read from the converted H5AD file. Set `PREPARE_MISSING_INPUTS = True` to build the GP pseudobulk NPZ inputs, then set `RUN_SCREENS = True` to run the GP screens. New results are written beneath `results/notebook_multimodal_atac_gp_all_conditions/`; existing production outputs are never overwritten.
# 

# In[7]:


import numpy as np
import math
from scipy import sparse
import re
from scipy.stats import kruskal
from statsmodels.stats.multitest import multipletests
import anndata as ad
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
sc.settings.set_figure_params(dpi=300)


# In[7]:


from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import json
import os
import shlex
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy import sparse


DEFAULT_WORKSPACE = Path("/scratch/welchjd_root/welchjd1/javidgmh/15515_Multiome/catatac_work/motif_analysis")


def resolve_workspace():
    configured = os.environ.get("MOTIF_ANALYSIS_DIR", "").strip()
    return Path(configured).expanduser() if configured else DEFAULT_WORKSPACE


def first_existing_file(candidates):
    for candidate in candidates:
        candidate = Path(candidate).expanduser()
        if candidate.is_file():
            return candidate
    return None


def missing_file_message(candidates, description):
    checked = "\n".join(f"  - {Path(candidate).expanduser()}" for candidate in candidates)
    return f"Could not find {description}. Checked:\n{checked}"


def resolve_gp_python(workspace):
    configured = os.environ.get("GP_PYTHON", "").strip()
    candidates = [Path(configured).expanduser()] if configured else []
    candidates.extend([
        workspace / "gpclust_spatial" / ".venv_gpclust" / "bin" / "python",
        workspace / ".venv_gpclust" / "bin" / "python",
        Path(sys.executable),
    ])
    resolved = first_existing_file(candidates)
    if resolved is not None:
        return resolved
    checked = "\n".join(f"  - {Path(candidate).expanduser()}" for candidate in candidates)
    raise FileNotFoundError(
        "Could not find a Linux Python executable for GP screens. "
        f"Set GP_PYTHON if needed. Checked:\n{checked}"
    )


def configured_candidates(env_var, default_candidates):
    configured = os.environ.get(env_var, "").strip()
    candidates = [Path(configured).expanduser()] if configured else []
    candidates.extend(default_candidates)
    return candidates


WORKSPACE = resolve_workspace()
GP_RUNNER_CANDIDATES = configured_candidates(
    "GP_RUNNER",
    [WORKSPACE / "gpclust_spatial" / "run_spatial_lrt.py", WORKSPACE / "run_spatial_lrt.py"],
)
GP_RUNNER = first_existing_file(GP_RUNNER_CANDIDATES)
GP_PYTHON = resolve_gp_python(WORKSPACE)

# Raw data available for this analysis.
SOURCE_H5MU_CANDIDATES = configured_candidates(
    "SOURCE_H5MU",
    [
        WORKSPACE / "all_lanes_merged_CATATAC_QC_RNA_processed_EB_manual_clustering_cell_types (3).h5mu",
        WORKSPACE / "all_lanes_merged_CATATAC_QC_RNA_processed_EB_manual_clustering_cell_types.h5mu",
    ],
)
SOURCE_H5MU = first_existing_file(SOURCE_H5MU_CANDIDATES)

TILE_DIR = Path(os.environ.get("SNAPATAC2_TILE_DIR", str(WORKSPACE))).expanduser()
TILE_FILES = [TILE_DIR / f"lane{i}_snapatac2_tiles.h5ad" for i in (1, 2, 3)]

GENE_ACTIVITY_H5AD_CANDIDATES = configured_candidates(
    "GENE_ACTIVITY_H5AD",
    [
        WORKSPACE / "signac_gene_activity_raw_counts.h5ad",
        WORKSPACE / "signac_gene_activity.h5ad",
    ],
)
GENE_ACTIVITY_H5AD = first_existing_file(GENE_ACTIVITY_H5AD_CANDIDATES)

FDR = 0.05
WORKERS_PER_SCREEN = 4
MAX_CONCURRENT_SCREENS = 2
CHUNK_SIZE = 25
MAX_OPTIMIZER_ITERS = 1000
RBF_START_LENGTHSCALES_UM = (10.0, 50.0, 200.0)
MIN_LENGTHSCALE_UM = None
DIAGNOSTIC_MAX_FEATURES = None

INCLUDE_NEGATIVE_CONTROLS = True
PREPARE_MISSING_PEAK_INPUTS = True
PREPARE_MISSING_INPUTS = True       # prepares 50-kb bins, chromVAR, and gene activity
RUN_SCREENS = True                  # set False if you only want to prepare/inspect inputs

OUTPUT_ROOT = WORKSPACE / "results" / "notebook_multimodal_atac_gp_all_conditions_50kb"
PEAK_BATCH_ROOT = WORKSPACE / "results" / "atac_peak_all_ligands_gp_assigned_50um"
NONPEAK_INPUT_ROOT = WORKSPACE / "results" / "multimodal_gp_inputs_from_available_data_50kb"

assert WORKSPACE.is_dir(), WORKSPACE
assert GP_PYTHON.is_file(), GP_PYTHON
print(f"Workspace: {WORKSPACE}")
print(f"Notebook interpreter: {Path(sys.executable).resolve()}")
print(f"GP runner: {GP_RUNNER if GP_RUNNER is not None else 'not found; internal GP runner will be used'}")
print(f"GP interpreter: {GP_PYTHON}")
print(f"H5Mu: {SOURCE_H5MU if SOURCE_H5MU is not None else 'NOT FOUND'}")
print(f"50-kb tiles: {[str(p) for p in TILE_FILES]}")
print(f"Gene activity H5AD: {GENE_ACTIVITY_H5AD if GENE_ACTIVITY_H5AD is not None else 'NOT FOUND'}")


# ## 2. Register all ligand and control datasets
# 
# The `condition_id` names the 50-kb-bin and gene-activity directory. `chromvar_dataset` names the corresponding chromVAR directory. Keeping this mapping explicit prevents the negative-control aliases (`negative_control_1` versus `NTC_1`, for example) from being mixed up.
# 

# In[8]:


LIGANDS = (
    "WNT5A", "WNT1", "BMP5", "BMP7", "JAG1",
    "DLL3", "VEGFB", "PDGFC", "EFNA3", "RSPO3",
    "SEMA6D", "NECTIN2", "NECTIN3", "ALB", "B2M",
)
NEGATIVE_CONTROLS = (
    ("NTC_1", "negative_control_1", "Negative control-1"),
    ("NTC_2", "negative_control_2", "Negative control-2"),
    ("NTC_3", "negative_control_3", "Negative control-3"),
    ("NTC_5", "negative_control_5", "Negative control-5"),
)

DATASETS = [
    {
        "dataset": ligand,
        "condition_id": ligand.lower(),
        "condition_label": ligand,
        "chromvar_dataset": ligand,
        "dataset_type": "ligand",
    }
    for ligand in LIGANDS
]
if INCLUDE_NEGATIVE_CONTROLS:
    DATASETS.extend(
        {
            "dataset": dataset,
            "condition_id": condition_id,
            "condition_label": condition_label,
            "chromvar_dataset": dataset,
            "dataset_type": "negative_control",
        }
        for dataset, condition_id, condition_label in NEGATIVE_CONTROLS
    )

dataset_table = pd.DataFrame(DATASETS)
dataset_table


# ## 3. Prepare GP inputs from the data you actually have
# 
# Called-peak inputs are prepared with the existing peak helper. The additional code below prepares the other three modalities directly from the available data: **50-kb SnapATAC2 tiles**, **chromVAR scores in the H5Mu**, and **Signac gene-activity counts in the converted H5AD**. All three use twelve 25-µm spatial distance bins (0–300 µm). Count modalities retain features detected in >5% of eligible binned cells and use pseudobulk library-size normalization `log1p(10000 * count / library_total)` with fixed noise derived from the pseudobulk SEM.
# 

# In[9]:


LEGACY_WNT5A_PEAK_INPUT = (
    WORKSPACE / "results" / "wnt5a_atac_peak_pseudobulk_logcpm_gp_assigned_50um"
    / "wnt5a_atac_peak_pseudobulk_logcpm_inputs.npz"
)


def peak_batch_input(dataset):
    return PEAK_BATCH_ROOT / "conditions" / dataset["condition_id"] / "gp_inputs.npz"


def peak_input_path(dataset):
    batch_input = peak_batch_input(dataset)
    if batch_input.is_file():
        return batch_input
    if dataset["dataset"] == "WNT5A" and LEGACY_WNT5A_PEAK_INPUT.is_file():
        return LEGACY_WNT5A_PEAK_INPUT
    return batch_input


def _peak_distance_bins(obs, spatial, condition_label, bin_width_um=50.0, max_distance_um=300.0):
    """Distance of assigned non-target cells to nearest target cell within the same EB."""
    from scipy.spatial import cKDTree

    if "target_call" not in obs.columns:
        raise KeyError("ATAC obs is missing target_call")
    eb_col = "EB_manual_clustering" if "EB_manual_clustering" in obs.columns else "EB_manual"
    if eb_col not in obs.columns:
        raise KeyError("ATAC obs needs EB_manual_clustering or EB_manual")

    calls = obs["target_call"]
    call_str = calls.astype(str)
    target = call_str.eq(str(condition_label)).to_numpy()
    assigned = calls.notna().to_numpy() & ~call_str.isin(["", "nan", "None"]).to_numpy()
    query = assigned & ~target

    if target.sum() == 0:
        raise ValueError(
            f"No target_call cells found for {condition_label!r}. "
            f"Check the exact target_call labels in the H5Mu file."
        )

    dist = np.full(len(obs), np.nan, dtype=float)
    eb_values = obs[eb_col].astype(str).to_numpy()
    for eb in pd.unique(eb_values):
        in_eb = eb_values == eb
        src = np.flatnonzero(in_eb & target)
        qry = np.flatnonzero(in_eb & query)
        if src.size == 0 or qry.size == 0:
            continue
        tree = cKDTree(np.asarray(spatial[src], dtype=float))
        dist[qry] = tree.query(np.asarray(spatial[qry], dtype=float), k=1)[0]

    edges = np.arange(0.0, max_distance_um + bin_width_um, bin_width_um)
    mids = ((edges[:-1] + edges[1:]) / 2.0)[:, None]
    valid = query & np.isfinite(dist) & (dist >= 0) & (dist < max_distance_um)
    bins = np.full(len(obs), -1, dtype=int)
    bins[valid] = np.floor(dist[valid] / bin_width_um).astype(int)
    return bins, valid, mids


def _peak_gp_summaries(count_matrix, library_size, bins, valid, feature_names,
                       detection_threshold=0.05, scale_factor=10_000.0, expected_bins=6):
    """Reproduce the peak pseudobulk log-library workflow directly from the H5Mu."""
    Xmat = sparse.csr_matrix(count_matrix)
    library = np.asarray(library_size, dtype=float).ravel()
    if Xmat.shape[0] != library.size:
        raise ValueError("Peak matrix and ATAC library-size vector have different numbers of cells")

    eligible = np.flatnonzero(valid)
    if eligible.size == 0:
        raise ValueError("No eligible assigned non-target cells fall within 300 um of target cells")

    # Detection is based on cell-level peak presence among all eligible binned cells.
    detected = np.asarray((Xmat[eligible] > 0).sum(axis=0)).ravel() / eligible.size
    keep = detected > detection_threshold
    kept_idx = np.flatnonzero(keep)
    if kept_idx.size == 0:
        raise ValueError("No called peaks pass the >5% detection threshold")

    Mkeep = Xmat[:, kept_idx]
    names = np.asarray(feature_names, dtype=str)[kept_idx]
    detection = detected[kept_idx]

    means, sems, bin_counts = [], [], []
    for b in range(expected_bins):
        rows = np.flatnonzero(valid & (bins == b))
        n = rows.size
        bin_counts.append(n)
        if n <= 1:
            raise ValueError(f"Peak distance bin {b} has only {n} eligible cells; at least 2 are required")

        M = Mkeep[rows]
        lib = library[rows]
        if np.any(~np.isfinite(lib)) or np.any(lib <= 0):
            raise ValueError(f"Peak distance bin {b} contains invalid ATAC_total_counts")

        count_sum = np.asarray(M.sum(axis=0)).ravel().astype(float)
        count_sq_sum = np.asarray(M.power(2).sum(axis=0)).ravel().astype(float)
        count_library_sum = np.asarray(M.T @ lib).ravel().astype(float)
        library_sum = float(lib.sum())
        library_sq_sum = float(np.square(lib).sum())

        ratio = count_sum / library_sum
        scaled = scale_factor * ratio
        mean = np.log1p(scaled)

        # Ratio-estimator delta-method variance, retaining count/library covariance.
        residual_ss = np.maximum(
            count_sq_sum - 2.0 * ratio * count_library_sum + np.square(ratio) * library_sq_sum,
            0.0,
        )
        ratio_variance = n / (n - 1.0) * residual_ss / (library_sum ** 2)
        sem = scale_factor / (1.0 + scaled) * np.sqrt(ratio_variance)
        means.append(mean)
        sems.append(sem)

    means = np.vstack(means)
    sems = np.vstack(sems)
    mean_sem_sq = np.mean(np.square(sems), axis=0)
    finite = (
        np.all(np.isfinite(means), axis=0)
        & np.all(np.isfinite(sems), axis=0)
        & np.all(sems >= 0, axis=0)
        & np.isfinite(mean_sem_sq)
        & (mean_sem_sq > 0)
    )

    return (
        means[:, finite], sems[:, finite], mean_sem_sq[finite], names[finite],
        kept_idx[finite], detection[finite], np.asarray(bin_counts, dtype=int),
    )


def prepare_all_peak_inputs():
    if SOURCE_H5MU is None or not SOURCE_H5MU.is_file():
        raise FileNotFoundError(missing_file_message(SOURCE_H5MU_CANDIDATES, "source H5Mu"))

    import mudata as md

    print(f"Loading peak matrix once from {SOURCE_H5MU}")
    mdata = md.read_h5mu(SOURCE_H5MU)
    atac = mdata.mod["atac"]
    rna = mdata.mod["rna"]
    obs = atac.obs.copy()

    # EB labels live in RNA obs, not ATAC obs.
    obs["EB_manual_clustering"] = (
        rna.obs["EB_manual_clustering"].reindex(obs.index)
    )
    if obs["EB_manual_clustering"].isna().any():
        n_missing = int(obs["EB_manual_clustering"].isna().sum())
        raise ValueError(
            f"{n_missing} ATAC cells could not be matched to RNA EB_manual_clustering labels"
        )
        
    spatial = np.asarray(atac.obsm["X_spatial"], dtype=float)

    # The ATAC modality contains the called-peak count matrix in X.
    peak_counts = sparse.csr_matrix(atac.X)
    if peak_counts.shape != atac.shape:
        raise ValueError("Unexpected ATAC peak matrix shape")
    if peak_counts.data.size and np.nanmin(peak_counts.data) < 0:
        raise ValueError("ATAC X contains negative values; raw/nonnegative peak counts are required")

    # Original workflow normalizes peak pseudobulks by total ATAC counts per cell.
    if "ATAC_total_counts" in obs.columns:
        library = pd.to_numeric(obs["ATAC_total_counts"], errors="coerce").to_numpy(dtype=float)
    else:
        print("WARNING: ATAC_total_counts not found; using row sums of the called-peak matrix as library size")
        library = np.asarray(peak_counts.sum(axis=1)).ravel().astype(float)

    feature_names = atac.var_names.astype(str).to_numpy()

    for dataset in (d for d in DATASETS if d["dataset_type"] == "ligand"):
        destination = peak_batch_input(dataset)
        if destination.is_file():
            print(f"Reusing called-peak input for {dataset['dataset']}: {destination}")
            continue

        print(f"\nPreparing called-peak input directly from H5Mu for {dataset['dataset']}")
        bins, valid, mids = _peak_distance_bins(
            obs, spatial, dataset["condition_label"], bin_width_um=50.0, max_distance_um=300.0
        )
        counts = np.bincount(bins[valid], minlength=6)
        print(f"eligible cells={int(valid.sum())}; bin counts={counts.tolist()}")

        vals = _peak_gp_summaries(
            peak_counts, library, bins, valid, feature_names,
            detection_threshold=0.05, scale_factor=10_000.0, expected_bins=6,
        )
        means, sems, noise, names, idx, detection, bin_counts = vals

        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination,
            X=np.asarray(mids, dtype=float),
            means=np.ascontiguousarray(means),
            sems=np.ascontiguousarray(sems),
            mean_sem_sq=np.ascontiguousarray(noise),
            gene_names=np.asarray(names, dtype=str),
            gene_indices=np.asarray(idx, dtype=int),
            bin_counts=np.asarray(bin_counts, dtype=int),
            detection=np.asarray(detection, dtype=float),
            feature_space=np.asarray("called ATAC peaks"),
            normalization_method=np.asarray("raw distance-bin pseudobulk, library-size normalize, log1p"),
            normalization_formula=np.asarray("log1p(10000 * peak_count / total_ATAC_count)"),
            normalization_order=np.asarray("bin_then_log_cpm"),
            normalization_scale_factor=np.float64(10_000.0),
            spatial_bin_width_um=np.float64(50.0),
            max_distance_um=np.float64(300.0),
        )
        print(f"wrote {destination} ({len(names)} peaks)")


if PREPARE_MISSING_PEAK_INPUTS:
    prepare_all_peak_inputs()
else:
    print("Peak preparation is disabled; existing peak inputs will be inventoried below.")


# ### Prepare 50-kb-bin, chromVAR, and gene-activity GP inputs
# 

# In[15]:


def nonpeak_input_path(dataset, modality_slug):
    return NONPEAK_INPUT_ROOT / modality_slug / "conditions" / dataset["condition_id"] / "gp_inputs.npz"


def _load_h5ad(path):
    # SnapATAC2 H5ADs may use HDF5 compression plugins.
    try:
        import hdf5plugin  # noqa: F401
    except ImportError:
        pass
    import anndata as ad
    return ad.read_h5ad(path)


def _canonical_cell_ids(obs, obs_names):
    """Create several candidate IDs for matching matrices back to the H5Mu cells."""
    candidates = {"obs_names": pd.Index(obs_names.astype(str))}
    for col in ("global_cell_id", "atac_barcode", "atac_barcode_linked", "raw_cell_barcode", "barcode_raw"):
        if col in obs.columns:
            candidates[col] = pd.Index(obs[col].astype(str))
    if "lane" in obs.columns:
        lane = obs["lane"].astype(str).str.replace("lane", "", regex=False)
        for col in ("atac_barcode", "raw_cell_barcode", "barcode_raw"):
            if col in obs.columns:
                raw = obs[col].astype(str)
                candidates[f"lane_prefixed_{col}"] = pd.Index("lane" + lane + "_" + raw)
    return candidates


def _indexer_to_h5mu(matrix_obs_names, h5_obs):
    matrix_ids = pd.Index(matrix_obs_names.astype(str))
    for label, ids in _canonical_cell_ids(h5_obs, h5_obs.index).items():
        if ids.is_unique:
            lookup = pd.Series(np.arange(len(ids)), index=ids)
            idx = lookup.reindex(matrix_ids).to_numpy()
            if pd.notna(idx).all():
                print(f"Matched cells using H5Mu {label}")
                return idx.astype(int)
    raise ValueError("Could not match all matrix cell IDs to the H5Mu metadata. Inspect obs_names/barcode columns.")


def _distance_bins(obs, spatial, condition_label, condition_col="target_call", bin_width_um=25.0, max_distance_um=300.0):
    from scipy.spatial import cKDTree

    eb_col = "EB_manual_clustering" if "EB_manual_clustering" in obs.columns else "EB_manual"
    if eb_col not in obs.columns:
        raise KeyError("H5Mu obs needs EB_manual_clustering or EB_manual")
    if condition_col not in obs.columns:
        raise KeyError(f"H5Mu obs is missing {condition_col}")

    calls = obs[condition_col]
    call_str = calls.astype(str)
    target = call_str.eq(str(condition_label)).to_numpy()
    assigned = calls.notna().to_numpy() & ~call_str.isin(["", "nan", "None", "Unassigned"]).to_numpy()
    query = assigned & ~target

    if target.sum() == 0:
        raise ValueError(f"No cells found with {condition_col} == {condition_label!r}")

    dist = np.full(len(obs), np.nan, dtype=float)
    eb_values = obs[eb_col].astype(str).to_numpy()

    for eb in pd.unique(eb_values):
        in_eb = eb_values == eb
        src = np.flatnonzero(in_eb & target)
        qry = np.flatnonzero(in_eb & query)
        if src.size == 0 or qry.size == 0:
            continue
        tree = cKDTree(np.asarray(spatial[src], dtype=float))
        dist[qry] = tree.query(np.asarray(spatial[qry], dtype=float), k=1)[0]

    edges = np.arange(0.0, max_distance_um + bin_width_um, bin_width_um)
    mids = ((edges[:-1] + edges[1:]) / 2.0)[:, None]
    valid = query & np.isfinite(dist) & (dist >= 0) & (dist < max_distance_um)

    bins = np.full(len(obs), -1, dtype=int)
    bins[valid] = np.floor(dist[valid] / bin_width_um).astype(int)

    return bins, valid, mids

def _count_gp_summaries(matrix, bins, valid, feature_names, detection_threshold=0.05, scale_factor=10000.0):
    Xmat = sparse.csr_matrix(matrix)
    eligible = np.flatnonzero(valid)
    if eligible.size == 0:
        raise ValueError("No eligible binned cells")

    detected = np.asarray((Xmat[eligible] > 0).sum(axis=0)).ravel() / eligible.size
    keep = detected > detection_threshold
    kept_idx = np.flatnonzero(keep)
    if kept_idx.size == 0:
        raise ValueError("No count features pass the >5% detection threshold")
    Xmat = Xmat[:, kept_idx]
    names = np.asarray(feature_names, dtype=str)[kept_idx]
    detection = detected[kept_idx]

    # Library total is calculated from the full source feature space, before feature filtering.
    full = sparse.csr_matrix(matrix)
    library = np.asarray(full.sum(axis=1)).ravel().astype(float)
    B = bins[valid].max() + 1
    # Force all expected 12 bins, including bins with no cells, to be checked explicitly.
    B = 12
    means, sems = [], []
    bin_counts = []
    for b in range(B):
        rows = np.flatnonzero(valid & (bins == b))
        n = rows.size
        bin_counts.append(n)
        if n <= 1:
            raise ValueError(f"Distance bin {b} has only {n} eligible cells; at least 2 are required")
        M = Xmat[rows]
        lib = library[rows]
        if np.any(lib <= 0):
            raise ValueError(f"Distance bin {b} contains cells with zero library size")
        count_sum = np.asarray(M.sum(axis=0)).ravel().astype(float)
        count_sq_sum = np.asarray(M.power(2).sum(axis=0)).ravel().astype(float)
        count_library_sum = np.asarray(M.T @ lib).ravel().astype(float)
        library_sum = float(lib.sum())
        library_sq_sum = float(np.square(lib).sum())
        ratio = count_sum / library_sum
        scaled = scale_factor * ratio
        mean = np.log1p(scaled)
        residual_ss = np.maximum(
            count_sq_sum - 2.0 * ratio * count_library_sum + np.square(ratio) * library_sq_sum,
            0.0,
        )
        ratio_variance = n / (n - 1.0) * residual_ss / (library_sum ** 2)
        sem = scale_factor / (1.0 + scaled) * np.sqrt(ratio_variance)
        means.append(mean)
        sems.append(sem)

    means = np.vstack(means)
    sems = np.vstack(sems)
    mean_sem_sq = np.mean(np.square(sems), axis=0)
    finite = np.all(np.isfinite(means), axis=0) & np.all(np.isfinite(sems), axis=0) & np.isfinite(mean_sem_sq) & (mean_sem_sq > 0)
    return means[:, finite], sems[:, finite], mean_sem_sq[finite], names[finite], kept_idx[finite], detection[finite], np.asarray(bin_counts)


def _score_gp_summaries(matrix, bins, valid, feature_names):
    M = np.asarray(matrix) if not sparse.issparse(matrix) else matrix
    means, sems, bin_counts = [], [], []
    for b in range(12):
        rows = np.flatnonzero(valid & (bins == b))
        n = rows.size
        bin_counts.append(n)
        if n <= 1:
            raise ValueError(f"Distance bin {b} has only {n} eligible cells; at least 2 are required")
        block = M[rows]
        if sparse.issparse(block):
            mean = np.asarray(block.mean(axis=0)).ravel()
            sqmean = np.asarray(block.power(2).mean(axis=0)).ravel()
            var = np.maximum((sqmean - mean**2) * n / (n - 1.0), 0.0)
        else:
            block = np.asarray(block, dtype=float)
            mean = np.mean(block, axis=0)
            var = np.var(block, axis=0, ddof=1)
        means.append(mean)
        sems.append(np.sqrt(var / n))
    means = np.vstack(means)
    sems = np.vstack(sems)
    mean_sem_sq = np.mean(np.square(sems), axis=0)
    keep = np.all(np.isfinite(means), axis=0) & np.all(np.isfinite(sems), axis=0) & np.isfinite(mean_sem_sq) & (mean_sem_sq > 0)
    names = np.asarray(feature_names, dtype=str)
    idx = np.flatnonzero(keep)
    return means[:, keep], sems[:, keep], mean_sem_sq[keep], names[keep], idx, np.asarray(bin_counts)


def _save_gp_input(path, mids, means, sems, mean_sem_sq, names, indices, bin_counts, **extra):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(
        X=np.asarray(mids, dtype=float),
        means=np.ascontiguousarray(means),
        sems=np.ascontiguousarray(sems),
        mean_sem_sq=np.ascontiguousarray(mean_sem_sq),
        gene_names=np.asarray(names, dtype=str),
        gene_indices=np.asarray(indices, dtype=int),
        bin_counts=np.asarray(bin_counts, dtype=int),
    )
    payload.update(extra)
    np.savez_compressed(path, **payload)
    print(f"wrote {path} ({len(names)} features)")


def prepare_nonpeak_inputs():
    if SOURCE_H5MU is None:
        raise FileNotFoundError(missing_file_message(SOURCE_H5MU_CANDIDATES, "source H5Mu"))
    if not all(p.is_file() for p in TILE_FILES):
        missing = [str(p) for p in TILE_FILES if not p.is_file()]
        raise FileNotFoundError(f"Missing 50-kb tile files: {missing}")
    if GENE_ACTIVITY_H5AD is None:
        raise FileNotFoundError(missing_file_message(GENE_ACTIVITY_H5AD_CANDIDATES, "gene-activity H5AD"))

    import mudata as md
    mdata = md.read_h5mu(SOURCE_H5MU)
    atac = mdata.mod["atac"]
    rna = mdata.mod["rna"]
    meta = atac.obs.copy()
    meta["EB_manual_clustering"] = (
        rna.obs["EB_manual_clustering"].reindex(meta.index)
    )
    spatial = np.asarray(atac.obsm["X_spatial"], dtype=float)
    
    # chromVAR is already aligned to the H5Mu ATAC cells.
    chrom = mdata.mod["chromvar"]
    if not chrom.obs_names.equals(mdata.mod["atac"].obs_names):
        chrom = chrom[mdata.mod["atac"].obs_names].copy()

    # Merge the three lane-specific 50-kb tile matrices and align them to H5Mu metadata.
    tile_ads = []
    for lane_i, path in enumerate(TILE_FILES, start=1):
        ad = _load_h5ad(path)
        ids = pd.Index(ad.obs_names.astype(str))
        if not ids.str.startswith(f"lane{lane_i}_").all():
            ad.obs_names = pd.Index([f"lane{lane_i}_{x}" for x in ids])
        tile_ads.append(ad)
    if not all(tile_ads[0].var_names.equals(ad.var_names) for ad in tile_ads[1:]):
        raise ValueError("The three tile files do not have identical 50-kb feature definitions")
    tile_names = tile_ads[0].var_names.astype(str).to_numpy()
    tile_matrix = sparse.vstack([sparse.csr_matrix(ad.X) for ad in tile_ads], format="csr")
    tile_ids = pd.Index(np.concatenate([ad.obs_names.astype(str).to_numpy() for ad in tile_ads]))
    tile_to_h5 = _indexer_to_h5mu(tile_ids, meta)
    inv = np.empty(len(meta), dtype=int)
    inv[tile_to_h5] = np.arange(len(tile_to_h5))
    tile_matrix = tile_matrix[inv]

    # Gene activity H5AD: cells x genes.
    ga = _load_h5ad(GENE_ACTIVITY_H5AD)
    ga_to_h5 = _indexer_to_h5mu(ga.obs_names, meta)
    inv_ga = np.empty(len(meta), dtype=int)
    inv_ga[ga_to_h5] = np.arange(len(ga_to_h5))
    ga_matrix = sparse.csr_matrix(ga.X)[inv_ga]
    ga_names = ga.var_names.astype(str).to_numpy()

    motif_ids = chrom.var_names.astype(str).to_numpy()
    motif_names = chrom.var["motif_name"].astype(str).to_numpy() if "motif_name" in chrom.var.columns else motif_ids.copy()

    for dataset in DATASETS:
        condition_col = (
            "guide_call"
            if dataset["dataset_type"] == "negative_control"
            else "target_call"
        )

        bins, valid, mids = _distance_bins(meta,spatial,dataset["condition_label"],condition_col=condition_col,bin_width_um=25.0,max_distance_um=300.0,)
        print(f"\n{dataset['dataset']}: eligible cells={int(valid.sum())}; bin counts={np.bincount(bins[valid], minlength=12).tolist()}")

        tile_out = nonpeak_input_path(dataset, "snapatac2_tiles_50000bp")
        if not tile_out.is_file():
            vals = _count_gp_summaries(tile_matrix, bins, valid, tile_names)
            means, sems, noise, names, idx, detection, counts = vals
            _save_gp_input(tile_out, mids, means, sems, noise, names, idx, counts,
                           detection=detection, feature_space=np.asarray("50-kb SnapATAC2 tiles"))

        chrom_out = nonpeak_input_path(dataset, "chromvar")
        if not chrom_out.is_file():
            vals = _score_gp_summaries(chrom.X, bins, valid, motif_ids)
            means, sems, noise, names, idx, counts = vals
            _save_gp_input(chrom_out, mids, means, sems, noise, names, idx, counts,
                           motif_ids=motif_ids[idx], motif_names=motif_names[idx], feature_space=np.asarray("chromVAR scores"))

        ga_out = nonpeak_input_path(dataset, "signac_gene_activity")
        if not ga_out.is_file():
            vals = _count_gp_summaries(ga_matrix, bins, valid, ga_names)
            means, sems, noise, names, idx, detection, counts = vals
            _save_gp_input(ga_out, mids, means, sems, noise, names, idx, counts,
                           detection=detection, feature_space=np.asarray("Signac gene activity"))


if PREPARE_MISSING_INPUTS:
    prepare_nonpeak_inputs()
else:
    print("PREPARE_MISSING_INPUTS is False; non-peak inputs were not generated.")


# ## 4. Build the perturbation-by-modality task matrix
# 
# This creates 60 ligand screens (15 ligands × 4 modalities) plus 12 negative-control screens (4 controls × 3 modalities). The fixed-bin modality now uses the **50-kb SnapATAC2 tiles actually present in the project**.
# 

# In[16]:


def modality_specs(dataset):
    specs = {
        "ATAC 50-kb bins": {
            "input": nonpeak_input_path(dataset, "snapatac2_tiles_50000bp"),
            "expected_bins": 12,
            "spatial_bin_width_um": 25,
            "reference_output": None,
        },
        "chromVAR scores": {
            "input": nonpeak_input_path(dataset, "chromvar"),
            "expected_bins": 12,
            "spatial_bin_width_um": 25,
            "reference_output": None,
        },
        "ATAC gene activity": {
            "input": nonpeak_input_path(dataset, "signac_gene_activity"),
            "expected_bins": 12,
            "spatial_bin_width_um": 25,
            "reference_output": None,
        },
    }
    if dataset["dataset_type"] == "ligand":
        specs = {
            "ATAC peaks": {
                "input": peak_input_path(dataset),
                "expected_bins": 6,
                "spatial_bin_width_um": 50,
                "reference_output": None,
            },
            **specs,
        }
    return specs


TASKS = []
for dataset in DATASETS:
    for modality, spec in modality_specs(dataset).items():
        TASKS.append({**dataset, "modality": modality, **spec})

task_table = pd.DataFrame(
    {
        "dataset": task["dataset"],
        "dataset_type": task["dataset_type"],
        "modality": task["modality"],
        "input": str(task["input"]),
    }
    for task in TASKS
)
task_table.groupby(["dataset_type", "modality"]).size().rename("screens").to_frame()


# In[17]:


REQUIRED_ARRAYS = {"X", "means", "sems", "mean_sem_sq", "gene_names", "gene_indices"}


def inspect_gp_input(task):
    path = task["input"]
    base = {"dataset": task["dataset"], "dataset_type": task["dataset_type"], "modality": task["modality"]}
    if not path.is_file():
        return {**base, "status": "missing", "features": np.nan, "distance_bins": task["expected_bins"], "input": str(path)}

    with np.load(path, allow_pickle=False) as data:
        missing = REQUIRED_ARRAYS.difference(data.files)
        if missing:
            raise ValueError(f"{task['dataset']} / {task['modality']} is missing {sorted(missing)}")
        X, means, sems = data["X"], data["means"], data["sems"]
        fixed_noise = data["mean_sem_sq"]
        feature_ids = data["gene_names"]
        if X.shape != (task["expected_bins"], 1):
            raise ValueError(f"Unexpected X shape for {task['dataset']} / {task['modality']}: {X.shape}")
        if means.shape != sems.shape or means.shape != (X.shape[0], feature_ids.size):
            raise ValueError(f"Incompatible feature arrays for {task['dataset']} / {task['modality']}")
        if fixed_noise.shape != (feature_ids.size,):
            raise ValueError(f"Incompatible fixed-noise array for {task['dataset']} / {task['modality']}")
        if not np.allclose(fixed_noise, np.mean(np.square(sems), axis=0), rtol=1e-8, atol=1e-12):
            raise ValueError(f"SEM calibration mismatch for {task['dataset']} / {task['modality']}")
        detection_min = float(np.min(data["detection"])) if "detection" in data.files else np.nan
        return {
            **base, "status": "ready", "features": feature_ids.size,
            "distance_bins": X.shape[0], "midpoint_min_um": float(X[0, 0]),
            "midpoint_max_um": float(X[-1, 0]), "minimum_detection": detection_min,
            "input": str(path),
        }


input_inventory = pd.DataFrame(inspect_gp_input(task) for task in TASKS)
input_inventory.groupby(["dataset_type", "modality", "status"]).size().rename("screens").to_frame()


# ### Processed-value interpretation
# 
# For called peaks and 50-kb bins, fragments are summed within each spatial pseudobulk, library-size normalized, and transformed as `log1p(10000 * feature_count / total_count)`. Signac gene-activity counts use the analogous gene-activity library total. Their SEMs propagate cell-level count and library-size variation while retaining count/library-size covariance. chromVAR inputs instead contain the mean and SEM of per-cell motif deviation scores in each distance bin. The GP screen does not renormalize these values.
# 

# ## 5. Build and execute all GP screens
# 
# The command matrix below contains one independent GP screen per perturbation and modality. Two screens run concurrently by default, with four feature-fitting workers in each. Adjust those values to fit the machine. A first-N diagnostic subset is useful for checking paths and dependencies, but its q-values must not be reported because it changes the multiple-testing universe.
# 

# In[ ]:


def output_directory(task):
    modality_slug = task["modality"].lower().replace(" ", "_").replace("-", "")
    subset = "full" if DIAGNOSTIC_MAX_FEATURES is None else f"first_{DIAGNOSTIC_MAX_FEATURES}"
    constraint = "unconstrained" if MIN_LENGTHSCALE_UM is None else f"min_l{MIN_LENGTHSCALE_UM:g}um"
    return OUTPUT_ROOT / task["dataset"] / modality_slug / f"gp_lrt_{subset}_{constraint}"


def build_gp_command(task):
    if GP_RUNNER is None:
        return None
    command = [
        str(GP_PYTHON), str(GP_RUNNER),
        "--input", str(task["input"]),
        "--output", str(output_directory(task)),
        "--fdr", str(FDR),
        "--rbf-start-lengthscales", *map(str, RBF_START_LENGTHSCALES_UM),
        "--max-optimizer-iters", str(MAX_OPTIMIZER_ITERS),
        "--workers", str(WORKERS_PER_SCREEN),
        "--parallel-backend", "process",
        "--chunk-size", str(CHUNK_SIZE),
        "--expected-bins", str(task["expected_bins"]),
    ]
    if MIN_LENGTHSCALE_UM is not None:
        command.extend(["--rbf-lengthscale-minimum", str(MIN_LENGTHSCALE_UM)])
    if DIAGNOSTIC_MAX_FEATURES is not None:
        command.extend(["--diagnostic-max-genes", str(DIAGNOSTIC_MAX_FEATURES)])
    return command


def _bh_adjust(p):
    p = np.asarray(p, dtype=float)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return q
    pv = p[ok]
    order = np.argsort(pv)
    ranked = pv[order]
    n = len(ranked)
    adj = ranked * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0.0, 1.0)
    restored = np.empty_like(adj)
    restored[order] = adj
    q[ok] = restored
    return q


def _by_adjust(p):
    p = np.asarray(p, dtype=float)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return q
    pv = p[ok]
    order = np.argsort(pv)
    ranked = pv[order]
    n = len(ranked)
    harmonic = np.sum(1.0 / np.arange(1, n + 1))
    adj = ranked * n * harmonic / np.arange(1, n + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0.0, 1.0)
    restored = np.empty_like(adj)
    restored[order] = adj
    q[ok] = restored
    return q


def _gp_loglik(y, K_signal, noise_var):
    """Zero-mean Gaussian-process log marginal likelihood."""
    from scipy.linalg import cho_factor, cho_solve

    n = len(y)
    K = np.asarray(K_signal, dtype=float).copy()
    K.flat[:: n + 1] += float(noise_var)
    # Tiny numerical jitter only; observation noise remains fixed.
    K.flat[:: n + 1] += 1e-10
    cf = cho_factor(K, lower=True, check_finite=False)
    alpha = cho_solve(cf, y, check_finite=False)
    logdet = 2.0 * np.log(np.diag(cf[0])).sum()
    return -0.5 * (y @ alpha + logdet + n * np.log(2.0 * np.pi))


def _fit_bias_gp(y, noise_var):
    from scipy.optimize import minimize_scalar

    yvar = max(float(np.var(y)), 1e-8)
    lo, hi = np.log(1e-10), np.log(max(1e4, yvar * 1e4))

    def objective(log_var):
        var = np.exp(log_var)
        K = np.full((len(y), len(y)), var, dtype=float)
        try:
            return -_gp_loglik(y, K, noise_var)
        except Exception:
            return np.inf

    fit = minimize_scalar(objective, bounds=(lo, hi), method="bounded", options={"maxiter": MAX_OPTIMIZER_ITERS})
    return -float(fit.fun), float(np.exp(fit.x)), bool(fit.success)


def _fit_rbf_gp(x, y, noise_var):
    from scipy.optimize import minimize

    x = np.asarray(x, dtype=float).ravel()
    d2 = np.square(x[:, None] - x[None, :])
    yvar = max(float(np.var(y)), 1e-8)
    min_ls = 1e-3 if MIN_LENGTHSCALE_UM is None else float(MIN_LENGTHSCALE_UM)
    max_ls = 1e6
    bounds = [(np.log(1e-10), np.log(max(1e4, yvar * 1e4))), (np.log(min_ls), np.log(max_ls))]

    def objective(theta):
        var, ls = np.exp(theta)
        K = var * np.exp(-0.5 * d2 / (ls * ls))
        try:
            return -_gp_loglik(y, K, noise_var)
        except Exception:
            return np.inf

    best = None
    for start_ls in RBF_START_LENGTHSCALES_UM:
        start_ls = max(float(start_ls), min_ls)
        theta0 = np.log([yvar, start_ls])
        fit = minimize(
            objective, theta0, method="L-BFGS-B", bounds=bounds,
            options={"maxiter": MAX_OPTIMIZER_ITERS},
        )
        if np.isfinite(fit.fun) and (best is None or fit.fun < best.fun):
            best = fit

    if best is None:
        return np.nan, np.nan, np.nan, False
    var, ls = np.exp(best.x)
    return -float(best.fun), float(var), float(ls), bool(best.success)


def _run_gp_screen_internal(task):
    """Self-contained implementation of the notebook's Bias-vs-RBF GP LRT."""
    from scipy.stats import chi2

    with np.load(task["input"], allow_pickle=False) as data:
        X = np.asarray(data["X"], dtype=float).ravel()
        means = np.asarray(data["means"], dtype=float)
        noise = np.asarray(data["mean_sem_sq"], dtype=float)
        genes = np.asarray(data["gene_names"], dtype=str)
        detection = np.asarray(data["detection"], dtype=float) if "detection" in data.files else np.full(genes.size, np.nan)

    if DIAGNOSTIC_MAX_FEATURES is not None:
        ntest = min(int(DIAGNOSTIC_MAX_FEATURES), genes.size)
        means, noise, genes, detection = means[:, :ntest], noise[:ntest], genes[:ntest], detection[:ntest]

    rows = []
    for j, gene in enumerate(genes):
        y = means[:, j]
        try:
            ll0, bias_var, ok0 = _fit_bias_gp(y, noise[j])
            ll1, rbf_var, ls, ok1 = _fit_rbf_gp(X, y, noise[j])
            lrt = max(0.0, 2.0 * (ll1 - ll0)) if np.isfinite(ll0) and np.isfinite(ll1) else np.nan
            # Boundary mixture: 0.5*chi^2_0 + 0.5*chi^2_1. For positive LRT, survival = 0.5*chi2_1.sf.
            p_boundary = 1.0 if lrt == 0 else (0.5 * chi2.sf(lrt, 1) if np.isfinite(lrt) else np.nan)
            p_chi1 = chi2.sf(lrt, 1) if np.isfinite(lrt) else np.nan
            success = bool(ok0 and ok1 and np.isfinite(lrt))
            err = ""
        except Exception as exc:
            ll0 = ll1 = bias_var = rbf_var = ls = lrt = p_boundary = p_chi1 = np.nan
            success = False
            err = f"{type(exc).__name__}: {exc}"

        rows.append({
            "gene": gene,
            "gene_index": j,
            "detection_fraction": detection[j],
            "expression_range": float(np.ptp(y)),
            "bias_log_likelihood": ll0,
            "rbf_log_likelihood": ll1,
            "likelihood_ratio_statistic": lrt,
            "p_value": p_boundary,
            "p_value_chi1": p_chi1,
            "bias_variance": bias_var,
            "rbf_variance": rbf_var,
            "rbf_lengthscale_um": ls,
            "fixed_noise_variance": noise[j],
            "fit_success": success,
            "fit_error": err,
        })

    table = pd.DataFrame(rows)
    table["fdr_q_value"] = _bh_adjust(table["p_value"].to_numpy())
    table["significant_fdr"] = table["fit_success"] & (table["fdr_q_value"] <= FDR)
    table["by_q_value"] = _by_adjust(table["p_value"].to_numpy())
    table["significant_by"] = table["fit_success"] & (table["by_q_value"] <= FDR)
    table["chi1_fdr_q_value"] = _bh_adjust(table["p_value_chi1"].to_numpy())

    output = output_directory(task)
    output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output / "spatial_lrt_results.tsv", sep="\t", index=False)
    config = {
        "status": "completed",
        "runner": "internal_notebook_gp",
        "input": str(task["input"]),
        "dataset": task["dataset"],
        "modality": task["modality"],
        "fdr": FDR,
        "rbf_start_lengthscales_um": list(RBF_START_LENGTHSCALES_UM),
        "max_optimizer_iters": MAX_OPTIMIZER_ITERS,
        "min_lengthscale_um": MIN_LENGTHSCALE_UM,
        "expected_bins": task["expected_bins"],
        "n_features_tested": int(len(table)),
    }
    (output / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    return f"completed internally: {task['dataset']} / {task['modality']}"


def run_gp_screen(task):
    if not task["input"].is_file():
        raise FileNotFoundError(f"Missing input for {task['dataset']} / {task['modality']}: {task['input']}")
    output = output_directory(task)
    config_path = output / "run_config.json"
    if config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("status") == "completed":
            return f"reused: {task['dataset']} / {task['modality']}"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite incomplete/nonempty output: {output}")

    command = build_gp_command(task)
    if command is not None:
        subprocess.run(command, cwd=WORKSPACE, check=True)
        return f"completed with external runner: {task['dataset']} / {task['modality']}"
    return _run_gp_screen_internal(task)


def preview_gp_command(task):
    command = build_gp_command(task)
    if command is None:
        return "INTERNAL NOTEBOOK GP RUNNER"
    return shlex.join(command)


command_table = pd.DataFrame(
    {
        "dataset": task["dataset"], "modality": task["modality"],
        "input_ready": task["input"].is_file(),
        "command": preview_gp_command(task),
    }
    for task in TASKS
)

if RUN_SCREENS:
    missing_tasks = [task for task in TASKS if not task["input"].is_file()]
    if missing_tasks:
        labels = ", ".join(f"{task['dataset']}/{task['modality']}" for task in missing_tasks)
        raise FileNotFoundError(
            f"{len(missing_tasks)} GP inputs are missing: {labels}. "
            "Set PREPARE_MISSING_PEAK_INPUTS=True and PREPARE_MISSING_INPUTS=True before the full run."
        )
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_SCREENS) as executor:
        futures = {executor.submit(run_gp_screen, task): task for task in TASKS}
        for future in as_completed(futures):
            print(future.result(), flush=True)
else:
    print("RUN_SCREENS is False; the all-condition command matrix was built but not executed.")

command_table.groupby(["modality", "input_ready"]).size().rename("screens").to_frame()


# ## 6. Aggregate completed results across all conditions
# 
# This section reads newly generated outputs under the notebook output tree. If you rerun the notebook later with `RUN_SCREENS=False`, completed notebook outputs are still discovered automatically.
# 

# In[ ]:


def available_result_directory(task):
    # Prefer outputs produced by this notebook, even when RUN_SCREENS=False on a later inspection run.
    candidates = [output_directory(task)]
    if task.get("reference_output") is not None:
        candidates.append(task["reference_output"])
    for candidate in candidates:
        table_path = candidate / "spatial_lrt_results.tsv"
        config_path = candidate / "run_config.json"
        if table_path.is_file() and config_path.is_file():
            return candidate
    return None


def load_results(task):
    directory = available_result_directory(task)
    if directory is None:
        return None
    table = pd.read_csv(directory / "spatial_lrt_results.tsv", sep="\t")
    table.insert(0, "dataset", task["dataset"])
    table.insert(1, "dataset_type", task["dataset_type"])
    table.insert(2, "modality", task["modality"])
    table.insert(4, "display_feature", table["gene"])
    if task["modality"] == "chromVAR scores":
        with np.load(task["input"], allow_pickle=False) as data:
            motif_names = dict(zip(data["motif_ids"].astype(str), data["motif_names"].astype(str)))
        table["display_feature"] = table["gene"].map(motif_names).fillna(table["gene"])
    return table


RESULTS = {}
for task in TASKS:
    table = load_results(task)
    if table is not None:
        RESULTS[(task["dataset"], task["modality"])] = table

run_summary = pd.DataFrame(
    {
        "dataset": dataset, "modality": modality,
        "tested": len(table),
        "fit_failures": int((~table["fit_success"]).sum()),
        "significant_at_5pct_fdr": int(table["significant_fdr"].sum()),
        "minimum_q_value": float(table["fdr_q_value"].min()),
    }
    for (dataset, modality), table in RESULTS.items()
)
print(f"Loaded {len(run_summary)} completed perturbation-by-modality screens.")
if len(run_summary):
    display(run_summary.sort_values(["modality", "dataset"]).reset_index(drop=True))
else:
    print("No completed GP result tables were found yet.")


# In[ ]:


TOP_COLUMNS = [
    "dataset", "dataset_type", "modality", "gene", "display_feature",
    "detection_fraction", "expression_range", "likelihood_ratio_statistic",
    "p_value", "fdr_q_value", "rbf_lengthscale_um", "significant_fdr",
]
if RESULTS:
    top_hit_per_screen = pd.concat(
        [
            table.loc[table["fit_success"]]
            .sort_values(["p_value", "likelihood_ratio_statistic"], ascending=[True, False])
            .head(1)[TOP_COLUMNS]
            for table in RESULTS.values()
        ],
        ignore_index=True,
    ).sort_values(["modality", "fdr_q_value", "dataset"])
    display(top_hit_per_screen.reset_index(drop=True))
else:
    top_hit_per_screen = pd.DataFrame(columns=TOP_COLUMNS)
    print("No completed results available for top-hit summary yet.")


# ## Reproducibility checklist
# 
# For the final all-ligand rerun, use `DIAGNOSTIC_MAX_FEATURES = None`, keep ligand/modality screens in separate output directories, and archive each `run_config.json` plus its input SHA-256 hash. Confirm that all 60 ligand screens completed, inspect optimizer failures and boundary flags, and analyze the negative controls alongside ligand results. If a minimum lengthscale is imposed, label that output as a constrained sensitivity analysis rather than combining it with the unconstrained primary screen.
# 

# In[2]:


from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path("/scratch/welchjd_root/welchjd1/javidgmh/15515_Multiome/catatac_work/motif_analysis")
RESULT_ROOT = ROOT / "results" / "notebook_multimodal_atac_gp_all_conditions_50kb"
PLOT_ROOT = RESULT_ROOT / "top8_gp_plots"
PLOT_ROOT.mkdir(parents=True, exist_ok=True)

# Rename these however you want in the figures
DISPLAY_MODALITY = {
    "ATAC peaks": "ATAC",
    "ATAC 50-kb bins": "ATAC_50kb",
    "chromVAR scores": "chromVAR",
    "ATAC gene activity": "ATAC_gene_activity",
}

for result_file in RESULT_ROOT.glob("*/**/gp_lrt_full_unconstrained/spatial_lrt_results.tsv"):
    outdir = result_file.parent
    config_file = outdir / "run_config.json"
    if not config_file.exists():
        continue

    config = json.loads(config_file.read_text())
    kd = config["dataset"]
    modality = config["modality"]
    input_file = Path(config["input"])

    df = pd.read_csv(result_file, sep="\t")
    df = (
        df[df["fit_success"]]
        .sort_values(["p_value", "likelihood_ratio_statistic"], ascending=[True, False])
        .head(8)
    )

    if df.empty:
        continue

    with np.load(input_file, allow_pickle=True) as d:
        x = np.asarray(d["X"]).ravel()
        means = np.asarray(d["means"])
        sems = np.asarray(d["sems"])
        genes = np.asarray(d["gene_names"]).astype(str)
        motif_map = {}
        if modality == "chromVAR scores" and "motif_ids" in d.files and "motif_names" in d.files:
            motif_map = dict(zip(
                d["motif_ids"].astype(str),
                d["motif_names"].astype(str)
            ))

    gene_to_idx = {g: i for i, g in enumerate(genes)}

    fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharex=True)
    axes = axes.ravel()

    for ax, (_, row) in zip(axes, df.iterrows()):
        gene = str(row["gene"])
        j = gene_to_idx[gene]

        y = means[:, j]
        se = sems[:, j]
        label = motif_map.get(gene, gene)

        ax.plot(x, y, marker="o", linewidth=2)
        ax.fill_between(x, y - se, y + se, alpha=0.2)

        ax.set_title(
            f"{label}\nq={row['fdr_q_value']:.2e}, LRT={row['likelihood_ratio_statistic']:.1f}",
            fontsize=9
        )
        ax.set_xlabel("Distance from KD cells (µm)")
        ax.set_ylabel("Mean signal")

    display_modality = DISPLAY_MODALITY.get(modality, modality)

    fig.suptitle(
        f"{kd} KD — Top 8 spatial GP hits — {display_modality}",
        fontsize=14
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    save_dir = PLOT_ROOT / kd
    save_dir.mkdir(parents=True, exist_ok=True)

    outfile = save_dir / f"{display_modality}_top8_GP.png"
    fig.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {outfile}")


# In[6]:


from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

ROOT = Path("/scratch/welchjd_root/welchjd1/javidgmh/15515_Multiome/catatac_work/motif_analysis")
RESULT_ROOT = ROOT / "results" / "notebook_multimodal_atac_gp_all_conditions_50kb"

records = []

for result_file in RESULT_ROOT.glob("*/**/gp_lrt_full_unconstrained/spatial_lrt_results.tsv"):
    run_dir = result_file.parent
    config_file = run_dir / "run_config.json"
    if not config_file.exists():
        continue

    cfg = json.loads(config_file.read_text())
    kd = cfg["dataset"]
    modality = cfg["modality"]

    # Biological KDs only
    if cfg.get("dataset_type") == "negative_control" or kd.startswith("NTC"):
        continue

    df = pd.read_csv(result_file, sep="\t")
    df = df[(df["fit_success"]) & (df["fdr_q_value"] <= 0.05)]
    if df.empty:
        continue

    with np.load(cfg["input"], allow_pickle=True) as d:
        x = np.asarray(d["X"]).ravel()
        means = np.asarray(d["means"])
        features = np.asarray(d["gene_names"]).astype(str)

    feature_to_idx = {g: i for i, g in enumerate(features)}

    inc = 0
    dec = 0

    for feature in df["gene"].astype(str):
        if feature not in feature_to_idx:
            continue

        y = means[:, feature_to_idx[feature]]
        rho, _ = spearmanr(x, y)

        if np.isfinite(rho):
            if rho > 0:
                inc += 1
            elif rho < 0:
                dec += 1

    records.append({
        "KD": kd,
        "modality": modality,
        "Increasing with distance": inc,
        "Decreasing with distance": dec,
    })

summary = pd.DataFrame(records)

modalities = [
    "ATAC peaks",
    "ATAC 50-kb bins",
    "ATAC gene activity",
    "chromVAR scores",
]

labels = {
    "ATAC peaks": "ATAC peaks",
    "ATAC 50-kb bins": "ATAC fixed bins",
    "ATAC gene activity": "ATAC gene activity",
    "chromVAR scores": "chromVAR",
}

# Same KD ordering in every panel
kd_order = sorted(summary["KD"].unique())

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

fig, axes = plt.subplots(
    2, 2,
    figsize=(7.2, 6.5),
    constrained_layout=True
)
axes = axes.ravel()

for ax, modality in zip(axes, modalities):
    sub = (
        summary[summary["modality"] == modality]
        .set_index("KD")
        .reindex(kd_order)
        .fillna(0)
    )

    y = np.arange(len(kd_order))
    h = 0.34

    ax.barh(
        y - h/2,
        sub["Increasing with distance"],
        height=h,
        label="Increasing"
    )

    ax.barh(
        y + h/2,
        sub["Decreasing with distance"],
        height=h,
        label="Decreasing"
    )

    ax.set_yticks(y)
    ax.set_yticklabels(kd_order)
    ax.invert_yaxis()

    ax.set_title(labels[modality], pad=5)
    ax.set_xlabel("FDR-significant features")

    # Nature-style cleanup
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="both", width=0.8, length=3)

    ax.legend(
        frameon=False,
        loc="lower right"
    )

fig.suptitle(
    "Direction of GP-significant spatial features",
    fontsize=11,
    y=1.02
)

png = RESULT_ROOT / "GP_significant_direction_summary_publication.png"
pdf = RESULT_ROOT / "GP_significant_direction_summary_publication.pdf"

fig.savefig(png, dpi=600, bbox_inches="tight")
fig.savefig(pdf, bbox_inches="tight")
plt.show()

print(png)
print(pdf)


# In[4]:





# In[2]:


#!/usr/bin/env python3
from pathlib import Path
import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import solve_triangular
WORKSPACE=Path("/scratch/welchjd_root/welchjd1/javidgmh/15515_Multiome/catatac_work/motif_analysis")
RESULT_ROOT=WORKSPACE/"results"/"notebook_multimodal_atac_gp_all_conditions_50kb"
PLOT_ROOT=RESULT_ROOT/"top8_gp_predictive_plots"
TOP_N=8
FDR_ONLY=False
FDR_THRESHOLD=0.05
LIGANDS=("WNT5A","WNT1","BMP5","BMP7","JAG1","DLL3","VEGFB","PDGFC","EFNA3","RSPO3","SEMA6D","NECTIN2","NECTIN3","ALB","B2M")
MODALITIES=("ATAC peaks","ATAC 50-kb bins","chromVAR scores","ATAC gene activity")
MODALITY_SLUG={"ATAC peaks":"atac_peaks","ATAC 50-kb bins":"atac_50kb_bins","chromVAR scores":"chromvar_scores","ATAC gene activity":"atac_gene_activity"}
FILE_SLUG={"ATAC peaks":"ATAC_peaks","ATAC 50-kb bins":"ATAC_50kb_bins","chromVAR scores":"chromVAR","ATAC gene activity":"ATAC_gene_activity"}
Y_LABEL={"ATAC peaks":"Log-normalized peak accessibility","ATAC 50-kb bins":"Log-normalized tile accessibility","chromVAR scores":"Mean chromVAR deviation score","ATAC gene activity":"Log-normalized ATAC gene activity"}
plt.rcParams.update({"font.family":"Arial","font.size":8,"axes.titlesize":8,"axes.labelsize":8.5,"xtick.labelsize":7.5,"ytick.labelsize":7.5,"legend.fontsize":7.5,"axes.linewidth":0.8,"pdf.fonttype":42,"ps.fonttype":42})
def _bool_series(s):
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)
    return s.astype(str).str.strip().str.lower().isin({"true","1","yes"})
def _rbf_kernel(a,b,variance,lengthscale):
    a=np.asarray(a,dtype=float).ravel()
    b=np.asarray(b,dtype=float).ravel()
    return float(variance)*np.exp(-0.5*np.square(a[:,None]-b[None,:])/float(lengthscale)**2)
def _rbf_predictive(x_train,y_train,x_pred,variance,lengthscale,noise_variance,jitter=1e-10):
    x_train=np.asarray(x_train,dtype=float).ravel()
    y_train=np.asarray(y_train,dtype=float).ravel()
    x_pred=np.asarray(x_pred,dtype=float).ravel()
    K=_rbf_kernel(x_train,x_train,variance,lengthscale)
    C=K.copy()
    C.flat[::C.shape[0]+1]+=float(noise_variance)+float(jitter)
    L=np.linalg.cholesky(C)
    alpha=solve_triangular(L.T,solve_triangular(L,y_train,lower=True,check_finite=False),lower=False,check_finite=False)
    Ksx=_rbf_kernel(x_pred,x_train,variance,lengthscale)
    mean=Ksx@alpha
    v=solve_triangular(L,Ksx.T,lower=True,check_finite=False)
    latent_var=np.maximum(float(variance)-np.sum(v*v,axis=0),0.0)
    predictive_var=np.maximum(latent_var+float(noise_variance),0.0)
    return mean,predictive_var
def _result_dir(ligand,modality):
    return RESULT_ROOT/ligand/MODALITY_SLUG[modality]/"gp_lrt_full_unconstrained"
def _variance_from_row(row):
    if "rbf_variance" in row.index and pd.notna(row["rbf_variance"]):
        return float(row["rbf_variance"])
    if "rbf_kernel_variance" in row.index and pd.notna(row["rbf_kernel_variance"]):
        return float(row["rbf_kernel_variance"])
    raise KeyError("Result table has neither rbf_variance nor rbf_kernel_variance")
def _load_input(path,modality):
    with np.load(path,allow_pickle=True) as d:
        required={"X","means","sems","mean_sem_sq","gene_names"}
        missing=required.difference(d.files)
        if missing:
            raise ValueError(f"{path} missing arrays: {sorted(missing)}")
        x=np.asarray(d["X"],dtype=float).ravel()
        means=np.asarray(d["means"],dtype=float)
        sems=np.asarray(d["sems"],dtype=float)
        noise=np.asarray(d["mean_sem_sq"],dtype=float).ravel()
        genes=np.asarray(d["gene_names"]).astype(str)
        motif_map={}
        if modality=="chromVAR scores" and "motif_ids" in d.files and "motif_names" in d.files:
            motif_map=dict(zip(np.asarray(d["motif_ids"]).astype(str),np.asarray(d["motif_names"]).astype(str)))
    if means.shape!=sems.shape or means.shape!=(x.size,genes.size):
        raise ValueError(f"Incompatible GP input shapes in {path}: X={x.shape}, means={means.shape}, sems={sems.shape}, genes={genes.shape}")
    if noise.size!=genes.size:
        raise ValueError(f"Noise vector length mismatch in {path}")
    return x,means,sems,noise,genes,motif_map
def _feature_position(gene,genes):
    hit=np.flatnonzero(genes==str(gene))
    if hit.size!=1:
        raise ValueError(f"Expected exactly one input feature named {gene!r}; found {hit.size}")
    return int(hit[0])
def _safe_name(text):
    return re.sub(r"[^A-Za-z0-9_.-]+","_",str(text)).strip("_")
def plot_screen(ligand,modality):
    run_dir=_result_dir(ligand,modality)
    result_file=run_dir/"spatial_lrt_results.tsv"
    config_file=run_dir/"run_config.json"
    if not result_file.is_file() or not config_file.is_file():
        print(f"SKIP missing result: {ligand} / {modality}")
        return None
    cfg=json.loads(config_file.read_text())
    input_path=Path(cfg["input"])
    if not input_path.is_file():
        raise FileNotFoundError(f"Missing GP input for {ligand} / {modality}: {input_path}")
    table=pd.read_csv(result_file,sep="\t")
    needed={"gene","fit_success","p_value","fdr_q_value","likelihood_ratio_statistic","rbf_lengthscale_um","fixed_noise_variance"}
    missing=needed.difference(table.columns)
    if missing:
        raise ValueError(f"{result_file} missing columns: {sorted(missing)}")
    good=_bool_series(table["fit_success"])
    ranked=table.loc[good].copy()
    if FDR_ONLY:
        ranked=ranked.loc[pd.to_numeric(ranked["fdr_q_value"],errors="coerce")<=FDR_THRESHOLD].copy()
    ranked=ranked.sort_values(["p_value","likelihood_ratio_statistic"],ascending=[True,False],kind="stable").head(TOP_N).reset_index(drop=True)
    if ranked.empty:
        print(f"SKIP no eligible fits: {ligand} / {modality}")
        return None
    x,means,sems,input_noise,genes,motif_map=_load_input(input_path,modality)
    prediction_x=np.linspace(float(x.min()),float(x.max()),401)
    fig,axes=plt.subplots(2,4,figsize=(13.5,7.4),sharex=True)
    axes=axes.ravel()
    for ax in axes[len(ranked):]:
        ax.axis("off")
    for ax,(_,row) in zip(axes,ranked.iterrows()):
        gene=str(row["gene"])
        j=_feature_position(gene,genes)
        y=means[:,j]
        sem=sems[:,j]
        noise=float(row["fixed_noise_variance"])
        if not np.isclose(noise,input_noise[j],rtol=1e-8,atol=1e-12):
            raise ValueError(f"Noise mismatch for {ligand} / {modality} / {gene}: result={noise}, input={input_noise[j]}")
        variance=_variance_from_row(row)
        lengthscale=float(row["rbf_lengthscale_um"])
        if not (np.isfinite(variance) and variance>0 and np.isfinite(lengthscale) and lengthscale>0 and np.isfinite(noise) and noise>0):
            raise ValueError(f"Invalid saved GP parameters for {ligand} / {modality} / {gene}")
        pred_mean,pred_var=_rbf_predictive(x,y,prediction_x,variance,lengthscale,noise)
        pred_sd=np.sqrt(pred_var)
        lower=pred_mean-1.96*pred_sd
        upper=pred_mean+1.96*pred_sd
        train_mean,train_var=_rbf_predictive(x,y,x,variance,lengthscale,noise)
        train_sd=np.sqrt(train_var)
        coverage=float(np.mean((y>=train_mean-1.96*train_sd)&(y<=train_mean+1.96*train_sd)))
        label=motif_map.get(gene,gene)
        q=float(row["fdr_q_value"])
        lrt=float(row["likelihood_ratio_statistic"])
        sig="yes" if np.isfinite(q) and q<=FDR_THRESHOLD else "no"
        ax.fill_between(prediction_x,lower,upper,color="#4C78A8",alpha=0.22,label="95% posterior predictive interval")
        ax.plot(prediction_x,pred_mean,color="#1F4E79",linewidth=1.8,label="RBF GP posterior mean")
        ax.errorbar(x,y,yerr=sem,fmt="o",markersize=3.2,color="black",ecolor="#555555",elinewidth=0.8,capsize=2,label="Bin mean ± SEM")
        ax.set_title(f"{label}\nBH FDR={sig}  |  q={q:.1e}  |  LRT={lrt:.1f}\nℓ={lengthscale:.1f} µm  |  coverage={coverage:.0%}",pad=7,linespacing=1.25)
        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both",width=0.8,length=3)
    for ax in axes[4:]:
        if ax.axison:
            ax.set_xlabel(f"Distance to nearest {ligand} KD cell (µm)")
    for ax in axes[::4]:
        if ax.axison:
            ax.set_ylabel(Y_LABEL[modality])
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="lower center",ncol=3,frameon=False,bbox_to_anchor=(0.5,-0.005))
    criterion="FDR-significant fits" if FDR_ONLY else "successful fits ranked by GP LRT p-value"
    fig.suptitle(f"{ligand} KD — top {len(ranked)} {modality}: {criterion}",fontsize=11,y=0.985)
    fig.subplots_adjust(left=0.075,right=0.995,bottom=0.13,top=0.86,wspace=0.30,hspace=0.60)
    out_dir=PLOT_ROOT/ligand
    out_dir.mkdir(parents=True,exist_ok=True)
    stem=f"{_safe_name(ligand)}_{FILE_SLUG[modality]}_top{len(ranked)}_GP_predictive"
    png=out_dir/f"{stem}.png"
    pdf=out_dir/f"{stem}.pdf"
    fig.savefig(png,dpi=600,bbox_inches="tight")
    fig.savefig(pdf,bbox_inches="tight")
    plt.close(fig)
    print(f"SAVED {ligand} / {modality}: {png}")
    return {"ligand":ligand,"modality":modality,"n_plotted":len(ranked),"png":str(png),"pdf":str(pdf)}
def main():
    if not RESULT_ROOT.is_dir():
        raise FileNotFoundError(RESULT_ROOT)
    PLOT_ROOT.mkdir(parents=True,exist_ok=True)
    outputs=[]
    failures=[]
    for ligand in LIGANDS:
        for modality in MODALITIES:
            try:
                out=plot_screen(ligand,modality)
                if out is not None:
                    outputs.append(out)
            except Exception as exc:
                failures.append((ligand,modality,f"{type(exc).__name__}: {exc}"))
                print(f"ERROR {ligand} / {modality}: {type(exc).__name__}: {exc}")
    if outputs:
        pd.DataFrame(outputs).to_csv(PLOT_ROOT/"plot_manifest.tsv",sep="\t",index=False)
    print(f"\nCreated {len(outputs)} KD-by-modality figures under: {PLOT_ROOT}")
    if failures:
        print(f"Failures: {len(failures)}")
        for ligand,modality,msg in failures:
            print(f"  {ligand} / {modality}: {msg}")
        raise SystemExit(1)
if __name__=="__main__":
    main()


# In[ ]:




