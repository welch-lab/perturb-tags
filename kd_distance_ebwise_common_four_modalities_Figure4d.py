# kd_distance_ebwise_common_four_modalities (1)
#!/usr/bin/env python3
from __future__ import annotations

import gc
import hashlib
import importlib.util
import os
import re
import shutil
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import anndata as ad
import matplotlib.pyplot as plt
import mudata as md
import numpy as np
import pandas as pd
from scipy.sparse import issparse
from scipy.stats import rankdata, spearmanr
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

DATA_DIR = Path(
    "/scratch/welchjd_root/welchjd1/javidgmh/15515_Multiome/catatac_work/motif_analysis"
)

H5MU_PATH = DATA_DIR / "all_lanes_merged_CATATAC_QC_RNA_processed_EB_manual_clustering_cell_types.h5mu"
TE_RNA_PATH = DATA_DIR / "aggregated_across_lanes_TE_RNA_length_transcript_CPM.h5ad"
TE_ATAC_PATH = DATA_DIR / "aggregated_atac_TE_completely_normalized_CPM.h5ad"

TOP10_LIGAND_KD_TARGETS = [
    "WNT5A",
    "WNT1",
    "BMP5",
    "BMP7",
    "JAG1",
    "DLL3",
    "VEGFB",
    "PDGFC",
    "EFNA3",
    "RSPO3",
]

MAX_DISTANCE = 300.0
DISTANCE_BINS: List[Tuple[float, float]] = [
    (0, 50),
    (50, 100),
    (100, 150),
    (150, 200),
    (200, 250),
    (250, 300),
]
BIN_LABELS = ["0-50", "50-100", "100-150", "150-200", "200-250", "250-300"]
BIN_MIDPOINTS = np.array([25, 75, 125, 175, 225, 275], dtype=float)

SEED = 1
MIN_KD_CELLS = 3
MIN_RECEIVER_CELLS = 10
FDR_ALPHA = 0.05
FEATURE_BLOCK_SIZE = 1000

GUIDE_COL = "guides_passing_str"
EB_COL = "EB_manual_clustering"
LANE_COL = "lane"
SPATIAL_KEY = "X_spatial"

ATAC_VALUE_LAYER: Optional[str] = None
ATAC_DETECTION_LAYER = "binary_accessibility"
TE_RNA_VALUE_LAYER: Optional[str] = None
TE_RNA_DETECTION_LAYER = "raw"
TE_ATAC_VALUE_LAYER: Optional[str] = None
CHROMVAR_VALUE_LAYER: Optional[str] = None


@dataclass(frozen=True)
class AnalysisConfig:
    modality: str
    default_output_dir_name: str
    detection_threshold: float
    n_perm: int
    save_full_nulls: bool = True
    compress_nulls: bool = False
    kd_targets: Tuple[str, ...] = tuple(TOP10_LIGAND_KD_TARGETS)


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")


def to_str_array(values: Iterable) -> np.ndarray:
    if isinstance(values, pd.Series):
        arr = values.astype(object).to_numpy()
    elif isinstance(values, pd.Index):
        arr = values.astype(object).to_numpy()
    else:
        arr = np.asarray(values, dtype=object)
    arr = np.asarray(arr, dtype=object)
    arr[pd.isna(arr)] = ""
    return np.asarray([str(x).strip() for x in arr], dtype=object)


def safe_filename(value: str, max_length: int = 180) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_")
    return cleaned[:max_length] if cleaned else "unnamed"


def get_matrix(adata: ad.AnnData, layer: Optional[str]):
    if layer is None:
        return adata.X
    if layer not in adata.layers:
        raise KeyError(f"Layer '{layer}' missing. Available: {list(adata.layers.keys())}")
    return adata.layers[layer]


def make_unique_names(names: Sequence[str]) -> np.ndarray:
    seen: Dict[str, int] = {}
    output: List[str] = []
    for value in names:
        name = str(value)
        seen[name] = seen.get(name, 0) + 1
        output.append(name if seen[name] == 1 else f"{name}__dup{seen[name]}")
    return np.asarray(output, dtype=object)


def parse_guide_target(guide_name: str) -> str:
    guide_name = str(guide_name).strip()
    if not guide_name or guide_name.lower().startswith("negative control"):
        return ""
    return re.sub(r"-\d+$", "", guide_name)


def target_guide_mask(guide_strings: np.ndarray, target: str) -> np.ndarray:
    output = np.zeros(len(guide_strings), dtype=bool)
    for i, raw in enumerate(guide_strings):
        tokens = re.split(r"[;,|\s]+", str(raw))
        output[i] = any(parse_guide_target(token) == target for token in tokens if token.strip())
    return output


def make_lane_barcode_keys(lanes: Iterable, barcodes: Iterable) -> np.ndarray:
    lane_arr = to_str_array(lanes)
    barcode_arr = to_str_array(barcodes)
    if len(lane_arr) != len(barcode_arr):
        raise ValueError("Lane and barcode lengths differ")
    return np.asarray([f"{lane}|{barcode}" for lane, barcode in zip(lane_arr, barcode_arr)])


def build_exact_external_map(
    main_keys: np.ndarray,
    external_keys: np.ndarray,
    external_name: str,
) -> np.ndarray:
    if pd.Index(external_keys).duplicated().any():
        duplicates = pd.Index(external_keys)[pd.Index(external_keys).duplicated(keep=False)]
        raise ValueError(f"{external_name} duplicate composite keys: {pd.unique(duplicates)[:10].tolist()}")
    lookup = {str(key): i for i, key in enumerate(external_keys)}
    mapped = np.asarray([lookup.get(str(key), -1) for key in main_keys], dtype=int)
    if np.any(mapped < 0):
        bad = np.where(mapped < 0)[0][:10]
        raise ValueError(
            f"{external_name} matched {(mapped >= 0).sum()}/{len(mapped)} cells. "
            f"Unmatched examples: {[str(main_keys[i]) for i in bad]}"
        )
    return mapped


def map_te_rna_cells(main_obs: pd.DataFrame, te_rna: ad.AnnData) -> np.ndarray:
    main_keys = make_lane_barcode_keys(main_obs[LANE_COL], main_obs["gex_barcode"])
    external_keys = make_lane_barcode_keys(te_rna.obs["lane"], te_rna.obs["barcode"])
    return build_exact_external_map(main_keys, external_keys, "TE RNA")


def map_te_atac_cells(main_obs: pd.DataFrame, te_atac: ad.AnnData) -> np.ndarray:
    main_keys = make_lane_barcode_keys(main_obs[LANE_COL], main_obs["raw_cell_barcode"])
    external_keys = make_lane_barcode_keys(te_atac.obs["lane"], te_atac.obs_names)
    return build_exact_external_map(main_keys, external_keys, "TE ATAC")


def subset_dense(matrix, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    sub = matrix[rows][:, cols]
    if issparse(sub):
        return sub.toarray().astype(float, copy=False)
    return np.asarray(sub, dtype=float)


def materialize_dense_matrix(
    matrix,
    rows: np.ndarray,
    cols: np.ndarray,
    dtype=np.float32,
) -> np.ndarray:
    if len(rows) == 0 or len(cols) == 0:
        return np.empty((len(rows), len(cols)), dtype=dtype)
    output = np.empty((len(rows), len(cols)), dtype=dtype)
    for start in range(0, len(cols), FEATURE_BLOCK_SIZE):
        stop = min(start + FEATURE_BLOCK_SIZE, len(cols))
        output[:, start:stop] = subset_dense(matrix, rows, cols[start:stop]).astype(dtype, copy=False)
    return output


def nonzero_detection_fraction(matrix, rows: np.ndarray) -> np.ndarray:
    if len(rows) == 0:
        return np.zeros(matrix.shape[1], dtype=float)
    sub = matrix[rows]
    if issparse(sub):
        return np.asarray((sub != 0).mean(axis=0)).ravel().astype(float)
    dense = np.asarray(sub, dtype=float)
    valid = np.isfinite(dense)
    detected = valid & (dense != 0)
    denominator = valid.sum(axis=0)
    return np.divide(
        detected.sum(axis=0),
        denominator,
        out=np.zeros(dense.shape[1], dtype=float),
        where=denominator > 0,
    )


def finite_coverage_fraction(matrix, rows: np.ndarray) -> np.ndarray:
    if len(rows) == 0:
        return np.zeros(matrix.shape[1], dtype=float)
    sub = matrix[rows]
    if issparse(sub):
        dense = sub.toarray().astype(float, copy=False)
    else:
        dense = np.asarray(sub, dtype=float)
    return np.isfinite(dense).mean(axis=0).astype(float)


def safe_spearman(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3:
        return np.nan, np.nan
    xv = x[valid]
    yv = y[valid]
    if np.allclose(xv, xv[0]) or np.allclose(yv, yv[0]):
        return np.nan, np.nan
    rho, pvalue = spearmanr(xv, yv)
    return float(rho), float(pvalue)


def stable_seed(*parts: object) -> int:
    joined = "|".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(joined).hexdigest()[:16]
    return int(digest, 16) % np.iinfo(np.int32).max


def resolve_kd_targets(all_targets: Sequence[str]) -> List[str]:
    explicit_target = os.environ.get("KD_DISTANCE_TARGET", "").strip()
    if explicit_target:
        if explicit_target not in all_targets:
            raise ValueError(f"Unknown KD_DISTANCE_TARGET: {explicit_target}")
        return [explicit_target]
    explicit_targets = os.environ.get("KD_DISTANCE_TARGETS", "").strip()
    if explicit_targets:
        parsed = [token.strip() for token in explicit_targets.split(",") if token.strip()]
        unknown = [token for token in parsed if token not in all_targets]
        if unknown:
            raise ValueError(f"Unknown KD_DISTANCE_TARGETS values: {unknown}")
        return parsed
    raw_index = os.environ.get("KD_DISTANCE_TARGET_INDEX", "").strip()
    if not raw_index:
        raw_index = os.environ.get("SLURM_ARRAY_TASK_ID", "").strip()
    if raw_index:
        idx = int(raw_index)
        if 1 <= idx <= len(all_targets):
            return [all_targets[idx - 1]]
        if 0 <= idx < len(all_targets):
            return [all_targets[idx]]
        raise ValueError(f"KD target index out of range: {idx}")
    return list(all_targets)


def current_mode() -> str:
    mode = os.environ.get("KD_DISTANCE_MODE", "full").strip().lower()
    if mode not in {"full", "compute", "aggregate"}:
        raise ValueError("KD_DISTANCE_MODE must be one of: full, compute, aggregate")
    return mode


def pairwise_distance_matrix(coords: np.ndarray) -> np.ndarray:
    coords32 = np.asarray(coords, dtype=np.float32)
    delta = coords32[:, None, :] - coords32[None, :, :]
    squared = np.sum(delta * delta, axis=2, dtype=np.float32)
    distances = np.sqrt(squared).astype(np.float32, copy=False)
    np.fill_diagonal(distances, 0.0)
    return distances


def nearest_reference_distances_from_matrix(
    distance_matrix: np.ndarray,
    reference_indices: np.ndarray,
) -> np.ndarray:
    if len(reference_indices) < 1:
        raise ValueError("No reference cells")
    if len(reference_indices) == 1:
        return distance_matrix[:, reference_indices[0]].astype(np.float32, copy=False)
    return distance_matrix[:, reference_indices].min(axis=1).astype(np.float32, copy=False)


def calculate_bin_profile(
    distances: np.ndarray,
    values: np.ndarray,
) -> Tuple[List[float], List[int]]:
    means: List[float] = []
    counts: List[int] = []
    for bin_index, (lo, hi) in enumerate(DISTANCE_BINS):
        if bin_index == len(DISTANCE_BINS) - 1:
            keep = (distances >= lo) & (distances <= hi)
        else:
            keep = (distances >= lo) & (distances < hi)
        valid = keep & np.isfinite(values)
        counts.append(int(valid.sum()))
        means.append(float(np.mean(values[valid])) if valid.any() else np.nan)
    return means, counts


def rank_vector(values: np.ndarray) -> np.ndarray:
    try:
        return np.asarray(rankdata(values, method="average"), dtype=np.float64)
    except TypeError:
        return np.asarray(rankdata(values), dtype=np.float64)


def rank_columns(values: np.ndarray) -> np.ndarray:
    try:
        return np.asarray(rankdata(values, axis=0, method="average"), dtype=np.float64)
    except TypeError:
        return pd.DataFrame(values).rank(axis=0, method="average").to_numpy(dtype=np.float64, copy=False)


def centered_rank_vector(values: np.ndarray) -> Tuple[np.ndarray, float]:
    ranks = rank_vector(values)
    ranks -= ranks.mean()
    scale = float(np.sqrt(np.dot(ranks, ranks)))
    return ranks, scale


def spearman_rho_block_no_nan(
    ranked_x: np.ndarray,
    scale_x: float,
    values: np.ndarray,
) -> np.ndarray:
    if values.shape[0] < 3 or not np.isfinite(scale_x) or scale_x == 0.0:
        return np.full(values.shape[1], np.nan, dtype=float)
    ranked_y = rank_columns(values)
    ranked_y -= ranked_y.mean(axis=0, keepdims=True)
    scale_y = np.sqrt((ranked_y * ranked_y).sum(axis=0))
    numerator = ranked_x @ ranked_y
    return np.divide(
        numerator,
        scale_x * scale_y,
        out=np.full(values.shape[1], np.nan, dtype=float),
        where=scale_y > 0,
    )


def spearman_rho_block_with_nan(
    x: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    output = np.full(values.shape[1], np.nan, dtype=float)
    finite = np.isfinite(values)
    for column in range(values.shape[1]):
        keep = finite[:, column]
        if keep.sum() < 3:
            continue
        xv = x[keep]
        yv = values[keep, column]
        if np.allclose(xv, xv[0]) or np.allclose(yv, yv[0]):
            continue
        ranked_x, scale_x = centered_rank_vector(xv)
        if not np.isfinite(scale_x) or scale_x == 0.0:
            continue
        ranked_y, scale_y = centered_rank_vector(yv)
        if not np.isfinite(scale_y) or scale_y == 0.0:
            continue
        output[column] = float(np.dot(ranked_x, ranked_y) / (scale_x * scale_y))
    return output


def write_dataframe_atomic(df: pd.DataFrame, path: Path) -> None:
    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, path)


def bh_fdr_with_nan(pvalues: pd.Series) -> np.ndarray:
    values = pd.to_numeric(pvalues, errors="coerce").to_numpy(dtype=float)
    output = np.full(len(values), np.nan, dtype=float)
    valid = np.isfinite(values)
    if valid.any():
        output[valid] = multipletests(values[valid], method="fdr_bh")[1]
    return output


def build_profile_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[dict] = []
    for _, row in df.iterrows():
        for label, mid in zip(BIN_LABELS, BIN_MIDPOINTS):
            rows.append({
                "KD_target": row["KD_target"],
                "modality": row["modality"],
                "feature": row["feature"],
                "feature_index": row["feature_index"],
                "bin_label": label,
                "bin_mid": mid,
                "mean_feature_value": row[f"mean_{label}um"],
                "n_cells": row[f"n_{label}um"],
                "rho": row["rho"],
                "p_positive_one_sided": row["p_positive_one_sided"],
                "p_negative_one_sided": row["p_negative_one_sided"],
                "fdr_positive_one_sided": row["fdr_positive_one_sided"],
                "fdr_negative_one_sided": row["fdr_negative_one_sided"],
            })
    return pd.DataFrame(rows)


def run_analysis(config: AnalysisConfig) -> None:
    for path in [H5MU_PATH, TE_RNA_PATH, TE_ATAC_PATH]:
        require_file(path)

    mode = current_mode()
    selected_targets = resolve_kd_targets(config.kd_targets)

    output_dir_name = os.environ.get("KD_DISTANCE_OUTPUT_DIR_NAME", config.default_output_dir_name)
    output_dir = DATA_DIR / output_dir_name
    result_dir = output_dir / "dataset_level_results"
    plot_dir = output_dir / "significant_feature_plots"
    profile_dir = output_dir / "dataset_level_bin_profiles"
    null_dir = output_dir / "full_null_rho_arrays"
    for directory in [output_dir, result_dir, plot_dir, profile_dir, null_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    detection_threshold = float(os.environ.get("KD_DISTANCE_MIN_DET", str(config.detection_threshold)))
    n_perm = int(os.environ.get("KD_DISTANCE_N_PERM", str(config.n_perm)))
    null_chunk_size = int(os.environ.get("KD_DISTANCE_NULL_CHUNK_SIZE", "1024"))
    save_full_nulls = os.environ.get("KD_DISTANCE_SAVE_FULL_NULLS", "1") != "0"
    compress_nulls = os.environ.get("KD_DISTANCE_COMPRESS_NULLS", "0") != "0"

    print(f"Run mode: {mode}")
    print(f"Modality: {config.modality}")
    print(f"Selected KD targets ({len(selected_targets)}): {selected_targets}")
    print(f"Detection / coverage threshold: {detection_threshold:.3f}")
    print(f"Permutations per KD test: {n_perm:,}")
    print(f"Save full nulls: {save_full_nulls}")
    print(f"Compress null chunks: {compress_nulls}")
    print(f"Null chunk size: {null_chunk_size:,}")
    print("Analysis design: dataset-level pooled, one-sided positive/negative tests, BH across features within each KD.")
    print("Permutation scheme: global spatial shuffle of whole cells across all valid positions, preserving each cell's full profile.")

    raw_result_pattern = "*_GLOBAL_feature_results_RAW.csv"

    def aggregate_only() -> None:
        raw_files = sorted(
            path for path in result_dir.glob(raw_result_pattern)
            if not path.name.startswith("ALL_KD_")
        )
        if not raw_files:
            raise FileNotFoundError("No per-target dataset-level raw result files were found for aggregation.")

        results = pd.concat([pd.read_csv(path) for path in raw_files], ignore_index=True)
        if results.empty:
            print("\nNo dataset-level KD x feature tests completed.")
            return

        results["fdr_positive_one_sided"] = np.nan
        results["fdr_negative_one_sided"] = np.nan
        for _, idx in results.groupby(["KD_target", "modality"]).groups.items():
            idx = list(idx)
            results.loc[idx, "fdr_positive_one_sided"] = bh_fdr_with_nan(results.loc[idx, "p_positive_one_sided"])
            results.loc[idx, "fdr_negative_one_sided"] = bh_fdr_with_nan(results.loc[idx, "p_negative_one_sided"])

        results["passes_fdr_positive"] = results["fdr_positive_one_sided"] < FDR_ALPHA
        results["passes_fdr_negative"] = results["fdr_negative_one_sided"] < FDR_ALPHA
        results["significant_any_direction"] = results["passes_fdr_positive"] | results["passes_fdr_negative"]
        results["significant_direction"] = np.select(
            [
                results["passes_fdr_positive"] & ~results["passes_fdr_negative"],
                results["passes_fdr_negative"] & ~results["passes_fdr_positive"],
                results["passes_fdr_positive"] & results["passes_fdr_negative"],
            ],
            ["positive", "negative", "both"],
            default="none",
        )

        write_dataframe_atomic(results, result_dir / "ALL_KD_dataset_level_feature_results.csv")
        write_dataframe_atomic(
            results.loc[results["significant_any_direction"]].copy(),
            result_dir / "ALL_FDR_passing_dataset_level_results.csv",
        )

        all_profiles = build_profile_rows(results)
        write_dataframe_atomic(all_profiles, profile_dir / "ALL_dataset_level_bin_profiles.csv")
        passing = results.loc[results["significant_any_direction"]].copy()
        passing_profiles = build_profile_rows(passing) if not passing.empty else pd.DataFrame()
        write_dataframe_atomic(passing_profiles, profile_dir / "FDR_passing_dataset_level_bin_profiles.csv")

        plot_manifest_rows: List[dict] = []
        for row in passing.itertuples(index=False):
            means = np.asarray([getattr(row, f"mean_{label}um") for label in BIN_LABELS], dtype=float)
            counts = np.asarray([getattr(row, f"n_{label}um") for label in BIN_LABELS], dtype=float)
            fig, ax = plt.subplots(figsize=(7.2, 5.2))
            color = "#b22222" if row.significant_direction == "positive" else "#1f4e79"
            ax.plot(BIN_MIDPOINTS, means, marker="o", linewidth=1.6, markersize=4.0, color=color)
            ax.set_xlabel("Distance to nearest KD cell (um)")
            ax.set_ylabel("Mean feature value")
            ax.set_xticks(BIN_MIDPOINTS)
            ax.set_xticklabels(BIN_LABELS, rotation=35, ha="right")
            ax.set_title(
                f"{row.KD_target} KD | {row.modality} | {row.feature}\n"
                f"rho={row.rho:.3f}; FDR+={row.fdr_positive_one_sided:.3g}; "
                f"FDR-={row.fdr_negative_one_sided:.3g}; direction={row.significant_direction}"
            )
            for x, y, n in zip(BIN_MIDPOINTS, means, counts):
                if np.isfinite(y):
                    ax.text(x, y, f"n={int(n)}", fontsize=7, ha="center", va="bottom")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            fig.tight_layout()

            base = safe_filename(f"{row.KD_target}_{row.modality}_{row.feature}_GLOBAL")
            png_path = plot_dir / f"{base}.png"
            pdf_path = plot_dir / f"{base}.pdf"
            fig.savefig(png_path, dpi=600, bbox_inches="tight")
            fig.savefig(pdf_path, bbox_inches="tight")
            plt.close(fig)

            plot_manifest_rows.append({
                "KD_target": row.KD_target,
                "modality": row.modality,
                "feature": row.feature,
                "feature_index": row.feature_index,
                "significant_direction": row.significant_direction,
                "png": str(png_path),
                "pdf": str(pdf_path),
            })

        write_dataframe_atomic(pd.DataFrame(plot_manifest_rows), result_dir / "dataset_level_plot_manifest.csv")
        print("\nDone.")
        print(f"Output directory: {output_dir}")
        print("Analysis: dataset-level pooled.")
        print("One-sided positive/negative empirical p-values are converted to BH-FDR across features within each KD.")

    if mode == "aggregate":
        aggregate_only()
        return

    print("Loading MuData...")
    mdata = md.read_h5mu(H5MU_PATH)
    for required_modality in ["rna", "atac", "chromvar"]:
        if required_modality not in mdata.mod:
            raise KeyError(f"Missing MuData modality: {required_modality}")

    rna = mdata.mod["rna"]
    atac = mdata.mod["atac"]
    chromvar = mdata.mod["chromvar"]

    if not np.array_equal(to_str_array(rna.obs_names), to_str_array(atac.obs_names)):
        raise ValueError("RNA and ATAC rows are not aligned")
    if not np.array_equal(to_str_array(rna.obs_names), to_str_array(chromvar.obs_names)):
        raise ValueError("RNA and chromVAR rows are not aligned")

    main_obs = rna.obs.copy()
    for column in [GUIDE_COL, EB_COL, LANE_COL, "gex_barcode", "raw_cell_barcode"]:
        if column not in main_obs.columns:
            raise KeyError(f"RNA obs missing required column: {column}")
    if SPATIAL_KEY not in rna.obsm:
        raise KeyError(f"RNA obsm missing required key: {SPATIAL_KEY}")

    coords = np.asarray(rna.obsm[SPATIAL_KEY], dtype=float)[:, :2]
    if not np.isfinite(coords).all():
        raise ValueError("Spatial coordinates contain non-finite values")

    guide_strings = to_str_array(main_obs[GUIDE_COL])
    eb_labels = to_str_array(main_obs[EB_COL])
    valid_eb_mask = eb_labels != "Unassigned"
    valid_ebs = sorted(pd.unique(eb_labels[valid_eb_mask]))

    print("Loading TE RNA...")
    te_rna = ad.read_h5ad(TE_RNA_PATH)
    print("Loading TE ATAC...")
    te_atac = ad.read_h5ad(TE_ATAC_PATH)

    te_rna_map = map_te_rna_cells(main_obs, te_rna)
    te_atac_map = map_te_atac_cells(main_obs, te_atac)

    if config.modality == "ATAC":
        info = {
            "value_matrix": get_matrix(atac, ATAC_VALUE_LAYER),
            "detection_matrix": get_matrix(atac, ATAC_DETECTION_LAYER),
            "feature_names": make_unique_names(to_str_array(atac.var_names)),
            "main_to_external": np.arange(rna.n_obs, dtype=int),
        }
    elif config.modality == "TE_RNA":
        info = {
            "value_matrix": get_matrix(te_rna, TE_RNA_VALUE_LAYER),
            "detection_matrix": get_matrix(te_rna, TE_RNA_DETECTION_LAYER),
            "feature_names": make_unique_names(to_str_array(te_rna.var_names)),
            "main_to_external": te_rna_map,
        }
    elif config.modality == "TE_ATAC":
        te_atac_values = get_matrix(te_atac, TE_ATAC_VALUE_LAYER)
        info = {
            "value_matrix": te_atac_values,
            "detection_matrix": te_atac_values,
            "feature_names": make_unique_names(to_str_array(te_atac.var_names)),
            "main_to_external": te_atac_map,
        }
    elif config.modality == "chromVAR":
        chromvar_names = (
            make_unique_names(to_str_array(chromvar.var["motif_name"]))
            if "motif_name" in chromvar.var.columns
            else make_unique_names(to_str_array(chromvar.var_names))
        )
        chromvar_values = get_matrix(chromvar, CHROMVAR_VALUE_LAYER)
        info = {
            "value_matrix": chromvar_values,
            "detection_matrix": chromvar_values,
            "feature_names": chromvar_names,
            "main_to_external": np.arange(rna.n_obs, dtype=int),
        }
    else:
        raise ValueError(f"Unsupported modality: {config.modality}")

    valid_rows = np.flatnonzero(valid_eb_mask)
    valid_cell_ids = to_str_array(rna.obs_names)[valid_rows]
    valid_guide_strings = guide_strings[valid_rows]
    valid_external_rows = info["main_to_external"][valid_rows]

    pd.DataFrame([
        {"object": "TE_RNA", "n_cells": len(te_rna_map), "n_matched": int((te_rna_map >= 0).sum())},
        {"object": "TE_ATAC", "n_cells": len(te_atac_map), "n_matched": int((te_atac_map >= 0).sum())},
    ]).to_csv(output_dir / "cell_mapping_summary.csv", index=False)

    print(f"Cells: {rna.n_obs:,}")
    print(f"Valid EBs: {pd.Series(eb_labels[valid_eb_mask]).nunique():,}")
    print(f"Features in {config.modality}: {len(info['feature_names']):,}")

    print("Precomputing valid-EB cell indices and pairwise distance matrices...")
    eb_valid_local_rows_map: Dict[str, np.ndarray] = {}
    eb_distance_matrices: Dict[str, np.ndarray] = {}
    for eb in valid_ebs:
        local_rows = np.where(eb_labels[valid_rows] == eb)[0]
        eb_valid_local_rows_map[str(eb)] = local_rows
        eb_distance_matrices[str(eb)] = pairwise_distance_matrix(coords[valid_rows[local_rows]])
    print(f"Precomputed pairwise distance matrices for {len(valid_ebs):,} EBs")

    for target in selected_targets:
        print("\n" + "=" * 100)
        print(f"KD target: {target}")
        print("=" * 100)

        target_mask_global = target_guide_mask(guide_strings, target)
        target_mask_valid = target_mask_global[valid_rows]

        observed_receiver_local_parts: List[np.ndarray] = []
        observed_distance_parts: List[np.ndarray] = []
        observed_eligible_ebs: List[str] = []

        for eb_key, eb_local_rows in eb_valid_local_rows_map.items():
            kd_flag_eb = target_mask_valid[eb_local_rows]
            n_kd = int(kd_flag_eb.sum())
            n_non_kd = int((~kd_flag_eb).sum())
            if n_kd < MIN_KD_CELLS or n_non_kd < MIN_RECEIVER_CELLS:
                continue
            observed_distances = nearest_reference_distances_from_matrix(
                eb_distance_matrices[eb_key],
                np.flatnonzero(kd_flag_eb),
            )
            observed_receiver_mask = (~kd_flag_eb) & (observed_distances <= MAX_DISTANCE)
            if int(observed_receiver_mask.sum()) < MIN_RECEIVER_CELLS:
                continue
            observed_eligible_ebs.append(eb_key)
            observed_receiver_local_parts.append(eb_local_rows[observed_receiver_mask])
            observed_distance_parts.append(observed_distances[observed_receiver_mask].astype(np.float64, copy=False))

        if not observed_receiver_local_parts:
            print(f"No eligible observed receiver cells for {target}; skipping")
            continue

        observed_receiver_local = np.concatenate(observed_receiver_local_parts)
        observed_distances = np.concatenate(observed_distance_parts)
        observed_external_rows = valid_external_rows[observed_receiver_local]

        if config.modality == "chromVAR":
            detection = finite_coverage_fraction(info["value_matrix"], observed_external_rows)
        else:
            detection = nonzero_detection_fraction(info["detection_matrix"], observed_external_rows)
        tested_feature_indices = np.where(detection >= detection_threshold)[0]
        if len(tested_feature_indices) == 0:
            print(f"No features passed the detection / coverage threshold for {target}; skipping")
            continue

        dense_values_all = materialize_dense_matrix(
            info["value_matrix"],
            valid_external_rows,
            tested_feature_indices,
            dtype=np.float32,
        )
        dense_feature_names = info["feature_names"][tested_feature_indices]
        dense_detection = detection[tested_feature_indices]
        observed_dense = dense_values_all[observed_receiver_local]

        ranked_x_obs, scale_x_obs = centered_rank_vector(observed_distances)
        observed_rows: List[dict] = []
        for start in range(0, observed_dense.shape[1], FEATURE_BLOCK_SIZE):
            stop = min(start + FEATURE_BLOCK_SIZE, observed_dense.shape[1])
            block = observed_dense[:, start:stop]
            rho_block = (
                spearman_rho_block_with_nan(observed_distances, block)
                if config.modality == "chromVAR"
                else spearman_rho_block_no_nan(ranked_x_obs, scale_x_obs, block)
            )
            for local_index in range(block.shape[1]):
                dense_position = start + local_index
                feature_index = int(tested_feature_indices[dense_position])
                values = block[:, local_index].astype(float, copy=False)
                rho = float(rho_block[local_index]) if np.isfinite(rho_block[local_index]) else np.nan
                _, analytical_p = safe_spearman(observed_distances, values)
                bin_means, bin_counts = calculate_bin_profile(observed_distances, values)
                observed_rows.append({
                    "KD_target": target,
                    "modality": config.modality,
                    "feature": str(dense_feature_names[dense_position]),
                    "feature_index": feature_index,
                    "detection_fraction": float(dense_detection[dense_position]),
                    "detection_filter_metric": (
                        "finite_score_coverage_fraction" if config.modality == "chromVAR"
                        else "nonzero_detection_fraction"
                    ),
                    "rho": rho,
                    "analytical_pvalue": analytical_p,
                    "n_valid_positions": len(valid_rows),
                    "n_eligible_EBs_observed": len(observed_eligible_ebs),
                    "n_observed_receivers_within_300um_total": len(observed_receiver_local),
                    **{f"mean_{label}um": value for label, value in zip(BIN_LABELS, bin_means)},
                    **{f"n_{label}um": value for label, value in zip(BIN_LABELS, bin_counts)},
                })

        observed_df = pd.DataFrame(observed_rows)
        observed_df = observed_df.loc[np.isfinite(observed_df["rho"].to_numpy(dtype=float))].reset_index(drop=True)
        if observed_df.empty:
            print(f"No finite observed statistics for {target}; skipping")
            continue

        dense_keep = np.isfinite(pd.DataFrame(observed_rows)["rho"].to_numpy(dtype=float))
        dense_values_all = dense_values_all[:, dense_keep]
        observed_rhos = observed_df["rho"].to_numpy(dtype=float)

        null_chunk_dir = null_dir / safe_filename(f"{target}_{config.modality}_GLOBAL_null_rhos_chunks")
        if null_chunk_dir.exists():
            shutil.rmtree(null_chunk_dir)
        null_chunk_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = null_chunk_dir / "manifest.csv"
        manifest_rows: List[dict] = []

        positive_tail_counts = np.zeros(len(observed_df), dtype=np.int64)
        negative_tail_counts = np.zeros(len(observed_df), dtype=np.int64)
        valid_perm_counts = np.zeros(len(observed_df), dtype=np.int64)

        print(
            f"Observed pooled test for {target}: "
            f"{len(observed_eligible_ebs)} eligible EBs, {len(observed_receiver_local):,} pooled receivers, "
            f"{len(observed_df):,} tested features"
        )

        for chunk_start in range(0, n_perm, null_chunk_size):
            chunk_stop = min(chunk_start + null_chunk_size, n_perm)
            chunk_width = chunk_stop - chunk_start
            chunk_null_rhos = (
                np.full((len(observed_df), chunk_width), np.nan, dtype=np.float32)
                if save_full_nulls else None
            )

            for permutation_index in range(chunk_start, chunk_stop):
                perm_rng = np.random.default_rng(
                    stable_seed(SEED, config.modality, permutation_index, "global_spatial_shuffle")
                )
                perm_order = perm_rng.permutation(len(valid_rows))
                perm_kd_valid = target_mask_valid[perm_order]

                perm_receiver_local_parts: List[np.ndarray] = []
                perm_distance_parts: List[np.ndarray] = []
                for eb_key, eb_local_rows in eb_valid_local_rows_map.items():
                    perm_kd_eb = perm_kd_valid[eb_local_rows]
                    n_kd = int(perm_kd_eb.sum())
                    n_non_kd = int((~perm_kd_eb).sum())
                    if n_kd < MIN_KD_CELLS or n_non_kd < MIN_RECEIVER_CELLS:
                        continue
                    perm_distances = nearest_reference_distances_from_matrix(
                        eb_distance_matrices[eb_key],
                        np.flatnonzero(perm_kd_eb),
                    )
                    perm_receiver_mask = (~perm_kd_eb) & (perm_distances <= MAX_DISTANCE)
                    if int(perm_receiver_mask.sum()) < MIN_RECEIVER_CELLS:
                        continue
                    perm_receiver_local_parts.append(eb_local_rows[perm_receiver_mask])
                    perm_distance_parts.append(perm_distances[perm_receiver_mask].astype(np.float64, copy=False))

                if not perm_receiver_local_parts:
                    continue

                perm_receiver_local = np.concatenate(perm_receiver_local_parts)
                perm_distances_all = np.concatenate(perm_distance_parts)
                if len(perm_receiver_local) < MIN_RECEIVER_CELLS:
                    continue

                ranked_x, scale_x = centered_rank_vector(perm_distances_all)
                if not np.isfinite(scale_x) or scale_x == 0.0:
                    continue

                perm_profile_rows = perm_order[perm_receiver_local]
                for start in range(0, dense_values_all.shape[1], FEATURE_BLOCK_SIZE):
                    stop = min(start + FEATURE_BLOCK_SIZE, dense_values_all.shape[1])
                    block = dense_values_all[perm_profile_rows, start:stop]
                    rho_block = (
                        spearman_rho_block_with_nan(perm_distances_all, block)
                        if config.modality == "chromVAR"
                        else spearman_rho_block_no_nan(ranked_x, scale_x, block)
                    )
                    if chunk_null_rhos is not None:
                        chunk_null_rhos[start:stop, permutation_index - chunk_start] = rho_block.astype(np.float32, copy=False)
                    finite = np.isfinite(rho_block)
                    valid_perm_counts[start:stop] += finite.astype(np.int64)
                    positive_tail_counts[start:stop] += (finite & (rho_block >= observed_rhos[start:stop])).astype(np.int64)
                    negative_tail_counts[start:stop] += (finite & (rho_block <= observed_rhos[start:stop])).astype(np.int64)

                if (permutation_index + 1) % 100 == 0:
                    print(f"  permutation {permutation_index + 1:,}/{n_perm:,}")

            if chunk_null_rhos is not None:
                if compress_nulls:
                    chunk_path = null_chunk_dir / (
                        f"{safe_filename(target)}_{safe_filename(config.modality)}"
                        f"_GLOBAL_null_rhos_perm_{chunk_start:06d}_{chunk_stop - 1:06d}.npz"
                    )
                    np.savez_compressed(chunk_path, null_rhos=chunk_null_rhos)
                else:
                    chunk_path = null_chunk_dir / (
                        f"{safe_filename(target)}_{safe_filename(config.modality)}"
                        f"_GLOBAL_null_rhos_perm_{chunk_start:06d}_{chunk_stop - 1:06d}.npy"
                    )
                    np.save(chunk_path, chunk_null_rhos, allow_pickle=False)
                manifest_rows.append({
                    "KD_target": target,
                    "modality": config.modality,
                    "perm_start_inclusive": chunk_start,
                    "perm_stop_exclusive": chunk_stop,
                    "n_features": int(chunk_null_rhos.shape[0]),
                    "n_permutations": int(chunk_null_rhos.shape[1]),
                    "dtype": "float32",
                    "path": str(chunk_path),
                })
                del chunk_null_rhos
                gc.collect()

        pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)

        valid = valid_perm_counts > 0
        observed_df["p_positive_one_sided"] = np.nan
        observed_df["p_negative_one_sided"] = np.nan
        observed_df.loc[valid, "p_positive_one_sided"] = (
            1.0 + positive_tail_counts[valid]
        ) / (1.0 + valid_perm_counts[valid])
        observed_df.loc[valid, "p_negative_one_sided"] = (
            1.0 + negative_tail_counts[valid]
        ) / (1.0 + valid_perm_counts[valid])
        observed_df["n_permutations_positive_tail"] = positive_tail_counts
        observed_df["n_permutations_negative_tail"] = negative_tail_counts
        observed_df["n_valid_permutations"] = valid_perm_counts
        observed_df["n_permutations_requested"] = n_perm
        observed_df["full_null_rho_array_path"] = str(manifest_path) if save_full_nulls else ""
        observed_df["full_null_rho_chunk_dir"] = str(null_chunk_dir) if save_full_nulls else ""
        observed_df["full_null_rho_array_shape"] = f"{len(observed_df)}x{n_perm}"
        observed_df["full_null_rho_dtype"] = "float32" if save_full_nulls else ""
        observed_df["null_chunk_size_permutations"] = null_chunk_size

        write_dataframe_atomic(
            observed_df,
            result_dir / f"{safe_filename(target)}_GLOBAL_feature_results_RAW.csv",
        )
        gc.collect()

    if mode == "full" and len(selected_targets) == len(config.kd_targets):
        aggregate_only()
    elif mode == "full":
        print("Skipping aggregate stage because a KD subset was selected in full mode.")


# KD_distance_chromVAR_TOP5_10K_TWO_SIDED_SAVE_FULL_NULLS
#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def load_common():
    common_path = Path(__file__).with_name("kd_distance_ebwise_common_four_modalities.py")
    spec = importlib.util.spec_from_file_location("kd_distance_ebwise_common", common_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load shared module: {common_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    common = load_common()
    common.run_analysis(
        common.AnalysisConfig(
            modality="chromVAR",
            default_output_dir_name="KD_distance_chromVAR_TOP10_LIGAND_KDs_DATASET_LEVEL_GLOBAL_SPATIAL_SHUFFLE_65K_ONE_SIDED_5pct_FULL_NULLS",
            detection_threshold=0.05,
            n_perm=65000,
        )
    )


# KD_distance_TE_ATAC_TOP5_10K_TWO_SIDED_SAVE_FULL_NULLS
#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def load_common():
    common_path = Path(__file__).with_name("kd_distance_ebwise_common_four_modalities.py")
    spec = importlib.util.spec_from_file_location("kd_distance_ebwise_common", common_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load shared module: {common_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    common = load_common()
    common.run_analysis(
        common.AnalysisConfig(
            modality="TE_ATAC",
            default_output_dir_name="KD_distance_TE_ATAC_TOP10_LIGAND_KDs_DATASET_LEVEL_GLOBAL_SPATIAL_SHUFFLE_120K_ONE_SIDED_10pct_FULL_NULLS",
            detection_threshold=0.10,
            n_perm=120000,
        )
    )


# KD_distance_ATAC_TOP5_10K_TWO_SIDED_SAVE_FULL_NULLS (1)
#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def load_common():
    common_path = Path(__file__).with_name("kd_distance_ebwise_common_four_modalities.py")
    spec = importlib.util.spec_from_file_location("kd_distance_ebwise_common", common_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load shared module: {common_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    common = load_common()
    common.run_analysis(
        common.AnalysisConfig(
            modality="ATAC",
            default_output_dir_name="KD_distance_ATAC_TOP10_LIGAND_KDs_DATASET_LEVEL_GLOBAL_SPATIAL_SHUFFLE_60K_ONE_SIDED_15pct_FULL_NULLS",
            detection_threshold=0.15,
            n_perm=60000,
        )
    )


# KD_distance_TE_RNA_TOP5_10K_TWO_SIDED_SAVE_FULL_NULLS
#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def load_common():
    common_path = Path(__file__).with_name("kd_distance_ebwise_common_four_modalities.py")
    spec = importlib.util.spec_from_file_location("kd_distance_ebwise_common", common_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load shared module: {common_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    common = load_common()
    common.run_analysis(
        common.AnalysisConfig(
            modality="TE_RNA",
            default_output_dir_name="KD_distance_TE_RNA_TOP10_LIGAND_KDs_DATASET_LEVEL_GLOBAL_SPATIAL_SHUFFLE_15K_ONE_SIDED_10pct_FULL_NULLS",
            detection_threshold=0.10,
            n_perm=15000,
        )
    )


# KD_distance_chromVAR_REMAINING5_LIGAND_KDs
#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def load_common():
    common_path = Path(__file__).with_name("kd_distance_ebwise_common_four_modalities.py")
    spec = importlib.util.spec_from_file_location("kd_distance_ebwise_common", common_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load shared module: {common_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    common = load_common()
    common.run_analysis(
        common.AnalysisConfig(
            modality="chromVAR",
            default_output_dir_name="KD_distance_chromVAR_REMAINING5_LIGAND_KDs_DATASET_LEVEL_GLOBAL_SPATIAL_SHUFFLE_65K_ONE_SIDED_5pct_FULL_NULLS",
            detection_threshold=0.05,
            n_perm=65000,
            kd_targets=("SEMA6D", "NECTIN2", "NECTIN3", "ALB", "B2M"),
        )
    )


# KD_distance_TE_ATAC_REMAINING5_LIGAND_KDs
#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def load_common():
    common_path = Path(__file__).with_name("kd_distance_ebwise_common_four_modalities.py")
    spec = importlib.util.spec_from_file_location("kd_distance_ebwise_common", common_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load shared module: {common_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    common = load_common()
    common.run_analysis(
        common.AnalysisConfig(
            modality="TE_ATAC",
            default_output_dir_name="KD_distance_TE_ATAC_REMAINING5_LIGAND_KDs_DATASET_LEVEL_GLOBAL_SPATIAL_SHUFFLE_120K_ONE_SIDED_10pct_FULL_NULLS",
            detection_threshold=0.10,
            n_perm=120000,
            kd_targets=("SEMA6D", "NECTIN2", "NECTIN3", "ALB", "B2M"),
        )
    )


# KD_distance_ATAC_REMAINING5_LIGAND_KDs
#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def load_common():
    common_path = Path(__file__).with_name("kd_distance_ebwise_common_four_modalities.py")
    spec = importlib.util.spec_from_file_location("kd_distance_ebwise_common", common_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load shared module: {common_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    common = load_common()
    common.run_analysis(
        common.AnalysisConfig(
            modality="ATAC",
            default_output_dir_name="KD_distance_ATAC_REMAINING5_LIGAND_KDs_DATASET_LEVEL_GLOBAL_SPATIAL_SHUFFLE_60K_ONE_SIDED_15pct_FULL_NULLS",
            detection_threshold=0.15,
            n_perm=60000,
            kd_targets=("SEMA6D", "NECTIN2", "NECTIN3", "ALB", "B2M"),
        )
    )


# KD_distance_TE_RNA_REMAINING5_LIGAND_KDs
#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def load_common():
    common_path = Path(__file__).with_name("kd_distance_ebwise_common_four_modalities.py")
    spec = importlib.util.spec_from_file_location("kd_distance_ebwise_common", common_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load shared module: {common_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    common = load_common()
    common.run_analysis(
        common.AnalysisConfig(
            modality="TE_RNA",
            default_output_dir_name="KD_distance_TE_RNA_REMAINING5_LIGAND_KDs_DATASET_LEVEL_GLOBAL_SPATIAL_SHUFFLE_15K_ONE_SIDED_10pct_FULL_NULLS",
            detection_threshold=0.10,
            n_perm=15000,
            kd_targets=("SEMA6D", "NECTIN2", "NECTIN3", "ALB", "B2M"),
        )
    )
