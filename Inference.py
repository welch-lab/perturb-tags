#!/usr/bin/env python3
from __future__ import annotations
import importlib.util
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import mudata as mu
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator
from scipy.stats import pearsonr, spearmanr
from sklearn.neighbors import NearestNeighbors

# Figure 6e-f: counterfactual guide-pair and distance-resolved analysis
TRAINING_SCRIPT_PATH = Path(
    "/scratch/welchjd_root/welchjd1/javidgmh/15515_Multiome/catatac_work/motif_analysis/joint_multiome_MODALITY_SPECIFIC_SPATIAL_RESIDUAL_TUNE_LOSS_WEIGHTS_10EB.py"
)
TRAINING_OUTDIR = Path(
    "multiome_joint_RNA_ATAC_MODALITY_SPECIFIC_SPATIAL_RESIDUAL_TUNE_LOSS_WEIGHTS_10EB"
)
OUTROOT = TRAINING_OUTDIR / "counterfactual_inference_publication"
DATA_PATH = Path(
    "/scratch/welchjd_root/welchjd1/javidgmh/15515_Multiome/"
    "catatac_work/motif_analysis/"
    "all_lanes_merged_CATATAC_QC_RNA_processed_EB_manual_clustering_cell_types.h5mu"
)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_SEED = 42
RUN_MODULE_1 = True
RUN_MODULE_2 = True
RUN_MODULE_3 = True
RUN_MODULE_4 = True
SELECT_RUNS: list[int] | None = None
CENTRAL_CELL_SCOPE = "all"
SELECTED_EBS: list[str] | None = None
SELECTED_CENTRAL_CELL_TYPES: list[str] | None = None
MAX_CENTRAL_CELLS_MODULE_1 = 500
MAX_CENTRAL_CELLS_MODULE_2 = None
MAX_CENTRAL_CELLS_MODULE_3 = 250
MAX_CENTRAL_CELLS_MODULE_4 = 1000
MODULE_1_GUIDE_SUBSTITUTIONS: list[tuple[str, str]] | None = None
MODULE_3_GUIDE_PAIRS: list[tuple[str, str]] | None = None
SCREEN_GUIDES: list[str] | None = None
N_AUTO_SCREEN_GUIDES = 10
CONTROL_GUIDE_PATTERNS = (
    "negative control",
    "non-targeting",
    "non targeting",
    "nontargeting",
    "ntc",
    "control",
)
COUNTERFACTUAL_EDIT_MODE = "replace_source_with_target"
EDIT_CENTRAL_CELL = False
MIN_SOURCE_POSITIVE_NEIGHBORS = 1
TOP_FEATURES_PER_CELL_SCENARIO = 30
MODULE_3_TOP_FEATURES_PER_PAIR = 100
MATCH_MIN_GUIDE_NEIGHBORS = 1
MATCH_REQUIRE_EXCLUSIVE_EXPOSURE = False
MATCH_WITHIN_EB = True
MATCH_WITHIN_CELL_TYPE = True
MATCH_WITHOUT_REPLACEMENT = True
MATCH_CALIPER = None
MATCH_INCLUDE_OWN_GUIDE_COUNT = True
DISTANCE_BANDS_UM: list[tuple[float, float]] | None = None
AUTO_DISTANCE_BAND_COUNT = 4
AUTO_DISTANCE_BAND_MIN_WIDTH_UM = 10.0
RADIUS_EDIT_DIRECTION = "guide_to_control"
RADIUS_GUIDES: list[str] | None = None
MIN_SOURCE_POSITIVE_NEIGHBORS_PER_BAND = 1
N_EB_BOOTSTRAPS_MODULE_4 = 1000
MODULE_4_BOOTSTRAP_SEED = 20260816
MODULE_4_FDR_THRESHOLD = 0.05
MODULE_4_TOP_FEATURES_PER_GUIDE = 15
PLOT_CMAP_DIVERGING = "RdBu_r"
PLOT_CMAP_SEQUENTIAL = "YlGnBu"
COLOR_MLP = "#7F8C8D"
COLOR_GAT = "#1F4E79"
COLOR_ACCENT = "#C44E52"
COLOR_NEUTRAL = "#D9D9D9"
def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 600,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
def save_plot(fig: plt.Figure, stem: Path) -> None:
    fig.tight_layout()
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
rng = np.random.default_rng(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
def dense32(x) -> np.ndarray:
    if sp.issparse(x):
        return x.toarray().astype(np.float32)
    return np.asarray(x, dtype=np.float32)
def parse_guides(value) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, (list, tuple, np.ndarray)):
        values = value
    else:
        text = str(value).strip()
        if text == "" or text.lower() in {"nan", "none", "unassigned", "no guide", "no_guide"}:
            return []
        values = text.split(";")
    return [str(x).strip() for x in values if str(x).strip()]
def build_guides(obs: pd.DataFrame, source_col: str) -> tuple[np.ndarray, list[str]]:
    rows = [parse_guides(value) for value in obs[source_col].astype(object).to_numpy()]
    names = sorted({guide for row in rows for guide in row})
    lookup = {name: idx for idx, name in enumerate(names)}
    matrix = np.zeros((len(rows), len(names)), dtype=np.float32)
    for row_idx, row in enumerate(rows):
        for guide in row:
            matrix[row_idx, lookup[guide]] = 1.0
    return matrix, names
def valid_eb_label(value: object) -> bool:
    text = str(value).strip()
    return text.startswith("EB_") and text.lower() not in {"eb_", "eb_nan", "eb_none", "eb_unassigned"}
def sample_indices(indices: np.ndarray, maximum: int | None) -> np.ndarray:
    indices = np.asarray(indices, dtype=int)
    if maximum is None or len(indices) <= maximum:
        return indices
    chosen = rng.choice(indices, size=maximum, replace=False)
    return np.sort(chosen.astype(int))
def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_")
def is_control_guide(guide_name: str) -> bool:
    lowered = guide_name.lower()
    return any(pattern in lowered for pattern in CONTROL_GUIDE_PATTERNS)
def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(p_values)
    ranked = p_values[order]
    n = len(ranked)
    adjusted = np.empty(n, dtype=np.float64)
    running = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        value = ranked[i] * n / rank
        running = min(running, value)
        adjusted[i] = running
    out = np.empty(n, dtype=np.float64)
    out[order] = np.clip(adjusted, 0.0, 1.0)
    return out
def infer_distance_bands_for_run(
    global_data: GlobalData,
    run_data: RunArtifacts,
) -> list[tuple[float, float]]:
    if DISTANCE_BANDS_UM is not None:
        return [(float(left), float(right)) for left, right in DISTANCE_BANDS_UM]
    max_distance = 0.0
    for central_cell, neighbors in enumerate(run_data.union_neighbors):
        if len(neighbors) == 0:
            continue
        distances = np.linalg.norm(
            global_data.coords[neighbors, :2] - global_data.coords[central_cell, :2],
            axis=1,
        ).astype(np.float32)
        if len(distances):
            max_distance = max(max_distance, float(distances.max()))
    if max_distance <= 0:
        return [(0.0, 1.0)]
    band_count = max(1, int(AUTO_DISTANCE_BAND_COUNT))
    if max_distance < band_count * AUTO_DISTANCE_BAND_MIN_WIDTH_UM:
        band_count = max(1, int(math.ceil(max_distance / AUTO_DISTANCE_BAND_MIN_WIDTH_UM)))
    edges = np.linspace(0.0, max_distance, band_count + 1, dtype=np.float64)
    bands = [
        (float(edges[i]), float(edges[i + 1]))
        for i in range(len(edges) - 1)
        if edges[i + 1] > edges[i] + 1e-8
    ]
    return bands or [(0.0, max_distance)]
def standardize_matrix(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    mean = matrix.mean(axis=0, keepdims=True)
    std = matrix.std(axis=0, keepdims=True, ddof=0)
    std[std == 0] = 1.0
    return (matrix - mean) / std
def top_feature_table(
    feature_names: np.ndarray,
    effect_matrix: np.ndarray,
    top_n: int,
    feature_label: str,
    extra: dict[str, object],
) -> pd.DataFrame:
    mean_effect = effect_matrix.mean(axis=0)
    median_effect = np.median(effect_matrix, axis=0)
    abs_mean = np.abs(mean_effect)
    order = np.argsort(abs_mean)[::-1][: min(top_n, len(feature_names))]
    rows = []
    for idx in order:
        row = dict(extra)
        row[feature_label] = str(feature_names[idx])
        row["mean_log1p_effect"] = float(mean_effect[idx])
        row["median_log1p_effect"] = float(median_effect[idx])
        row["absolute_mean_effect"] = float(abs_mean[idx])
        rows.append(row)
    return pd.DataFrame(rows)
def heatmap_from_long_table(
    table: pd.DataFrame,
    row_col: str,
    feature_col: str,
    value_col: str,
    title: str,
    colorbar_label: str,
    outpath: Path,
    top_features: int = 40,
) -> None:
    if table.empty:
        return
    feature_scores = (
        table.groupby(feature_col)[value_col]
        .apply(lambda x: float(np.mean(np.abs(x))))
        .sort_values(ascending=False)
    )
    keep_features = feature_scores.head(top_features).index
    matrix = (
        table[table[feature_col].isin(keep_features)]
        .pivot_table(index=row_col, columns=feature_col, values=value_col, aggfunc="mean")
        .fillna(0.0)
    )
    if matrix.empty:
        return
    vmax = float(np.abs(matrix.to_numpy()).max())
    vmax = max(vmax, 1e-6)
    fig, ax = plt.subplots(
        figsize=(max(8, 0.28 * matrix.shape[1]), max(4, 0.35 * matrix.shape[0]))
    )
    image = ax.imshow(matrix.to_numpy(), aspect="auto", cmap=PLOT_CMAP_DIVERGING, vmin=-vmax, vmax=vmax)
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels(matrix.columns, rotation=90)
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels(matrix.index)
    ax.set_title(title)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label(colorbar_label)
    save_plot(fig, outpath)
def matrix_plot(
    matrix: pd.DataFrame,
    title: str,
    colorbar_label: str,
    outpath: Path,
    cmap: str = PLOT_CMAP_SEQUENTIAL,
) -> None:
    if matrix.empty:
        return
    fig, ax = plt.subplots(
        figsize=(max(5.5, 0.55 * matrix.shape[1]), max(4.5, 0.45 * matrix.shape[0]))
    )
    image = ax.imshow(matrix.to_numpy(), aspect="auto", cmap=cmap)
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels(matrix.index)
    ax.set_title(title)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label(colorbar_label)
    save_plot(fig, outpath)
def bar_plot_from_series(series: pd.Series, title: str, ylabel: str, outpath: Path) -> None:
    if series.empty:
        return
    fig, ax = plt.subplots(figsize=(max(6.0, 0.4 * len(series)), 4.0))
    ax.bar(np.arange(len(series)), series.to_numpy(), color=COLOR_GAT, edgecolor="black", linewidth=0.6)
    ax.set_xticks(np.arange(len(series)))
    ax.set_xticklabels(series.index, rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    save_plot(fig, outpath)
def load_training_module(script_path: Path):
    spec = importlib.util.spec_from_file_location("joint_multiome_training_module", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load training script from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
@dataclass
class GlobalData:
    obs: pd.DataFrame
    coords: np.ndarray
    eb: np.ndarray
    cell_type: np.ndarray
    guide_names: list[str]
    guide_lookup: dict[str, int]
    guides_np: np.ndarray
    celltype_names: list[str]
    celltypes_np: np.ndarray
    x_np: np.ndarray
    x_tensor: torch.Tensor
    guides_tensor: torch.Tensor
    celltypes_tensor: torch.Tensor
    rna_counts: sp.csr_matrix
    atac_counts: sp.csr_matrix
    rna_feature_names_all: np.ndarray
    atac_feature_names_all: np.ndarray
    rna_library_all: np.ndarray
    atac_library_all: np.ndarray
@dataclass
class RunArtifacts:
    run: int
    validation_eb: str
    test_eb: str
    outdir: Path
    rna_feature_names: np.ndarray
    atac_feature_names: np.ndarray
    rna_feature_idx: np.ndarray
    atac_feature_idx: np.ndarray
    rna_mu0: np.ndarray
    atac_mu0: np.ndarray
    rna_log_library: np.ndarray
    atac_log_library: np.ndarray
    rna_bundle: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    atac_bundle: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    rna_neighbors: list[np.ndarray]
    rna_neighbor_distances: list[np.ndarray]
    atac_neighbors: list[np.ndarray]
    atac_neighbor_distances: list[np.ndarray]
    union_neighbors: list[np.ndarray]
    final_seeds: list[int]
def load_global_data(training) -> GlobalData:
    if not DATA_PATH.exists():
        raise FileNotFoundError(DATA_PATH)
    mu.set_options(pull_on_update=False)
    mdata = mu.read_h5mu(DATA_PATH)
    rna = mdata.mod["rna"]
    atac = mdata.mod["atac"]
    rna_names = np.asarray(rna.obs_names.astype(str))
    atac_names = np.asarray(atac.obs_names.astype(str))
    main_names = np.asarray(mdata.obs_names.astype(str))
    if not np.array_equal(rna_names, atac_names) or not np.array_equal(rna_names, main_names):
        raise ValueError("RNA, ATAC, and MuData cells are not in identical order")
    obs = mdata.obs.copy()
    coords = training.dense32(rna.obsm[training.SPATIAL_KEY])
    eb_all = obs[training.EB_KEY].astype(str).to_numpy()
    valid_mask = np.asarray([valid_eb_label(value) for value in eb_all], dtype=bool)
    obs = obs.iloc[np.flatnonzero(valid_mask)].copy()
    coords = coords[valid_mask]
    rna_counts = training.get_counts(rna, training.RNA_COUNTS_LAYER, "RNA", False)[valid_mask]
    atac_counts = training.get_counts(atac, training.ATAC_COUNTS_LAYER, "ATAC", True)[valid_mask]
    guides_np, guide_names = build_guides(obs, training.GUIDE_SOURCE_COL)
    celltype_df = pd.get_dummies(obs[training.CELL_TYPE_KEY].astype(str), dtype=np.float32)
    celltypes_np = celltype_df.to_numpy(np.float32)
    x_np = np.concatenate([guides_np, celltypes_np], axis=1).astype(np.float32)
    return GlobalData(
        obs=obs,
        coords=coords,
        eb=obs[training.EB_KEY].astype(str).to_numpy(),
        cell_type=obs[training.CELL_TYPE_KEY].astype(str).to_numpy(),
        guide_names=guide_names,
        guide_lookup={name: idx for idx, name in enumerate(guide_names)},
        guides_np=guides_np,
        celltype_names=list(map(str, celltype_df.columns)),
        celltypes_np=celltypes_np,
        x_np=x_np,
        x_tensor=torch.tensor(x_np, dtype=torch.float32, device=DEVICE),
        guides_tensor=torch.tensor(guides_np, dtype=torch.float32, device=DEVICE),
        celltypes_tensor=torch.tensor(celltypes_np, dtype=torch.float32, device=DEVICE),
        rna_counts=rna_counts,
        atac_counts=atac_counts,
        rna_feature_names_all=np.asarray(rna.var_names.astype(str)),
        atac_feature_names_all=np.asarray(atac.var_names.astype(str)),
        rna_library_all=np.asarray(rna_counts.sum(1)).ravel().astype(np.float32),
        atac_library_all=np.asarray(atac_counts.sum(1)).ravel().astype(np.float32),
    )
def build_incoming_neighbor_lists(
    edge_index: torch.Tensor,
    edge_distance: torch.Tensor,
    n_cells: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    src = edge_index[0].detach().cpu().numpy().astype(int)
    dst = edge_index[1].detach().cpu().numpy().astype(int)
    dist = edge_distance.detach().cpu().numpy().reshape(-1).astype(np.float32)
    neighbors = [[] for _ in range(n_cells)]
    distances = [[] for _ in range(n_cells)]
    for s, d, value in zip(src, dst, dist, strict=False):
        neighbors[d].append(s)
        distances[d].append(value)
    return (
        [np.asarray(values, dtype=int) for values in neighbors],
        [np.asarray(values, dtype=np.float32) for values in distances],
    )
def make_union_neighbors(
    left_neighbors: list[np.ndarray],
    right_neighbors: list[np.ndarray],
) -> list[np.ndarray]:
    union_neighbors: list[np.ndarray] = []
    for left, right in zip(left_neighbors, right_neighbors, strict=False):
        if len(left) == 0 and len(right) == 0:
            union_neighbors.append(np.asarray([], dtype=int))
        else:
            union_neighbors.append(np.unique(np.concatenate([left, right]).astype(int)))
    return union_neighbors
def load_run_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)
def build_run_artifacts(
    training,
    global_data: GlobalData,
) -> tuple[list[RunArtifacts], dict[tuple[int, float], tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[np.ndarray], list[np.ndarray]]]]:
    config = json.loads((TRAINING_OUTDIR / "configuration.json").read_text())
    final_seeds = list(map(int, config["final_seeds"]))
    manifest = load_run_table(TRAINING_OUTDIR / "split_manifest.csv")
    selected_runs = set(range(1, len(manifest) + 1)) if SELECT_RUNS is None else set(map(int, SELECT_RUNS))
    graph_cache: dict[
        tuple[int, float],
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[np.ndarray], list[np.ndarray]],
    ] = {}
    def graph_bundle(k: int, decay: float):
        key = (int(k), float(decay))
        if key not in graph_cache:
            edge_cpu, dist_cpu = training.build_knn_graph(global_data.coords, global_data.eb, int(k))
            context_np = training.weighted_neighbor_context(
                global_data.guides_np,
                global_data.celltypes_np,
                edge_cpu,
                dist_cpu,
                float(decay),
            )
            neighbors, distances = build_incoming_neighbor_lists(edge_cpu, dist_cpu, len(global_data.obs))
            graph_cache[key] = (
                edge_cpu.to(DEVICE),
                dist_cpu.to(DEVICE),
                torch.tensor(context_np, dtype=torch.float32, device=DEVICE),
                neighbors,
                distances,
            )
        return graph_cache[key]
    runs: list[RunArtifacts] = []
    for row in manifest.itertuples(index=False):
        run = int(row.run)
        if run not in selected_runs:
            continue
        selected_graphs = load_run_table(TRAINING_OUTDIR / f"run{run}_selected_modality_graphs.csv")
        graph_lookup = {
            str(entry.modality): (int(entry.k_neighbors), float(entry.distance_decay))
            for entry in selected_graphs.itertuples(index=False)
        }
        rna_k, rna_decay = graph_lookup["rna"]
        atac_k, atac_decay = graph_lookup["atac"]
        rna_bundle = graph_bundle(rna_k, rna_decay)
        atac_bundle = graph_bundle(atac_k, atac_decay)
        rna_bundle_torch = rna_bundle[:3]
        atac_bundle_torch = atac_bundle[:3]
        rna_neighbors, rna_distances = rna_bundle[3], rna_bundle[4]
        atac_neighbors, atac_distances = atac_bundle[3], atac_bundle[4]
        union_neighbors = make_union_neighbors(rna_neighbors, atac_neighbors)
        rna_targets = load_run_table(TRAINING_OUTDIR / f"run{run}_rna_hvgs.csv")
        atac_targets = load_run_table(TRAINING_OUTDIR / f"run{run}_atac_deviance_features.csv")
        offsets = np.load(TRAINING_OUTDIR / f"run{run}_offsets_and_targets.npz")
        runs.append(
            RunArtifacts(
                run=run,
                validation_eb=str(row.validation_eb),
                test_eb=str(row.test_eb),
                outdir=OUTROOT / f"run{run:02d}_{safe_name(str(row.test_eb))}",
                rna_feature_names=rna_targets["gene"].astype(str).to_numpy(),
                atac_feature_names=atac_targets["atac_feature"].astype(str).to_numpy(),
                rna_feature_idx=offsets["rna_target_idx"].astype(int),
                atac_feature_idx=offsets["atac_target_idx"].astype(int),
                rna_mu0=offsets["rna_mu0"].astype(np.float32),
                atac_mu0=offsets["atac_mu0"].astype(np.float32),
                rna_log_library=offsets["rna_log_library"].astype(np.float32),
                atac_log_library=offsets["atac_log_library"].astype(np.float32),
                rna_bundle=rna_bundle_torch,
                atac_bundle=atac_bundle_torch,
                rna_neighbors=rna_neighbors,
                rna_neighbor_distances=rna_distances,
                atac_neighbors=atac_neighbors,
                atac_neighbor_distances=atac_distances,
                union_neighbors=union_neighbors,
                final_seeds=final_seeds,
            )
        )
    return runs, graph_cache
def eligible_central_cells_for_run(global_data: GlobalData, run_data: RunArtifacts) -> np.ndarray:
    mask = np.ones(len(global_data.obs), dtype=bool)
    if CENTRAL_CELL_SCOPE == "test_eb":
        mask &= global_data.eb == run_data.test_eb
    elif CENTRAL_CELL_SCOPE == "selected_ebs":
        if not SELECTED_EBS:
            raise ValueError("SELECTED_EBS must be set when CENTRAL_CELL_SCOPE='selected_ebs'")
        mask &= np.isin(global_data.eb, np.asarray(SELECTED_EBS, dtype=str))
    elif CENTRAL_CELL_SCOPE == "all":
        pass
    else:
        raise ValueError(f"Unknown CENTRAL_CELL_SCOPE: {CENTRAL_CELL_SCOPE}")
    if SELECTED_CENTRAL_CELL_TYPES is not None:
        mask &= np.isin(global_data.cell_type, np.asarray(SELECTED_CENTRAL_CELL_TYPES, dtype=str))
    indices = np.flatnonzero(mask)
    if len(indices) == 0:
        raise ValueError(f"No central cells available for run {run_data.run}")
    return indices
def load_gat_models(training, global_data: GlobalData, run_data: RunArtifacts):
    models = []
    context_input_dim = global_data.guides_np.shape[1] + global_data.celltypes_np.shape[1] + 3
    for seed in run_data.final_seeds:
        checkpoint_path = TRAINING_OUTDIR / f"run{run_data.run}_seed{seed}_gat_best.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(checkpoint_path)
        model = training.MultiomicSpatialResidualGAT(
            global_data.x_tensor.shape[1],
            global_data.guides_np.shape[1],
            global_data.celltypes_np.shape[1],
            len(run_data.rna_feature_names),
            len(run_data.atac_feature_names),
            context_input_dim,
            float(load_run_table(TRAINING_OUTDIR / f"run{run_data.run}_selected_modality_graphs.csv").query("modality == 'rna'")["distance_decay"].iloc[0]),
            float(load_run_table(TRAINING_OUTDIR / f"run{run_data.run}_selected_modality_graphs.csv").query("modality == 'atac'")["distance_decay"].iloc[0]),
        ).to(DEVICE)
        state = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(state, strict=True)
        model.eval()
        models.append(model)
    return models
@torch.no_grad()
def baseline_predictions(
    training,
    global_data: GlobalData,
    run_data: RunArtifacts,
    models,
) -> dict[str, np.ndarray]:
    rna_loglib = torch.tensor(run_data.rna_log_library, dtype=torch.float32, device=DEVICE)
    atac_loglib = torch.tensor(run_data.atac_log_library, dtype=torch.float32, device=DEVICE)
    rna_mu0 = torch.tensor(run_data.rna_mu0, dtype=torch.float32, device=DEVICE)
    atac_mu0 = torch.tensor(run_data.atac_mu0, dtype=torch.float32, device=DEVICE)
    rna_predictions = []
    atac_predictions = []
    for model in models:
        rna_residual, atac_residual = model(
            global_data.x_tensor,
            global_data.guides_tensor,
            global_data.celltypes_tensor,
            run_data.rna_bundle,
            run_data.atac_bundle,
        )
        rna_predictions.append(training.make_mu(rna_residual, rna_loglib, rna_mu0).detach().cpu().numpy())
        atac_predictions.append(training.make_mu(atac_residual, atac_loglib, atac_mu0).detach().cpu().numpy())
    return {
        "rna": np.mean(np.stack(rna_predictions, axis=0), axis=0).astype(np.float32),
        "atac": np.mean(np.stack(atac_predictions, axis=0), axis=0).astype(np.float32),
    }
def compute_branch_context_from_local_neighbors(
    edited_guides: np.ndarray,
    neighbor_indices: np.ndarray,
    neighbor_distances: np.ndarray,
    celltypes_np: np.ndarray,
    decay: float,
) -> torch.Tensor:
    if len(neighbor_indices) == 0:
        guide_mean = np.zeros((edited_guides.shape[1],), dtype=np.float32)
        celltype_mean = np.zeros((celltypes_np.shape[1],), dtype=np.float32)
        degree = 0.0
        mean_distance = 0.0
        min_distance = 0.0
    else:
        weights = np.exp(-neighbor_distances / float(decay)).astype(np.float32)
        weights_sum = float(np.clip(weights.sum(), 1e-8, None))
        guide_mean = (edited_guides[neighbor_indices] * weights[:, None]).sum(axis=0) / weights_sum
        celltype_mean = (celltypes_np[neighbor_indices] * weights[:, None]).sum(axis=0) / weights_sum
        degree = float(len(neighbor_indices))
        mean_distance = float((neighbor_distances * weights).sum() / weights_sum)
        min_distance = float(neighbor_distances.min())
    context = np.concatenate(
        [
            guide_mean.astype(np.float32),
            celltype_mean.astype(np.float32),
            np.asarray(
                [math.log1p(degree), mean_distance / float(decay), min_distance / float(decay)],
                dtype=np.float32,
            ),
        ]
    )
    return torch.tensor(context[None, :], dtype=torch.float32, device=DEVICE)
def predict_branch_for_central_cell_local(
    branch,
    x_np: np.ndarray,
    global_data: GlobalData,
    central_cell: int,
    neighbor_indices: np.ndarray,
    neighbor_distances: np.ndarray,
    edited_cells: np.ndarray,
    source_index: int,
    target_index: int,
    edit_mode: str,
    log_library_value: float,
    mu0_np: np.ndarray,
) -> np.ndarray:
    local_indices = np.concatenate([[central_cell], neighbor_indices]).astype(int)
    local_x = x_np[local_indices].copy()
    if len(edited_cells):
        local_lookup = {global_idx: local_idx for local_idx, global_idx in enumerate(local_indices)}
        overlapping = [local_lookup[idx] for idx in edited_cells if idx in local_lookup]
        if overlapping:
            overlapping = np.asarray(overlapping, dtype=int)
            if edit_mode == "replace_source_with_target":
                local_x[overlapping, source_index] = 0.0
                local_x[overlapping, target_index] = 1.0
            elif edit_mode == "set_all_neighbors_to_target":
                local_x[overlapping, : len(global_data.guide_names)] = 0.0
                local_x[overlapping, target_index] = 1.0
            else:
                raise ValueError(f"Unknown COUNTERFACTUAL_EDIT_MODE: {edit_mode}")
    local_x_tensor = torch.tensor(local_x, dtype=torch.float32, device=DEVICE)
    local_guides_tensor = local_x_tensor[:, : len(global_data.guide_names)]
    local_celltypes_tensor = local_x_tensor[:, len(global_data.guide_names):]
    base = branch.base(
        local_x_tensor[0:1],
        local_guides_tensor[0:1],
        local_celltypes_tensor[0:1],
    )
    receiver = branch.receiver_encoder(local_celltypes_tensor[0:1])
    if len(neighbor_indices) == 0:
        spatial = torch.zeros((1, training_module.SPATIAL_DIM), dtype=torch.float32, device=DEVICE)
    else:
        source_tensor = branch.source_encoder(local_x_tensor[1:])
        scaled = torch.tensor(neighbor_distances[:, None], dtype=torch.float32, device=DEVICE) / branch.distance_decay
        edge_raw = torch.cat(
            [
                scaled,
                torch.exp(-scaled),
                1.0 / (1.0 + scaled),
                torch.log1p(torch.tensor(neighbor_distances[:, None], dtype=torch.float32, device=DEVICE))
                / math.log1p(branch.distance_decay),
            ],
            dim=1,
        )
        edge_attr = branch.edge_encoder(edge_raw)
        local_edge_index = torch.stack(
            [
                torch.arange(len(neighbor_indices), dtype=torch.long, device=DEVICE),
                torch.zeros(len(neighbor_indices), dtype=torch.long, device=DEVICE),
            ],
            dim=0,
        )
        spatial = F.elu(branch.gat((source_tensor, receiver), local_edge_index, edge_attr=edge_attr))
    edited_guides = local_x[:, : len(global_data.guide_names)].astype(np.float32)
    context = branch.context_encoder(
        compute_branch_context_from_local_neighbors(
            edited_guides,
            neighbor_indices=np.arange(1, len(local_indices), dtype=int),
            neighbor_distances=neighbor_distances,
            celltypes_np=local_x[:, len(global_data.guide_names):].astype(np.float32),
            decay=branch.distance_decay,
        )
    )
    gate = branch.gate(torch.cat([receiver, spatial, context], dim=1))
    correction = branch.correction_head(
        branch.correction_encoder(torch.cat([receiver, gate * spatial, context], dim=1))
    )
    residual = base + correction
    mu = torch.exp(
        (residual + float(log_library_value) + torch.tensor(mu0_np[None, :], dtype=torch.float32, device=DEVICE)).clamp(-12, 12)
    )[0]
    return mu.detach().cpu().numpy().astype(np.float32)
def strict_receiver_ok(global_data: GlobalData, central_cell: int, source_index: int, target_index: int) -> bool:
    return (
        global_data.guides_np[central_cell, source_index] <= 0
        and global_data.guides_np[central_cell, target_index] <= 0
    )
def neighborhood_cells_for_edit(
    global_data: GlobalData,
    run_data: RunArtifacts,
    central_cell: int,
    distance_band: tuple[float, float] | None = None,
    inclusive_right: bool = False,
) -> np.ndarray:
    neighbors = run_data.union_neighbors[central_cell]
    if len(neighbors) == 0:
        return neighbors
    if distance_band is None:
        return neighbors
    left_um, right_um = distance_band
    distances = np.linalg.norm(
        global_data.coords[neighbors, :2] - global_data.coords[central_cell, :2],
        axis=1,
    ).astype(np.float32)
    if np.isinf(right_um):
        keep = distances >= left_um
    elif inclusive_right:
        keep = (distances >= left_um) & (distances <= right_um)
    else:
        keep = (distances >= left_um) & (distances < right_um if not np.isclose(right_um, left_um) else distances == right_um)
    return neighbors[keep]
def edited_cells_for_counterfactual(
    global_data: GlobalData,
    run_data: RunArtifacts,
    central_cell: int,
    source_index: int,
    target_index: int,
    min_source_neighbors: int,
    distance_band: tuple[float, float] | None = None,
    inclusive_right: bool = False,
) -> tuple[np.ndarray, int]:
    candidate_cells = neighborhood_cells_for_edit(global_data, run_data, central_cell, distance_band, inclusive_right)
    if EDIT_CENTRAL_CELL:
        candidate_cells = np.unique(np.concatenate([candidate_cells, np.asarray([central_cell], dtype=int)]))
    if len(candidate_cells) == 0:
        return np.asarray([], dtype=int), 0
    if COUNTERFACTUAL_EDIT_MODE == "replace_source_with_target":
        source_positive = global_data.guides_np[candidate_cells, source_index] > 0
        edited = candidate_cells[source_positive]
        n_source_positive = int(source_positive.sum())
    elif COUNTERFACTUAL_EDIT_MODE == "set_all_neighbors_to_target":
        edited = candidate_cells
        n_source_positive = int((global_data.guides_np[candidate_cells, source_index] > 0).sum())
    else:
        raise ValueError(f"Unknown COUNTERFACTUAL_EDIT_MODE: {COUNTERFACTUAL_EDIT_MODE}")
    if n_source_positive < min_source_neighbors:
        return np.asarray([], dtype=int), n_source_positive
    return edited.astype(int), n_source_positive
def predict_counterfactual_for_central_cell(
    training,
    global_data: GlobalData,
    run_data: RunArtifacts,
    models,
    baseline: dict[str, np.ndarray],
    central_cell: int,
    source_guide: str,
    target_guide: str,
    distance_band: tuple[float, float] | None = None,
    min_source_neighbors: int = MIN_SOURCE_POSITIVE_NEIGHBORS,
    inclusive_right: bool = False,
) -> dict[str, object] | None:
    source_index = global_data.guide_lookup[source_guide]
    target_index = global_data.guide_lookup[target_guide]
    if not strict_receiver_ok(global_data, central_cell, source_index, target_index):
        return None
    edited_cells, n_source_positive = edited_cells_for_counterfactual(
        global_data,
        run_data,
        central_cell,
        source_index,
        target_index,
        min_source_neighbors=min_source_neighbors,
        distance_band=distance_band,
        inclusive_right=inclusive_right,
    )
    if n_source_positive < min_source_neighbors:
        return None
    rna_predictions = []
    atac_predictions = []
    for model in models:
        rna_predictions.append(
            predict_branch_for_central_cell_local(
                model.rna,
                global_data.x_np,
                global_data,
                central_cell,
                run_data.rna_neighbors[central_cell],
                run_data.rna_neighbor_distances[central_cell],
                edited_cells,
                source_index,
                target_index,
                COUNTERFACTUAL_EDIT_MODE,
                run_data.rna_log_library[central_cell],
                run_data.rna_mu0,
            )
        )
        atac_predictions.append(
            predict_branch_for_central_cell_local(
                model.atac,
                global_data.x_np,
                global_data,
                central_cell,
                run_data.atac_neighbors[central_cell],
                run_data.atac_neighbor_distances[central_cell],
                edited_cells,
                source_index,
                target_index,
                COUNTERFACTUAL_EDIT_MODE,
                run_data.atac_log_library[central_cell],
                run_data.atac_mu0,
            )
        )
    counterfactual_rna = np.mean(np.stack(rna_predictions, axis=0), axis=0).astype(np.float32)
    counterfactual_atac = np.mean(np.stack(atac_predictions, axis=0), axis=0).astype(np.float32)
    baseline_rna = baseline["rna"][central_cell]
    baseline_atac = baseline["atac"][central_cell]
    return {
        "central_cell": int(central_cell),
        "source_guide": source_guide,
        "target_guide": target_guide,
        "edited_cells": edited_cells,
        "n_edited_cells": int(len(edited_cells)),
        "n_source_positive_neighbors": int(n_source_positive),
        "rna_log1p_effect": (np.log1p(counterfactual_rna) - np.log1p(baseline_rna)).astype(np.float32),
        "atac_log1p_effect": (np.log1p(counterfactual_atac) - np.log1p(baseline_atac)).astype(np.float32),
        "rna_raw_effect": (counterfactual_rna - baseline_rna).astype(np.float32),
        "atac_raw_effect": (counterfactual_atac - baseline_atac).astype(np.float32),
    }
def choose_default_control_guide(global_data: GlobalData) -> str | None:
    controls = [guide for guide in global_data.guide_names if is_control_guide(guide)]
    if not controls:
        return None
    counts = global_data.guides_np.sum(axis=0)
    ranked = sorted(controls, key=lambda x: float(counts[global_data.guide_lookup[x]]), reverse=True)
    return ranked[0]
def choose_screen_guides(global_data: GlobalData, control_guide: str | None) -> list[str]:
    if SCREEN_GUIDES is not None:
        guides = list(SCREEN_GUIDES)
    else:
        counts = pd.Series(global_data.guides_np.sum(axis=0), index=global_data.guide_names)
        counts = counts.sort_values(ascending=False)
        guides = []
        for guide in counts.index:
            if control_guide is not None and guide == control_guide:
                continue
            if is_control_guide(guide):
                continue
            guides.append(str(guide))
            if len(guides) >= N_AUTO_SCREEN_GUIDES:
                break
    for guide in guides:
        if guide not in global_data.guide_lookup:
            raise KeyError(f"Unknown guide: {guide}")
    return guides
def choose_module_pairs(global_data: GlobalData) -> tuple[list[tuple[str, str]], list[tuple[str, str]], list[str], str | None]:
    control_guide = choose_default_control_guide(global_data)
    screen_guides = choose_screen_guides(global_data, control_guide)
    if MODULE_1_GUIDE_SUBSTITUTIONS is not None:
        module_1_pairs = list(MODULE_1_GUIDE_SUBSTITUTIONS)
    else:
        if control_guide is None:
            raise ValueError("No control guide detected; set MODULE_1_GUIDE_SUBSTITUTIONS explicitly.")
        module_1_pairs = [(guide, control_guide) for guide in screen_guides]
    if MODULE_3_GUIDE_PAIRS is not None:
        module_3_pairs = list(MODULE_3_GUIDE_PAIRS)
    else:
        module_3_pairs = [(left, right) for left in screen_guides for right in screen_guides if left != right]
    return module_1_pairs, module_3_pairs, screen_guides, control_guide
def guide_neighbor_count(
    global_data: GlobalData,
    run_data: RunArtifacts,
    guide_index: int,
) -> np.ndarray:
    counts = np.zeros(len(global_data.obs), dtype=np.int32)
    guide_vector = global_data.guides_np[:, guide_index]
    for central_cell, neighbors in enumerate(run_data.union_neighbors):
        if len(neighbors):
            counts[central_cell] = int(guide_vector[neighbors].sum())
    return counts
def make_matching_covariates(
    global_data: GlobalData,
    run_data: RunArtifacts,
    candidate_indices: np.ndarray,
) -> np.ndarray:
    union_degree = np.asarray([len(run_data.union_neighbors[idx]) for idx in range(len(global_data.obs))], dtype=np.float32)
    neighbor_celltype_props = np.zeros((len(global_data.obs), len(global_data.celltype_names)), dtype=np.float32)
    for central_cell, neighbors in enumerate(run_data.union_neighbors):
        if len(neighbors):
            neighbor_celltype_props[central_cell] = global_data.celltypes_np[neighbors].mean(axis=0)
    columns = [
        np.log(np.clip(global_data.rna_library_all[candidate_indices], 1.0, None))[:, None],
        np.log(np.clip(global_data.atac_library_all[candidate_indices], 1.0, None))[:, None],
        union_degree[candidate_indices, None],
        neighbor_celltype_props[candidate_indices],
    ]
    if MATCH_INCLUDE_OWN_GUIDE_COUNT:
        columns.append(global_data.guides_np[candidate_indices].sum(axis=1, keepdims=True))
    matrix = np.concatenate(columns, axis=1).astype(np.float64)
    return standardize_matrix(matrix)
def greedy_match_two_groups(
    global_data: GlobalData,
    run_data: RunArtifacts,
    source_indices: np.ndarray,
    target_indices: np.ndarray,
) -> pd.DataFrame:
    source_indices = np.asarray(source_indices, dtype=int)
    target_indices = np.asarray(target_indices, dtype=int)
    if len(source_indices) == 0 or len(target_indices) == 0:
        return pd.DataFrame()
    source_covariates = make_matching_covariates(global_data, run_data, source_indices)
    target_covariates = make_matching_covariates(global_data, run_data, target_indices)
    strata_source = []
    strata_target = []
    for cell_idx in source_indices:
        key = []
        if MATCH_WITHIN_EB:
            key.append(global_data.eb[cell_idx])
        if MATCH_WITHIN_CELL_TYPE:
            key.append(global_data.cell_type[cell_idx])
        strata_source.append(tuple(key))
    for cell_idx in target_indices:
        key = []
        if MATCH_WITHIN_EB:
            key.append(global_data.eb[cell_idx])
        if MATCH_WITHIN_CELL_TYPE:
            key.append(global_data.cell_type[cell_idx])
        strata_target.append(tuple(key))
    unique_strata = sorted(set(strata_source) & set(strata_target))
    match_rows = []
    match_id = 0
    for stratum in unique_strata:
        source_positions = np.flatnonzero(np.asarray([value == stratum for value in strata_source], dtype=bool))
        target_positions = np.flatnonzero(np.asarray([value == stratum for value in strata_target], dtype=bool))
        if len(source_positions) == 0 or len(target_positions) == 0:
            continue
        nn = NearestNeighbors(
            n_neighbors=min(len(target_positions), 10 if MATCH_WITHOUT_REPLACEMENT else 1),
            metric="euclidean",
        )
        nn.fit(target_covariates[target_positions])
        distances, neighbors = nn.kneighbors(source_covariates[source_positions])
        used_target_local: set[int] = set()
        order = np.argsort(distances[:, 0])
        for local_source in order:
            chosen_target_local = None
            chosen_distance = None
            for rank in range(neighbors.shape[1]):
                candidate = int(neighbors[local_source, rank])
                if MATCH_WITHOUT_REPLACEMENT and candidate in used_target_local:
                    continue
                distance_value = float(distances[local_source, rank])
                if MATCH_CALIPER is not None and distance_value > MATCH_CALIPER:
                    continue
                chosen_target_local = candidate
                chosen_distance = distance_value
                break
            if chosen_target_local is None:
                continue
            if MATCH_WITHOUT_REPLACEMENT:
                used_target_local.add(chosen_target_local)
            source_cell = int(source_indices[source_positions[local_source]])
            target_cell = int(target_indices[target_positions[chosen_target_local]])
            match_rows.append(
                {
                    "match_id": match_id,
                    "source_exposed_cell_index": source_cell,
                    "target_exposed_cell_index": target_cell,
                    "matching_distance": chosen_distance,
                    "EB": global_data.eb[source_cell],
                    "cell_type": global_data.cell_type[source_cell],
                }
            )
            match_id += 1
    return pd.DataFrame(match_rows)
def eb_bootstrap_feature_summary(
    effect_matrix: np.ndarray,
    eb_labels: np.ndarray,
    n_bootstraps: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    effect_matrix = np.asarray(effect_matrix, dtype=np.float32)
    eb_labels = np.asarray(eb_labels, dtype=object)
    unique_eb = np.asarray(pd.unique(eb_labels), dtype=object)
    mean_effect = effect_matrix.mean(axis=0)
    median_effect = np.median(effect_matrix, axis=0)
    rng_local = np.random.default_rng(seed)
    draws = np.empty((n_bootstraps, effect_matrix.shape[1]), dtype=np.float32)
    for draw_idx in range(n_bootstraps):
        sampled_ebs = rng_local.choice(unique_eb, size=len(unique_eb), replace=True)
        sampled_rows = np.concatenate([np.flatnonzero(eb_labels == eb_name) for eb_name in sampled_ebs])
        draws[draw_idx] = effect_matrix[sampled_rows].mean(axis=0)
    ci_lower = np.quantile(draws, 0.025, axis=0)
    ci_upper = np.quantile(draws, 0.975, axis=0)
    bootstrap_se = draws.std(axis=0, ddof=1)
    z_score = np.divide(mean_effect, bootstrap_se, out=np.zeros_like(mean_effect, dtype=np.float64), where=bootstrap_se > 0)
    p_values = np.asarray([math.erfc(abs(float(z)) / math.sqrt(2.0)) for z in z_score], dtype=np.float64)
    zero_se = bootstrap_se <= 0
    p_values[zero_se & (mean_effect != 0)] = 0.0
    p_values[zero_se & (mean_effect == 0)] = 1.0
    return mean_effect, median_effect, ci_lower, ci_upper, p_values
def run_module_1(
    training,
    global_data: GlobalData,
    run_data: RunArtifacts,
    models,
    baseline: dict[str, np.ndarray],
    eligible_cells: np.ndarray,
    module_1_pairs: list[tuple[str, str]],
) -> None:
    module_dir = run_data.outdir / "module_1_neighborhood_substitution"
    module_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    aggregate_tables: dict[str, list[pd.DataFrame]] = {"rna": [], "atac": []}
    top_rows: list[pd.DataFrame] = []
    for source_guide, target_guide in module_1_pairs:
        retained = []
        rna_effects = []
        atac_effects = []
        cells = sample_indices(eligible_cells, MAX_CENTRAL_CELLS_MODULE_1)
        for central_cell in cells:
            result = predict_counterfactual_for_central_cell(
                training, global_data, run_data, models, baseline, central_cell, source_guide, target_guide
            )
            if result is None:
                continue
            retained.append(result)
            rna_effects.append(result["rna_log1p_effect"])
            atac_effects.append(result["atac_log1p_effect"])
            top_rows.append(
                top_feature_table(
                    run_data.rna_feature_names,
                    result["rna_log1p_effect"][None, :],
                    TOP_FEATURES_PER_CELL_SCENARIO,
                    "feature",
                    {
                        "modality": "rna",
                        "source_guide": source_guide,
                        "target_guide": target_guide,
                        "central_cell_index": int(result["central_cell"]),
                        "central_cell_id": str(global_data.obs.index[int(result["central_cell"])]),
                        "EB": global_data.eb[int(result["central_cell"])],
                        "cell_type": global_data.cell_type[int(result["central_cell"])],
                    },
                )
            )
            top_rows.append(
                top_feature_table(
                    run_data.atac_feature_names,
                    result["atac_log1p_effect"][None, :],
                    TOP_FEATURES_PER_CELL_SCENARIO,
                    "feature",
                    {
                        "modality": "atac",
                        "source_guide": source_guide,
                        "target_guide": target_guide,
                        "central_cell_index": int(result["central_cell"]),
                        "central_cell_id": str(global_data.obs.index[int(result["central_cell"])]),
                        "EB": global_data.eb[int(result["central_cell"])],
                        "cell_type": global_data.cell_type[int(result["central_cell"])],
                    },
                )
            )
        summary_rows.append(
            {
                "source_guide": source_guide,
                "target_guide": target_guide,
                "n_valid_central_cells": len(retained),
            }
        )
        if not retained:
            continue
        rna_matrix = np.stack(rna_effects, axis=0)
        atac_matrix = np.stack(atac_effects, axis=0)
        rna_table = top_feature_table(
            run_data.rna_feature_names,
            rna_matrix,
            len(run_data.rna_feature_names),
            "feature",
            {"modality": "rna", "source_guide": source_guide, "target_guide": target_guide},
        )
        atac_table = top_feature_table(
            run_data.atac_feature_names,
            atac_matrix,
            len(run_data.atac_feature_names),
            "feature",
            {"modality": "atac", "source_guide": source_guide, "target_guide": target_guide},
        )
        rna_table.to_csv(module_dir / f"aggregate_rna_{safe_name(source_guide)}_vs_{safe_name(target_guide)}.csv", index=False)
        atac_table.to_csv(module_dir / f"aggregate_atac_{safe_name(source_guide)}_vs_{safe_name(target_guide)}.csv", index=False)
        aggregate_tables["rna"].append(rna_table)
        aggregate_tables["atac"].append(atac_table)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(module_dir / "cell_level_scenario_summary.csv", index=False)
    if top_rows:
        pd.concat(top_rows, ignore_index=True).to_csv(module_dir / "top_features_per_cell_scenario.csv", index=False)
    if not summary_df.empty:
        pairs = summary_df["source_guide"] + " -> " + summary_df["target_guide"]
        counts = pd.Series(summary_df["n_valid_central_cells"].to_numpy(), index=pairs)
        bar_plot_from_series(counts, "Valid counterfactual receiver scenarios", "Receiver cells", module_dir / "module_1_valid_scenario_counts")
    for modality in ["rna", "atac"]:
        if aggregate_tables[modality]:
            combined = pd.concat(aggregate_tables[modality], ignore_index=True)
            combined["pair"] = combined["source_guide"] + " -> " + combined["target_guide"]
            combined.to_csv(module_dir / f"module_1_all_{modality}_features.csv", index=False)
            heatmap_from_long_table(
                combined,
                row_col="pair",
                feature_col="feature",
                value_col="mean_log1p_effect",
                title=f"Module 1 {modality.upper()} counterfactual effects",
                colorbar_label="Mean log1p counterfactual effect",
                outpath=module_dir / f"module_1_{modality}_feature_heatmap",
                top_features=40,
            )
def run_module_2(
    global_data: GlobalData,
    run_data: RunArtifacts,
    baseline: dict[str, np.ndarray],
    eligible_cells: np.ndarray,
    module_1_pairs: list[tuple[str, str]],
) -> None:
    module_dir = run_data.outdir / "module_2_matched_observed_comparison"
    module_dir.mkdir(parents=True, exist_ok=True)
    all_match_rows = []
    all_gene_rows: list[pd.DataFrame] = []
    rna_true = global_data.rna_counts[:, run_data.rna_feature_idx].toarray().astype(np.float32)
    atac_true = global_data.atac_counts[:, run_data.atac_feature_idx].toarray().astype(np.float32)
    rna_norm = np.log1p(rna_true * (1e4 / np.maximum(global_data.rna_library_all[:, None], 1.0)))
    atac_norm = np.log1p(atac_true * (1e4 / np.maximum(global_data.atac_library_all[:, None], 1.0)))
    central_pool = eligible_cells.copy()
    for source_guide, target_guide in module_1_pairs:
        source_index = global_data.guide_lookup[source_guide]
        target_index = global_data.guide_lookup[target_guide]
        source_neighbor_count = guide_neighbor_count(global_data, run_data, source_index)
        target_neighbor_count = guide_neighbor_count(global_data, run_data, target_index)
        if MATCH_REQUIRE_EXCLUSIVE_EXPOSURE:
            source_exposed = (source_neighbor_count >= MATCH_MIN_GUIDE_NEIGHBORS) & (target_neighbor_count == 0)
            target_exposed = (target_neighbor_count >= MATCH_MIN_GUIDE_NEIGHBORS) & (source_neighbor_count == 0)
        else:
            source_exposed = (source_neighbor_count >= MATCH_MIN_GUIDE_NEIGHBORS) & (source_neighbor_count > target_neighbor_count)
            target_exposed = (target_neighbor_count >= MATCH_MIN_GUIDE_NEIGHBORS) & (target_neighbor_count > source_neighbor_count)
        receiver_mask = np.asarray(
            [strict_receiver_ok(global_data, idx, source_index, target_index) for idx in range(len(global_data.obs))],
            dtype=bool,
        )
        source_indices = central_pool[receiver_mask[central_pool] & source_exposed[central_pool]]
        target_indices = central_pool[receiver_mask[central_pool] & target_exposed[central_pool]]
        matches = greedy_match_two_groups(global_data, run_data, source_indices, target_indices)
        if matches.empty:
            continue
        source_cells = matches["source_exposed_cell_index"].to_numpy(dtype=int)
        target_cells = matches["target_exposed_cell_index"].to_numpy(dtype=int)
        for modality, names, observed_matrix, predicted_matrix in [
            ("rna", run_data.rna_feature_names, rna_norm, baseline["rna"]),
            ("atac", run_data.atac_feature_names, atac_norm, baseline["atac"]),
        ]:
            observed_difference = observed_matrix[target_cells] - observed_matrix[source_cells]
            predicted_difference = np.log1p(predicted_matrix[target_cells]) - np.log1p(predicted_matrix[source_cells])
            table = pd.DataFrame(
                {
                    "modality": modality,
                    "source_neighbor_guide": source_guide,
                    "target_neighbor_guide": target_guide,
                    "feature": names,
                    "n_matched_pairs": len(matches),
                    "mean_observed_log1p_difference_target_minus_source": observed_difference.mean(axis=0),
                    "median_observed_log1p_difference_target_minus_source": np.median(observed_difference, axis=0),
                    "mean_predicted_log1p_difference_target_minus_source": predicted_difference.mean(axis=0),
                    "median_predicted_log1p_difference_target_minus_source": np.median(predicted_difference, axis=0),
                }
            )
            table["absolute_mean_observed_difference"] = np.abs(table["mean_observed_log1p_difference_target_minus_source"])
            table["observed_predicted_direction_agreement"] = (
                np.sign(table["mean_observed_log1p_difference_target_minus_source"])
                == np.sign(table["mean_predicted_log1p_difference_target_minus_source"])
            )
            table.to_csv(
                module_dir / f"matched_{modality}_{safe_name(source_guide)}_vs_{safe_name(target_guide)}.csv",
                index=False,
            )
            all_gene_rows.append(table)
        matches = matches.copy()
        matches["source_neighbor_guide"] = source_guide
        matches["target_neighbor_guide"] = target_guide
        matches["source_neighbor_count"] = source_neighbor_count[source_cells]
        matches["target_neighbor_count"] = target_neighbor_count[target_cells]
        matches.to_csv(
            module_dir / f"matched_cell_pairs_{safe_name(source_guide)}_vs_{safe_name(target_guide)}.csv",
            index=False,
        )
        all_match_rows.append(matches)
    if not all_gene_rows:
        return
    combined = pd.concat(all_gene_rows, ignore_index=True)
    combined.to_csv(module_dir / "all_matched_guide_pair_feature_effects.csv", index=False)
    if all_match_rows:
        pd.concat(all_match_rows, ignore_index=True).to_csv(module_dir / "all_matched_cell_pairs.csv", index=False)
    for modality in ["rna", "atac"]:
        subset = combined[combined["modality"] == modality].copy()
        if subset.empty:
            continue
        fig, ax = plt.subplots(figsize=(5.25, 5.0))
        ax.scatter(
            subset["mean_observed_log1p_difference_target_minus_source"],
            subset["mean_predicted_log1p_difference_target_minus_source"],
            s=14,
            alpha=0.6,
            color=COLOR_GAT,
            edgecolors="none",
        )
        bounds = np.asarray(
            [
                subset["mean_observed_log1p_difference_target_minus_source"].min(),
                subset["mean_observed_log1p_difference_target_minus_source"].max(),
                subset["mean_predicted_log1p_difference_target_minus_source"].min(),
                subset["mean_predicted_log1p_difference_target_minus_source"].max(),
            ],
            dtype=float,
        )
        low, high = float(bounds.min()), float(bounds.max())
        ax.plot([low, high], [low, high], linestyle="--", color=COLOR_ACCENT, linewidth=1.0)
        corr = np.corrcoef(
            subset["mean_observed_log1p_difference_target_minus_source"],
            subset["mean_predicted_log1p_difference_target_minus_source"],
        )[0, 1]
        ax.text(0.03, 0.97, f"Pearson r = {corr:.2f}", transform=ax.transAxes, ha="left", va="top")
        ax.set_xlabel("Observed matched log1p difference")
        ax.set_ylabel("GAT-predicted matched log1p difference")
        ax.set_title(f"Module 2 {modality.upper()} observed vs predicted")
        save_plot(fig, module_dir / f"module_2_{modality}_observed_vs_predicted")
        subset["pair"] = subset["source_neighbor_guide"] + " -> " + subset["target_neighbor_guide"]
        heatmap_from_long_table(
            subset,
            row_col="pair",
            feature_col="feature",
            value_col="mean_observed_log1p_difference_target_minus_source",
            title=f"Module 2 {modality.upper()} matched observed differences",
            colorbar_label="Mean observed log1p difference",
            outpath=module_dir / f"module_2_{modality}_observed_heatmap",
            top_features=40,
        )
def run_module_3(
    training,
    global_data: GlobalData,
    run_data: RunArtifacts,
    models,
    baseline: dict[str, np.ndarray],
    eligible_cells: np.ndarray,
    module_3_pairs: list[tuple[str, str]],
) -> None:
    module_dir = run_data.outdir / "module_3_guide_pair_screen"
    module_dir.mkdir(parents=True, exist_ok=True)
    pair_rows = []
    top_rows: list[pd.DataFrame] = []
    for source_guide, target_guide in module_3_pairs:
        rna_effects = []
        atac_effects = []
        cells = sample_indices(eligible_cells, MAX_CENTRAL_CELLS_MODULE_3)
        for central_cell in cells:
            result = predict_counterfactual_for_central_cell(
                training, global_data, run_data, models, baseline, central_cell, source_guide, target_guide
            )
            if result is None:
                continue
            rna_effects.append(result["rna_log1p_effect"])
            atac_effects.append(result["atac_log1p_effect"])
        if not rna_effects:
            continue
        rna_matrix = np.stack(rna_effects, axis=0)
        atac_matrix = np.stack(atac_effects, axis=0)
        pair_rows.append(
            {
                "source_guide": source_guide,
                "target_guide": target_guide,
                "n_receiver_cells": rna_matrix.shape[0],
                "mean_absolute_effect_rna": float(np.mean(np.abs(rna_matrix))),
                "max_absolute_effect_rna": float(np.max(np.abs(rna_matrix))),
                "mean_absolute_effect_atac": float(np.mean(np.abs(atac_matrix))),
                "max_absolute_effect_atac": float(np.max(np.abs(atac_matrix))),
            }
        )
        top_rows.append(
            top_feature_table(
                run_data.rna_feature_names,
                rna_matrix,
                MODULE_3_TOP_FEATURES_PER_PAIR,
                "feature",
                {"modality": "rna", "source_guide": source_guide, "target_guide": target_guide},
            )
        )
        top_rows.append(
            top_feature_table(
                run_data.atac_feature_names,
                atac_matrix,
                MODULE_3_TOP_FEATURES_PER_PAIR,
                "feature",
                {"modality": "atac", "source_guide": source_guide, "target_guide": target_guide},
            )
        )
    if not pair_rows:
        return
    pair_df = pd.DataFrame(pair_rows)
    pair_df.to_csv(module_dir / "guide_pair_global_summary.csv", index=False)
    if top_rows:
        top_df = pd.concat(top_rows, ignore_index=True)
        top_df.to_csv(module_dir / "top_response_features_for_all_guide_pairs.csv", index=False)
    for modality, value_col in [
        ("rna", "mean_absolute_effect_rna"),
        ("atac", "mean_absolute_effect_atac"),
    ]:
        matrix = pair_df.pivot_table(index="source_guide", columns="target_guide", values=value_col, aggfunc="mean").fillna(0.0)
        matrix_plot(
            matrix,
            title=f"Module 3 {modality.upper()} guide-pair effect magnitude",
            colorbar_label="Mean absolute log1p effect",
            outpath=module_dir / f"module_3_{modality}_magnitude_matrix",
        )
        if top_rows:
            subset = top_df[top_df["modality"] == modality].copy()
            subset["pair"] = subset["source_guide"] + " -> " + subset["target_guide"]
            heatmap_from_long_table(
                subset,
                row_col="pair",
                feature_col="feature",
                value_col="mean_log1p_effect",
                title=f"Module 3 {modality.upper()} guide-pair feature effects",
                colorbar_label="Mean log1p counterfactual effect",
                outpath=module_dir / f"module_3_{modality}_feature_heatmap",
                top_features=45,
            )
def run_module_4(
    training,
    global_data: GlobalData,
    run_data: RunArtifacts,
    models,
    baseline: dict[str, np.ndarray],
    eligible_cells: np.ndarray,
    screen_guides: list[str],
    control_guide: str | None,
) -> None:
    if control_guide is None:
        return
    module_dir = run_data.outdir / "module_4_distance_resolved_effect_radius"
    plot_dir = module_dir / "plots"
    module_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    distance_bands_um = infer_distance_bands_for_run(global_data, run_data)
    pd.DataFrame(
        [
            {"band_index": idx + 1, "distance_left_um": left, "distance_right_um": right}
            for idx, (left, right) in enumerate(distance_bands_um)
        ]
    ).to_csv(module_dir / "distance_bands_used.csv", index=False)
    radius_guides = screen_guides if RADIUS_GUIDES is None else list(RADIUS_GUIDES)
    if RADIUS_EDIT_DIRECTION == "guide_to_control":
        pairs = [(guide, control_guide) for guide in radius_guides]
    elif RADIUS_EDIT_DIRECTION == "control_to_guide":
        pairs = [(control_guide, guide) for guide in radius_guides]
    else:
        raise ValueError("RADIUS_EDIT_DIRECTION must be 'guide_to_control' or 'control_to_guide'")
    results_by_modality: dict[str, list[pd.DataFrame]] = {"rna": [], "atac": []}
    cell_rows = []
    for source_guide, target_guide in pairs:
        perturbation_guide = source_guide if RADIUS_EDIT_DIRECTION == "guide_to_control" else target_guide
        source_index = global_data.guide_lookup[source_guide]
        target_index = global_data.guide_lookup[target_guide]
        receiver_mask = np.asarray(
            [strict_receiver_ok(global_data, idx, source_index, target_index) for idx in range(len(global_data.obs))],
            dtype=bool,
        )
        for band_index, (left_um, right_um) in enumerate(distance_bands_um):
            retained_cells = []
            rna_effects = []
            atac_effects = []
            for central_cell in sample_indices(eligible_cells, MAX_CENTRAL_CELLS_MODULE_4):
                if not receiver_mask[central_cell]:
                    continue
                result = predict_counterfactual_for_central_cell(
                    training,
                    global_data,
                    run_data,
                    models,
                    baseline,
                    central_cell,
                    source_guide,
                    target_guide,
                    distance_band=(left_um, right_um),
                    min_source_neighbors=MIN_SOURCE_POSITIVE_NEIGHBORS_PER_BAND,
                    inclusive_right=band_index == len(distance_bands_um) - 1,
                )
                if result is None:
                    continue
                retained_cells.append(central_cell)
                rna_effects.append(result["rna_log1p_effect"])
                atac_effects.append(result["atac_log1p_effect"])
                cell_rows.append(
                    {
                        "perturbation_guide": perturbation_guide,
                        "source_guide": source_guide,
                        "target_guide": target_guide,
                        "distance_left_um": left_um,
                        "distance_right_um": right_um,
                        "central_cell_index": int(central_cell),
                        "central_cell_id": str(global_data.obs.index[central_cell]),
                        "central_EB": global_data.eb[central_cell],
                        "central_cell_type": global_data.cell_type[central_cell],
                        "n_edited_cells": int(result["n_edited_cells"]),
                        "n_source_positive_neighbors": int(result["n_source_positive_neighbors"]),
                        "mean_absolute_rna_effect": float(np.mean(np.abs(result["rna_log1p_effect"]))),
                        "mean_absolute_atac_effect": float(np.mean(np.abs(result["atac_log1p_effect"]))),
                    }
                )
            if not retained_cells:
                continue
            eb_labels = global_data.eb[np.asarray(retained_cells, dtype=int)]
            for modality, names, matrix in [
                ("rna", run_data.rna_feature_names, np.stack(rna_effects, axis=0)),
                ("atac", run_data.atac_feature_names, np.stack(atac_effects, axis=0)),
            ]:
                mean_effect, median_effect, ci_lower, ci_upper, p_values = eb_bootstrap_feature_summary(
                    matrix,
                    eb_labels,
                    N_EB_BOOTSTRAPS_MODULE_4,
                    MODULE_4_BOOTSTRAP_SEED + run_data.run,
                )
                table = pd.DataFrame(
                    {
                        "modality": modality,
                        "perturbation_guide": perturbation_guide,
                        "source_guide": source_guide,
                        "target_guide": target_guide,
                        "distance_left_um": left_um,
                        "distance_right_um": right_um,
                        "feature": names,
                        "n_receiver_cells": len(retained_cells),
                        "mean_log1p_predicted_effect": mean_effect,
                        "median_log1p_predicted_effect": median_effect,
                        "ci_lower": ci_lower,
                        "ci_upper": ci_upper,
                        "p_value": p_values,
                    }
                )
                table["absolute_mean_effect"] = np.abs(table["mean_log1p_predicted_effect"])
                table["fdr_bh"] = benjamini_hochberg(table["p_value"].to_numpy())
                table["significant_fdr_0_05"] = table["fdr_bh"] < MODULE_4_FDR_THRESHOLD
                results_by_modality[modality].append(table)
    if not cell_rows:
        return
    pd.DataFrame(cell_rows).to_csv(module_dir / "cell_level_distance_band_summary.csv", index=False)
    for modality in ["rna", "atac"]:
        if not results_by_modality[modality]:
            continue
        results = pd.concat(results_by_modality[modality], ignore_index=True)
        results.to_csv(module_dir / f"all_distance_resolved_{modality}_effects.csv", index=False)
        results[results["significant_fdr_0_05"]].to_csv(
            module_dir / f"significant_distance_resolved_{modality}_effects.csv",
            index=False,
        )
        sig_counts = (
            results.groupby(["perturbation_guide", "distance_left_um", "distance_right_um"])["significant_fdr_0_05"]
            .sum()
            .reset_index()
        )
        sig_counts["band"] = sig_counts["distance_left_um"].map("{:g}".format) + "-" + sig_counts["distance_right_um"].map("{:g}".format) + " um"
        count_matrix = sig_counts.pivot_table(index="perturbation_guide", columns="band", values="significant_fdr_0_05", fill_value=0)
        matrix_plot(
            count_matrix,
            title=f"Module 4 {modality.upper()} significant features by distance band",
            colorbar_label="Significant features (FDR < 0.05)",
            outpath=plot_dir / f"module_4_{modality}_significant_feature_count_heatmap",
        )
        magnitude = (
            results.groupby(["perturbation_guide", "distance_left_um", "distance_right_um"])["absolute_mean_effect"]
            .mean()
            .reset_index()
        )
        magnitude["band"] = magnitude["distance_left_um"].map("{:g}".format) + "-" + magnitude["distance_right_um"].map("{:g}".format) + " um"
        magnitude_matrix = magnitude.pivot_table(index="perturbation_guide", columns="band", values="absolute_mean_effect", fill_value=0.0)
        matrix_plot(
            magnitude_matrix,
            title=f"Module 4 {modality.upper()} effect magnitude by distance band",
            colorbar_label="Mean absolute log1p effect",
            outpath=plot_dir / f"module_4_{modality}_magnitude_heatmap",
        )
        guide_summary = (
            results[results["significant_fdr_0_05"]]
            .groupby(["perturbation_guide", "distance_left_um", "distance_right_um"])["absolute_mean_effect"]
            .mean()
            .reset_index()
        )
        if not guide_summary.empty:
            fig, ax = plt.subplots(figsize=(6.5, max(4.0, 0.55 * guide_summary["perturbation_guide"].nunique())))
            for guide in sorted(guide_summary["perturbation_guide"].unique()):
                subset = guide_summary[guide_summary["perturbation_guide"] == guide].copy()
                midpoint = 0.5 * (subset["distance_left_um"] + subset["distance_right_um"])
                ax.plot(midpoint, subset["absolute_mean_effect"], marker="o", linewidth=1.4, label=guide)
            ax.set_xlabel("Distance band midpoint (um)")
            ax.set_ylabel("Mean absolute significant effect")
            ax.set_title(f"Module 4 {modality.upper()} guide-level distance profile")
            ax.legend(frameon=False, loc="best", ncol=1)
            save_plot(fig, plot_dir / f"module_4_{modality}_guide_distance_profile")
        top_pdf = plot_dir / f"module_4_{modality}_top_feature_curves.pdf"
        with PdfPages(top_pdf) as pdf:
            for guide in sorted(results["perturbation_guide"].unique()):
                guide_results = results[results["perturbation_guide"] == guide].copy()
                top_features = (
                    guide_results.groupby("feature")["absolute_mean_effect"]
                    .mean()
                    .sort_values(ascending=False)
                    .head(MODULE_4_TOP_FEATURES_PER_GUIDE)
                    .index
                )
                for feature in top_features:
                    feature_table = guide_results[guide_results["feature"] == feature].sort_values(["distance_left_um", "distance_right_um"])
                    midpoint = 0.5 * (feature_table["distance_left_um"].to_numpy() + feature_table["distance_right_um"].to_numpy())
                    fig, ax = plt.subplots(figsize=(4.6, 3.6))
                    ax.plot(midpoint, feature_table["mean_log1p_predicted_effect"], marker="o", color=COLOR_GAT, linewidth=1.5)
                    ax.fill_between(midpoint, feature_table["ci_lower"], feature_table["ci_upper"], color=COLOR_GAT, alpha=0.18)
                    ax.axhline(0.0, color="black", linewidth=0.7, linestyle="--")
                    ax.set_xlabel("Distance band midpoint (um)")
                    ax.set_ylabel("Mean log1p effect")
                    ax.set_title(f"{guide} | {feature}")
                    fig.tight_layout()
                    pdf.savefig(fig)
                    plt.close(fig)
def write_run_configuration(
    run_data: RunArtifacts,
    global_data: GlobalData,
    module_1_pairs: list[tuple[str, str]],
    module_3_pairs: list[tuple[str, str]],
    control_guide: str | None,
) -> None:
    payload = {
        "training_script_path": str(TRAINING_SCRIPT_PATH),
        "training_outdir": str(TRAINING_OUTDIR),
        "outdir": str(run_data.outdir),
        "data_path": str(DATA_PATH),
        "run": run_data.run,
        "validation_eb": run_data.validation_eb,
        "test_eb": run_data.test_eb,
        "n_cells": int(len(global_data.obs)),
        "n_guides": int(len(global_data.guide_names)),
        "n_cell_types": int(len(global_data.celltype_names)),
        "n_rna_targets": int(len(run_data.rna_feature_names)),
        "n_atac_targets": int(len(run_data.atac_feature_names)),
        "module_1_pairs": module_1_pairs,
        "module_3_pairs": module_3_pairs,
        "control_guide": control_guide,
        "central_cell_scope": CENTRAL_CELL_SCOPE,
        "selected_ebs": SELECTED_EBS,
        "selected_central_cell_types": SELECTED_CENTRAL_CELL_TYPES,
        "counterfactual_edit_mode": COUNTERFACTUAL_EDIT_MODE,
        "edit_central_cell": EDIT_CENTRAL_CELL,
        "distance_bands_um": DISTANCE_BANDS_UM,
        "auto_distance_band_count": AUTO_DISTANCE_BAND_COUNT,
        "auto_distance_band_min_width_um": AUTO_DISTANCE_BAND_MIN_WIDTH_UM,
        "run_module_1": RUN_MODULE_1,
        "run_module_2": RUN_MODULE_2,
        "run_module_3": RUN_MODULE_3,
        "run_module_4": RUN_MODULE_4,
    }
    (run_data.outdir / "counterfactual_inference_configuration.json").write_text(json.dumps(payload, indent=2))
def run_one_analysis(
    training,
    global_data: GlobalData,
    run_data: RunArtifacts,
    module_1_pairs: list[tuple[str, str]],
    module_3_pairs: list[tuple[str, str]],
    screen_guides: list[str],
    control_guide: str | None,
) -> None:
    run_data.outdir.mkdir(parents=True, exist_ok=True)
    print(f"\nRUN {run_data.run}: validation={run_data.validation_eb} test={run_data.test_eb}", flush=True)
    models = load_gat_models(training, global_data, run_data)
    baseline = baseline_predictions(training, global_data, run_data, models)
    np.save(run_data.outdir / "baseline_rna_mu.npy", baseline["rna"].astype(np.float32))
    np.save(run_data.outdir / "baseline_atac_mu.npy", baseline["atac"].astype(np.float32))
    eligible_cells = eligible_central_cells_for_run(global_data, run_data)
    print("Eligible central cells:", len(eligible_cells), flush=True)
    write_run_configuration(run_data, global_data, module_1_pairs, module_3_pairs, control_guide)
    if RUN_MODULE_1:
        print("  Module 1", flush=True)
        run_module_1(training, global_data, run_data, models, baseline, eligible_cells, module_1_pairs)
    if RUN_MODULE_2:
        print("  Module 2", flush=True)
        run_module_2(global_data, run_data, baseline, eligible_cells, module_1_pairs)
    if RUN_MODULE_3:
        print("  Module 3", flush=True)
        run_module_3(training, global_data, run_data, models, baseline, eligible_cells, module_3_pairs)
    if RUN_MODULE_4:
        print("  Module 4", flush=True)
        run_module_4(training, global_data, run_data, models, baseline, eligible_cells, screen_guides, control_guide)
    limitations = [
        "These are model-based conditional predictions rather than causal identification.",
        "Analyses are run-specific because modeled RNA genes and ATAC features differ across held-out-EB runs.",
        "RNA and ATAC branches use modality-specific learned graphs; a shared sender edit is applied to the union of branch-relevant neighbors.",
        "Module 4 distance bands are evaluated across the full empirical kNN neighbor-distance range of each held-out run rather than being truncated at 75 microns.",
    ]
    (run_data.outdir / "INTERPRETATION_AND_LIMITATIONS.txt").write_text("\n".join(f"- {line}" for line in limitations))
def main() -> None:
    configure_matplotlib()
    global training_module
    training_module = load_training_module(TRAINING_SCRIPT_PATH)
    if not TRAINING_OUTDIR.exists():
        raise FileNotFoundError(TRAINING_OUTDIR)
    OUTROOT.mkdir(parents=True, exist_ok=True)
    global_data = load_global_data(training_module)
    module_1_pairs, module_3_pairs, screen_guides, control_guide = choose_module_pairs(global_data)
    runs, _ = build_run_artifacts(training_module, global_data)
    if not runs:
        raise ValueError("No runs selected for analysis.")
    print("Device:", DEVICE, flush=True)
    print("Selected runs:", [run.run for run in runs], flush=True)
    print("Control guide:", control_guide, flush=True)
    print("Module 1 pairs:", len(module_1_pairs), flush=True)
    print("Module 3 ordered pairs:", len(module_3_pairs), flush=True)
    for run_data in runs:
        run_one_analysis(training_module, global_data, run_data, module_1_pairs, module_3_pairs, screen_guides, control_guide)
    print("\nOutputs:", OUTROOT.resolve(), flush=True)
if __name__ == "__main__":
    main()

# Figure 6e-g: variance-ranked guide-pair analysis
WORKDIR = Path("/gpfs/accounts/welchjd_root/welchjd1/javidgmh/15515_Multiome/catatac_work/motif_analysis")
TRAINING_SCRIPT_PATH = Path("/scratch/welchjd_root/welchjd1/javidgmh/15515_Multiome/catatac_work/motif_analysis/joint_multiome_MODALITY_SPECIFIC_SPATIAL_RESIDUAL_TUNE_LOSS_WEIGHTS_10EB.py")
TRAINING_OUTDIR = Path("multiome_joint_RNA_ATAC_MODALITY_SPECIFIC_SPATIAL_RESIDUAL_TUNE_LOSS_WEIGHTS_10EB")
OUTROOT = TRAINING_OUTDIR / "counterfactual_inference_publication"
DATA_PATH = Path("/scratch/welchjd_root/welchjd1/javidgmh/15515_Multiome/catatac_work/motif_analysis/all_lanes_merged_CATATAC_QC_RNA_processed_EB_manual_clustering_cell_types.h5mu")
MODULE3_DIRNAME = "module_3_guide_pair_screen_variance_ranked"
OVERALL_SUBDIR = "module_3_variance_ranked"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_SEED = 42
CENTRAL_CELL_SCOPE = "all"
MAX_CENTRAL_CELLS_MODULE_1 = 500
MAX_CENTRAL_CELLS_MODULE_3 = 250
MAX_CENTRAL_CELLS_MODULE_4 = 1000
MODULE_3_TOP_FEATURES_PER_PAIR = 100
N_AUTO_SCREEN_GUIDES = 10
SCREEN_GUIDES: list[str] | None = None
MODULE_3_GUIDE_PAIRS: list[tuple[str, str]] | None = None
CONTROL_GUIDE_PATTERNS = ("negative control","non-targeting","non targeting","nontargeting","ntc","control")
COUNTERFACTUAL_EDIT_MODE = "replace_source_with_target"
EDIT_CENTRAL_CELL = False
MIN_SOURCE_POSITIVE_NEIGHBORS = 1
DISTANCE_BANDS_UM: list[tuple[float,float]] | None = None
AUTO_DISTANCE_BAND_COUNT = 4
AUTO_DISTANCE_BAND_MIN_WIDTH_UM = 10.0
RADIUS_GUIDES: list[str] | None = None
HOUSEKEEPING_GENES = {"ACTB","ACTG1","B2M","EEF1A1","GAPDH","GUSB","HMBS","HPRT1","MALAT1","NONO","PGK1","PPIA","SDHA","TBP","TFRC","TUBA1A","TUBA1B","TUBB","TUBB4B","UBC","VCP","YWHAZ"}
EXCLUDED_PREFIXES = ("MT-","RPL","RPS","MRPL","MRPS")
rng = np.random.default_rng(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
@dataclass
class GlobalData:
    obs: pd.DataFrame
    coords: np.ndarray
    eb: np.ndarray
    cell_type: np.ndarray
    guide_names: list[str]
    guide_lookup: dict[str,int]
    guides_np: np.ndarray
    celltype_names: list[str]
    celltypes_np: np.ndarray
    x_np: np.ndarray
    x_tensor: torch.Tensor
    guides_tensor: torch.Tensor
    celltypes_tensor: torch.Tensor
    rna_counts: sp.csr_matrix
    atac_counts: sp.csr_matrix
    rna_library_all: np.ndarray
    atac_library_all: np.ndarray
@dataclass
class RunArtifacts:
    run: int
    validation_eb: str
    test_eb: str
    outdir: Path
    rna_feature_names: np.ndarray
    atac_feature_names: np.ndarray
    rna_feature_idx: np.ndarray
    atac_feature_idx: np.ndarray
    rna_mu0: np.ndarray
    atac_mu0: np.ndarray
    rna_log_library: np.ndarray
    atac_log_library: np.ndarray
    rna_bundle: tuple[torch.Tensor,torch.Tensor,torch.Tensor]
    atac_bundle: tuple[torch.Tensor,torch.Tensor,torch.Tensor]
    rna_neighbors: list[np.ndarray]
    rna_neighbor_distances: list[np.ndarray]
    atac_neighbors: list[np.ndarray]
    atac_neighbor_distances: list[np.ndarray]
    union_neighbors: list[np.ndarray]
    final_seeds: list[int]
def configure_matplotlib():
    mpl.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Helvetica","Arial","DejaVu Sans"],"axes.linewidth":0.8,"axes.spines.top":False,"axes.spines.right":False,"axes.labelsize":10,"axes.titlesize":11,"xtick.labelsize":9,"ytick.labelsize":9,"legend.fontsize":9,"pdf.fonttype":42,"ps.fonttype":42,"figure.facecolor":"white","axes.facecolor":"white","savefig.dpi":600})
def save_plot(fig,stem):
    fig.tight_layout()
    fig.savefig(stem.with_suffix(".png"),bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"),bbox_inches="tight")
    plt.close(fig)
def heatmap_from_long_table(table,row_col,feature_col,value_col,title,colorbar_label,outpath,top_features=60):
    if table.empty:
        return
    feature_scores = table.groupby(feature_col)[value_col].apply(lambda x:float(np.mean(np.abs(x)))).sort_values(ascending=False)
    keep_features = feature_scores.head(top_features).index
    matrix = table[table[feature_col].isin(keep_features)].pivot_table(index=row_col,columns=feature_col,values=value_col,aggfunc="mean").fillna(0.0)
    if matrix.empty:
        return
    vmax = max(float(np.abs(matrix.to_numpy()).max()),1e-6)
    fig,ax = plt.subplots(figsize=(max(8,0.28*matrix.shape[1]),max(4,0.35*matrix.shape[0])))
    image = ax.imshow(matrix.to_numpy(),aspect="auto",cmap="RdBu_r",vmin=-vmax,vmax=vmax)
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels(matrix.columns,rotation=90)
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels(matrix.index)
    ax.set_title(title)
    colorbar = fig.colorbar(image,ax=ax,fraction=0.046,pad=0.04)
    colorbar.set_label(colorbar_label)
    save_plot(fig,outpath)
def heatmap_from_variability_table(table,row_col,feature_col,value_col,variability_col,title,colorbar_label,outpath,top_features=60):
    if table.empty:
        return
    feature_scores = table.groupby(feature_col)[variability_col].mean().dropna().sort_values(ascending=False)
    keep_features = feature_scores.head(top_features).index
    matrix = table[table[feature_col].isin(keep_features)].pivot_table(index=row_col,columns=feature_col,values=value_col,aggfunc="mean").fillna(0.0)
    if matrix.empty:
        return
    ordered = [x for x in keep_features if x in matrix.columns]
    matrix = matrix.loc[:,ordered]
    vmax = max(float(np.abs(matrix.to_numpy()).max()),1e-6)
    fig,ax = plt.subplots(figsize=(max(8,0.28*matrix.shape[1]),max(4,0.35*matrix.shape[0])))
    image = ax.imshow(matrix.to_numpy(),aspect="auto",cmap="RdBu_r",vmin=-vmax,vmax=vmax)
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels(matrix.columns,rotation=90)
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels(matrix.index)
    ax.set_title(title)
    colorbar = fig.colorbar(image,ax=ax,fraction=0.046,pad=0.04)
    colorbar.set_label(colorbar_label)
    save_plot(fig,outpath)
def matrix_plot(matrix,title,colorbar_label,outpath,cmap="YlGnBu"):
    if matrix.empty:
        return
    fig,ax = plt.subplots(figsize=(max(5.5,0.55*matrix.shape[1]),max(4.5,0.45*matrix.shape[0])))
    image = ax.imshow(matrix.to_numpy(),aspect="auto",cmap=cmap)
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels(matrix.columns,rotation=45,ha="right")
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels(matrix.index)
    ax.set_title(title)
    colorbar = fig.colorbar(image,ax=ax,fraction=0.046,pad=0.04)
    colorbar.set_label(colorbar_label)
    save_plot(fig,outpath)
def load_training_module():
    spec = importlib.util.spec_from_file_location("joint_multiome_training_module",TRAINING_SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {TRAINING_SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
def parse_guides(value):
    if value is None or (isinstance(value,float) and np.isnan(value)):
        return []
    if isinstance(value,(list,tuple,np.ndarray)):
        values = value
    else:
        text = str(value).strip()
        if text == "" or text.lower() in {"nan","none","unassigned","no guide","no_guide"}:
            return []
        values = text.split(";")
    return [str(x).strip() for x in values if str(x).strip()]
def build_guides(obs,source_col):
    rows = [parse_guides(value) for value in obs[source_col].astype(object).to_numpy()]
    names = sorted({guide for row in rows for guide in row})
    lookup = {name:idx for idx,name in enumerate(names)}
    matrix = np.zeros((len(rows),len(names)),dtype=np.float32)
    for row_idx,row in enumerate(rows):
        for guide in row:
            matrix[row_idx,lookup[guide]] = 1.0
    return matrix,names
def valid_eb_label(value):
    text = str(value).strip()
    return text.startswith("EB_") and text.lower() not in {"eb_","eb_nan","eb_none","eb_unassigned"}
def safe_name(value):
    return re.sub(r"[^A-Za-z0-9._-]+","_",str(value)).strip("_")
def sample_indices(indices,maximum):
    indices = np.asarray(indices,dtype=int)
    if maximum is None or len(indices) <= maximum:
        return indices
    chosen = rng.choice(indices,size=maximum,replace=False)
    return np.sort(chosen.astype(int))
def is_control_guide(name):
    lowered = name.lower()
    return any(pattern in lowered for pattern in CONTROL_GUIDE_PATTERNS)
def load_global_data(training):
    mu.set_options(pull_on_update=False)
    mdata = mu.read_h5mu(DATA_PATH)
    rna = mdata.mod["rna"]
    atac = mdata.mod["atac"]
    if not np.array_equal(np.asarray(rna.obs_names.astype(str)),np.asarray(atac.obs_names.astype(str))) or not np.array_equal(np.asarray(rna.obs_names.astype(str)),np.asarray(mdata.obs_names.astype(str))):
        raise ValueError("RNA, ATAC and MuData cells are not in identical order")
    obs = mdata.obs.copy()
    coords = training.dense32(rna.obsm[training.SPATIAL_KEY])
    eb_all = obs[training.EB_KEY].astype(str).to_numpy()
    valid_mask = np.asarray([valid_eb_label(x) for x in eb_all],dtype=bool)
    obs = obs.iloc[np.flatnonzero(valid_mask)].copy()
    coords = coords[valid_mask]
    rna_counts = training.get_counts(rna,training.RNA_COUNTS_LAYER,"RNA",False)[valid_mask]
    atac_counts = training.get_counts(atac,training.ATAC_COUNTS_LAYER,"ATAC",True)[valid_mask]
    guides_np,guide_names = build_guides(obs,training.GUIDE_SOURCE_COL)
    celltype_df = pd.get_dummies(obs[training.CELL_TYPE_KEY].astype(str),dtype=np.float32)
    celltypes_np = celltype_df.to_numpy(np.float32)
    x_np = np.concatenate([guides_np,celltypes_np],axis=1).astype(np.float32)
    return GlobalData(obs=obs,coords=coords,eb=obs[training.EB_KEY].astype(str).to_numpy(),cell_type=obs[training.CELL_TYPE_KEY].astype(str).to_numpy(),guide_names=guide_names,guide_lookup={x:i for i,x in enumerate(guide_names)},guides_np=guides_np,celltype_names=list(map(str,celltype_df.columns)),celltypes_np=celltypes_np,x_np=x_np,x_tensor=torch.tensor(x_np,dtype=torch.float32,device=DEVICE),guides_tensor=torch.tensor(guides_np,dtype=torch.float32,device=DEVICE),celltypes_tensor=torch.tensor(celltypes_np,dtype=torch.float32,device=DEVICE),rna_counts=rna_counts,atac_counts=atac_counts,rna_library_all=np.asarray(rna_counts.sum(1)).ravel().astype(np.float32),atac_library_all=np.asarray(atac_counts.sum(1)).ravel().astype(np.float32))
def build_incoming_neighbor_lists(edge_index,edge_distance,n_cells):
    src = edge_index[0].detach().cpu().numpy().astype(int)
    dst = edge_index[1].detach().cpu().numpy().astype(int)
    dist = edge_distance.detach().cpu().numpy().reshape(-1).astype(np.float32)
    neighbors = [[] for _ in range(n_cells)]
    distances = [[] for _ in range(n_cells)]
    for s,d,value in zip(src,dst,dist,strict=False):
        neighbors[d].append(s)
        distances[d].append(value)
    return [np.asarray(x,dtype=int) for x in neighbors],[np.asarray(x,dtype=np.float32) for x in distances]
def make_union_neighbors(left,right):
    out = []
    for a,b in zip(left,right,strict=False):
        out.append(np.asarray([],dtype=int) if len(a)==0 and len(b)==0 else np.unique(np.concatenate([a,b]).astype(int)))
    return out
def load_run_table(path):
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)
def build_run_artifacts(training,global_data):
    config = json.loads((TRAINING_OUTDIR/"configuration.json").read_text())
    final_seeds = list(map(int,config["final_seeds"]))
    manifest = load_run_table(TRAINING_OUTDIR/"split_manifest.csv")
    graph_cache = {}
    def graph_bundle(k,decay):
        key = (int(k),float(decay))
        if key not in graph_cache:
            edge_cpu,dist_cpu = training.build_knn_graph(global_data.coords,global_data.eb,int(k))
            context_np = training.weighted_neighbor_context(global_data.guides_np,global_data.celltypes_np,edge_cpu,dist_cpu,float(decay))
            neighbors,distances = build_incoming_neighbor_lists(edge_cpu,dist_cpu,len(global_data.obs))
            graph_cache[key] = (edge_cpu.to(DEVICE),dist_cpu.to(DEVICE),torch.tensor(context_np,dtype=torch.float32,device=DEVICE),neighbors,distances)
        return graph_cache[key]
    runs = []
    for row in manifest.itertuples(index=False):
        run = int(row.run)
        selected = load_run_table(TRAINING_OUTDIR/f"run{run}_selected_modality_graphs.csv")
        lookup = {str(x.modality):(int(x.k_neighbors),float(x.distance_decay)) for x in selected.itertuples(index=False)}
        rna_k,rna_decay = lookup["rna"]
        atac_k,atac_decay = lookup["atac"]
        rna_bundle = graph_bundle(rna_k,rna_decay)
        atac_bundle = graph_bundle(atac_k,atac_decay)
        offsets = np.load(TRAINING_OUTDIR/f"run{run}_offsets_and_targets.npz")
        rna_targets = load_run_table(TRAINING_OUTDIR/f"run{run}_rna_hvgs.csv")
        atac_targets = load_run_table(TRAINING_OUTDIR/f"run{run}_atac_deviance_features.csv")
        runs.append(RunArtifacts(run=run,validation_eb=str(row.validation_eb),test_eb=str(row.test_eb),outdir=OUTROOT/f"run{run:02d}_{safe_name(str(row.test_eb))}",rna_feature_names=rna_targets["gene"].astype(str).to_numpy(),atac_feature_names=atac_targets["atac_feature"].astype(str).to_numpy(),rna_feature_idx=offsets["rna_target_idx"].astype(int),atac_feature_idx=offsets["atac_target_idx"].astype(int),rna_mu0=offsets["rna_mu0"].astype(np.float32),atac_mu0=offsets["atac_mu0"].astype(np.float32),rna_log_library=offsets["rna_log_library"].astype(np.float32),atac_log_library=offsets["atac_log_library"].astype(np.float32),rna_bundle=rna_bundle[:3],atac_bundle=atac_bundle[:3],rna_neighbors=rna_bundle[3],rna_neighbor_distances=rna_bundle[4],atac_neighbors=atac_bundle[3],atac_neighbor_distances=atac_bundle[4],union_neighbors=make_union_neighbors(rna_bundle[3],atac_bundle[3]),final_seeds=final_seeds))
    return runs
def eligible_central_cells(global_data,run_data):
    if CENTRAL_CELL_SCOPE == "all":
        return np.arange(len(global_data.obs),dtype=int)
    if CENTRAL_CELL_SCOPE == "test_eb":
        return np.flatnonzero(global_data.eb == run_data.test_eb)
    raise ValueError(CENTRAL_CELL_SCOPE)
def load_gat_models(training,global_data,run_data):
    selected = load_run_table(TRAINING_OUTDIR/f"run{run_data.run}_selected_modality_graphs.csv")
    rna_decay = float(selected.query("modality == 'rna'")["distance_decay"].iloc[0])
    atac_decay = float(selected.query("modality == 'atac'")["distance_decay"].iloc[0])
    context_input_dim = global_data.guides_np.shape[1]+global_data.celltypes_np.shape[1]+3
    models = []
    for seed in run_data.final_seeds:
        path = TRAINING_OUTDIR/f"run{run_data.run}_seed{seed}_gat_best.pt"
        model = training.MultiomicSpatialResidualGAT(global_data.x_tensor.shape[1],global_data.guides_np.shape[1],global_data.celltypes_np.shape[1],len(run_data.rna_feature_names),len(run_data.atac_feature_names),context_input_dim,rna_decay,atac_decay).to(DEVICE)
        model.load_state_dict(torch.load(path,map_location=DEVICE),strict=True)
        model.eval()
        models.append(model)
    return models
@torch.no_grad()
def baseline_predictions(training,global_data,run_data,models):
    rna_loglib = torch.tensor(run_data.rna_log_library,dtype=torch.float32,device=DEVICE)
    atac_loglib = torch.tensor(run_data.atac_log_library,dtype=torch.float32,device=DEVICE)
    rna_mu0 = torch.tensor(run_data.rna_mu0,dtype=torch.float32,device=DEVICE)
    atac_mu0 = torch.tensor(run_data.atac_mu0,dtype=torch.float32,device=DEVICE)
    rna_pred = []
    atac_pred = []
    for model in models:
        rna_residual,atac_residual = model(global_data.x_tensor,global_data.guides_tensor,global_data.celltypes_tensor,run_data.rna_bundle,run_data.atac_bundle)
        rna_pred.append(training.make_mu(rna_residual,rna_loglib,rna_mu0).detach().cpu().numpy())
        atac_pred.append(training.make_mu(atac_residual,atac_loglib,atac_mu0).detach().cpu().numpy())
    return {"rna":np.mean(np.stack(rna_pred),axis=0).astype(np.float32),"atac":np.mean(np.stack(atac_pred),axis=0).astype(np.float32)}
def compute_branch_context_from_local_neighbors(edited_guides,neighbor_indices,neighbor_distances,celltypes_np,decay):
    if len(neighbor_indices)==0:
        guide_mean = np.zeros(edited_guides.shape[1],dtype=np.float32)
        celltype_mean = np.zeros(celltypes_np.shape[1],dtype=np.float32)
        degree = mean_distance = min_distance = 0.0
    else:
        weights = np.exp(-neighbor_distances/float(decay)).astype(np.float32)
        weights_sum = float(np.clip(weights.sum(),1e-8,None))
        guide_mean = (edited_guides[neighbor_indices]*weights[:,None]).sum(axis=0)/weights_sum
        celltype_mean = (celltypes_np[neighbor_indices]*weights[:,None]).sum(axis=0)/weights_sum
        degree = float(len(neighbor_indices))
        mean_distance = float((neighbor_distances*weights).sum()/weights_sum)
        min_distance = float(neighbor_distances.min())
    context = np.concatenate([guide_mean.astype(np.float32),celltype_mean.astype(np.float32),np.asarray([math.log1p(degree),mean_distance/float(decay),min_distance/float(decay)],dtype=np.float32)])
    return torch.tensor(context[None,:],dtype=torch.float32,device=DEVICE)
def predict_branch_for_central_cell_local(training,branch,x_np,global_data,central_cell,neighbor_indices,neighbor_distances,edited_cells,source_index,target_index,log_library_value,mu0_np):
    local_indices = np.concatenate([[central_cell],neighbor_indices]).astype(int)
    local_x = x_np[local_indices].copy()
    if len(edited_cells):
        local_lookup = {global_idx:local_idx for local_idx,global_idx in enumerate(local_indices)}
        overlapping = [local_lookup[idx] for idx in edited_cells if idx in local_lookup]
        if overlapping:
            overlapping = np.asarray(overlapping,dtype=int)
            local_x[overlapping,source_index] = 0.0
            local_x[overlapping,target_index] = 1.0
    local_x_tensor = torch.tensor(local_x,dtype=torch.float32,device=DEVICE)
    local_guides_tensor = local_x_tensor[:,:len(global_data.guide_names)]
    local_celltypes_tensor = local_x_tensor[:,len(global_data.guide_names):]
    base = branch.base(local_x_tensor[0:1],local_guides_tensor[0:1],local_celltypes_tensor[0:1])
    receiver = branch.receiver_encoder(local_celltypes_tensor[0:1])
    if len(neighbor_indices)==0:
        spatial = torch.zeros((1,training.SPATIAL_DIM),dtype=torch.float32,device=DEVICE)
    else:
        source_tensor = branch.source_encoder(local_x_tensor[1:])
        scaled = torch.tensor(neighbor_distances[:,None],dtype=torch.float32,device=DEVICE)/branch.distance_decay
        edge_raw = torch.cat([scaled,torch.exp(-scaled),1.0/(1.0+scaled),torch.log1p(torch.tensor(neighbor_distances[:,None],dtype=torch.float32,device=DEVICE))/math.log1p(branch.distance_decay)],dim=1)
        edge_attr = branch.edge_encoder(edge_raw)
        local_edge_index = torch.stack([torch.arange(len(neighbor_indices),dtype=torch.long,device=DEVICE),torch.zeros(len(neighbor_indices),dtype=torch.long,device=DEVICE)],dim=0)
        spatial = F.elu(branch.gat((source_tensor,receiver),local_edge_index,edge_attr=edge_attr))
    edited_guides = local_x[:,:len(global_data.guide_names)].astype(np.float32)
    context = branch.context_encoder(compute_branch_context_from_local_neighbors(edited_guides,np.arange(1,len(local_indices),dtype=int),neighbor_distances,local_x[:,len(global_data.guide_names):].astype(np.float32),branch.distance_decay))
    gate = branch.gate(torch.cat([receiver,spatial,context],dim=1))
    correction = branch.correction_head(branch.correction_encoder(torch.cat([receiver,gate*spatial,context],dim=1)))
    residual = base+correction
    mu = torch.exp((residual+float(log_library_value)+torch.tensor(mu0_np[None,:],dtype=torch.float32,device=DEVICE)).clamp(-12,12))[0]
    return mu.detach().cpu().numpy().astype(np.float32)
def strict_receiver_ok(global_data,central_cell,source_index,target_index):
    return global_data.guides_np[central_cell,source_index] <= 0 and global_data.guides_np[central_cell,target_index] <= 0
def edited_cells_for_counterfactual(global_data,run_data,central_cell,source_index):
    candidate_cells = run_data.union_neighbors[central_cell]
    if EDIT_CENTRAL_CELL:
        candidate_cells = np.unique(np.concatenate([candidate_cells,np.asarray([central_cell],dtype=int)]))
    if len(candidate_cells)==0:
        return np.asarray([],dtype=int),0
    source_positive = global_data.guides_np[candidate_cells,source_index] > 0
    return candidate_cells[source_positive].astype(int),int(source_positive.sum())
def predict_counterfactual(training,global_data,run_data,models,baseline,central_cell,source_guide,target_guide):
    source_index = global_data.guide_lookup[source_guide]
    target_index = global_data.guide_lookup[target_guide]
    if not strict_receiver_ok(global_data,central_cell,source_index,target_index):
        return None
    edited_cells,n_source_positive = edited_cells_for_counterfactual(global_data,run_data,central_cell,source_index)
    if n_source_positive < MIN_SOURCE_POSITIVE_NEIGHBORS:
        return None
    rna_predictions = []
    atac_predictions = []
    for model in models:
        rna_predictions.append(predict_branch_for_central_cell_local(training,model.rna,global_data.x_np,global_data,central_cell,run_data.rna_neighbors[central_cell],run_data.rna_neighbor_distances[central_cell],edited_cells,source_index,target_index,run_data.rna_log_library[central_cell],run_data.rna_mu0))
        atac_predictions.append(predict_branch_for_central_cell_local(training,model.atac,global_data.x_np,global_data,central_cell,run_data.atac_neighbors[central_cell],run_data.atac_neighbor_distances[central_cell],edited_cells,source_index,target_index,run_data.atac_log_library[central_cell],run_data.atac_mu0))
    counterfactual_rna = np.mean(np.stack(rna_predictions),axis=0).astype(np.float32)
    counterfactual_atac = np.mean(np.stack(atac_predictions),axis=0).astype(np.float32)
    return {"rna_log1p_effect":(np.log1p(counterfactual_rna)-np.log1p(baseline["rna"][central_cell])).astype(np.float32),"atac_log1p_effect":(np.log1p(counterfactual_atac)-np.log1p(baseline["atac"][central_cell])).astype(np.float32)}
def choose_pairs(global_data):
    controls = [x for x in global_data.guide_names if is_control_guide(x)]
    control = sorted(controls,key=lambda x:float(global_data.guides_np[:,global_data.guide_lookup[x]].sum()),reverse=True)[0]
    counts = pd.Series(global_data.guides_np.sum(axis=0),index=global_data.guide_names).sort_values(ascending=False)
    if SCREEN_GUIDES is None:
        guides = [str(x) for x in counts.index if x != control and not is_control_guide(str(x))][:N_AUTO_SCREEN_GUIDES]
    else:
        guides = list(SCREEN_GUIDES)
    pairs = list(MODULE_3_GUIDE_PAIRS) if MODULE_3_GUIDE_PAIRS is not None else [(a,b) for a in guides for b in guides if a != b]
    module1_pairs = [(guide,control) for guide in guides]
    return module1_pairs,pairs,guides,control
def informative_rna_gene_mask(feature_names):
    keep = np.ones(len(feature_names),dtype=bool)
    for idx,gene in enumerate(feature_names):
        name = str(gene).upper()
        if name.startswith(EXCLUDED_PREFIXES) or name in HOUSEKEEPING_GENES:
            keep[idx] = False
    return keep
def mean_adjusted_variability_scores(global_data,run_data,cell_indices):
    cell_indices = np.asarray(cell_indices,dtype=int)
    x = global_data.rna_counts[cell_indices][:,run_data.rna_feature_idx]
    if not sp.issparse(x):
        x = sp.csr_matrix(x)
    x = x.astype(np.float64).tocsr(copy=True)
    library_sizes = np.asarray(global_data.rna_library_all[cell_indices],dtype=np.float64)
    x = sp.diags(1e4/np.maximum(library_sizes,1.0)).dot(x).tocsr()
    x.data = np.log1p(x.data)
    mean_expression = np.asarray(x.mean(axis=0)).ravel()
    variance = np.maximum(np.asarray(x.power(2).mean(axis=0)).ravel()-mean_expression**2,0.0)
    dispersion = np.divide(variance,mean_expression,out=np.zeros_like(variance),where=mean_expression>0)
    log_dispersion = np.log1p(dispersion)
    n_genes = len(mean_expression)
    n_bins = min(20,max(1,n_genes))
    mean_rank = pd.Series(mean_expression).rank(method="first")
    mean_bins = pd.qcut(mean_rank,q=n_bins,labels=False,duplicates="drop").to_numpy()
    score = np.zeros(n_genes,dtype=np.float64)
    for bin_value in np.unique(mean_bins):
        mask = mean_bins == bin_value
        values = log_dispersion[mask]
        if len(values)<=1:
            score[mask] = 0.0
            continue
        spread = float(np.std(values,ddof=1))
        score[mask] = 0.0 if not np.isfinite(spread) or spread<=0 else (values-float(np.mean(values)))/spread
    return score
def select_variable_rna_indices(feature_names,variability_score,top_n):
    valid = informative_rna_gene_mask(feature_names) & np.isfinite(variability_score)
    indices = np.flatnonzero(valid)
    if len(indices)==0:
        raise ValueError("No informative RNA genes remained after filtering")
    return indices[np.argsort(variability_score[indices])[::-1]][:min(top_n,len(indices))]
def top_feature_table(feature_names,effect_matrix,top_n,modality,source_guide,target_guide):
    mean_effect = effect_matrix.mean(axis=0)
    median_effect = np.median(effect_matrix,axis=0)
    absolute = np.abs(mean_effect)
    order = np.argsort(absolute)[::-1][:min(top_n,len(feature_names))]
    return pd.DataFrame({"modality":modality,"source_guide":source_guide,"target_guide":target_guide,"feature":feature_names[order].astype(str),"mean_log1p_effect":mean_effect[order],"median_log1p_effect":median_effect[order],"absolute_mean_effect":absolute[order]})
def variable_rna_feature_table(feature_names,effect_matrix,indices,variability_score,source_guide,target_guide):
    mean_effect = effect_matrix.mean(axis=0)
    median_effect = np.median(effect_matrix,axis=0)
    absolute = np.abs(mean_effect)
    return pd.DataFrame({"modality":"rna","source_guide":source_guide,"target_guide":target_guide,"feature":feature_names[indices].astype(str),"mean_log1p_effect":mean_effect[indices],"median_log1p_effect":median_effect[indices],"absolute_mean_effect":absolute[indices],"variability_score":variability_score[indices]})
def run_module3(training,global_data,run_data,models,baseline,eligible_cells,module3_pairs):
    module_dir = run_data.outdir/MODULE3_DIRNAME
    module_dir.mkdir(parents=True,exist_ok=True)
    variability_score = mean_adjusted_variability_scores(global_data,run_data,eligible_cells)
    selected_rna = select_variable_rna_indices(run_data.rna_feature_names,variability_score,MODULE_3_TOP_FEATURES_PER_PAIR)
    pd.DataFrame({"feature":run_data.rna_feature_names.astype(str),"variability_score":variability_score,"selected_top100":np.isin(np.arange(len(run_data.rna_feature_names)),selected_rna),"excluded_gene_class":~informative_rna_gene_mask(run_data.rna_feature_names)}).sort_values("variability_score",ascending=False).to_csv(module_dir/"rna_variability_ranking.csv",index=False)
    pair_rows = []
    top_rows = []
    for source_guide,target_guide in module3_pairs:
        rna_effects = []
        atac_effects = []
        for central_cell in sample_indices(eligible_cells,MAX_CENTRAL_CELLS_MODULE_3):
            result = predict_counterfactual(training,global_data,run_data,models,baseline,central_cell,source_guide,target_guide)
            if result is None:
                continue
            rna_effects.append(result["rna_log1p_effect"])
            atac_effects.append(result["atac_log1p_effect"])
        if not rna_effects:
            continue
        rna_matrix = np.stack(rna_effects,axis=0)
        atac_matrix = np.stack(atac_effects,axis=0)
        pair_rows.append({"source_guide":source_guide,"target_guide":target_guide,"n_receiver_cells":rna_matrix.shape[0],"mean_absolute_effect_rna":float(np.mean(np.abs(rna_matrix))),"max_absolute_effect_rna":float(np.max(np.abs(rna_matrix))),"mean_absolute_effect_atac":float(np.mean(np.abs(atac_matrix))),"max_absolute_effect_atac":float(np.max(np.abs(atac_matrix)))})
        top_rows.append(variable_rna_feature_table(run_data.rna_feature_names,rna_matrix,selected_rna,variability_score,source_guide,target_guide))
        top_rows.append(top_feature_table(run_data.atac_feature_names,atac_matrix,MODULE_3_TOP_FEATURES_PER_PAIR,"atac",source_guide,target_guide))
    if not pair_rows:
        return
    pair_df = pd.DataFrame(pair_rows)
    top_df = pd.concat(top_rows,ignore_index=True)
    pair_df.to_csv(module_dir/"guide_pair_global_summary.csv",index=False)
    top_df.to_csv(module_dir/"top_response_features_for_all_guide_pairs.csv",index=False)
    for modality,value_col in [("rna","mean_absolute_effect_rna"),("atac","mean_absolute_effect_atac")]:
        matrix = pair_df.pivot_table(index="source_guide",columns="target_guide",values=value_col,aggfunc="mean").fillna(0.0)
        matrix_plot(matrix,f"Module 3 {modality.upper()} guide-pair effect magnitude","Mean absolute log1p effect",module_dir/f"module_3_{modality}_magnitude_matrix")
        subset = top_df[top_df["modality"]==modality].copy()
        subset["pair"] = subset["source_guide"]+" -> "+subset["target_guide"]
        if modality=="rna":
            heatmap_from_variability_table(subset,"pair","feature","mean_log1p_effect","variability_score","Module 3 RNA guide-pair feature effects","Mean log1p counterfactual effect",module_dir/"module_3_rna_feature_heatmap",45)
        else:
            heatmap_from_long_table(subset,"pair","feature","mean_log1p_effect","Module 3 ATAC guide-pair feature effects","Mean log1p counterfactual effect",module_dir/"module_3_atac_feature_heatmap",45)
def infer_distance_bands(global_data,run_data):
    if DISTANCE_BANDS_UM is not None:
        return DISTANCE_BANDS_UM
    max_distance = 0.0
    for central_cell,neighbors in enumerate(run_data.union_neighbors):
        if len(neighbors):
            distances = np.linalg.norm(global_data.coords[neighbors,:2]-global_data.coords[central_cell,:2],axis=1)
            if len(distances):
                max_distance = max(max_distance,float(distances.max()))
    if max_distance<=0:
        return [(0.0,1.0)]
    band_count = max(1,int(AUTO_DISTANCE_BAND_COUNT))
    if max_distance < band_count*AUTO_DISTANCE_BAND_MIN_WIDTH_UM:
        band_count = max(1,int(math.ceil(max_distance/AUTO_DISTANCE_BAND_MIN_WIDTH_UM)))
    edges = np.linspace(0,max_distance,band_count+1)
    return [(float(edges[i]),float(edges[i+1])) for i in range(len(edges)-1)]
def advance_rng_before_module3(eligible_cells,module1_pairs):
    for _ in module1_pairs:
        sample_indices(eligible_cells,MAX_CENTRAL_CELLS_MODULE_1)
def advance_rng_after_module3(global_data,run_data,eligible_cells,screen_guides):
    radius_guides = screen_guides if RADIUS_GUIDES is None else RADIUS_GUIDES
    for _guide in radius_guides:
        for _band in infer_distance_bands(global_data,run_data):
            sample_indices(eligible_cells,MAX_CENTRAL_CELLS_MODULE_4)
def collect_module3_csv(run_dirs,filename):
    tables = []
    for run_dir in run_dirs:
        path = run_dir/MODULE3_DIRNAME/filename
        if not path.exists():
            continue
        table = pd.read_csv(path)
        config_path = run_dir/"counterfactual_inference_configuration.json"
        if config_path.exists():
            config = json.loads(config_path.read_text())
            table["run"] = int(config["run"])
            table["test_eb"] = str(config["test_eb"])
            table["validation_eb"] = str(config["validation_eb"])
        else:
            match = re.match(r"run(\d+)_",run_dir.name)
            table["run"] = int(match.group(1)) if match else -1
            table["test_eb"] = run_dir.name.split("_",1)[1] if "_" in run_dir.name else ""
        tables.append(table)
    return pd.concat(tables,ignore_index=True) if tables else pd.DataFrame()
def aggregate_module3(run_dirs):
    overall_dir = OUTROOT/"overall_across_runs"/OVERALL_SUBDIR
    overall_dir.mkdir(parents=True,exist_ok=True)
    pair_table = collect_module3_csv(run_dirs,"guide_pair_global_summary.csv")
    feature_table = collect_module3_csv(run_dirs,"top_response_features_for_all_guide_pairs.csv")
    if not pair_table.empty:
        pair_summary = pair_table.groupby(["source_guide","target_guide"],as_index=False).agg(mean_n_receiver_cells=("n_receiver_cells","mean"),mean_absolute_effect_rna=("mean_absolute_effect_rna","mean"),max_absolute_effect_rna=("max_absolute_effect_rna","mean"),mean_absolute_effect_atac=("mean_absolute_effect_atac","mean"),max_absolute_effect_atac=("max_absolute_effect_atac","mean"),n_runs_present=("run","nunique"),n_test_ebs_present=("test_eb","nunique"))
        pair_summary.to_csv(overall_dir/"module_3_overall_guide_pair_summary.csv",index=False)
        for modality,value_col in [("rna","mean_absolute_effect_rna"),("atac","mean_absolute_effect_atac")]:
            matrix = pair_summary.pivot_table(index="source_guide",columns="target_guide",values=value_col,aggfunc="mean").fillna(0.0)
            matrix_plot(matrix,f"Overall Module 3 {modality.upper()} guide-pair effect magnitude","Mean absolute log1p effect",overall_dir/f"module_3_overall_{modality}_magnitude_matrix")
    if feature_table.empty:
        return overall_dir
    if "variability_score" not in feature_table.columns:
        feature_table["variability_score"] = np.nan
    feature_summary = feature_table.groupby(["modality","source_guide","target_guide","feature"],as_index=False).agg(mean_log1p_effect=("mean_log1p_effect","mean"),median_log1p_effect=("median_log1p_effect","mean"),absolute_mean_effect=("absolute_mean_effect","mean"),variability_score=("variability_score","mean"),n_rows=("feature","size"),n_runs_present=("run","nunique"),n_test_ebs_present=("test_eb","nunique")).sort_values("absolute_mean_effect",ascending=False)
    feature_summary.to_csv(overall_dir/"module_3_overall_top_response_features.csv",index=False)
    rna = feature_summary[feature_summary["modality"]=="rna"].copy()
    if not rna.empty:
        rna = rna.loc[informative_rna_gene_mask(rna["feature"].astype(str).to_numpy())].copy()
        rna["pair"] = rna["source_guide"]+" -> "+rna["target_guide"]
        heatmap_from_variability_table(rna,"pair","feature","mean_log1p_effect","variability_score","Overall Module 3 RNA guide-pair feature effects","Mean log1p counterfactual effect",overall_dir/"module_3_overall_rna_feature_heatmap",60)
    atac = feature_summary[feature_summary["modality"]=="atac"].copy()
    if not atac.empty:
        atac["pair"] = atac["source_guide"]+" -> "+atac["target_guide"]
        heatmap_from_long_table(atac,"pair","feature","mean_log1p_effect","Overall Module 3 ATAC guide-pair feature effects","Mean log1p counterfactual effect",overall_dir/"module_3_overall_atac_feature_heatmap",60)
    return overall_dir
def main():
    global rng
    os.chdir(WORKDIR)
    configure_matplotlib()
    rng = np.random.default_rng(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    training = load_training_module()
    global_data = load_global_data(training)
    module1_pairs,module3_pairs,screen_guides,control_guide = choose_pairs(global_data)
    runs = build_run_artifacts(training,global_data)
    print("Device:",DEVICE,flush=True)
    print("Selected runs:",[x.run for x in runs],flush=True)
    print("Control guide:",control_guide,flush=True)
    print("Module 3 ordered pairs:",len(module3_pairs),flush=True)
    for run_data in runs:
        print(f"\nRUN {run_data.run}: validation={run_data.validation_eb} test={run_data.test_eb}",flush=True)
        eligible_cells = eligible_central_cells(global_data,run_data)
        print("Eligible central cells:",len(eligible_cells),flush=True)
        advance_rng_before_module3(eligible_cells,module1_pairs)
        models = load_gat_models(training,global_data,run_data)
        baseline = baseline_predictions(training,global_data,run_data,models)
        print("  Module 3 variance-ranked",flush=True)
        run_module3(training,global_data,run_data,models,baseline,eligible_cells,module3_pairs)
        advance_rng_after_module3(global_data,run_data,eligible_cells,screen_guides)
        del models,baseline
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    overall_dir = aggregate_module3([x.outdir for x in runs])
    print("\nOutputs:",overall_dir.resolve(),flush=True)
if __name__ == "__main__":
    main()

# Figure 6f: distance-band profiles across runs
ROOT=Path("/gpfs/accounts/welchjd_root/welchjd1/javidgmh/15515_Multiome/catatac_work/motif_analysis/multiome_joint_RNA_ATAC_MODALITY_SPECIFIC_SPATIAL_RESIDUAL_TUNE_LOSS_WEIGHTS_10EB/counterfactual_inference_publication")
OVERALL=ROOT/"overall_across_runs"
OUT=OVERALL/"module_4_band_diagnostics"
OUT.mkdir(parents=True,exist_ok=True)
MODDIR="module_4_distance_resolved_effect_radius"
RNA_EXCLUDE_PREFIXES=("MT-","RPL","RPS","MRPL","MRPS")
def run_number(path):
    m=re.search(r"run(\d+)",path.name)
    return int(m.group(1)) if m else -1
def run_metadata(path):
    config=path/"counterfactual_inference_configuration.json"
    run=run_number(path)
    test_eb=""
    validation_eb=""
    if config.exists():
        x=json.loads(config.read_text())
        run=int(x.get("run",run))
        test_eb=str(x.get("test_eb",""))
        validation_eb=str(x.get("validation_eb",""))
    return run,test_eb,validation_eb
def bool_column(x):
    if pd.api.types.is_bool_dtype(x):
        return x.fillna(False)
    return x.astype(str).str.lower().isin(["true","1","1.0","yes"])
def savefig(fig,name):
    fig.tight_layout()
    fig.savefig(OUT/f"{name}.png",dpi=300,bbox_inches="tight")
    fig.savefig(OUT/f"{name}.pdf",bbox_inches="tight")
    plt.close(fig)
RUN_DIRS=sorted([x for x in ROOT.iterdir() if x.is_dir() and re.match(r"run\d+",x.name)],key=run_number)
print("RUN DIRECTORIES:",len(RUN_DIRS))
print([x.name for x in RUN_DIRS])
band_tables=[]
for rd in RUN_DIRS:
    path=rd/MODDIR/"distance_bands_used.csv"
    if not path.exists():
        print("MISSING:",path)
        continue
    run,test_eb,validation_eb=run_metadata(rd)
    x=pd.read_csv(path)
    x["run"]=run
    x["test_eb"]=test_eb
    x["validation_eb"]=validation_eb
    x["midpoint_um"]=0.5*(x["distance_left_um"]+x["distance_right_um"])
    x["width_um"]=x["distance_right_um"]-x["distance_left_um"]
    x["band_key"]=x["distance_left_um"].round(6).astype(str)+"_"+x["distance_right_um"].round(6).astype(str)
    band_tables.append(x)
bands=pd.concat(band_tables,ignore_index=True)
bands=bands.sort_values(["run","band_index"])
bands.to_csv(OUT/"01_distance_bands_by_run.csv",index=False)
print("\n================ DISTANCE BANDS USED IN EACH RUN ================")
print(bands[["run","test_eb","band_index","distance_left_um","distance_right_um","midpoint_um","width_um"]].to_string(index=False))
coverage=bands.groupby(["distance_left_um","distance_right_um"],as_index=False).agg(n_runs=("run","nunique"),runs=("run",lambda x:",".join(map(str,sorted(set(x))))))
coverage["midpoint_um"]=0.5*(coverage["distance_left_um"]+coverage["distance_right_um"])
coverage=coverage.sort_values("midpoint_um")
coverage.to_csv(OUT/"02_exact_distance_band_run_coverage.csv",index=False)
print("\n================ EXACT BAND COVERAGE ACROSS RUNS ================")
print(coverage.to_string(index=False))
print("\nNumber of unique exact band definitions:",len(coverage))
print("Distribution of number of runs contributing to an exact band:")
print(coverage["n_runs"].value_counts().sort_index().to_string())
fig,ax=plt.subplots(figsize=(7,5))
for band_index,sub in bands.groupby("band_index"):
    ax.scatter(sub["midpoint_um"],sub["run"],s=45,label=f"Band {band_index}")
ax.set_xlabel("Distance-band midpoint (µm)")
ax.set_ylabel("Run")
ax.set_yticks(sorted(bands["run"].unique()))
ax.set_title("Module 4 distance-band midpoints used in each run")
ax.legend(frameon=False)
savefig(fig,"01_band_midpoints_by_run")
def load_modality(modality):
    rows=[]
    for rd in RUN_DIRS:
        path=rd/MODDIR/f"all_distance_resolved_{modality}_effects.csv"
        if not path.exists():
            continue
        run,test_eb,validation_eb=run_metadata(rd)
        x=pd.read_csv(path)
        x["run"]=run
        x["test_eb"]=test_eb
        x["validation_eb"]=validation_eb
        if modality=="rna":
            feature_upper=x["feature"].astype(str).str.upper()
            keep=~feature_upper.str.startswith(RNA_EXCLUDE_PREFIXES)
            x=x.loc[keep].copy()
        x["significant"]=bool_column(x["significant_fdr_0_05"])
        x["significant_absolute_effect"]=x["absolute_mean_effect"].where(x["significant"])
        x["band_key"]=x["distance_left_um"].round(6).astype(str)+"_"+x["distance_right_um"].round(6).astype(str)
        rows.append(x)
    x=pd.concat(rows,ignore_index=True)
    band_lookup=bands[["run","band_key","band_index","midpoint_um","width_um"]].drop_duplicates()
    x=x.merge(band_lookup,on=["run","band_key"],how="left",validate="many_to_one")
    return x
def make_profiles(modality):
    x=load_modality(modality)
    x.to_csv(OUT/f"03_{modality}_all_per_run_feature_effects.csv",index=False)
    profile=x.groupby(["run","test_eb","perturbation_guide","band_index","distance_left_um","distance_right_um","midpoint_um"],as_index=False).agg(mean_absolute_effect_all_features=("absolute_mean_effect","mean"),mean_absolute_effect_significant_only=("significant_absolute_effect","mean"),n_features=("feature","nunique"),n_significant_features=("significant","sum"),n_receiver_cells=("n_receiver_cells","max"))
    profile.to_csv(OUT/f"04_{modality}_per_run_guide_band_profiles.csv",index=False)
    run_band=profile.groupby(["run","test_eb","band_index","distance_left_um","distance_right_um","midpoint_um"],as_index=False).agg(mean_absolute_effect_all_features=("mean_absolute_effect_all_features","mean"),mean_absolute_effect_significant_only=("mean_absolute_effect_significant_only","mean"),mean_significant_feature_count=("n_significant_features","mean"),mean_receiver_cells=("n_receiver_cells","mean"))
    run_band.to_csv(OUT/f"05_{modality}_per_run_band_summary.csv",index=False)
    fig,ax=plt.subplots(figsize=(7,5))
    for run,sub in run_band.groupby("run"):
        sub=sub.sort_values("midpoint_um")
        ax.plot(sub["midpoint_um"],sub["mean_absolute_effect_all_features"],marker="o",linewidth=1.2,label=f"Run {run}")
    ax.set_xlabel("Distance-band midpoint (µm)")
    ax.set_ylabel("Mean absolute effect")
    ax.set_title(f"{modality.upper()}: each run separately\nALL features")
    ax.legend(frameon=False,ncol=2,fontsize=8)
    savefig(fig,f"02_{modality}_per_run_ALL_features")
    fig,ax=plt.subplots(figsize=(7,5))
    for run,sub in run_band.groupby("run"):
        sub=sub.sort_values("midpoint_um")
        ax.plot(sub["midpoint_um"],sub["mean_absolute_effect_significant_only"],marker="o",linewidth=1.2,label=f"Run {run}")
    ax.set_xlabel("Distance-band midpoint (µm)")
    ax.set_ylabel("Mean absolute significant effect")
    ax.set_title(f"{modality.upper()}: each run separately\nSIGNIFICANT features only")
    ax.legend(frameon=False,ncol=2,fontsize=8)
    savefig(fig,f"03_{modality}_per_run_SIGNIFICANT_features")
    band_index_summary=profile.groupby("band_index",as_index=False).agg(mean_all=("mean_absolute_effect_all_features","mean"),sd_all=("mean_absolute_effect_all_features","std"),n_all=("mean_absolute_effect_all_features","count"),mean_significant=("mean_absolute_effect_significant_only","mean"),sd_significant=("mean_absolute_effect_significant_only","std"),n_significant=("mean_absolute_effect_significant_only","count"),mean_receiver_cells=("n_receiver_cells","mean"),mean_significant_feature_count=("n_significant_features","mean"))
    band_index_summary["sem_all"]=band_index_summary["sd_all"]/np.sqrt(band_index_summary["n_all"].clip(lower=1))
    band_index_summary["sem_significant"]=band_index_summary["sd_significant"]/np.sqrt(band_index_summary["n_significant"].clip(lower=1))
    band_index_summary.to_csv(OUT/f"06_{modality}_band_index_summary.csv",index=False)
    fig,ax=plt.subplots(figsize=(6,4.5))
    ax.errorbar(band_index_summary["band_index"],band_index_summary["mean_all"],yerr=band_index_summary["sem_all"],marker="o",linewidth=1.5,capsize=3,label="All features")
    ax.errorbar(band_index_summary["band_index"],band_index_summary["mean_significant"],yerr=band_index_summary["sem_significant"],marker="o",linewidth=1.5,capsize=3,label="Significant only")
    ax.set_xticks(sorted(band_index_summary["band_index"].unique()))
    ax.set_xlabel("Band index within run")
    ax.set_ylabel("Mean absolute effect")
    ax.set_title(f"{modality.upper()}: effect by band index across runs")
    ax.legend(frameon=False)
    savefig(fig,f"04_{modality}_effect_by_band_index")
    return x,profile,run_band,band_index_summary
rna,rna_profile,rna_run_band,rna_band_summary=make_profiles("rna")
atac,atac_profile,atac_run_band,atac_band_summary=make_profiles("atac")
cell_tables=[]
for rd in RUN_DIRS:
    path=rd/MODDIR/"cell_level_distance_band_summary.csv"
    if not path.exists():
        continue
    run,test_eb,validation_eb=run_metadata(rd)
    x=pd.read_csv(path)
    x["run"]=run
    x["test_eb"]=test_eb
    x["band_key"]=x["distance_left_um"].round(6).astype(str)+"_"+x["distance_right_um"].round(6).astype(str)
    cell_tables.append(x)
cells=pd.concat(cell_tables,ignore_index=True)
band_lookup=bands[["run","band_key","band_index","midpoint_um"]].drop_duplicates()
cells=cells.merge(band_lookup,on=["run","band_key"],how="left",validate="many_to_one")
cell_profile=cells.groupby(["run","test_eb","perturbation_guide","band_index","distance_left_um","distance_right_um","midpoint_um"],as_index=False).agg(n_receiver_rows=("central_cell_index","size"),n_unique_receiver_cells=("central_cell_index","nunique"),mean_source_positive_neighbors=("n_source_positive_neighbors","mean"),median_source_positive_neighbors=("n_source_positive_neighbors","median"),mean_edited_cells=("n_edited_cells","mean"),mean_absolute_rna_effect_cell_level=("mean_absolute_rna_effect","mean"),mean_absolute_atac_effect_cell_level=("mean_absolute_atac_effect","mean"))
cell_profile.to_csv(OUT/"07_cell_level_band_support_by_run_and_guide.csv",index=False)
support=cell_profile.groupby("band_index",as_index=False).agg(mean_receiver_cells=("n_unique_receiver_cells","mean"),mean_source_positive_neighbors=("mean_source_positive_neighbors","mean"),mean_edited_cells=("mean_edited_cells","mean"),mean_absolute_rna_effect=("mean_absolute_rna_effect_cell_level","mean"),mean_absolute_atac_effect=("mean_absolute_atac_effect_cell_level","mean"))
support.to_csv(OUT/"08_cell_level_support_by_band_index.csv",index=False)
fig,ax=plt.subplots(figsize=(6,4.5))
ax.plot(support["band_index"],support["mean_receiver_cells"],marker="o",label="Receiver cells")
ax.plot(support["band_index"],support["mean_source_positive_neighbors"],marker="o",label="Source-positive neighbors")
ax.set_xticks(sorted(support["band_index"].unique()))
ax.set_xlabel("Band index within run")
ax.set_ylabel("Mean count")
ax.set_title("Module 4 support by distance-band index")
ax.legend(frameon=False)
savefig(fig,"05_support_by_band_index")
def inspect_current_overall_plot(modality):
    if modality=="rna":
        path=OVERALL/"module_4_overall_rna_effects_no_mito_ribo.csv"
        if not path.exists():
            path=OVERALL/"module_4_overall_rna_effects.csv"
    else:
        path=OVERALL/"module_4_overall_atac_effects.csv"
    if not path.exists():
        print("Could not find:",path)
        return pd.DataFrame()
    x=pd.read_csv(path)
    x["significant"]=bool_column(x["significant_fdr_0_05"])
    x=x[x["significant"]].copy()
    summary=x.groupby(["perturbation_guide","distance_left_um","distance_right_um"],as_index=False).agg(mean_absolute_significant_effect=("absolute_mean_effect","mean"),n_significant_features=("feature","nunique"),minimum_runs_per_feature=("n_runs_present","min"),median_runs_per_feature=("n_runs_present","median"),maximum_runs_per_feature=("n_runs_present","max"))
    summary["midpoint_um"]=0.5*(summary["distance_left_um"]+summary["distance_right_um"])
    summary.to_csv(OUT/f"09_{modality}_CURRENT_plot_point_diagnostics.csv",index=False)
    near=summary[(summary["midpoint_um"]>=175)&(summary["midpoint_um"]<=225)].sort_values(["midpoint_um","perturbation_guide"])
    print(f"\n================ {modality.upper()} CURRENT PLOT POINTS 175-225 µm ================")
    print(near.to_string(index=False))
    return summary
rna_current=inspect_current_overall_plot("rna")
atac_current=inspect_current_overall_plot("atac")
print("\n================ BAND-INDEX RNA SUMMARY ================")
print(rna_band_summary.to_string(index=False))
print("\n================ BAND-INDEX ATAC SUMMARY ================")
print(atac_band_summary.to_string(index=False))
print("\n================ CELL/EXPOSURE SUPPORT ================")
print(support.to_string(index=False))
print("\nDONE")
print("Diagnostic outputs:",OUT)

# Figure 6f: common-band RNA and ATAC distance profiles
ROOT=Path("/gpfs/accounts/welchjd_root/welchjd1/javidgmh/15515_Multiome/catatac_work/motif_analysis/multiome_joint_RNA_ATAC_MODALITY_SPECIFIC_SPATIAL_RESIDUAL_TUNE_LOSS_WEIGHTS_10EB/counterfactual_inference_publication")
DIAG=ROOT/"overall_across_runs"/"module_4_band_diagnostics"
OUT=ROOT/"overall_across_runs"/"module_4_common_bands_VARIANCE_SELECTED_RNA"
OUT.mkdir(parents=True,exist_ok=True)
MODULE3_DIR="module_3_guide_pair_screen_variance_ranked"
MODULE4_DIR="module_4_distance_resolved_effect_radius"
ATAC_PROFILE_FILE=DIAG/"04_atac_per_run_guide_band_profiles.csv"
MIN_RUNS_TO_PLOT=2
def configure_plotting():
    plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","Helvetica","DejaVu Sans"],"font.size":9,"axes.labelsize":10,"axes.titlesize":11,"xtick.labelsize":9,"ytick.labelsize":9,"legend.fontsize":8,"axes.linewidth":0.8,"axes.spines.top":False,"axes.spines.right":False,"pdf.fonttype":42,"ps.fonttype":42,"figure.facecolor":"white","axes.facecolor":"white"})
def savefig(fig,name):
    fig.tight_layout()
    fig.savefig(OUT/f"{name}.png",dpi=600,bbox_inches="tight")
    fig.savefig(OUT/f"{name}.pdf",bbox_inches="tight")
    plt.close(fig)
def run_number(path):
    m=re.match(r"run(\d+)_",path.name)
    return int(m.group(1)) if m else -1
def bool_column(series):
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.lower().isin(["true","1","1.0","yes"])
def find_runs():
    return sorted([x for x in ROOT.iterdir() if x.is_dir() and re.match(r"run\d+_",x.name)],key=run_number)
def collect_band_metadata(run_dirs):
    rows=[]
    for rd in run_dirs:
        path=rd/MODULE4_DIR/"distance_bands_used.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        x=pd.read_csv(path)
        x["run"]=run_number(rd)
        x["midpoint_um"]=0.5*(x["distance_left_um"]+x["distance_right_um"])
        rows.append(x)
    bands=pd.concat(rows,ignore_index=True)
    bands.to_csv(OUT/"distance_bands_by_run.csv",index=False)
    return bands
def make_band_labels(bands):
    labels={}
    for band,g in bands.groupby("band_index"):
        midpoint=float(g["midpoint_um"].mean())
        labels[int(band)]={"band_index":int(band),"mean_midpoint_um":midpoint,"min_midpoint_um":float(g["midpoint_um"].min()),"max_midpoint_um":float(g["midpoint_um"].max()),"label":f"Band {int(band)}\n~{midpoint:.0f} µm"}
    return labels
def load_module3_selected_genes(run_dir):
    top_path=run_dir/MODULE3_DIR/"top_response_features_for_all_guide_pairs.csv"
    if top_path.exists():
        table=pd.read_csv(top_path)
        if "modality" not in table.columns or "feature" not in table.columns:
            raise ValueError(f"Unexpected columns in {top_path}")
        genes=sorted(table.loc[table["modality"].astype(str).str.lower()=="rna","feature"].astype(str).unique())
        if genes:
            return genes,"Module 3 variance-ranked feature table"
    ranking_path=run_dir/MODULE3_DIR/"rna_variability_ranking.csv"
    if ranking_path.exists():
        table=pd.read_csv(ranking_path)
        selected_col=None
        for candidate in ["selected_top100","selected"]:
            if candidate in table.columns:
                selected_col=candidate
                break
        if selected_col is not None and "feature" in table.columns:
            selected=bool_column(table[selected_col])
            genes=sorted(table.loc[selected,"feature"].astype(str).unique())
            if genes:
                return genes,"Module 3 RNA variability ranking"
    raise FileNotFoundError(f"Could not find the variance-selected Module 3 RNA genes for {run_dir.name}. Expected {top_path} or {ranking_path}")
def attach_band_index(table,band_path):
    bands=pd.read_csv(band_path)
    left=table["distance_left_um"].round(6)
    right=table["distance_right_um"].round(6)
    table=table.copy()
    table["_left"]=left
    table["_right"]=right
    bands=bands.copy()
    bands["_left"]=bands["distance_left_um"].round(6)
    bands["_right"]=bands["distance_right_um"].round(6)
    table=table.merge(bands[["_left","_right","band_index"]].drop_duplicates(),on=["_left","_right"],how="left",validate="many_to_one")
    if table["band_index"].isna().any():
        bad=table.loc[table["band_index"].isna(),["distance_left_um","distance_right_um"]].drop_duplicates()
        raise ValueError(f"Could not assign band index:\n{bad}")
    table["band_index"]=table["band_index"].astype(int)
    table["midpoint_um"]=0.5*(table["distance_left_um"]+table["distance_right_um"])
    return table.drop(columns=["_left","_right"])
def make_rna_run_profile(run_dir):
    run=run_number(run_dir)
    effects_path=run_dir/MODULE4_DIR/"all_distance_resolved_rna_effects.csv"
    band_path=run_dir/MODULE4_DIR/"distance_bands_used.csv"
    if not effects_path.exists():
        raise FileNotFoundError(effects_path)
    selected_genes,selection_source=load_module3_selected_genes(run_dir)
    selected_set=set(selected_genes)
    effects=pd.read_csv(effects_path)
    effects["feature"]=effects["feature"].astype(str)
    present=set(effects["feature"].unique())
    missing=sorted(selected_set-present)
    matched=selected_set&present
    if not matched:
        raise ValueError(f"Run {run}: none of the Module 3 variance-selected RNA genes are present in Module 4")
    if missing:
        print(f"WARNING run {run}: {len(missing)} selected genes missing from Module 4")
    effects=effects[effects["feature"].isin(matched)].copy()
    effects=attach_band_index(effects,band_path)
    effects["significant"]=bool_column(effects["significant_fdr_0_05"])
    effects["significant_absolute_effect"]=effects["absolute_mean_effect"].where(effects["significant"])
    profile=effects.groupby(["perturbation_guide","band_index","distance_left_um","distance_right_um","midpoint_um"],as_index=False).agg(mean_absolute_effect_variance_selected=("absolute_mean_effect","mean"),mean_absolute_effect_significant_variance_selected=("significant_absolute_effect","mean"),n_variance_selected_genes=("feature","nunique"),n_significant_variance_selected_genes=("significant","sum"),n_receiver_cells=("n_receiver_cells","max"))
    profile["run"]=run
    profile["test_eb"]=run_dir.name.split("_",1)[1] if "_" in run_dir.name else ""
    selection=pd.DataFrame({"run":run,"gene":sorted(matched),"selection_source":selection_source})
    return profile,selection
def build_rna_profiles(run_dirs):
    profiles=[]
    selections=[]
    for rd in run_dirs:
        profile,selection=make_rna_run_profile(rd)
        profiles.append(profile)
        selections.append(selection)
        print(f"{rd.name}: {selection['gene'].nunique()} variance-selected RNA genes",flush=True)
    profiles=pd.concat(profiles,ignore_index=True)
    selections=pd.concat(selections,ignore_index=True)
    profiles.to_csv(OUT/"rna_per_run_guide_band_profiles_VARIANCE_SELECTED.csv",index=False)
    selections.to_csv(OUT/"rna_variance_selected_genes_by_run.csv",index=False)
    return profiles
def summarize_rna_common_bands(profile):
    rows=[]
    for (guide,band),g in profile.groupby(["perturbation_guide","band_index"]):
        all_values=g["mean_absolute_effect_variance_selected"].dropna().to_numpy(float)
        sig_values=g["mean_absolute_effect_significant_variance_selected"].dropna().to_numpy(float)
        all_sd=float(np.std(all_values,ddof=1)) if len(all_values)>1 else np.nan
        sig_sd=float(np.std(sig_values,ddof=1)) if len(sig_values)>1 else np.nan
        rows.append({"modality":"rna","perturbation_guide":str(guide),"band_index":int(band),"n_runs_all_selected":len(all_values),"n_runs_significant_selected":len(sig_values),"runs_present":",".join(map(str,sorted(g["run"].unique()))),"mean_midpoint_um":float(g["midpoint_um"].mean()),"mean_receiver_cells":float(g["n_receiver_cells"].mean()),"mean_selected_gene_count":float(g["n_variance_selected_genes"].mean()),"mean_significant_selected_gene_count":float(g["n_significant_variance_selected_genes"].mean()),"mean_variance_selected":float(np.mean(all_values)) if len(all_values) else np.nan,"sd_variance_selected":all_sd,"sem_variance_selected":all_sd/np.sqrt(len(all_values)) if len(all_values)>1 else np.nan,"mean_significant_variance_selected":float(np.mean(sig_values)) if len(sig_values) else np.nan,"sd_significant_variance_selected":sig_sd,"sem_significant_variance_selected":sig_sd/np.sqrt(len(sig_values)) if len(sig_values)>1 else np.nan})
    summary=pd.DataFrame(rows).sort_values(["perturbation_guide","band_index"]).reset_index(drop=True)
    summary.to_csv(OUT/"rna_corrected_common_band_guide_summary_VARIANCE_SELECTED.csv",index=False)
    return summary
def summarize_atac_common_bands():
    if not ATAC_PROFILE_FILE.exists():
        print("ATAC diagnostic profile not found; skipping ATAC:",ATAC_PROFILE_FILE)
        return pd.DataFrame()
    df=pd.read_csv(ATAC_PROFILE_FILE)
    rows=[]
    for (guide,band),g in df.groupby(["perturbation_guide","band_index"]):
        all_values=g["mean_absolute_effect_all_features"].dropna().to_numpy(float)
        sig_values=g["mean_absolute_effect_significant_only"].dropna().to_numpy(float)
        all_sd=float(np.std(all_values,ddof=1)) if len(all_values)>1 else np.nan
        sig_sd=float(np.std(sig_values,ddof=1)) if len(sig_values)>1 else np.nan
        rows.append({"modality":"atac","perturbation_guide":str(guide),"band_index":int(band),"n_runs_all":len(all_values),"n_runs_significant":len(sig_values),"runs_present":",".join(map(str,sorted(g["run"].unique()))),"mean_midpoint_um":float(g["midpoint_um"].mean()),"mean_receiver_cells":float(g["n_receiver_cells"].mean()),"mean_all_features":float(np.mean(all_values)) if len(all_values) else np.nan,"sem_all_features":all_sd/np.sqrt(len(all_values)) if len(all_values)>1 else np.nan,"mean_significant_only":float(np.mean(sig_values)) if len(sig_values) else np.nan,"sem_significant_only":sig_sd/np.sqrt(len(sig_values)) if len(sig_values)>1 else np.nan})
    summary=pd.DataFrame(rows).sort_values(["perturbation_guide","band_index"]).reset_index(drop=True)
    summary.to_csv(OUT/"atac_corrected_common_band_guide_summary.csv",index=False)
    return summary
def plot_profiles(summary,value_col,sem_col,n_col,band_labels,title,ylabel,filename,show_sem=False):
    fig,ax=plt.subplots(figsize=(7.0,5.2))
    for guide in sorted(summary["perturbation_guide"].unique()):
        sub=summary[summary["perturbation_guide"]==guide].sort_values("band_index").copy()
        sub=sub[sub[n_col]>=MIN_RUNS_TO_PLOT]
        if sub.empty:
            continue
        x=sub["band_index"].to_numpy(float)
        y=sub[value_col].to_numpy(float)
        ax.plot(x,y,marker="o",linewidth=1.4,markersize=4,label=guide)
        if show_sem:
            sem=sub[sem_col].fillna(0).to_numpy(float)
            ax.fill_between(x,y-sem,y+sem,alpha=0.12)
    used=sorted(summary["band_index"].unique())
    ax.set_xticks(used)
    ax.set_xticklabels([band_labels[int(x)]["label"] for x in used])
    ax.set_xlabel("Common within-run distance band")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False,loc="best",ncol=1)
    savefig(fig,filename)
def plot_run_coverage(summary,n_col,band_labels,title,filename):
    matrix=summary.pivot_table(index="perturbation_guide",columns="band_index",values=n_col,aggfunc="max",fill_value=0)
    matrix=matrix.reindex(columns=sorted(matrix.columns))
    fig,ax=plt.subplots(figsize=(5.5,max(4.0,0.4*matrix.shape[0])))
    im=ax.imshow(matrix.to_numpy(),aspect="auto",cmap="viridis",vmin=0,vmax=max(10,int(matrix.to_numpy().max())))
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels([band_labels[int(x)]["label"] for x in matrix.columns])
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels(matrix.index)
    ax.set_xlabel("Common within-run distance band")
    ax.set_ylabel("Perturbation guide")
    ax.set_title(title)
    cbar=fig.colorbar(im,ax=ax,fraction=0.046,pad=0.04)
    cbar.set_label("Number of contributing runs")
    savefig(fig,filename)
def main():
    configure_plotting()
    run_dirs=find_runs()
    if not run_dirs:
        raise ValueError(f"No run directories found under {ROOT}")
    print("Runs:",[x.name for x in run_dirs],flush=True)
    bands=collect_band_metadata(run_dirs)
    band_labels=make_band_labels(bands)
    pd.DataFrame(list(band_labels.values())).sort_values("band_index").to_csv(OUT/"common_band_definitions.csv",index=False)
    print("\nCommon bands:",flush=True)
    for band in sorted(band_labels):
        x=band_labels[band]
        print(f"Band {band}: mean midpoint ~{x['mean_midpoint_um']:.1f} µm",flush=True)
    print("\nBuilding RNA profiles using EXACT Module 3 variance-selected genes...",flush=True)
    rna_profile=build_rna_profiles(run_dirs)
    rna=summarize_rna_common_bands(rna_profile)
    plot_profiles(rna,"mean_variance_selected","sem_variance_selected","n_runs_all_selected",band_labels,"Overall Module 4 RNA guide-level distance profile\nvariance-selected RNA genes; common band index; equal weight per run","Mean absolute effect","module_4_overall_rna_guide_distance_profile_COMMON_BANDS_VARIANCE_SELECTED",False)
    plot_profiles(rna,"mean_variance_selected","sem_variance_selected","n_runs_all_selected",band_labels,"Overall Module 4 RNA guide-level distance profile\nvariance-selected RNA genes; common band index; equal weight per run ± SEM","Mean absolute effect","module_4_overall_rna_guide_distance_profile_COMMON_BANDS_VARIANCE_SELECTED_SEM",True)
    plot_profiles(rna,"mean_significant_variance_selected","sem_significant_variance_selected","n_runs_significant_selected",band_labels,"Overall Module 4 RNA guide-level distance profile\nsignificant variance-selected RNA genes; common band index; equal weight per run","Mean absolute significant effect","module_4_overall_rna_guide_distance_profile_COMMON_BANDS_VARIANCE_SELECTED_SIGNIFICANT",False)
    plot_run_coverage(rna,"n_runs_all_selected",band_labels,"RNA run coverage | variance-selected genes","rna_run_coverage_heatmap_VARIANCE_SELECTED")
    print("\nRegenerating unchanged ATAC common-band plots...",flush=True)
    atac=summarize_atac_common_bands()
    if not atac.empty:
        plot_profiles(atac,"mean_all_features","sem_all_features","n_runs_all",band_labels,"Overall Module 4 ATAC guide-level distance profile\nall ATAC features; common band index; equal weight per run","Mean absolute effect","module_4_overall_atac_guide_distance_profile_COMMON_BANDS_ALL_FEATURES",False)
        plot_profiles(atac,"mean_significant_only","sem_significant_only","n_runs_significant",band_labels,"Overall Module 4 ATAC guide-level distance profile\nsignificant features; common band index; equal weight per run","Mean absolute significant effect","module_4_overall_atac_guide_distance_profile_COMMON_BANDS",False)
        plot_profiles(atac,"mean_significant_only","sem_significant_only","n_runs_significant",band_labels,"Overall Module 4 ATAC guide-level distance profile\nsignificant features; common band index; equal weight per run ± SEM","Mean absolute significant effect","module_4_overall_atac_guide_distance_profile_COMMON_BANDS_SEM",True)
        plot_run_coverage(atac,"n_runs_all",band_labels,"ATAC run coverage","atac_run_coverage_heatmap")
    print("\nDONE",flush=True)
    print("Outputs:",OUT.resolve(),flush=True)
    print("\nPRIMARY UPDATED RNA FIGURE:",flush=True)
    print("module_4_overall_rna_guide_distance_profile_COMMON_BANDS_VARIANCE_SELECTED.png",flush=True)
if __name__=="__main__":
    main()

# Figure 6f: variance-selected RNA spatial effective-radius maps
WORKDIR=Path("/gpfs/accounts/welchjd_root/welchjd1/javidgmh/15515_Multiome/catatac_work/motif_analysis")
TRAINING_SCRIPT=Path("/scratch/welchjd_root/welchjd1/javidgmh/15515_Multiome/catatac_work/motif_analysis/joint_multiome_MODALITY_SPECIFIC_SPATIAL_RESIDUAL_TUNE_LOSS_WEIGHTS_10EB.py")
TRAINING_OUTDIR=WORKDIR/"multiome_joint_RNA_ATAC_MODALITY_SPECIFIC_SPATIAL_RESIDUAL_TUNE_LOSS_WEIGHTS_10EB"
COUNTERFACTUAL_ROOT=TRAINING_OUTDIR/"counterfactual_inference_publication"
DATA_PATH=Path("/scratch/welchjd_root/welchjd1/javidgmh/15515_Multiome/catatac_work/motif_analysis/all_lanes_merged_CATATAC_QC_RNA_processed_EB_manual_clustering_cell_types.h5mu")
OVERALL_OUT=COUNTERFACTUAL_ROOT/"overall_across_runs"/"module_4_spatial_effective_radius_maps_overall_VARIANCE_SELECTED_RNA"
TOP_VARIABLE_RNA_GENES=100
EXCLUDED_PREFIXES=("MT-","RPL","RPS","MRPL","MRPS")
HOUSEKEEPING_GENES={"ACTB","ACTG1","B2M","EEF1A1","GAPDH","GUSB","HMBS","HPRT1","MALAT1","NONO","PGK1","PPIA","SDHA","TBP","TFRC","TUBA1A","TUBA1B","TUBB","TUBB4B","UBC","VCP","YWHAZ"}
POINT_SIZE_MIN=14.0
POINT_SIZE_MAX=78.0
BACKGROUND_POINT_SIZE=10.0
BACKGROUND_POINT_ALPHA=0.35
WEIGHTED_RADIUS_CMAP="magma"
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
def load_training():
    spec=importlib.util.spec_from_file_location("training_module",TRAINING_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(TRAINING_SCRIPT)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
def valid_eb_label(value):
    x=str(value).strip()
    return x.startswith("EB_") and x.lower() not in {"eb_","eb_nan","eb_none","eb_unassigned"}
def parse_guides(value):
    if value is None or (isinstance(value,float) and np.isnan(value)):
        return []
    if isinstance(value,(list,tuple,np.ndarray)):
        values=value
    else:
        text=str(value).strip()
        if text=="" or text.lower() in {"nan","none","unassigned","no guide","no_guide"}:
            return []
        values=text.split(";")
    return [str(x).strip() for x in values if str(x).strip()]
def build_guides(obs,source_col):
    rows=[parse_guides(x) for x in obs[source_col].astype(object).to_numpy()]
    names=sorted({g for row in rows for g in row})
    lookup={g:i for i,g in enumerate(names)}
    matrix=np.zeros((len(rows),len(names)),dtype=np.float32)
    for i,row in enumerate(rows):
        for g in row:
            matrix[i,lookup[g]]=1.0
    return matrix,names
def incoming_neighbors(edge_index,edge_distance,n_cells):
    src=edge_index[0].detach().cpu().numpy().astype(int)
    dst=edge_index[1].detach().cpu().numpy().astype(int)
    dist=edge_distance.detach().cpu().numpy().reshape(-1).astype(np.float32)
    neighbors=[[] for _ in range(n_cells)]
    distances=[[] for _ in range(n_cells)]
    for s,d,x in zip(src,dst,dist,strict=False):
        neighbors[d].append(s)
        distances[d].append(x)
    return [np.asarray(x,dtype=int) for x in neighbors],[np.asarray(x,dtype=np.float32) for x in distances]
def union_neighbors(a,b):
    out=[]
    for x,y in zip(a,b,strict=False):
        if len(x)==0 and len(y)==0:
            out.append(np.asarray([],dtype=int))
        else:
            out.append(np.unique(np.concatenate([x,y]).astype(int)))
    return out
def load_global_data(training):
    mu.set_options(pull_on_update=False)
    mdata=mu.read_h5mu(DATA_PATH)
    rna=mdata.mod["rna"]
    atac=mdata.mod["atac"]
    if not np.array_equal(np.asarray(rna.obs_names.astype(str)),np.asarray(atac.obs_names.astype(str))):
        raise ValueError("RNA and ATAC cell ordering differs")
    obs=mdata.obs.copy()
    coords=training.dense32(rna.obsm[training.SPATIAL_KEY])
    eb_all=obs[training.EB_KEY].astype(str).to_numpy()
    valid=np.asarray([valid_eb_label(x) for x in eb_all],dtype=bool)
    obs=obs.iloc[np.flatnonzero(valid)].copy()
    coords=coords[valid]
    rna_counts=training.get_counts(rna,training.RNA_COUNTS_LAYER,"RNA",False)[valid]
    guides_np,guide_names=build_guides(obs,training.GUIDE_SOURCE_COL)
    celltype_df=pd.get_dummies(obs[training.CELL_TYPE_KEY].astype(str),dtype=np.float32)
    celltypes_np=celltype_df.to_numpy(np.float32)
    x_np=np.concatenate([guides_np,celltypes_np],axis=1).astype(np.float32)
    return SimpleNamespace(obs=obs,coords=coords,eb=obs[training.EB_KEY].astype(str).to_numpy(),cell_type=obs[training.CELL_TYPE_KEY].astype(str).to_numpy(),guide_names=guide_names,guide_lookup={g:i for i,g in enumerate(guide_names)},guides_np=guides_np,celltypes_np=celltypes_np,x_np=x_np,x_tensor=torch.tensor(x_np,dtype=torch.float32,device=DEVICE),guides_tensor=torch.tensor(guides_np,dtype=torch.float32,device=DEVICE),celltypes_tensor=torch.tensor(celltypes_np,dtype=torch.float32,device=DEVICE),rna_counts=rna_counts,rna_library_all=np.asarray(rna_counts.sum(1)).ravel().astype(np.float32))
def build_run(training,global_data,run_dir,final_seeds):
    m=re.match(r"run(\d+)_",run_dir.name)
    if not m:
        raise ValueError(run_dir.name)
    run=int(m.group(1))
    selected=pd.read_csv(TRAINING_OUTDIR/f"run{run}_selected_modality_graphs.csv")
    lookup={str(x.modality):(int(x.k_neighbors),float(x.distance_decay)) for x in selected.itertuples(index=False)}
    rna_k,rna_decay=lookup["rna"]
    atac_k,atac_decay=lookup["atac"]
    rna_edge,rna_dist=training.build_knn_graph(global_data.coords,global_data.eb,rna_k)
    rna_context=training.weighted_neighbor_context(global_data.guides_np,global_data.celltypes_np,rna_edge,rna_dist,rna_decay)
    rna_neighbors,rna_neighbor_distances=incoming_neighbors(rna_edge,rna_dist,len(global_data.obs))
    atac_edge,atac_dist=training.build_knn_graph(global_data.coords,global_data.eb,atac_k)
    atac_neighbors,_=incoming_neighbors(atac_edge,atac_dist,len(global_data.obs))
    unions=union_neighbors(rna_neighbors,atac_neighbors)
    offsets=np.load(TRAINING_OUTDIR/f"run{run}_offsets_and_targets.npz")
    rna_targets=pd.read_csv(TRAINING_OUTDIR/f"run{run}_rna_hvgs.csv")
    atac_targets=pd.read_csv(TRAINING_OUTDIR/f"run{run}_atac_deviance_features.csv")
    return SimpleNamespace(run=run,outdir=run_dir,rna_feature_names=rna_targets["gene"].astype(str).to_numpy(),atac_feature_names=atac_targets["atac_feature"].astype(str).to_numpy(),rna_feature_idx=offsets["rna_target_idx"].astype(int),rna_mu0=offsets["rna_mu0"].astype(np.float32),rna_log_library=offsets["rna_log_library"].astype(np.float32),rna_decay=rna_decay,atac_decay=atac_decay,rna_bundle=(rna_edge.to(DEVICE),rna_dist.to(DEVICE),torch.tensor(rna_context,dtype=torch.float32,device=DEVICE)),rna_neighbors=rna_neighbors,rna_neighbor_distances=rna_neighbor_distances,union_neighbors=unions,final_seeds=final_seeds)
def load_models(training,global_data,run_data):
    context_dim=global_data.guides_np.shape[1]+global_data.celltypes_np.shape[1]+3
    models=[]
    for seed in run_data.final_seeds:
        path=TRAINING_OUTDIR/f"run{run_data.run}_seed{seed}_gat_best.pt"
        model=training.MultiomicSpatialResidualGAT(global_data.x_tensor.shape[1],global_data.guides_np.shape[1],global_data.celltypes_np.shape[1],len(run_data.rna_feature_names),len(run_data.atac_feature_names),context_dim,run_data.rna_decay,run_data.atac_decay).to(DEVICE)
        model.load_state_dict(torch.load(path,map_location=DEVICE),strict=True)
        model.eval()
        models.append(model)
    return models
@torch.no_grad()
def baseline_rna(training,global_data,run_data,models):
    loglib=torch.tensor(run_data.rna_log_library,dtype=torch.float32,device=DEVICE)
    mu0=torch.tensor(run_data.rna_mu0,dtype=torch.float32,device=DEVICE)
    predictions=[]
    for model in models:
        edge,dist,context=run_data.rna_bundle
        residual=model.rna(global_data.x_tensor,global_data.guides_tensor,global_data.celltypes_tensor,edge,dist,context)
        predictions.append(training.make_mu(residual,loglib,mu0).detach().cpu().numpy())
    return np.mean(np.stack(predictions),axis=0).astype(np.float32)
def context_from_neighbors(edited_guides,neighbor_indices,neighbor_distances,celltypes_np,decay):
    if len(neighbor_indices)==0:
        guide_mean=np.zeros(edited_guides.shape[1],dtype=np.float32)
        celltype_mean=np.zeros(celltypes_np.shape[1],dtype=np.float32)
        degree=mean_distance=min_distance=0.0
    else:
        weights=np.exp(-neighbor_distances/float(decay)).astype(np.float32)
        weight_sum=float(np.clip(weights.sum(),1e-8,None))
        guide_mean=(edited_guides[neighbor_indices]*weights[:,None]).sum(axis=0)/weight_sum
        celltype_mean=(celltypes_np[neighbor_indices]*weights[:,None]).sum(axis=0)/weight_sum
        degree=float(len(neighbor_indices))
        mean_distance=float((neighbor_distances*weights).sum()/weight_sum)
        min_distance=float(neighbor_distances.min())
    value=np.concatenate([guide_mean.astype(np.float32),celltype_mean.astype(np.float32),np.asarray([math.log1p(degree),mean_distance/float(decay),min_distance/float(decay)],dtype=np.float32)])
    return torch.tensor(value[None,:],dtype=torch.float32,device=DEVICE)
@torch.no_grad()
def predict_local_rna(training,branch,global_data,run_data,central_cell,edited_cells,source_index,target_index,edit_mode):
    neighbors=run_data.rna_neighbors[central_cell]
    distances=run_data.rna_neighbor_distances[central_cell]
    local_indices=np.concatenate([[central_cell],neighbors]).astype(int)
    local_x=global_data.x_np[local_indices].copy()
    if len(edited_cells):
        lookup={g:i for i,g in enumerate(local_indices)}
        overlapping=[lookup[x] for x in edited_cells if x in lookup]
        if overlapping:
            overlapping=np.asarray(overlapping,dtype=int)
            if edit_mode=="replace_source_with_target":
                local_x[overlapping,source_index]=0.0
                local_x[overlapping,target_index]=1.0
            elif edit_mode=="set_all_neighbors_to_target":
                local_x[overlapping,:len(global_data.guide_names)]=0.0
                local_x[overlapping,target_index]=1.0
            else:
                raise ValueError(edit_mode)
    local_x_t=torch.tensor(local_x,dtype=torch.float32,device=DEVICE)
    local_guides=local_x_t[:,:len(global_data.guide_names)]
    local_celltypes=local_x_t[:,len(global_data.guide_names):]
    base=branch.base(local_x_t[0:1],local_guides[0:1],local_celltypes[0:1])
    receiver=branch.receiver_encoder(local_celltypes[0:1])
    if len(neighbors)==0:
        spatial=torch.zeros((1,training.SPATIAL_DIM),dtype=torch.float32,device=DEVICE)
    else:
        source=branch.source_encoder(local_x_t[1:])
        scaled=torch.tensor(distances[:,None],dtype=torch.float32,device=DEVICE)/branch.distance_decay
        raw=torch.cat([scaled,torch.exp(-scaled),1.0/(1.0+scaled),torch.log1p(torch.tensor(distances[:,None],dtype=torch.float32,device=DEVICE))/math.log1p(branch.distance_decay)],dim=1)
        edge_attr=branch.edge_encoder(raw)
        local_edge=torch.stack([torch.arange(len(neighbors),dtype=torch.long,device=DEVICE),torch.zeros(len(neighbors),dtype=torch.long,device=DEVICE)],dim=0)
        spatial=F.elu(branch.gat((source,receiver),local_edge,edge_attr=edge_attr))
    edited_guides=local_x[:,:len(global_data.guide_names)].astype(np.float32)
    context=branch.context_encoder(context_from_neighbors(edited_guides,np.arange(1,len(local_indices),dtype=int),distances,local_x[:,len(global_data.guide_names):].astype(np.float32),branch.distance_decay))
    gate=branch.gate(torch.cat([receiver,spatial,context],dim=1))
    correction=branch.correction_head(branch.correction_encoder(torch.cat([receiver,gate*spatial,context],dim=1)))
    residual=base+correction
    mu=torch.exp((residual+float(run_data.rna_log_library[central_cell])+torch.tensor(run_data.rna_mu0[None,:],dtype=torch.float32,device=DEVICE)).clamp(-12,12))[0]
    return mu.detach().cpu().numpy().astype(np.float32)
def selected_edit_cells(global_data,run_data,central_cell,source_index,target_index,left,right,inclusive_right,edit_mode,edit_central):
    candidates=run_data.union_neighbors[central_cell]
    if len(candidates):
        distances=np.linalg.norm(global_data.coords[candidates,:2]-global_data.coords[central_cell,:2],axis=1)
        if inclusive_right:
            candidates=candidates[(distances>=left)&(distances<=right)]
        else:
            candidates=candidates[(distances>=left)&(distances<right)]
    if edit_central:
        candidates=np.unique(np.concatenate([candidates,np.asarray([central_cell],dtype=int)]))
    if len(candidates)==0:
        return np.asarray([],dtype=int)
    if edit_mode=="replace_source_with_target":
        return candidates[global_data.guides_np[candidates,source_index]>0].astype(int)
    if edit_mode=="set_all_neighbors_to_target":
        return candidates.astype(int)
    raise ValueError(edit_mode)
def informative_mask(names):
    keep=np.ones(len(names),dtype=bool)
    for i,gene in enumerate(names):
        x=str(gene).upper()
        if x.startswith(EXCLUDED_PREFIXES) or x in HOUSEKEEPING_GENES:
            keep[i]=False
    return keep
def compute_variability(global_data,run_data):
    x=global_data.rna_counts[:,run_data.rna_feature_idx]
    if not sp.issparse(x):
        x=sp.csr_matrix(x)
    x=x.astype(np.float64).tocsr(copy=True)
    scale=1e4/np.maximum(global_data.rna_library_all.astype(np.float64),1.0)
    x=sp.diags(scale).dot(x).tocsr()
    x.data=np.log1p(x.data)
    mean=np.asarray(x.mean(axis=0)).ravel()
    var=np.maximum(np.asarray(x.power(2).mean(axis=0)).ravel()-mean**2,0.0)
    dispersion=np.divide(var,mean,out=np.zeros_like(var),where=mean>0)
    log_dispersion=np.log1p(dispersion)
    n_bins=min(20,max(1,len(mean)))
    ranks=pd.Series(mean).rank(method="first")
    bins=pd.qcut(ranks,q=n_bins,labels=False,duplicates="drop").to_numpy()
    score=np.zeros(len(mean),dtype=np.float64)
    for b in np.unique(bins):
        mask=bins==b
        values=log_dispersion[mask]
        if len(values)<=1:
            continue
        sd=float(np.std(values,ddof=1))
        if np.isfinite(sd) and sd>0:
            score[mask]=(values-float(np.mean(values)))/sd
    return score
def choose_variable_genes(global_data,run_data,new_run_out):
    module3_ranking=run_data.outdir/"module_3_guide_pair_screen_variance_ranked"/"rna_variability_ranking.csv"
    if module3_ranking.exists():
        ranking=pd.read_csv(module3_ranking)
        if {"feature","selected_top100"}.issubset(ranking.columns):
            selected_flag=ranking["selected_top100"] if pd.api.types.is_bool_dtype(ranking["selected_top100"]) else ranking["selected_top100"].astype(str).str.lower().isin(["true","1","1.0","yes"])
            selected_names=ranking.loc[selected_flag,"feature"].astype(str).tolist()
            lookup={g:i for i,g in enumerate(run_data.rna_feature_names)}
            indices=np.asarray([lookup[g] for g in selected_names if g in lookup],dtype=int)
            if len(indices):
                pd.DataFrame({"feature":run_data.rna_feature_names[indices],"selected_index":indices,"selection_source":"Module 3 saved variance ranking"}).to_csv(new_run_out/"selected_variable_rna_genes.csv",index=False)
                print(f"Run {run_data.run}: using {len(indices)} RNA genes from Module 3 saved variance selection",flush=True)
                return indices
    score=compute_variability(global_data,run_data)
    valid=informative_mask(run_data.rna_feature_names)&np.isfinite(score)
    candidates=np.flatnonzero(valid)
    indices=candidates[np.argsort(score[candidates])[::-1]][:min(TOP_VARIABLE_RNA_GENES,len(candidates))]
    pd.DataFrame({"feature":run_data.rna_feature_names,"variability_score":score,"excluded":~informative_mask(run_data.rna_feature_names),"selected":np.isin(np.arange(len(score)),indices)}).sort_values("variability_score",ascending=False).to_csv(new_run_out/"rna_variability_ranking.csv",index=False)
    pd.DataFrame({"feature":run_data.rna_feature_names[indices],"selected_index":indices,"selection_source":"recomputed Module 3-style variability"}).to_csv(new_run_out/"selected_variable_rna_genes.csv",index=False)
    print(f"Run {run_data.run}: recomputed and selected {len(indices)} variable RNA genes",flush=True)
    return indices
def add_band_index(old,bands):
    old=old.copy()
    bands=bands.copy()
    old["_left"]=old["distance_left_um"].round(6)
    old["_right"]=old["distance_right_um"].round(6)
    bands["_left"]=bands["distance_left_um"].round(6)
    bands["_right"]=bands["distance_right_um"].round(6)
    out=old.merge(bands[["_left","_right","band_index"]],on=["_left","_right"],how="left",validate="many_to_one")
    if out["band_index"].isna().any():
        raise ValueError("Could not match some cell-level rows to distance_bands_used.csv")
    return out.drop(columns=["_left","_right"])
def recompute_run(training,global_data,run_data):
    original_path=run_data.outdir/"module_4_distance_resolved_effect_radius"/"cell_level_distance_band_summary.csv"
    band_path=run_data.outdir/"module_4_distance_resolved_effect_radius"/"distance_bands_used.csv"
    if not original_path.exists() or not band_path.exists():
        raise FileNotFoundError(f"Missing Module 4 outputs for run {run_data.run}")
    new_out=run_data.outdir/"module_4_distance_resolved_effect_radius_VARIANCE_SELECTED_RNA"
    new_out.mkdir(parents=True,exist_ok=True)
    original=add_band_index(pd.read_csv(original_path),pd.read_csv(band_path))
    config_path=run_data.outdir/"counterfactual_inference_configuration.json"
    config=json.loads(config_path.read_text()) if config_path.exists() else {}
    edit_mode=str(config.get("counterfactual_edit_mode","replace_source_with_target"))
    edit_central=bool(config.get("edit_central_cell",False))
    selected_idx=choose_variable_genes(global_data,run_data,new_out)
    models=load_models(training,global_data,run_data)
    baseline=baseline_rna(training,global_data,run_data,models)
    rows=[]
    max_band=int(pd.read_csv(band_path)["band_index"].max())
    total=len(original)
    print(f"Run {run_data.run}: recomputing {total} existing Module 4 RNA cell/band scenarios",flush=True)
    for count,row in enumerate(original.itertuples(index=False),1):
        central=int(row.central_cell_index)
        source=str(row.source_guide)
        target=str(row.target_guide)
        source_index=global_data.guide_lookup[source]
        target_index=global_data.guide_lookup[target]
        edited=selected_edit_cells(global_data,run_data,central,source_index,target_index,float(row.distance_left_um),float(row.distance_right_um),int(row.band_index)==max_band,edit_mode,edit_central)
        if len(edited)==0:
            continue
        cf=[]
        for model in models:
            cf.append(predict_local_rna(training,model.rna,global_data,run_data,central,edited,source_index,target_index,edit_mode))
        cf=np.mean(np.stack(cf),axis=0).astype(np.float32)
        effect=np.log1p(cf)-np.log1p(baseline[central])
        selected_effect=effect[selected_idx]
        d=row._asdict()
        d["run"]=run_data.run
        d["mean_absolute_rna_effect_variance_selected"]=float(np.mean(np.abs(selected_effect)))
        d["mean_signed_rna_effect_variance_selected"]=float(np.mean(selected_effect))
        d["n_variance_selected_rna_genes"]=int(len(selected_idx))
        rows.append(d)
        if count%250==0 or count==total:
            print(f"  {count}/{total}",flush=True)
    result=pd.DataFrame(rows)
    result.to_csv(new_out/"cell_level_distance_band_summary_variance_selected_rna.csv",index=False)
    del models,baseline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result
def common_band_pool(all_rows):
    metadata=all_rows[["run","band_index","distance_left_um","distance_right_um"]].drop_duplicates()
    metadata["midpoint_um"]=0.5*(metadata["distance_left_um"]+metadata["distance_right_um"])
    common_midpoints=metadata.groupby("band_index")["midpoint_um"].mean().to_dict()
    group_cols=["perturbation_guide","central_cell_index","central_cell_id","central_EB","central_cell_type","band_index"]
    pooled=all_rows.groupby(group_cols,as_index=False).agg(mean_absolute_rna_effect_variance_selected=("mean_absolute_rna_effect_variance_selected","mean"),n_runs_present=("run","nunique"))
    pooled["common_band_midpoint_um"]=pooled["band_index"].map(common_midpoints).astype(float)
    return pooled,metadata,common_midpoints
def summarize_radius(pooled):
    rows=[]
    for (_, _),group in pooled.groupby(["perturbation_guide","central_cell_index"],sort=False):
        group=group.sort_values("band_index")
        effects=group["mean_absolute_rna_effect_variance_selected"].to_numpy(float)
        if len(effects)==0 or np.all(~np.isfinite(effects)):
            continue
        best=int(np.nanargmax(effects))
        weights=np.clip(effects,0,None)
        if weights.sum()>0:
            radius=float(np.sum(weights*group["common_band_midpoint_um"].to_numpy(float))/weights.sum())
        else:
            radius=float(group.iloc[best]["common_band_midpoint_um"])
        row=group.iloc[best]
        rows.append({"perturbation_guide":str(row["perturbation_guide"]),"central_cell_index":int(row["central_cell_index"]),"central_cell_id":str(row["central_cell_id"]),"central_EB":str(row["central_EB"]),"central_cell_type":str(row["central_cell_type"]),"dominant_band_index":int(row["band_index"]),"dominant_radius_um":float(row["common_band_midpoint_um"]),"weighted_effective_radius_um":radius,"max_effect":float(effects[best]),"total_effect":float(np.nansum(effects)),"n_common_bands_present":int(group["band_index"].nunique()),"max_runs_present_for_any_band":int(group["n_runs_present"].max())})
    return pd.DataFrame(rows)
def configure_matplotlib():
    mpl.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Helvetica","Arial","DejaVu Sans"],"axes.linewidth":0.8,"axes.spines.top":False,"axes.spines.right":False,"axes.labelsize":10,"axes.titlesize":11,"xtick.labelsize":9,"ytick.labelsize":9,"legend.fontsize":9,"pdf.fonttype":42,"ps.fonttype":42,"figure.facecolor":"white","axes.facecolor":"white","savefig.dpi":600})
def scale_sizes(values):
    values=np.asarray(values,float)
    if len(values)==0:
        return values
    if np.allclose(values.max(),values.min()):
        return np.full(len(values),0.5*(POINT_SIZE_MIN+POINT_SIZE_MAX))
    x=(values-values.min())/(values.max()-values.min())
    return POINT_SIZE_MIN+x*(POINT_SIZE_MAX-POINT_SIZE_MIN)
def plot_grid(summary,global_data,outpath,radius_limits):
    guides=sorted(summary["perturbation_guide"].unique())
    ncols=min(4,len(guides))
    nrows=int(np.ceil(len(guides)/ncols))
    fig,axes=plt.subplots(nrows,ncols,figsize=(4.15*ncols,4.0*nrows),squeeze=False)
    letters="ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    scatter=None
    for i,guide in enumerate(guides):
        ax=axes[i//ncols,i%ncols]
        sub=summary[summary["perturbation_guide"]==guide].copy()
        idx=sub["central_cell_index"].to_numpy(int)
        xy=global_data.coords[idx,:2]
        ebs=set(sub["central_EB"].astype(str))
        bg=global_data.coords[np.isin(global_data.eb,np.asarray(list(ebs),dtype=str)),:2]
        x_min,x_max=float(bg[:,0].min()),float(bg[:,0].max())
        y_min,y_max=float(bg[:,1].min()),float(bg[:,1].max())
        x_pad=max(5,0.04*(x_max-x_min))
        y_pad=max(5,0.04*(y_max-y_min))
        ax.scatter(bg[:,0],bg[:,1],s=BACKGROUND_POINT_SIZE,color="#C9CED6",alpha=BACKGROUND_POINT_ALPHA,edgecolors="none",rasterized=True)
        sizes=scale_sizes(sub["max_effect"].to_numpy(float))
        scatter=ax.scatter(xy[:,0],xy[:,1],c=sub["weighted_effective_radius_um"].to_numpy(float),s=sizes,cmap=WEIGHTED_RADIUS_CMAP,vmin=radius_limits[0],vmax=radius_limits[1],edgecolors="white",linewidths=0.25,alpha=0.96,rasterized=True)
        ax.set_aspect("equal",adjustable="box")
        ax.set_xlim(x_min-x_pad,x_max+x_pad)
        ax.set_ylim(y_min-y_pad,y_max+y_pad)
        ax.set_xlabel("Spatial x (um)")
        ax.set_ylabel("Spatial y (um)" if i%ncols==0 else "")
        if i%ncols!=0:
            ax.set_yticklabels([])
        ax.set_title(guide,pad=7,fontsize=10.5)
        ax.text(0.01,0.99,letters[i],transform=ax.transAxes,ha="left",va="top",fontsize=11,fontweight="bold")
        span=(x_max+x_pad)-(x_min-x_pad)
        candidates=np.asarray([25,50,100,200,250,500,1000],float)
        eligible=candidates[candidates<=max(span*0.24,1)]
        bar=float(eligible[-1] if len(eligible) else candidates[0])
        xx=x_min-x_pad+0.06*span
        yy=y_min-y_pad+0.07*((y_max+y_pad)-(y_min-y_pad))
        ax.plot([xx,xx+bar],[yy,yy],color="black",linewidth=2)
        ax.text(xx+0.5*bar,yy+0.03*((y_max+y_pad)-(y_min-y_pad)),f"{bar:g} um",ha="center",va="bottom",fontsize=8.5)
    for i in range(len(guides),nrows*ncols):
        axes[i//ncols,i%ncols].axis("off")
    if scatter is not None:
        cbar=fig.colorbar(scatter,ax=axes.ravel().tolist(),fraction=0.022,pad=0.02)
        cbar.set_label("Weighted RNA effective radius (um)")
    fig.suptitle("Overall pooled held-out EBs | spatial perturbation distance maps (RNA; variance-selected)",y=0.995,fontsize=12)
    fig.savefig(outpath.with_suffix(".png"),dpi=600,bbox_inches="tight")
    fig.savefig(outpath.with_suffix(".pdf"),bbox_inches="tight")
    plt.close(fig)
def main():
    configure_matplotlib()
    OVERALL_OUT.mkdir(parents=True,exist_ok=True)
    training=load_training()
    global_data=load_global_data(training)
    config=json.loads((TRAINING_OUTDIR/"configuration.json").read_text())
    final_seeds=list(map(int,config["final_seeds"]))
    run_dirs=sorted([x for x in COUNTERFACTUAL_ROOT.iterdir() if x.is_dir() and re.match(r"run\d+_",x.name)],key=lambda x:int(re.match(r"run(\d+)_",x.name).group(1)))
    print("Device:",DEVICE,flush=True)
    print("Runs:",[x.name for x in run_dirs],flush=True)
    all_results=[]
    for run_dir in run_dirs:
        run_data=build_run(training,global_data,run_dir,final_seeds)
        result=recompute_run(training,global_data,run_data)
        all_results.append(result)
    all_rows=pd.concat(all_results,ignore_index=True)
    all_rows.to_csv(OVERALL_OUT/"all_runs_cell_level_variance_selected_rna_effects.csv",index=False)
    pooled,band_metadata,common_midpoints=common_band_pool(all_rows)
    pooled.to_csv(OVERALL_OUT/"pooled_common_band_cell_level_variance_selected_rna_effects.csv",index=False)
    band_metadata.to_csv(OVERALL_OUT/"run_specific_band_metadata.csv",index=False)
    pd.DataFrame({"band_index":list(common_midpoints.keys()),"common_midpoint_um":list(common_midpoints.values())}).sort_values("band_index").to_csv(OVERALL_OUT/"common_band_midpoints.csv",index=False)
    summary=summarize_radius(pooled)
    summary.to_csv(OVERALL_OUT/"rna_variance_selected_effective_radius_summary.csv",index=False)
    used_midpoints=pooled["common_band_midpoint_um"].dropna().to_numpy(float)
    radius_limits=(float(used_midpoints.min()),float(used_midpoints.max()))
    plot_grid(summary,global_data,OVERALL_OUT/"overall_all_guides_rna_weighted_effective_radius_grid_VARIANCE_SELECTED_COMMON_BANDS",radius_limits)
    print("\nDONE",flush=True)
    print("Output:",OVERALL_OUT.resolve(),flush=True)
    print("Main figure: overall_all_guides_rna_weighted_effective_radius_grid_VARIANCE_SELECTED_COMMON_BANDS.png",flush=True)
if __name__=="__main__":
    main()

# Figure 6d: observed and GAT-predicted TRPM3 spatial expression
TRAINING_SCRIPT = Path("joint_multiome_MODALITY_SPECIFIC_SPATIAL_RESIDUAL_TUNE_LOSS_WEIGHTS_10EB.py")
RESULTS_DIR = Path("multiome_joint_RNA_ATAC_MODALITY_SPECIFIC_SPATIAL_RESIDUAL_TUNE_LOSS_WEIGHTS_10EB")
OUTDIR = RESULTS_DIR / "publication_panels"
FOCUS_TEST_EB = "EB_34"
FOCUS_GENE = "TRPM3"
COLOR_TEXT = "#222222"
CMAP_EXPR = "magma"
def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.dpi": 600,
            "axes.grid": False,
            "text.color": COLOR_TEXT,
            "axes.labelcolor": COLOR_TEXT,
            "xtick.color": COLOR_TEXT,
            "ytick.color": COLOR_TEXT,
        }
    )
def save_plot(fig: plt.Figure, stem: Path) -> None:
    fig.tight_layout()
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
def load_training_module(path: Path):
    spec = importlib.util.spec_from_file_location("joint_multiome_training_module", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load training script from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
def dense32(x) -> np.ndarray:
    if sp.issparse(x):
        return x.toarray().astype(np.float32)
    return np.asarray(x, dtype=np.float32)
def valid_eb_label(value: object) -> bool:
    text = str(value).strip()
    return text.startswith("EB_") and text.lower() not in {"eb_", "eb_nan", "eb_none", "eb_unassigned"}
def parse_guides(value) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, (list, tuple, np.ndarray)):
        values = value
    else:
        text = str(value).strip()
        if text == "" or text.lower() in {"nan", "none", "unassigned", "no guide", "no_guide"}:
            return []
        values = text.split(";")
    return [str(x).strip() for x in values if str(x).strip()]
def build_guides(obs: pd.DataFrame, source_col: str) -> tuple[np.ndarray, list[str]]:
    rows = [parse_guides(value) for value in obs[source_col].astype(object).to_numpy()]
    names = sorted({guide for row in rows for guide in row})
    lookup = {name: idx for idx, name in enumerate(names)}
    matrix = np.zeros((len(rows), len(names)), dtype=np.float32)
    for row_idx, row in enumerate(rows):
        for guide in row:
            matrix[row_idx, lookup[guide]] = 1.0
    return matrix, names
class GlobalData:
    obs: pd.DataFrame
    coords: np.ndarray
    eb: np.ndarray
    guide_names: list[str]
    guides_np: np.ndarray
    rna_counts: sp.csr_matrix
    atac_counts: sp.csr_matrix
    rna_feature_names_all: np.ndarray
    cell_indices_by_eb: dict[str, np.ndarray]
@dataclass
class FocusRun:
    run: int
    validation_eb: str
    test_eb: str
    rna_graph_k: int
    rna_graph_decay: float
    rna_feature_names: np.ndarray
    rna_feature_idx: np.ndarray
    rna_log_library: np.ndarray
    rna_mu0: np.ndarray
    true_rna: np.ndarray
    gat_pred_rna: np.ndarray
    mlp_pred_rna: np.ndarray
def load_global_data(training) -> GlobalData:
    data_path = Path(training.DATA_PATH)
    if not data_path.exists():
        raise FileNotFoundError(data_path)
    mu.set_options(pull_on_update=False)
    mdata = mu.read_h5mu(data_path)
    rna = mdata.mod["rna"]
    atac = mdata.mod["atac"]
    rna_names = np.asarray(rna.obs_names.astype(str))
    atac_names = np.asarray(atac.obs_names.astype(str))
    main_names = np.asarray(mdata.obs_names.astype(str))
    if not np.array_equal(rna_names, atac_names) or not np.array_equal(rna_names, main_names):
        raise ValueError("RNA, ATAC, and MuData cells are not in identical order")
    obs = mdata.obs.copy()
    coords = dense32(rna.obsm[training.SPATIAL_KEY])
    eb_all = obs[training.EB_KEY].astype(str).to_numpy()
    valid_mask = np.asarray([valid_eb_label(value) for value in eb_all], dtype=bool)
    obs = obs.iloc[np.flatnonzero(valid_mask)].copy()
    coords = coords[valid_mask]
    rna_counts = training.get_counts(rna, training.RNA_COUNTS_LAYER, "RNA", False)[valid_mask]
    atac_counts = training.get_counts(atac, training.ATAC_COUNTS_LAYER, "ATAC", True)[valid_mask]
    guides_np, guide_names = build_guides(obs, training.GUIDE_SOURCE_COL)
    eb = obs[training.EB_KEY].astype(str).to_numpy()
    cell_indices_by_eb = {label: np.flatnonzero(eb == label) for label in pd.unique(eb)}
    return GlobalData(
        obs=obs,
        coords=coords,
        eb=eb,
        guide_names=guide_names,
        guides_np=guides_np,
        rna_counts=rna_counts,
        atac_counts=atac_counts,
        rna_feature_names_all=np.asarray(rna.var_names.astype(str)),
        cell_indices_by_eb=cell_indices_by_eb,
    )
def load_focus_run() -> FocusRun:
    manifest = pd.read_csv(RESULTS_DIR / "split_manifest.csv")
    focus = manifest[manifest["test_eb"].astype(str) == FOCUS_TEST_EB].copy()
    if focus.empty:
        raise ValueError(f"No run found with test EB {FOCUS_TEST_EB}")
    row = focus.iloc[0]
    run = int(row["run"])
    graphs = pd.read_csv(RESULTS_DIR / f"run{run}_selected_modality_graphs.csv")
    rna_graph = graphs[graphs["modality"].astype(str) == "rna"].iloc[0]
    rna_targets = pd.read_csv(RESULTS_DIR / f"run{run}_rna_hvgs.csv")
    offsets = np.load(RESULTS_DIR / f"run{run}_offsets_and_targets.npz")
    return FocusRun(
        run=run,
        validation_eb=str(row["validation_eb"]),
        test_eb=str(row["test_eb"]),
        rna_graph_k=int(rna_graph["k_neighbors"]),
        rna_graph_decay=float(rna_graph["distance_decay"]),
        rna_feature_names=rna_targets["gene"].astype(str).to_numpy(),
        rna_feature_idx=offsets["rna_target_idx"].astype(int),
        rna_log_library=offsets["rna_log_library"].astype(np.float32),
        rna_mu0=offsets["rna_mu0"].astype(np.float32),
        true_rna=np.load(RESULTS_DIR / f"run{run}_heldout_true_rna_counts.npy").astype(np.float32),
        gat_pred_rna=np.load(RESULTS_DIR / f"run{run}_gat_3seed_ensemble_rna_pred.npy").astype(np.float32),
        mlp_pred_rna=np.load(RESULTS_DIR / f"run{run}_mlp_3seed_ensemble_rna_pred.npy").astype(np.float32),
    )
def panel_h_and_i_spatial_expression(global_data: GlobalData, focus_run: FocusRun) -> None:
    eb_indices = global_data.cell_indices_by_eb[focus_run.test_eb]
    gene_global_mask = global_data.rna_feature_names_all == FOCUS_GENE
    if not gene_global_mask.any():
        raise ValueError(f"{FOCUS_GENE} not found in full RNA feature list.")
    gene_global_idx = int(np.flatnonzero(gene_global_mask)[0])
    gene_run_mask = focus_run.rna_feature_names == FOCUS_GENE
    if not gene_run_mask.any():
        raise ValueError(f"{FOCUS_GENE} not found in modeled RNA targets for focus run.")
    gene_run_idx = int(np.flatnonzero(gene_run_mask)[0])
    observed_counts = global_data.rna_counts[eb_indices, gene_global_idx].toarray().ravel().astype(np.float32)
    library = np.asarray(global_data.rna_counts[eb_indices].sum(1)).ravel().astype(np.float32)
    observed_expr = np.log1p(observed_counts * (1e4 / np.maximum(library, 1.0)))
    predicted_expr = np.log1p(np.clip(focus_run.gat_pred_rna[:, gene_run_idx], 1e-8, None))
    xy = global_data.coords[eb_indices, :2]
    vmax = float(max(observed_expr.max(), predicted_expr.max()))
    vmin = 0.0
    fig, ax = plt.subplots(figsize=(4.4, 4.2))
    ax.scatter(xy[:, 0], xy[:, 1], s=12, color="black", alpha=0.92, edgecolors="none")
    mask = observed_expr > 0
    sc = ax.scatter(xy[mask, 0], xy[mask, 1], c=observed_expr[mask], cmap=CMAP_EXPR, s=18, edgecolors="none", vmin=vmin, vmax=vmax)
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ["top", "right", "left", "bottom"]:
        ax.spines[side].set_visible(False)
    ax.set_title(f"Observed {FOCUS_GENE} expression in held-out {focus_run.test_eb}")
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("log1p normalized expression")
    save_plot(fig, OUTDIR / f"panel_H_observed_{FOCUS_GENE}_spatial_expression")
    fig, ax = plt.subplots(figsize=(4.4, 4.2))
    ax.scatter(xy[:, 0], xy[:, 1], s=12, color="black", alpha=0.92, edgecolors="none")
    mask = predicted_expr > 0
    sc = ax.scatter(xy[mask, 0], xy[mask, 1], c=predicted_expr[mask], cmap=CMAP_EXPR, s=18, edgecolors="none", vmin=vmin, vmax=vmax)
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ["top", "right", "left", "bottom"]:
        ax.spines[side].set_visible(False)
    ax.set_title(f"GAT predicted {FOCUS_GENE} expression")
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("log1p predicted expression")
    save_plot(fig, OUTDIR / f"panel_I_gat_predicted_{FOCUS_GENE}_spatial_expression")
def main():
    configure_matplotlib()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    training = load_training_module(TRAINING_SCRIPT)
    global_data = load_global_data(training)
    focus_run = load_focus_run()
    panel_h_and_i_spatial_expression(global_data, focus_run)
    print("Outputs:", OUTDIR.resolve(), flush=True)
if __name__ == "__main__":
    main()
