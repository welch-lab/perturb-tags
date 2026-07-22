# Perturb-Tags Analysis Notebooks

Reproducible analysis notebooks for the Perturb-Tags spatial multiome experiment.

## Data

All data is automatically downloaded from Google Drive when running the notebooks. The following files are used:

| File | Description | Size |
|------|-------------|------|
| `all_lanes_CATATAC.h5mu` | Multiome data (RNA + ATAC + ChromVAR) for 17,043 cells across 3 lanes/EBs | ~615 MB |
| `aggregated_TE_RNA.h5ad` | TE family-level RNA expression (CPM) | ~16 MB |
| `aggregated_TE_ATAC.h5ad` | TE family-level ATAC accessibility (CPM) | ~130 MB |

## Notebooks

### 1. `ligand_distance_analysis.ipynb`

Distance-dependent analysis of gene expression, chromatin accessibility, and transposable elements
as a function of spatial distance from ligand knockdown cells.

**Analyses:**
- Monotonic feature detection across 6 distance bins (0-300 µm) for 13 ligand KD targets
- Four modalities: RNA genes, ATAC peaks, TE RNA families, TE ATAC families
- Negative control: same analysis using distance to Non-Targeting (NTC) cells
- Within-EB normalization (per-lane pseudobulk averaged across lanes)

**Key outputs:**
- Summary bar plots with NTC reference lines
- RNA closing/opening gene line plots (top 6 per target)
- ATAC closing/opening peak line plots (top 6 per target)
- TE family line plots for top 4 targets
- Recurrent TE heatmap across ligand KDs

### 2. `receptor_interaction_analysis.ipynb`

Receptor KD × ligand microenvironment interaction analysis to detect non-cell-autonomous
signaling effects.

**Analyses:**
- Gaussian kernel ligand scoring across 20 radii (25-500 µm)
- OLS interaction model: `expression ~ β_KD + β_ligand + β_KD×ligand`
- 8 receptor KDs × cognate ligand families
- RNA, ATAC, and TE family-level interaction testing with FDR correction

**Key outputs:**
- Radius sweep heatmaps (interaction and ligand effect counts)
- Interaction coefficient bar plots per receptor
- TE interaction heatmap
- Full results Excel file

## Requirements

```
numpy
scipy
pandas
matplotlib
h5py
gdown
statsmodels
openpyxl
nbformat
```

## Usage

```bash
pip install jupyter
jupyter notebook ligand_distance_analysis.ipynb
```
