# Perturb-Tags Analysis Notebooks

Reproducible analysis notebooks for the Perturb-Tags spatial multiome experiment.

## Data

All data is downloaded from Google Drive when running the notebooks. The following files are used:

### Multiome data (ligand/receptor analyses)
| File | Google Drive ID | Description | Size |
|------|----------------|-------------|------|
| `all_lanes_CATATAC.h5mu` | `1NZzzEX9zHSj5Cp5ewuHRataW0jkIA-bo` | Multiome (RNA + ATAC + ChromVAR), 17,043 cells | ~615 MB |
| `aggregated_TE_RNA.h5ad` | `1PU3dJ0n3siOUqt67knTc3mXWe1k1gROV` | TE family RNA expression (CPM) | ~16 MB |
| `aggregated_TE_ATAC.h5ad` | `1H_7eV_ZynrXNHy0P9YAb66aTWLA9fvIg` | TE family ATAC accessibility (CPM) | ~130 MB |

### Kinase perturbation slides
| File | Google Drive ID | Description | Size |
|------|----------------|-------------|------|
| `slide1_manual_EBs.h5ad` | `11IA4l5KVsvGCq0BjdQ7DCCzWOWfjme-n` | Slide 1, 2,583 cells | ~23 MB |
| `slide2_manual_EBs.h5ad` | `1lwxZkS6cAZbcAEeI6OqJn2qxmpFFAs2B` | Slide 2, 11,908 cells | ~283 MB |
| `slide3_manual_EBs.h5ad` | `15CCMZzP0EBN7EvGSlu3mgj_wuZq6eTL3` | Slide 3, 5,213 cells | ~123 MB |
| `slide4_manual_EBs.h5ad` | `1M9RhXsbqSvOZbsGKTxeZ88gDxHmeqaav` | Slide 4, 3,659 cells | ~85 MB |

## Notebooks

### 1. `ligand_distance_analysis.ipynb`

Distance-dependent analysis of gene expression, chromatin accessibility, and transposable elements
as a function of spatial distance from ligand knockdown cells.

- 15 ligand KD targets (WNT5A, WNT1, BMP5, BMP7, JAG1, DLL3, VEGFB, PDGFC, EFNA3, RSPO3, SEMA6D, NECTIN2, NECTIN3, ALB, B2M)
- Within-EB distance computation using EB_manual_clustering (52 EBs)
- Feature selection: Spearman rho >= 0.8, fold change >= 1.2, detection > 5% (RNA) or > 1% (ATAC)
- Negative control: same analysis using distance to Non-Targeting cells
- RNA (log1p_normalized from Seurat SCT), ATAC (raw counts), TE RNA, TE ATAC

### 2. `receptor_interaction_analysis.ipynb`

Receptor KD × ligand microenvironment interaction analysis.

- 8 receptor KD targets with cognate ligand families
- Gaussian kernel ligand scoring across 11 radii (25-500 µm)
- Vectorized OLS interaction model: `expression ~ β_KD + β_ligand + β_KD×ligand`
- FDR correction per receptor per radius

### 3. `kinase_perturbation_qc.ipynb`

QC and summary analyses for 4 spatial kinase perturbation slides (~300 kinase targets).

- EB and cell counts, spatial plots by EB and cell type
- sgRNA and target gene summaries (boxplots by slide)
- Knockdown efficiency (Wilcoxon rank-sum, volcano plot)
- sgRNA+ cell distribution across cell types

### 4. `gp_distance_analysis.ipynb`

WNT5A spatial-expression screening and hierarchical Gaussian-process clustering.

- Fixed-noise constant-versus-RBF GP likelihood-ratio screen over 10,789 genes
- Benjamini-Hochberg selection at 5% FDR
- Top-ten posterior predictive GP fits
- Stabilized weighted-mean centering without variance normalization
- Four-cluster GPClust/MOHGP fit with ten independent restarts
- Centered spatial trajectories and separate original-scale cluster profiles

## Requirements

```
numpy scipy pandas matplotlib h5py statsmodels openpyxl gdown GPy
```

The GP clustering notebook additionally uses the pinned SheffieldML/GPclust
commit documented in its Setup section and should be run in the isolated
Python 3.12 environment described there.
