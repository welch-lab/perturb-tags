# Perturb-Tags Analysis Notebooks

Reproducible analysis scripts and notebooks used to generate the analyses presented in the Perturb-Tags manuscript.

## Analysis files

### Figure 1

#### `Belayer (HMM) Spatial Guide Clustering`

Spatial guide-clustering analysis used for Figure 1e.

- HMM-based identification of spatially clustered perturbation populations
- Spatial organization of guide-positive cells within individual EBs
- Identification and visualization of spatial perturbation clones

#### `distance_dependent_expression`

Distance-dependent receiver-cell expression analysis used for Figure 1.

- Defines cells proximal and distal to perturbation-positive cells
- Tests transcriptional responses in neighboring non-perturbed receiver cells
- Evaluates expression changes as a function of distance from perturbation-positive cells
- Generates the distance-dependent expression and representative spatial analyses shown in Figure 1

---

### Figure 2

#### `KD_effect_direct_compensatory`

Ligand-receptor analysis used to generate Figure 2.

- Quantifies direct ligand-receptor signaling responses following ligand knockdown
- Identifies compensatory ligand-receptor interactions
- Evaluates spatial LR-score changes surrounding perturbation-positive cells
- Generates direct and compensatory interaction summaries and distance-dependent LR profiles

---

### Figure 3

#### `multimodal_gp_50kb.py`

Multimodal Gaussian-process analysis used for Figure 3.

- Tests distance-dependent perturbation responses across ATAC-derived modalities
- Includes called ATAC peaks, fixed genomic bins, chromVAR motif activity, and ATAC gene activity
- Compares constant and radial-basis-function Gaussian-process models
- Uses Benjamini-Hochberg FDR correction to identify significant distance-dependent regulatory features
- Generates representative spatial regulatory trajectories and multimodal GP summaries

#### `receptor_interaction_analysis.ipynb`

Receptor KD × ligand microenvironment interaction analysis associated with the Figure 3 spatial signaling analyses.

- Receptor-specific ligand-family analysis
- Gaussian-kernel ligand scoring across spatial radii
- Tests receptor KD × local ligand-context interactions
- Vectorized OLS model: `expression ~ KD + ligand + KD×ligand`
- FDR correction across tested interactions

#### `SCENIC.ipynb`

Regulatory-network analysis used for Figure 3i.

- Transcription-factor/regulon analysis
- Integrates perturbation-associated transcriptional and regulatory responses
- Generates the regulatory-network analysis shown in Figure 3i

---

### Figure 4

#### `KD_effect_Cell_type_analysis.py`

Cell-state and spatial analyses used for Figure 4a-c and Figure 4e-g.

- Spatial visualization of perturbation and neighboring cell states
- Cell-type proportion analysis as a function of distance from perturbation-positive cells
- EB-level statistical analysis of distance-dependent cell-state changes
- Targeted pre-ranked gene-set enrichment analysis
- Common-coordinate spatial registration and visualization of perturbation-associated cell-state distributions

#### `kd_distance_ebwise_common_four_modalities.py`

Distance-dependent multimodal analysis used for Figure 4d.

- Evaluates perturbation-associated effects across multiple molecular modalities
- Performs EB-aware spatial distance analysis
- Integrates RNA, chromatin-accessibility, motif-activity, and related regulatory measurements
- Generates the multimodal distance-dependent results shown in Figure 4d

---

### Figure 5

#### `kinase_perturbation_qc.ipynb`

QC, spatial-distribution, and perturbation-summary analyses used for the kinase screen in Figure 5.

- EB and cell-count summaries
- Spatial distributions by EB and cell type
- sgRNA and target-gene summaries
- Knockdown-efficiency analysis
- Distribution of perturbation-positive cells across cell types
- Supports the kinase-screen characterization shown in Figure 5a-e

#### `cytosignal_9slides_diffusion_scores_100_200_400.ipynb`

CytoSignal ligand-receptor scoring for the nine-slide kinase perturbation dataset used for Figure 5.

- Computes spatial ligand-receptor scores at 100, 200, and 400 µm diffusion scales
- Generates the LR-score inputs used for downstream kinase-context analyses
- Provides upstream signaling features for Figure 5f-i

#### `Kinase_CytoSignal_LR_ElasticNet_Screen.ipynb`

First-stage kinase × ligand-receptor interaction screen used for Figure 5.

- Uses CytoSignal LR scores as spatial microenvironment features
- Elastic-net screening identifies candidate LR interactions associated with kinase-dependent receiver-cell states
- Prioritizes kinase-LR combinations for downstream gene-level modeling

#### `Kinase_CytoSignal_TopLR_GeneLevel_OLS.ipynb`

Gene-level kinase × ligand-receptor interaction analysis used for Figure 5f-g and to identify interactions visualized in Figure 5h-i.

- Fits gene-level OLS models to prioritized LR interactions
- Model includes kinase KD, LR score, and KD × LR interaction terms
- Applies multiple-testing correction across gene-level interaction effects
- Generates the kinase-context interaction summaries used in Figure 5

#### `Kinase_CytoSignal_LR_EB_Spatial_Score_KD_Maps.ipynb`

Spatial visualization of kinase × ligand-receptor interactions used for Figure 5h-i.

- Maps kinase perturbations and local LR signaling within individual EBs
- Visualizes representative context-dependent transcriptional responses
- Generates the representative spatial kinase-LR examples shown in Figure 5h-i

#### `cytosignal_diffusion_scores_only_100_200_400.ipynb`

CytoSignal ligand-receptor scoring for the receptor/multiome dataset.

- Computes LR scores at 100, 200, and 400 µm diffusion scales
- Provides spatial signaling features for the receptor-context analyses described in the Methods

#### `Receptor_CytoSignal_LR_ElasticNet_Screen.ipynb`

Receptor-context ElasticNet analysis corresponding to the interaction-modeling framework described with the Figure 5 analyses.

- Screens receptor KD × LR-context relationships
- Identifies candidate spatial ligand-receptor interactions for downstream testing
- Provides the receptor counterpart to the kinase interaction-screening workflow

#### `Receptor_CytoSignal_TopLR_GeneLevel_OLS.ipynb`

Gene-level receptor KD × ligand-receptor interaction analysis corresponding to the Figure 5 Methods framework.

- Fits gene-level receptor KD + LR + KD × LR interaction models
- Applies multiple-testing correction to interaction effects
- Provides the receptor-context counterpart to the kinase gene-level OLS analysis

#### `Receptor_CytoSignal_LR_EB_Spatial_Score_KD_Maps.ipynb`

Spatial visualization of receptor KD × ligand-receptor context effects.

- Maps receptor perturbations and local LR signaling within EBs
- Visualizes representative receptor-context associations
- Supports the receptor-context analyses described in the Methods

---

### Figure 6

#### `Multimodal_GAT.py`

Multimodal graph-attention-network training and evaluation used for Figure 6.

- Trains non-spatial MLP and spatial-residual GAT models for RNA and ATAC
- Uses ten held-out EBs with separate validation EBs
- Performs modality-specific graph and loss-weight tuning
- Ensembles three independently trained models
- Compares mean-baseline, MLP, and GAT held-out negative-binomial likelihood
- Performs the targeted ATAC analysis using the 402 FDR-significant ATAC peaks
- Generates the RNA and ATAC prediction-performance analyses used in Figure 6c
- Produces the trained models used for the downstream Figure 6d-g counterfactual analyses

#### `Inference.py`

Post-training GAT inference and counterfactual analyses used for Figure 6d-g.

- Figure 6d: observed versus GAT-predicted spatial expression of TRPM3 in a representative held-out EB
- Figure 6e: ordered guide-to-guide counterfactual substitution analysis for RNA and ATAC
- Figure 6f: distance-resolved guide-to-control counterfactual analysis and spatial effective-radius estimation
- Figure 6g: gene-level RNA responses to ordered source-to-target guide substitutions
- Uses the saved multimodal GAT models and held-out EB splits generated by `Multimodal_GAT_fig6.py`

## Requirements

Core analyses use:

```text
numpy
scipy
pandas
matplotlib
seaborn
h5py
statsmodels
scikit-learn
scanpy
anndata
mudata
torch
torch-geometric
openpyxl
gdown
GPy
