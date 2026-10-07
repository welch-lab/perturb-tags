from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
GP_OUTDIR=Path("GP_ligand_KD_distance_effect_on_GEX_50um_bins")
GO_OUTDIR=GP_OUTDIR/"GO_enrichment_increasing_decreasing"
PUBLICATION_OUTDIR=GO_OUTDIR/"publication_tables"
FIGURE_OUTDIR=GO_OUTDIR/"publication_figures"
PUBLICATION_OUTDIR.mkdir(parents=True,exist_ok=True)
FIGURE_OUTDIR.mkdir(parents=True,exist_ok=True)
GO_FDR_THRESHOLD=0.05
TOP_TERMS_PER_DIRECTION=15
def clean_go_term(term):
    term=str(term).replace("GOBP_","").replace("GO_","").replace("_"," ").lower()
    return term[0].upper()+term[1:] if term else term
def format_pvalue(x):
    if pd.isna(x):
        return ""
    x=float(x)
    if x<1e-300:
        return "<1 × 10^-300"
    if x<0.001:
        exponent=int(np.floor(np.log10(x)))
        coefficient=x/(10**exponent)
        return f"{coefficient:.2f} × 10^{exponent}"
    return f"{x:.3f}"
def make_diverging_plot(df,kd_gene,output_path):
    plot_parts=[]
    for direction in ["Increasing","Decreasing"]:
        sub=df[df["Direction"]==direction].copy()
        if sub.empty:
            continue
        sub=sub.sort_values(["FDR_numeric","P_numeric","Enrichment ratio"],ascending=[True,True,False]).head(TOP_TERMS_PER_DIRECTION)
        sub["plot_value"]=-np.log10(np.maximum(sub["FDR_numeric"].astype(float),1e-300))
        if direction=="Decreasing":
            sub["plot_value"]=-sub["plot_value"]
        plot_parts.append(sub)
    if not plot_parts:
        return
    plot_df=pd.concat(plot_parts,ignore_index=True)
    plot_df=plot_df.sort_values("plot_value").reset_index(drop=True)
    labels=plot_df["GO biological process"].tolist()
    values=plot_df["plot_value"].to_numpy()
    fig_height=max(6,0.38*len(plot_df)+1.5)
    fig,ax=plt.subplots(figsize=(10,fig_height))
    bars=ax.barh(labels,values)
    for bar,value,direction in zip(bars,values,plot_df["Direction"]):
        if direction=="Increasing":
            bar.set_color("#B2182B")
        else:
            bar.set_color("#2166AC")
    ax.axvline(0,color="black",linewidth=0.8)
    max_abs=max(abs(values.min()),abs(values.max())) if len(values)>0 else 1
    ax.set_xlim(-max_abs*1.15,max_abs*1.15)
    ticks=ax.get_xticks()
    ax.set_xticklabels([f"{abs(x):.1f}" for x in ticks])
    ax.set_xlabel("−log10(FDR)")
    ax.set_ylabel("GO biological process")
    ax.set_title(f"{kd_gene} KD")
    ax.grid(axis="x",alpha=0.2)
    ax.text(0.02,1.01,"Decreasing",transform=ax.transAxes,ha="left",va="bottom",fontweight="bold")
    ax.text(0.98,1.01,"Increasing",transform=ax.transAxes,ha="right",va="bottom",fontweight="bold")
    for spine in ["top","right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path,dpi=600,bbox_inches="tight")
    plt.close(fig)
if not GO_OUTDIR.exists():
    raise FileNotFoundError(f"GO output directory does not exist: {GO_OUTDIR}")
kd_directories=sorted([x for x in GO_OUTDIR.iterdir() if x.is_dir() and x.name not in ["publication_tables","publication_figures"]])
if not kd_directories:
    raise FileNotFoundError(f"No KD directories found inside: {GO_OUTDIR}")
all_publication_tables=[]
for kd_dir in kd_directories:
    kd_gene=kd_dir.name
    print("\n"+"="*80)
    print("Processing:",kd_gene)
    print("="*80)
    direction_tables=[]
    for direction in ["increasing","decreasing"]:
        go_file=kd_dir/f"{kd_gene}_{direction}_GO_FDR_0.05.csv"
        if not go_file.exists():
            print(f"Missing {direction} table: {go_file.name}")
            continue
        try:
            df=pd.read_csv(go_file)
        except pd.errors.EmptyDataError:
            print(f"Empty {direction} table.")
            continue
        if df.empty:
            print(f"No significant {direction} GO terms.")
            continue
        if "fdr" in df.columns:
            df=df[df["fdr"]<GO_FDR_THRESHOLD].copy()
        if df.empty:
            continue
        df=df.sort_values(["fdr","pval","enrichment_ratio"],ascending=[True,True,False]).reset_index(drop=True)
        df["Direction"]=direction.capitalize()
        df["Rank"]=np.arange(1,len(df)+1)
        df["GO biological process"]=df["go_term"].astype(str).map(clean_go_term)
        df["Gene overlap"]=df["overlap_count"].astype(int).astype(str)+"/"+df["query_gene_count"].astype(int).astype(str)
        df["GO set size"]=df["term_background_count"].astype(int)
        df["Enrichment ratio"]=df["enrichment_ratio"].astype(float).round(2)
        df["Odds ratio"]=df["odds_ratio"].astype(float).replace([np.inf,-np.inf],np.nan).round(2)
        df["P value"]=df["pval"].map(format_pvalue)
        df["FDR"]=df["fdr"].map(format_pvalue)
        df["P_numeric"]=df["pval"].astype(float)
        df["FDR_numeric"]=df["fdr"].astype(float)
        df["Genes"]=df["overlap_genes"]
        publication_df=df[["Direction","Rank","GO biological process","Gene overlap","GO set size","Enrichment ratio","Odds ratio","P value","FDR","P_numeric","FDR_numeric","Genes"]].copy()
        direction_tables.append(publication_df)
        print(f"{direction.capitalize()}: {len(publication_df)} significant GO terms")
    if not direction_tables:
        print("No significant GO terms; skipping.")
        continue
    kd_publication=pd.concat(direction_tables,ignore_index=True)
    kd_publication["Direction"]=pd.Categorical(kd_publication["Direction"],categories=["Increasing","Decreasing"],ordered=True)
    kd_publication=kd_publication.sort_values(["Direction","Rank"]).reset_index(drop=True)
    table_output=PUBLICATION_OUTDIR/f"{kd_gene}_GO_publication_table.csv"
    kd_publication.drop(columns=["P_numeric","FDR_numeric"]).to_csv(table_output,index=False)
    print("Saved table:",table_output)
    figure_output=FIGURE_OUTDIR/f"{kd_gene}_GO_diverging_barplot.png"
    make_diverging_plot(kd_publication,kd_gene,figure_output)
    print("Saved figure:",figure_output)
    master_df=kd_publication.copy()
    master_df.insert(0,"KD",kd_gene)
    all_publication_tables.append(master_df)
if all_publication_tables:
    all_kd_table=pd.concat(all_publication_tables,ignore_index=True)
    master_output=PUBLICATION_OUTDIR/"all_KD_publication_GO_table.csv"
    all_kd_table.drop(columns=["P_numeric","FDR_numeric"]).to_csv(master_output,index=False)
    print("\nMaster publication table saved:",master_output)
print("\nDone.")
import re
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mudata as md
from scipy.spatial import KDTree
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.stats.multitest import multipletests
warnings.filterwarnings("ignore")
H5MU_PATH = Path(
    "all_lanes_merged_CATATAC_QC_RNA_processed_EB_manual_clustering_cell_types.h5mu"
)
RNA_MODALITY = "rna"
GP_OUTDIR = Path(
    "GP_ligand_KD_distance_effect_on_GEX_50um_bins"
)
GO_OUTDIR = GP_OUTDIR / "GO_enrichment_increasing_decreasing"
GO_SIGNIFICANT_FILE = Path(
    "/scratch/welchjd_root/welchjd1/javidgmh/15515_Multiome/"
    "catatac_work/motif_analysis/"
    "GP_ligand_KD_distance_effect_on_GEX_50um_bins/"
    "GO_enrichment_increasing_decreasing/"
    "all_KD_direction_GO_results_FDR_0.05_combined.csv"
)
OUTDIR = Path(
    "MULTIOME_KD_distance_cell_type_composition_and_GO"
)
OUTDIR.mkdir(parents=True, exist_ok=True)
GUIDE_COL = "guides_passing_str"
EB_COL = "EB_manual_clustering"
CELL_TYPE_COL = "cell_type"
SPATIAL_KEY = "X_spatial"
CELL_TYPES = [
    "Neurons",
    "Pluripotent",
    "Neural Crest",
    "Endoderm",
    "Early Ectoderm",
    "Mesoderm",
]
BINS = [
    (0, 50),
    (50, 100),
    (100, 150),
    (150, 200),
    (200, 250),
    (250, 300),
]
BIN_LABELS = [
    "0-50",
    "50-100",
    "100-150",
    "150-200",
    "200-250",
    "250-300",
]
BIN_MIDS = np.array(
    [(lo + hi) / 2 for lo, hi in BINS],
    dtype=float
)
MAX_DISTANCE = 300.0
MIN_TOTAL_RECEIVER_CELLS = 20
MIN_EBS = 2
MIN_CELL_TYPE_POSITIVES = 5
MIN_CELL_TYPE_NEGATIVES = 5
FDR_THRESHOLD = 0.05
def parse_guides(guide_string):
    if pd.isna(guide_string):
        return []
    guide_string = str(guide_string).strip()
    if guide_string == "" or guide_string.lower() in {
        "nan",
        "none",
        "na",
    }:
        return []
    return [
        guide.strip()
        for guide in re.split(r"[;,|\s]+", guide_string)
        if guide.strip()
    ]
def guide_to_target(guide):
    return re.sub(
        r"-\d+$",
        "",
        str(guide).strip()
    )
def is_kd_for_gene(guide_string, gene):
    return gene in {
        guide_to_target(guide)
        for guide in parse_guides(guide_string)
    }
def normalize_cell_type(value):
    value = str(value).strip()
    value_lower = value.lower()
    lookup = {
        cell_type.lower(): cell_type
        for cell_type in CELL_TYPES
    }
    return lookup.get(
        value_lower,
        value
    )
def assign_distance_bin(distances):
    result = np.full(
        len(distances),
        -1,
        dtype=int
    )
    for bin_index, (lo, hi) in enumerate(BINS):
        if bin_index == len(BINS) - 1:
            in_bin = (
                (distances >= lo)
                & (distances <= hi)
            )
        else:
            in_bin = (
                (distances >= lo)
                & (distances < hi)
            )
        result[in_bin] = bin_index
    return result
def safe_sem(values):
    values = np.asarray(
        values,
        dtype=float
    )
    values = values[
        np.isfinite(values)
    ]
    if len(values) <= 1:
        return np.nan
    return (
        values.std(ddof=1)
        / np.sqrt(len(values))
    )
if not GO_SIGNIFICANT_FILE.exists():
    raise FileNotFoundError(
        "The completed significant GO result file was not found:\n"
        f"{GO_SIGNIFICANT_FILE}\n\n"
        "This script is designed to use the GO analysis that has already run."
    )
go_sig = pd.read_csv(
    GO_SIGNIFICANT_FILE
)
required_go_columns = {
    "kd_gene",
    "direction",
}
missing_go_columns = (
    required_go_columns
    - set(go_sig.columns)
)
if missing_go_columns:
    raise ValueError(
        "The GO file is missing required columns: "
        f"{sorted(missing_go_columns)}"
    )
go_sig["kd_gene"] = (
    go_sig["kd_gene"]
    .astype(str)
    .str.strip()
)
go_sig["direction"] = (
    go_sig["direction"]
    .astype(str)
    .str.strip()
    .str.lower()
)
go_sig = go_sig[
    go_sig["direction"].isin(
        ["increasing", "decreasing"]
    )
].copy()
KDS_TO_ANALYZE = sorted(
    go_sig["kd_gene"].dropna().unique()
)
print("KDs with significant GO results:")
print(KDS_TO_ANALYZE)
if len(KDS_TO_ANALYZE) == 0:
    raise ValueError(
        "No KD targets with significant GO results were found."
    )
if not H5MU_PATH.exists():
    raise FileNotFoundError(f"MuData file not found: {H5MU_PATH}")
print("Loading MuData:", H5MU_PATH)
mdata = md.read_h5mu(H5MU_PATH)
if RNA_MODALITY not in mdata.mod:
    raise ValueError(f"MuData is missing modality: {RNA_MODALITY}")
adata = mdata[RNA_MODALITY]
for required_obs_column in [
    GUIDE_COL,
    EB_COL,
    CELL_TYPE_COL,
]:
    if required_obs_column not in adata.obs.columns:
        raise ValueError(
            f"Missing mdata['rna'].obs['{required_obs_column}']"
        )
if SPATIAL_KEY not in adata.obsm:
    raise ValueError(
        f"Missing mdata['rna'].obsm['{SPATIAL_KEY}']"
    )
guide_strings = (
    adata.obs[GUIDE_COL]
    .fillna("")
    .astype(str)
    .to_numpy()
)
eb_labels = (
    adata.obs[EB_COL]
    .astype(str)
    .to_numpy()
)
cell_types = np.array(
    [
        normalize_cell_type(value)
        for value in adata.obs[CELL_TYPE_COL]
    ],
    dtype=object
)
spatial_coords = np.asarray(
    adata.obsm[SPATIAL_KEY][:, :2],
    dtype=float
)
print(f"Loaded {adata.n_obs} cells.")
print("Cell-type counts:")
print(
    pd.Series(cell_types)
    .value_counts()
    .to_string()
)
all_cell_level_tables = []
all_bin_summary_tables = []
for kd_gene in KDS_TO_ANALYZE:
    print("\n" + "=" * 80)
    print(f"Preparing {kd_gene} KD")
    print("=" * 80)
    kd_mask = np.array(
        [
            is_kd_for_gene(
                guide_string,
                kd_gene
            )
            for guide_string in guide_strings
        ],
        dtype=bool
    )
    non_kd_mask = ~kd_mask
    distances = np.full(
        adata.n_obs,
        np.inf,
        dtype=float
    )
    for eb in np.unique(eb_labels):
        eb_mask = (
            eb_labels == eb
        )
        reference_mask = (
            kd_mask
            & eb_mask
        )
        query_mask = (
            non_kd_mask
            & eb_mask
        )
        if (
            reference_mask.sum() < 1
            or query_mask.sum() < 1
        ):
            continue
        tree = KDTree(
            spatial_coords[
                reference_mask
            ]
        )
        eb_distances, _ = tree.query(
            spatial_coords[
                query_mask
            ]
        )
        distances[
            np.where(query_mask)[0]
        ] = eb_distances
    valid_mask = (
        non_kd_mask
        & np.isfinite(distances)
        & (distances >= 0)
        & (distances <= MAX_DISTANCE)
    )
    valid_indices = np.where(
        valid_mask
    )[0]
    if len(valid_indices) == 0:
        print(
            "No non-KD receiver cells within 300 um. Skipping."
        )
        continue
    distance_bins = assign_distance_bin(
        distances[
            valid_indices
        ]
    )
    cell_table = pd.DataFrame({
        "cell_index": valid_indices,
        "kd_gene": kd_gene,
        "EB": eb_labels[
            valid_indices
        ],
        "cell_type": cell_types[
            valid_indices
        ],
        "distance_um": distances[
            valid_indices
        ],
        "distance_100um": (
            distances[
                valid_indices
            ]
            / 100.0
        ),
        "bin_index": distance_bins,
    })
    cell_table = cell_table[
        cell_table["bin_index"] >= 0
    ].copy()
    cell_table["distance_bin"] = pd.Categorical(
        [
            BIN_LABELS[index]
            for index in cell_table["bin_index"]
        ],
        categories=BIN_LABELS,
        ordered=True
    )
    cell_table["bin_mid_um"] = (
        cell_table["bin_index"]
        .map(
            {
                index: midpoint
                for index, midpoint
                in enumerate(BIN_MIDS)
            }
        )
        .astype(float)
    )
    all_cell_level_tables.append(
        cell_table
    )
    print(f"KD cells: {kd_mask.sum()}")
    print(
        f"Receiver cells within 300 um: {len(cell_table)}"
    )
    print(
        f"Contributing EBs: {cell_table['EB'].nunique()}"
    )
    rows = []
    grouped = cell_table.groupby(
        [
            "EB",
            "bin_index",
            "distance_bin",
            "bin_mid_um",
        ],
        observed=True
    )
    for (
        eb,
        bin_index,
        distance_bin,
        bin_mid_um,
    ), group in grouped:
        total_cells = len(group)
        for cell_type in CELL_TYPES:
            cell_type_count = int(
                (
                    group["cell_type"]
                    == cell_type
                ).sum()
            )
            rows.append({
                "kd_gene": kd_gene,
                "EB": eb,
                "bin_index": int(
                    bin_index
                ),
                "distance_bin": str(
                    distance_bin
                ),
                "bin_mid_um": float(
                    bin_mid_um
                ),
                "cell_type": cell_type,
                "cell_type_count": cell_type_count,
                "total_cells": total_cells,
                "proportion": (
                    cell_type_count
                    / total_cells
                ),
            })
    kd_bin_summary = pd.DataFrame(
        rows
    )
    all_bin_summary_tables.append(
        kd_bin_summary
    )
if len(all_cell_level_tables) == 0:
    raise ValueError(
        "No KD target produced usable receiver cells."
    )
cell_level_data = pd.concat(
    all_cell_level_tables,
    ignore_index=True
)
bin_level_data = pd.concat(
    all_bin_summary_tables,
    ignore_index=True
)
cell_level_data.to_csv(
    OUTDIR
    / "all_KD_receiver_cells_with_distance_and_cell_type.csv",
    index=False
)
bin_level_data.to_csv(
    OUTDIR
    / "all_KD_EB_bin_cell_type_counts_and_proportions.csv",
    index=False
)
model_rows = []
for kd_gene in KDS_TO_ANALYZE:
    kd_cells = cell_level_data[
        cell_level_data["kd_gene"]
        == kd_gene
    ].copy()
    if len(kd_cells) < MIN_TOTAL_RECEIVER_CELLS:
        continue
    for cell_type in CELL_TYPES:
        model_data = kd_cells[
            [
                "EB",
                "distance_100um",
                "cell_type",
            ]
        ].copy()
        model_data["is_target_cell_type"] = (
            model_data["cell_type"]
            == cell_type
        ).astype(int)
        n_cells = len(
            model_data
        )
        n_ebs = model_data[
            "EB"
        ].nunique()
        n_positive = int(
            model_data[
                "is_target_cell_type"
            ].sum()
        )
        n_negative = (
            n_cells
            - n_positive
        )
        base_row = {
            "kd_gene": kd_gene,
            "cell_type": cell_type,
            "n_receiver_cells": n_cells,
            "n_EBs": n_ebs,
            "n_cell_type_cells": n_positive,
            "n_other_cells": n_negative,
        }
        if n_ebs < MIN_EBS:
            base_row["status"] = (
                "too_few_EBs"
            )
            model_rows.append(
                base_row
            )
            continue
        if n_positive < MIN_CELL_TYPE_POSITIVES:
            base_row["status"] = (
                "too_few_cell_type_cells"
            )
            model_rows.append(
                base_row
            )
            continue
        if n_negative < MIN_CELL_TYPE_NEGATIVES:
            base_row["status"] = (
                "too_few_other_cells"
            )
            model_rows.append(
                base_row
            )
            continue
        exog = pd.DataFrame({
            "intercept": np.ones(
                n_cells,
                dtype=float
            ),
            "distance_100um": model_data[
                "distance_100um"
            ].to_numpy(
                dtype=float
            ),
        })
        try:
            gee_model = GEE(
                endog=model_data[
                    "is_target_cell_type"
                ].to_numpy(
                    dtype=float
                ),
                exog=exog,
                groups=model_data[
                    "EB"
                ].astype(str).to_numpy(),
                family=Binomial(),
                cov_struct=Exchangeable(),
            )
            gee_result = gee_model.fit()
            beta = float(
                gee_result.params[
                    "distance_100um"
                ]
            )
            standard_error = float(
                gee_result.bse[
                    "distance_100um"
                ]
            )
            p_value = float(
                gee_result.pvalues[
                    "distance_100um"
                ]
            )
            odds_ratio_per_100um = float(
                np.exp(beta)
            )
            direction = (
                "increasing"
                if beta > 0
                else "decreasing"
            )
            base_row.update({
                "status": "ok",
                "log_odds_slope_per_100um": beta,
                "slope_SE": standard_error,
                "odds_ratio_per_100um": odds_ratio_per_100um,
                "pval": p_value,
                "direction": direction,
            })
        except Exception as error:
            base_row.update({
                "status": (
                    "model_failure"
                ),
                "error": (
                    f"{type(error).__name__}: "
                    f"{error}"
                ),
            })
        model_rows.append(
            base_row
        )
model_results = pd.DataFrame(
    model_rows
)
model_results["fdr"] = np.nan
successful_mask = (
    model_results["status"]
    == "ok"
)
if successful_mask.sum() > 0:
    model_results.loc[
        successful_mask,
        "fdr"
    ] = multipletests(
        model_results.loc[
            successful_mask,
            "pval"
        ].to_numpy(),
        method="fdr_bh"
    )[1]
model_results["significant"] = (
    model_results["fdr"]
    < FDR_THRESHOLD
)
model_results = model_results.sort_values(
    [
        "fdr",
        "pval",
        "kd_gene",
        "cell_type",
    ],
    na_position="last"
).reset_index(
    drop=True
)
model_results.to_csv(
    OUTDIR
    / "KD_cell_type_distance_GEE_results.csv",
    index=False
)
model_results[
    model_results["significant"]
].to_csv(
    OUTDIR
    / "KD_cell_type_distance_GEE_results_FDR_0.05.csv",
    index=False
)
plot_summary = (
    bin_level_data
    .groupby(
        [
            "kd_gene",
            "cell_type",
            "bin_index",
            "distance_bin",
            "bin_mid_um",
        ],
        observed=True,
        as_index=False
    )
    .agg(
        mean_proportion=(
            "proportion",
            "mean"
        ),
        median_proportion=(
            "proportion",
            "median"
        ),
        n_EBs=(
            "EB",
            "nunique"
        ),
        sem_proportion=(
            "proportion",
            safe_sem
        ),
        total_cell_type_cells=(
            "cell_type_count",
            "sum"
        ),
        total_cells=(
            "total_cells",
            "sum"
        ),
    )
)
plot_summary.to_csv(
    OUTDIR
    / "KD_cell_type_proportion_plot_summary.csv",
    index=False
)
for kd_gene in KDS_TO_ANALYZE:
    kd_plot = plot_summary[
        plot_summary["kd_gene"]
        == kd_gene
    ].copy()
    if kd_plot.empty:
        continue
    fig, ax = plt.subplots(
        figsize=(10, 7)
    )
    for cell_type in CELL_TYPES:
        subset = kd_plot[
            kd_plot["cell_type"]
            == cell_type
        ].sort_values(
            "bin_mid_um"
        )
        if subset.empty:
            continue
        ax.errorbar(
            subset["bin_mid_um"],
            subset["mean_proportion"],
            yerr=subset[
                "sem_proportion"
            ],
            marker="o",
            linewidth=2,
            capsize=3,
            label=cell_type,
        )
    ax.set_xlabel(
        f"Distance to nearest {kd_gene} KD cell (um)"
    )
    ax.set_ylabel(
        "Cell-type proportion within distance bin"
    )
    ax.set_title(
        (
            f"{kd_gene} KD: cell-type composition "
            "as a function of distance"
        )
    )
    ax.set_xticks(
        BIN_MIDS
    )
    ax.set_xticklabels(
        BIN_LABELS
    )
    ax.set_ylim(
        bottom=0
    )
    ax.grid(
        alpha=0.25
    )
    ax.legend(
        bbox_to_anchor=(
            1.02,
            1
        ),
        loc="upper left",
        frameon=False
    )
    fig.tight_layout()
    fig.savefig(
        OUTDIR
        / f"{kd_gene}_cell_type_proportions_by_distance.png",
        dpi=220,
        bbox_inches="tight"
    )
    plt.close(
        fig
    )
successful_results = model_results[
    model_results["status"]
    == "ok"
].copy()
if not successful_results.empty:
    slope_matrix = (
        successful_results
        .pivot(
            index="kd_gene",
            columns="cell_type",
            values="log_odds_slope_per_100um"
        )
        .reindex(
            index=KDS_TO_ANALYZE,
            columns=CELL_TYPES
        )
    )
    fdr_matrix = (
        successful_results
        .pivot(
            index="kd_gene",
            columns="cell_type",
            values="fdr"
        )
        .reindex(
            index=KDS_TO_ANALYZE,
            columns=CELL_TYPES
        )
    )
    slope_matrix.to_csv(
        OUTDIR
        / "KD_by_cell_type_log_odds_slope_matrix.csv"
    )
    fdr_matrix.to_csv(
        OUTDIR
        / "KD_by_cell_type_FDR_matrix.csv"
    )
    matrix_values = slope_matrix.to_numpy(
        dtype=float
    )
    finite_values = matrix_values[
        np.isfinite(
            matrix_values
        )
    ]
    if len(finite_values) > 0:
        max_abs = np.max(
            np.abs(
                finite_values
            )
        )
        if max_abs == 0:
            max_abs = 1.0
        fig_width = max(
            9,
            len(CELL_TYPES) * 1.5
        )
        fig_height = max(
            7,
            len(KDS_TO_ANALYZE) * 0.55
        )
        fig, ax = plt.subplots(
            figsize=(
                fig_width,
                fig_height
            )
        )
        image = ax.imshow(
            matrix_values,
            aspect="auto",
            cmap="coolwarm",
            vmin=-max_abs,
            vmax=max_abs,
        )
        ax.set_xticks(
            np.arange(
                len(CELL_TYPES)
            )
        )
        ax.set_xticklabels(
            CELL_TYPES,
            rotation=35,
            ha="right"
        )
        ax.set_yticks(
            np.arange(
                len(KDS_TO_ANALYZE)
            )
        )
        ax.set_yticklabels(
            KDS_TO_ANALYZE
        )
        ax.set_title(
            (
                "Cell-type composition trend with distance\n"
                "log-odds slope per 100 um"
            )
        )
        for row_index, kd_gene in enumerate(
            KDS_TO_ANALYZE
        ):
            for column_index, cell_type in enumerate(
                CELL_TYPES
            ):
                value = slope_matrix.loc[
                    kd_gene,
                    cell_type
                ]
                fdr = fdr_matrix.loc[
                    kd_gene,
                    cell_type
                ]
                if pd.isna(value):
                    label = ""
                else:
                    label = (
                        f"{value:.2f}"
                    )
                    if (
                        pd.notna(fdr)
                        and fdr < 0.05
                    ):
                        label += "*"
                ax.text(
                    column_index,
                    row_index,
                    label,
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black",
                )
        colorbar = fig.colorbar(
            image,
            ax=ax
        )
        colorbar.set_label(
            "Log-odds slope per 100 um"
        )
        fig.tight_layout()
        fig.savefig(
            OUTDIR
            / "all_KD_cell_type_distance_slope_heatmap.png",
            dpi=240,
            bbox_inches="tight"
        )
        plt.close(
            fig
        )
possible_term_columns = [
    "go_term",
    "term",
    "gene_set",
    "pathway",
    "name",
]
go_term_column = next(
    (
        column
        for column in possible_term_columns
        if column in go_sig.columns
    ),
    None
)
go_direction_summary_rows = []
for (
    kd_gene,
    go_direction,
), group in go_sig.groupby(
    [
        "kd_gene",
        "direction",
    ],
    observed=True
):
    row = {
        "kd_gene": kd_gene,
        "go_direction": go_direction,
        "n_significant_GO_terms": len(
            group
        ),
    }
    if go_term_column is not None:
        row["significant_GO_terms"] = "; ".join(
            group[
                go_term_column
            ]
            .astype(str)
            .drop_duplicates()
            .tolist()
        )
    for candidate_fdr_column in [
        "fdr",
        "go_fdr",
        "FDR",
    ]:
        if candidate_fdr_column in group.columns:
            row["minimum_GO_FDR"] = pd.to_numeric(
                group[
                    candidate_fdr_column
                ],
                errors="coerce"
            ).min()
            break
    go_direction_summary_rows.append(
        row
    )
go_direction_summary = pd.DataFrame(
    go_direction_summary_rows
)
go_direction_summary.to_csv(
    OUTDIR
    / "significant_GO_direction_summary_used_for_comparison.csv",
    index=False
)
successful_for_merge = model_results[
    model_results["status"]
    == "ok"
][
    [
        "kd_gene",
        "cell_type",
        "log_odds_slope_per_100um",
        "odds_ratio_per_100um",
        "pval",
        "fdr",
        "significant",
        "direction",
        "n_receiver_cells",
        "n_EBs",
    ]
].rename(
    columns={
        "direction": "cell_type_direction",
        "fdr": "cell_type_FDR",
        "pval": "cell_type_pval",
        "significant": "cell_type_significant",
    }
)
go_cell_type_comparison = (
    go_direction_summary
    .merge(
        successful_for_merge,
        on="kd_gene",
        how="left"
    )
)
go_cell_type_comparison[
    "direction_concordance"
] = np.where(
    (
        go_cell_type_comparison[
            "go_direction"
        ]
        == go_cell_type_comparison[
            "cell_type_direction"
        ]
    ),
    "same direction",
    "opposite direction"
)
go_cell_type_comparison[
    "composition_interpretation"
] = np.where(
    (
        go_cell_type_comparison[
            "cell_type_significant"
        ].fillna(False)
    )
    & (
        go_cell_type_comparison[
            "direction_concordance"
        ]
        == "same direction"
    ),
    (
        "Significant concordant composition change; "
        "the GO distance pattern may be partly explained "
        "by changing abundance of this cell type."
    ),
    np.where(
        (
            go_cell_type_comparison[
                "cell_type_significant"
            ].fillna(False)
        )
        & (
            go_cell_type_comparison[
                "direction_concordance"
            ]
            == "opposite direction"
        ),
        (
            "Significant but opposite composition change; "
            "this cell type does not provide a simple "
            "composition explanation for the GO direction."
        ),
        (
            "No significant cell-type composition trend; "
            "the GO distance pattern is more consistent "
            "with a within-cell-type transcriptional effect, "
            "although cell-type-specific GEX analysis is needed "
            "to confirm this."
        )
    )
)
go_cell_type_comparison.to_csv(
    OUTDIR
    / "GO_direction_vs_cell_type_composition_comparison.csv",
    index=False
)
go_cell_type_comparison[
    go_cell_type_comparison[
        "cell_type_significant"
    ].fillna(False)
].to_csv(
    OUTDIR
    / "GO_direction_vs_significant_cell_type_composition_changes.csv",
    index=False
)
print("\nAnalysis complete.")
print("Output directory:")
print(OUTDIR.resolve())
print(
    "\nSignificant KD x cell-type composition trends:"
)
significant_models = model_results[
    model_results["significant"]
]
if significant_models.empty:
    print(
        "None at FDR < 0.05."
    )
else:
    print(
        significant_models[
            [
                "kd_gene",
                "cell_type",
                "direction",
                "odds_ratio_per_100um",
                "fdr",
            ]
        ].to_string(
            index=False
        )
    )
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
INPUT_FILE = Path(
    "MULTIOME_KD_distance_cell_type_composition_and_GO/"
    "all_KD_EB_bin_cell_type_counts_and_proportions.csv"
)
OUTDIR = Path(
    "MULTIOME_KD_distance_cell_type_composition_and_GO/"
    "publication_style_selected_plots"
)
OUTDIR.mkdir(
    parents=True,
    exist_ok=True
)
if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Input file not found:\n{INPUT_FILE.resolve()}"
    )
bin_level_data = pd.read_csv(
    INPUT_FILE
)
required_columns = {
    "kd_gene",
    "EB",
    "bin_index",
    "distance_bin",
    "bin_mid_um",
    "cell_type",
    "proportion",
}
missing_columns = required_columns - set(bin_level_data.columns)
if missing_columns:
    raise ValueError(
        "Input CSV is missing required columns: "
        f"{sorted(missing_columns)}"
    )
BIN_LABELS = [
    "0-50",
    "50-100",
    "100-150",
    "150-200",
    "200-250",
    "250-300",
]
BIN_MIDS = np.array(
    [25, 75, 125, 175, 225, 275],
    dtype=float
)
plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "axes.linewidth": 1.1,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})
def safe_sem(values):
    values = np.asarray(
        values,
        dtype=float
    )
    values = values[
        np.isfinite(values)
    ]
    if len(values) <= 1:
        return np.nan
    return (
        values.std(ddof=1)
        / np.sqrt(len(values))
    )
plot_summary = (
    bin_level_data
    .groupby(
        [
            "kd_gene",
            "cell_type",
            "bin_index",
            "distance_bin",
            "bin_mid_um",
        ],
        observed=True,
        as_index=False
    )
    .agg(
        mean_proportion=(
            "proportion",
            "mean"
        ),
        sem_proportion=(
            "proportion",
            safe_sem
        ),
        n_EBs=(
            "EB",
            "nunique"
        ),
    )
)
PLOT_GROUPS = [
    {
        "cell_type": "Neural Crest",
        "kd_genes": [
            "ALB",
            "BMP5",
            "DLL3",
            "EFNA3",
            "NECTIN2",
            "NECTIN3",
            "SEMA6D",
            "VEGFB",
        ],
        "title": "Neural Crest",
        "filename": (
            "Neural_Crest_proportion_"
            "ALB_BMP5_DLL3_EFNA3_NECTIN2_NECTIN3_SEMA6D_VEGFB"
        ),
    },
    {
        "cell_type": "Early Ectoderm",
        "kd_genes": [
            "NECTIN2",
            "SEMA6D",
        ],
        "title": "Early Ectoderm",
        "filename": (
            "Early_Ectoderm_proportion_NECTIN2_SEMA6D"
        ),
    },
    {
        "cell_type": "Early Ectoderm",
        "kd_genes": [
            "WNT5A",
            "EFNA3",
        ],
        "title": "Early Ectoderm",
        "filename": (
            "Early_Ectoderm_proportion_WNT5A_EFNA3"
        ),
    },
]
for plot_group in PLOT_GROUPS:
    target_cell_type = plot_group["cell_type"]
    selected_kds = plot_group["kd_genes"]
    selected_data = plot_summary[
        (
            plot_summary["cell_type"]
            == target_cell_type
        )
        & (
            plot_summary["kd_gene"]
            .isin(selected_kds)
        )
    ].copy()
    if selected_data.empty:
        print(
            f"No data found for {target_cell_type}: "
            f"{selected_kds}"
        )
        continue
    fig, ax = plt.subplots(
        figsize=(7.2, 5.2)
    )
    plotted_any = False
    for kd_gene in selected_kds:
        subset = selected_data[
            selected_data["kd_gene"]
            == kd_gene
        ].sort_values(
            "bin_mid_um"
        )
        if subset.empty:
            print(
                f"Skipping {kd_gene}: no "
                f"{target_cell_type} data."
            )
            continue
        plotted_any = True
        ax.errorbar(
            subset["bin_mid_um"],
            subset["mean_proportion"],
            yerr=subset["sem_proportion"],
            marker="o",
            markersize=5,
            linewidth=1.8,
            elinewidth=1.0,
            capsize=2.5,
            capthick=1.0,
            label=kd_gene,
        )
    if not plotted_any:
        plt.close(fig)
        continue
    ax.set_xlabel(
        "Distance to nearest KD cell (µm)"
    )
    ax.set_ylabel(
        f"{target_cell_type} proportion"
    )
    ax.set_title(
        plot_group["title"],
        fontweight="bold",
        pad=10
    )
    ax.set_xticks(
        BIN_MIDS
    )
    ax.set_xticklabels(
        BIN_LABELS
    )
    ax.set_xlim(
        10,
        290
    )
    ax.set_ylim(
        bottom=0
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(
        axis="both",
        direction="out"
    )
    ax.grid(
        axis="y",
        linewidth=0.7,
        alpha=0.2
    )
    ax.legend(
        title="KD",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=False,
        borderaxespad=0
    )
    fig.tight_layout()
    fig.savefig(
        OUTDIR / f"{plot_group['filename']}.png",
        dpi=600,
        bbox_inches="tight",
        facecolor="white"
    )
    fig.savefig(
        OUTDIR / f"{plot_group['filename']}.pdf",
        bbox_inches="tight",
        facecolor="white"
    )
    plt.close(fig)
print(
    "Finished. Figures saved to:"
)
print(
    OUTDIR.resolve()
)
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
INPUT_FILE=Path("MULTIOME_KD_distance_cell_type_composition_and_GO/all_KD_EB_bin_cell_type_counts_and_proportions.csv")
OUTDIR=Path("MULTIOME_KD_distance_cell_type_composition_and_GO/publication_style_selected_plots")
OUTDIR.mkdir(parents=True,exist_ok=True)
if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Input file not found:\n{INPUT_FILE.resolve()}")
bin_level_data=pd.read_csv(INPUT_FILE)
required_columns={"kd_gene","EB","bin_index","distance_bin","bin_mid_um","cell_type","proportion"}
missing_columns=required_columns-set(bin_level_data.columns)
if missing_columns:
    raise ValueError("Input CSV is missing required columns: "+f"{sorted(missing_columns)}")
BIN_LABELS=["0-50","50-100","100-150","150-200","200-250","250-300"]
BIN_MIDS=np.array([25,75,125,175,225,275],dtype=float)
plt.rcParams.update({"font.family":"Arial","font.size":11,"axes.titlesize":14,"axes.labelsize":12,"xtick.labelsize":10,"ytick.labelsize":10,"legend.fontsize":9,"axes.linewidth":1.1,"xtick.major.width":1.0,"ytick.major.width":1.0,"xtick.major.size":4,"ytick.major.size":4,"pdf.fonttype":42,"ps.fonttype":42})
def safe_sem(values):
    values=np.asarray(values,dtype=float)
    values=values[np.isfinite(values)]
    if len(values)<=1:
        return np.nan
    return values.std(ddof=1)/np.sqrt(len(values))
plot_summary=(bin_level_data.groupby(["kd_gene","cell_type","bin_index","distance_bin","bin_mid_um"],observed=True,as_index=False).agg(mean_proportion=("proportion","mean"),sem_proportion=("proportion",safe_sem),n_EBs=("EB","nunique")))
PLOT_GROUPS=[
    {"cell_type":"Pluripotent","kd_genes":["EFNA3","WNT5A"],"title":"Pluripotent","filename":"Pluripotent_proportion_EFNA3_WNT5A"},
    {"cell_type":"Early Ectoderm","kd_genes":["EFNA3","WNT5A"],"title":"Early Ectoderm","filename":"Early_Ectoderm_proportion_EFNA3_WNT5A"},
    {"cell_type":"Early Ectoderm","kd_genes":["NECTIN2","SEMA6D"],"title":"Early Ectoderm","filename":"Early_Ectoderm_proportion_NECTIN2_SEMA6D"},
    {"cell_type":"Neural Crest","kd_genes":["NECTIN2","SEMA6D"],"title":"Neural Crest","filename":"Neural_Crest_proportion_NECTIN2_SEMA6D"},
]
for plot_group in PLOT_GROUPS:
    target_cell_type=plot_group["cell_type"]
    selected_kds=plot_group["kd_genes"]
    selected_data=plot_summary[(plot_summary["cell_type"]==target_cell_type)&(plot_summary["kd_gene"].isin(selected_kds))].copy()
    if selected_data.empty:
        print(f"No data found for {target_cell_type}: {selected_kds}")
        continue
    fig,ax=plt.subplots(figsize=(7.2,5.2))
    plotted_any=False
    for kd_gene in selected_kds:
        subset=selected_data[selected_data["kd_gene"]==kd_gene].sort_values("bin_mid_um")
        if subset.empty:
            print(f"Skipping {kd_gene}: no {target_cell_type} data.")
            continue
        plotted_any=True
        ax.errorbar(subset["bin_mid_um"],subset["mean_proportion"],yerr=subset["sem_proportion"],marker="o",markersize=5,linewidth=1.8,elinewidth=1.0,capsize=2.5,capthick=1.0,label=kd_gene)
    if not plotted_any:
        plt.close(fig)
        continue
    ax.set_xlabel("Distance to nearest KD cell (µm)")
    ax.set_ylabel(f"{target_cell_type} proportion")
    ax.set_title(plot_group["title"],fontweight="bold",pad=10)
    ax.set_xticks(BIN_MIDS)
    ax.set_xticklabels(BIN_LABELS)
    ax.set_xlim(10,290)
    ax.set_ylim(bottom=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both",direction="out")
    ax.grid(axis="y",linewidth=0.7,alpha=0.2)
    ax.legend(title="KD",bbox_to_anchor=(1.02,1),loc="upper left",frameon=False,borderaxespad=0)
    fig.tight_layout()
    fig.savefig(OUTDIR/f"{plot_group['filename']}.png",dpi=600,bbox_inches="tight",facecolor="white")
    fig.savefig(OUTDIR/f"{plot_group['filename']}.pdf",bbox_inches="tight",facecolor="white")
    plt.close(fig)
print("Finished. Figures saved to:")
print(OUTDIR.resolve())
from pathlib import Path
import re
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.backends.backend_pdf import PdfPages
import mudata as md
H5MU_PATH=Path("all_lanes_merged_CATATAC_QC_RNA_processed_EB_manual_clustering_cell_types.h5mu")
OUTDIR=Path("MULTIOME_KD_distance_cell_type_composition_and_GO/publication_spatial_maps_combined_KDs")
OUTDIR.mkdir(parents=True,exist_ok=True)
RNA_MODALITY="rna"
GUIDE_COL="guides_passing_str"
EB_COL="EB_manual_clustering"
CELL_TYPE_COL="cell_type"
SPATIAL_KEY="X_spatial"
TARGET_CELL_TYPE="Pluripotent"
BACKGROUND_COLOR="#D9D9D9"
CELL_TYPE_COLOR="#009E73"
KD1_COLOR="#E41A1C"
KD2_COLOR="#984EA3"
BACKGROUND_SIZE=3.5
CELL_TYPE_SIZE=6
KD_SIZE=9
SCALE_BAR_UM=100
MAX_COLS=4
SAVE_DPI=300
ANNOTATION_SPACE_FRAC=0.26
plt.rcParams.update({"font.family":"Arial","font.size":9,"axes.titlesize":10,"pdf.fonttype":42,"ps.fonttype":42})
PLOT_GROUPS=[
    {"kd_genes":["NECTIN2","SEMA6D"],"title":"Pluripotent — NECTIN2 and SEMA6D KD","filename":"Pluripotent_NECTIN2_SEMA6D_all_EBs"},
    {"kd_genes":["EFNA3","WNT5A"],"title":"Pluripotent — EFNA3 and WNT5A KD","filename":"Pluripotent_EFNA3_WNT5A_all_EBs"},
]
def parse_guides(x):
    if pd.isna(x):
        return []
    x=str(x).strip()
    if x=="" or x.lower() in {"nan","none","na"}:
        return []
    return [g.strip() for g in re.split(r"[;,|\s]+",x) if g.strip()]
def guide_to_target(g):
    return re.sub(r"-\d+$","",str(g).strip())
def is_kd_for_gene(x,gene):
    return gene in {guide_to_target(g) for g in parse_guides(x)}
def normalize_cell_type(x):
    return str(x).strip()
def get_combined_kd_ebs(kd_masks,eb_labels):
    combined=np.logical_or.reduce(kd_masks)
    return sorted([eb for eb in np.unique(eb_labels) if np.sum(combined & (eb_labels==eb))>0],key=str)
def add_scale_bar(ax,x,y,length_um):
    ax.plot([x,x+length_um],[y,y],color="black",linewidth=1.8,solid_capstyle="butt",clip_on=False)
    ax.text(x+length_um/2,y-8,f"{length_um} µm",ha="center",va="top",fontsize=7,clip_on=False)
def make_combined_kd_figure(kd_genes,target_cell_type,guide_strings,eb_labels,cell_types,coords):
    kd1,kd2=kd_genes
    kd1_mask=np.array([is_kd_for_gene(x,kd1) for x in guide_strings],dtype=bool)
    kd2_mask=np.array([is_kd_for_gene(x,kd2) for x in guide_strings],dtype=bool)
    combined_kd_mask=kd1_mask | kd2_mask
    target_mask=cell_types==target_cell_type
    kd_ebs=get_combined_kd_ebs([kd1_mask,kd2_mask],eb_labels)
    if len(kd_ebs)==0:
        print(f"{kd1} + {kd2}: no EBs contain KD cells.")
        return None,None
    n_panels=len(kd_ebs)
    ncols=min(MAX_COLS,n_panels)
    nrows=math.ceil(n_panels/ncols)
    fig,axes=plt.subplots(nrows,ncols,figsize=(3.0*ncols,3.05*nrows),squeeze=False)
    axes=axes.ravel()
    for i,eb in enumerate(kd_ebs):
        ax=axes[i]
        eb_mask=eb_labels==eb
        kd1_eb=kd1_mask & eb_mask
        kd2_eb=kd2_mask & eb_mask
        combined_kd_eb=combined_kd_mask & eb_mask
        target_eb=target_mask & eb_mask
        other_mask=eb_mask & ~target_eb & ~combined_kd_eb
        target_only=target_eb & ~combined_kd_eb
        kd1_only=kd1_eb & ~kd2_eb
        kd2_only=kd2_eb & ~kd1_eb
        both_kd=kd1_eb & kd2_eb
        ax.scatter(coords[other_mask,0],coords[other_mask,1],s=BACKGROUND_SIZE,c=BACKGROUND_COLOR,alpha=0.5,linewidths=0,rasterized=True)
        ax.scatter(coords[target_only,0],coords[target_only,1],s=CELL_TYPE_SIZE,c=CELL_TYPE_COLOR,alpha=0.85,linewidths=0,rasterized=True)
        ax.scatter(coords[kd1_only,0],coords[kd1_only,1],s=KD_SIZE,c=KD1_COLOR,alpha=0.95,linewidths=0,rasterized=True)
        ax.scatter(coords[kd2_only,0],coords[kd2_only,1],s=KD_SIZE,c=KD2_COLOR,alpha=0.95,linewidths=0,rasterized=True)
        if both_kd.sum()>0:
            ax.scatter(coords[both_kd,0],coords[both_kd,1],s=KD_SIZE+2,c=KD1_COLOR,edgecolors=KD2_COLOR,linewidths=0.8,alpha=1.0,rasterized=True)
        xvals=coords[eb_mask,0]
        yvals=coords[eb_mask,1]
        xmin,xmax=np.nanmin(xvals),np.nanmax(xvals)
        ymin,ymax=np.nanmin(yvals),np.nanmax(yvals)
        xrange=xmax-xmin
        yrange=ymax-ymin
        pad_x=max(xrange*0.04,10)
        pad_y=max(yrange*0.04,10)
        annotation_space=max(yrange*ANNOTATION_SPACE_FRAC,40)
        scale_space=max(yrange*0.12,35)
        ax.set_xlim(xmin-pad_x,xmax+pad_x)
        ax.set_ylim(ymin-pad_y-scale_space,ymax+pad_y+annotation_space)
        if xrange>=SCALE_BAR_UM*1.2:
            scale_x=xmin+0.06*xrange
            scale_y=ymin-pad_y-scale_space*0.35
            add_scale_bar(ax,scale_x,scale_y,SCALE_BAR_UM)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(f"EB {eb}",fontweight="bold",pad=3)
        ax.text(0.02,0.97,f"{kd1}: {kd1_eb.sum()}\n{kd2}: {kd2_eb.sum()}\n{target_cell_type}: {target_eb.sum()}",transform=ax.transAxes,ha="left",va="top",fontsize=6.8)
        print(f"{kd1}+{kd2} | EB {eb} | {kd1}={kd1_eb.sum()} | {kd2}={kd2_eb.sum()} | {target_cell_type}={target_eb.sum()}")
    for j in range(n_panels,len(axes)):
        axes[j].axis("off")
    legend_handles=[
        mlines.Line2D([],[],marker="o",linestyle="None",markersize=4.5,markerfacecolor=BACKGROUND_COLOR,markeredgecolor="none",label="Other cells"),
        mlines.Line2D([],[],marker="o",linestyle="None",markersize=4.5,markerfacecolor=CELL_TYPE_COLOR,markeredgecolor="none",label=target_cell_type),
        mlines.Line2D([],[],marker="o",linestyle="None",markersize=4.5,markerfacecolor=KD1_COLOR,markeredgecolor="none",label=f"{kd1} KD"),
        mlines.Line2D([],[],marker="o",linestyle="None",markersize=4.5,markerfacecolor=KD2_COLOR,markeredgecolor="none",label=f"{kd2} KD"),
    ]
    fig.legend(handles=legend_handles,loc="upper center",bbox_to_anchor=(0.5,0.975),ncol=4,frameon=False,handletextpad=0.3,columnspacing=1.2)
    fig.suptitle(f"{target_cell_type} — {kd1} and {kd2} KD",fontsize=13,fontweight="bold",y=0.995)
    fig.tight_layout(rect=[0,0,1,0.93])
    return fig,kd_ebs
if not H5MU_PATH.exists():
    raise FileNotFoundError(f"MuData file not found: {H5MU_PATH.resolve()}")
mdata=md.read_h5mu(H5MU_PATH)
if RNA_MODALITY not in mdata.mod:
    raise ValueError(f"Missing modality: {RNA_MODALITY}")
adata=mdata[RNA_MODALITY]
for col in [GUIDE_COL,EB_COL,CELL_TYPE_COL]:
    if col not in adata.obs.columns:
        raise ValueError(f"Missing mdata['rna'].obs['{col}']")
if SPATIAL_KEY not in adata.obsm:
    raise ValueError(f"Missing mdata['rna'].obsm['{SPATIAL_KEY}']")
guide_strings=adata.obs[GUIDE_COL].fillna("").astype(str).to_numpy()
eb_labels=adata.obs[EB_COL].astype(str).to_numpy()
cell_types=adata.obs[CELL_TYPE_COL].map(normalize_cell_type).to_numpy()
coords=np.asarray(adata.obsm[SPATIAL_KEY][:,:2],dtype=float)
print(f"Loaded {adata.n_obs:,} cells.")
summary_rows=[]
for group in PLOT_GROUPS:
    kd_genes=group["kd_genes"]
    kd1,kd2=kd_genes
    group_dir=OUTDIR/group["filename"]
    group_dir.mkdir(parents=True,exist_ok=True)
    fig,kd_ebs=make_combined_kd_figure(kd_genes,TARGET_CELL_TYPE,guide_strings,eb_labels,cell_types,coords)
    if fig is None:
        continue
    png_file=group_dir/f"{group['filename']}.png"
    pdf_file=group_dir/f"{group['filename']}.pdf"
    fig.savefig(png_file,dpi=SAVE_DPI,bbox_inches="tight",facecolor="white")
    fig.savefig(pdf_file,dpi=SAVE_DPI,bbox_inches="tight",facecolor="white")
    plt.close(fig)
    summary_rows.append({"cell_type":TARGET_CELL_TYPE,"kd_1":kd1,"kd_2":kd2,"n_EBs":len(kd_ebs),"EBs":";".join(map(str,kd_ebs))})
    print("Saved:",png_file)
    print("Saved:",pdf_file)
summary_df=pd.DataFrame(summary_rows)
summary_df.to_csv(OUTDIR/"combined_KD_spatial_plot_EB_summary_Pluripotent.csv",index=False)
print("\nFinished.")
print("Figures saved to:",OUTDIR.resolve())
from pathlib import Path
import math
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import hypergeom
from statsmodels.stats.multitest import multipletests
GP_OUTDIR=Path("GP_ligand_KD_distance_effect_on_GEX_50um_bins")
GO_GMT_PATH=Path("/scratch/welchjd_root/welchjd1/javidgmh/13701-JG_300gRNA_100Perturbed_EB/c5.go.bp.v2026.1.Hs.symbols.gmt")
GO_OUTDIR=GP_OUTDIR/"TARGETED_GO_EFNA3_WNT5A"
GO_OUTDIR.mkdir(parents=True,exist_ok=True)
KDS_TO_ANALYZE={"EFNA3","WNT5A"}
GP_FDR_THRESHOLD=0.05
GO_FDR_THRESHOLD=0.05
MIN_GENES_PER_GO_TERM=5
MAX_GENES_PER_GO_TERM=500
MIN_OVERLAP_GENES=2
TOP_TERMS_TO_PLOT=20
TARGET_THEMES={
"Pluripotency":[
"PLURIPOTEN",
"EMBRYONIC_STEM_CELL",
"STEM_CELL_MAINTENANCE",
"STEM_CELL_POPULATION_MAINTENANCE",
"MAINTENANCE_OF_STEM_CELL",
"STEM_CELL_DIFFERENTIATION",
"STEM_CELL_FATE",
"STEM_CELL_PROLIFERATION"
],
"Ectoderm":[
"ECTODERM",
"ECTODERMAL"
],
"Neuroectoderm_neural_plate":[
"NEUROECTODERM",
"NEURAL_PLATE",
"NEURAL_TUBE",
"NEURAL_FOLD"
],
"Neural_progenitor":[
"NEURAL_PRECURSOR",
"NEURAL_PROGENITOR",
"NEURAL_STEM_CELL",
"NEURAL_STEM",
"NEUROBLAST"
],
"Neural_differentiation":[
"NEURON_DIFFERENTIATION",
"NEURON_FATE",
"NEUROGENESIS",
"FOREBRAIN_DEVELOPMENT",
"BRAIN_DEVELOPMENT",
"NEURON_MIGRATION",
"AXON_DEVELOPMENT",
"NEURON_PROJECTION",
"NEURON_MORPHOGENESIS"
],
"BMP_signaling":[
"BMP_SIGNALING",
"BONE_MORPHOGENETIC_PROTEIN_SIGNALING"
],
"WNT_signaling":[
"WNT_SIGNALING",
"CANONICAL_WNT",
"NON_CANONICAL_WNT"
],
"FGF_signaling":[
"FGF_RECEPTOR_SIGNALING",
"FIBROBLAST_GROWTH_FACTOR_RECEPTOR_SIGNALING"
],
"Cell_fate_differentiation":[
"CELL_FATE_COMMITMENT",
"CELL_FATE_SPECIFICATION",
"CELL_FATE_DETERMINATION",
"GERM_LAYER",
"EMBRYONIC_MORPHOGENESIS"
]
}
def read_gmt(gmt_path):
    gene_sets={}
    with open(gmt_path,"r") as handle:
        for line in handle:
            fields=line.rstrip("\n").split("\t")
            if len(fields)<3:
                continue
            term=fields[0]
            genes={str(g).strip().upper() for g in fields[2:] if str(g).strip()}
            if genes:
                gene_sets[term]=genes
    return gene_sets
def clean_term(term):
    x=str(term).upper()
    x=x.replace("GOBP_","")
    x=x.replace("GO_","")
    return x
def select_targeted_sets(gene_sets):
    selected={}
    theme_map={}
    for term,genes in gene_sets.items():
        clean=clean_term(term)
        matched=[]
        for theme,patterns in TARGET_THEMES.items():
            if any(pattern in clean for pattern in patterns):
                matched.append(theme)
        if matched:
            selected[term]=genes
            theme_map[term]=";".join(matched)
    return selected,theme_map
def run_go_ora(query_genes,background_genes,gene_sets):
    query={str(g).strip().upper() for g in query_genes if str(g).strip()}
    background={str(g).strip().upper() for g in background_genes if str(g).strip()}
    query&=background
    M=len(background)
    N=len(query)
    if M==0 or N==0:
        return pd.DataFrame()
    rows=[]
    for term,term_genes in gene_sets.items():
        term_bg=term_genes&background
        K=len(term_bg)
        if K<MIN_GENES_PER_GO_TERM or K>MAX_GENES_PER_GO_TERM:
            continue
        overlap=query&term_bg
        k=len(overlap)
        if k<MIN_OVERLAP_GENES:
            continue
        pval=float(hypergeom.sf(k-1,M,K,N))
        expected=N*K/M
        enrichment_ratio=k/expected if expected>0 else np.nan
        a=k
        b=N-k
        c=K-k
        d=M-K-N+k
        odds_ratio=(a*d)/(b*c) if b>0 and c>0 else np.inf
        rows.append({
            "go_term":term,
            "overlap_count":k,
            "query_gene_count":N,
            "term_background_count":K,
            "background_gene_count":M,
            "expected_overlap":expected,
            "enrichment_ratio":enrichment_ratio,
            "odds_ratio":odds_ratio,
            "pval":pval,
            "overlap_genes":";".join(sorted(overlap))
        })
    if not rows:
        return pd.DataFrame()
    df=pd.DataFrame(rows)
    df["fdr"]=multipletests(df["pval"].to_numpy(),method="fdr_bh")[1]
    return df.sort_values(["fdr","pval","enrichment_ratio"],ascending=[True,True,False]).reset_index(drop=True)
def pretty_term(term):
    return clean_term(term).replace("_"," ").title()
def plot_targeted_go(df,kd,direction,outfile):
    sig=df[df["fdr"]<GO_FDR_THRESHOLD].copy()
    if sig.empty:
        return
    plot_df=sig.head(TOP_TERMS_TO_PLOT).iloc[::-1].copy()
    plot_df["score"]=-np.log10(np.maximum(plot_df["fdr"].astype(float),1e-300))
    plot_df["label"]=plot_df["go_term"].map(pretty_term)
    fig,ax=plt.subplots(figsize=(10,max(5,0.42*len(plot_df)+1.5)))
    ax.barh(plot_df["label"],plot_df["score"])
    ax.axvline(-math.log10(GO_FDR_THRESHOLD),linestyle="--",linewidth=1)
    location="FARTHER FROM KD" if direction=="increasing" else "NEAR KD"
    ax.set_xlabel("-log10(FDR)")
    ax.set_ylabel("")
    ax.set_title(f"{kd} KD: targeted developmental programs {location}\nGenes {direction} with distance")
    ax.grid(axis="x",alpha=0.25)
    fig.tight_layout()
    fig.savefig(outfile,dpi=300,bbox_inches="tight")
    plt.close(fig)
if not GO_GMT_PATH.exists():
    raise FileNotFoundError(GO_GMT_PATH)
print("Loading GO gene sets...")
all_go_sets=read_gmt(GO_GMT_PATH)
targeted_go_sets,theme_map=select_targeted_sets(all_go_sets)
print("All GO BP terms:",len(all_go_sets))
print("Targeted GO BP terms:",len(targeted_go_sets))
selected_terms=pd.DataFrame({
"go_term":list(targeted_go_sets.keys()),
"theme":[theme_map[x] for x in targeted_go_sets]
})
selected_terms["clean_name"]=selected_terms["go_term"].map(pretty_term)
selected_terms=selected_terms.sort_values(["theme","clean_name"])
selected_terms.to_csv(GO_OUTDIR/"TARGETED_GO_TERMS_TESTED.csv",index=False)
print("\nTargeted terms that will be tested:")
print(selected_terms[["theme","clean_name"]].to_string(index=False))
all_results=[]
summary=[]
for kd in sorted(KDS_TO_ANALYZE):
    sig_file=GP_OUTDIR/f"{kd}_significant_GEX_GP_results_FDR_0.05.csv"
    gp_file=GP_OUTDIR/f"{kd}_GP_inputs_50um_bins.npz"
    if not sig_file.exists():
        print(f"\nMissing {sig_file}; skipping {kd}")
        continue
    if not gp_file.exists():
        print(f"\nMissing {gp_file}; skipping {kd}")
        continue
    print("\n"+"="*80)
    print(kd)
    print("="*80)
    sig=pd.read_csv(sig_file)
    sig=sig[sig["fdr"]<GP_FDR_THRESHOLD].copy()
    sig["direction"]=sig["direction"].astype(str).str.lower()
    with np.load(gp_file,allow_pickle=True) as x:
        background=[str(g) for g in x["gene_names"]]
    kd_dir=GO_OUTDIR/kd
    kd_dir.mkdir(parents=True,exist_ok=True)
    for direction in ["decreasing","increasing"]:
        query=sig.loc[sig["direction"]==direction,"gene"].dropna().astype(str).unique().tolist()
        location="near_KD" if direction=="decreasing" else "far_from_KD"
        print(f"{direction} ({location}): {len(query)} genes")
        res=run_go_ora(query,background,targeted_go_sets)
        if res.empty:
            print("No targeted GO results.")
            continue
        res.insert(0,"theme",res["go_term"].map(theme_map))
        res.insert(0,"spatial_interpretation",location)
        res.insert(0,"direction",direction)
        res.insert(0,"kd_gene",kd)
        res["go_label"]=res["go_term"].map(pretty_term)
        res.to_csv(kd_dir/f"{kd}_{direction}_TARGETED_GO_all.csv",index=False)
        sig_res=res[res["fdr"]<GO_FDR_THRESHOLD].copy()
        sig_res.to_csv(kd_dir/f"{kd}_{direction}_TARGETED_GO_FDR_0.05.csv",index=False)
        plot_targeted_go(res,kd,direction,kd_dir/f"{kd}_{direction}_TARGETED_GO.png")
        print(f"Tested: {len(res)} | FDR<0.05: {len(sig_res)}")
        if not sig_res.empty:
            print(sig_res[["theme","go_label","overlap_count","enrichment_ratio","fdr"]].head(20).to_string(index=False))
        all_results.append(res)
        summary.append({
            "kd_gene":kd,
            "direction":direction,
            "spatial_interpretation":location,
            "n_query_genes":len(query),
            "n_targeted_terms_tested":len(res),
            "n_significant_targeted_terms":len(sig_res)
        })
if all_results:
    combined=pd.concat(all_results,ignore_index=True)
    combined.to_csv(GO_OUTDIR/"EFNA3_WNT5A_TARGETED_GO_all.csv",index=False)
    combined[combined["fdr"]<GO_FDR_THRESHOLD].to_csv(GO_OUTDIR/"EFNA3_WNT5A_TARGETED_GO_FDR_0.05.csv",index=False)
if summary:
    pd.DataFrame(summary).to_csv(GO_OUTDIR/"TARGETED_GO_summary.csv",index=False)
print("\nDone.")
print("Results:",GO_OUTDIR.resolve())
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
try:
    import gseapy as gp
except ImportError:
    raise ImportError("Install gseapy first: pip install gseapy")
GP_OUTDIR=Path("GP_ligand_KD_distance_effect_on_GEX_50um_bins")
GO_GMT_PATH=Path("/scratch/welchjd_root/welchjd1/javidgmh/13701-JG_300gRNA_100Perturbed_EB/c5.go.bp.v2026.1.Hs.symbols.gmt")
OUTDIR=GP_OUTDIR/"TARGETED_GSEA_EFNA3_WNT5A"
OUTDIR.mkdir(parents=True,exist_ok=True)
KDS=["EFNA3","WNT5A"]
FDR_CUTOFF=0.05
MIN_SIZE=5
MAX_SIZE=500
PERMUTATIONS=1000
SEED=1
MAX_TERMS_TO_PLOT=15
TARGET_THEMES={
"Pluripotency / stem cell":[
"PLURIPOTEN",
"EMBRYONIC_STEM",
"STEM_CELL_MAINTENANCE",
"MAINTENANCE_OF_STEM_CELL",
"STEM_CELL_DIFFERENTIATION",
"STEM_CELL_FATE"
],
"Ectoderm / neuroectoderm":[
"ECTODERM",
"NEUROECTODERM",
"NEURAL_PLATE",
"NEURAL_TUBE",
"NEURAL_FOLD"
],
"Neural precursor":[
"NEURAL_PRECURSOR",
"NEURAL_PROGENITOR",
"NEURAL_STEM",
"NEUROBLAST"
],
"Neural differentiation":[
"NEURON_DIFFERENTIATION",
"NEURON_FATE",
"NEUROGENESIS",
"FOREBRAIN_DEVELOPMENT",
"BRAIN_DEVELOPMENT"
],
"Migration / neurite development":[
"NEURON_MIGRATION",
"AXON_DEVELOPMENT",
"AXON_GUIDANCE",
"NEURON_PROJECTION",
"NEURON_MORPHOGENESIS"
],
"WNT signaling":[
"WNT_SIGNALING",
"CANONICAL_WNT",
"NON_CANONICAL_WNT",
"PLANAR_CELL_POLARITY"
],
"BMP signaling":[
"BMP_SIGNALING",
"BONE_MORPHOGENETIC_PROTEIN"
],
"Cell fate / embryonic development":[
"CELL_FATE_COMMITMENT",
"CELL_FATE_SPECIFICATION",
"CELL_FATE_DETERMINATION",
"GERM_LAYER",
"EMBRYONIC_MORPHOGENESIS"
]
}
def read_gmt(path):
    gene_sets={}
    with open(path,"r") as f:
        for line in f:
            x=line.rstrip("\n").split("\t")
            if len(x)>=3:
                gene_sets[x[0]]={g.upper() for g in x[2:] if g}
    return gene_sets
def clean_term(term):
    x=str(term).replace("GOBP_","").replace("GO_","").replace("_"," ").lower()
    return x[0].upper()+x[1:] if x else x
def select_targeted_sets(gene_sets):
    selected={}
    themes={}
    for term,genes in gene_sets.items():
        u=term.upper()
        for theme,patterns in TARGET_THEMES.items():
            if any(p in u for p in patterns):
                selected[term]=genes
                themes[term]=theme
                break
    return selected,themes
if not GO_GMT_PATH.exists():
    raise FileNotFoundError(GO_GMT_PATH)
go_all=read_gmt(GO_GMT_PATH)
target_sets,theme_map=select_targeted_sets(go_all)
print("Targeted gene sets:",len(target_sets))
pd.DataFrame({
    "GO_term":list(target_sets),
    "GO_process":[clean_term(x) for x in target_sets],
    "Theme":[theme_map[x] for x in target_sets]
}).to_csv(OUTDIR/"targeted_gene_sets_tested.csv",index=False)
all_results=[]
for kd in KDS:
    infile=GP_OUTDIR/f"{kd}_all_GEX_GP_results.csv"
    if not infile.exists():
        raise FileNotFoundError(infile)
    df=pd.read_csv(infile)
    required={"gene","lr_stat","direction"}
    missing=required-set(df.columns)
    if missing:
        raise ValueError(f"{infile.name} missing: {missing}")
    df=df.dropna(subset=["gene","lr_stat","direction"]).copy()
    df=df[df["direction"].isin(["increasing","decreasing"])].copy()
    df["gene"]=df["gene"].astype(str).str.upper()
    df["sign"]=df["direction"].map({"increasing":1.0,"decreasing":-1.0})
    df["rank_score"]=df["sign"]*np.sqrt(np.maximum(df["lr_stat"].astype(float),0))
    ranking=df.groupby("gene",as_index=False)["rank_score"].mean()
    ranking=ranking.sort_values("rank_score",ascending=False)
    ranking.to_csv(OUTDIR/f"{kd}_GSEA_ranked_genes.csv",index=False)
    print(f"\n{kd}: {len(ranking)} ranked genes")
    pre=gp.prerank(
        rnk=ranking[["gene","rank_score"]],
        gene_sets=target_sets,
        min_size=MIN_SIZE,
        max_size=MAX_SIZE,
        permutation_num=PERMUTATIONS,
        seed=SEED,
        threads=4,
        verbose=False
    )
    res=pre.res2d.reset_index().copy()
    rename={
        "Term":"go_term",
        "NES":"NES",
        "NOM p-val":"pval",
        "FDR q-val":"fdr",
        "Lead_genes":"leading_edge"
    }
    res=res.rename(columns={k:v for k,v in rename.items() if k in res.columns})
    if "go_term" not in res.columns:
        if "Name" in res.columns:
            res=res.rename(columns={"Name":"go_term"})
        else:
            res=res.rename(columns={res.columns[0]:"go_term"})
    res["KD"]=kd
    res["Theme"]=res["go_term"].map(theme_map)
    res["GO process"]=res["go_term"].map(clean_term)
    res["NES"]=pd.to_numeric(res["NES"],errors="coerce")
    res["fdr"]=pd.to_numeric(res["fdr"],errors="coerce")
    res["pval"]=pd.to_numeric(res["pval"],errors="coerce")
    res["Spatial enrichment"]=np.where(res["NES"]<0,"Near KD","Far from KD")
    res.to_csv(OUTDIR/f"{kd}_targeted_GSEA_all.csv",index=False)
    all_results.append(res)
combined=pd.concat(all_results,ignore_index=True)
combined.to_csv(OUTDIR/"EFNA3_WNT5A_targeted_GSEA_all.csv",index=False)
sig=combined[combined["fdr"]<FDR_CUTOFF].copy()
sig.to_csv(OUTDIR/"EFNA3_WNT5A_targeted_GSEA_FDR_0.05.csv",index=False)
print("\nSignificant targeted programs:")
if sig.empty:
    print("None at FDR < 0.05")
else:
    print(sig[["KD","Theme","GO process","NES","fdr","Spatial enrichment"]].sort_values(["KD","fdr"]).to_string(index=False))
term_stats=combined.groupby(["go_term","Theme","GO process"],as_index=False).agg(
    min_fdr=("fdr","min"),
    max_abs_NES=("NES",lambda x:np.nanmax(np.abs(x)))
)
term_stats=term_stats[term_stats["min_fdr"]<FDR_CUTOFF].copy()
if term_stats.empty:
    term_stats=combined.groupby(["go_term","Theme","GO process"],as_index=False).agg(
        min_fdr=("fdr","min"),
        max_abs_NES=("NES",lambda x:np.nanmax(np.abs(x)))
    )
selected=[]
for theme in TARGET_THEMES:
    sub=term_stats[term_stats["Theme"]==theme].sort_values(["min_fdr","max_abs_NES"],ascending=[True,False]).head(2)
    selected.extend(sub["go_term"].tolist())
selected=list(dict.fromkeys(selected))[:MAX_TERMS_TO_PLOT]
plot_df=combined[combined["go_term"].isin(selected)].copy()
plot_df["significant"]=plot_df["fdr"]<FDR_CUTOFF
meta=(plot_df.groupby(["go_term","Theme","GO process"],as_index=False)
      .agg(min_fdr=("fdr","min"),max_abs_NES=("NES",lambda x:np.nanmax(np.abs(x)))))
theme_order=list(TARGET_THEMES.keys())
meta["theme_rank"]=meta["Theme"].map({x:i for i,x in enumerate(theme_order)})
meta=meta.sort_values(["theme_rank","min_fdr","max_abs_NES"],ascending=[True,True,False])
row_order=meta["go_term"].tolist()
label_map=dict(zip(meta["go_term"],meta["GO process"]))
nes=plot_df.pivot_table(index="go_term",columns="KD",values="NES",aggfunc="first").reindex(index=row_order,columns=KDS)
fdr=plot_df.pivot_table(index="go_term",columns="KD",values="fdr",aggfunc="first").reindex(index=row_order,columns=KDS)
publication_table=[]
for term in row_order:
    row={"Theme":theme_map.get(term,""),"GO process":label_map[term]}
    for kd in KDS:
        row[f"{kd} NES"]=nes.loc[term,kd] if kd in nes.columns else np.nan
        row[f"{kd} FDR"]=fdr.loc[term,kd] if kd in fdr.columns else np.nan
        if kd in nes.columns and pd.notna(nes.loc[term,kd]):
            row[f"{kd} enrichment"]="Near KD" if nes.loc[term,kd]<0 else "Far from KD"
        else:
            row[f"{kd} enrichment"]=""
    publication_table.append(row)
pd.DataFrame(publication_table).to_csv(OUTDIR/"EFNA3_WNT5A_GSEA_publication_table.csv",index=False)
fig_h=max(5.5,0.42*len(row_order)+1.5)
fig,ax=plt.subplots(figsize=(7.2,fig_h))
values=nes.to_numpy(dtype=float)
vmax=np.nanmax(np.abs(values))
vmax=max(vmax,1)
norm=TwoSlopeNorm(vmin=-vmax,vcenter=0,vmax=vmax)
im=ax.imshow(values,aspect="auto",cmap="RdBu_r",norm=norm)
ax.set_xticks(np.arange(len(KDS)))
ax.set_xticklabels(["EFNA3 KD","WNT5A KD"],fontsize=11,fontweight="bold")
ax.set_yticks(np.arange(len(row_order)))
ax.set_yticklabels([label_map[x] for x in row_order],fontsize=9)
ax.tick_params(length=0)
for i,term in enumerate(row_order):
    for j,kd in enumerate(KDS):
        if kd in fdr.columns and pd.notna(fdr.loc[term,kd]) and fdr.loc[term,kd]<FDR_CUTOFF:
            ax.text(j,i,"●",ha="center",va="center",fontsize=7,color="black")
for i in range(len(row_order)+1):
    ax.axhline(i-0.5,color="white",linewidth=0.8)
for j in range(len(KDS)+1):
    ax.axvline(j-0.5,color="white",linewidth=0.8)
for spine in ax.spines.values():
    spine.set_visible(False)
cbar=fig.colorbar(im,ax=ax,fraction=0.045,pad=0.04)
cbar.set_label("Normalized enrichment score (NES)",fontsize=10)
ax.set_title("Distance-dependent developmental programs",fontsize=14,fontweight="bold",pad=12)
ax.text(0.5,-0.075,"Negative NES = enriched near KD     |     Positive NES = enriched farther from KD",transform=ax.transAxes,ha="center",va="top",fontsize=9)
ax.text(0.5,-0.12,"● FDR < 0.05",transform=ax.transAxes,ha="center",va="top",fontsize=9)
fig.tight_layout()
fig.savefig(OUTDIR/"EFNA3_WNT5A_targeted_GSEA_NES_heatmap.pdf",bbox_inches="tight")
fig.savefig(OUTDIR/"EFNA3_WNT5A_targeted_GSEA_NES_heatmap.svg",bbox_inches="tight")
fig.savefig(OUTDIR/"EFNA3_WNT5A_targeted_GSEA_NES_heatmap.png",dpi=600,bbox_inches="tight")
plt.show()
print("\nSaved to:",OUTDIR.resolve())
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mudata as md
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
H5MU_PATH=Path("all_lanes_merged_CATATAC_QC_RNA_processed_EB_manual_clustering_cell_types.h5mu")
ORIENTATION_FILE=Path("MULTIOME_KD_distance_cell_type_composition_and_GO/common_coordinate_cell_type_distance_maps/EB_common_coordinate_orientation_summary.csv")
OUTDIR=Path("MULTIOME_KD_distance_cell_type_composition_and_GO/common_coordinate_axis_QC")
OUTDIR.mkdir(parents=True,exist_ok=True)
RNA_MODALITY="rna"
EB_COL="EB_manual_clustering"
SPATIAL_KEY="X_spatial"
EXPRESSION_LAYER=None
MAJOR_MARKERS=["GATA4","TNC","CDH2"]
MINOR_MARKERS=["GRID2","SNCA","LHFPL6"]
ALL_MARKERS=MAJOR_MARKERS+MINOR_MARKERS
MIN_CELLS_PER_EB=50
MIN_EXPR_FRAC=0.10
ABS_RHO_THRESHOLD=0.20
plt.rcParams.update({"font.family":"Arial","font.size":9,"pdf.fonttype":42,"ps.fonttype":42})
def build_gene_lookup(adata):
    lookup={}
    for i,g in enumerate(adata.var_names.astype(str)):
        lookup.setdefault(g.upper(),i)
    for col in ["gene_names","gene_name","symbol","gene_symbol"]:
        if col in adata.var.columns:
            for i,g in enumerate(adata.var[col]):
                if pd.notna(g):
                    lookup.setdefault(str(g).strip().upper(),i)
    return lookup
def get_gene_expression(adata,gene_lookup,gene):
    idx=gene_lookup.get(gene.upper())
    if idx is None:
        return None
    matrix=adata.X if EXPRESSION_LAYER is None else adata.layers[EXPRESSION_LAYER]
    values=matrix[:,idx]
    if sparse.issparse(values):
        values=values.toarray().ravel()
    else:
        values=np.asarray(values).ravel()
    return values.astype(float)
if not H5MU_PATH.exists():
    raise FileNotFoundError(H5MU_PATH)
mdata=md.read_h5mu(H5MU_PATH)
adata=mdata[RNA_MODALITY]
eb_labels=adata.obs[EB_COL].astype(str).to_numpy()
coords=np.asarray(adata.obsm[SPATIAL_KEY][:,:2],dtype=float)
gene_lookup=build_gene_lookup(adata)
expr={}
for gene in ALL_MARKERS:
    x=get_gene_expression(adata,gene_lookup,gene)
    if x is None:
        print(f"WARNING: {gene} not found.")
    else:
        expr[gene]=x
print("Orientation genes found:",sorted(expr))
rows=[]
eb_rows=[]
for eb in sorted(np.unique(eb_labels),key=str):
    idx=np.where(eb_labels==eb)[0]
    if len(idx)<MIN_CELLS_PER_EB:
        continue
    xy=coords[idx]
    if not np.isfinite(xy).all():
        continue
    pca=PCA(n_components=2)
    proj=pca.fit_transform(xy)
    evr1=float(pca.explained_variance_ratio_[0])
    evr2=float(pca.explained_variance_ratio_[1])
    eb_rows.append({"EB":eb,"n_cells":len(idx),"PC1_variance_fraction":evr1,"PC2_variance_fraction":evr2,"PC1_PC2_variance_fraction":evr1+evr2})
    for gene in ALL_MARKERS:
        if gene not in expr:
            continue
        gexpr=np.asarray(expr[gene][idx],dtype=float)
        valid=np.isfinite(gexpr)&np.isfinite(proj[:,0])&np.isfinite(proj[:,1])
        if valid.sum()<10:
            continue
        expr_frac=float(np.mean(gexpr[valid]>0))
        rho1,p1=spearmanr(proj[valid,0],gexpr[valid])
        rho2,p2=spearmanr(proj[valid,1],gexpr[valid])
        intended_axis="PC1" if gene in MAJOR_MARKERS else "PC2"
        intended_rho=rho1 if intended_axis=="PC1" else rho2
        other_rho=rho2 if intended_axis=="PC1" else rho1
        rows.append({"EB":eb,"n_cells":len(idx),"gene":gene,"marker_group":"major" if gene in MAJOR_MARKERS else "minor","intended_axis":intended_axis,"expression_fraction":expr_frac,"rho_PC1":rho1,"p_PC1":p1,"rho_PC2":rho2,"p_PC2":p2,"abs_rho_PC1":abs(rho1) if np.isfinite(rho1) else np.nan,"abs_rho_PC2":abs(rho2) if np.isfinite(rho2) else np.nan,"intended_rho":intended_rho,"abs_intended_rho":abs(intended_rho) if np.isfinite(intended_rho) else np.nan,"other_axis_rho":other_rho,"abs_other_axis_rho":abs(other_rho) if np.isfinite(other_rho) else np.nan,"expressed_enough":expr_frac>=MIN_EXPR_FRAC,"passes_rho_threshold":expr_frac>=MIN_EXPR_FRAC and np.isfinite(intended_rho) and abs(intended_rho)>=ABS_RHO_THRESHOLD,"axis_specific":np.isfinite(intended_rho) and np.isfinite(other_rho) and abs(intended_rho)>abs(other_rho)})
corr_df=pd.DataFrame(rows)
eb_pca_df=pd.DataFrame(eb_rows)
corr_df.to_csv(OUTDIR/"all_EB_orientation_marker_spatial_PC_correlations.csv",index=False)
eb_pca_df.to_csv(OUTDIR/"EB_spatial_PCA_variance_QC.csv",index=False)
marker_summary=corr_df.groupby(["marker_group","gene"],as_index=False).agg(n_EBs=("EB","nunique"),median_expression_fraction=("expression_fraction","median"),median_abs_intended_rho=("abs_intended_rho","median"),mean_abs_intended_rho=("abs_intended_rho","mean"),median_abs_other_axis_rho=("abs_other_axis_rho","median"),fraction_EBs_abs_rho_ge_threshold=("passes_rho_threshold","mean"),fraction_EBs_axis_specific=("axis_specific","mean"))
marker_summary.to_csv(OUTDIR/"orientation_marker_QC_summary.csv",index=False)
if ORIENTATION_FILE.exists():
    orientation=pd.read_csv(ORIENTATION_FILE)
    orientation["EB"]=orientation["EB"].astype(str)
    selected_rows=[]
    for row in orientation.itertuples(index=False):
        eb=str(row.EB)
        major_gene=str(row.major_marker)
        minor_gene=str(row.minor_marker)
        major=corr_df[(corr_df["EB"]==eb)&(corr_df["gene"]==major_gene)]
        minor=corr_df[(corr_df["EB"]==eb)&(corr_df["gene"]==minor_gene)]
        out={"EB":eb,"n_cells":getattr(row,"n_cells",np.nan),"major_marker":major_gene,"minor_marker":minor_gene}
        if len(major):
            r=major.iloc[0]
            out.update({"major_rho_PC1":r["rho_PC1"],"major_rho_PC2":r["rho_PC2"],"major_abs_rho_PC1":r["abs_rho_PC1"],"major_expression_fraction":r["expression_fraction"],"major_pass_abs_rho":bool(r["expressed_enough"] and r["abs_rho_PC1"]>=ABS_RHO_THRESHOLD),"major_axis_specific":bool(r["abs_rho_PC1"]>r["abs_rho_PC2"])})
        else:
            out.update({"major_rho_PC1":np.nan,"major_rho_PC2":np.nan,"major_abs_rho_PC1":np.nan,"major_expression_fraction":np.nan,"major_pass_abs_rho":False,"major_axis_specific":False})
        if len(minor):
            r=minor.iloc[0]
            out.update({"minor_rho_PC1":r["rho_PC1"],"minor_rho_PC2":r["rho_PC2"],"minor_abs_rho_PC2":r["abs_rho_PC2"],"minor_expression_fraction":r["expression_fraction"],"minor_pass_abs_rho":bool(r["expressed_enough"] and r["abs_rho_PC2"]>=ABS_RHO_THRESHOLD),"minor_axis_specific":bool(r["abs_rho_PC2"]>r["abs_rho_PC1"])})
        else:
            out.update({"minor_rho_PC1":np.nan,"minor_rho_PC2":np.nan,"minor_abs_rho_PC2":np.nan,"minor_expression_fraction":np.nan,"minor_pass_abs_rho":False,"minor_axis_specific":False})
        out["both_axes_pass_rho"]=out["major_pass_abs_rho"] and out["minor_pass_abs_rho"]
        out["both_axes_specific"]=out["major_axis_specific"] and out["minor_axis_specific"]
        out["overall_axis_QC_pass"]=out["both_axes_pass_rho"] and out["both_axes_specific"]
        selected_rows.append(out)
    selected_qc=pd.DataFrame(selected_rows)
    selected_qc.to_csv(OUTDIR/"selected_orientation_marker_per_EB_QC.csv",index=False)
    overall=pd.DataFrame([{"n_registered_EBs":len(selected_qc),"major_axis_pass_fraction":selected_qc["major_pass_abs_rho"].mean(),"minor_axis_pass_fraction":selected_qc["minor_pass_abs_rho"].mean(),"major_axis_specific_fraction":selected_qc["major_axis_specific"].mean(),"minor_axis_specific_fraction":selected_qc["minor_axis_specific"].mean(),"both_axes_rho_pass_fraction":selected_qc["both_axes_pass_rho"].mean(),"both_axes_specific_fraction":selected_qc["both_axes_specific"].mean(),"overall_axis_QC_pass_fraction":selected_qc["overall_axis_QC_pass"].mean(),"median_major_abs_rho":selected_qc["major_abs_rho_PC1"].median(),"median_minor_abs_rho":selected_qc["minor_abs_rho_PC2"].median()}])
    overall.to_csv(OUTDIR/"overall_axis_QC_summary.csv",index=False)
    print("\nOVERALL QC")
    print(overall.T.to_string(header=False))
else:
    print("WARNING: saved orientation summary not found; marker-level QC was still generated.")
def make_heatmap(markers,axis_col,title,filename):
    sub=corr_df[corr_df["gene"].isin(markers)].pivot(index="EB",columns="gene",values=axis_col)
    sub=sub.reindex(columns=markers)
    if sub.empty:
        return
    fig,ax=plt.subplots(figsize=(max(5,len(markers)*1.4),max(6,len(sub)*0.15)))
    im=ax.imshow(sub.to_numpy(dtype=float),aspect="auto",cmap="coolwarm",vmin=-1,vmax=1)
    ax.set_xticks(np.arange(len(markers)))
    ax.set_xticklabels(markers,rotation=45,ha="right")
    if len(sub)<=50:
        ax.set_yticks(np.arange(len(sub)))
        ax.set_yticklabels(sub.index,fontsize=6)
    else:
        ax.set_yticks([])
    ax.set_title(title,fontweight="bold")
    cbar=fig.colorbar(im,ax=ax)
    cbar.set_label("Spearman rho")
    fig.tight_layout()
    fig.savefig(OUTDIR/filename,dpi=300,bbox_inches="tight")
    plt.close(fig)
make_heatmap(MAJOR_MARKERS,"rho_PC1","Major-axis markers vs spatial PC1","major_markers_vs_PC1_heatmap.png")
make_heatmap(MAJOR_MARKERS,"rho_PC2","Major-axis markers vs spatial PC2","major_markers_vs_PC2_heatmap.png")
make_heatmap(MINOR_MARKERS,"rho_PC2","Minor-axis markers vs spatial PC2","minor_markers_vs_PC2_heatmap.png")
make_heatmap(MINOR_MARKERS,"rho_PC1","Minor-axis markers vs spatial PC1","minor_markers_vs_PC1_heatmap.png")
fig,axes=plt.subplots(1,2,figsize=(9,4))
major_vals=corr_df[(corr_df["marker_group"]=="major")&corr_df["expressed_enough"]]["abs_intended_rho"].dropna()
minor_vals=corr_df[(corr_df["marker_group"]=="minor")&corr_df["expressed_enough"]]["abs_intended_rho"].dropna()
axes[0].hist(major_vals,bins=25,edgecolor="black")
axes[0].axvline(ABS_RHO_THRESHOLD,linestyle="--",linewidth=1.5)
axes[0].set_xlabel("|Spearman rho| with PC1")
axes[0].set_ylabel("EB × marker observations")
axes[0].set_title("Major-axis marker strength")
axes[1].hist(minor_vals,bins=25,edgecolor="black")
axes[1].axvline(ABS_RHO_THRESHOLD,linestyle="--",linewidth=1.5)
axes[1].set_xlabel("|Spearman rho| with PC2")
axes[1].set_ylabel("EB × marker observations")
axes[1].set_title("Minor-axis marker strength")
fig.tight_layout()
fig.savefig(OUTDIR/"orientation_marker_correlation_strength_histograms.png",dpi=300,bbox_inches="tight")
plt.close(fig)
print("\nMARKER-LEVEL QC")
print(marker_summary.to_string(index=False))
print("\nSaved QC results to:",OUTDIR.resolve())
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mudata as md
from scipy import sparse,stats
from sklearn.decomposition import PCA
from sklearn.utils.sparsefuncs import mean_variance_axis
from statsmodels.stats.multitest import multipletests
warnings.filterwarnings("ignore")
H5MU_PATH=Path("all_lanes_merged_CATATAC_QC_RNA_processed_EB_manual_clustering_cell_types.h5mu")
OUTDIR=Path("MULTIOME_KD_distance_cell_type_composition_and_GO/common_coordinate_data_driven_axes")
OUTDIR.mkdir(parents=True,exist_ok=True)
RNA_MODALITY="rna"
EB_COL="EB_manual_clustering"
CELL_TYPE_COL="cell_type"
SPATIAL_KEY="X_spatial"
EXPRESSION_LAYER=None
MIN_CELLS_PER_EB=50
TOP_N_GENES=5000
MIN_GLOBAL_DETECTION=0.02
MIN_EB_DETECTION=0.05
MIN_PC1_VARIANCE_FRACTION=0.55
CONSENSUS_TOP_GENES=500
N_MARKERS_PER_AXIS=20
MIN_MARKER_EB_FRACTION=0.20
MIN_META_ABS_RHO=0.08
MIN_SIGN_CONSISTENCY=0.70
MIN_AXIS_SPECIFICITY=0.03
CONSISTENCY_MIN_ABS_RHO=0.05
MIN_MARKERS_PER_EB=3
MIN_ORIENTATION_AGREEMENT=0.60
MIN_ORIENTATION_STRENGTH=0.05
REQUIRE_DISCOVERY_AGREEMENT=True
EXCLUDE_TECHNICAL_GENES=True
plt.rcParams.update({"font.family":"Arial","font.size":9,"pdf.fonttype":42,"ps.fonttype":42})
def get_gene_labels(adata):
    labels=adata.var_names.astype(str).to_numpy()
    for col in ["gene_names","gene_name","gene_symbol","symbol"]:
        if col in adata.var.columns:
            vals=adata.var[col].astype(str).to_numpy()
            good=np.array([x not in {"nan","None",""} for x in vals])
            labels=np.where(good,vals,labels)
            break
    return labels
def fast_spearman_matrix(X,y):
    y_rank=stats.rankdata(y).astype(float)
    y_rank-=y_rank.mean()
    y_den=np.sqrt(np.sum(y_rank**2))
    if y_den==0:
        return np.full(X.shape[1],np.nan)
    y_rank/=y_den
    X_rank=stats.rankdata(X,axis=0).astype(float)
    X_rank-=X_rank.mean(axis=0,keepdims=True)
    den=np.sqrt(np.sum(X_rank**2,axis=0))
    out=np.full(X.shape[1],np.nan,dtype=float)
    good=den>0
    out[good]=(X_rank[:,good].T@y_rank)/den[good]
    return np.clip(out,-1,1)
def infer_consensus_flips(rho_target,rho_other,n_cells,geometry_pass):
    use=np.where(geometry_pass)[0]
    med_target=np.nanmedian(np.abs(rho_target[use]),axis=0)
    med_other=np.nanmedian(np.abs(rho_other[use]),axis=0)
    strength=med_target-med_other
    good=np.isfinite(strength)&np.isfinite(med_target)&(med_target>0)
    candidate=np.where(good)[0]
    candidate=candidate[np.argsort(strength[candidate])[::-1]]
    candidate=candidate[:min(CONSENSUS_TOP_GENES,len(candidate))]
    if len(candidate)<10:
        raise ValueError("Too few genes available for transcriptome-wide axis synchronization.")
    P=np.nan_to_num(rho_target[np.ix_(use,candidate)],nan=0.0)
    norms=np.sqrt(np.sum(P**2,axis=1))
    P=P/(norms[:,None]+1e-12)
    similarity=P@P.T
    _,vecs=np.linalg.eigh(similarity)
    flips=np.where(vecs[:,-1]>=0,1.0,-1.0)
    anchor=np.argmax(n_cells[use])
    if flips[anchor]<0:
        flips*=-1
    weights=np.sqrt(np.maximum(n_cells[use]-3,1))
    for _ in range(50):
        consensus=np.average(P*flips[:,None],axis=0,weights=weights)
        new=np.where(P@consensus>=0,1.0,-1.0)
        if new[anchor]<0:
            new*=-1
        if np.array_equal(new,flips):
            break
        flips=new
    full=np.full(len(n_cells),np.nan,dtype=float)
    full[use]=flips
    return full,candidate
def meta_axis_scores(rho_target,rho_other,det_frac,target_flips,other_flips,n_cells,geometry_pass,gene_labels,var_names):
    use=np.where(geometry_pass)[0]
    rows=[]
    for j in range(rho_target.shape[1]):
        rt=rho_target[use,j]
        ro=rho_other[use,j]
        det=det_frac[use,j]
        ft=target_flips[use]
        fo=other_flips[use]
        valid=np.isfinite(rt)&np.isfinite(ft)&(det>=MIN_EB_DETECTION)
        valid_other=np.isfinite(ro)&np.isfinite(fo)&(det>=MIN_EB_DETECTION)
        n_valid=int(valid.sum())
        if n_valid<2:
            continue
        aligned=rt[valid]*ft[valid]
        weights=np.maximum(n_cells[use][valid]-3,1).astype(float)
        fisher=np.arctanh(np.clip(aligned,-0.999999,0.999999))
        meta_fisher=float(np.sum(weights*fisher)/np.sum(weights))
        meta_r=float(np.tanh(meta_fisher))
        meta_z=float(meta_fisher*np.sqrt(np.sum(weights)))
        pval=float(2*stats.norm.sf(abs(meta_z)))
        if valid_other.sum()>=2:
            aligned_other=ro[valid_other]*fo[valid_other]
            weights_other=np.maximum(n_cells[use][valid_other]-3,1).astype(float)
            fisher_other=np.arctanh(np.clip(aligned_other,-0.999999,0.999999))
            other_meta_r=float(np.tanh(np.sum(weights_other*fisher_other)/np.sum(weights_other)))
            median_abs_other=float(np.nanmedian(np.abs(aligned_other)))
        else:
            other_meta_r=np.nan
            median_abs_other=np.nan
        consistency_mask=np.abs(aligned)>=CONSISTENCY_MIN_ABS_RHO
        if consistency_mask.sum()>0:
            sign_consistency=float(np.mean(np.sign(aligned[consistency_mask])==np.sign(meta_r)))
        else:
            sign_consistency=np.nan
        fraction_informative=n_valid/len(use)
        axis_specificity=abs(meta_r)-abs(other_meta_r) if np.isfinite(other_meta_r) else np.nan
        marker_score=abs(meta_r)*max(sign_consistency if np.isfinite(sign_consistency) else 0,0)*max(axis_specificity if np.isfinite(axis_specificity) else 0,0)*np.sqrt(fraction_informative)
        rows.append({"candidate_col":j,"var_name":var_names[j],"gene":gene_labels[j],"n_informative_EBs":n_valid,"fraction_informative_EBs":fraction_informative,"meta_r":meta_r,"meta_z":meta_z,"pval":pval,"median_abs_target_rho":float(np.nanmedian(np.abs(aligned))),"other_axis_meta_r":other_meta_r,"median_abs_other_rho":median_abs_other,"axis_specificity":axis_specificity,"sign_consistency":sign_consistency,"marker_score":marker_score})
    out=pd.DataFrame(rows)
    out["fdr"]=multipletests(out["pval"].to_numpy(),method="fdr_bh")[1] if len(out)>0 else np.nan
    out=out.sort_values(["marker_score","fdr","meta_r"],ascending=[False,True,False]).reset_index(drop=True)
    return out
def select_markers(score_df,n_markers=N_MARKERS_PER_AXIS):
    strict=score_df[(score_df["fraction_informative_EBs"]>=MIN_MARKER_EB_FRACTION)&(score_df["meta_r"].abs()>=MIN_META_ABS_RHO)&(score_df["sign_consistency"]>=MIN_SIGN_CONSISTENCY)&(score_df["axis_specificity"]>=MIN_AXIS_SPECIFICITY)&(score_df["fdr"]<0.05)].copy()
    strict["selection_tier"]="strict"
    selected=strict.drop_duplicates("gene").head(n_markers)
    if len(selected)<n_markers:
        relaxed=score_df[(score_df["fraction_informative_EBs"]>=0.40)&(score_df["meta_r"].abs()>=0.08)&(score_df["sign_consistency"]>=0.60)&(score_df["axis_specificity"]>=0)&(score_df["fdr"]<0.05)].copy()
        relaxed["selection_tier"]="relaxed"
        relaxed=relaxed[~relaxed["gene"].isin(selected["gene"])]
        selected=pd.concat([selected,relaxed.drop_duplicates("gene").head(n_markers-len(selected))],ignore_index=True)
    return selected.head(n_markers).copy()
def orient_ebs_from_markers(rho,det_frac,selected,broad_flips):
    cols=selected["candidate_col"].astype(int).to_numpy()
    expected=np.sign(selected["meta_r"].to_numpy(dtype=float))
    marker_weights=np.abs(selected["meta_r"].to_numpy(dtype=float))*selected["sign_consistency"].fillna(0).to_numpy(dtype=float)
    marker_weights=np.where(marker_weights>0,marker_weights,1e-6)
    rows=[]
    flips=np.full(rho.shape[0],np.nan,dtype=float)
    for i in range(rho.shape[0]):
        r=rho[i,cols]
        d=det_frac[i,cols]
        valid=np.isfinite(r)&(d>=MIN_EB_DETECTION)
        n_valid=int(valid.sum())
        if n_valid==0:
            rows.append({"n_markers":0,"orientation_score":np.nan,"marker_agreement":np.nan,"orientation_strength":np.nan,"marker_flip":np.nan,"discovery_flip":broad_flips[i],"agrees_with_discovery":False,"marker_orientation_pass":False})
            continue
        evidence=r[valid]*expected[valid]
        w=marker_weights[valid]
        score=float(np.sum(w*evidence)/np.sum(w))
        flip=1.0 if score>=0 else -1.0
        oriented=evidence*flip
        agreement=float(np.sum(w*(oriented>0))/np.sum(w))
        strength=float(np.sum(w*np.abs(r[valid]))/np.sum(w))
        agrees=bool(np.isfinite(broad_flips[i]) and flip==broad_flips[i])
        passed=bool(n_valid>=MIN_MARKERS_PER_EB and agreement>=MIN_ORIENTATION_AGREEMENT and strength>=MIN_ORIENTATION_STRENGTH)
        flips[i]=flip
        rows.append({"n_markers":n_valid,"orientation_score":score,"marker_agreement":agreement,"orientation_strength":strength,"marker_flip":flip,"discovery_flip":broad_flips[i],"agrees_with_discovery":agrees,"marker_orientation_pass":passed})
    return flips,pd.DataFrame(rows)
def plot_selected_heatmap(rho,flips,geometry_pass,selected,eb_order,title,path):
    use=np.where(geometry_pass)[0]
    cols=selected["candidate_col"].astype(int).to_numpy()
    arr=rho[np.ix_(use,cols)]*flips[use,None]
    fig,ax=plt.subplots(figsize=(max(8,len(cols)*0.5),max(7,len(use)*0.18)))
    im=ax.imshow(arr,aspect="auto",cmap="coolwarm",vmin=-0.6,vmax=0.6)
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(selected["gene"],rotation=60,ha="right",fontsize=7)
    ax.set_yticks(np.arange(len(use)))
    ax.set_yticklabels(np.asarray(eb_order)[use],fontsize=6)
    ax.set_title(title,fontweight="bold")
    cbar=fig.colorbar(im,ax=ax)
    cbar.set_label("Aligned Spearman rho")
    fig.tight_layout()
    fig.savefig(path,dpi=300,bbox_inches="tight")
    plt.close(fig)
if not H5MU_PATH.exists():
    raise FileNotFoundError(H5MU_PATH)
print("Loading MuData:",H5MU_PATH)
mdata=md.read_h5mu(H5MU_PATH)
if RNA_MODALITY not in mdata.mod:
    raise ValueError(f"Missing modality: {RNA_MODALITY}")
adata=mdata[RNA_MODALITY]
for col in [EB_COL,CELL_TYPE_COL]:
    if col not in adata.obs.columns:
        raise ValueError(f"Missing obs column: {col}")
if SPATIAL_KEY not in adata.obsm:
    raise ValueError(f"Missing obsm key: {SPATIAL_KEY}")
X=adata.X if EXPRESSION_LAYER is None else adata.layers[EXPRESSION_LAYER]
if sparse.issparse(X):
    X=X.tocsr()
coords=np.asarray(adata.obsm[SPATIAL_KEY][:,:2],dtype=float)
eb_labels=adata.obs[EB_COL].astype(str).to_numpy()
cell_types=adata.obs[CELL_TYPE_COL].astype(str).str.strip().to_numpy()
gene_labels_all=get_gene_labels(adata)
var_names_all=adata.var_names.astype(str).to_numpy()
if sparse.issparse(X):
    means,variances=mean_variance_axis(X,axis=0)
    global_detection=np.asarray(X.getnnz(axis=0)).ravel()/X.shape[0]
else:
    means=np.nanmean(np.asarray(X),axis=0)
    variances=np.nanvar(np.asarray(X),axis=0)
    global_detection=np.mean(np.asarray(X)!=0,axis=0)
cv=np.sqrt(np.maximum(variances,0))/(np.abs(means)+1e-8)
valid_gene=np.isfinite(cv)&(global_detection>=MIN_GLOBAL_DETECTION)
if EXCLUDE_TECHNICAL_GENES:
    upper=np.char.upper(gene_labels_all.astype(str))
    technical=np.array([g.startswith("MT-") or g.startswith("RPL") or g.startswith("RPS") or g=="MALAT1" for g in upper])
    valid_gene&=~technical
valid_idx=np.where(valid_gene)[0]
top_idx=valid_idx[np.argsort(cv[valid_idx])[::-1][:min(TOP_N_GENES,len(valid_idx))]]
candidate_gene_labels=gene_labels_all[top_idx]
candidate_var_names=var_names_all[top_idx]
print(f"Candidate genes: {len(top_idx):,}")
eb_order=[]
eb_indices=[]
proj_list=[]
pca_rows=[]
rho1_list=[]
rho2_list=[]
det_list=[]
for eb in sorted(np.unique(eb_labels),key=str):
    idx=np.where(eb_labels==eb)[0]
    if len(idx)<MIN_CELLS_PER_EB:
        continue
    xy=coords[idx]
    if not np.isfinite(xy).all():
        continue
    pca=PCA(n_components=2)
    proj=pca.fit_transform(xy)
    pc1_fraction=float(pca.explained_variance_ratio_[0])
    pc2_fraction=float(pca.explained_variance_ratio_[1])
    aspect=float(np.sqrt(pca.explained_variance_[1]/pca.explained_variance_[0]))
    Xeb=X[idx][:,top_idx]
    if sparse.issparse(Xeb):
        Xeb=Xeb.toarray()
    else:
        Xeb=np.asarray(Xeb)
    detection=np.mean(Xeb!=0,axis=0)
    rho1=fast_spearman_matrix(Xeb,proj[:,0])
    rho2=fast_spearman_matrix(Xeb,proj[:,1])
    rho1[detection<MIN_EB_DETECTION]=np.nan
    rho2[detection<MIN_EB_DETECTION]=np.nan
    eb_order.append(eb)
    eb_indices.append(idx)
    proj_list.append(proj)
    rho1_list.append(rho1)
    rho2_list.append(rho2)
    det_list.append(detection)
    pca_rows.append({"EB":eb,"n_cells":len(idx),"PC1_variance_fraction":pc1_fraction,"PC2_variance_fraction":pc2_fraction,"aspect_ratio":aspect,"geometry_pass":pc1_fraction>=MIN_PC1_VARIANCE_FRACTION})
rho1=np.vstack(rho1_list)
rho2=np.vstack(rho2_list)
det_frac=np.vstack(det_list)
pca_df=pd.DataFrame(pca_rows)
n_cells=pca_df["n_cells"].to_numpy(dtype=int)
geometry_pass=pca_df["geometry_pass"].to_numpy(dtype=bool)
print(f"EBs analyzed: {len(eb_order)}")
print(f"EBs passing geometry QC: {geometry_pass.sum()}/{len(geometry_pass)}")
pca_df.to_csv(OUTDIR/"data_driven_EB_spatial_PCA_QC.csv",index=False)
flip1_broad,consensus_genes1=infer_consensus_flips(rho1,rho2,n_cells,geometry_pass)
flip2_broad,consensus_genes2=infer_consensus_flips(rho2,rho1,n_cells,geometry_pass)
pc1_scores=meta_axis_scores(rho1,rho2,det_frac,flip1_broad,flip2_broad,n_cells,geometry_pass,candidate_gene_labels,candidate_var_names)
pc2_scores=meta_axis_scores(rho2,rho1,det_frac,flip2_broad,flip1_broad,n_cells,geometry_pass,candidate_gene_labels,candidate_var_names)
pc1_scores.to_csv(OUTDIR/"PC1_data_driven_gene_scores.csv",index=False)
pc2_scores.to_csv(OUTDIR/"PC2_data_driven_gene_scores.csv",index=False)
pc1_markers=select_markers(pc1_scores)
pc2_markers=select_markers(pc2_scores)
pc1_markers.to_csv(OUTDIR/"PC1_selected_orientation_markers.csv",index=False)
pc2_markers.to_csv(OUTDIR/"PC2_selected_orientation_markers.csv",index=False)
print("\nSelected PC1 markers:")
print(pc1_markers[["gene","meta_r","sign_consistency","axis_specificity","fraction_informative_EBs","selection_tier"]].to_string(index=False))
print("\nSelected PC2 markers:")
print(pc2_markers[["gene","meta_r","sign_consistency","axis_specificity","fraction_informative_EBs","selection_tier"]].to_string(index=False))
flip1_marker,qc1=orient_ebs_from_markers(rho1,det_frac,pc1_markers,flip1_broad)
flip2_marker,qc2=orient_ebs_from_markers(rho2,det_frac,pc2_markers,flip2_broad)
qc=pca_df.copy()
qc["PC1_marker_flip"]=qc1["marker_flip"]
qc["PC1_discovery_flip"]=qc1["discovery_flip"]
qc["PC1_n_markers"]=qc1["n_markers"]
qc["PC1_marker_agreement"]=qc1["marker_agreement"]
qc["PC1_orientation_strength"]=qc1["orientation_strength"]
qc["PC1_marker_pass"]=qc1["marker_orientation_pass"]
qc["PC1_agrees_with_discovery"]=qc1["agrees_with_discovery"]
qc["PC2_marker_flip"]=qc2["marker_flip"]
qc["PC2_discovery_flip"]=qc2["discovery_flip"]
qc["PC2_n_markers"]=qc2["n_markers"]
qc["PC2_marker_agreement"]=qc2["marker_agreement"]
qc["PC2_orientation_strength"]=qc2["orientation_strength"]
qc["PC2_marker_pass"]=qc2["marker_orientation_pass"]
qc["PC2_agrees_with_discovery"]=qc2["agrees_with_discovery"]
qc["axis_marker_pass"]=qc["PC1_marker_pass"]&qc["PC2_marker_pass"]
qc["discovery_agreement_pass"]=qc["PC1_agrees_with_discovery"]&qc["PC2_agrees_with_discovery"]
qc["final_registration_pass"]=qc["geometry_pass"]&qc["axis_marker_pass"]
if REQUIRE_DISCOVERY_AGREEMENT:
    qc["final_registration_pass"]&=qc["discovery_agreement_pass"]
qc.to_csv(OUTDIR/"data_driven_EB_axis_orientation_QC.csv",index=False)
print("\nFINAL ORIENTATION QC")
print("Geometry pass:",int(qc["geometry_pass"].sum()),"/",len(qc))
print("PC1 marker pass:",int(qc["PC1_marker_pass"].sum()),"/",len(qc))
print("PC2 marker pass:",int(qc["PC2_marker_pass"].sum()),"/",len(qc))
print("Both marker axes pass:",int(qc["axis_marker_pass"].sum()),"/",len(qc))
print("Final registration pass:",int(qc["final_registration_pass"].sum()),"/",len(qc))
if qc["final_registration_pass"].sum()<3:
    raise ValueError("Fewer than 3 EBs passed final axis QC. Inspect the marker tables and relax thresholds only if biologically justified.")
plot_selected_heatmap(rho1,flip1_marker,geometry_pass,pc1_markers,eb_order,"Data-driven PC1 markers after EB orientation",OUTDIR/"PC1_selected_marker_aligned_heatmap.png")
plot_selected_heatmap(rho2,flip2_marker,geometry_pass,pc2_markers,eb_order,"Data-driven PC2 markers after EB orientation",OUTDIR/"PC2_selected_marker_aligned_heatmap.png")
included=np.where(qc["final_registration_pass"].to_numpy(dtype=bool))[0]
canonical_aspect=float(np.mean(qc.loc[qc["final_registration_pass"],"aspect_ratio"]))
common_x=np.full(adata.n_obs,np.nan,dtype=float)
common_y=np.full(adata.n_obs,np.nan,dtype=float)
for i in included:
    idx=eb_indices[i]
    proj=proj_list[i].copy()
    for d in range(2):
        mx=np.max(np.abs(proj[:,d]))
        if mx>0:
            proj[:,d]/=mx
    common_x[idx]=proj[:,0]*flip1_marker[i]
    common_y[idx]=proj[:,1]*flip2_marker[i]*canonical_aspect
registered=pd.DataFrame({"cell_index":np.arange(adata.n_obs,dtype=int),"EB_registered":eb_labels,"cell_type_registered":cell_types,"common_x":common_x,"common_y":common_y})
registered=registered[np.isfinite(registered["common_x"])&np.isfinite(registered["common_y"])].copy()
registered.to_csv(OUTDIR/"registered_all_cells_common_coordinates_DATA_DRIVEN.csv.gz",index=False,compression="gzip")
summary=pd.DataFrame([{"n_EBs_analyzed":len(qc),"n_geometry_pass":int(qc["geometry_pass"].sum()),"n_PC1_marker_pass":int(qc["PC1_marker_pass"].sum()),"n_PC2_marker_pass":int(qc["PC2_marker_pass"].sum()),"n_both_marker_pass":int(qc["axis_marker_pass"].sum()),"n_final_registered_EBs":int(qc["final_registration_pass"].sum()),"fraction_final_registered_EBs":float(qc["final_registration_pass"].mean()),"canonical_aspect_ratio":canonical_aspect,"n_PC1_markers":len(pc1_markers),"n_PC2_markers":len(pc2_markers)}])
summary.to_csv(OUTDIR/"data_driven_axis_summary.csv",index=False)
print("\nTop PC1-positive genes:")
print(pc1_scores.sort_values("meta_r",ascending=False)[["gene","meta_r","sign_consistency","axis_specificity"]].head(15).to_string(index=False))
print("\nTop PC1-negative genes:")
print(pc1_scores.sort_values("meta_r")[["gene","meta_r","sign_consistency","axis_specificity"]].head(15).to_string(index=False))
print("\nTop PC2-positive genes:")
print(pc2_scores.sort_values("meta_r",ascending=False)[["gene","meta_r","sign_consistency","axis_specificity"]].head(15).to_string(index=False))
print("\nTop PC2-negative genes:")
print(pc2_scores.sort_values("meta_r")[["gene","meta_r","sign_consistency","axis_specificity"]].head(15).to_string(index=False))
print("\nFinished.")
print("Results:",OUTDIR.resolve())
print("New registered coordinate file:",(OUTDIR/"registered_all_cells_common_coordinates_DATA_DRIVEN.csv.gz").resolve())
from pathlib import Path
import pandas as pd
QC_FILE=Path("MULTIOME_KD_distance_cell_type_composition_and_GO/common_coordinate_data_driven_axes/data_driven_EB_axis_orientation_QC.csv")
DISTANCE_FILE=Path("MULTIOME_KD_distance_cell_type_composition_and_GO/all_KD_receiver_cells_with_distance_and_cell_type.csv")
OUTDIR=Path("MULTIOME_KD_distance_cell_type_composition_and_GO/common_coordinate_data_driven_axes/KD_EB_coverage_QC")
OUTDIR.mkdir(parents=True,exist_ok=True)
MIN_CELLS_PER_EB_GROUP=5
DISTANCE_BINS=["0-50","50-100","100-150","150-200","200-250","250-300"]
NEAR_BINS=["0-50","50-100"]
FAR_BINS=["200-250","250-300"]
TREND_GROUPS=[
    {"name":"Early_Ectoderm_NECTIN2_SEMA6D","cell_type":"Early Ectoderm","kds":["NECTIN2","SEMA6D"]},
    {"name":"Early_Ectoderm_WNT5A_EFNA3","cell_type":"Early Ectoderm","kds":["WNT5A","EFNA3"]},
    {"name":"Neural_Crest_multiKD","cell_type":"Neural Crest","kds":["ALB","BMP5","DLL3","EFNA3","NECTIN2","NECTIN3","SEMA6D","VEGFB"]},
]
qc=pd.read_csv(QC_FILE)
dist=pd.read_csv(DISTANCE_FILE)
qc["EB"]=qc["EB"].astype(str)
dist["EB"]=dist["EB"].astype(str)
dist["kd_gene"]=dist["kd_gene"].astype(str).str.strip()
dist["cell_type"]=dist["cell_type"].astype(str).str.strip()
dist["distance_bin"]=dist["distance_bin"].astype(str)
accepted=set(qc.loc[qc["final_registration_pass"].astype(bool),"EB"])
print(f"Final registered EBs: {len(accepted)}")
print(", ".join(sorted(accepted)))
dist=dist[dist["EB"].isin(accepted)].copy()
kd_rows=[]
bin_rows=[]
near_far_rows=[]
for group in TREND_GROUPS:
    target=group["cell_type"]
    for kd in group["kds"]:
        kd_df=dist[dist["kd_gene"]==kd].copy()
        ebs=sorted(kd_df["EB"].unique())
        target_df=kd_df[kd_df["cell_type"]==target]
        kd_rows.append({"group":group["name"],"cell_type":target,"kd_gene":kd,"n_final_registered_EBs":len(ebs),"EBs":";".join(ebs),"n_receiver_cells":len(kd_df),"n_target_cells":len(target_df)})
        for dbin in DISTANCE_BINS:
            b=kd_df[kd_df["distance_bin"]==dbin]
            bt=b[b["cell_type"]==target]
            bin_rows.append({"group":group["name"],"cell_type":target,"kd_gene":kd,"distance_bin":dbin,"n_EBs":b["EB"].nunique(),"EBs":";".join(sorted(b["EB"].unique())),"n_receiver_cells":len(b),"n_target_cells":len(bt)})
        eligible=[]
        for eb,eb_df in kd_df.groupby("EB"):
            n_near=int(eb_df["distance_bin"].isin(NEAR_BINS).sum())
            n_far=int(eb_df["distance_bin"].isin(FAR_BINS).sum())
            if n_near>=MIN_CELLS_PER_EB_GROUP and n_far>=MIN_CELLS_PER_EB_GROUP:
                eligible.append(eb)
        near_far_rows.append({"group":group["name"],"cell_type":target,"kd_gene":kd,"n_near_far_eligible_EBs":len(eligible),"eligible_EBs":";".join(sorted(eligible))})
kd_summary=pd.DataFrame(kd_rows)
bin_summary=pd.DataFrame(bin_rows)
near_far_summary=pd.DataFrame(near_far_rows)
kd_summary.to_csv(OUTDIR/"KD_final_registered_EB_coverage.csv",index=False)
bin_summary.to_csv(OUTDIR/"KD_distance_bin_final_registered_EB_coverage.csv",index=False)
near_far_summary.to_csv(OUTDIR/"KD_near_far_final_registered_EB_coverage.csv",index=False)
print("\nOVERALL KD COVERAGE")
print(kd_summary[["group","kd_gene","n_final_registered_EBs","n_receiver_cells","n_target_cells"]].to_string(index=False))
print("\nNEAR/FAR ELIGIBLE EBs")
print(near_far_summary[["group","kd_gene","n_near_far_eligible_EBs","eligible_EBs"]].to_string(index=False))
print("\nDISTANCE-BIN COVERAGE")
print(bin_summary[["group","kd_gene","distance_bin","n_EBs","n_receiver_cells","n_target_cells"]].to_string(index=False))
print("\nSaved to:",OUTDIR.resolve())
from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
DISTANCE_FILE=Path("MULTIOME_KD_distance_cell_type_composition_and_GO/all_KD_receiver_cells_with_distance_and_cell_type.csv")
REG_FILE=Path("MULTIOME_KD_distance_cell_type_composition_and_GO/common_coordinate_data_driven_axes/registered_all_cells_common_coordinates_DATA_DRIVEN.csv.gz")
OUTDIR=Path("MULTIOME_KD_distance_cell_type_composition_and_GO/common_coordinate_data_driven_axes/final_cell_type_KD_maps")
OUTDIR.mkdir(parents=True,exist_ok=True)
GRID_NX=300
GRID_NY=225
KERNEL_BW=0.15
KERNEL_RADIUS_MULTIPLIER=3.0
MIN_CELLS_PER_EB_GROUP=5
MIN_EFFECTIVE_CELLS=3.0
MIN_EBS_PER_LOCATION=2
GRID_CHUNK=1000
PROPORTION_CMAP="inferno"
DELTA_CMAP="coolwarm"
DISTANCE_BINS=["0-50","50-100","100-150","150-200","200-250","250-300"]
NEAR_BINS=["0-50","50-100"]
FAR_BINS=["200-250","250-300"]
TREND_GROUPS=[
    {"name":"Neural_Crest_NECTIN2_SEMA6D","cell_type":"Neural Crest","kds":["NECTIN2","SEMA6D"]},
    {"name":"Pluripotent_WNT5A_EFNA3","cell_type":"Pluripotent","kds":["WNT5A","EFNA3"]},
    {"name":"Early_Ectoderm_NECTIN2_SEMA6D","cell_type":"Early Ectoderm","kds":["NECTIN2","SEMA6D"]},
    {"name":"Early_Ectoderm_WNT5A_EFNA3","cell_type":"Early Ectoderm","kds":["WNT5A","EFNA3"]},
]
plt.rcParams.update({"font.family":"Arial","font.size":9,"axes.titlesize":10,"pdf.fonttype":42,"ps.fonttype":42})
def make_grid(aspect):
    x=np.linspace(-1,1,GRID_NX)
    y=np.linspace(-aspect,aspect,GRID_NY)
    xx,yy=np.meshgrid(x,y)
    inside=(xx**2+(yy/aspect)**2)<=1.0
    return x,y,xx,yy,inside
def kernel_surface(x,y,target,xx,yy,inside):
    grid=np.column_stack([xx.ravel(),yy.ravel()])
    cells=np.column_stack([np.asarray(x,dtype=float),np.asarray(y,dtype=float)])
    target=np.asarray(target,dtype=float)
    surface=np.full(len(grid),np.nan,dtype=float)
    radius2=(KERNEL_RADIUS_MULTIPLIER*KERNEL_BW)**2
    for start in range(0,len(grid),GRID_CHUNK):
        end=min(start+GRID_CHUNK,len(grid))
        g=grid[start:end]
        d2=(g[:,None,0]-cells[None,:,0])**2+(g[:,None,1]-cells[None,:,1])**2
        w=np.exp(-d2/(2*KERNEL_BW**2))
        w[d2>radius2]=0.0
        sw=w.sum(axis=1)
        sw2=(w*w).sum(axis=1)
        neff=np.zeros_like(sw)
        good=sw2>0
        neff[good]=(sw[good]**2)/sw2[good]
        valid=(sw>0)&(neff>=MIN_EFFECTIVE_CELLS)
        if valid.any():
            out_idx=np.where(valid)[0]+start
            surface[out_idx]=(w[valid]@target)/sw[valid]
    surface=surface.reshape(xx.shape)
    surface[~inside]=np.nan
    return surface
def mean_surfaces(arrays,min_count=1):
    if not arrays:
        return None,None
    stack=np.stack(arrays,axis=0)
    count=np.isfinite(stack).sum(axis=0)
    total=np.nansum(stack,axis=0)
    mean=np.full(stack.shape[1:],np.nan,dtype=float)
    np.divide(total,count,out=mean,where=count>0)
    mean[count<min_count]=np.nan
    return mean,count
def kd_distance_surface(df,kd,target_cell_type,distance_bins,xx,yy,inside):
    sub=df[(df["kd_gene"]==kd)&(df["distance_bin"].isin(distance_bins))]
    surfaces=[]
    for eb,eb_df in sub.groupby("EB",observed=True):
        if len(eb_df)<MIN_CELLS_PER_EB_GROUP:
            continue
        target=(eb_df["cell_type"].to_numpy()==target_cell_type).astype(float)
        surf=kernel_surface(eb_df["common_x"],eb_df["common_y"],target,xx,yy,inside)
        if np.isfinite(surf).any():
            surfaces.append(surf)
    if not surfaces:
        return None,None
    return mean_surfaces(surfaces,MIN_EBS_PER_LOCATION)
def group_distance_surface(df,kds,target_cell_type,distance_bins,xx,yy,inside):
    kd_maps=[]
    for kd in kds:
        surf,_=kd_distance_surface(df,kd,target_cell_type,distance_bins,xx,yy,inside)
        if surf is not None and np.isfinite(surf).any():
            kd_maps.append(surf)
    if not kd_maps:
        return None,None
    return mean_surfaces(kd_maps,1)
def kd_near_far_surfaces(df,kd,target_cell_type,xx,yy,inside):
    kd_df=df[df["kd_gene"]==kd]
    eligible=[]
    for eb,eb_df in kd_df.groupby("EB",observed=True):
        n_near=int(eb_df["distance_bin"].isin(NEAR_BINS).sum())
        n_far=int(eb_df["distance_bin"].isin(FAR_BINS).sum())
        if n_near>=MIN_CELLS_PER_EB_GROUP and n_far>=MIN_CELLS_PER_EB_GROUP:
            eligible.append(eb)
    near_surfaces=[]
    far_surfaces=[]
    for eb in eligible:
        eb_df=kd_df[kd_df["EB"]==eb]
        near_df=eb_df[eb_df["distance_bin"].isin(NEAR_BINS)]
        far_df=eb_df[eb_df["distance_bin"].isin(FAR_BINS)]
        near_target=(near_df["cell_type"].to_numpy()==target_cell_type).astype(float)
        far_target=(far_df["cell_type"].to_numpy()==target_cell_type).astype(float)
        near_surf=kernel_surface(near_df["common_x"],near_df["common_y"],near_target,xx,yy,inside)
        far_surf=kernel_surface(far_df["common_x"],far_df["common_y"],far_target,xx,yy,inside)
        if np.isfinite(near_surf).any():
            near_surfaces.append(near_surf)
        if np.isfinite(far_surf).any():
            far_surfaces.append(far_surf)
    if not near_surfaces or not far_surfaces:
        return None,None,None,None
    near,near_support=mean_surfaces(near_surfaces,MIN_EBS_PER_LOCATION)
    far,far_support=mean_surfaces(far_surfaces,MIN_EBS_PER_LOCATION)
    delta=far-near
    delta[~(np.isfinite(near)&np.isfinite(far))]=np.nan
    return near,far,delta,np.minimum(near_support,far_support)
def group_near_far_surfaces(df,kds,target_cell_type,xx,yy,inside):
    near_maps=[]
    far_maps=[]
    delta_maps=[]
    per_kd={}
    for kd in kds:
        near,far,delta,support=kd_near_far_surfaces(df,kd,target_cell_type,xx,yy,inside)
        per_kd[kd]=(near,far,delta,support)
        if near is not None and np.isfinite(near).any():
            near_maps.append(near)
        if far is not None and np.isfinite(far).any():
            far_maps.append(far)
        if delta is not None and np.isfinite(delta).any():
            delta_maps.append(delta)
    near,near_support=mean_surfaces(near_maps,1) if near_maps else (None,None)
    far,far_support=mean_surfaces(far_maps,1) if far_maps else (None,None)
    delta,delta_support=mean_surfaces(delta_maps,1) if delta_maps else (None,None)
    return near,far,delta,near_support,far_support,delta_support,per_kd
def get_surface_limits(surfaces,lower=1,upper=99,min_span=0.08,clip0=True,clip1=True):
    vals=[s[np.isfinite(s)] for s in surfaces if s is not None and np.isfinite(s).any()]
    if len(vals)==0:
        return 0.0,1.0
    vals=np.concatenate(vals)
    vmin=float(np.nanpercentile(vals,lower))
    vmax=float(np.nanpercentile(vals,upper))
    if vmax-vmin<min_span:
        center=float(np.nanmean(vals))
        half=min_span/2
        vmin=center-half
        vmax=center+half
    if clip0:
        vmin=max(0.0,vmin)
    if clip1:
        vmax=min(1.0,vmax)
    if vmax<=vmin:
        vmax=min(1.0,vmin+min_span)
    return vmin,vmax
def style_axis(ax,title,aspect,show_y_ticks=False):
    ax.add_patch(Ellipse((0,0),width=2,height=2*aspect,fill=False,edgecolor="black",linewidth=1.2,zorder=5))
    ax.axhline(0,color="black",linewidth=0.35,alpha=0.25)
    ax.axvline(0,color="black",linewidth=0.35,alpha=0.25)
    ax.set_xlim(-1.03,1.03)
    ax.set_ylim(-aspect*1.03,aspect*1.03)
    ax.set_aspect("equal")
    ax.set_xticks([-1,-0.5,0,0.5,1])
    if show_y_ticks:
        yticks=[-aspect,-aspect/2,0,aspect/2,aspect]
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"{v:.2f}" for v in yticks])
    else:
        ax.set_yticks([])
    ax.set_title(title,fontweight="bold")
def draw_surface(ax,surface,xx,yy,aspect,title,cmap,vmin,vmax,show_y_ticks=False):
    masked=np.ma.masked_invalid(surface) if surface is not None else np.ma.masked_all(xx.shape)
    clipped=np.ma.clip(masked,vmin,vmax)
    mesh=ax.pcolormesh(xx,yy,clipped,cmap=cmap,vmin=vmin,vmax=vmax,shading="gouraud",edgecolors="none",linewidth=0,antialiased=False,rasterized=True)
    style_axis(ax,title,aspect,show_y_ticks=show_y_ticks)
    return mesh
def surface_to_df(surface,support,x,y,label,value_col):
    rows=[]
    for yi,yv in enumerate(y):
        for xi,xv in enumerate(x):
            rows.append({"map":label,"common_x":xv,"common_y":yv,value_col:surface[yi,xi] if surface is not None else np.nan,"support":int(support[yi,xi]) if support is not None else 0})
    return pd.DataFrame(rows)
if not DISTANCE_FILE.exists():
    raise FileNotFoundError(DISTANCE_FILE)
if not REG_FILE.exists():
    raise FileNotFoundError(f"Data-driven registered coordinates not found: {REG_FILE}")
distance_df=pd.read_csv(DISTANCE_FILE)
registered=pd.read_csv(REG_FILE)
distance_df["cell_index"]=pd.to_numeric(distance_df["cell_index"],errors="raise").astype(int)
registered["cell_index"]=pd.to_numeric(registered["cell_index"],errors="raise").astype(int)
distance_df["kd_gene"]=distance_df["kd_gene"].astype(str).str.strip()
distance_df["EB"]=distance_df["EB"].astype(str)
distance_df["cell_type"]=distance_df["cell_type"].astype(str).str.strip()
distance_df["distance_bin"]=distance_df["distance_bin"].astype(str)
registered["EB_registered"]=registered["EB_registered"].astype(str)
registered["cell_type_registered"]=registered["cell_type_registered"].astype(str).str.strip()
mapped=distance_df.merge(registered[["cell_index","EB_registered","cell_type_registered","common_x","common_y"]],on="cell_index",how="inner",validate="many_to_one")
if (mapped["EB"]!=mapped["EB_registered"]).any():
    raise ValueError("EB mismatch between KD-distance results and data-driven registered coordinates.")
if (mapped["cell_type"]!=mapped["cell_type_registered"]).any():
    raise ValueError("Cell-type mismatch between KD-distance results and data-driven registered coordinates.")
mapped=mapped[np.isfinite(mapped["common_x"])&np.isfinite(mapped["common_y"])].copy()
aspect=float(np.nanmax(np.abs(registered["common_y"].to_numpy(dtype=float))))
x,y,xx,yy,inside=make_grid(aspect)
available_kds=set(mapped["kd_gene"].unique())
available_cell_types=set(mapped["cell_type"].unique())
print(f"Using {len(mapped):,} receiver-cell rows from QC-passing registered EBs.")
print(f"Registered EBs: {mapped['EB'].nunique()}")
print(f"Canonical aspect ratio: {aspect:.3f}")
print("Available target cell types:",sorted(available_cell_types))
for group in TREND_GROUPS:
    name=group["name"]
    target=group["cell_type"]
    kds=[kd for kd in group["kds"] if kd in available_kds]
    print("\n"+"="*80)
    print(name)
    print("Cell type:",target)
    print("KDs:",kds)
    if target not in available_cell_types:
        print(f"WARNING: cell type '{target}' was not found. Skipping.")
        continue
    if not kds:
        print("No requested KDs found. Skipping.")
        continue
    out=OUTDIR/name
    out.mkdir(parents=True,exist_ok=True)
    bin_maps=[]
    bin_support=[]
    bin_tables=[]
    for dbin in DISTANCE_BINS:
        surf,support=group_distance_surface(mapped,kds,target,[dbin],xx,yy,inside)
        bin_maps.append(surf)
        bin_support.append(support)
        if surf is not None:
            bin_tables.append(surface_to_df(surf,support,x,y,dbin,"mean_proportion"))
    prop_vmin,prop_vmax=get_surface_limits(bin_maps,lower=1,upper=99,min_span=0.08)
    fig,axes=plt.subplots(2,3,figsize=(13.5,7.5),squeeze=False,layout="constrained")
    axes=axes.ravel()
    mesh=None
    for i,(dbin,surf) in enumerate(zip(DISTANCE_BINS,bin_maps)):
        mesh=draw_surface(axes[i],surf,xx,yy,aspect,f"{dbin} µm",PROPORTION_CMAP,prop_vmin,prop_vmax,show_y_ticks=(i%3==0))
    if mesh is not None:
        cbar=fig.colorbar(mesh,ax=axes.tolist(),shrink=0.82,pad=0.02)
        cbar.set_label(f"{target} local proportion")
    fig.supxlabel("Common major axis",fontsize=10)
    fig.supylabel("Common minor axis",fontsize=10)
    fig.suptitle(f"{target}: spatial proportion by distance from KD\n{' + '.join(kds)}",fontsize=13,fontweight="bold")
    fig.savefig(out/f"{name}_six_distance_bin_common_maps.png",dpi=400,bbox_inches="tight",facecolor="white")
    fig.savefig(out/f"{name}_six_distance_bin_common_maps.pdf",bbox_inches="tight",facecolor="white")
    plt.close(fig)
    if bin_tables:
        pd.concat(bin_tables,ignore_index=True).to_csv(out/f"{name}_six_distance_bin_surfaces.csv",index=False)
    near,far,delta,near_support,far_support,delta_support,per_kd=group_near_far_surfaces(mapped,kds,target,xx,yy,inside)
    if near is None or far is None or delta is None:
        print("Insufficient near/far data.")
        continue
    nf_vmin,nf_vmax=get_surface_limits([near,far],lower=1,upper=99,min_span=0.08)
    absvals=np.abs(delta[np.isfinite(delta)])
    delta_lim=max(float(np.nanpercentile(absvals,98)) if len(absvals)>0 else 0.05,0.05)
    fig,axes=plt.subplots(1,3,figsize=(13.5,4.5),layout="constrained")
    m1=draw_surface(axes[0],near,xx,yy,aspect,"Near KD: 0–100 µm",PROPORTION_CMAP,nf_vmin,nf_vmax,show_y_ticks=True)
    draw_surface(axes[1],far,xx,yy,aspect,"Far from KD: 200–300 µm",PROPORTION_CMAP,nf_vmin,nf_vmax,show_y_ticks=False)
    m2=draw_surface(axes[2],delta,xx,yy,aspect,"Far − Near",DELTA_CMAP,-delta_lim,delta_lim,show_y_ticks=False)
    c1=fig.colorbar(m1,ax=axes[:2].tolist(),shrink=0.78,pad=0.02)
    c1.set_label(f"{target} local proportion")
    c2=fig.colorbar(m2,ax=axes[2],shrink=0.78,pad=0.02)
    c2.set_label("Δ local proportion: far − near")
    fig.supxlabel("Common major axis",fontsize=10)
    fig.supylabel("Common minor axis",fontsize=10)
    fig.suptitle(f"{target}: common-coordinate KD-distance effect\n{' + '.join(kds)}",fontsize=13,fontweight="bold")
    fig.savefig(out/f"{name}_near_far_common_maps.png",dpi=400,bbox_inches="tight",facecolor="white")
    fig.savefig(out/f"{name}_near_far_common_maps.pdf",bbox_inches="tight",facecolor="white")
    plt.close(fig)
    pd.concat([surface_to_df(near,near_support,x,y,"near_0_100","mean_proportion"),surface_to_df(far,far_support,x,y,"far_200_300","mean_proportion"),surface_to_df(delta,delta_support,x,y,"far_minus_near","delta_far_minus_near")],ignore_index=True).to_csv(out/f"{name}_near_far_surfaces.csv",index=False)
    per_maps=[]
    for kd in kds:
        kd_delta=per_kd[kd][2]
        per_maps.append((kd,kd_delta))
    finite=[np.abs(arr[np.isfinite(arr)]) for _,arr in per_maps if arr is not None and np.isfinite(arr).any()]
    per_lim=max(float(np.nanpercentile(np.concatenate(finite),98)) if finite else 0.05,0.05)
    ncols=len(kds)
    fig,axes=plt.subplots(1,ncols,figsize=(4.8*ncols,4.1),squeeze=False,layout="constrained")
    axes=axes.ravel()
    mesh=None
    for i,(kd,arr) in enumerate(per_maps):
        mesh=draw_surface(axes[i],arr,xx,yy,aspect,kd,DELTA_CMAP,-per_lim,per_lim,show_y_ticks=(i==0))
    if mesh is not None:
        cbar=fig.colorbar(mesh,ax=axes.tolist(),shrink=0.82,pad=0.02)
        cbar.set_label("Δ local proportion: far − near")
    fig.supxlabel("Common major axis",fontsize=10)
    fig.supylabel("Common minor axis",fontsize=10)
    fig.suptitle(f"{target}: per-KD spatial far-minus-near effect",fontsize=13,fontweight="bold")
    fig.savefig(out/f"{name}_per_KD_far_minus_near_common_maps.png",dpi=400,bbox_inches="tight",facecolor="white")
    fig.savefig(out/f"{name}_per_KD_far_minus_near_common_maps.pdf",bbox_inches="tight",facecolor="white")
    plt.close(fig)
    print("Saved:",out.resolve())
print("\nFinished.")
print("All maps saved to:",OUTDIR.resolve())
from pathlib import Path
import sys
import subprocess
import numpy as np
import pandas as pd
try:
    import gseapy as gp
except ImportError:
    subprocess.check_call([sys.executable,"-m","pip","install","gseapy"])
    import gseapy as gp
BASE=Path("MULTIOME_KD_distance_cell_type_composition_and_GO/common_coordinate_data_driven_axes")
PC1_FILE=BASE/"PC1_data_driven_gene_scores.csv"
PC2_FILE=BASE/"PC2_data_driven_gene_scores.csv"
OUTDIR=BASE/"axis_ranked_GSEA"
OUTDIR.mkdir(parents=True,exist_ok=True)
SEED=20260828
N_PERM=1000
MIN_SIZE=10
MAX_SIZE=500
FDR_CUTOFF=0.05
TOP_N=20
pc1=pd.read_csv(PC1_FILE)
pc2=pd.read_csv(PC2_FILE)
print(f"PC1: {len(pc1):,} genes")
print(f"PC2: {len(pc2):,} genes")
available=gp.get_library_name(organism="human")
def choose_library(preferred,label):
    for x in preferred:
        if x in available:
            print(f"{label}: {x}")
            return x
    for x in preferred:
        hits=[y for y in available if x.lower() in y.lower()]
        if hits:
            print(f"{label}: {hits[0]}")
            return hits[0]
    print(f"WARNING: no library found for {label}")
    return None
libraries=[]
x=choose_library(["GO_Biological_Process_2025","GO_Biological_Process_2023"],"GO biological process")
if x: libraries.append(("GO_BP",x))
x=choose_library(["Reactome_Pathways_2024","Reactome_2022"],"Reactome")
if x: libraries.append(("Reactome",x))
x=choose_library(["MSigDB_Hallmark_2020"],"Hallmark")
if x: libraries.append(("Hallmark",x))
x=choose_library(["Descartes_Cell_Types_and_Tissue_2021"],"Developmental cell types")
if x: libraries.append(("Descartes",x))
x=choose_library(["PanglaoDB_Augmented_2021"],"Cell types")
if x: libraries.append(("PanglaoDB",x))
def make_rank(df,axis):
    d=df[["gene","meta_r"]].copy()
    d["gene"]=d["gene"].astype(str).str.strip()
    d["meta_r"]=pd.to_numeric(d["meta_r"],errors="coerce")
    d=d.replace([np.inf,-np.inf],np.nan).dropna()
    d=d[~d["gene"].isin(["","nan","None"])]
    d=d.groupby("gene",as_index=False)["meta_r"].mean()
    d=d.sort_values("meta_r",ascending=False).reset_index(drop=True)
    d.to_csv(OUTDIR/f"{axis}_ranked_genes.csv",index=False)
    return d
all_results=[]
for axis,df in [("PC1",pc1),("PC2",pc2)]:
    rank=make_rank(df,axis)
    print(f"\n{axis}: ranked {len(rank):,} genes")
    for lib_label,lib_name in libraries:
        print(f"  Running {lib_label}: {lib_name}")
        try:
            gene_sets=gp.get_library(name=lib_name,organism="human")
            pre=gp.prerank(rnk=rank[["gene","meta_r"]],gene_sets=gene_sets,min_size=MIN_SIZE,max_size=MAX_SIZE,permutation_num=N_PERM,seed=SEED,outdir=None,verbose=False)
            res=pre.res2d.copy()
        except Exception as e:
            print(f"  FAILED: {e}")
            continue
        if res is None or len(res)==0:
            continue
        res["Axis"]=axis
        res["Library_type"]=lib_label
        res["Library"]=lib_name
        all_results.append(res)
if not all_results:
    raise RuntimeError("No GSEA results returned.")
results=pd.concat(all_results,ignore_index=True)
for c in ["NES","NOM p-val","FDR q-val"]:
    if c in results.columns:
        results[c]=pd.to_numeric(results[c],errors="coerce")
results["Axis_direction"]=np.where(results["NES"]>0,"positive","negative")
results.to_csv(OUTDIR/"axis_GSEA_ALL.csv",index=False)
sig=results[results["FDR q-val"]<FDR_CUTOFF].copy()
sig=sig.sort_values(["Axis","Axis_direction","Library_type","FDR q-val","NES"],ascending=[True,True,True,True,False])
sig.to_csv(OUTDIR/"axis_GSEA_FDR05.csv",index=False)
top=[]
for (axis,direction,library),sub in sig.groupby(["Axis","Axis_direction","Library_type"],observed=True):
    s=sub.copy()
    s["abs_NES"]=s["NES"].abs()
    s=s.sort_values(["FDR q-val","abs_NES"],ascending=[True,False]).head(TOP_N)
    top.append(s)
top=pd.concat(top,ignore_index=True) if top else pd.DataFrame()
top.to_csv(OUTDIR/"axis_GSEA_TOP_TERMS.csv",index=False)
summary=[]
for axis in ["PC1","PC2"]:
    for direction in ["positive","negative"]:
        for lib_label,_ in libraries:
            s=sig[(sig["Axis"]==axis)&(sig["Axis_direction"]==direction)&(sig["Library_type"]==lib_label)].copy()
            if len(s):
                s["abs_NES"]=s["NES"].abs()
                s=s.sort_values(["FDR q-val","abs_NES"],ascending=[True,False])
            terms="; ".join(s["Term"].astype(str).head(10)) if len(s) else ""
            summary.append({"axis":axis,"direction":direction,"library":lib_label,"n_FDR05":len(s),"top_terms":terms})
summary=pd.DataFrame(summary)
summary.to_csv(OUTDIR/"axis_GSEA_SUMMARY.csv",index=False)
print("\n"+"="*90)
print("SIGNIFICANT AXIS ENRICHMENTS")
print("="*90)
for axis in ["PC1","PC2"]:
    for direction in ["positive","negative"]:
        print(f"\n{axis} {direction.upper()}")
        sub=sig[(sig["Axis"]==axis)&(sig["Axis_direction"]==direction)]
        if len(sub)==0:
            print("  No significant terms at FDR < 0.05")
            continue
        for lib_label,_ in libraries:
            s=sub[sub["Library_type"]==lib_label].copy()
            if len(s)==0:
                continue
            s["abs_NES"]=s["NES"].abs()
            s=s.sort_values(["FDR q-val","abs_NES"],ascending=[True,False]).head(8)
            print(f"\n  {lib_label}")
            for _,r in s.iterrows():
                print(f"    {r['Term']} | NES={r['NES']:.2f} | FDR={r['FDR q-val']:.3g}")
print("\nSaved to:",OUTDIR)
print("Main file to inspect:",OUTDIR/"axis_GSEA_SUMMARY.csv")
from pathlib import Path
import pandas as pd
BASE=Path("MULTIOME_KD_distance_cell_type_composition_and_GO/common_coordinate_data_driven_axes/axis_ranked_GSEA")
df=pd.read_csv(BASE/"axis_GSEA_ALL.csv")
x=df[(df["Axis"]=="PC1")&(df["NES"]>0)&(df["NOM p-val"]<0.05)].copy()
x=x.sort_values(["Library_type","NOM p-val","NES"],ascending=[True,True,False])
for lib in x["Library_type"].unique():
    print("\n"+"="*80)
    print(lib)
    print("="*80)
    s=x[x["Library_type"]==lib].head(25)
    print(s[["Term","NES","NOM p-val","FDR q-val"]].to_string(index=False))
