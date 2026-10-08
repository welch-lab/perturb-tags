#!/usr/bin/env python3
from __future__ import annotations
# Joint multiome RNA + ATAC GAT training
import argparse,copy,json,math,os,random
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter,MaxNLocator
import mudata as mu
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import cKDTree
from scipy.stats import pearsonr,spearmanr
from sklearn.metrics import mean_absolute_error,mean_squared_error
try:
    from torch_geometric.nn import GATv2Conv
    from torch_geometric.utils import scatter
except ImportError as exc:
    raise ImportError("This script requires PyTorch Geometric with GATv2Conv.") from exc
MODEL_VERSION="modality_specific_spatial_residual_tune_loss_weights_10EB_v1"
DATA_PATH=Path("/scratch/welchjd_root/welchjd1/javidgmh/15515_Multiome/catatac_work/motif_analysis/all_lanes_merged_CATATAC_QC_RNA_processed_EB_manual_clustering_cell_types.h5mu")
OUTDIR=Path("multiome_joint_RNA_ATAC_MODALITY_SPECIFIC_SPATIAL_RESIDUAL_TUNE_LOSS_WEIGHTS_10EB")
OUTDIR.mkdir(parents=True,exist_ok=True)
RNA_COUNTS_LAYER="counts"
ATAC_COUNTS_LAYER="counts"
SPATIAL_KEY="X_spatial"
EB_KEY="EB_manual_clustering"
CELL_TYPE_KEY="cell_type"
GUIDE_SOURCE_COL="guides_passing_str"
N_RNA_HVGS=1000
N_ATAC_HVFS=1000
N_HELDOUT_EBS=10
ANCHOR_TEST_EBS=["EB_42","EB_8","EB_34"]
ANCHOR_VALIDATION_EBS=["EB_49","EB_38","EB_9"]
EB_SELECTION_SEED=42
FINAL_SEEDS=[17,42,89]
RNA_GRAPH_CANDIDATES=[(8,25.0),(12,50.0),(20,50.0),(30,100.0)]
ATAC_GRAPH_CANDIDATES=[(4,25.0),(8,25.0),(12,50.0),(20,50.0),(30,100.0)]
TUNE_SEED_BASE=5000
TUNE_MLP_MAX_EPOCHS=300
TUNE_SPATIAL_MAX_EPOCHS=150
FINAL_MLP_MAX_EPOCHS=500
FINAL_SPATIAL_MAX_EPOCHS=350
EARLY_STOPPING_PATIENCE=25
LR_PATIENCE=7
MLP_LEARNING_RATE=0.003
SPATIAL_LEARNING_RATE=0.003
WEIGHT_DECAY=1e-5
DROPOUT=0.15
LOG1P_MSE_WEIGHT=0.03
RAW_CORR_WEIGHT=0.02
RAW_NMSE_WEIGHT=0.005
DEFAULT_LOSS_WEIGHTS=(LOG1P_MSE_WEIGHT,RAW_CORR_WEIGHT,RAW_NMSE_WEIGHT)
LOSS_WEIGHT_CANDIDATES=[(0.03,0.02,0.005),(0.03,0.05,0.005),(0.03,0.10,0.010),(0.05,0.05,0.010),(0.05,0.10,0.020)]
OWN_HIDDEN_DIM=128
OWN_LATENT_DIM=64
MESSAGE_HIDDEN_DIM=128
SPATIAL_DIM=128
CONTEXT_DIM=64
SPATIAL_DECODER_HIDDEN=64
EDGE_HIDDEN_DIM=16
EDGE_DIM=8
GAT_HEADS=4
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_THREADS=int(os.environ.get("SLURM_CPUS_PER_TASK",os.environ.get("OMP_NUM_THREADS","8")))
torch.set_num_threads(max(1,N_THREADS))
try:torch.set_num_interop_threads(max(1,min(4,N_THREADS)))
except RuntimeError:pass
SUMMARY_METRICS=["test_joint_nb_nll","test_rna_nb_nll","test_atac_nb_nll","rna_raw_pearson","rna_raw_spearman","rna_raw_rmse","rna_raw_mae","rna_log1p_pearson","rna_log1p_spearman","rna_log1p_rmse","rna_log1p_mae","atac_raw_pearson","atac_raw_spearman","atac_raw_rmse","atac_raw_mae","atac_log1p_pearson","atac_log1p_spearman","atac_log1p_rmse","atac_log1p_mae"]
def seed_all(seed:int)->None:
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
def to_csr(x)->sp.csr_matrix:
    if sp.issparse(x):return x.tocsr().astype(np.float32)
    try:x=x[:]
    except Exception:pass
    if sp.issparse(x):return x.tocsr().astype(np.float32)
    return sp.csr_matrix(np.asarray(x,dtype=np.float32))
def dense32(x)->np.ndarray:
    return x.toarray().astype(np.float32) if sp.issparse(x) else np.asarray(x,dtype=np.float32)
def validate_count_matrix(name:str,x:sp.csr_matrix,require_nonbinary:bool)->None:
    if x.ndim!=2 or x.shape[0]<1 or x.shape[1]<1:raise ValueError(f"{name} matrix has invalid shape {x.shape}")
    if x.nnz==0:raise ValueError(f"{name} matrix has no nonzero counts")
    data=np.asarray(x.data,dtype=np.float64)
    if not np.all(np.isfinite(data)):raise ValueError(f"{name} matrix contains non-finite values")
    if np.any(data<0):raise ValueError(f"{name} matrix contains negative values")
    if not np.allclose(data,np.rint(data),atol=1e-5,rtol=0):raise ValueError(f"{name} matrix is not integer-like count data")
    if require_nonbinary and float(data.max())<=1.0:raise ValueError(f"{name} matrix is binary; NB count loss requires non-binary counts")
def get_counts(adata,layer:str,name:str,require_nonbinary:bool)->sp.csr_matrix:
    counts=to_csr(adata.layers[layer] if layer in adata.layers else adata.X)
    validate_count_matrix(name,counts,require_nonbinary)
    return counts
def valid_eb_label(value:object)->bool:
    s=str(value).strip()
    return s.startswith("EB_") and s.lower() not in {"eb_","eb_nan","eb_none","eb_unassigned"}
def parse_guides(value)->list[str]:
    if value is None or (isinstance(value,float) and np.isnan(value)):return []
    if isinstance(value,(list,tuple,np.ndarray)):values=value
    else:
        s=str(value).strip()
        if s=="" or s.lower() in {"nan","none","unassigned","no guide","no_guide"}:return []
        values=s.split(";")
    return [str(x).strip() for x in values if str(x).strip()]
def build_guides(obs:pd.DataFrame)->tuple[np.ndarray,list[str]]:
    if GUIDE_SOURCE_COL not in obs.columns:raise KeyError(f"Missing obs[{GUIDE_SOURCE_COL!r}]")
    rows=[parse_guides(v) for v in obs[GUIDE_SOURCE_COL].astype(object).to_numpy()]
    names=sorted({g for row in rows for g in row})
    if not names:raise ValueError("No guides were parsed")
    lookup={g:i for i,g in enumerate(names)}
    matrix=np.zeros((len(rows),len(names)),dtype=np.float32)
    for i,row in enumerate(rows):
        for guide in row:matrix[i,lookup[guide]]=1.0
    return matrix,names
def log_normalize_counts(counts:sp.csr_matrix,target_sum:float=1e4)->sp.csr_matrix:
    totals=np.asarray(counts.sum(1)).ravel().astype(np.float32)
    if np.any(totals<=0):raise ValueError("Cannot normalize cells with non-positive library size")
    out=(sp.diags(target_sum/totals)@counts).tocsr().astype(np.float32)
    out.data=np.log1p(out.data)
    return out
def sparse_variance(x:sp.csr_matrix)->np.ndarray:
    mean=np.asarray(x.mean(0)).ravel();mean2=np.asarray(x.power(2).mean(0)).ravel()
    return np.maximum(mean2-mean**2,0)
def select_rna_hvgs(xlog:sp.csr_matrix,train_mask:np.ndarray,n:int)->tuple[np.ndarray,np.ndarray]:
    train=xlog[train_mask];variance=sparse_variance(train);detected=np.asarray((train>0).sum(0)).ravel()>=10;variance[~detected]=-np.inf
    available=np.flatnonzero(np.isfinite(variance))
    if len(available)<n:raise ValueError(f"Only {len(available)} eligible RNA features are available; {n} requested")
    values=variance[available];chosen=available[np.argpartition(values,-n)[-n:]];chosen=chosen[np.argsort(variance[chosen])[::-1]].astype(int)
    return chosen,variance[chosen].astype(np.float64)
def select_atac_deviance_features(counts:sp.csr_matrix,train_mask:np.ndarray,n:int)->tuple[np.ndarray,np.ndarray]:
    train=counts[train_mask].tocsr()
    library=np.asarray(train.sum(1)).ravel()
    if np.any(library<=0):raise ValueError("ATAC training cells contain non-positive library sizes")
    feature_total=np.asarray(train.sum(0)).ravel();detected=np.asarray((train>0).sum(0)).ravel();valid=(detected>=10)&(feature_total>0)
    total=float(library.sum())
    if total<=0:raise ValueError("ATAC training total is non-positive")
    probabilities=feature_total/total
    coo=train.tocoo();mu_nonzero=library[coo.row]*probabilities[coo.col]
    if np.any(mu_nonzero<=0):raise ValueError("Non-positive ATAC null expectation during deviance selection")
    contribution=2.0*coo.data*np.log(coo.data/mu_nonzero)
    deviance=np.bincount(coo.col,weights=contribution,minlength=train.shape[1]).astype(np.float64)
    deviance=np.maximum(deviance,0.0);deviance[~valid]=-np.inf
    available=np.flatnonzero(np.isfinite(deviance))
    if len(available)<n:raise ValueError(f"Only {len(available)} eligible ATAC features are available; {n} requested")
    values=deviance[available];chosen=available[np.argpartition(values,-n)[-n:]];chosen=chosen[np.argsort(deviance[chosen])[::-1]].astype(int)
    return chosen,deviance[chosen]
def eb_sort_key(label:str)->tuple[int,str]:
    s=str(label)
    try:return int(s.split("_",1)[1]),s
    except Exception:return 10**9,s
def split_coverage_ok(matrix:np.ndarray,train:np.ndarray,val:np.ndarray,test:np.ndarray)->bool:
    train_present=matrix[train].sum(0)>0;return bool(np.all(~((matrix[val].sum(0)>0)&~train_present)) and np.all(~((matrix[test].sum(0)>0)&~train_present)))
def choose_splits(labels:np.ndarray,guides:np.ndarray,celltypes:np.ndarray)->list[tuple[np.ndarray,np.ndarray,np.ndarray,str,str]]:
    unique=sorted(set(map(str,pd.unique(labels))),key=eb_sort_key);anchors=list(zip(ANCHOR_TEST_EBS,ANCHOR_VALIDATION_EBS));required=set(ANCHOR_TEST_EBS+ANCHOR_VALIDATION_EBS);missing=sorted(required-set(unique),key=eb_sort_key)
    if missing:raise ValueError(f"Requested anchor test/validation EBs are missing: {missing}")
    if len(ANCHOR_TEST_EBS)!=len(ANCHOR_VALIDATION_EBS):raise ValueError("Anchor test and validation EB lists must have equal length")
    if set(ANCHOR_TEST_EBS)&set(ANCHOR_VALIDATION_EBS):raise ValueError("An anchor EB cannot be both test and validation")
    n_extra=N_HELDOUT_EBS-len(anchors)
    if n_extra<0:raise ValueError("N_HELDOUT_EBS is smaller than the number of anchor test EBs")
    remaining=[eb for eb in unique if eb not in required];rng=np.random.default_rng(EB_SELECTION_SEED);remaining=np.asarray(remaining,dtype=object);rng.shuffle(remaining);unused=list(map(str,remaining));pairs=list(anchors)
    while len(pairs)<N_HELDOUT_EBS:
        found=False
        for i,test_eb in enumerate(unused):
            for j,val_eb in enumerate(unused):
                if i==j:continue
                train=(labels!=test_eb)&(labels!=val_eb);val=labels==val_eb;test=labels==test_eb
                if train.any() and val.any() and test.any() and split_coverage_ok(guides,train,val,test) and split_coverage_ok(celltypes,train,val,test):
                    pairs.append((test_eb,val_eb));unused=[eb for eb in unused if eb not in {test_eb,val_eb}];found=True;break
            if found:break
        if not found:raise ValueError(f"Could select only {len(pairs)} valid test/validation EB pairs; {N_HELDOUT_EBS} requested")
    if len({x for pair in pairs for x in pair})!=2*len(pairs):raise RuntimeError("Test and validation EB selection contains duplicates")
    splits=[]
    for test_eb,val_eb in pairs:
        train=(labels!=test_eb)&(labels!=val_eb);val=labels==val_eb;test=labels==test_eb
        if not split_coverage_ok(guides,train,val,test) or not split_coverage_ok(celltypes,train,val,test):raise ValueError(f"Guide/cell-type coverage failed for test={test_eb}, validation={val_eb}")
        splits.append((train,val,test,val_eb,test_eb))
    return splits
def build_knn_graph(coords:np.ndarray,labels:np.ndarray,k:int)->tuple[torch.Tensor,torch.Tensor]:
    if k<1:raise ValueError("k must be at least 1")
    if coords.ndim!=2 or coords.shape[1]<2 or len(coords)!=len(labels):raise ValueError("Spatial coordinates and EB labels are misaligned")
    if not np.all(np.isfinite(coords[:,:2])):raise ValueError("Spatial coordinates contain non-finite values")
    sources=[];destinations=[];distances=[]
    for eb in pd.unique(labels):
        idx=np.flatnonzero(labels==eb)
        if len(idx)<2:raise ValueError(f"EB {eb} has fewer than 2 cells")
        kk=min(k,len(idx)-1);d,nn_idx=cKDTree(coords[idx,:2]).query(coords[idx,:2],k=kk+1);d=np.atleast_2d(d);nn_idx=np.atleast_2d(nn_idx)
        for local_i,global_i in enumerate(idx):
            keep=nn_idx[local_i]!=local_i;chosen_nn=nn_idx[local_i][keep][:kk];chosen_d=d[local_i][keep][:kk]
            if len(chosen_nn)!=kk:raise RuntimeError(f"Could not construct {kk} neighbors for EB {eb}, cell {global_i}")
            sources.extend([global_i]*kk);destinations.extend(idx[chosen_nn].tolist());distances.extend(np.asarray(chosen_d,dtype=np.float32).tolist())
    src=np.asarray(sources,dtype=np.int64);dst=np.asarray(destinations,dtype=np.int64);dist=np.asarray(distances,dtype=np.float32)
    pairs=np.concatenate([np.stack([src,dst],1),np.stack([dst,src],1)],0);pair_dist=np.concatenate([dist,dist]);order=np.lexsort((pairs[:,1],pairs[:,0]));pairs=pairs[order];pair_dist=pair_dist[order]
    keep=np.ones(len(pairs),dtype=bool);keep[1:]=np.any(pairs[1:]!=pairs[:-1],axis=1);pairs=pairs[keep];pair_dist=pair_dist[keep]
    if np.any(pairs[:,0]==pairs[:,1]):raise RuntimeError("Graph contains self-edges")
    if np.any(labels[pairs[:,0]]!=labels[pairs[:,1]]):raise RuntimeError("Graph contains cross-EB edges")
    return torch.tensor(pairs.T,dtype=torch.long),torch.tensor(pair_dist[:,None],dtype=torch.float32)
def weighted_neighbor_context(guides:np.ndarray,celltypes:np.ndarray,edge_index:torch.Tensor,edge_distance:torch.Tensor,decay:float)->np.ndarray:
    src,dst=edge_index;distance=edge_distance.reshape(-1).float();weights=torch.exp(-distance/float(decay))[:,None]
    guide_tensor=torch.tensor(guides,dtype=torch.float32);celltype_tensor=torch.tensor(celltypes,dtype=torch.float32)
    weighted_guides=scatter(guide_tensor[src]*weights,dst,dim=0,dim_size=len(guides),reduce="sum")
    weighted_celltypes=scatter(celltype_tensor[src]*weights,dst,dim=0,dim_size=len(guides),reduce="sum")
    weight_sum=scatter(weights,dst,dim=0,dim_size=len(guides),reduce="sum").clamp_min(1e-8)
    degree=scatter(torch.ones_like(weights),dst,dim=0,dim_size=len(guides),reduce="sum").clamp_min(1.0)
    mean_distance=scatter(distance[:,None]*weights,dst,dim=0,dim_size=len(guides),reduce="sum")/weight_sum
    min_distance=scatter(distance[:,None],dst,dim=0,dim_size=len(guides),reduce="min")
    context=torch.cat([weighted_guides/weight_sum,weighted_celltypes/weight_sum,torch.log1p(degree),mean_distance/float(decay),min_distance/float(decay)],dim=1)
    out=context.numpy().astype(np.float32)
    if not np.all(np.isfinite(out)):raise ValueError("Non-finite weighted neighborhood context")
    return out
def check_split_feature_coverage(matrix:np.ndarray,names:list[str],train:np.ndarray,val:np.ndarray,test:np.ndarray,label:str)->None:
    train_present=matrix[train].sum(0)>0;val_present=matrix[val].sum(0)>0;test_present=matrix[test].sum(0)>0
    missing_val=[names[i] for i in np.flatnonzero(val_present&~train_present)];missing_test=[names[i] for i in np.flatnonzero(test_present&~train_present)]
    if missing_val or missing_test:raise ValueError(f"{label} categories absent from training: validation={missing_val[:10]}, test={missing_test[:10]}")
def positive_theta(raw_theta:torch.Tensor)->torch.Tensor:
    return F.softplus(raw_theta)+1e-4
def nb_log_prob_from_mu(y:torch.Tensor,mu:torch.Tensor,theta:torch.Tensor)->torch.Tensor:
    mu=mu.clamp_min(1e-12);theta=theta.clamp_min(1e-4);logmu=torch.log(mu);logtheta=torch.log(theta);log_theta_plus_mu=torch.logaddexp(logtheta[None,:],logmu)
    return torch.lgamma(y+theta[None,:])-torch.lgamma(theta[None,:])-torch.lgamma(y+1)+theta[None,:]*(logtheta[None,:]-log_theta_plus_mu)+y*(logmu-log_theta_plus_mu)
def nb_nll(residual:torch.Tensor,y:torch.Tensor,log_library:torch.Tensor,mu0:torch.Tensor,raw_theta:torch.Tensor)->torch.Tensor:
    mu=torch.exp((residual+log_library[:,None]+mu0[None,:]).clamp(-12,12))
    return -nb_log_prob_from_mu(y,mu,positive_theta(raw_theta)).mean()
def make_mu(residual:torch.Tensor,log_library:torch.Tensor,mu0:torch.Tensor)->torch.Tensor:
    return torch.exp((residual+log_library[:,None]+mu0[None,:]).clamp(-12,12))
def torch_pearson(x:torch.Tensor,y:torch.Tensor)->torch.Tensor:
    x=x.reshape(-1);y=y.reshape(-1);xc=x-x.mean();yc=y-y.mean();den=torch.sqrt((xc.square().mean()+1e-8)*(yc.square().mean()+1e-8))
    return (xc*yc).mean()/den
def modality_terms(residual:torch.Tensor,y:torch.Tensor,log_library:torch.Tensor,mu0:torch.Tensor,raw_theta:torch.Tensor,loss_weights:tuple[float,float,float]=DEFAULT_LOSS_WEIGHTS)->dict[str,torch.Tensor]:
    log1p_mse_weight,corr_weight,nmse_weight=map(float,loss_weights);mu=make_mu(residual,log_library,mu0);nb=nb_nll(residual,y,log_library,mu0,raw_theta);log_mse=F.mse_loss(torch.log1p(mu),torch.log1p(y));corr=torch_pearson(mu,y);nmse=F.mse_loss(mu,y)/(torch.var(y,unbiased=False)+1e-6);objective=nb+log1p_mse_weight*log_mse+corr_weight*(1.0-corr)+nmse_weight*nmse
    return {"mu":mu,"nb":nb,"log_mse":log_mse,"corr":corr,"nmse":nmse,"objective":objective}
def safe_corr(x:np.ndarray,y:np.ndarray,method:str)->float:
    if len(x)<2 or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)) or np.std(x)==0 or np.std(y)==0:return np.nan
    return float(pearsonr(x,y).statistic if method=="pearson" else spearmanr(x,y).statistic)
def metrics(y:np.ndarray,mu:np.ndarray,prefix:str)->dict[str,float]:
    yt=np.asarray(y,dtype=np.float64).ravel();yp=np.asarray(mu,dtype=np.float64).ravel();valid=np.isfinite(yt)&np.isfinite(yp)&(yt>=0)&(yp>=0);yt=yt[valid];yp=yp[valid]
    if yt.size==0:raise ValueError(f"No valid {prefix} observations")
    xl=np.log1p(yt);yl=np.log1p(yp)
    return {f"{prefix}_raw_pearson":safe_corr(yt,yp,"pearson"),f"{prefix}_raw_spearman":safe_corr(yt,yp,"spearman"),f"{prefix}_raw_rmse":float(np.sqrt(mean_squared_error(yt,yp))),f"{prefix}_raw_mae":float(mean_absolute_error(yt,yp)),f"{prefix}_log1p_pearson":safe_corr(xl,yl,"pearson"),f"{prefix}_log1p_spearman":safe_corr(xl,yl,"spearman"),f"{prefix}_log1p_rmse":float(np.sqrt(mean_squared_error(xl,yl))),f"{prefix}_log1p_mae":float(mean_absolute_error(xl,yl))}
class NonSpatialModality(nn.Module):
    def __init__(self,input_dim:int,n_guides:int,n_celltypes:int,n_targets:int):
        super().__init__();self.guide_main=nn.Linear(n_guides,n_targets,bias=False);self.celltype_main=nn.Linear(n_celltypes,n_targets,bias=False);self.own_encoder=nn.Sequential(nn.Linear(input_dim,OWN_HIDDEN_DIM),nn.LayerNorm(OWN_HIDDEN_DIM),nn.ELU(),nn.Dropout(DROPOUT),nn.Linear(OWN_HIDDEN_DIM,OWN_LATENT_DIM),nn.ELU());self.own_head=nn.Linear(OWN_LATENT_DIM,n_targets,bias=False);self.raw_theta=nn.Parameter(torch.full((n_targets,),10.0))
    def forward(self,x:torch.Tensor,guides:torch.Tensor,celltypes:torch.Tensor)->torch.Tensor:
        return self.guide_main(guides)+self.celltype_main(celltypes)+self.own_head(self.own_encoder(x))
class SpatialResidualModality(nn.Module):
    def __init__(self,input_dim:int,n_guides:int,n_celltypes:int,n_targets:int,context_input_dim:int,distance_decay:float):
        super().__init__();self.distance_decay=float(distance_decay);self.base=NonSpatialModality(input_dim,n_guides,n_celltypes,n_targets);self.source_encoder=nn.Sequential(nn.Linear(input_dim,MESSAGE_HIDDEN_DIM),nn.LayerNorm(MESSAGE_HIDDEN_DIM),nn.ELU());self.receiver_encoder=nn.Sequential(nn.Linear(n_celltypes,MESSAGE_HIDDEN_DIM),nn.LayerNorm(MESSAGE_HIDDEN_DIM),nn.ELU());self.edge_encoder=nn.Sequential(nn.Linear(4,EDGE_HIDDEN_DIM),nn.ELU(),nn.Linear(EDGE_HIDDEN_DIM,EDGE_DIM));self.gat=GATv2Conv((MESSAGE_HIDDEN_DIM,MESSAGE_HIDDEN_DIM),SPATIAL_DIM//GAT_HEADS,heads=GAT_HEADS,concat=True,dropout=DROPOUT,add_self_loops=False,edge_dim=EDGE_DIM,bias=True,share_weights=False);self.context_encoder=nn.Sequential(nn.Linear(context_input_dim,CONTEXT_DIM),nn.LayerNorm(CONTEXT_DIM),nn.ELU());self.gate=nn.Sequential(nn.Linear(MESSAGE_HIDDEN_DIM+SPATIAL_DIM+CONTEXT_DIM,SPATIAL_DIM),nn.Sigmoid());self.correction_encoder=nn.Sequential(nn.Linear(MESSAGE_HIDDEN_DIM+SPATIAL_DIM+CONTEXT_DIM,SPATIAL_DECODER_HIDDEN),nn.LayerNorm(SPATIAL_DECODER_HIDDEN),nn.ELU(),nn.Dropout(DROPOUT));self.correction_head=nn.Linear(SPATIAL_DECODER_HIDDEN,n_targets,bias=False);nn.init.zeros_(self.correction_head.weight);nn.init.constant_(self.gate[0].bias,-1.5)
    def forward(self,x:torch.Tensor,guides:torch.Tensor,celltypes:torch.Tensor,edge_index:torch.Tensor,edge_distance:torch.Tensor,neighbor_context:torch.Tensor)->torch.Tensor:
        base=self.base(x,guides,celltypes);source=self.source_encoder(x);receiver=self.receiver_encoder(celltypes);scaled=edge_distance/self.distance_decay;edge_raw=torch.cat([scaled,torch.exp(-scaled),1.0/(1.0+scaled),torch.log1p(edge_distance)/math.log1p(self.distance_decay)],dim=1);edge_attr=self.edge_encoder(edge_raw);spatial=F.elu(self.gat((source,receiver),edge_index,edge_attr=edge_attr));context=self.context_encoder(neighbor_context);gate=self.gate(torch.cat([receiver,spatial,context],dim=1));correction=self.correction_head(self.correction_encoder(torch.cat([receiver,gate*spatial,context],dim=1)));return base+correction
    def spatial_parameters(self):
        for name,param in self.named_parameters():
            if not name.startswith("base."):yield param
class MultiomicMLP(nn.Module):
    def __init__(self,input_dim:int,n_guides:int,n_celltypes:int,n_rna:int,n_atac:int):
        super().__init__();self.rna=NonSpatialModality(input_dim,n_guides,n_celltypes,n_rna);self.atac=NonSpatialModality(input_dim,n_guides,n_celltypes,n_atac)
    def forward(self,x:torch.Tensor,guides:torch.Tensor,celltypes:torch.Tensor):
        return self.rna(x,guides,celltypes),self.atac(x,guides,celltypes)
class MultiomicSpatialResidualGAT(nn.Module):
    def __init__(self,input_dim:int,n_guides:int,n_celltypes:int,n_rna:int,n_atac:int,context_input_dim:int,rna_decay:float,atac_decay:float):
        super().__init__();self.rna=SpatialResidualModality(input_dim,n_guides,n_celltypes,n_rna,context_input_dim,rna_decay);self.atac=SpatialResidualModality(input_dim,n_guides,n_celltypes,n_atac,context_input_dim,atac_decay)
    def warm_start_from_mlp(self,mlp:MultiomicMLP)->None:
        self.rna.base.load_state_dict(copy.deepcopy(mlp.rna.state_dict()));self.atac.base.load_state_dict(copy.deepcopy(mlp.atac.state_dict()))
    def forward(self,x:torch.Tensor,guides:torch.Tensor,celltypes:torch.Tensor,rna_bundle:tuple,atac_bundle:tuple):
        rna_edge,rna_dist,rna_context=rna_bundle;atac_edge,atac_dist,atac_context=atac_bundle
        return self.rna(x,guides,celltypes,rna_edge,rna_dist,rna_context),self.atac(x,guides,celltypes,atac_edge,atac_dist,atac_context)
def copy_state(module:nn.Module)->dict[str,torch.Tensor]:
    return {key:value.detach().cpu().clone() for key,value in module.state_dict().items()}
def evaluate_mlp(model:MultiomicMLP,x,guides,celltypes,y_rna,y_atac,loglib_rna,mu0_rna,loglib_atac,mu0_atac,mask,rna_weights:tuple[float,float,float]=DEFAULT_LOSS_WEIGHTS,atac_weights:tuple[float,float,float]=DEFAULT_LOSS_WEIGHTS)->tuple[dict,dict]:
    model.eval()
    with torch.no_grad():rna_residual,atac_residual=model(x,guides,celltypes);rna_terms=modality_terms(rna_residual[mask],y_rna[mask],loglib_rna[mask],mu0_rna,model.rna.raw_theta,rna_weights);atac_terms=modality_terms(atac_residual[mask],y_atac[mask],loglib_atac[mask],mu0_atac,model.atac.raw_theta,atac_weights)
    return rna_terms,atac_terms
def evaluate_gat(model:MultiomicSpatialResidualGAT,x,guides,celltypes,rna_bundle,atac_bundle,y_rna,y_atac,loglib_rna,mu0_rna,loglib_atac,mu0_atac,mask,rna_weights:tuple[float,float,float]=DEFAULT_LOSS_WEIGHTS,atac_weights:tuple[float,float,float]=DEFAULT_LOSS_WEIGHTS)->tuple[dict,dict]:
    model.eval()
    with torch.no_grad():rna_residual,atac_residual=model(x,guides,celltypes,rna_bundle,atac_bundle);rna_terms=modality_terms(rna_residual[mask],y_rna[mask],loglib_rna[mask],mu0_rna,model.rna.base.raw_theta,rna_weights);atac_terms=modality_terms(atac_residual[mask],y_atac[mask],loglib_atac[mask],mu0_atac,model.atac.base.raw_theta,atac_weights)
    return rna_terms,atac_terms
def train_mlp(model:MultiomicMLP,x,guides,celltypes,y_rna,y_atac,loglib_rna,mu0_rna,loglib_atac,mu0_atac,train_mask,val_mask,max_epochs:int,save_prefix:str|None,rna_weights:tuple[float,float,float]=DEFAULT_LOSS_WEIGHTS,atac_weights:tuple[float,float,float]=DEFAULT_LOSS_WEIGHTS)->tuple[MultiomicMLP,float,float]:
    model=model.to(DEVICE);optimizer=torch.optim.AdamW(model.parameters(),lr=MLP_LEARNING_RATE,weight_decay=WEIGHT_DECAY);scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="min",patience=LR_PATIENCE,factor=0.5,min_lr=1e-7);best_rna=np.inf;best_atac=np.inf;best_rna_state=None;best_atac_state=None;stale_rna=0;stale_atac=0;history=[]
    for epoch in range(1,max_epochs+1):
        model.train();optimizer.zero_grad(set_to_none=True);rna_residual,atac_residual=model(x,guides,celltypes);rna_train=modality_terms(rna_residual[train_mask],y_rna[train_mask],loglib_rna[train_mask],mu0_rna,model.rna.raw_theta,rna_weights);atac_train=modality_terms(atac_residual[train_mask],y_atac[train_mask],loglib_atac[train_mask],mu0_atac,model.atac.raw_theta,atac_weights);loss=0.5*(rna_train["objective"]+atac_train["objective"])
        if not torch.isfinite(loss):raise FloatingPointError(f"Non-finite MLP loss for {save_prefix or 'tuning'} epoch {epoch}")
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step();rna_val,atac_val=evaluate_mlp(model,x,guides,celltypes,y_rna,y_atac,loglib_rna,mu0_rna,loglib_atac,mu0_atac,val_mask,rna_weights,atac_weights);rna_score=float(rna_val["objective"]);atac_score=float(atac_val["objective"]);scheduler.step(0.5*(rna_score+atac_score));history.append({"epoch":epoch,"train_rna_objective":float(rna_train["objective"].detach()),"train_atac_objective":float(atac_train["objective"].detach()),"val_rna_score":rna_score,"val_atac_score":atac_score,"val_rna_nb_nll":float(rna_val["nb"]),"val_atac_nb_nll":float(atac_val["nb"]),"val_rna_raw_pearson":float(rna_val["corr"]),"val_atac_raw_pearson":float(atac_val["corr"]),"lr":optimizer.param_groups[0]["lr"]})
        if rna_score<best_rna-1e-7:best_rna=rna_score;best_rna_state=copy_state(model.rna);stale_rna=0
        else:stale_rna+=1
        if atac_score<best_atac-1e-7:best_atac=atac_score;best_atac_state=copy_state(model.atac);stale_atac=0
        else:stale_atac+=1
        if epoch==1 or epoch%25==0:print(save_prefix or "TUNE_MLP",epoch,float(loss.detach()),rna_score,atac_score,flush=True)
        if stale_rna>=EARLY_STOPPING_PATIENCE and stale_atac>=EARLY_STOPPING_PATIENCE:break
    if best_rna_state is None or best_atac_state is None:raise RuntimeError("MLP did not produce valid branch checkpoints")
    model.rna.load_state_dict(best_rna_state);model.atac.load_state_dict(best_atac_state)
    if save_prefix is not None:torch.save(model.state_dict(),OUTDIR/f"{save_prefix}_mlp_best.pt");pd.DataFrame(history).to_csv(OUTDIR/f"{save_prefix}_mlp_history.csv",index=False)
    return model,best_rna,best_atac
def train_single_spatial(branch:SpatialResidualModality,x,guides,celltypes,bundle,y,loglib,mu0,train_mask,val_mask,max_epochs:int,loss_weights:tuple[float,float,float]=DEFAULT_LOSS_WEIGHTS)->tuple[SpatialResidualModality,dict[str,float]]:
    branch=branch.to(DEVICE)
    for parameter in branch.base.parameters():parameter.requires_grad=False
    optimizer=torch.optim.AdamW(list(branch.spatial_parameters()),lr=SPATIAL_LEARNING_RATE,weight_decay=WEIGHT_DECAY);scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="min",patience=LR_PATIENCE,factor=0.5,min_lr=1e-7);edge_index,edge_distance,context=bundle
    branch.eval()
    with torch.no_grad():initial_residual=branch(x,guides,celltypes,edge_index,edge_distance,context);initial=modality_terms(initial_residual[val_mask],y[val_mask],loglib[val_mask],mu0,branch.base.raw_theta,loss_weights)
    best=float(initial["objective"]);best_state=copy_state(branch);best_metrics={"best_val_score":best,"best_val_nb_nll":float(initial["nb"]),"best_val_raw_pearson":float(initial["corr"]),"best_val_raw_rmse":float(torch.sqrt(F.mse_loss(initial["mu"],y[val_mask])))};stale=0
    for epoch in range(1,max_epochs+1):
        branch.train();branch.base.eval();optimizer.zero_grad(set_to_none=True);residual=branch(x,guides,celltypes,edge_index,edge_distance,context);train_terms=modality_terms(residual[train_mask],y[train_mask],loglib[train_mask],mu0,branch.base.raw_theta,loss_weights);loss=train_terms["objective"]
        if not torch.isfinite(loss):raise FloatingPointError(f"Non-finite spatial tuning loss at epoch {epoch}")
        loss.backward();torch.nn.utils.clip_grad_norm_(list(branch.spatial_parameters()),1.0);optimizer.step();branch.eval()
        with torch.no_grad():val_residual=branch(x,guides,celltypes,edge_index,edge_distance,context);val_terms=modality_terms(val_residual[val_mask],y[val_mask],loglib[val_mask],mu0,branch.base.raw_theta,loss_weights)
        score=float(val_terms["objective"]);scheduler.step(score)
        if score<best-1e-7:best=score;best_state=copy_state(branch);best_metrics={"best_val_score":best,"best_val_nb_nll":float(val_terms["nb"]),"best_val_raw_pearson":float(val_terms["corr"]),"best_val_raw_rmse":float(torch.sqrt(F.mse_loss(val_terms["mu"],y[val_mask])))};stale=0
        else:stale+=1
        if stale>=EARLY_STOPPING_PATIENCE:break
    branch.load_state_dict(best_state)
    for parameter in branch.base.parameters():parameter.requires_grad=True
    return branch,best_metrics
def train_gat(model:MultiomicSpatialResidualGAT,x,guides,celltypes,rna_bundle,atac_bundle,y_rna,y_atac,loglib_rna,mu0_rna,loglib_atac,mu0_atac,train_mask,val_mask,max_epochs:int,save_prefix:str,rna_weights:tuple[float,float,float]=DEFAULT_LOSS_WEIGHTS,atac_weights:tuple[float,float,float]=DEFAULT_LOSS_WEIGHTS)->MultiomicSpatialResidualGAT:
    model=model.to(DEVICE)
    for parameter in model.rna.base.parameters():parameter.requires_grad=False
    for parameter in model.atac.base.parameters():parameter.requires_grad=False
    spatial_params=list(model.rna.spatial_parameters())+list(model.atac.spatial_parameters());optimizer=torch.optim.AdamW(spatial_params,lr=SPATIAL_LEARNING_RATE,weight_decay=WEIGHT_DECAY);scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="min",patience=LR_PATIENCE,factor=0.5,min_lr=1e-7);rna_initial,atac_initial=evaluate_gat(model,x,guides,celltypes,rna_bundle,atac_bundle,y_rna,y_atac,loglib_rna,mu0_rna,loglib_atac,mu0_atac,val_mask,rna_weights,atac_weights);best_rna=float(rna_initial["objective"]);best_atac=float(atac_initial["objective"]);best_rna_state=copy_state(model.rna);best_atac_state=copy_state(model.atac);stale_rna=0;stale_atac=0;history=[{"epoch":0,"val_rna_score":best_rna,"val_atac_score":best_atac,"val_rna_nb_nll":float(rna_initial["nb"]),"val_atac_nb_nll":float(atac_initial["nb"]),"val_rna_raw_pearson":float(rna_initial["corr"]),"val_atac_raw_pearson":float(atac_initial["corr"]),"lr":SPATIAL_LEARNING_RATE}]
    for epoch in range(1,max_epochs+1):
        model.train();model.rna.base.eval();model.atac.base.eval();optimizer.zero_grad(set_to_none=True);rna_residual,atac_residual=model(x,guides,celltypes,rna_bundle,atac_bundle);rna_train=modality_terms(rna_residual[train_mask],y_rna[train_mask],loglib_rna[train_mask],mu0_rna,model.rna.base.raw_theta,rna_weights);atac_train=modality_terms(atac_residual[train_mask],y_atac[train_mask],loglib_atac[train_mask],mu0_atac,model.atac.base.raw_theta,atac_weights);loss=0.5*(rna_train["objective"]+atac_train["objective"])
        if not torch.isfinite(loss):raise FloatingPointError(f"Non-finite GAT loss for {save_prefix} epoch {epoch}")
        loss.backward();torch.nn.utils.clip_grad_norm_(spatial_params,1.0);optimizer.step();rna_val,atac_val=evaluate_gat(model,x,guides,celltypes,rna_bundle,atac_bundle,y_rna,y_atac,loglib_rna,mu0_rna,loglib_atac,mu0_atac,val_mask,rna_weights,atac_weights);rna_score=float(rna_val["objective"]);atac_score=float(atac_val["objective"]);scheduler.step(0.5*(rna_score+atac_score));history.append({"epoch":epoch,"train_rna_objective":float(rna_train["objective"].detach()),"train_atac_objective":float(atac_train["objective"].detach()),"val_rna_score":rna_score,"val_atac_score":atac_score,"val_rna_nb_nll":float(rna_val["nb"]),"val_atac_nb_nll":float(atac_val["nb"]),"val_rna_raw_pearson":float(rna_val["corr"]),"val_atac_raw_pearson":float(atac_val["corr"]),"lr":optimizer.param_groups[0]["lr"]})
        if rna_score<best_rna-1e-7:best_rna=rna_score;best_rna_state=copy_state(model.rna);stale_rna=0
        else:stale_rna+=1
        if atac_score<best_atac-1e-7:best_atac=atac_score;best_atac_state=copy_state(model.atac);stale_atac=0
        else:stale_atac+=1
        if epoch==1 or epoch%25==0:print(save_prefix,"gat",epoch,float(loss.detach()),rna_score,atac_score,flush=True)
        if stale_rna>=EARLY_STOPPING_PATIENCE and stale_atac>=EARLY_STOPPING_PATIENCE:break
    model.rna.load_state_dict(best_rna_state);model.atac.load_state_dict(best_atac_state)
    for parameter in model.rna.base.parameters():parameter.requires_grad=True
    for parameter in model.atac.base.parameters():parameter.requires_grad=True
    torch.save(model.state_dict(),OUTDIR/f"{save_prefix}_gat_best.pt");pd.DataFrame(history).to_csv(OUTDIR/f"{save_prefix}_gat_history.csv",index=False)
    return model
def mixture_nb_nll(y_np:np.ndarray,mu_list:list[np.ndarray],theta_list:list[np.ndarray])->float:
    y=torch.tensor(y_np,dtype=torch.float32);log_probs=[]
    for mu_np,theta_np in zip(mu_list,theta_list):log_probs.append(nb_log_prob_from_mu(y,torch.tensor(mu_np,dtype=torch.float32),torch.tensor(theta_np,dtype=torch.float32)))
    return float(-(torch.logsumexp(torch.stack(log_probs,dim=0),dim=0)-math.log(len(log_probs))).mean())
def save_loss_plot(summary:pd.DataFrame,metric:str,label:str,stem:str,decimals:int)->None:
    means=summary.xs("mean",axis=1,level=1)[metric].loc[["mlp","gat"]];sds=summary.xs("std",axis=1,level=1)[metric].loc[["mlp","gat"]];fig,ax=plt.subplots(figsize=(3.2,3.2));x=np.array([0.0,1.0]);ax.bar(x,means.to_numpy(),yerr=sds.to_numpy(),width=0.55,color=["#969696","#4C78A8"],edgecolor="black",linewidth=0.7,capsize=3);spread=max(float(means.max()-means.min()),1e-6);pad=max(0.8*spread,10**(-decimals));ax.set_ylim(float(means.min()-pad),float(means.max()+pad));ax.set_xticks(x,["MLP","GAT"]);ax.set_ylabel(label);ax.yaxis.set_major_locator(MaxNLocator(6));ax.yaxis.set_major_formatter(FormatStrFormatter(f"%.{decimals}f"));ax.spines[["top","right"]].set_visible(False);ax.grid(False);fig.tight_layout();fig.savefig(OUTDIR/f"{stem}.png",dpi=600,bbox_inches="tight");fig.savefig(OUTDIR/f"{stem}.pdf",bbox_inches="tight");plt.close(fig)
def save_sem_loss_plot(means:pd.Series,sems:pd.Series,label:str,stem:str,decimals:int)->None:
    means=means.loc[["mlp","gat"]];sems=sems.loc[["mlp","gat"]];fig,ax=plt.subplots(figsize=(3.2,3.2));x=np.array([0.0,1.0]);ax.bar(x,means.to_numpy(),yerr=sems.to_numpy(),width=0.55,color=["#969696","#4C78A8"],edgecolor="black",linewidth=0.7,capsize=3);spread=max(float(means.max()-means.min()),1e-6);pad=max(0.8*spread,10**(-decimals));ax.set_ylim(float(means.min()-pad),float(means.max()+pad));ax.set_xticks(x,["MLP","GAT"]);ax.set_ylabel(label);ax.yaxis.set_major_locator(MaxNLocator(6));ax.yaxis.set_major_formatter(FormatStrFormatter(f"%.{decimals}f"));ax.spines[["top","right"]].set_visible(False);ax.grid(False);fig.tight_layout();fig.savefig(OUTDIR/f"{stem}.png",dpi=600,bbox_inches="tight");fig.savefig(OUTDIR/f"{stem}.pdf",bbox_inches="tight");plt.close(fig)
def paired_sem_by_model(checked:pd.DataFrame)->pd.DataFrame:
    if checked.duplicated(["test_eb","model"]).any():raise ValueError("Each test EB must have exactly one row per model for paired SEMs")
    sem_by_metric={}
    for metric in SUMMARY_METRICS:
        paired=checked.pivot(index="test_eb",columns="model",values=metric)
        if set(paired.columns)!={"mlp","gat"}:raise ValueError(f"Paired SEM requires both MLP and GAT values for {metric}")
        paired=paired.loc[:,["mlp","gat"]].sort_index()
        if paired.isnull().any().any():raise ValueError(f"Missing paired held-out EB values for {metric}")
        diffs=paired["mlp"]-paired["gat"]
        sem=float(diffs.std(ddof=1)/math.sqrt(len(diffs))) if len(diffs)>=2 else float("nan")
        sem_by_metric[metric]=pd.Series({"mlp":sem,"gat":sem})
    paired_sem=pd.DataFrame(sem_by_metric)
    paired_sem.index.name="model"
    return paired_sem
def summarize_ensemble_results(res:pd.DataFrame)->pd.DataFrame:
    required={"run","model","validation_eb","test_eb",*SUMMARY_METRICS};missing=sorted(required-set(res.columns))
    if missing:raise KeyError(f"Missing ensemble columns: {missing}")
    for model_name in ["mlp","gat"]:
        subset=res[res["model"]==model_name]
        if len(subset)!=N_HELDOUT_EBS or subset["test_eb"].nunique()!=N_HELDOUT_EBS:raise ValueError(f"Expected {N_HELDOUT_EBS} held-out EB rows for {model_name}, found {len(subset)}")
    if set(res.loc[res["model"]=="mlp","test_eb"])!=set(res.loc[res["model"]=="gat","test_eb"]):raise ValueError("MLP and GAT test EBs differ")
    checked=res.copy();checked[SUMMARY_METRICS]=checked[SUMMARY_METRICS].apply(pd.to_numeric,errors="raise");summary=checked.groupby("model")[SUMMARY_METRICS].agg(["mean","std"]);paired_sem=paired_sem_by_model(checked);summary.to_csv(OUTDIR/f"model_summary_across_{N_HELDOUT_EBS}_heldout_EBs_3seed_ensembles.csv");paired_sem.to_csv(OUTDIR/f"model_SEM_across_{N_HELDOUT_EBS}_heldout_EBs_3seed_ensembles.csv");save_loss_plot(summary,"test_joint_nb_nll","Held-out joint NB NLL","joint_NB_loss_spatial_residual_GAT_vs_MLP",4);save_loss_plot(summary,"test_rna_nb_nll","Held-out RNA NB NLL","RNA_NB_loss_spatial_residual_GAT_vs_MLP",4);save_loss_plot(summary,"test_atac_nb_nll","Held-out ATAC NB NLL","ATAC_NB_loss_spatial_residual_GAT_vs_MLP",6);means=summary.xs("mean",axis=1,level=1);save_sem_loss_plot(means["test_joint_nb_nll"],paired_sem["test_joint_nb_nll"],"Held-out joint NB NLL","joint_NB_loss_spatial_residual_GAT_vs_MLP_SEM",4);save_sem_loss_plot(means["test_rna_nb_nll"],paired_sem["test_rna_nb_nll"],"Held-out RNA NB NLL","RNA_NB_loss_spatial_residual_GAT_vs_MLP_SEM",4);save_sem_loss_plot(means["test_atac_nb_nll"],paired_sem["test_atac_nb_nll"],"Held-out ATAC NB NLL","ATAC_NB_loss_spatial_residual_GAT_vs_MLP_SEM",6);return summary
def load_csv(path:Path)->pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()
def main(preflight_only:bool=False,plots_only:bool=False)->None:
    if plots_only:
        ensemble_path=OUTDIR/"per_eb_ensemble_metrics.csv"
        if not ensemble_path.exists():raise FileNotFoundError(ensemble_path)
        ensemble_results=pd.read_csv(ensemble_path);ensemble_results=ensemble_results[ensemble_results["model_version"]==MODEL_VERSION].copy();summary=summarize_ensemble_results(ensemble_results);print(f"\nPER-EB THREE-SEED ENSEMBLE METRICS ({N_HELDOUT_EBS} held-out EBs)");print(ensemble_results.sort_values(["run","model"]).to_string(index=False));print("\nSUMMARY");print(summary);print("\nOutputs:",OUTDIR.resolve());return
    print("Device:",DEVICE,flush=True);print("Torch threads:",torch.get_num_threads(),flush=True)
    if not DATA_PATH.exists():raise FileNotFoundError(DATA_PATH)
    mu.set_options(pull_on_update=False);mdata=mu.read_h5mu(DATA_PATH)
    if "rna" not in mdata.mod or "atac" not in mdata.mod:raise KeyError("MuData must contain RNA and ATAC modalities")
    rna=mdata.mod["rna"];atac=mdata.mod["atac"];rna_names=np.asarray(rna.obs_names.astype(str));atac_names=np.asarray(atac.obs_names.astype(str));main_names=np.asarray(mdata.obs_names.astype(str))
    if not np.array_equal(rna_names,atac_names) or not np.array_equal(rna_names,main_names):raise ValueError("RNA, ATAC, and MuData cells are not in identical order")
    required_obs=[EB_KEY,CELL_TYPE_KEY,GUIDE_SOURCE_COL];missing=[column for column in required_obs if column not in mdata.obs.columns]
    if missing:raise KeyError(f"Missing MuData obs columns: {missing}")
    if SPATIAL_KEY not in rna.obsm:raise KeyError(f"Missing RNA obsm[{SPATIAL_KEY!r}]")
    rna_counts=get_counts(rna,RNA_COUNTS_LAYER,"RNA",False);atac_counts=get_counts(atac,ATAC_COUNTS_LAYER,"ATAC",True);obs=mdata.obs.copy();coords=dense32(rna.obsm[SPATIAL_KEY]);eb_all=obs[EB_KEY].astype(str).to_numpy();valid_mask=np.asarray([valid_eb_label(value) for value in eb_all],dtype=bool)
    if valid_mask.sum()==0:raise ValueError("No valid EB_* cells found")
    obs=obs.iloc[np.flatnonzero(valid_mask)].copy();coords=coords[valid_mask];rna_counts=rna_counts[valid_mask];atac_counts=atac_counts[valid_mask];eb=obs[EB_KEY].astype(str).to_numpy();guides_np,guide_names=build_guides(obs);celltype_df=pd.get_dummies(obs[CELL_TYPE_KEY].astype(str),dtype=np.float32);celltypes_np=celltype_df.to_numpy(np.float32);celltype_names=list(map(str,celltype_df.columns));splits=choose_splits(eb,guides_np,celltypes_np)
    for train,val,test,_,_ in splits:check_split_feature_coverage(guides_np,guide_names,train,val,test,"Guide");check_split_feature_coverage(celltypes_np,celltype_names,train,val,test,"Cell type")
    pd.DataFrame([{"run":i+1,"validation_eb":split[3],"test_eb":split[4],"n_train_cells":int(split[0].sum()),"n_validation_cells":int(split[1].sum()),"n_test_cells":int(split[2].sum())} for i,split in enumerate(splits)]).to_csv(OUTDIR/"split_manifest.csv",index=False)
    config={"model_version":MODEL_VERSION,"data_path":str(DATA_PATH),"output_directory":str(OUTDIR),"inputs_for_both_models":["guide_binary","own_cell_type_onehot"],"gat_specific_design":["parallel_RNA_and_ATAC_one_layer_GATv2_branches","receiver_aware_sender_guide_sender_celltype_receiver_celltype_messages","distance_weighted_neighbor_context","zero_initialized_spatial_residual","MLP_warm_start","frozen_nonspatial_base_during_spatial_training"],"rna_feature_selection":"training_only_log_normalized_variance","atac_feature_selection":"training_only_Poisson_deviance","n_heldout_ebs":N_HELDOUT_EBS,"anchor_test_ebs":ANCHOR_TEST_EBS,"anchor_validation_ebs":ANCHOR_VALIDATION_EBS,"eb_selection_seed":EB_SELECTION_SEED,"selected_test_ebs":[split[4] for split in splits],"selected_validation_ebs":[split[3] for split in splits],"final_seeds":FINAL_SEEDS,"rna_graph_candidates":RNA_GRAPH_CANDIDATES,"atac_graph_candidates":ATAC_GRAPH_CANDIDATES,"auxiliary_weights":{"log1p_mse":LOG1P_MSE_WEIGHT,"raw_correlation":RAW_CORR_WEIGHT,"raw_nmse":RAW_NMSE_WEIGHT},"loss_weight_candidates":LOSS_WEIGHT_CANDIDATES}
    (OUTDIR/"configuration.json").write_text(json.dumps(config,indent=2));pd.DataFrame({"guide":guide_names}).to_csv(OUTDIR/"guide_names.csv",index=False);pd.DataFrame({"cell_type":celltype_names}).to_csv(OUTDIR/"cell_type_names.csv",index=False)
    x_np=np.concatenate([guides_np,celltypes_np],axis=1).astype(np.float32);x=torch.tensor(x_np,dtype=torch.float32,device=DEVICE);guides=torch.tensor(guides_np,dtype=torch.float32,device=DEVICE);celltypes=torch.tensor(celltypes_np,dtype=torch.float32,device=DEVICE);rna_log=log_normalize_counts(rna_counts);graph_cache={}
    def graph_bundle(k:int,decay:float):
        key=(int(k),float(decay))
        if key not in graph_cache:
            edge_cpu,dist_cpu=build_knn_graph(coords,eb,k);context_np=weighted_neighbor_context(guides_np,celltypes_np,edge_cpu,dist_cpu,decay);graph_cache[key]=(edge_cpu.to(DEVICE),dist_cpu.to(DEVICE),torch.tensor(context_np,dtype=torch.float32,device=DEVICE))
        return graph_cache[key]
    if preflight_only:
        for run_idx,(train_np,_,_,val_eb,test_eb) in enumerate(splits,1):
            rna_idx,_=select_rna_hvgs(rna_log,train_np,N_RNA_HVGS);atac_idx,_=select_atac_deviance_features(atac_counts,train_np,N_ATAC_HVFS);print(f"Preflight run {run_idx}: validation={val_eb}, test={test_eb}, RNA targets={len(rna_idx)}, ATAC targets={len(atac_idx)}",flush=True)
        rna_k,rna_decay=RNA_GRAPH_CANDIDATES[0];atac_k,atac_decay=ATAC_GRAPH_CANDIDATES[0];rna_bundle=graph_bundle(rna_k,rna_decay);atac_bundle=graph_bundle(atac_k,atac_decay);context_input_dim=guides_np.shape[1]+celltypes_np.shape[1]+3;seed_all(123);mlp_test=MultiomicMLP(x.shape[1],guides_np.shape[1],celltypes_np.shape[1],10,10).to(DEVICE);gat_test=MultiomicSpatialResidualGAT(x.shape[1],guides_np.shape[1],celltypes_np.shape[1],10,10,context_input_dim,rna_decay,atac_decay).to(DEVICE);gat_test.warm_start_from_mlp(mlp_test)
        with torch.no_grad():
            mr,ma=mlp_test(x,guides,celltypes);gr,ga=gat_test(x,guides,celltypes,rna_bundle,atac_bundle)
        if mr.shape!=(len(obs),10) or ma.shape!=(len(obs),10) or gr.shape!=(len(obs),10) or ga.shape!=(len(obs),10):raise RuntimeError("Preflight model output shapes are invalid")
        if bool(gat_test.rna.gat.add_self_loops) or bool(gat_test.atac.gat.add_self_loops):raise RuntimeError("Preflight detected GAT self-loops")
        print("PREFLIGHT PASSED: data, splits, feature selection, graphs, GATv2 tuple input, and model forward shapes are valid",flush=True);return
    per_seed_path=OUTDIR/"per_seed_test_metrics.csv";ensemble_path=OUTDIR/"per_eb_ensemble_metrics.csv";per_seed_rows=load_csv(per_seed_path).to_dict("records");ensemble_rows=load_csv(ensemble_path).to_dict("records")
    context_input_dim=guides_np.shape[1]+celltypes_np.shape[1]+3
    for run_idx,(train_np,val_np,test_np,val_eb,test_eb) in enumerate(splits,1):
        print(f"\nRUN {run_idx} validation={val_eb} test={test_eb}",flush=True);rna_idx,rna_var=select_rna_hvgs(rna_log,train_np,N_RNA_HVGS);atac_idx,atac_dev=select_atac_deviance_features(atac_counts,train_np,N_ATAC_HVFS);y_rna_np=rna_counts[:,rna_idx].toarray().astype(np.float32);y_atac_np=atac_counts[:,atac_idx].toarray().astype(np.float32);y_rna=torch.tensor(y_rna_np,dtype=torch.float32,device=DEVICE);y_atac=torch.tensor(y_atac_np,dtype=torch.float32,device=DEVICE);rna_library=(np.asarray(rna_counts.sum(1)).ravel()-y_rna_np.sum(1)).astype(np.float32);atac_library=(np.asarray(atac_counts.sum(1)).ravel()-y_atac_np.sum(1)).astype(np.float32)
        if np.any(rna_library<=0) or np.any(atac_library<=0):raise ValueError("Non-positive non-target library size")
        loglib_rna_np=np.log(rna_library).astype(np.float32);loglib_atac_np=np.log(atac_library).astype(np.float32);mu0_rna_np=np.log(np.clip(y_rna_np[train_np].sum(0,dtype=np.float64)/rna_library[train_np].sum(dtype=np.float64),1e-12,None)).astype(np.float32);mu0_atac_np=np.log(np.clip(y_atac_np[train_np].sum(0,dtype=np.float64)/atac_library[train_np].sum(dtype=np.float64),1e-12,None)).astype(np.float32);loglib_rna=torch.tensor(loglib_rna_np,dtype=torch.float32,device=DEVICE);loglib_atac=torch.tensor(loglib_atac_np,dtype=torch.float32,device=DEVICE);mu0_rna=torch.tensor(mu0_rna_np,dtype=torch.float32,device=DEVICE);mu0_atac=torch.tensor(mu0_atac_np,dtype=torch.float32,device=DEVICE);train=torch.tensor(train_np,dtype=torch.bool,device=DEVICE);val=torch.tensor(val_np,dtype=torch.bool,device=DEVICE);test=torch.tensor(test_np,dtype=torch.bool,device=DEVICE)
        pd.DataFrame({"gene":np.asarray(rna.var_names.astype(str))[rna_idx],"feature_index":rna_idx,"training_log_variance":rna_var}).to_csv(OUTDIR/f"run{run_idx}_rna_hvgs.csv",index=False);pd.DataFrame({"atac_feature":np.asarray(atac.var_names.astype(str))[atac_idx],"feature_index":atac_idx,"training_poisson_deviance":atac_dev}).to_csv(OUTDIR/f"run{run_idx}_atac_deviance_features.csv",index=False);np.savez_compressed(OUTDIR/f"run{run_idx}_offsets_and_targets.npz",rna_target_idx=rna_idx,atac_target_idx=atac_idx,rna_mu0=mu0_rna_np,atac_mu0=mu0_atac_np,rna_log_library=loglib_rna_np,atac_log_library=loglib_atac_np);np.save(OUTDIR/f"run{run_idx}_heldout_true_rna_counts.npy",y_rna_np[test_np]);np.save(OUTDIR/f"run{run_idx}_heldout_true_atac_counts.npy",y_atac_np[test_np])
        tune_seed=TUNE_SEED_BASE+run_idx;tune_mlp_path=OUTDIR/f"run{run_idx}_tune_mlp_best.pt";seed_all(tune_seed);tune_mlp=MultiomicMLP(x.shape[1],guides_np.shape[1],celltypes_np.shape[1],len(rna_idx),len(atac_idx))
        if tune_mlp_path.exists():tune_mlp.load_state_dict(torch.load(tune_mlp_path,map_location=DEVICE));tune_mlp=tune_mlp.to(DEVICE)
        else:tune_mlp,_,_=train_mlp(tune_mlp,x,guides,celltypes,y_rna,y_atac,loglib_rna,mu0_rna,loglib_atac,mu0_atac,train,val,TUNE_MLP_MAX_EPOCHS,f"run{run_idx}_tune");torch.save(tune_mlp.state_dict(),tune_mlp_path)
        tuning_path=OUTDIR/f"run{run_idx}_modality_graph_tuning.csv";tuning_rows=load_csv(tuning_path).to_dict("records")
        for modality,candidates in [("rna",RNA_GRAPH_CANDIDATES),("atac",ATAC_GRAPH_CANDIDATES)]:
            for k,decay in candidates:
                if any(str(row.get("model_version"))==MODEL_VERSION and str(row.get("modality"))==modality and int(row.get("k_neighbors"))==k and abs(float(row.get("distance_decay"))-decay)<1e-9 for row in tuning_rows):continue
                seed_all(tune_seed);branch=SpatialResidualModality(x.shape[1],guides_np.shape[1],celltypes_np.shape[1],len(rna_idx) if modality=="rna" else len(atac_idx),context_input_dim,decay);branch.base.load_state_dict(copy.deepcopy(tune_mlp.rna.state_dict() if modality=="rna" else tune_mlp.atac.state_dict()));bundle=graph_bundle(k,decay);target=y_rna if modality=="rna" else y_atac;loglib=loglib_rna if modality=="rna" else loglib_atac;mu0=mu0_rna if modality=="rna" else mu0_atac;branch,best=train_single_spatial(branch,x,guides,celltypes,bundle,target,loglib,mu0,train,val,TUNE_SPATIAL_MAX_EPOCHS);tuning_rows.append({"model_version":MODEL_VERSION,"run":run_idx,"validation_eb":val_eb,"test_eb":test_eb,"modality":modality,"k_neighbors":k,"distance_decay":decay,**best});pd.DataFrame(tuning_rows).to_csv(tuning_path,index=False);del branch
                if torch.cuda.is_available():torch.cuda.empty_cache()
        tuning_df=pd.DataFrame(tuning_rows);selected_graphs={}
        for modality in ["rna","atac"]:
            subset=tuning_df[(tuning_df["model_version"]==MODEL_VERSION)&(tuning_df["modality"]==modality)].copy();best_row=subset.sort_values(["best_val_score","best_val_nb_nll","k_neighbors","distance_decay"]).iloc[0];selected_graphs[modality]=(int(best_row["k_neighbors"]),float(best_row["distance_decay"]));print(f"Selected {modality.upper()} graph: k={selected_graphs[modality][0]}, decay={selected_graphs[modality][1]}, validation score={float(best_row['best_val_score']):.6f}",flush=True)
        pd.DataFrame([{"run":run_idx,"modality":modality,"k_neighbors":values[0],"distance_decay":values[1]} for modality,values in selected_graphs.items()]).to_csv(OUTDIR/f"run{run_idx}_selected_modality_graphs.csv",index=False);rna_bundle=graph_bundle(*selected_graphs["rna"]);atac_bundle=graph_bundle(*selected_graphs["atac"])
        loss_tuning_path=OUTDIR/f"run{run_idx}_modality_loss_weight_tuning.csv";loss_tuning_rows=load_csv(loss_tuning_path).to_dict("records")
        for modality in ["rna","atac"]:
            bundle=rna_bundle if modality=="rna" else atac_bundle;target=y_rna if modality=="rna" else y_atac;loglib=loglib_rna if modality=="rna" else loglib_atac;mu0=mu0_rna if modality=="rna" else mu0_atac;base_state=tune_mlp.rna.state_dict() if modality=="rna" else tune_mlp.atac.state_dict();n_targets=len(rna_idx) if modality=="rna" else len(atac_idx);decay=selected_graphs[modality][1]
            for logw,corrw,nmsew in LOSS_WEIGHT_CANDIDATES:
                if any(str(row.get("model_version"))==MODEL_VERSION and str(row.get("modality"))==modality and abs(float(row.get("log1p_mse_weight"))-logw)<1e-12 and abs(float(row.get("corr_weight"))-corrw)<1e-12 and abs(float(row.get("nmse_weight"))-nmsew)<1e-12 for row in loss_tuning_rows):continue
                seed_all(tune_seed);branch=SpatialResidualModality(x.shape[1],guides_np.shape[1],celltypes_np.shape[1],n_targets,context_input_dim,decay);branch.base.load_state_dict(copy.deepcopy(base_state));branch,best=train_single_spatial(branch,x,guides,celltypes,bundle,target,loglib,mu0,train,val,TUNE_SPATIAL_MAX_EPOCHS,(logw,corrw,nmsew));loss_tuning_rows.append({"model_version":MODEL_VERSION,"run":run_idx,"validation_eb":val_eb,"test_eb":test_eb,"modality":modality,"log1p_mse_weight":logw,"corr_weight":corrw,"nmse_weight":nmsew,**best});pd.DataFrame(loss_tuning_rows).to_csv(loss_tuning_path,index=False);del branch
                if torch.cuda.is_available():torch.cuda.empty_cache()
        loss_tuning_df=pd.DataFrame(loss_tuning_rows);selected_weights={}
        for modality in ["rna","atac"]:
            subset=loss_tuning_df[(loss_tuning_df["model_version"]==MODEL_VERSION)&(loss_tuning_df["modality"]==modality)].copy();subset["rank_sum"]=subset["best_val_nb_nll"].rank(method="min",ascending=True)+subset["best_val_raw_pearson"].rank(method="min",ascending=False)+subset["best_val_raw_rmse"].rank(method="min",ascending=True);best_row=subset.sort_values(["rank_sum","best_val_nb_nll","best_val_raw_rmse"],ascending=[True,True,True]).iloc[0];selected_weights[modality]=(float(best_row["log1p_mse_weight"]),float(best_row["corr_weight"]),float(best_row["nmse_weight"]));print(f"Selected {modality.upper()} loss weights: logMSE={selected_weights[modality][0]}, Pearson={selected_weights[modality][1]}, NMSE={selected_weights[modality][2]}",flush=True)
        pd.DataFrame([{"run":run_idx,"modality":m,"log1p_mse_weight":w[0],"corr_weight":w[1],"nmse_weight":w[2]} for m,w in selected_weights.items()]).to_csv(OUTDIR/f"run{run_idx}_selected_loss_weights.csv",index=False);seed_predictions={"mlp":{"rna":[],"atac":[],"rna_theta":[],"atac_theta":[]},"gat":{"rna":[],"atac":[],"rna_theta":[],"atac_theta":[]}}
        for seed in FINAL_SEEDS:
            prefix=f"run{run_idx}_seed{seed}";mlp_checkpoint=OUTDIR/f"{prefix}_mlp_best.pt";seed_all(seed);mlp=MultiomicMLP(x.shape[1],guides_np.shape[1],celltypes_np.shape[1],len(rna_idx),len(atac_idx))
            if mlp_checkpoint.exists():mlp.load_state_dict(torch.load(mlp_checkpoint,map_location=DEVICE));mlp=mlp.to(DEVICE)
            else:mlp,_,_=train_mlp(mlp,x,guides,celltypes,y_rna,y_atac,loglib_rna,mu0_rna,loglib_atac,mu0_atac,train,val,FINAL_MLP_MAX_EPOCHS,prefix,selected_weights["rna"],selected_weights["atac"])
            for name in ["mlp","gat"]:
                model=None;pred_rna_path=OUTDIR/f"{prefix}_{name}_heldout_rna_pred.npy";pred_atac_path=OUTDIR/f"{prefix}_{name}_heldout_atac_pred.npy";theta_rna_path=OUTDIR/f"{prefix}_{name}_rna_theta.npy";theta_atac_path=OUTDIR/f"{prefix}_{name}_atac_theta.npy";existing=next((row for row in per_seed_rows if str(row.get("model_version"))==MODEL_VERSION and int(row["run"])==run_idx and int(row["seed"])==seed and str(row["model"])==name),None)
                if existing is not None and pred_rna_path.exists() and pred_atac_path.exists() and theta_rna_path.exists() and theta_atac_path.exists():rna_mu=np.load(pred_rna_path);atac_mu=np.load(pred_atac_path);rna_theta=np.load(theta_rna_path);atac_theta=np.load(theta_atac_path)
                else:
                    if name=="mlp":model=mlp;model.eval();
                    else:
                        seed_all(seed);model=MultiomicSpatialResidualGAT(x.shape[1],guides_np.shape[1],celltypes_np.shape[1],len(rna_idx),len(atac_idx),context_input_dim,selected_graphs["rna"][1],selected_graphs["atac"][1]);model.warm_start_from_mlp(mlp);model=train_gat(model,x,guides,celltypes,rna_bundle,atac_bundle,y_rna,y_atac,loglib_rna,mu0_rna,loglib_atac,mu0_atac,train,val,FINAL_SPATIAL_MAX_EPOCHS,prefix,selected_weights["rna"],selected_weights["atac"]);model.eval()
                    with torch.no_grad():
                        if name=="mlp":rna_residual,atac_residual=model(x,guides,celltypes);rna_theta_tensor=model.rna.raw_theta;atac_theta_tensor=model.atac.raw_theta
                        else:rna_residual,atac_residual=model(x,guides,celltypes,rna_bundle,atac_bundle);rna_theta_tensor=model.rna.base.raw_theta;atac_theta_tensor=model.atac.base.raw_theta
                        rna_loss=float(nb_nll(rna_residual[test],y_rna[test],loglib_rna[test],mu0_rna,rna_theta_tensor));atac_loss=float(nb_nll(atac_residual[test],y_atac[test],loglib_atac[test],mu0_atac,atac_theta_tensor));rna_mu=make_mu(rna_residual[test],loglib_rna[test],mu0_rna).cpu().numpy();atac_mu=make_mu(atac_residual[test],loglib_atac[test],mu0_atac).cpu().numpy();rna_theta=positive_theta(rna_theta_tensor).cpu().numpy();atac_theta=positive_theta(atac_theta_tensor).cpu().numpy()
                    row={"model_version":MODEL_VERSION,"run":run_idx,"seed":seed,"model":name,"validation_eb":val_eb,"test_eb":test_eb,"rna_k_neighbors":selected_graphs["rna"][0],"rna_distance_decay":selected_graphs["rna"][1],"atac_k_neighbors":selected_graphs["atac"][0],"atac_distance_decay":selected_graphs["atac"][1],"rna_log1p_mse_weight":selected_weights["rna"][0],"rna_corr_weight":selected_weights["rna"][1],"rna_nmse_weight":selected_weights["rna"][2],"atac_log1p_mse_weight":selected_weights["atac"][0],"atac_corr_weight":selected_weights["atac"][1],"atac_nmse_weight":selected_weights["atac"][2],"test_joint_nb_nll":0.5*(rna_loss+atac_loss),"test_rna_nb_nll":rna_loss,"test_atac_nb_nll":atac_loss,"n_parameters":sum(parameter.numel() for parameter in model.parameters())};row.update(metrics(y_rna_np[test_np],rna_mu,"rna"));row.update(metrics(y_atac_np[test_np],atac_mu,"atac"));per_seed_rows=[r for r in per_seed_rows if not (str(r.get("model_version"))==MODEL_VERSION and int(r["run"])==run_idx and int(r["seed"])==seed and str(r["model"])==name)];per_seed_rows.append(row);pd.DataFrame(per_seed_rows).sort_values(["run","model","seed"]).to_csv(per_seed_path,index=False);np.save(pred_rna_path,rna_mu.astype(np.float32));np.save(pred_atac_path,atac_mu.astype(np.float32));np.save(theta_rna_path,rna_theta.astype(np.float32));np.save(theta_atac_path,atac_theta.astype(np.float32))
                seed_predictions[name]["rna"].append(rna_mu);seed_predictions[name]["atac"].append(atac_mu);seed_predictions[name]["rna_theta"].append(rna_theta);seed_predictions[name]["atac_theta"].append(atac_theta)
                if model is not None and name=="gat":del model
                if torch.cuda.is_available():torch.cuda.empty_cache()
            del mlp
        for name in ["mlp","gat"]:
            ensemble_rna=np.mean(np.stack(seed_predictions[name]["rna"],axis=0),axis=0);ensemble_atac=np.mean(np.stack(seed_predictions[name]["atac"],axis=0),axis=0);rna_loss=mixture_nb_nll(y_rna_np[test_np],seed_predictions[name]["rna"],seed_predictions[name]["rna_theta"]);atac_loss=mixture_nb_nll(y_atac_np[test_np],seed_predictions[name]["atac"],seed_predictions[name]["atac_theta"]);row={"model_version":MODEL_VERSION,"run":run_idx,"model":name,"validation_eb":val_eb,"test_eb":test_eb,"rna_k_neighbors":selected_graphs["rna"][0],"rna_distance_decay":selected_graphs["rna"][1],"atac_k_neighbors":selected_graphs["atac"][0],"atac_distance_decay":selected_graphs["atac"][1],"n_seeds":len(FINAL_SEEDS),"test_joint_nb_nll":0.5*(rna_loss+atac_loss),"test_rna_nb_nll":rna_loss,"test_atac_nb_nll":atac_loss};row.update(metrics(y_rna_np[test_np],ensemble_rna,"rna"));row.update(metrics(y_atac_np[test_np],ensemble_atac,"atac"));ensemble_rows=[r for r in ensemble_rows if not (str(r.get("model_version"))==MODEL_VERSION and int(r["run"])==run_idx and str(r["model"])==name)];ensemble_rows.append(row);pd.DataFrame(ensemble_rows).sort_values(["run","model"]).to_csv(ensemble_path,index=False);np.save(OUTDIR/f"run{run_idx}_{name}_3seed_ensemble_rna_pred.npy",ensemble_rna.astype(np.float32));np.save(OUTDIR/f"run{run_idx}_{name}_3seed_ensemble_atac_pred.npy",ensemble_atac.astype(np.float32))
    ensemble_results=pd.DataFrame(ensemble_rows);ensemble_results=ensemble_results[ensemble_results["model_version"]==MODEL_VERSION].copy();summary=summarize_ensemble_results(ensemble_results);print(f"\nPER-EB THREE-SEED ENSEMBLE METRICS ({N_HELDOUT_EBS} held-out EBs)");print(ensemble_results.sort_values(["run","model"]).to_string(index=False));print("\nSUMMARY");print(summary);print("\nOutputs:",OUTDIR.resolve())

main(False,False)

# RNA mean baseline, MLP, and GAT
from pathlib import Path
import math
import numpy as np
import pandas as pd
import mudata as mu
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter,MaxNLocator
from scipy.special import gammaln,logsumexp
RESULTS_DIR=Path("multiome_joint_RNA_ATAC_MODALITY_SPECIFIC_SPATIAL_RESIDUAL_TUNE_LOSS_WEIGHTS_10EB")
DATA_PATH=Path("/scratch/welchjd_root/welchjd1/javidgmh/Non_Autonomous_perturbation_project/15515_Multiome/catatac_work/motif_analysis/all_lanes_merged_CATATAC_QC_RNA_processed_EB_manual_clustering_cell_types.h5mu")
OUTDIR=RESULTS_DIR/"publication_panels"
OUTDIR.mkdir(parents=True,exist_ok=True)
EB_KEY="EB_manual_clustering"
SEEDS=[17,42,89]
COLOR_BASELINE="#D9D9D9"
COLOR_MLP="#969696"
COLOR_GAT="#4C78A8"
df=pd.read_csv(RESULTS_DIR/"per_eb_ensemble_metrics.csv")
mdata=mu.read_h5mu(DATA_PATH)
eb_all=mdata.obs[EB_KEY].astype(str).to_numpy()
valid=np.asarray([x.startswith("EB_") and x.lower() not in {"eb_","eb_nan","eb_none","eb_unassigned"} for x in eb_all])
eb=eb_all[valid]
def nb_log_prob(y,mu,theta):
    y=np.asarray(y,dtype=np.float64)
    mu=np.clip(np.asarray(mu,dtype=np.float64),1e-12,None)
    theta=np.clip(np.asarray(theta,dtype=np.float64),1e-4,None)[None,:]
    return gammaln(y+theta)-gammaln(theta)-gammaln(y+1)+theta*(np.log(theta)-np.log(theta+mu))+y*(np.log(mu)-np.log(theta+mu))
def mixture_nb_nll(y,mu_list,theta_list):
    lp=np.stack([nb_log_prob(y,mu,theta) for mu,theta in zip(mu_list,theta_list)],axis=0)
    return float(-(logsumexp(lp,axis=0)-math.log(len(lp))).mean())
def baseline_losses(modality):
    out={}
    for run in range(1,11):
        row=df[df["run"]==run].iloc[0]
        test_eb=str(row["test_eb"])
        test_mask=eb==test_eb
        z=np.load(RESULTS_DIR/f"run{run}_offsets_and_targets.npz")
        y=np.load(RESULTS_DIR/f"run{run}_heldout_true_{modality}_counts.npy").astype(np.float64)
        loglib=z[f"{modality}_log_library"].astype(np.float64)[test_mask]
        mu0=z[f"{modality}_mu0"].astype(np.float64)
        if len(loglib)!=len(y):raise ValueError(f"Run {run} {modality}: test mask={len(loglib)}, heldout={len(y)}")
        baseline_mu=np.exp(np.clip(loglib[:,None]+mu0[None,:],-12,12))
        mu_list=[baseline_mu for _ in SEEDS]
        theta_list=[np.load(RESULTS_DIR/f"run{run}_seed{seed}_mlp_{modality}_theta.npy").astype(np.float64) for seed in SEEDS]
        out[test_eb]=mixture_nb_nll(y,mu_list,theta_list)
    return pd.Series(out,name="baseline")
def make_plot(metric,modality,ylabel,filename,decimals):
    x=df.pivot(index="test_eb",columns="model",values=metric)[["mlp","gat"]].dropna()
    baseline=baseline_losses(modality)
    x=x.join(baseline,how="inner")[["baseline","mlp","gat"]]
    if len(x)!=10:raise ValueError(f"Expected 10 held-out EBs, found {len(x)}")
    means=x.mean()
    mlp_gat_sem=float((x["mlp"]-x["gat"]).std(ddof=1)/math.sqrt(len(x)))
    baseline_mlp_sem=float((x["baseline"]-x["mlp"]).std(ddof=1)/math.sqrt(len(x)))
    sems=np.array([baseline_mlp_sem,mlp_gat_sem,mlp_gat_sem])
    fig,ax=plt.subplots(figsize=(4.0,3.2))
    xpos=np.arange(3,dtype=float)
    ax.bar(xpos,means.to_numpy(),yerr=sems,width=0.55,color=[COLOR_BASELINE,COLOR_MLP,COLOR_GAT],edgecolor="black",linewidth=0.7,capsize=3,error_kw={"elinewidth":1.3,"ecolor":"black","capthick":1.3})
    low=float(np.min(means.to_numpy()-sems))
    high=float(np.max(means.to_numpy()+sems))
    spread=max(high-low,10**(-decimals))
    ax.set_ylim(low-0.20*spread,high+0.20*spread)
    ax.set_xticks(xpos,["Mean\nbaseline","MLP","GAT"])
    ax.set_ylabel(ylabel)
    ax.yaxis.set_major_locator(MaxNLocator(6))
    ax.yaxis.set_major_formatter(FormatStrFormatter(f"%.{decimals}f"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(OUTDIR/f"{filename}.png",dpi=600,bbox_inches="tight")
    fig.savefig(OUTDIR/f"{filename}.pdf",bbox_inches="tight")
    plt.close(fig)
    print(f"\n{modality.upper()}")
    print(x.to_string())
    print("\nMeans:")
    print(means.to_string())
    print(f"\nBaseline-vs-MLP paired SEM: {baseline_mlp_sem:.10f}")
    print(f"MLP-vs-GAT paired SEM: {mlp_gat_sem:.10f}")
make_plot("test_rna_nb_nll","rna","Held-out RNA NB NLL","RNA_NB_loss_mean_baseline_MLP_GAT_SEM",4)
print("\nSaved to:",OUTDIR.resolve())

# ATAC FDR-significant peak union selection

#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path('/scratch/welchjd_root/welchjd1/javidgmh/Non_Autonomous_perturbation_project/15515_Multiome/catatac_work/motif_analysis')
DIST=ROOT/'KD_distance_ATAC_TOP10_LIGAND_KDs_DATASET_LEVEL_GLOBAL_SPATIAL_SHUFFLE_60K_ONE_SIDED_15pct_FULL_NULLS/dataset_level_results/ALL_KD_dataset_level_feature_results.csv'
GP_ROOT=ROOT/'results/notebook_multimodal_atac_gp_all_conditions_50kb'
OUT=ROOT/'ATAC_FDR_SIGNIFICANT_PEAKS_KD_GP_UNION.csv'
ALPHA=0.05

def bool_mask(s:pd.Series)->pd.Series:
    if pd.api.types.is_bool_dtype(s):return s.fillna(False)
    x=s.astype(str).str.strip().str.lower()
    return x.isin({'true','1','yes','y','t'})

if not DIST.exists():raise FileNotFoundError(DIST)
if not GP_ROOT.exists():raise FileNotFoundError(GP_ROOT)

dist=pd.read_csv(DIST)
required={'KD_target','feature','fdr_positive_one_sided','fdr_negative_one_sided'}
if not required.issubset(dist.columns):raise KeyError(f'Distance results missing columns: {sorted(required-set(dist.columns))}')
if 'modality' in dist.columns:dist=dist[dist['modality'].astype(str).eq('ATAC')].copy()
dist['fdr_positive_one_sided']=pd.to_numeric(dist['fdr_positive_one_sided'],errors='coerce')
dist['fdr_negative_one_sided']=pd.to_numeric(dist['fdr_negative_one_sided'],errors='coerce')
dist['KD_best_fdr']=dist[['fdr_positive_one_sided','fdr_negative_one_sided']].min(axis=1,skipna=True)
dist_sig=dist[(dist['fdr_positive_one_sided']<ALPHA)|(dist['fdr_negative_one_sided']<ALPHA)].copy()
dist_sig['peak']=dist_sig['feature'].astype(str).str.strip()

gp_files=sorted(GP_ROOT.glob('*/atac_peaks/gp_lrt_full_unconstrained/spatial_lrt_results.tsv'))
if not gp_files:gp_files=sorted(p for p in GP_ROOT.rglob('spatial_lrt_results.tsv') if 'atac_peaks' in {x.name for x in p.parents})
if not gp_files:raise FileNotFoundError(f'No GP ATAC peak result files found under {GP_ROOT}')
gp_parts=[]
for f in gp_files:
    d=pd.read_csv(f,sep='\t')
    if not {'gene','fdr_q_value'}.issubset(d.columns):continue
    d['KD_target']=f.parents[2].name
    d['peak']=d['gene'].astype(str).str.strip()
    d['fdr_q_value']=pd.to_numeric(d['fdr_q_value'],errors='coerce')
    d=d[d['fdr_q_value']<=ALPHA].copy()
    if 'fit_success' in d.columns:d=d[bool_mask(d['fit_success'])].copy()
    gp_parts.append(d)
if not gp_parts:raise RuntimeError('GP result files were found, but no usable ATAC peak tables were loaded')
gp_sig=pd.concat(gp_parts,ignore_index=True)

kd=dist_sig.groupby('peak',as_index=False).agg(KD_best_fdr=('KD_best_fdr','min'),KD_n_significant_perturbations=('KD_target','nunique'),KD_significant_perturbations=('KD_target',lambda x:';'.join(sorted(set(map(str,x))))))
if 'rho' in dist_sig.columns:
    rho=dist_sig.groupby('peak')['rho'].apply(lambda x:float(np.nanmax(np.abs(pd.to_numeric(x,errors='coerce'))))).rename('KD_max_abs_rho').reset_index()
    kd=kd.merge(rho,on='peak',how='left')
gp=gp_sig.groupby('peak',as_index=False).agg(GP_best_fdr=('fdr_q_value','min'),GP_n_significant_perturbations=('KD_target','nunique'),GP_significant_perturbations=('KD_target',lambda x:';'.join(sorted(set(map(str,x))))))
if 'likelihood_ratio_statistic' in gp_sig.columns:
    lrt=gp_sig.groupby('peak')['likelihood_ratio_statistic'].max().rename('GP_max_LRT').reset_index()
    gp=gp.merge(lrt,on='peak',how='left')

out=kd.merge(gp,on='peak',how='outer')
out['KD_FDR_significant']=out['KD_best_fdr'].notna()
out['GP_FDR_significant']=out['GP_best_fdr'].notna()
out['significant_source']=np.select([out['KD_FDR_significant']&out['GP_FDR_significant'],out['KD_FDR_significant'],out['GP_FDR_significant']],['KD_permutation;GP','KD_permutation','GP'],default='')
out=out.sort_values(['KD_FDR_significant','GP_FDR_significant','KD_best_fdr','GP_best_fdr'],ascending=[False,False,True,True],na_position='last').reset_index(drop=True)
out.insert(0,'peak_rank',np.arange(1,len(out)+1))
out.to_csv(OUT,index=False)

n_kd=dist_sig['peak'].nunique();n_gp=gp_sig['peak'].nunique();n_both=int((out['KD_FDR_significant']&out['GP_FDR_significant']).sum());n_union=len(out)
print('KD permutation FDR-significant unique peaks:',n_kd)
print('GP FDR-significant unique peaks:',n_gp)
print('FDR-significant in both methods:',n_both)
print('TOTAL UNIQUE UNION:',n_union)
print('Saved:',OUT)
assert n_kd==64,f'Expected 64 KD FDR-significant peaks, found {n_kd}'
assert n_gp==340,f'Expected 340 GP FDR-significant peaks, found {n_gp}'
assert n_both==2,f'Expected 2 peaks significant in both methods, found {n_both}'
assert n_union==402,f'Expected 402 unique union peaks, found {n_union}'

# ATAC-only GAT on 402 FDR-significant peaks
import argparse,copy,json,math,os,random
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter,MaxNLocator
import mudata as mu
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import cKDTree
from scipy.stats import pearsonr,spearmanr
from sklearn.metrics import mean_absolute_error,mean_squared_error
try:
    from torch_geometric.nn import GATv2Conv
    from torch_geometric.utils import scatter
except ImportError as exc:
    raise ImportError("This script requires PyTorch Geometric with GATv2Conv.") from exc
MODEL_VERSION="atac_only_402_FDR_KD_GP_union_10EB_v1"
DATA_PATH=Path("/scratch/welchjd_root/welchjd1/javidgmh/Non_Autonomous_perturbation_project/15515_Multiome/catatac_work/motif_analysis/all_lanes_merged_CATATAC_QC_RNA_processed_EB_manual_clustering_cell_types.h5mu")
BASE_DIR=Path("/scratch/welchjd_root/welchjd1/javidgmh/Non_Autonomous_perturbation_project/15515_Multiome/catatac_work/motif_analysis")
ORIGINAL_RESULTS_DIR=BASE_DIR/"multiome_joint_RNA_ATAC_MODALITY_SPECIFIC_SPATIAL_RESIDUAL_TUNE_LOSS_WEIGHTS_10EB"
PEAK_FILE=BASE_DIR/"ATAC_FDR_SIGNIFICANT_PEAKS_KD_GP_UNION.csv"
OUTDIR=BASE_DIR/"ATAC_ONLY_402_FDR_PEAKS_10EB"
OUTDIR.mkdir(parents=True,exist_ok=True)
ATAC_COUNTS_LAYER="counts"
SPATIAL_KEY="X_spatial"
EB_KEY="EB_manual_clustering"
CELL_TYPE_KEY="cell_type"
GUIDE_SOURCE_COL="guides_passing_str"
N_ATAC_FEATURES=402
N_HELDOUT_EBS=10
FINAL_SEEDS=[17,42,89]
ATAC_GRAPH_CANDIDATES=[(4,25.0),(8,25.0),(12,50.0),(20,50.0),(30,100.0)]
TUNE_SEED_BASE=5000
TUNE_MLP_MAX_EPOCHS=300
TUNE_SPATIAL_MAX_EPOCHS=150
FINAL_MLP_MAX_EPOCHS=500
FINAL_SPATIAL_MAX_EPOCHS=350
EARLY_STOPPING_PATIENCE=25
LR_PATIENCE=7
MLP_LEARNING_RATE=0.003
SPATIAL_LEARNING_RATE=0.003
WEIGHT_DECAY=1e-5
DROPOUT=0.15
LOG1P_MSE_WEIGHT=0.03
RAW_CORR_WEIGHT=0.02
RAW_NMSE_WEIGHT=0.005
DEFAULT_LOSS_WEIGHTS=(LOG1P_MSE_WEIGHT,RAW_CORR_WEIGHT,RAW_NMSE_WEIGHT)
LOSS_WEIGHT_CANDIDATES=[(0.03,0.02,0.005),(0.03,0.05,0.005),(0.03,0.10,0.010),(0.05,0.05,0.010),(0.05,0.10,0.020)]
OWN_HIDDEN_DIM=128
OWN_LATENT_DIM=64
MESSAGE_HIDDEN_DIM=128
SPATIAL_DIM=128
CONTEXT_DIM=64
SPATIAL_DECODER_HIDDEN=64
EDGE_HIDDEN_DIM=16
EDGE_DIM=8
GAT_HEADS=4
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_THREADS=int(os.environ.get("SLURM_CPUS_PER_TASK",os.environ.get("OMP_NUM_THREADS","8")))
torch.set_num_threads(max(1,N_THREADS))
try:torch.set_num_interop_threads(max(1,min(4,N_THREADS)))
except RuntimeError:pass
ATAC_METRICS=["test_atac_nb_nll","atac_raw_pearson","atac_raw_spearman","atac_raw_rmse","atac_raw_mae","atac_log1p_pearson","atac_log1p_spearman","atac_log1p_rmse","atac_log1p_mae"]
def seed_all(seed:int)->None:
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
def to_csr(x)->sp.csr_matrix:
    if sp.issparse(x):return x.tocsr().astype(np.float32)
    try:x=x[:]
    except Exception:pass
    if sp.issparse(x):return x.tocsr().astype(np.float32)
    return sp.csr_matrix(np.asarray(x,dtype=np.float32))
def dense32(x)->np.ndarray:
    return x.toarray().astype(np.float32) if sp.issparse(x) else np.asarray(x,dtype=np.float32)
def validate_count_matrix(name:str,x:sp.csr_matrix,require_nonbinary:bool)->None:
    if x.ndim!=2 or x.shape[0]<1 or x.shape[1]<1:raise ValueError(f"{name} matrix has invalid shape {x.shape}")
    if x.nnz==0:raise ValueError(f"{name} matrix has no nonzero counts")
    data=np.asarray(x.data,dtype=np.float64)
    if not np.all(np.isfinite(data)):raise ValueError(f"{name} matrix contains non-finite values")
    if np.any(data<0):raise ValueError(f"{name} matrix contains negative values")
    if not np.allclose(data,np.rint(data),atol=1e-5,rtol=0):raise ValueError(f"{name} matrix is not integer-like count data")
    if require_nonbinary and float(data.max())<=1.0:raise ValueError(f"{name} matrix is binary; NB count loss requires non-binary counts")
def get_counts(adata,layer:str,name:str,require_nonbinary:bool)->sp.csr_matrix:
    counts=to_csr(adata.layers[layer] if layer in adata.layers else adata.X);validate_count_matrix(name,counts,require_nonbinary);return counts
def valid_eb_label(value:object)->bool:
    s=str(value).strip();return s.startswith("EB_") and s.lower() not in {"eb_","eb_nan","eb_none","eb_unassigned"}
def parse_guides(value)->list[str]:
    if value is None or (isinstance(value,float) and np.isnan(value)):return []
    if isinstance(value,(list,tuple,np.ndarray)):values=value
    else:
        s=str(value).strip()
        if s=="" or s.lower() in {"nan","none","unassigned","no guide","no_guide"}:return []
        values=s.split(";")
    return [str(x).strip() for x in values if str(x).strip()]
def build_guides(obs:pd.DataFrame)->tuple[np.ndarray,list[str]]:
    if GUIDE_SOURCE_COL not in obs.columns:raise KeyError(f"Missing obs[{GUIDE_SOURCE_COL!r}]")
    rows=[parse_guides(v) for v in obs[GUIDE_SOURCE_COL].astype(object).to_numpy()];names=sorted({g for row in rows for g in row})
    if not names:raise ValueError("No guides were parsed")
    lookup={g:i for i,g in enumerate(names)};matrix=np.zeros((len(rows),len(names)),dtype=np.float32)
    for i,row in enumerate(rows):
        for guide in row:matrix[i,lookup[guide]]=1.0
    return matrix,names
def log_normalize_counts(counts:sp.csr_matrix,target_sum:float=1e4)->sp.csr_matrix:
    totals=np.asarray(counts.sum(1)).ravel().astype(np.float32)
    if np.any(totals<=0):raise ValueError("Cannot normalize cells with non-positive library size")
    out=(sp.diags(target_sum/totals)@counts).tocsr().astype(np.float32);out.data=np.log1p(out.data);return out
def build_knn_graph(coords:np.ndarray,labels:np.ndarray,k:int)->tuple[torch.Tensor,torch.Tensor]:
    if k<1:raise ValueError("k must be at least 1")
    if coords.ndim!=2 or coords.shape[1]<2 or len(coords)!=len(labels):raise ValueError("Spatial coordinates and EB labels are misaligned")
    if not np.all(np.isfinite(coords[:,:2])):raise ValueError("Spatial coordinates contain non-finite values")
    sources=[];destinations=[];distances=[]
    for eb in pd.unique(labels):
        idx=np.flatnonzero(labels==eb)
        if len(idx)<2:raise ValueError(f"EB {eb} has fewer than 2 cells")
        kk=min(k,len(idx)-1);d,nn_idx=cKDTree(coords[idx,:2]).query(coords[idx,:2],k=kk+1);d=np.atleast_2d(d);nn_idx=np.atleast_2d(nn_idx)
        for local_i,global_i in enumerate(idx):
            keep=nn_idx[local_i]!=local_i;chosen_nn=nn_idx[local_i][keep][:kk];chosen_d=d[local_i][keep][:kk]
            if len(chosen_nn)!=kk:raise RuntimeError(f"Could not construct {kk} neighbors for EB {eb}, cell {global_i}")
            sources.extend([global_i]*kk);destinations.extend(idx[chosen_nn].tolist());distances.extend(np.asarray(chosen_d,dtype=np.float32).tolist())
    src=np.asarray(sources,dtype=np.int64);dst=np.asarray(destinations,dtype=np.int64);dist=np.asarray(distances,dtype=np.float32);pairs=np.concatenate([np.stack([src,dst],1),np.stack([dst,src],1)],0);pair_dist=np.concatenate([dist,dist]);order=np.lexsort((pairs[:,1],pairs[:,0]));pairs=pairs[order];pair_dist=pair_dist[order];keep=np.ones(len(pairs),dtype=bool);keep[1:]=np.any(pairs[1:]!=pairs[:-1],axis=1);pairs=pairs[keep];pair_dist=pair_dist[keep]
    if np.any(pairs[:,0]==pairs[:,1]):raise RuntimeError("Graph contains self-edges")
    if np.any(labels[pairs[:,0]]!=labels[pairs[:,1]]):raise RuntimeError("Graph contains cross-EB edges")
    return torch.tensor(pairs.T,dtype=torch.long),torch.tensor(pair_dist[:,None],dtype=torch.float32)
def weighted_neighbor_context(guides:np.ndarray,celltypes:np.ndarray,edge_index:torch.Tensor,edge_distance:torch.Tensor,decay:float)->np.ndarray:
    src,dst=edge_index;distance=edge_distance.reshape(-1).float();weights=torch.exp(-distance/float(decay))[:,None];guide_tensor=torch.tensor(guides,dtype=torch.float32);celltype_tensor=torch.tensor(celltypes,dtype=torch.float32);weighted_guides=scatter(guide_tensor[src]*weights,dst,dim=0,dim_size=len(guides),reduce="sum");weighted_celltypes=scatter(celltype_tensor[src]*weights,dst,dim=0,dim_size=len(guides),reduce="sum");weight_sum=scatter(weights,dst,dim=0,dim_size=len(guides),reduce="sum").clamp_min(1e-8);degree=scatter(torch.ones_like(weights),dst,dim=0,dim_size=len(guides),reduce="sum").clamp_min(1.0);mean_distance=scatter(distance[:,None]*weights,dst,dim=0,dim_size=len(guides),reduce="sum")/weight_sum;min_distance=scatter(distance[:,None],dst,dim=0,dim_size=len(guides),reduce="min");context=torch.cat([weighted_guides/weight_sum,weighted_celltypes/weight_sum,torch.log1p(degree),mean_distance/float(decay),min_distance/float(decay)],dim=1);out=context.numpy().astype(np.float32)
    if not np.all(np.isfinite(out)):raise ValueError("Non-finite weighted neighborhood context")
    return out
def load_atac_features(atac_var_names:np.ndarray)->tuple[np.ndarray,pd.DataFrame]:
    if not PEAK_FILE.exists():raise FileNotFoundError(f"FDR peak file not found: {PEAK_FILE}")
    table=pd.read_csv(PEAK_FILE)
    if "peak" not in table.columns:raise KeyError(f"Missing 'peak' column in {PEAK_FILE}")
    peaks=table["peak"].astype(str).str.strip().tolist()
    if len(peaks)!=N_ATAC_FEATURES:raise ValueError(f"Expected {N_ATAC_FEATURES} peaks, found {len(peaks)} in {PEAK_FILE}")
    if len(set(peaks))!=len(peaks):raise ValueError(f"Duplicate peak names in {PEAK_FILE}")
    names=np.asarray(atac_var_names,dtype=str);lookup={name:i for i,name in enumerate(names)};missing=[peak for peak in peaks if peak not in lookup]
    if missing:raise KeyError(f"{len(missing)} selected peaks are absent from ATAC var_names. Example: {missing[:10]}")
    idx=np.asarray([lookup[peak] for peak in peaks],dtype=int);table=table.copy()
    if "feature_index" in table.columns:table=table.drop(columns=["feature_index"])
    table.insert(1,"feature_index",idx)
    return idx,table
def check_split_feature_coverage(matrix:np.ndarray,names:list[str],train:np.ndarray,val:np.ndarray,test:np.ndarray,label:str)->None:
    train_present=matrix[train].sum(0)>0;val_present=matrix[val].sum(0)>0;test_present=matrix[test].sum(0)>0;missing_val=[names[i] for i in np.flatnonzero(val_present&~train_present)];missing_test=[names[i] for i in np.flatnonzero(test_present&~train_present)]
    if missing_val or missing_test:raise ValueError(f"{label} categories absent from training: validation={missing_val[:10]}, test={missing_test[:10]}")
def positive_theta(raw_theta:torch.Tensor)->torch.Tensor:
    return F.softplus(raw_theta)+1e-4
def nb_log_prob_from_mu(y:torch.Tensor,mu:torch.Tensor,theta:torch.Tensor)->torch.Tensor:
    mu=mu.clamp_min(1e-12);theta=theta.clamp_min(1e-4);logmu=torch.log(mu);logtheta=torch.log(theta);log_theta_plus_mu=torch.logaddexp(logtheta[None,:],logmu);return torch.lgamma(y+theta[None,:])-torch.lgamma(theta[None,:])-torch.lgamma(y+1)+theta[None,:]*(logtheta[None,:]-log_theta_plus_mu)+y*(logmu-log_theta_plus_mu)
def nb_nll(residual:torch.Tensor,y:torch.Tensor,log_library:torch.Tensor,mu0:torch.Tensor,raw_theta:torch.Tensor)->torch.Tensor:
    mu=torch.exp((residual+log_library[:,None]+mu0[None,:]).clamp(-12,12));return -nb_log_prob_from_mu(y,mu,positive_theta(raw_theta)).mean()
def make_mu(residual:torch.Tensor,log_library:torch.Tensor,mu0:torch.Tensor)->torch.Tensor:
    return torch.exp((residual+log_library[:,None]+mu0[None,:]).clamp(-12,12))
def torch_pearson(x:torch.Tensor,y:torch.Tensor)->torch.Tensor:
    x=x.reshape(-1);y=y.reshape(-1);xc=x-x.mean();yc=y-y.mean();den=torch.sqrt((xc.square().mean()+1e-8)*(yc.square().mean()+1e-8));return (xc*yc).mean()/den
def modality_terms(residual:torch.Tensor,y:torch.Tensor,log_library:torch.Tensor,mu0:torch.Tensor,raw_theta:torch.Tensor,loss_weights:tuple[float,float,float]=DEFAULT_LOSS_WEIGHTS)->dict[str,torch.Tensor]:
    log1p_mse_weight,corr_weight,nmse_weight=map(float,loss_weights);mu=make_mu(residual,log_library,mu0);nb=nb_nll(residual,y,log_library,mu0,raw_theta);log_mse=F.mse_loss(torch.log1p(mu),torch.log1p(y));corr=torch_pearson(mu,y);nmse=F.mse_loss(mu,y)/(torch.var(y,unbiased=False)+1e-6);objective=nb+log1p_mse_weight*log_mse+corr_weight*(1.0-corr)+nmse_weight*nmse;return {"mu":mu,"nb":nb,"log_mse":log_mse,"corr":corr,"nmse":nmse,"objective":objective}
def safe_corr(x:np.ndarray,y:np.ndarray,method:str)->float:
    if len(x)<2 or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)) or np.std(x)==0 or np.std(y)==0:return np.nan
    return float(pearsonr(x,y).statistic if method=="pearson" else spearmanr(x,y).statistic)
def metrics(y:np.ndarray,mu_pred:np.ndarray)->dict[str,float]:
    yt=np.asarray(y,dtype=np.float64).ravel();yp=np.asarray(mu_pred,dtype=np.float64).ravel();valid=np.isfinite(yt)&np.isfinite(yp)&(yt>=0)&(yp>=0);yt=yt[valid];yp=yp[valid]
    if yt.size==0:raise ValueError("No valid ATAC observations")
    xl=np.log1p(yt);yl=np.log1p(yp);return {"atac_raw_pearson":safe_corr(yt,yp,"pearson"),"atac_raw_spearman":safe_corr(yt,yp,"spearman"),"atac_raw_rmse":float(np.sqrt(mean_squared_error(yt,yp))),"atac_raw_mae":float(mean_absolute_error(yt,yp)),"atac_log1p_pearson":safe_corr(xl,yl,"pearson"),"atac_log1p_spearman":safe_corr(xl,yl,"spearman"),"atac_log1p_rmse":float(np.sqrt(mean_squared_error(xl,yl))),"atac_log1p_mae":float(mean_absolute_error(xl,yl))}
class NonSpatialModality(nn.Module):
    def __init__(self,input_dim:int,n_guides:int,n_celltypes:int,n_targets:int):
        super().__init__();self.guide_main=nn.Linear(n_guides,n_targets,bias=False);self.celltype_main=nn.Linear(n_celltypes,n_targets,bias=False);self.own_encoder=nn.Sequential(nn.Linear(input_dim,OWN_HIDDEN_DIM),nn.LayerNorm(OWN_HIDDEN_DIM),nn.ELU(),nn.Dropout(DROPOUT),nn.Linear(OWN_HIDDEN_DIM,OWN_LATENT_DIM),nn.ELU());self.own_head=nn.Linear(OWN_LATENT_DIM,n_targets,bias=False);self.raw_theta=nn.Parameter(torch.full((n_targets,),10.0))
    def forward(self,x:torch.Tensor,guides:torch.Tensor,celltypes:torch.Tensor)->torch.Tensor:
        return self.guide_main(guides)+self.celltype_main(celltypes)+self.own_head(self.own_encoder(x))
class SpatialResidualModality(nn.Module):
    def __init__(self,input_dim:int,n_guides:int,n_celltypes:int,n_targets:int,context_input_dim:int,distance_decay:float):
        super().__init__();self.distance_decay=float(distance_decay);self.base=NonSpatialModality(input_dim,n_guides,n_celltypes,n_targets);self.source_encoder=nn.Sequential(nn.Linear(input_dim,MESSAGE_HIDDEN_DIM),nn.LayerNorm(MESSAGE_HIDDEN_DIM),nn.ELU());self.receiver_encoder=nn.Sequential(nn.Linear(n_celltypes,MESSAGE_HIDDEN_DIM),nn.LayerNorm(MESSAGE_HIDDEN_DIM),nn.ELU());self.edge_encoder=nn.Sequential(nn.Linear(4,EDGE_HIDDEN_DIM),nn.ELU(),nn.Linear(EDGE_HIDDEN_DIM,EDGE_DIM));self.gat=GATv2Conv((MESSAGE_HIDDEN_DIM,MESSAGE_HIDDEN_DIM),SPATIAL_DIM//GAT_HEADS,heads=GAT_HEADS,concat=True,dropout=DROPOUT,add_self_loops=False,edge_dim=EDGE_DIM,bias=True,share_weights=False);self.context_encoder=nn.Sequential(nn.Linear(context_input_dim,CONTEXT_DIM),nn.LayerNorm(CONTEXT_DIM),nn.ELU());self.gate=nn.Sequential(nn.Linear(MESSAGE_HIDDEN_DIM+SPATIAL_DIM+CONTEXT_DIM,SPATIAL_DIM),nn.Sigmoid());self.correction_encoder=nn.Sequential(nn.Linear(MESSAGE_HIDDEN_DIM+SPATIAL_DIM+CONTEXT_DIM,SPATIAL_DECODER_HIDDEN),nn.LayerNorm(SPATIAL_DECODER_HIDDEN),nn.ELU(),nn.Dropout(DROPOUT));self.correction_head=nn.Linear(SPATIAL_DECODER_HIDDEN,n_targets,bias=False);nn.init.zeros_(self.correction_head.weight);nn.init.constant_(self.gate[0].bias,-1.5)
    def forward(self,x:torch.Tensor,guides:torch.Tensor,celltypes:torch.Tensor,edge_index:torch.Tensor,edge_distance:torch.Tensor,neighbor_context:torch.Tensor)->torch.Tensor:
        base=self.base(x,guides,celltypes);source=self.source_encoder(x);receiver=self.receiver_encoder(celltypes);scaled=edge_distance/self.distance_decay;edge_raw=torch.cat([scaled,torch.exp(-scaled),1.0/(1.0+scaled),torch.log1p(edge_distance)/math.log1p(self.distance_decay)],dim=1);edge_attr=self.edge_encoder(edge_raw);spatial=F.elu(self.gat((source,receiver),edge_index,edge_attr=edge_attr));context=self.context_encoder(neighbor_context);gate=self.gate(torch.cat([receiver,spatial,context],dim=1));correction=self.correction_head(self.correction_encoder(torch.cat([receiver,gate*spatial,context],dim=1)));return base+correction
    def spatial_parameters(self):
        for name,param in self.named_parameters():
            if not name.startswith("base."):yield param
def copy_state(module:nn.Module)->dict[str,torch.Tensor]:
    return {key:value.detach().cpu().clone() for key,value in module.state_dict().items()}
def evaluate_mlp(model:NonSpatialModality,x,guides,celltypes,y,loglib,mu0,mask,weights=DEFAULT_LOSS_WEIGHTS)->dict:
    model.eval()
    with torch.no_grad():residual=model(x,guides,celltypes);terms=modality_terms(residual[mask],y[mask],loglib[mask],mu0,model.raw_theta,weights)
    return terms
def evaluate_gat(model:SpatialResidualModality,x,guides,celltypes,bundle,y,loglib,mu0,mask,weights=DEFAULT_LOSS_WEIGHTS)->dict:
    model.eval();edge,dist,context=bundle
    with torch.no_grad():residual=model(x,guides,celltypes,edge,dist,context);terms=modality_terms(residual[mask],y[mask],loglib[mask],mu0,model.base.raw_theta,weights)
    return terms
def train_mlp(model:NonSpatialModality,x,guides,celltypes,y,loglib,mu0,train_mask,val_mask,max_epochs:int,save_prefix:str|None,weights=DEFAULT_LOSS_WEIGHTS)->tuple[NonSpatialModality,float]:
    model=model.to(DEVICE);optimizer=torch.optim.AdamW(model.parameters(),lr=MLP_LEARNING_RATE,weight_decay=WEIGHT_DECAY);scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="min",patience=LR_PATIENCE,factor=0.5,min_lr=1e-7);best=np.inf;best_state=None;stale=0;history=[]
    for epoch in range(1,max_epochs+1):
        model.train();optimizer.zero_grad(set_to_none=True);residual=model(x,guides,celltypes);train_terms=modality_terms(residual[train_mask],y[train_mask],loglib[train_mask],mu0,model.raw_theta,weights);loss=train_terms["objective"]
        if not torch.isfinite(loss):raise FloatingPointError(f"Non-finite ATAC MLP loss for {save_prefix or 'tuning'} epoch {epoch}")
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step();val_terms=evaluate_mlp(model,x,guides,celltypes,y,loglib,mu0,val_mask,weights);score=float(val_terms["objective"]);scheduler.step(score);history.append({"epoch":epoch,"train_atac_objective":float(train_terms["objective"].detach()),"val_atac_score":score,"val_atac_nb_nll":float(val_terms["nb"]),"val_atac_raw_pearson":float(val_terms["corr"]),"lr":optimizer.param_groups[0]["lr"]})
        if score<best-1e-7:best=score;best_state=copy_state(model);stale=0
        else:stale+=1
        if epoch==1 or epoch%25==0:print(save_prefix or "TUNE_ATAC_MLP",epoch,float(loss.detach()),score,flush=True)
        if stale>=EARLY_STOPPING_PATIENCE:break
    if best_state is None:raise RuntimeError("ATAC MLP did not produce a valid checkpoint")
    model.load_state_dict(best_state)
    if save_prefix is not None:torch.save(model.state_dict(),OUTDIR/f"{save_prefix}_mlp_best.pt");pd.DataFrame(history).to_csv(OUTDIR/f"{save_prefix}_mlp_history.csv",index=False)
    return model,best
def train_spatial(model:SpatialResidualModality,x,guides,celltypes,bundle,y,loglib,mu0,train_mask,val_mask,max_epochs:int,weights=DEFAULT_LOSS_WEIGHTS,save_prefix:str|None=None)->tuple[SpatialResidualModality,dict[str,float]]:
    model=model.to(DEVICE)
    for parameter in model.base.parameters():parameter.requires_grad=False
    optimizer=torch.optim.AdamW(list(model.spatial_parameters()),lr=SPATIAL_LEARNING_RATE,weight_decay=WEIGHT_DECAY);scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="min",patience=LR_PATIENCE,factor=0.5,min_lr=1e-7);edge,dist,context=bundle;model.eval()
    with torch.no_grad():initial_residual=model(x,guides,celltypes,edge,dist,context);initial=modality_terms(initial_residual[val_mask],y[val_mask],loglib[val_mask],mu0,model.base.raw_theta,weights)
    best=float(initial["objective"]);best_state=copy_state(model);best_metrics={"best_val_score":best,"best_val_nb_nll":float(initial["nb"]),"best_val_raw_pearson":float(initial["corr"]),"best_val_raw_rmse":float(torch.sqrt(F.mse_loss(initial["mu"],y[val_mask])))};stale=0;history=[{"epoch":0,"val_atac_score":best,"val_atac_nb_nll":float(initial["nb"]),"val_atac_raw_pearson":float(initial["corr"]),"lr":SPATIAL_LEARNING_RATE}]
    for epoch in range(1,max_epochs+1):
        model.train();model.base.eval();optimizer.zero_grad(set_to_none=True);residual=model(x,guides,celltypes,edge,dist,context);train_terms=modality_terms(residual[train_mask],y[train_mask],loglib[train_mask],mu0,model.base.raw_theta,weights);loss=train_terms["objective"]
        if not torch.isfinite(loss):raise FloatingPointError(f"Non-finite ATAC spatial loss at epoch {epoch}")
        loss.backward();torch.nn.utils.clip_grad_norm_(list(model.spatial_parameters()),1.0);optimizer.step();val_terms=evaluate_gat(model,x,guides,celltypes,bundle,y,loglib,mu0,val_mask,weights);score=float(val_terms["objective"]);scheduler.step(score);history.append({"epoch":epoch,"train_atac_objective":float(train_terms["objective"].detach()),"val_atac_score":score,"val_atac_nb_nll":float(val_terms["nb"]),"val_atac_raw_pearson":float(val_terms["corr"]),"lr":optimizer.param_groups[0]["lr"]})
        if score<best-1e-7:best=score;best_state=copy_state(model);best_metrics={"best_val_score":best,"best_val_nb_nll":float(val_terms["nb"]),"best_val_raw_pearson":float(val_terms["corr"]),"best_val_raw_rmse":float(torch.sqrt(F.mse_loss(val_terms["mu"],y[val_mask])))};stale=0
        else:stale+=1
        if epoch==1 or epoch%25==0:print(save_prefix or "TUNE_ATAC_GAT",epoch,float(loss.detach()),score,flush=True)
        if stale>=EARLY_STOPPING_PATIENCE:break
    model.load_state_dict(best_state)
    for parameter in model.base.parameters():parameter.requires_grad=True
    if save_prefix is not None:torch.save(model.state_dict(),OUTDIR/f"{save_prefix}_gat_best.pt");pd.DataFrame(history).to_csv(OUTDIR/f"{save_prefix}_gat_history.csv",index=False)
    return model,best_metrics
def mixture_nb_nll(y_np:np.ndarray,mu_list:list[np.ndarray],theta_list:list[np.ndarray])->float:
    y=torch.tensor(y_np,dtype=torch.float32);log_probs=[]
    for mu_np,theta_np in zip(mu_list,theta_list):log_probs.append(nb_log_prob_from_mu(y,torch.tensor(mu_np,dtype=torch.float32),torch.tensor(theta_np,dtype=torch.float32)))
    return float(-(torch.logsumexp(torch.stack(log_probs,dim=0),dim=0)-math.log(len(log_probs))).mean())
def load_csv(path:Path)->pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()
def summarize(res:pd.DataFrame)->pd.DataFrame:
    for model_name in ["mlp","gat"]:
        sub=res[res["model"]==model_name]
        if len(sub)!=N_HELDOUT_EBS or sub["test_eb"].nunique()!=N_HELDOUT_EBS:raise ValueError(f"Expected {N_HELDOUT_EBS} held-out EB rows for {model_name}, found {len(sub)}")
    checked=res.copy();checked[ATAC_METRICS]=checked[ATAC_METRICS].apply(pd.to_numeric,errors="raise");summary=checked.groupby("model")[ATAC_METRICS].agg(["mean","std"]);paired=checked.pivot(index="test_eb",columns="model",values="test_atac_nb_nll")[["mlp","gat"]];sem=float((paired["mlp"]-paired["gat"]).std(ddof=1)/math.sqrt(len(paired)));pd.DataFrame({"model":["mlp","gat"],"paired_sem":[sem,sem]}).to_csv(OUTDIR/"ATAC_NB_loss_paired_SEM.csv",index=False);summary.to_csv(OUTDIR/"ATAC_model_summary_across_10_heldout_EBs_3seed_ensembles.csv")
    means=summary.xs("mean",axis=1,level=1)["test_atac_nb_nll"].loc[["mlp","gat"]];fig,ax=plt.subplots(figsize=(3.2,3.2));x=np.array([0.0,1.0]);ax.bar(x,means.to_numpy(),yerr=[sem,sem],width=0.55,color=["#969696","#4C78A8"],edgecolor="black",linewidth=0.7,capsize=3);spread=max(float(means.max()-means.min()),1e-6);pad=max(0.8*spread,1e-6);ax.set_ylim(float(means.min()-pad),float(means.max()+pad));ax.set_xticks(x,["MLP","GAT"]);ax.set_ylabel("Held-out ATAC NB NLL");ax.yaxis.set_major_locator(MaxNLocator(6));ax.yaxis.set_major_formatter(FormatStrFormatter("%.6f"));ax.spines[["top","right"]].set_visible(False);ax.grid(False);fig.tight_layout();fig.savefig(OUTDIR/"ATAC_NB_loss_402_FDR_peaks_GAT_vs_MLP_SEM.png",dpi=600,bbox_inches="tight");fig.savefig(OUTDIR/"ATAC_NB_loss_402_FDR_peaks_GAT_vs_MLP_SEM.pdf",bbox_inches="tight");plt.close(fig);return summary
def main(preflight_only:bool=False,plots_only:bool=False)->None:
    ensemble_path=OUTDIR/"per_eb_ensemble_metrics.csv"
    if plots_only:
        if not ensemble_path.exists():raise FileNotFoundError(ensemble_path)
        res=pd.read_csv(ensemble_path);res=res[res["model_version"]==MODEL_VERSION].copy();summary=summarize(res);print(summary);return
    print("Device:",DEVICE,flush=True);print("Torch threads:",torch.get_num_threads(),flush=True)
    if not DATA_PATH.exists():raise FileNotFoundError(DATA_PATH)
    split_path=ORIGINAL_RESULTS_DIR/"split_manifest.csv"
    if not split_path.exists():raise FileNotFoundError(f"Original split manifest not found: {split_path}")
    original_manifest=pd.read_csv(split_path).sort_values("run")
    if len(original_manifest)!=N_HELDOUT_EBS:raise ValueError(f"Expected {N_HELDOUT_EBS} rows in original split manifest, found {len(original_manifest)}")
    mu.set_options(pull_on_update=False);mdata=mu.read_h5mu(DATA_PATH)
    if "atac" not in mdata.mod or "rna" not in mdata.mod:raise KeyError("MuData must contain ATAC plus RNA modality for the stored spatial coordinates")
    atac=mdata.mod["atac"];rna=mdata.mod["rna"];atac_names=np.asarray(atac.obs_names.astype(str));rna_names=np.asarray(rna.obs_names.astype(str));main_names=np.asarray(mdata.obs_names.astype(str))
    if not np.array_equal(atac_names,rna_names) or not np.array_equal(atac_names,main_names):raise ValueError("ATAC, RNA, and MuData cells are not in identical order")
    required_obs=[EB_KEY,CELL_TYPE_KEY,GUIDE_SOURCE_COL];missing=[c for c in required_obs if c not in mdata.obs.columns]
    if missing:raise KeyError(f"Missing MuData obs columns: {missing}")
    if SPATIAL_KEY not in rna.obsm:raise KeyError(f"Missing RNA obsm[{SPATIAL_KEY!r}]")
    atac_counts=get_counts(atac,ATAC_COUNTS_LAYER,"ATAC",True);obs=mdata.obs.copy();coords=dense32(rna.obsm[SPATIAL_KEY]);eb_all=obs[EB_KEY].astype(str).to_numpy();valid_mask=np.asarray([valid_eb_label(v) for v in eb_all],dtype=bool)
    if valid_mask.sum()==0:raise ValueError("No valid EB_* cells found")
    obs=obs.iloc[np.flatnonzero(valid_mask)].copy();coords=coords[valid_mask];atac_counts=atac_counts[valid_mask];eb=obs[EB_KEY].astype(str).to_numpy();guides_np,guide_names=build_guides(obs);celltype_df=pd.get_dummies(obs[CELL_TYPE_KEY].astype(str),dtype=np.float32);celltypes_np=celltype_df.to_numpy(np.float32);celltype_names=list(map(str,celltype_df.columns));splits=[]
    for row in original_manifest.itertuples(index=False):
        val_eb=str(row.validation_eb);test_eb=str(row.test_eb);train=(eb!=test_eb)&(eb!=val_eb);val=eb==val_eb;test=eb==test_eb
        if not train.any() or not val.any() or not test.any():raise ValueError(f"Missing cells for original split run {int(row.run)}: val={val_eb}, test={test_eb}")
        if int(test.sum())!=int(row.n_test_cells) or int(val.sum())!=int(row.n_validation_cells):raise ValueError(f"Cell counts do not reproduce original split run {int(row.run)}")
        check_split_feature_coverage(guides_np,guide_names,train,val,test,"Guide");check_split_feature_coverage(celltypes_np,celltype_names,train,val,test,"Cell type");splits.append((train,val,test,val_eb,test_eb))
    pd.DataFrame([{"run":i+1,"validation_eb":s[3],"test_eb":s[4],"n_train_cells":int(s[0].sum()),"n_validation_cells":int(s[1].sum()),"n_test_cells":int(s[2].sum())} for i,s in enumerate(splits)]).to_csv(OUTDIR/"split_manifest.csv",index=False)
    config={"model_version":MODEL_VERSION,"data_path":str(DATA_PATH),"source_split_manifest":str(split_path),"feature_selection_file":str(PEAK_FILE),"output_directory":str(OUTDIR),"modality":"ATAC_only","architecture":"same_NonSpatialModality_and_SpatialResidualModality_as_previous_ATAC_only_script","atac_feature_selection":"fixed_union_of_402_ATAC_peaks_FDR_significant_in_KD_permutation_or_GP_analysis","candidate_universe_full_dataset_derived":True,"n_atac_features":N_ATAC_FEATURES,"n_heldout_ebs":N_HELDOUT_EBS,"final_seeds":FINAL_SEEDS,"atac_graph_candidates":ATAC_GRAPH_CANDIDATES,"loss_weight_candidates":LOSS_WEIGHT_CANDIDATES}
    (OUTDIR/"configuration.json").write_text(json.dumps(config,indent=2));pd.DataFrame({"guide":guide_names}).to_csv(OUTDIR/"guide_names.csv",index=False);pd.DataFrame({"cell_type":celltype_names}).to_csv(OUTDIR/"cell_type_names.csv",index=False)
    x_np=np.concatenate([guides_np,celltypes_np],axis=1).astype(np.float32);x=torch.tensor(x_np,dtype=torch.float32,device=DEVICE);guides=torch.tensor(guides_np,dtype=torch.float32,device=DEVICE);celltypes=torch.tensor(celltypes_np,dtype=torch.float32,device=DEVICE);graph_cache={}
    def graph_bundle(k:int,decay:float):
        key=(int(k),float(decay))
        if key not in graph_cache:
            edge_cpu,dist_cpu=build_knn_graph(coords,eb,k);context_np=weighted_neighbor_context(guides_np,celltypes_np,edge_cpu,dist_cpu,decay);graph_cache[key]=(edge_cpu.to(DEVICE),dist_cpu.to(DEVICE),torch.tensor(context_np,dtype=torch.float32,device=DEVICE))
        return graph_cache[key]
    if preflight_only:
        train_np,_,_,val_eb,test_eb=splits[0];idx,feature_table=load_atac_features(np.asarray(atac.var_names.astype(str)));print(f"PREFLIGHT PASSED: run1 validation={val_eb}, test={test_eb}, selected ATAC peaks={len(idx)}, source={PEAK_FILE}");return
    per_seed_path=OUTDIR/"per_seed_test_metrics.csv";per_seed_rows=load_csv(per_seed_path).to_dict("records");ensemble_rows=load_csv(ensemble_path).to_dict("records");context_input_dim=guides_np.shape[1]+celltypes_np.shape[1]+3
    for run_idx,(train_np,val_np,test_np,val_eb,test_eb) in enumerate(splits,1):
        print(f"\nRUN {run_idx} validation={val_eb} test={test_eb}",flush=True);atac_idx,feature_table=load_atac_features(np.asarray(atac.var_names.astype(str)));y_np=atac_counts[:,atac_idx].toarray().astype(np.float32);y=torch.tensor(y_np,dtype=torch.float32,device=DEVICE);library=(np.asarray(atac_counts.sum(1)).ravel()-y_np.sum(1)).astype(np.float32)
        if np.any(library<=0):raise ValueError("Non-positive non-target ATAC library size")
        loglib_np=np.log(library).astype(np.float32);mu0_np=np.log(np.clip(y_np[train_np].sum(0,dtype=np.float64)/library[train_np].sum(dtype=np.float64),1e-12,None)).astype(np.float32);loglib=torch.tensor(loglib_np,dtype=torch.float32,device=DEVICE);mu0=torch.tensor(mu0_np,dtype=torch.float32,device=DEVICE);train=torch.tensor(train_np,dtype=torch.bool,device=DEVICE);val=torch.tensor(val_np,dtype=torch.bool,device=DEVICE);test=torch.tensor(test_np,dtype=torch.bool,device=DEVICE)
        feature_table.to_csv(OUTDIR/f"run{run_idx}_atac_402_FDR_features.csv",index=False);np.savez_compressed(OUTDIR/f"run{run_idx}_atac_offsets_and_targets.npz",atac_target_idx=atac_idx,atac_mu0=mu0_np,atac_log_library=loglib_np);np.save(OUTDIR/f"run{run_idx}_heldout_true_atac_counts.npy",y_np[test_np])
        tune_seed=TUNE_SEED_BASE+run_idx;tune_mlp_path=OUTDIR/f"run{run_idx}_tune_atac_mlp_best.pt";seed_all(tune_seed);tune_mlp=NonSpatialModality(x.shape[1],guides_np.shape[1],celltypes_np.shape[1],len(atac_idx))
        if tune_mlp_path.exists():tune_mlp.load_state_dict(torch.load(tune_mlp_path,map_location=DEVICE));tune_mlp=tune_mlp.to(DEVICE)
        else:tune_mlp,_=train_mlp(tune_mlp,x,guides,celltypes,y,loglib,mu0,train,val,TUNE_MLP_MAX_EPOCHS,None);torch.save(tune_mlp.state_dict(),tune_mlp_path)
        tuning_path=OUTDIR/f"run{run_idx}_atac_graph_tuning.csv";tuning_rows=load_csv(tuning_path).to_dict("records")
        for k,decay in ATAC_GRAPH_CANDIDATES:
            if any(str(r.get("model_version"))==MODEL_VERSION and int(r.get("k_neighbors"))==k and abs(float(r.get("distance_decay"))-decay)<1e-9 for r in tuning_rows):continue
            seed_all(tune_seed);branch=SpatialResidualModality(x.shape[1],guides_np.shape[1],celltypes_np.shape[1],len(atac_idx),context_input_dim,decay);branch.base.load_state_dict(copy.deepcopy(tune_mlp.state_dict()));branch,best=train_spatial(branch,x,guides,celltypes,graph_bundle(k,decay),y,loglib,mu0,train,val,TUNE_SPATIAL_MAX_EPOCHS);tuning_rows.append({"model_version":MODEL_VERSION,"run":run_idx,"validation_eb":val_eb,"test_eb":test_eb,"k_neighbors":k,"distance_decay":decay,**best});pd.DataFrame(tuning_rows).to_csv(tuning_path,index=False);del branch
            if torch.cuda.is_available():torch.cuda.empty_cache()
        tuning_df=pd.DataFrame(tuning_rows);subset=tuning_df[tuning_df["model_version"]==MODEL_VERSION].copy();best_row=subset.sort_values(["best_val_score","best_val_nb_nll","k_neighbors","distance_decay"]).iloc[0];selected_graph=(int(best_row["k_neighbors"]),float(best_row["distance_decay"]));pd.DataFrame([{"run":run_idx,"k_neighbors":selected_graph[0],"distance_decay":selected_graph[1]}]).to_csv(OUTDIR/f"run{run_idx}_selected_atac_graph.csv",index=False);bundle=graph_bundle(*selected_graph);print(f"Selected ATAC graph: k={selected_graph[0]}, decay={selected_graph[1]}",flush=True)
        loss_tuning_path=OUTDIR/f"run{run_idx}_atac_loss_weight_tuning.csv";loss_rows=load_csv(loss_tuning_path).to_dict("records");base_state=tune_mlp.state_dict()
        for logw,corrw,nmsew in LOSS_WEIGHT_CANDIDATES:
            if any(str(r.get("model_version"))==MODEL_VERSION and abs(float(r.get("log1p_mse_weight"))-logw)<1e-12 and abs(float(r.get("corr_weight"))-corrw)<1e-12 and abs(float(r.get("nmse_weight"))-nmsew)<1e-12 for r in loss_rows):continue
            seed_all(tune_seed);branch=SpatialResidualModality(x.shape[1],guides_np.shape[1],celltypes_np.shape[1],len(atac_idx),context_input_dim,selected_graph[1]);branch.base.load_state_dict(copy.deepcopy(base_state));branch,best=train_spatial(branch,x,guides,celltypes,bundle,y,loglib,mu0,train,val,TUNE_SPATIAL_MAX_EPOCHS,(logw,corrw,nmsew));loss_rows.append({"model_version":MODEL_VERSION,"run":run_idx,"validation_eb":val_eb,"test_eb":test_eb,"log1p_mse_weight":logw,"corr_weight":corrw,"nmse_weight":nmsew,**best});pd.DataFrame(loss_rows).to_csv(loss_tuning_path,index=False);del branch
            if torch.cuda.is_available():torch.cuda.empty_cache()
        loss_df=pd.DataFrame(loss_rows);subset=loss_df[loss_df["model_version"]==MODEL_VERSION].copy();subset["rank_sum"]=subset["best_val_nb_nll"].rank(method="min",ascending=True)+subset["best_val_raw_pearson"].rank(method="min",ascending=False)+subset["best_val_raw_rmse"].rank(method="min",ascending=True);best_row=subset.sort_values(["rank_sum","best_val_nb_nll","best_val_raw_rmse"],ascending=[True,True,True]).iloc[0];selected_weights=(float(best_row["log1p_mse_weight"]),float(best_row["corr_weight"]),float(best_row["nmse_weight"]));pd.DataFrame([{"run":run_idx,"log1p_mse_weight":selected_weights[0],"corr_weight":selected_weights[1],"nmse_weight":selected_weights[2]}]).to_csv(OUTDIR/f"run{run_idx}_selected_atac_loss_weights.csv",index=False);print(f"Selected ATAC loss weights: logMSE={selected_weights[0]}, Pearson={selected_weights[1]}, NMSE={selected_weights[2]}",flush=True)
        seed_predictions={"mlp":{"mu":[],"theta":[]},"gat":{"mu":[],"theta":[]}}
        for seed in FINAL_SEEDS:
            prefix=f"run{run_idx}_seed{seed}";mlp_checkpoint=OUTDIR/f"{prefix}_mlp_best.pt";seed_all(seed);mlp=NonSpatialModality(x.shape[1],guides_np.shape[1],celltypes_np.shape[1],len(atac_idx))
            if mlp_checkpoint.exists():mlp.load_state_dict(torch.load(mlp_checkpoint,map_location=DEVICE));mlp=mlp.to(DEVICE)
            else:mlp,_=train_mlp(mlp,x,guides,celltypes,y,loglib,mu0,train,val,FINAL_MLP_MAX_EPOCHS,prefix,selected_weights)
            for name in ["mlp","gat"]:
                model=None;pred_path=OUTDIR/f"{prefix}_{name}_heldout_atac_pred.npy";theta_path=OUTDIR/f"{prefix}_{name}_atac_theta.npy";existing=next((r for r in per_seed_rows if str(r.get("model_version"))==MODEL_VERSION and int(r["run"])==run_idx and int(r["seed"])==seed and str(r["model"])==name),None)
                if existing is not None and pred_path.exists() and theta_path.exists():pred=np.load(pred_path);theta_np=np.load(theta_path)
                else:
                    if name=="mlp":model=mlp;model.eval()
                    else:
                        gat_checkpoint=OUTDIR/f"{prefix}_gat_best.pt";model=SpatialResidualModality(x.shape[1],guides_np.shape[1],celltypes_np.shape[1],len(atac_idx),context_input_dim,selected_graph[1]);model.base.load_state_dict(copy.deepcopy(mlp.state_dict()))
                        if gat_checkpoint.exists():model.load_state_dict(torch.load(gat_checkpoint,map_location=DEVICE));model=model.to(DEVICE)
                        else:seed_all(seed);model,_=train_spatial(model,x,guides,celltypes,bundle,y,loglib,mu0,train,val,FINAL_SPATIAL_MAX_EPOCHS,selected_weights,prefix)
                        model.eval()
                    with torch.no_grad():
                        if name=="mlp":residual=model(x,guides,celltypes);theta_tensor=model.raw_theta
                        else:edge,dist,context=bundle;residual=model(x,guides,celltypes,edge,dist,context);theta_tensor=model.base.raw_theta
                        loss=float(nb_nll(residual[test],y[test],loglib[test],mu0,theta_tensor));pred=make_mu(residual[test],loglib[test],mu0).cpu().numpy();theta_np=positive_theta(theta_tensor).cpu().numpy()
                    row={"model_version":MODEL_VERSION,"run":run_idx,"seed":seed,"model":name,"validation_eb":val_eb,"test_eb":test_eb,"atac_k_neighbors":selected_graph[0],"atac_distance_decay":selected_graph[1],"atac_log1p_mse_weight":selected_weights[0],"atac_corr_weight":selected_weights[1],"atac_nmse_weight":selected_weights[2],"test_atac_nb_nll":loss,"n_parameters":sum(p.numel() for p in model.parameters())};row.update(metrics(y_np[test_np],pred));per_seed_rows=[r for r in per_seed_rows if not (str(r.get("model_version"))==MODEL_VERSION and int(r["run"])==run_idx and int(r["seed"])==seed and str(r["model"])==name)];per_seed_rows.append(row);pd.DataFrame(per_seed_rows).sort_values(["run","model","seed"]).to_csv(per_seed_path,index=False);np.save(pred_path,pred.astype(np.float32));np.save(theta_path,theta_np.astype(np.float32))
                seed_predictions[name]["mu"].append(pred);seed_predictions[name]["theta"].append(theta_np)
                if name=="gat" and model is not None:del model
                if torch.cuda.is_available():torch.cuda.empty_cache()
            del mlp
        for name in ["mlp","gat"]:
            ensemble_pred=np.mean(np.stack(seed_predictions[name]["mu"],axis=0),axis=0);loss=mixture_nb_nll(y_np[test_np],seed_predictions[name]["mu"],seed_predictions[name]["theta"]);row={"model_version":MODEL_VERSION,"run":run_idx,"model":name,"validation_eb":val_eb,"test_eb":test_eb,"atac_k_neighbors":selected_graph[0],"atac_distance_decay":selected_graph[1],"n_seeds":len(FINAL_SEEDS),"test_atac_nb_nll":loss};row.update(metrics(y_np[test_np],ensemble_pred));ensemble_rows=[r for r in ensemble_rows if not (str(r.get("model_version"))==MODEL_VERSION and int(r["run"])==run_idx and str(r["model"])==name)];ensemble_rows.append(row);pd.DataFrame(ensemble_rows).sort_values(["run","model"]).to_csv(ensemble_path,index=False);np.save(OUTDIR/f"run{run_idx}_{name}_3seed_ensemble_atac_pred.npy",ensemble_pred.astype(np.float32))
    res=pd.DataFrame(ensemble_rows);res=res[res["model_version"]==MODEL_VERSION].copy();summary=summarize(res);print(f"\nPER-EB THREE-SEED ATAC ENSEMBLE METRICS ({N_HELDOUT_EBS} held-out EBs)");print(res.sort_values(["run","model"]).to_string(index=False));print("\nSUMMARY");print(summary);print("\nOutputs:",OUTDIR.resolve())

main(False,False)

# ATAC 402-peak mean baseline
from pathlib import Path
import json,math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter,MaxNLocator
import mudata as mu
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr,spearmanr
from sklearn.metrics import mean_absolute_error,mean_squared_error
OUTDIR=Path("/scratch/welchjd_root/welchjd1/javidgmh/Non_Autonomous_perturbation_project/15515_Multiome/catatac_work/motif_analysis/ATAC_ONLY_402_FDR_PEAKS_10EB")
EB_KEY="EB_manual_clustering"
def valid_eb_label(value):
    s=str(value).strip()
    return s.startswith("EB_") and s.lower() not in {"eb_","eb_nan","eb_none","eb_unassigned"}
def nb_log_prob_from_mu(y,mu,theta):
    mu=mu.clamp_min(1e-12);theta=theta.clamp_min(1e-4);logmu=torch.log(mu);logtheta=torch.log(theta);log_theta_plus_mu=torch.logaddexp(logtheta[None,:],logmu)
    return torch.lgamma(y+theta[None,:])-torch.lgamma(theta[None,:])-torch.lgamma(y+1)+theta[None,:]*(logtheta[None,:]-log_theta_plus_mu)+y*(logmu-log_theta_plus_mu)
def mixture_nb_nll(y_np,mu_list,theta_list):
    y=torch.tensor(y_np,dtype=torch.float32);log_probs=[]
    for mu_np,theta_np in zip(mu_list,theta_list):
        log_probs.append(nb_log_prob_from_mu(y,torch.tensor(mu_np,dtype=torch.float32),torch.tensor(theta_np,dtype=torch.float32)))
    return float(-(torch.logsumexp(torch.stack(log_probs),dim=0)-math.log(len(log_probs))).mean())
def safe_corr(x,y,method):
    if len(x)<2 or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)) or np.std(x)==0 or np.std(y)==0:return np.nan
    return float(pearsonr(x,y).statistic if method=="pearson" else spearmanr(x,y).statistic)
def metrics(y,mu_pred):
    yt=np.asarray(y,dtype=np.float64).ravel();yp=np.asarray(mu_pred,dtype=np.float64).ravel();valid=np.isfinite(yt)&np.isfinite(yp)&(yt>=0)&(yp>=0);yt=yt[valid];yp=yp[valid];xl=np.log1p(yt);yl=np.log1p(yp)
    return {"atac_raw_pearson":safe_corr(yt,yp,"pearson"),"atac_raw_spearman":safe_corr(yt,yp,"spearman"),"atac_raw_rmse":float(np.sqrt(mean_squared_error(yt,yp))),"atac_raw_mae":float(mean_absolute_error(yt,yp)),"atac_log1p_pearson":safe_corr(xl,yl,"pearson"),"atac_log1p_spearman":safe_corr(xl,yl,"spearman"),"atac_log1p_rmse":float(np.sqrt(mean_squared_error(xl,yl))),"atac_log1p_mae":float(mean_absolute_error(xl,yl))}
def main():
    config=json.loads((OUTDIR/"configuration.json").read_text());model_version=config["model_version"];data_path=Path(config["data_path"]);seeds=[int(x) for x in config["final_seeds"]];n_ebs=int(config["n_heldout_ebs"])
    ensemble_path=OUTDIR/"per_eb_ensemble_metrics.csv";split_path=OUTDIR/"split_manifest.csv"
    if not ensemble_path.exists():raise FileNotFoundError(ensemble_path)
    if not split_path.exists():raise FileNotFoundError(split_path)
    res=pd.read_csv(ensemble_path);res=res[(res["model_version"]==model_version)&res["model"].isin(["mlp","gat"])].copy()
    for name in ["mlp","gat"]:
        sub=res[res["model"]==name]
        if len(sub)!=n_ebs:raise ValueError(f"Expected {n_ebs} rows for {name}, found {len(sub)}")
    splits=pd.read_csv(split_path).sort_values("run")
    mu.set_options(pull_on_update=False);mdata=mu.read_h5mu(data_path);eb_all=mdata.obs[EB_KEY].astype(str).to_numpy();valid=np.asarray([valid_eb_label(x) for x in eb_all]);eb=eb_all[valid];del mdata
    baseline_rows=[]
    for row in splits.itertuples(index=False):
        run=int(row.run);test_eb=str(row.test_eb);val_eb=str(row.validation_eb);test_mask=eb==test_eb
        z=np.load(OUTDIR/f"run{run}_atac_offsets_and_targets.npz");mu0=np.asarray(z["atac_mu0"],dtype=np.float64);loglib=np.asarray(z["atac_log_library"],dtype=np.float64);y=np.load(OUTDIR/f"run{run}_heldout_true_atac_counts.npy").astype(np.float64)
        if len(loglib)!=len(eb):raise ValueError(f"Run {run}: saved library-size vector does not match filtered cells")
        if int(test_mask.sum())!=len(y):raise ValueError(f"Run {run}: test-cell count mismatch")
        pred=np.exp(np.clip(loglib[test_mask,None]+mu0[None,:],-12,12))
        theta_list=[]
        for seed in seeds:
            p=OUTDIR/f"run{run}_seed{seed}_mlp_atac_theta.npy"
            if not p.exists():raise FileNotFoundError(p)
            theta_list.append(np.load(p).astype(np.float64))
        nll=mixture_nb_nll(y,[pred]*len(theta_list),theta_list)
        out={"model_version":model_version,"run":run,"model":"mean_baseline","validation_eb":val_eb,"test_eb":test_eb,"n_seeds":0,"test_atac_nb_nll":nll}
        out.update(metrics(y,pred));baseline_rows.append(out);np.save(OUTDIR/f"run{run}_mean_baseline_atac_pred.npy",pred.astype(np.float32))
        print(f"Run {run} {test_eb}: mean baseline NB NLL = {nll:.8f}",flush=True)
    baseline=pd.DataFrame(baseline_rows);combined=pd.concat([res,baseline],ignore_index=True,sort=False);combined=combined.sort_values(["run","model"])
    combined.to_csv(OUTDIR/"per_eb_ensemble_metrics_with_mean_baseline.csv",index=False)
    metric_cols=["test_atac_nb_nll","atac_raw_pearson","atac_raw_spearman","atac_raw_rmse","atac_raw_mae","atac_log1p_pearson","atac_log1p_spearman","atac_log1p_rmse","atac_log1p_mae"]
    summary=combined.groupby("model")[metric_cols].agg(["mean","std","sem"]);summary.to_csv(OUTDIR/"ATAC_model_summary_with_mean_baseline.csv")
    order=["mean_baseline","mlp","gat"];labels=["Mean\nbaseline","MLP","GAT"];colors=["#D9D9D9","#969696","#4C78A8"]
    stats=combined.groupby("model")["test_atac_nb_nll"].agg(["mean","sem"]).loc[order];means=stats["mean"].to_numpy();sems=stats["sem"].to_numpy()
    paired=combined[combined["model"].isin(["mlp","gat"])].pivot(index="run",columns="model",values="test_atac_nb_nll").sort_index()[["mlp","gat"]]
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(7.0,3.4),gridspec_kw={"width_ratios":[1.05,1]})
    x=np.arange(3);ax1.bar(x,means,yerr=sems,width=0.62,color=colors,edgecolor="black",linewidth=0.7,capsize=3);ax1.set_xticks(x,labels);ax1.set_ylabel("Held-out ATAC NB NLL");ax1.set_ylim(0,float(np.max(means+sems))*1.08);ax1.set_title("Full scale");ax1.spines[["top","right"]].set_visible(False);ax1.grid(False)
    for _,r in paired.iterrows():ax2.plot([0,1],[r["mlp"],r["gat"]],color="0.75",linewidth=0.8,zorder=1)
    ax2.scatter(np.zeros(len(paired)),paired["mlp"],s=18,color="#969696",edgecolor="black",linewidth=0.35,zorder=2);ax2.scatter(np.ones(len(paired)),paired["gat"],s=18,color="#4C78A8",edgecolor="black",linewidth=0.35,zorder=2)
    zoom_means=paired.mean().to_numpy();zoom_sems=paired.sem().to_numpy();ax2.errorbar([0,1],zoom_means,yerr=zoom_sems,fmt="o",markersize=6,color="black",capsize=4,linewidth=1.4,zorder=3)
    vals=paired.to_numpy().ravel();spread=max(float(vals.max()-vals.min()),1e-6);pad=max(0.12*spread,1e-6);ax2.set_ylim(float(vals.min()-pad),float(vals.max()+pad));ax2.set_xticks([0,1],["MLP","GAT"]);ax2.set_title(f"MLP vs GAT zoom\npaired held-out EBs (n={len(paired)})");ax2.yaxis.set_major_locator(MaxNLocator(6));ax2.yaxis.set_major_formatter(FormatStrFormatter("%.6f"));ax2.spines[["top","right"]].set_visible(False);ax2.grid(False)
    fig.tight_layout();fig.savefig(OUTDIR/"ATAC_NB_loss_mean_baseline_full_and_MLP_GAT_zoom.png",dpi=600,bbox_inches="tight");fig.savefig(OUTDIR/"ATAC_NB_loss_mean_baseline_full_and_MLP_GAT_zoom.pdf",bbox_inches="tight");plt.close(fig)
    print("\nNB NLL summary:")
    print(stats)
    print("\nSaved:",OUTDIR/"ATAC_NB_loss_mean_baseline_full_and_MLP_GAT_zoom.pdf")
main()

# ATAC 402-peak mean baseline, MLP, and GAT within-EB SEM
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter,MaxNLocator
OUTDIR=Path("/scratch/welchjd_root/welchjd1/javidgmh/Non_Autonomous_perturbation_project/15515_Multiome/catatac_work/motif_analysis/ATAC_ONLY_402_FDR_PEAKS_10EB")
MODEL_VERSION=json.loads((OUTDIR/"configuration.json").read_text())["model_version"]
orig=pd.read_csv(OUTDIR/"per_eb_ensemble_metrics.csv")
orig=orig[(orig["model_version"]==MODEL_VERSION)&orig["model"].isin(["mlp","gat"])].copy()
all_df=pd.read_csv(OUTDIR/"per_eb_ensemble_metrics_with_mean_baseline.csv")
base=all_df[(all_df["model_version"]==MODEL_VERSION)&(all_df["model"]=="mean_baseline")].copy()
df=pd.concat([base,orig],ignore_index=True)
wide=df.pivot(index="run",columns="model",values="test_atac_nb_nll")[["mean_baseline","mlp","gat"]].sort_index()
if len(wide)!=10 or wide.isna().any().any():raise ValueError("Need all 3 models for exactly 10 held-out EBs")
means=wide.mean()
grand=wide.to_numpy().mean()
normalized=wide.sub(wide.mean(axis=1),axis=0)+grand
k=wide.shape[1]
sems=normalized.sem(axis=0)*np.sqrt(k/(k-1))
print("RAW MEANS:")
print(means)
print("\nWITHIN-EB SEM:")
print(sems)
print("\nMean paired improvements:")
print("Baseline - GAT =", (wide["mean_baseline"]-wide["gat"]).mean())
print("MLP - GAT      =", (wide["mlp"]-wide["gat"]).mean())
x=np.arange(3);labels=["Mean baseline","MLP","GAT"];colors=["#D9D9D9","#969696","#4C78A8"]
y=means.to_numpy();err=sems.to_numpy()
fig,ax=plt.subplots(figsize=(4.4,3.6))
ax.bar(x,y,yerr=err,width=0.60,color=colors,edgecolor="black",linewidth=0.7,capsize=3)
low=np.min(y-err);high=np.max(y+err);pad=max((high-low)*0.35,0.00002)
ax.set_ylim(low-pad,high+pad)
ax.set_xticks(x,labels)
ax.set_ylabel("Held-out ATAC NB NLL")
ax.yaxis.set_major_locator(MaxNLocator(6))
ax.yaxis.set_major_formatter(FormatStrFormatter("%.6f"))
ax.spines[["top","right"]].set_visible(False)
ax.grid(False)
fig.tight_layout()
fig.savefig(OUTDIR/"ATAC_NB_baseline_MLP_GAT_WITHIN_EB_SEM.png",dpi=600,bbox_inches="tight")
fig.savefig(OUTDIR/"ATAC_NB_baseline_MLP_GAT_WITHIN_EB_SEM.pdf",bbox_inches="tight")
plt.close(fig)

