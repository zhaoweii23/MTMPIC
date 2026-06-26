#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import json
import argparse
import logging
import numpy as np
import torch
import pandas as pd
import pickle
from collections import defaultdict
from typing import Dict, Optional, List, Any, Tuple
from torch.utils.data import Dataset, DataLoader

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import Data
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
from torch_scatter import scatter_mean

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.preprocessing import label_binarize
import time
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

class ModelConfig:
    device: torch.device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    use_amp: bool = True

    mol_atom_dim: int = 75
    mol_bond_dim: int = 8
    prot_atom_dim: int = 6
    prot_seq_dim: int = 1280
    mol_seq_dim: int = 384
    prot_bond_dim: int = 32


    hidden_dim: int = 384
    num_classes: int = 3
    dropout_rate: float = 0.5
    gnn_layers: int = 2
    num_heads: int = 8

    learning_rate: float = 5e-5
    weight_decay: float = 1e-3
    batch_size: int = 32
    grad_clip: float = 1.0
    epochs: int = 100
    early_stop_patient: int = 7

    save_dir: str = "experiment_results_NN_42/saved_models"
    log_dir: str = "experiment_results_NN_42/logs"
    plot_dir: str = "experiment_results_NN_42/plots"

    label_mapping: Dict[int, str] = {0: "inactive", 1: "inhibitor", 2: "activator"}

    @classmethod
    def setup_dirs(cls):
        os.makedirs(cls.save_dir, exist_ok=True)
        os.makedirs(cls.log_dir, exist_ok=True)
        os.makedirs(cls.plot_dir, exist_ok=True)


DEVICE_TYPE = "cuda" if torch.cuda.is_available() else "cpu"

FEATURE_CACHE_DIR = "feature_cache/high_volume"
SPLIT_DIR = "split_datasets"
RESULT_DIR = "experiment_results"
os.makedirs(RESULT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(RESULT_DIR, "experiment.log")),
        logging.StreamHandler(),
    ],
)

ABB_TO_GROUP = {
    "KCa_CNG": "tiny_volume", "CaCC": "tiny_volume", "AQP": "tiny_volume",
    "CFTR": "low_volume", "5-HT3R": "low_volume", "GABA_Gly": "low_volume", "VGKCs": "low_volume",
    "Kir": "mid_volume", "K2P": "mid_volume", "nAChRs": "mid_volume", "VGCCs": "mid_volume", "i-GluRs": "mid_volume",
    "TRP": "high_volume", "P2X": "high_volume", "VGSCs": "high_volume", "Herg": "high_volume",
}


def create_defaultdict_int():
    return defaultdict(int)


class AttributeSubset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]
    def __len__(self):
        return len(self.indices)
    def __getattr__(self, name):
        if name in ("dataset", "indices"):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        return getattr(self.dataset, name)


class PKLDataset(Dataset):
    def __init__(self, root_path: str):
        self.root_path = root_path
        self.task_label_counts = defaultdict(create_defaultdict_int)
        self._scanned_tasks = set()
        self.file_list = []
        for root, dirs, files in os.walk(root_path):
            for file in files:
                if file.endswith('.pkl'):
                    self.file_list.append(os.path.join(root, file))
        self.file_list = sorted(self.file_list, key=str.lower)
        self._scan_task_distribution()

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        file_path = self.file_list[idx]
        try:
            with open(file_path, 'rb') as f:
                sample = pickle.load(f)
            mol_data = sample['mol_fea']
            prot_data = sample['prot_fea']
            return {
                'mol_data': Data(
                    mol_x=mol_data.mol_atoms_feats,
                    mol_edge_attr=mol_data.mol_edges_feats,
                    mol_edge_index=mol_data.mol_edge_index,
                    atoinmol_index=mol_data.atoinmol_index,
                    mol_pos=mol_data.mol_coords_feats,
                    mol_dist=mol_data.mol_dist,
                    mol_theta=mol_data.mol_theta,
                    mol_phi=mol_data.mol_phi,
                    mol_tau=mol_data.mol_tau,
                    atom_types=mol_data.atom_types,
                    mol_embedding=mol_data.mol_embedding,
                    mol_smiles=sample.get('smiles', '')
                ),
                'prot_data': Data(
                    prot_atom_feats_s=prot_data.pro_atoms_feats_s,
                    prot_atom_feats_v=prot_data.pro_atoms_feats_v,
                    prot_edge_feats_s=prot_data.pro_edges_feats_s,
                    prot_edge_feats_v=prot_data.pro_edges_feats_v,
                    pro_index=prot_data.pro_index,
                    pro_edge_index=prot_data.pro_edge_index,
                    pol_pos=prot_data.pro_coords_feats,
                    pro_dist=prot_data.pr_dist,
                    pro_theta=prot_data.pr_theta,
                    pro_phi=prot_data.pr_phi,
                    pro_tau=prot_data.pr_tau,
                    pro_seq_embedding=prot_data.pro_token_repre,
                    pro_seq=prot_data.pro_fp
                ),
                'label': torch.tensor(sample['label'], dtype=torch.long),
                'task': sample['task']
            }
        except Exception as e:
            logging.error(f"Failed to load sample {file_path}: {str(e)}")
            return None

    def _scan_task_distribution(self):
        cache_dir = os.path.join(self.root_path, ".cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, "task_distribution.pkl")
        try:
            if os.path.exists(cache_file):
                latest_mtime = max(os.path.getmtime(f) for f in self.file_list)
                if os.path.getmtime(cache_file) > latest_mtime:
                    with open(cache_file, 'rb') as f:
                        cached = pickle.load(f)
                    self.task_label_counts = cached['counts']
                    self._scanned_tasks = cached['tasks']
                    return
        except Exception:
            pass
        for file_path in tqdm(self.file_list, desc="Scanning tasks", leave=False):
            try:
                with open(file_path, 'rb') as f:
                    sample = pickle.load(f)
                if 'task' in sample and 'label' in sample:
                    task = sample['task']
                    label = sample['label']
                    if torch.is_tensor(label):
                        label = label.item()
                    self.task_label_counts[task][int(label)] += 1
                    self._scanned_tasks.add(task)
            except Exception:
                continue
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump({'counts': self.task_label_counts, 'tasks': self._scanned_tasks}, f)
        except Exception:
            pass


class CSVSubsetDataset(Dataset):
    def __init__(self, csv_path: str, cache_root: str, group: Optional[str] = None,
                 df_with_group: Optional[pd.DataFrame] = None):
        self.cache_root = cache_root
        self.group = group
        if df_with_group is not None:
            self.df = df_with_group.copy()
        else:
            self.df = pd.read_csv(csv_path)
        if "group" not in self.df.columns:
            if "ABB" in self.df.columns:
                self.df["group"] = self.df["ABB"].map(ABB_TO_GROUP).fillna("unknown")
            else:
                self.df["group"] = "unknown"
        self.file_mapping = self._build_file_mapping(cache_root)
        self._add_match_keys()
        if group is not None:
            self.df = self.df[self.df["group"] == group].reset_index(drop=True)
        self._filter_missing_files()
        self.task_label_counts = self._scan_task_distribution()
        self._scanned_tasks = set(self.task_label_counts.keys())
        logging.info(f"CSVSubsetDataset (group={group or 'all'}): {len(self.df)} samples")

    def _build_file_mapping(self, cache_root: str) -> Dict[Tuple[str, str], str]:
        mapping = {}
        for root, _, files in os.walk(cache_root):
            for f in files:
                if not f.endswith(".pkl"): continue
                name = f[:-4]
                if "_" not in name: continue
                target_id = name.split("_", 1)[0]
                drug_id = name.rsplit("_", 1)[-1]
                mapping[(target_id, drug_id)] = os.path.join(root, f)
        logging.info(f"Found {len(mapping)} pkl files in {cache_root}")
        return mapping

    def _add_match_keys(self):
        target_col = "Target_ID" if "Target_ID" in self.df.columns else "UniProt ID"
        drug_col = "Drug_ID" if "Drug_ID" in self.df.columns else "SMILES"
        self.df["_target_id"] = self.df[target_col].astype(str)
        self.df["_drug_id"] = self.df[drug_col].astype(str)

    def _filter_missing_files(self):
        def exists(row):
            key = (row["_target_id"], row["_drug_id"])
            return key in self.file_mapping
        mask = self.df.apply(exists, axis=1)
        dropped = (~mask).sum()
        if dropped > 0:
            logging.warning(f"Dropping {dropped} samples without pkl files")
            self.df = self.df[mask].reset_index(drop=True)

    def _scan_task_distribution(self):
        task_counts = defaultdict(lambda: defaultdict(int))
        for _, row in self.df.iterrows():
            task = row["ABB"]
            label = int(row["Label"])
            task_counts[task][label] += 1
        return task_counts

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        key = (row["_target_id"], row["_drug_id"])
        with open(self.file_mapping[key], "rb") as f:
            sample = pickle.load(f)
        mol_data = sample["mol_fea"]
        prot_data = sample["prot_fea"]
        mol_obj = Data(
            mol_x=mol_data.mol_atoms_feats,
            mol_edge_attr=mol_data.mol_edges_feats,
            mol_edge_index=mol_data.mol_edge_index,
            atoinmol_index=mol_data.atoinmol_index,
            mol_pos=mol_data.mol_coords_feats,
            mol_dist=mol_data.mol_dist,
            mol_theta=mol_data.mol_theta,
            mol_phi=mol_data.mol_phi,
            mol_tau=mol_data.mol_tau,
            atom_types=mol_data.atom_types,
            mol_embedding=mol_data.mol_embedding,
            mol_smiles=sample.get("smiles", ""),
        )
        prot_obj = Data(
            prot_atom_feats_s=prot_data.pro_atoms_feats_s,
            prot_atom_feats_v=prot_data.pro_atoms_feats_v,
            prot_edge_feats_s=prot_data.pro_edges_feats_s,
            prot_edge_feats_v=prot_data.pro_edges_feats_v,
            pro_index=prot_data.pro_index,
            pro_edge_index=prot_data.pro_edge_index,
            pol_pos=prot_data.pro_coords_feats,
            pro_dist=prot_data.pr_dist,
            pro_theta=prot_data.pr_theta,
            pro_phi=prot_data.pr_phi,
            pro_tau=prot_data.pr_tau,
            pro_seq_embedding=prot_data.pro_token_repre,
            pro_seq=prot_data.pro_fp,
        )
        return {
            "mol_data": mol_obj,
            "prot_data": prot_obj,
            "label": torch.tensor(sample["label"], dtype=torch.long),
            "task": sample.get("task", row["ABB"]),
            "group": sample.get("group", row["group"]),
        }


def create_dataset_from_csv(csv_path: str, cache_root: str) -> CSVSubsetDataset:
    return CSVSubsetDataset(csv_path, cache_root, group=None)


class SafeResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(ModelConfig.dropout_rate)
        self.shortcut = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        def _inner_forward(x):
            x = self.linear(x)
            x = torch.nan_to_num(x, nan=0.0, posinf=1e5, neginf=-1e5)
            x = self.norm(x)
            x = torch.clamp(x, -10, 10)
            x = self.activation(x)
            return self.dropout(x)
        x = checkpoint(_inner_forward, x, use_reentrant=False)
        return residual + x


class GNNLayer(MessagePassing):
    def __init__(self, in_dim, out_dim):
        super().__init__(aggr="mean")
        self.message_net = nn.Sequential(nn.Linear(in_dim * 2, out_dim), nn.LayerNorm(out_dim), nn.GELU())
        self.update_net = nn.Sequential(nn.Linear(in_dim + out_dim, out_dim), nn.LayerNorm(out_dim), nn.GELU())

    def forward(self, x, edge_index, edge_attr):
        edge_index, edge_attr = add_self_loops(edge_index, edge_attr, fill_value=torch.mean(edge_attr, dim=0))
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        return self.message_net(torch.cat([x_i, x_j, edge_attr], dim=-1))

    def update(self, aggr_out, x):
        return self.update_net(torch.cat([x, aggr_out], dim=-1))


class GNNEncoder(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, num_layers=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(ModelConfig.dropout_rate)
        self.node_proj = nn.Sequential(nn.Linear(node_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(ModelConfig.dropout_rate))
        self.edge_proj = nn.Sequential(nn.Linear(edge_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(ModelConfig.dropout_rate))
        self.layers = nn.ModuleList([self._build_gnn_layer(hidden_dim) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(hidden_dim)

    def _build_gnn_layer(self, hidden_dim):
        return nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())

    def forward(self, node_feats, edge_feats, edge_index, batch_index):
        x = self.node_proj(node_feats)
        edge_attr = self.edge_proj(edge_feats)
        row, col = edge_index
        for layer in self.layers:
            messages = torch.cat([x[row], edge_attr], dim=-1)
            messages = layer(messages)
            aggregated = scatter_mean(messages, col, dim=0, dim_size=x.size(0))
            x = x + self.dropout(aggregated)
        x = self.norm(x)
        return scatter_mean(x, batch_index, dim=0)


class AttentionPooling(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads=4):
        super().__init__()
        assert output_dim % num_heads == 0
        self.num_heads = num_heads              
        self.head_dim = output_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.input_proj = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
        self.query = nn.Linear(output_dim, output_dim)
        self.key = nn.Linear(output_dim, output_dim)
        self.value = nn.Linear(output_dim, output_dim)
        self.proj = nn.Sequential(nn.Linear(output_dim, output_dim), nn.LayerNorm(output_dim))
        self.pool_vector = nn.Parameter(torch.randn(1, 1, output_dim))

    def forward(self, x):
        x = self.input_proj(x)
        batch_size = x.size(0)
        q = self.query(self.pool_vector).expand(batch_size, -1, -1)
        q = q.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(batch_size, 1, -1)
        return self.proj(out).squeeze(1)


class MoleculeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_dim = ModelConfig.hidden_dim
        self.atom_encoder = nn.Sequential(nn.Linear(ModelConfig.mol_atom_dim, self.hidden_dim // 2), nn.LayerNorm(self.hidden_dim // 2), nn.GELU())
        self.bond_encoder = nn.Sequential(nn.Linear(ModelConfig.mol_bond_dim, self.hidden_dim // 4), nn.LayerNorm(self.hidden_dim // 4), nn.GELU())
        spatial_input_dim = 4 + self.hidden_dim // 4
        self.spatial_encoder = nn.Sequential(nn.Linear(spatial_input_dim, self.hidden_dim // 2), nn.LayerNorm(self.hidden_dim // 2), nn.GELU())
        self.gnn = GNNEncoder(node_dim=self.hidden_dim // 2, edge_dim=self.hidden_dim // 2, hidden_dim=self.hidden_dim, num_layers=ModelConfig.gnn_layers)
        self.seq_encoder = nn.Sequential(
            nn.Linear(ModelConfig.mol_seq_dim, self.hidden_dim),
            nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=self.hidden_dim, nhead=8, dim_feedforward=self.hidden_dim * 4, dropout=ModelConfig.dropout_rate, batch_first=True), num_layers=2),
        )
        self.fusion = nn.Sequential(nn.Linear(self.hidden_dim * 2, self.hidden_dim), SafeResidualBlock(self.hidden_dim, self.hidden_dim))

    def forward(self, mol_data):
        mol_data.atoinmol_index = torch.zeros(mol_data.mol_x.size(0), dtype=torch.long, device=ModelConfig.device)
        if not isinstance(mol_data.mol_edge_index, torch.Tensor):
            mol_data.mol_edge_index = torch.as_tensor(mol_data.mol_edge_index, dtype=torch.long, device=ModelConfig.device)
        else:
            mol_data.mol_edge_index = mol_data.mol_edge_index.to(device=ModelConfig.device, dtype=torch.long)

        atom_feats = self.atom_encoder(mol_data.mol_x)
        bond_feats = self.bond_encoder(mol_data.mol_edge_attr)
        spatial_input = torch.cat([
            mol_data.mol_dist.unsqueeze(-1), mol_data.mol_theta.unsqueeze(-1),
            mol_data.mol_phi.unsqueeze(-1), mol_data.mol_tau.unsqueeze(-1), bond_feats
        ], dim=-1)
        spatial_feats = self.spatial_encoder(spatial_input)
        gnn_feats = self.gnn(atom_feats, spatial_feats, mol_data.mol_edge_index, mol_data.atoinmol_index)
        expanded = gnn_feats[mol_data.atoinmol_index]
        mol_feats = scatter_mean(expanded, mol_data.atoinmol_index, dim=0)
        seq_feats = self.seq_encoder(mol_data.mol_embedding).mean(dim=1)
        combined = torch.cat([mol_feats, seq_feats.expand(mol_feats.size(0), -1)], dim=-1)
        return self.fusion(combined)


class ProteinEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_dim = ModelConfig.hidden_dim
        self.atom_encoder = nn.Sequential(nn.Linear(ModelConfig.prot_atom_dim, self.hidden_dim // 2), nn.LayerNorm(self.hidden_dim // 2), nn.GELU())
        self.edge_encoder = nn.Sequential(nn.Linear(ModelConfig.prot_bond_dim, self.hidden_dim // 4), nn.LayerNorm(self.hidden_dim // 4), nn.GELU())
        spatial_input_dim = 4 + self.hidden_dim // 4
        self.spatial_encoder = nn.Sequential(nn.Linear(spatial_input_dim, self.hidden_dim // 2), nn.LayerNorm(self.hidden_dim // 2), nn.GELU())
        self.gnn = GNNEncoder(node_dim=self.hidden_dim // 2, edge_dim=self.hidden_dim // 2, hidden_dim=self.hidden_dim, num_layers=ModelConfig.gnn_layers)
        self.seq_encoder = nn.Sequential(
            nn.Linear(ModelConfig.prot_seq_dim, self.hidden_dim),
            nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=self.hidden_dim, nhead=8, dim_feedforward=self.hidden_dim * 4, dropout=ModelConfig.dropout_rate, batch_first=True), num_layers=2),
        )
        self.fusion = AttentionPooling(input_dim=2 * self.hidden_dim, output_dim=self.hidden_dim, num_heads=4)

    def forward(self, prot_data):
        prot_data.pro_index = torch.zeros(prot_data.prot_atom_feats_s.size(0), dtype=torch.long, device=ModelConfig.device)
        atom_feats = self.atom_encoder(prot_data.prot_atom_feats_s)
        edge_feats = self.edge_encoder(prot_data.prot_edge_feats_s)
        spatial_feats = torch.cat([
            prot_data.pro_dist.unsqueeze(-1), prot_data.pro_theta.unsqueeze(-1),
            prot_data.pro_phi.unsqueeze(-1), prot_data.pro_tau.unsqueeze(-1), edge_feats
        ], dim=-1)
        spatial_feats = self.spatial_encoder(spatial_feats)
        gnn_feats = self.gnn(atom_feats, spatial_feats, prot_data.pro_edge_index, prot_data.pro_index)
        seq_feats = self.seq_encoder(prot_data.pro_seq_embedding).mean(dim=1)
        expanded = gnn_feats[prot_data.pro_index]
        pro_feats = scatter_mean(expanded, prot_data.pro_index, dim=0)
        combined = torch.cat([pro_feats, seq_feats.expand(pro_feats.size(0), -1)], dim=-1)
        return self.fusion(combined)


class CrossAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
    def forward(self, x, context):
        q = self.query(x); k = self.key(context); v = self.value(context)
        attn = torch.softmax(q @ k.transpose(-2, -1) / (x.size(-1)**0.5), dim=-1)
        return attn @ v


class CrossModalFusion(nn.Module):
    def __init__(self, mol_feat_dim=384, prot_feat_dim=384, hidden_dim=384):
        super().__init__()
        self.mol_proj = nn.Sequential(nn.Linear(mol_feat_dim, hidden_dim), nn.LayerNorm(hidden_dim))
        self.prot_proj = nn.Sequential(nn.Linear(prot_feat_dim, hidden_dim), nn.LayerNorm(hidden_dim))
        self.mol_to_prot = CrossAttention(hidden_dim)
        self.prot_to_mol = CrossAttention(hidden_dim)
        self.mol_global = nn.Linear(hidden_dim, hidden_dim)
        self.prot_global = nn.Linear(hidden_dim, hidden_dim)
        self.fusion = nn.Sequential(nn.Linear(hidden_dim*2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))

    def forward(self, mol_feat, prot_feat):
        mol = self.mol_proj(mol_feat); prot = self.prot_proj(prot_feat)
        mol_enhanced = self.prot_to_mol(mol, prot)
        prot_enhanced = self.mol_to_prot(prot, mol)
        mol_g = self.mol_global(mol_enhanced.mean(dim=0))
        prot_g = self.prot_global(prot_enhanced.mean(dim=0))
        return self.fusion(torch.cat([mol_g, prot_g], dim=-1))


class MTLModel(nn.Module):
    def __init__(self, tasks: List[str]):
        super().__init__()
        self.tasks = tasks
        self.mol_encoder = MoleculeEncoder()
        self.prot_encoder = ProteinEncoder()
        self.fusion = CrossModalFusion(mol_feat_dim=ModelConfig.hidden_dim, prot_feat_dim=ModelConfig.hidden_dim, hidden_dim=ModelConfig.hidden_dim)
        self.heads = nn.ModuleDict()
        for task in tasks:
            self.heads[task] = nn.Sequential(
                nn.Linear(ModelConfig.hidden_dim, ModelConfig.hidden_dim),
                nn.LayerNorm(ModelConfig.hidden_dim),
                nn.GELU(),
                nn.Dropout(ModelConfig.dropout_rate),
                nn.Linear(ModelConfig.hidden_dim, ModelConfig.num_classes),
            )
        self.scaler = torch.amp.GradScaler(DEVICE_TYPE, enabled=ModelConfig.use_amp)

    def forward(self, mol_data, prot_data, task):
        with torch.amp.autocast(DEVICE_TYPE, enabled=ModelConfig.use_amp):
            mol_feat = self.mol_encoder(mol_data)
            prot_feat = self.prot_encoder(prot_data)
            joint_feat = self.fusion(mol_feat, prot_feat)
            output = self.heads[task](joint_feat)
        return output


class TrainingMonitor:
    def __init__(self):
        self.grad_history = []
        self.nan_count = 0
    def check_gradients(self, model):
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                    self.nan_count += 1
        total_norm = total_norm ** 0.5
        self.grad_history.append(total_norm)
        if self.nan_count > 5:
            raise RuntimeError("Continuous gradient NaN detected")


class Trainer:
    def __init__(self, model: MTLModel, train_dataset: Dataset, val_dataset: Dataset):
        self.device = ModelConfig.device
        self.model = model.to(self.device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.monitor = TrainingMonitor()
        self.optimizer = optim.AdamW(model.parameters(), lr=ModelConfig.learning_rate, weight_decay=ModelConfig.weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="max", factor=0.5, patience=5, verbose=True)
        self.scaler = torch.amp.GradScaler(DEVICE_TYPE, enabled=ModelConfig.use_amp)
        self.best_metrics = {"balanced_accuracy": 0.0}
        self.early_stop_counter = 0
        self.history = defaultdict(list)
        self.train_loader = DataLoader(train_dataset, batch_size=ModelConfig.batch_size, shuffle=True, collate_fn=self._collate_fn, num_workers=0)
        self.val_loader = DataLoader(val_dataset, batch_size=ModelConfig.batch_size, shuffle=False, collate_fn=self._collate_fn, num_workers=0)

    def _collate_fn(self, batch):
        batch = [b for b in batch if b is not None]
        mol_feats = [b["mol_data"].cpu() if b["mol_data"].is_cuda else b["mol_data"] for b in batch]
        prot_feats = [b["prot_data"].cpu() if b["prot_data"].is_cuda else b["prot_data"] for b in batch]
        labels = torch.stack([b["label"] for b in batch])
        tasks = [b["task"] for b in batch]
        return {"mol_feat": mol_feats, "prot_feat": prot_feats, "label": labels, "tasks": tasks}

    def _compute_loss(self, outputs, labels, tasks):
        return F.cross_entropy(outputs, labels)

    def train_step(self, batch):
        with torch.amp.autocast(DEVICE_TYPE, enabled=ModelConfig.use_amp):
            outputs = []
            for i in range(len(batch["tasks"])):
                mol_data = batch["mol_feat"][i].to(self.device)
                prot_data = batch["prot_feat"][i].to(self.device)
                out = self.model(mol_data=mol_data, prot_data=prot_data, task=batch["tasks"][i])
                outputs.append(out)
            outputs = torch.stack(outputs)
            loss = self._compute_loss(outputs, batch["label"].to(self.device), batch["tasks"])
            if torch.isnan(loss):
                return None
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), ModelConfig.grad_clip)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        return loss.item()

    def train_epoch(self):
        self.model.train()
        total_loss, valid_steps = 0.0, 0
        for batch in tqdm(self.train_loader, desc="Training"):
            if batch is None: continue
            self.monitor.check_gradients(self.model)
            loss = self.train_step(batch)
            if loss is not None:
                total_loss += loss
                valid_steps += 1
        return total_loss / valid_steps if valid_steps > 0 else float("nan")

    def evaluate(self, dataloader=None):
        if dataloader is None:
            dataloader = self.val_loader
        self.model.eval()
        all_preds, all_labels, all_tasks = [], [], []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                if batch is None: continue
                for i in range(len(batch["tasks"])):
                    mol_feats = batch["mol_feat"][i].to(self.device)
                    prot_feats = batch["prot_feat"][i].to(self.device)
                    out = self.model(mol_data=mol_feats, prot_data=prot_feats, task=batch["tasks"][i])
                    all_preds.append(out.cpu())
                    all_labels.append(batch["label"][i].cpu().item())
                    all_tasks.append(batch["tasks"][i])
        y_true = np.array(all_labels)
        y_pred = torch.stack(all_preds).numpy()
        global_metrics = self._compute_metrics(y_true, y_pred)
        task_metrics = {}
        for task in set(all_tasks):
            mask = np.array(all_tasks) == task
            if mask.sum() == 0: continue
            task_y_true, task_y_pred = y_true[mask], y_pred[mask]
            if len(np.unique(task_y_true)) < 2: continue
            task_metrics[task] = self._compute_metrics(task_y_true, task_y_pred)
            task_metrics[task]["support"] = mask.sum()
        return {"global": global_metrics, "tasks": task_metrics}

    def _compute_metrics(self, y_true, y_pred):
        metrics = {}
        unique_classes = np.unique(y_true)
        label_map = {c: i for i, c in enumerate(sorted(unique_classes))}
        y_true_mapped = np.array([label_map[c] for c in y_true])
        y_pred_class = np.argmax(y_pred, axis=1)
        metrics["accuracy"] = accuracy_score(y_true_mapped, y_pred_class)
        metrics["balanced_accuracy"] = balanced_accuracy_score(y_true_mapped, y_pred_class)
        metrics["macro_f1"] = f1_score(y_true_mapped, y_pred_class, average="macro")
        metrics["confusion_matrix"] = confusion_matrix(y_true_mapped, y_pred_class)
        try:
            if len(unique_classes) > 2:
                y_onehot = label_binarize(y_true_mapped, classes=list(range(len(unique_classes))))
                metrics["roc_auc_ovr"] = roc_auc_score(y_onehot, y_pred, multi_class="ovr", average="macro")
            else:
                metrics["roc_auc"] = roc_auc_score(y_true_mapped, y_pred[:, 1])
        except Exception:
            metrics["roc_auc_ovr"] = float("nan")
        return metrics

    def save_model(self, epoch, metrics):
        checkpoint = {"epoch": epoch, "model_state_dict": self.model.state_dict(), "optimizer_state_dict": self.optimizer.state_dict(), "metrics": metrics}
        torch.save(checkpoint, os.path.join(ModelConfig.save_dir, "best_model.pth"))

    def train(self):
        ModelConfig.setup_dirs()
        for epoch in range(ModelConfig.epochs):
            logging.info(f"\nEpoch {epoch+1}/{ModelConfig.epochs}")
            start = time.time()
            train_loss = self.train_epoch()
            self.history["train_loss"].append(train_loss)
            val_metrics = self.evaluate()
            self.history["val_balanced_accuracy"].append(val_metrics["global"]["balanced_accuracy"])
            self.history["val_macro_f1"].append(val_metrics["global"]["macro_f1"])
            logging.info(f"Time: {time.time()-start:.1f}s | Train Loss: {train_loss:.4f} | Val BAcc: {val_metrics['global']['balanced_accuracy']:.4f} | Val Macro F1: {val_metrics['global']['macro_f1']:.4f}")
            self.scheduler.step(val_metrics["global"]["balanced_accuracy"])
            if val_metrics["global"]["balanced_accuracy"] > self.best_metrics["balanced_accuracy"]:
                self.best_metrics = val_metrics["global"]
                self.early_stop_counter = 0
                self.save_model(epoch, val_metrics)
                logging.info("🌟 New best model saved!")
            else:
                self.early_stop_counter += 1
                if self.early_stop_counter >= ModelConfig.early_stop_patient:
                    logging.info(f"🛑 Early stopping at epoch {epoch+1}")
                    break


def run_single_experiment(scenario, seed, train_csv, test_csv):
    torch.manual_seed(seed); np.random.seed(seed)
    if torch.cuda.is_available():
        device_index = ModelConfig.device.index if ModelConfig.device.index is not None else 0
        torch.cuda.set_device(device_index)
        torch.cuda.manual_seed_all(seed)
    train_dataset = create_dataset_from_csv(train_csv, FEATURE_CACHE_DIR)
    test_dataset = create_dataset_from_csv(test_csv, FEATURE_CACHE_DIR)
    n_val = max(1, int(len(train_dataset) * 0.1))
    indices = np.random.permutation(len(train_dataset))
    val_dataset = AttributeSubset(train_dataset, indices[:n_val])
    train_dataset = AttributeSubset(train_dataset, indices[n_val:])
    tasks = list(train_dataset._scanned_tasks) or ["default"]
    model = MTLModel(tasks).to(ModelConfig.device)
    trainer = Trainer(model, train_dataset, val_dataset)
    trainer.train()
    test_loader = DataLoader(test_dataset, batch_size=ModelConfig.batch_size, shuffle=False, collate_fn=trainer._collate_fn, num_workers=0)
    test_metrics = trainer.evaluate(test_loader)
    return {
        "balanced_accuracy": test_metrics["global"]["balanced_accuracy"],
        "macro_f1": test_metrics["global"]["macro_f1"],
        "macro_auc": test_metrics["global"].get("roc_auc_ovr", float("nan"))
    }


def generate_latex_table(results):
    latex = ["\\begin{table}[htbp]", "\\centering", "\\caption{Performance across different data split scenarios (mean ± std).}", "\\label{tab:cold_start_results}", "\\begin{tabular}{lccc}", "\\hline", "Scenario & Balanced Accuracy & Macro F1 & Macro AUC \\\\", "\\hline"]
    names = {"drug_cs": "Drug Cold-Start", "target_cs": "Target Cold-Start", "pair_cs": "Pair Cold-Start"}
    for key, name in names.items():
        if key in results:
            r = results[key]
            latex.append(f"{name} & ${r['balanced_accuracy_mean']:.3f} \\pm {r['balanced_accuracy_std']:.3f}$ & ${r['macro_f1_mean']:.3f} \\pm {r['macro_f1_std']:.3f}$ & ${r['macro_auc_mean']:.3f} \\pm {r['macro_auc_std']:.3f}$ \\\\")
    latex.extend(["\\hline", "\\end{tabular}", "\\end{table}"])
    with open(os.path.join(RESULT_DIR, "results_table.tex"), "w") as f:
        f.write("\n".join(latex))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", nargs="+", default=["train_random"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[456])
    args = parser.parse_args()
    all_results = {}
    for scenario in args.scenarios:
        scenario_results = []
        for seed in args.seeds:
            logging.info(f"Running {scenario} seed {seed}")
            name_map = {"train_random": "random"}
            base = name_map[scenario]
            train_csv = os.path.join(SPLIT_DIR, f"train_{base}.csv")
            test_csv = os.path.join(SPLIT_DIR, f"test_{base}.csv")
            metrics = run_single_experiment(scenario, seed, train_csv, test_csv)
            metrics["seed"] = seed
            scenario_results.append(metrics)
            with open(os.path.join(RESULT_DIR, f"{scenario}_seed{seed}_metrics.json"), "w") as f:
                json.dump(metrics, f, indent=2)
        baccs = [r["balanced_accuracy"] for r in scenario_results]
        f1s = [r["macro_f1"] for r in scenario_results]
        aucs = [r["macro_auc"] for r in scenario_results]
        all_results[scenario] = {"balanced_accuracy_mean": np.mean(baccs), "balanced_accuracy_std": np.std(baccs), "macro_f1_mean": np.mean(f1s), "macro_f1_std": np.std(f1s), "macro_auc_mean": np.mean(aucs), "macro_auc_std": np.std(aucs), "per_seed": scenario_results}
        logging.info(f"{scenario} summary: BAcc = {np.mean(baccs):.4f} ± {np.std(baccs):.4f}")
    with open(os.path.join(RESULT_DIR, "all_scenarios_summary.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    generate_latex_table(all_results)


if __name__ == "__main__":
    main()