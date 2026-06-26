#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List
from torch_geometric.data import Data
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
import logging
import math
from torch.nn.utils import weight_norm, spectral_norm
import os
import pickle
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    balanced_accuracy_score, 
    roc_auc_score, 
    confusion_matrix
)
from sklearn.preprocessing import label_binarize
from datetime import datetime
import sys
import torch.multiprocessing as mp
import json
from torch_scatter import scatter_mean


class ModelConfig:

    device: torch.device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')
    strict_mode: bool = False
    use_amp = True
    

    mol_atom_dim: int = 75      
    mol_bond_dim: int = 8    
    prot_atom_dim: int = 6    
    prot_seq_dim: int = 1280   
    mol_seq_dim: int = 384   
    prot_bond_dim = 32  
    
    hidden_dim: int = 384
    num_classes: int = 3
    dropout_rate: float = 0.5
    gnn_layers: int = 2
    num_heads: int = 8         
    

    learning_rates: Dict[str, float] = None 
    weight_decay: float = 1e-3
    batch_sizes: Dict[str, int] = None      
    grad_clip: float = 1.0       
    epochs: int = 100
    early_stop_patient: int = 7
    
    # Path settings
    save_dir: str = 'MTLION_Fig_Seq_20260422/saved_models'  
    log_dir: str = 'MTLION_Fig_Seq_20260422/logs'          
    plot_dir: str = 'MTLION_Fig_Seq_20260422/plots'     
    
    # Label mapping
    label_mapping: Dict[int, str] = None

    TASK_LOSS_CONFIG = {
        "P2X": {
            "type": "focal",
            "alpha": [0.1, 0.5, 3.0],
            "gamma": 2.0
        },
        "Kir": {
            "type": "weighted_ce",
            "weight": [1.0, 0.5, 2.0]
        },
        "VGSC": {
            "type": "focal",
            "alpha": [0.2, 0.5, 5.0],
            "gamma": 2.0
        },
        "VGCC": {
            "type": "focal",
            "alpha": [0.2, 0.5, 5.0],
            "gamma": 2.0
        },
        "VGKC": {
            "type": "focal",
            "alpha": [0.2, 0.5, 5.0],
            "gamma": 2.0
        },        
        "K2P": {
            "type": "focal",
            "alpha": [0.2, 0.5, 5.0],
            "gamma": 2.0
        },
        "default": {
            "type": "ce"
        }
    }
    
    def __post_init__(self):
        if self.learning_rates is None:
            self.learning_rates = {        
                'base': 5e-5,
                'high': 1e-4, 
                'mid': 5e-5,     
                'low': 1e-5,     
                'tiny': 5e-6                                    
            }
        
        if self.batch_sizes is None:
            self.batch_sizes = {
                'high_volume': 32,
                'mid_volume': 16,
                'low_volume': 8,
                'tiny_volume': 4
            }
        
        if self.label_mapping is None:
            self.label_mapping = {
                0: 'inactive',
                1: 'inhibitor', 
                2: 'activator'
            }
    
    @classmethod
    def setup_dirs(cls):
        os.makedirs(cls.save_dir, exist_ok=True)
        os.makedirs(cls.log_dir, exist_ok=True)
        os.makedirs(cls.plot_dir, exist_ok=True)
    
def create_defaultdict_int():
    return defaultdict(int)

class PKLGroupDataset(Dataset):
    def __init__(self, root_path: str, group: str):
        self.group_path = os.path.join(root_path, group)
        self.group = group
        self.task_label_counts = defaultdict(create_defaultdict_int) 
        self._scanned_tasks = set()
        
        self.file_list = sorted([
            f for f in os.listdir(self.group_path)
            if f.endswith('.pkl') and ('batch_' in f or '_' in f.replace('.pkl', ''))
        ], key=str.lower)
        
        self._scan_task_distribution()
    
    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        file_path = os.path.join(self.group_path, self.file_list[idx])
        try:
            with open(file_path, 'rb') as f:
                sample = pickle.load(f)          
            
            mol_data = sample['mol_fea']
            prot_data = sample['prot_fea']           
            
            mol_embedding = mol_data.mol_embedding
            if 'train' in self.group_path.lower():  
                noise = torch.randn_like(mol_embedding) * 0.01
                mol_embedding = mol_embedding + noise
            
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
                    mol_embedding=mol_embedding,
                    mol_smiles=sample['smiles']
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
                'task': sample['task'],
                'group': sample['group'],
                '_debug_file': file_path 
            }
        except Exception as e:
            logging.error(f"Failed to load sample {file_path}: {str(e)}")
            return None

    def _scan_task_distribution(self):
        cache_dir = os.path.join(self.group_path, ".cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f"{self.group}_task_distribution.pkl")
        
        try:
            if os.path.exists(cache_file):
                latest_data_mtime = max(os.path.getmtime(os.path.join(self.group_path, f)) 
                                        for f in self.file_list)
                cache_mtime = os.path.getmtime(cache_file)
                
                if cache_mtime > latest_data_mtime:
                    with open(cache_file, 'rb') as f:
                        cached_data = pickle.load(f)
                        self.task_label_counts = cached_data['counts']
                        self._scanned_tasks = cached_data['tasks']
                        logging.info(f"Loaded cached distribution for {self.group} (samples: {len(self.file_list)})")
                        return
        except Exception as e:
            logging.warning(f"Cache loading failed, will rescan: {str(e)}")
        
        num_samples = len(self.file_list)
        progress = tqdm(range(num_samples), desc=f"Scanning {self.group} tasks", leave=False)
        
        for idx in progress:
            file_path = os.path.join(self.group_path, self.file_list[idx])
            try:
                with open(file_path, 'rb') as f:
                    sample = pickle.load(f)

                if not all(field in sample for field in ['task', 'label']):
                    continue

                task = sample['task']
                label = sample['label']
                if torch.is_tensor(label):
                    label = label.item()
                
                self.task_label_counts[task][int(label)] += 1
                self._scanned_tasks.add(task)
            except Exception as e:
                logging.warning(f"Error processing {file_path}: {str(e)}")
                continue

        progress.close()
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'counts': self.task_label_counts,
                    'tasks': self._scanned_tasks
                }, f)
            logging.info(f"Cached distribution for {self.group} (samples: {num_samples})")
        except Exception as e:
            logging.warning(f"Failed to save cache: {str(e)}")

        logging.info(f"Scanned {num_samples} samples in {self.group} group")


class SafeResidualBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(ModelConfig.dropout_rate)
        self.shortcut = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)

        def _inner_forward(x):
            x = self.linear(x)
            x = torch.nan_to_num(x, nan=0.0, posinf=1e5, neginf=-1e5) 
            x = self.norm(x)
            x = torch.clamp(x, -10, 10)  
            x = self.activation(x)
            return self.dropout(x)
            
        x = checkpoint(_inner_forward, x)
        return residual + x


class GNNLayer(MessagePassing):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__(aggr='mean')
        self.message_net = nn.Sequential(
            nn.Linear(in_dim * 2, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU()
        )
        self.update_net = nn.Sequential(
            nn.Linear(in_dim + out_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU()
        )

    def forward(self, x, edge_index, edge_attr):
        edge_index, edge_attr = add_self_loops(
            edge_index, edge_attr, 
            fill_value=torch.mean(edge_attr, dim=0)
        )
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        return self.message_net(torch.cat([x_i, x_j, edge_attr], dim=-1))

    def update(self, aggr_out, x):
        return self.update_net(torch.cat([x, aggr_out], dim=-1))


class GNNEncoder(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int, num_layers: int = 3):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.dropout = nn.Dropout(ModelConfig.dropout_rate)

        self.node_proj = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(ModelConfig.dropout_rate)   
        )
        self.edge_proj = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(ModelConfig.dropout_rate)   
        )

        self.layers = nn.ModuleList([
            self._build_gnn_layer(hidden_dim)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(hidden_dim)

    def _build_gnn_layer(self, hidden_dim):
        return nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

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

    def __init__(self, input_dim: int, output_dim: int, num_heads: int = 4):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        assert output_dim % num_heads == 0, "output_dim must be divisible by num_heads"
        self.head_dim = output_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.input_proj = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
        
        self.query = nn.Linear(output_dim, output_dim)
        self.key = nn.Linear(output_dim, output_dim)
        self.value = nn.Linear(output_dim, output_dim)
        
        self.proj = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        self.pool_vector = nn.Parameter(torch.randn(1, 1, output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        
        self.atom_encoder = nn.Sequential(
            nn.Linear(ModelConfig.mol_atom_dim, self.hidden_dim//2),
            nn.LayerNorm(self.hidden_dim//2),
            nn.GELU()
        )
        
        self.bond_encoder = nn.Sequential(
            nn.Linear(ModelConfig.mol_bond_dim, self.hidden_dim//4),
            nn.LayerNorm(self.hidden_dim//4),
            nn.GELU()
        )

        spatial_input_dim = 4 + self.hidden_dim // 4
        self.spatial_encoder = nn.Sequential(
            nn.Linear(spatial_input_dim, self.hidden_dim // 2),
            nn.LayerNorm(self.hidden_dim // 2),
            nn.GELU()
        )

        self.gnn = GNNEncoder(
            node_dim=self.hidden_dim//2,
            edge_dim=self.hidden_dim//2,
            hidden_dim=self.hidden_dim,
            num_layers=ModelConfig.gnn_layers
        )
        
        self.seq_encoder = nn.Sequential(
            nn.Linear(ModelConfig.mol_seq_dim, self.hidden_dim),
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=self.hidden_dim,
                    nhead=8,
                    dim_feedforward=self.hidden_dim*4,
                    dropout=ModelConfig.dropout_rate,  
                    batch_first=True
                ),
                num_layers=2
            )
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(self.hidden_dim*2, self.hidden_dim),
            SafeResidualBlock(self.hidden_dim, self.hidden_dim)
        )

    def forward(self, mol_data):
        mol_data.atoinmol_index = self._create_batch_index(mol_data.atom_types)
        mol_data.atoinmol_index = mol_data.atoinmol_index.to(ModelConfig.device)

        self._validate_batch_index(mol_data.atoinmol_index, mol_data.mol_x.size(0))
        mol_data.mol_edge_index = self._ensure_tensor(mol_data.mol_edge_index, 'edge_index')
        
        atom_feats = self.atom_encoder(mol_data.mol_x)
        bond_feats = self.bond_encoder(mol_data.mol_edge_attr)

        try:
            spatial_input = torch.cat([
                mol_data.mol_dist.unsqueeze(-1),
                mol_data.mol_theta.unsqueeze(-1),
                mol_data.mol_phi.unsqueeze(-1),
                mol_data.mol_tau.unsqueeze(-1),
                bond_feats
            ], dim=-1)
            spatial_feats = self.spatial_encoder(spatial_input)
        except Exception as e:
            logging.error(f"Spatial feature concatenation failed: {str(e)}")
            raise
        
        gnn_feats = self.gnn(
            node_feats=atom_feats,
            edge_feats=spatial_feats,
            edge_index=mol_data.mol_edge_index,
            batch_index=mol_data.atoinmol_index
        )
                
        expanded_feats = gnn_feats[mol_data.atoinmol_index] 
        mol_feats = scatter_mean(
            src=expanded_feats,
            index=mol_data.atoinmol_index,
            dim=0
        )

        seq_feats = self.seq_encoder(mol_data.mol_embedding).mean(dim=1)
        
        combined = torch.cat([
            mol_feats,
            seq_feats.expand(mol_feats.size(0), -1)
        ], dim=-1)
        
        molfusionx = self.fusion(combined)

        return molfusionx

    def _create_batch_index(self, atom_types):
        if not isinstance(atom_types, (list, tuple)) or len(atom_types) == 0:
            raise ValueError("atom_types must be a non-empty list or tuple")
        
        supported_atoms = {'H', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Ru', 'Br', 'I', 'Hg', 'B', 'Si', 'Na', 'K', 'Ca', 'Mg', 'Fe', 'Zn', 'Cu', 'Mn', 'Mo', 'Ni', 'Se', 'As', 'V', 'Al', 'Ti', 'Ge', 'Sn', 'Pb', 'Cd', 'W', 'Cr', 'Co', 'Li', 'Y', 'Zr', 'Nb', 'Ta', 'Re', 'Pd', 'Ag', 'Au', 'Pt', 'Rh', 'Ir'}
        invalid_atoms = set(atom_types) - supported_atoms
        if invalid_atoms:
            raise ValueError(f"Unsupported atom types: {invalid_atoms}. Supported types: {sorted(supported_atoms)}")
        
        try:
            batch_index = torch.arange(
                start=0,
                end=len(atom_types),
                dtype=torch.long
            )
        except Exception as e:
            raise RuntimeError(f"Failed to create numbering tensor: {e}")
        
        return batch_index

    def _validate_batch_index(self, batch_index, num_atoms):
        if not isinstance(batch_index, torch.Tensor):
            batch_index = torch.as_tensor(batch_index, dtype=torch.long)
        assert batch_index.dim() == 1, "Batch index should be 1D"
        assert len(batch_index) == num_atoms, \
            f"Expected {num_atoms} indices, got {len(batch_index)}"
        assert batch_index.min() >= 0, "Negative batch indices"
    
    def _ensure_tensor(self, data, name):
        if isinstance(data, torch.Tensor):
            return data
        try:
            return torch.as_tensor(data, dtype=torch.long)
        except Exception as e:
            logging.error(f"Failed to convert {name} to tensor: {str(e)}")
            raise ValueError(f"Could not convert {name} to tensor. Got: {type(data)}")


class ProteinEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.atom_encoder = nn.Sequential(
            nn.Linear(ModelConfig.prot_atom_dim, ModelConfig.hidden_dim // 2),
            nn.LayerNorm(ModelConfig.hidden_dim // 2),
            nn.GELU()
        )
        
        self.edge_encoder = nn.Sequential(
            nn.Linear(ModelConfig.prot_bond_dim, ModelConfig.hidden_dim // 4),
            nn.LayerNorm(ModelConfig.hidden_dim // 4),
            nn.GELU()
        )
        
        spatial_input_dim = 4 + ModelConfig.hidden_dim // 4
        self.spatial_encoder = nn.Sequential(
            nn.Linear(spatial_input_dim, ModelConfig.hidden_dim // 2),
            nn.LayerNorm(ModelConfig.hidden_dim // 2),
            nn.GELU()
        )
        
        self.gnn = GNNEncoder(
            node_dim=ModelConfig.hidden_dim // 2,
            edge_dim=ModelConfig.hidden_dim // 2,
            hidden_dim=ModelConfig.hidden_dim,
            num_layers=ModelConfig.gnn_layers
        )
        
        self.seq_encoder = nn.Sequential(
            nn.Linear(ModelConfig.prot_seq_dim, ModelConfig.hidden_dim),
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=ModelConfig.hidden_dim,
                    nhead=8,
                    dim_feedforward=ModelConfig.hidden_dim * 4,
                    dropout=ModelConfig.dropout_rate,
                    batch_first=True
                ),
                num_layers=2
            )
        )
        
        self.fusion = AttentionPooling(
            input_dim=2 * ModelConfig.hidden_dim,  
            output_dim=ModelConfig.hidden_dim, 
            num_heads=4
        )

    def forward(self, prot_data: Data) -> torch.Tensor:
        debug_file = getattr(prot_data, '_debug_file', 'unknown')
        prot_data.pro_index = self._ensure_tensor(prot_data.pro_index, debug_file)
        atom_feats = self.atom_encoder(prot_data.prot_atom_feats_s)
        edge_feats = self.edge_encoder(prot_data.prot_edge_feats_s)

        spatial_feats = torch.cat([
            prot_data.pro_dist.unsqueeze(-1),
            prot_data.pro_theta.unsqueeze(-1),
            prot_data.pro_phi.unsqueeze(-1),
            prot_data.pro_tau.unsqueeze(-1),
            edge_feats
        ], dim=-1)
        
        spatial_feats = self.spatial_encoder(spatial_feats)

        gnn_feats = self.gnn(
            node_feats=atom_feats,
            edge_feats=spatial_feats,
            batch_index=prot_data.pro_index,
            edge_index=prot_data.pro_edge_index
        )
        
        seq_feats = self.seq_encoder(prot_data.pro_seq_embedding).mean(dim=1)

        expanded_feats = gnn_feats[prot_data.pro_index] 
        pro_feats = scatter_mean(
            src=expanded_feats,
            index=prot_data.pro_index,
            dim=0
        )        
        
        combined = torch.cat([
            pro_feats,
            seq_feats.expand(pro_feats.size(0), -1)
        ], dim=-1)       
        
        try:
            profusionx = self.fusion(combined)
        except Exception as e:
            logging.error(f"Feature fusion failed: {str(e)}")
            raise
            
        return profusionx
    
    def _ensure_tensor(self, string_list, debug_file='unknown', dtype=torch.long, device=None):
        device = device or ('cuda:7' if torch.cuda.is_available() else 'cpu')
        
        numbers = []
        for s in string_list:
            s = s.strip()
            if not (s.lstrip('-').isdigit() and (s.count('-') <= 1) and (s.find('-') <= 0)):
                raise ValueError(f"Illegal numeric string: '{s}'")
            num = int(s)
            if num < 0:
                logging.warning(f"Found negative pro_index: {num} in file {debug_file}")
            numbers.append(num)
        
        tensor = torch.tensor(numbers, dtype=dtype, device='cpu')
               
        return tensor.to(device)

from einops.layers.torch import Rearrange


class CrossAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        
    def forward(self, x, context):
        q = self.query(x)
        k = self.key(context)
        v = self.value(context)
        attn = torch.softmax(q @ k.transpose(-2, -1) / (x.size(-1)**0.5), dim=-1)
        return attn @ v



class CrossModalFusion(nn.Module):
    def __init__(self, mol_feat_dim=512, prot_feat_dim=512, hidden_dim=512):
        super().__init__()
        self.mol_proj = nn.Sequential(
            nn.Linear(mol_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        self.prot_proj = nn.Sequential(
            nn.Linear(prot_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        self.mol_to_prot = CrossAttention(hidden_dim)
        self.prot_to_mol = CrossAttention(hidden_dim)
        
        self.mol_global = nn.Linear(hidden_dim, hidden_dim)
        self.prot_global = nn.Linear(hidden_dim, hidden_dim)
        
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, mol_feat, prot_feat):
        mol = self.mol_proj(mol_feat)
        prot = self.prot_proj(prot_feat)
        
        mol_enhanced = self.prot_to_mol(mol, prot)  
        prot_enhanced = self.mol_to_prot(prot, mol) 
         
        mol_g = self.mol_global(mol_enhanced.mean(dim=0))
        prot_g = self.prot_global(prot_enhanced.mean(dim=0))

        return self.fusion(torch.cat([mol_g, prot_g], dim=-1))


class MTLModel(nn.Module):
    def __init__(self, task_groups: Dict[str, Dict[str, Any]]):
        super().__init__()
        self.task_groups = task_groups
        
        self.mol_encoder = MoleculeEncoder()
        self.prot_encoder = ProteinEncoder()
        
        try:
            self.fusion = CrossModalFusion(
                mol_feat_dim=ModelConfig.hidden_dim,
                prot_feat_dim=ModelConfig.hidden_dim,
                hidden_dim=ModelConfig.hidden_dim
            )
        except Exception as e:
            logging.error(f"Failed to initialize CrossModalFusion: {str(e)}")
            raise
        
        self._init_task_heads()
        self._register_hooks()
        self.scaler = torch.amp.GradScaler('cuda:7', enabled=ModelConfig.use_amp)

    def _init_task_heads(self):
        self.group_layers = nn.ModuleDict()
        self.heads = nn.ModuleDict()
        
        group_counts = {
            'high_volume': 40471,
            'mid_volume': 8067,
            'low_volume': 2793,
            'tiny_volume': 682
        }
        max_count = max(group_counts.values())

        self.base_proj = nn.Sequential(
            SafeResidualBlock(ModelConfig.hidden_dim, ModelConfig.hidden_dim),
            nn.LayerNorm(ModelConfig.hidden_dim),
            nn.GELU()
        )

        for group, config in self.task_groups.items():
            count = group_counts[group]
            log_ratio = math.log10(count + 1) / math.log10(max_count + 1)
            capacity = 0.3 + 0.7 * log_ratio
            
            dim1 = ModelConfig.hidden_dim  
            dim2 = int(dim1 * (0.3 + 0.2*capacity))

            if group == 'high_volume':
                self.group_layers[group] = nn.ModuleDict({
                    task: nn.Sequential(
                        AttentionPooling(ModelConfig.hidden_dim, dim1),
                        nn.Linear(dim1, dim2),
                        nn.Dropout(0.1)
                    )
                    for task in config['tasks']
                })
            else:
                layers = [nn.Linear(ModelConfig.hidden_dim, dim1)]
                
                if group == 'mid_volume':
                    layers.extend([
                        nn.LayerNorm(dim1),
                        nn.Dropout(0.4)
                    ])
                elif group == 'low_volume':
                    layers.extend([
                        spectral_norm(nn.Linear(dim1, dim1)),
                        nn.Dropout(0.6)
                    ])
                else:  # tiny_volume
                    layers.extend([
                        weight_norm(nn.Linear(dim1, dim1)),
                        nn.Dropout(0.8)
                    ])
                
                self.group_layers[group] = nn.Sequential(*layers)

            for task in config['tasks']:
                self.heads[task] = nn.Sequential(
                    nn.Linear(dim1, dim2),
                    nn.LayerNorm(dim2),
                    nn.GELU() if count > 1000 else nn.SELU(),
                    weight_norm(nn.Linear(dim2, ModelConfig.num_classes)) if count < 5000 else 
                    nn.Linear(dim2, ModelConfig.num_classes)
                )

        for group in self.task_groups:
            count = group_counts[group]
            for name, param in self.named_parameters():
                if group in name:
                    param.reg_lambda = 0.1 * (1 - math.log10(count+1)/math.log10(max_count+1))
                    param.sample_weight = count / max_count

    def _register_hooks(self):
        def shape_hook(module, input, output):
            logging.debug(f"{module.__class__.__name__} output shape: {output.shape}")
        
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.LayerNorm, nn.AdaptiveAvgPool1d)):
                module.register_forward_hook(shape_hook)

    def forward(self, mol_data, prot_data, task, group):
        with torch.amp.autocast('cuda:7', enabled=ModelConfig.use_amp):
            mol_feat = self.mol_encoder(mol_data)
            prot_feat = self.prot_encoder(prot_data)

            try:
                joint_feat = self.fusion(mol_feat, prot_feat) 
            except Exception as e:
                raise
            
            try:
                deal_jointfeat = self.heads[task](joint_feat)
            except Exception as e:
                logging.error(f"Task head processing failed: {str(e)}")
                raise
            
            return deal_jointfeat


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
            raise RuntimeError("Continuous gradient NaN detected, stopping training")


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)[targets]
            focal_loss = alpha * (1 - pt) ** self.gamma * ce_loss
        else:
            focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, reduction='mean'):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        return F.cross_entropy(inputs, targets, weight=self.weight, reduction=self.reduction)


class Trainer:
    def __init__(self, 
                 model: nn.Module,
                 train_datasets: Dict[str, Dataset],
                 val_datasets: Dict[str, Dataset]):
        self.device = ModelConfig.device
        self.model = model.to(self.device)
        self.train_datasets = train_datasets
        self.val_datasets = val_datasets
        self.monitor = TrainingMonitor()
        
        if not hasattr(ModelConfig, 'learning_rates') or ModelConfig.learning_rates is None:
            ModelConfig.__post_init__(ModelConfig)
            
        self.optimizer = self._create_optimizer()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='max',
            factor=0.5,
            patience=3,     
            verbose=True
        )

        self.scaler = torch.amp.GradScaler('cuda:7', enabled=ModelConfig.use_amp)
        if ModelConfig.use_amp:
            logging.info("AMP GradScaler initialized")
        
        self.best_metrics = {'balanced_accuracy': 0.0}
        self.early_stop_counter = 0
        self.history = defaultdict(list)
        
        self.train_loaders = self._create_data_loaders(train=True)
        self.val_loader = self._create_data_loaders(train=False)
        logging.info(f"Created {len(self.train_loaders)} training loaders and {len(self.val_loader)} validation loaders")
    
    def _create_optimizer(self) -> optim.Optimizer:
        param_groups = [
            {'params': [], 'lr': lr, 'name': name}
            for name, lr in ModelConfig.learning_rates.items()
        ]
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
                
            if 'mol_encoder' in name or 'prot_encoder' in name:
                param_groups[0]['params'].append(param)
            elif 'group_layers.high_volume' in name or any(
                f'heads.{task}' in name 
                for task in self.model.task_groups['high_volume']['tasks']
            ):
                param_groups[1]['params'].append(param)
            elif 'group_layers.mid_volume' in name or any(
                f'heads.{task}' in name 
                for task in self.model.task_groups['mid_volume']['tasks']
            ):
                param_groups[2]['params'].append(param)
            elif 'group_layers.low_volume' in name or any(
                f'heads.{task}' in name 
                for task in self.model.task_groups['low_volume']['tasks']
            ):
                param_groups[3]['params'].append(param)
            else:
                param_groups[4]['params'].append(param)
        
        return optim.AdamW(
            [g for g in param_groups if g['params']],
            weight_decay=ModelConfig.weight_decay,
            eps=1e-7
        )

    def _create_data_loaders(self, train: bool) -> Dict[str, DataLoader]:
        datasets = self.train_datasets if train else self.val_datasets
        loaders = {}
        
        for group, dataset in datasets.items():
            try:
                if train:
                    weights = [
                        1.0 / (dataset.task_label_counts[dataset[idx]['task']]
                            [dataset[idx]['label'].item()] + 1e-5)
                        for idx in range(len(dataset))
                    ]

                    sampler = WeightedRandomSampler(
                        weights=weights,
                        num_samples=len(dataset),
                        replacement=True
                    )
                else:
                    sampler = None
                
                loaders[group] = DataLoader(
                    dataset=dataset,
                    batch_size=ModelConfig.batch_sizes[group],
                    sampler=sampler,
                    shuffle=(not train and sampler is None),
                    num_workers=0,
                    collate_fn=self._collate_fn,
                    pin_memory=False,
                    persistent_workers=False
                )
            except Exception as e:
                logging.error(f"Failed to create {group} loader: {str(e)}")
                if hasattr(dataset, '__getitem__'):
                    for idx in range(min(5, len(dataset))):
                        sample = dataset[idx]
                        logging.error(f"Sample {idx} content: {sample}")
                if hasattr(ModelConfig, 'strict_mode') and ModelConfig.strict_mode:
                    raise
                else:
                    continue 
        
        return loaders
    
    def _collate_fn(self, batch: List[Dict]) -> Dict[str, Any]:
        try:
            batch = [b for b in batch if b is not None]
            
            mol_feats = [b['mol_data'].cpu() if b['mol_data'].is_cuda else b['mol_data'] for b in batch]
            prot_feats = [b['prot_data'].cpu() if b['prot_data'].is_cuda else b['prot_data'] for b in batch]
            
            labels = torch.stack([b['label'] for b in batch])
            
            tasks = [b['task'] for b in batch]
            groups = [b['group'] for b in batch]

            return {
                'mol_feat': mol_feats,
                'prot_feat': prot_feats,
                'label': labels,
                'tasks': tasks,
                'groups': groups
            }
        except Exception as e:
            logging.error(f"Batch collate error: {str(e)}")
            return None

    def _compute_loss(self, outputs: torch.Tensor, labels: torch.Tensor, tasks: List[str]) -> torch.Tensor:
        assert isinstance(outputs, torch.Tensor), "Output must be Tensor"
        assert isinstance(labels, torch.Tensor), "Labels must be Tensor"
        assert isinstance(tasks, list), "Tasks must be list"
        assert labels.device == outputs.device, "Device mismatch"
        
        TASK_LOSS_CONFIG = {
            'P2X': {'type': 'focal', 'alpha': [0.1, 0.5, 3.0], 'gamma': 2.0},
            'Kir': {'type': 'weighted', 'weight': [1.0, 0.5, 2.0]},
            'VGSC': {'type': 'focal', 'alpha': [0.2, 0.5, 5.0], 'gamma': 2.0},
            'default': {'type': 'ce'}
        }
        
        task_weights = {
            'high_volume': 1.0,
            'mid_volume': 1.5, 
            'low_volume': 2.0,
            'tiny_volume': 3.0
        }
        
        total_loss = 0.0
        valid_samples = 0
        
        for i, task in enumerate(tasks):
            config = TASK_LOSS_CONFIG.get(task, TASK_LOSS_CONFIG['default'])
            
            group = next(
                g for g in self.model.task_groups 
                if task in self.model.task_groups[g]['tasks']
            )
            group_weight = task_weights[group]
            
            try:
                if config['type'] == 'focal':
                    alpha = torch.tensor(config['alpha'], device=outputs.device)
                    loss = FocalLoss(alpha=alpha, gamma=config['gamma'])(
                        outputs[i].unsqueeze(0),
                        labels[i].unsqueeze(0)
                    )
                elif config['type'] == 'weighted':
                    weight = torch.tensor(config['weight'], device=outputs.device)
                    loss = F.cross_entropy(
                        outputs[i].unsqueeze(0),
                        labels[i].unsqueeze(0),
                        weight=weight
                    )
                else:

                    loss = F.cross_entropy(
                        outputs[i].unsqueeze(0),
                        labels[i].unsqueeze(0),
                        label_smoothing=0.1   
                    )
                
                total_loss += group_weight * loss
                valid_samples += 1
                
            except Exception as e:
                logging.error(f"Task {task} loss calculation failed: {str(e)}")
                continue
        
        if valid_samples == 0:
            raise ValueError("No valid samples for loss calculation")
        
        return total_loss / valid_samples

    def train_step(self, batch):
        with torch.amp.autocast('cuda:7', enabled=ModelConfig.use_amp):
            outputs = []
            for i in range(len(batch['tasks'])):
                mol_data = batch['mol_feat'][i].to(self.device)
                prot_data = batch['prot_feat'][i].to(self.device)
                
                output = self.model(
                    mol_data=mol_data,
                    prot_data=prot_data,
                    task=batch['tasks'][i],
                    group=batch['groups'][i]
                )
                              
                outputs.append(output)
                
            outputs = torch.stack(outputs)
            loss = self._compute_loss(outputs, batch['label'].to(self.device), batch['tasks'])
            
            if torch.isnan(loss):
                logging.warning("Detected NaN loss, skipping this update")
                return None
        
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=ModelConfig.grad_clip,
            norm_type=2.0
        )
        
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        
        return loss.item()
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        valid_steps = 0
        
        for loader in self.train_loaders.values():
            for batch in tqdm(loader, desc="Training"):
                if batch is None:
                    continue
                self.monitor.check_gradients(self.model)    
                try:
                    loss = self.train_step(batch)
                    if loss is not None:
                        total_loss += loss
                        valid_steps += 1
                except Exception as e:
                    logging.error(f"Training step failed: {str(e)}")
                    self._recover_from_nan()
                    continue
                    
        return total_loss / valid_steps if valid_steps > 0 else float('nan')

    def _recover_from_nan(self):
        logging.warning("Attempting to recover from NaN state...")
        checkpoint = torch.load(os.path.join(ModelConfig.save_dir, 'best_model_1230.pth'))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        for g in self.optimizer.param_groups:
            g['lr'] *= 0.5
        
        self.monitor.nan_count = 0  
    
    def evaluate(self, dataloader: DataLoader = None) -> Dict[str, Any]:
        
        self.model.eval()
        all_preds = []
        all_labels = []
        all_tasks = []

        for dataloader in self.val_loader.values():       
            with torch.no_grad():
                for batch in dataloader:
                    if batch is None:
                        continue
                                        
                    try:
                        batch_outputs = []
                        for i in range(len(batch['tasks'])):
                            
                            mol_feats = batch['mol_feat'][i].to(self.device)
                            prot_feats = batch['prot_feat'][i].to(self.device)

                            try:
                                with torch.no_grad():
                                    output = self.model(
                                        mol_data=mol_feats,
                                        prot_data=prot_feats,
                                        task=batch['tasks'][i],
                                        group=batch['groups'][i]
                                    )
                                    output = output.cpu()
                                    batch_outputs.append(output)
                                    
                            except Exception as e:
                                logging.error(f"- Output error details: {str(e)}")                               
                        
                        all_preds.extend(batch_outputs)
                        all_labels.extend(batch['label'].cpu().numpy().ravel())
                        all_tasks.extend(batch['tasks'])
                        
                    except Exception as e:
                        logging.error(f"Evaluation error - task {batch['tasks'][i] if 'tasks' in batch else 'N/A'}:\n"
                                    f"- Error details: {str(e)}")
                        continue
                        
        y_true = np.array(all_labels)
    
        y_pred_uniform = [pred.squeeze() if pred.ndim == 2 else pred for pred in all_preds]
        y_pred_cpu = [pred.cpu() for pred in y_pred_uniform]
        y_pred = np.array(y_pred_cpu)
        
        global_metrics = self._compute_metrics(y_true, y_pred)
        
        task_metrics = {}
        for task in set(all_tasks):
            task_mask = np.array(all_tasks) == task
            if sum(task_mask) == 0:
                continue
                
            task_y_true = y_true[task_mask]
            task_y_pred = y_pred[task_mask]
                        
            if len(np.unique(task_y_true)) < 2:
                continue
                
            task_metrics[task] = self._compute_metrics(task_y_true, task_y_pred)
            task_metrics[task]['support'] = len(task_y_true)
        
        return {
            'global': global_metrics,
            'tasks': task_metrics
        }
    
    def _compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        metrics = {}
        unique_classes = np.unique(y_true)
        num_classes = len(unique_classes)
        
        label_map = {cls:i for i, cls in enumerate(sorted(unique_classes))}
        y_true_mapped = np.array([label_map[cls] for cls in y_true])
        
        y_pred_class = np.argmax(y_pred, axis=1)
        metrics.update({
            'accuracy': accuracy_score(y_true_mapped, y_pred_class),
            'balanced_accuracy': balanced_accuracy_score(y_true, y_pred_class),
            'macro_f1': f1_score(y_true, y_pred_class, average='macro'),
            'confusion_matrix': confusion_matrix(y_true_mapped, y_pred_class)
        })
        
        f1_scores = f1_score(y_true_mapped, y_pred_class, average=None)
        for i, cls in enumerate(unique_classes):
            metrics[f'f1_class{cls}'] = float(f1_scores[i])  
        metrics['f1_macro'] = float(np.mean(f1_scores))  
        
        try:
            valid_mask = ~(np.isnan(y_pred).any(axis=1) | np.isnan(y_true_mapped))
            if not valid_mask.all():
                invalid_indices = np.where(~valid_mask)[0]
                
                nan_samples = []
                for idx in invalid_indices:
                    try:
                        sample_info = {
                            'index': int(idx),
                            'y_true': float(y_true[idx]),  
                            'y_pred': [float(x) for x in y_pred[idx]],  
                            'is_y_true_nan': bool(np.isnan(y_true[idx])),
                            'is_y_pred_nan': bool(np.isnan(y_pred[idx]).any())
                        }
                        nan_samples.append(sample_info)
                    except Exception as e:
                        logging.error(f"Error recording NaN sample (index {idx}): {str(e)}")
                        logging.error(f"Problem sample data - y_true: {y_true[idx]}, y_pred: {y_pred[idx]}")
                        continue
                
                log_path = os.path.join(ModelConfig.log_dir, 'nan_samples.json')
                try:
                    with open(log_path, 'a') as f:
                        json.dump({
                            'timestamp': datetime.now().isoformat(),
                            'total_nan_samples': len(invalid_indices),
                            'samples': nan_samples
                        }, f, default=str)  
                        f.write('\n')
                    logging.warning(f"Found {len(invalid_indices)} NaN samples, recorded to {log_path}")
                except Exception as e:
                    logging.error(f"Failed to write NaN sample log: {str(e)}")
                
                y_true_mapped = y_true_mapped[valid_mask]
                y_pred = y_pred[valid_mask]
                y_pred_class = y_pred_class[valid_mask]
                
                if len(y_true_mapped) == 0:
                    raise ValueError("No valid samples after filtering NaN")
            
            if num_classes > 2:
                y_true_onehot = label_binarize(y_true_mapped, classes=[0,1,2])
                metrics['roc_auc_ovr'] = float(roc_auc_score(
                    y_true_onehot, y_pred,
                    multi_class='ovr',
                    average='macro'
                ))
            else:
                metrics['roc_auc'] = float(roc_auc_score(
                    y_true_mapped, 
                    y_pred[:, 1]  
                ))
                
        except Exception as e:
            logging.error(f"AUC calculation failed: {str(e)}")
            debug_info = {
                'y_true_sample': [float(x) for x in y_true[:3]] if len(y_true) > 0 else [],
                'y_pred_sample': [list(map(float, x)) for x in y_pred[:3]] if len(y_pred) > 0 else [],
                'num_classes': num_classes,
                'error': str(e)
            }
            logging.info(f"Debug information: {debug_info}")
            metrics['roc_auc'] = float('nan')
        
        metrics['class_distribution'] = {
            str(cls): int(sum(y_true == cls)) for cls in unique_classes
        }
        
        return metrics
    
    def save_model(self, epoch: int, metrics: Dict[str, Any]):
        task_metrics = metrics.get('tasks', {})

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_metrics': {
                'balanced_accuracy': metrics.get('balanced_accuracy', 0.0),
                'macro_f1': metrics.get('macro_f1', 0.0)
            },
            'config': self._get_serializable_config()
        }

        torch.save(checkpoint, os.path.join(ModelConfig.save_dir, 'best_model_20260413.pth'))

        try:
            if task_metrics:
                metrics_data = []
                for task_name, task_data in task_metrics.items():
                    if isinstance(task_data, dict):
                        row = {'task': task_name}
                        row.update({k: v for k, v in task_data.items() 
                                if isinstance(v, (int, float, str))})
                        metrics_data.append(row)
                
                if metrics_data:
                    pd.DataFrame(metrics_data).to_csv(
                        os.path.join(ModelConfig.log_dir, 'task_metrics_20260413.csv'),
                        index=False
                    )
            
            if hasattr(self, 'history') and self.history:
                history_df = pd.DataFrame(self.history)

                for col in ['val_balanced_accuracy', 'val_macro_f1']:
                    if col not in history_df.columns:
                        history_df[col] = None
                history_df.to_csv(
                    os.path.join(ModelConfig.log_dir, 'training_history_20260413.csv'),
                    index=False
                )

        except Exception as e:
            logging.error(f"Failed to save metric data: {str(e)}")

            with open(os.path.join(ModelConfig.log_dir, 'save_error.log'), 'a') as f:
                f.write(f"{datetime.now()} - {str(e)}\n")
                f.write(f"Metrics structure: {str(metrics.keys())}\n")

    def _get_serializable_config(self):
        return {
            k: v for k, v in ModelConfig.__dict__.items()
            if not k.startswith('__')
            and not isinstance(v, (type, classmethod, staticmethod, property))
            and isinstance(v, (int, float, str, bool, list, dict, tuple, type(None)))
        }
    
    def plot_training_curves(self):
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(self.history['train_loss'], label='Train Loss')
        plt.title("Training Loss")
        plt.xlabel("Epoch")
        plt.legend()
        
        plt.subplot(1, 3, 2)
        plt.plot(self.history['val_balanced_accuracy'], 
                color='orange', label='Val Balanced Accuracy')
        plt.title("Validation Accuracy")
        plt.xlabel("Epoch")
        plt.legend()
        
        plt.subplot(1, 3, 3)
        plt.plot(self.history['val_macro_f1'], 
                color='green', label='Val Macro F1')
        plt.title("Validation F1 Score")
        plt.xlabel("Epoch")
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(
            os.path.join(ModelConfig.plot_dir, 'training_curves_20260413.png'),
            bbox_inches='tight'
        )
        plt.close()
    
    def plot_confusion_matrix(self, cm: np.ndarray):
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=ModelConfig.label_mapping.values(),
            yticklabels=ModelConfig.label_mapping.values()
        )
        plt.title("Normalized Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.savefig(
            os.path.join(ModelConfig.plot_dir, 'confusion_matrix_20260413.png'),
            bbox_inches='tight'
        )
        plt.close()
    
    def train(self):
        ModelConfig.setup_dirs()
        
        for epoch in range(ModelConfig.epochs):
            logging.info(f"\nEpoch {epoch+1}/{ModelConfig.epochs}")
            start_time = time.time()
            
            train_loss = self.train_epoch()
            self.history['train_loss'].append(train_loss)
            
            val_metrics = self.evaluate()
            self.history['val_balanced_accuracy'].append(
                val_metrics['global']['balanced_accuracy']
            )
            self.history['val_macro_f1'].append(
                val_metrics['global']['macro_f1']
            )
            
            logging.info(f"Time: {time.time()-start_time:.1f}s | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val BAcc: {val_metrics['global']['balanced_accuracy']:.4f} | "
                  f"Val Macro F1: {val_metrics['global']['macro_f1']:.4f}")
            
            self.scheduler.step(val_metrics['global']['balanced_accuracy'])
            
            if val_metrics['global']['balanced_accuracy'] > self.best_metrics['balanced_accuracy']:
                self.best_metrics = val_metrics['global']
                self.early_stop_counter = 0
                self.save_model(epoch, val_metrics)
                self.plot_training_curves()
                self.plot_confusion_matrix(val_metrics['global']['confusion_matrix'])
                logging.info("New best model saved!")
            else:
                self.early_stop_counter += 1
                if self.early_stop_counter >= ModelConfig.early_stop_patient:
                    logging.info(f"Early stopping at epoch {epoch+1}")
                    break


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(ModelConfig.log_dir, 'training_20260413.log')),
            logging.StreamHandler()
        ]
    )
    
    try:
        train_root = 'feature_ESM2_train_New_20260326_R42_Real_OPT'
        val_root = 'feature_ESM2_val_New_20260326_R42_Real_OPT'
        
        logging.info("Loading datasets...")
        train_datasets = {
            group: PKLGroupDataset(train_root, group)
            for group in ['high_volume', 'mid_volume', 'low_volume', 'tiny_volume']
        }
        val_datasets = {
            group: PKLGroupDataset(val_root, group)
            for group in ['high_volume', 'mid_volume', 'low_volume', 'tiny_volume']
        }
        
        task_groups = {}
        for group, dataset in train_datasets.items():
            tasks = set()
            for idx in range(len(dataset)):
                sample = dataset[idx]
                if sample is not None:
                    tasks.add(sample['task'])
            
            task_groups[group] = {
                'tasks': list(tasks) if tasks else [f'default_{group}'],
                'class_weights': dataset.task_label_counts
            }
            logging.info(f"Group {group} tasks: {task_groups[group]['tasks']}")
        
        logging.info("Initializing model...")
        model = MTLModel(task_groups)
        
        logging.info("Initializing trainer...")
        trainer = Trainer(model, train_datasets, val_datasets)
        
        logging.info("Starting training...")
        start_time = time.time()
        try:
            trainer.train()
        except Exception as e:
            logging.error("Training interrupted!", exc_info=True)
            
        training_time = time.time() - start_time
        
        logging.info("Generating final report...")
        report = generate_report(trainer)
        logging.info("\n=== Training Report ===\n" + report)
        
        logging.info("Saving final model and metrics...")
        trainer.save_model(ModelConfig.epochs, trainer.best_metrics)
        
        logging.info(f"\n=== Training Summary ===")
        logging.info(f"Total training time: {training_time/60:.2f} minutes")
        logging.info(f"Best balanced accuracy: {trainer.best_metrics['balanced_accuracy']:.4f}")
        logging.info(f"Best macro F1: {trainer.best_metrics['macro_f1']:.4f}")
        
    except Exception as e:
        logging.error("Training failed!", exc_info=True)
        sys.exit(1)


def generate_report(trainer: Trainer) -> str:
    try:
        metrics_df = pd.read_csv(os.path.join(ModelConfig.log_dir, 'task_metrics_20260413.csv'))
        history_df = pd.read_csv(os.path.join(ModelConfig.log_dir, 'training_history_20260413.csv'))
    except Exception as e:
        logging.error(f"Failed to load metrics: {str(e)}")
        return "Error generating report"
    
    report = [
        "="*50,
        "Training Report",
        f"Best Balanced Accuracy: {trainer.best_metrics['balanced_accuracy']:.4f}",
        f"Final Training Loss: {history_df['train_loss'].iloc[-1]:.4f}",
        "="*50,
        "\nTop Performing Tasks:"
    ]
    
    top_tasks = metrics_df.sort_values('balanced_accuracy', ascending=False).head(5)
    for _, row in top_tasks.iterrows():
        report.append(
            f"{row['task']}: "
            f"Acc={row['accuracy']:.3f}, "
            f"BAcc={row['balanced_accuracy']:.3f}, "
            f"F1={row['macro_f1']:.3f}"
        )
    
    report.extend([
        "\nTraining History:",
        f"Epochs completed: {len(history_df)}",
        f"Final validation F1: {history_df['val_macro_f1'].iloc[-1]:.4f}",
        f"Best validation F1: {history_df['val_macro_f1'].max():.4f}"
    ])
    
    return "\n".join(report)


if __name__ == "__main__":
    mp.set_start_method('fork', force=True)
    main()