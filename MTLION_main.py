#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any
from torch_geometric.data import Data
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
import logging
import math
from torch.nn.utils import weight_norm, spectral_norm
import torch
from typing import Dict
import os
import os
import pickle
import logging
import torch
from torch.utils.data import Dataset
from typing import Dict, Any
from collections import defaultdict
from tqdm import tqdm
from torch_geometric.data import Data
import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from typing import Dict, List, Any
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time
from sklearn.metrics import (accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score, confusion_matrix)
from sklearn.preprocessing import label_binarize
from datetime import datetime
import os
import logging
import pandas as pd
import time
import sys
import torch.multiprocessing as mp
import json
from datetime import datetime
from torch_scatter import scatter_mean


class ModelConfig:
    # Device settings
    device: torch.device = torch.device('cuda:6' if torch.cuda.is_available() else 'cpu')
    strict_mode: bool = False
    use_amp = True  # Use Automatic Mixed Precision (AMP) for training
    
    # Feature dimensions
    mol_atom_dim: int = 75      # Dimension of atom features
    mol_bond_dim: int = 8      # Dimension of bond features
    prot_atom_dim: int = 6      # Dimension of protein atom features
    prot_seq_dim: int = 1280    # Dimension of protein sequence embeddings
    mol_seq_dim: int = 384    # Dimension of molecule sequence embeddings
    prot_bond_dim = 32  # Dimension of protein bond features
    
    # Model architecture
    hidden_dim: int = 512       # Base hidden dimension
    num_classes: int = 3        # Number of output classes
    dropout_rate: float = 0.1   # Dropout rate
    gnn_layers: int = 3         # Number of GNN layers
    num_heads: int = 8          # Number of attention heads
    
    # Training parameters
    learning_rates: Dict[str, float] = None  # Learning rates for different groups
    weight_decay: float = 1e-5   # Weight decay
    batch_sizes: Dict[str, int] = None      # Batch sizes for different groups
    grad_clip: float = 1.0       # Gradient clipping
    epochs: int = 100             # Total training epochs
    early_stop_patient: int = 30 # Early stopping patience
    
    # Path settings
    save_dir: str = 'MTLION_Fig_Seq/saved_models'  # Directory to save models
    log_dir: str = 'MTLION_Fig_Seq/logs'           # Directory for log files
    plot_dir: str = 'MTLION_Fig_Seq/plots'         # Directory for output plots
    
    # Label mapping
    label_mapping: Dict[int, str] = None  # Class label to name mapping

    TASK_LOSS_CONFIG = {
        "P2X": {
            "type": "focal",
            "alpha": [0.1, 0.5, 3.0],  # Enhanced focus on rare classes
            "gamma": 2.0
        },
        "Kir": {
            "type": "weighted_ce",
            "weight": [1.0, 0.5, 2.0]  # Adjusted class weights
        },
        "VGSC": {
            "type": "focal",
            "alpha": [0.2, 0.5, 5.0],  # Extremely imbalanced task
            "gamma": 2.0
        },
        "VGCC": {
            "type": "focal",
            "alpha": [0.2, 0.5, 5.0],  # Extremely imbalanced task
            "gamma": 2.0
        },
        "K2P": {
            "type": "focal",
            "alpha": [0.2, 0.5, 5.0],  # Extremely imbalanced task
            "gamma": 2.0
        },
        "default": {
            "type": "ce"  # Default to standard cross-entropy
        }
    }
    
    def __post_init__(self):
        # Initialize default learning rates
        if self.learning_rates is None:
            self.learning_rates = {        
                'base': 5e-5,
                'high': 1e-4, 
                'mid': 5e-5,     
                'low': 1e-5,     
                'tiny': 5e-6                                    
            }
        
        # Initialize default batch sizes
        if self.batch_sizes is None:
            self.batch_sizes = {
                'high_volume': 32,
                'mid_volume': 16,
                'low_volume': 8,
                'tiny_volume': 4
            }
        
        # Initialize label mapping
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
        
        # Get file list
        self.file_list = sorted([
            f for f in os.listdir(self.group_path)
            if f.endswith('.pkl') and ('batch_' in f or '_' in f.replace('.pkl', ''))
        ], key=str.lower)        
        # Pre-scan task distribution
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
                'group': sample['group']
            }
        except Exception as e:
            logging.error(f"Failed to load sample {file_path}: {str(e)}")
            return None

    def _scan_task_distribution(self):
        """Pre-scan tasks and label distribution with caching support"""
        cache_dir = os.path.join(self.group_path, ".cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f"{self.group}_task_distribution.pkl")
        
        # 1. Attempt to load cache
        try:
            if os.path.exists(cache_file):
                # Check if cache is expired (compare with latest data file modification time)
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
        
        # 2. Perform scan when no valid cache exists
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
        
        # 3. Save cache
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
    """Residual block with gradient checkpointing"""
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
    """Base Graph Neural Network layer"""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__(aggr='mean')  # Use mean aggregation
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
    """Complete Graph Neural Network encoder"""
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int, num_layers: int = 3):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        
        # Initial projection
        self.node_proj = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        self.edge_proj = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # Multi-layer graph convolutions
        self.layers = nn.ModuleList([
            self._build_gnn_layer(hidden_dim)
            for _ in range(num_layers)
        ])
        
        # Output normalization
        self.norm = nn.LayerNorm(hidden_dim)

    def _build_gnn_layer(self, hidden_dim):
        """Build single GNN layer"""
        return nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

    def forward(self, node_feats, edge_feats, edge_index, batch_index):
        # Feature projection
        x = self.node_proj(node_feats)
        edge_attr = self.edge_proj(edge_feats)
        
        # Multi-layer message passing
        row, col = edge_index
        for layer in self.layers:
            messages = torch.cat([x[row], edge_attr], dim=-1)
            messages = layer(messages)
            aggregated = scatter_mean(messages, col, dim=0, dim_size=x.size(0))
            x = x + aggregated  # Residual connection
        
        # Normalization
        x = self.norm(x)
        
        # Global pooling
        return scatter_mean(x, batch_index, dim=0)

class AttentionPooling(nn.Module):
    """Multi-head attention pooling layer"""
    def __init__(self, input_dim: int, output_dim: int, num_heads: int = 4):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        assert output_dim % num_heads == 0, "output_dim must be divisible by num_heads"
        self.head_dim = output_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Input projection layer (handles arbitrary input dimensions)
        self.input_proj = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
        
        # Attention weight matrices
        self.query = nn.Linear(output_dim, output_dim)
        self.key = nn.Linear(output_dim, output_dim)
        self.value = nn.Linear(output_dim, output_dim)
        
        # Output transformation
        self.proj = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        # Learnable pooling vector
        self.pool_vector = nn.Parameter(torch.randn(1, 1, output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Input: [batch_size, seq_len, input_dim] Output: [batch_size, output_dim]"""
        # Dimension projection
        x = self.input_proj(x)  # [B, L, output_dim]
        batch_size = x.size(0)
        
        # Generate query vector
        q = self.query(self.pool_vector).expand(batch_size, -1, -1)  # [B, 1, D_out]
        
        # Multi-head projection
        q = q.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, 1, D/H]
        k = self.key(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, L, D/H]
        v = self.value(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention calculation
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, H, 1, L]
        attn = attn.softmax(dim=-1)
        
        # Weighted aggregation
        out = (attn @ v).transpose(1, 2).reshape(batch_size, 1, -1)  # [B, 1, D_out]
        
        # Output transformation
        return self.proj(out).squeeze(1)

class MoleculeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Initialize configuration
        self.hidden_dim = ModelConfig.hidden_dim
        
        # 1. Atom feature encoder
        self.atom_encoder = nn.Sequential(
            nn.Linear(ModelConfig.mol_atom_dim, self.hidden_dim//2),
            nn.LayerNorm(self.hidden_dim//2),
            nn.GELU()
        )
        
        # 2. Bond feature encoder
        self.bond_encoder = nn.Sequential(
            nn.Linear(ModelConfig.mol_bond_dim, self.hidden_dim//4),
            nn.LayerNorm(self.hidden_dim//4),
            nn.GELU()
        )
        
        # 3. Spatial feature encoder
        self.spatial_encoder = nn.Sequential(
            nn.Linear(132, self.hidden_dim//2),
            nn.LayerNorm(self.hidden_dim//2),
            nn.GELU()
        )
        
        # 4. Graph Neural Network
        self.gnn = GNNEncoder(
            node_dim=self.hidden_dim//2,
            edge_dim=self.hidden_dim//2,
            hidden_dim=self.hidden_dim            
        )
        
        # 5. Sequence encoder
        self.seq_encoder = nn.Sequential(
            nn.Linear(ModelConfig.mol_seq_dim, self.hidden_dim),
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=self.hidden_dim,
                    nhead=8,
                    dim_feedforward=self.hidden_dim*4
                ),
                num_layers=2
            )
        )
        
        # 6. Feature fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(self.hidden_dim*2, self.hidden_dim),
            SafeResidualBlock(self.hidden_dim, self.hidden_dim)
        )

    def forward(self, mol_data):
        # Automatically generate batch index
        mol_data.atoinmol_index = self._create_batch_index(mol_data.atom_types)
        mol_data.atoinmol_index = mol_data.atoinmol_index.to(ModelConfig.device)

        # Validate batch index
        self._validate_batch_index(mol_data.atoinmol_index, mol_data.mol_x.size(0))
        mol_data.mol_edge_index = self._ensure_tensor(mol_data.mol_edge_index, 'edge_index')
        
        # Atom feature encoding
        atom_feats = self.atom_encoder(mol_data.mol_x)

        # Bond feature encoding
        bond_feats = self.bond_encoder(mol_data.mol_edge_attr)

        # Spatial feature processing
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
        
        # GNN processing        
        gnn_feats = self.gnn(
            node_feats=atom_feats,
            edge_feats=spatial_feats,
            edge_index=mol_data.mol_edge_index,
            batch_index=mol_data.atoinmol_index
        )
                
        # Molecule-level aggregation
        expanded_feats = gnn_feats[mol_data.atoinmol_index] 
        mol_feats = scatter_mean(
            src=expanded_feats,
            index=mol_data.atoinmol_index,
            dim=0
        )

        # Sequence feature processing
        seq_feats = self.seq_encoder(mol_data.mol_embedding).mean(dim=1)
        
        # Feature fusion
        combined = torch.cat([
            mol_feats,
            seq_feats.expand(mol_feats.size(0), -1)
        ], dim=-1)
        
        # Multi-modal fusion
        molfusionx = self.fusion(combined)
        return molfusionx

    def _create_batch_index(self, atom_types):
        # Input validation
        if not isinstance(atom_types, (list, tuple)) or len(atom_types) == 0:
            raise ValueError("atom_types must be a non-empty list or tuple")
        
        # Supported atom type check
        supported_atoms = {'H', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Ru', 'Br', 'I', 'Hg', 'B', 'Si', 'Na', 'K', 'Ca', 'Mg', 'Fe', 'Zn', 'Cu', 'Mn', 'Mo', 'Ni', 'Se', 'As', 'V', 'Al', 'Ti', 'Ge', 'Sn', 'Pb', 'Cd', 'W', 'Cr', 'Co', 'Li', 'Y', 'Zr', 'Nb', 'Ta', 'Re', 'Pd', 'Ag', 'Au', 'Pt', 'Rh', 'Ir'}
        invalid_atoms = set(atom_types) - supported_atoms
        if invalid_atoms:
            raise ValueError(f"Unsupported atom types: {invalid_atoms}. Supported types: {sorted(supported_atoms)}")
        
        # Create continuous numbering
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
        """Validate batch index validity"""
        if not isinstance(batch_index, torch.Tensor):
            batch_index = torch.as_tensor(batch_index, dtype=torch.long)
        assert batch_index.dim() == 1, "Batch index should be 1D"
        assert len(batch_index) == num_atoms, \
            f"Expected {num_atoms} indices, got {len(batch_index)}"
        assert batch_index.min() >= 0, "Negative batch indices"
    
    def _ensure_tensor(self, data, name):
        """Internal utility method: ensure input is tensor"""
        if isinstance(data, torch.Tensor):
            return data
        try:
            return torch.as_tensor(data, dtype=torch.long)
        except Exception as e:
            logging.error(f"Failed to convert {name} to tensor: {str(e)}")
            raise ValueError(f"Could not convert {name} to tensor. Got: {type(data)}")

class ProteinEncoder(nn.Module):
    """Integrated optimized protein encoder"""
    def __init__(self):
        super().__init__()
        # Atom feature encoding
        self.atom_encoder = nn.Sequential(
            nn.Linear(ModelConfig.prot_atom_dim, ModelConfig.hidden_dim // 2),
            nn.LayerNorm(ModelConfig.hidden_dim // 2),
            nn.GELU()
        )
        
        # Edge feature encoding
        self.edge_encoder = nn.Sequential(
            nn.Linear(ModelConfig.prot_bond_dim, ModelConfig.hidden_dim // 4),
            nn.LayerNorm(ModelConfig.hidden_dim // 4),
            nn.GELU()
        )
        
        # Spatial feature encoding
        self.spatial_encoder = nn.Sequential(
            nn.Linear(132, ModelConfig.hidden_dim // 2),  # dist,theta,phi,tau + edge
            nn.LayerNorm(ModelConfig.hidden_dim // 2),
            nn.GELU()
        )
        
        # Graph Neural Network
        self.gnn = GNNEncoder(
            node_dim=ModelConfig.hidden_dim // 2,
            edge_dim=ModelConfig.hidden_dim // 2,
            hidden_dim=ModelConfig.hidden_dim
        )
        
        # Sequence encoding
        self.seq_encoder = nn.Sequential(
            nn.Linear(ModelConfig.prot_seq_dim, ModelConfig.hidden_dim),
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=ModelConfig.hidden_dim,
                    nhead=8,
                    dim_feedforward=ModelConfig.hidden_dim * 4
                ),
                num_layers=2
            )
        )
        
        # Attention pooling
        self.fusion = AttentionPooling(
            input_dim=1024,
            output_dim=512, 
            num_heads=4
        )

    def forward(self, prot_data: Data) -> torch.Tensor:
        prot_data.pro_index = self._ensure_tensor(prot_data.pro_index)
        # Atom features
        atom_feats = self.atom_encoder(prot_data.prot_atom_feats_s)

        # Edge features
        edge_feats = self.edge_encoder(prot_data.prot_edge_feats_s)

        # Spatial feature concatenation
        spatial_feats = torch.cat([
            prot_data.pro_dist.unsqueeze(-1),
            prot_data.pro_theta.unsqueeze(-1),
            prot_data.pro_phi.unsqueeze(-1),
            prot_data.pro_tau.unsqueeze(-1),
            edge_feats
        ], dim=-1)
        
        # Spatial feature encoding
        spatial_feats = self.spatial_encoder(spatial_feats)

        # Graph Neural Network processing
        gnn_feats = self.gnn(
            node_feats=atom_feats,
            edge_feats=spatial_feats,
            batch_index=prot_data.pro_index,
            edge_index=prot_data.pro_edge_index
        )
        
        # Sequence feature processing
        seq_feats = self.seq_encoder(prot_data.pro_seq_embedding).mean(dim=1)

        expanded_feats = gnn_feats[prot_data.pro_index] 
        pro_feats = scatter_mean(
            src=expanded_feats,
            index=prot_data.pro_index,
            dim=0
        )        
        
        # Feature fusion
        combined = torch.cat([
            pro_feats,
            seq_feats.expand(pro_feats.size(0), -1)
        ], dim=-1)       
        
        # Multi-modal fusion
        try:
            profusionx = self.fusion(combined)
        except Exception as e:
            logging.error(f"Feature fusion failed: {str(e)}")
            raise
            
        return profusionx
    
    def _ensure_tensor(self, string_list, dtype=torch.long, device=None):
        device = device or ('cuda:6' if torch.cuda.is_available() else 'cpu')
        
        # Step-by-step conversion and validation
        numbers = []
        for s in string_list:
            s = s.strip()
            if not s.isdigit():
                raise ValueError(f"Illegal numeric string: '{s}'")
            num = int(s)
            if num < 0:
                raise ValueError(f"Number cannot be negative: {num}")
            numbers.append(num)
        
        # First create tensor on CPU
        tensor = torch.tensor(numbers, dtype=dtype, device='cpu')
        
        # Validate conversion result
        if (tensor < 0).any():
            bad_indices = (tensor < 0).nonzero().flatten()
            raise RuntimeError(
                f"Negative values found after conversion at positions: {bad_indices.tolist()}"
            )
        
        # Finally move to target device
        return tensor.to(device)

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
        attn = torch.softmax(q @ k.t() / (x.size(-1)**0.5), dim=-1)
        return attn @ v

class CrossModalFusion(nn.Module):
    def __init__(self, mol_feat_dim=512, prot_feat_dim=512, hidden_dim=512):
        super().__init__()
        # Molecule projection
        self.mol_proj = nn.Sequential(
            nn.Linear(mol_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Protein projection
        self.prot_proj = nn.Sequential(
            nn.Linear(prot_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Bidirectional attention
        self.mol_to_prot = CrossAttention(hidden_dim)
        self.prot_to_mol = CrossAttention(hidden_dim)
        
        # Global aggregation
        self.mol_global = nn.Linear(hidden_dim, hidden_dim)
        self.prot_global = nn.Linear(hidden_dim, hidden_dim)
        
        # Final fusion
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, mol_feat, prot_feat):
        """Corrected forward propagation"""
        
        # Projection
        mol = self.mol_proj(mol_feat)  # [N,512]
        prot = self.prot_proj(prot_feat)  # [M,512]
        
        # Bidirectional attention
        mol_enhanced = self.prot_to_mol(mol, prot)  
        prot_enhanced = self.mol_to_prot(prot, mol) 
         
        # Global representation
        mol_g = self.mol_global(mol_enhanced.mean(dim=0))  # [512]
        prot_g = self.prot_global(prot_enhanced.mean(dim=0))  # [512]

        # Symmetric fusion
        return self.fusion(torch.cat([mol_g, prot_g], dim=-1))  # [512]


class MTLModel(nn.Module):
    def __init__(self, task_groups: Dict[str, Dict[str, Any]]):
        super().__init__()
        self.task_groups = task_groups
        
        # Shared encoders
        self.mol_encoder = MoleculeEncoder()
        self.prot_encoder = ProteinEncoder()
        
        # Feature fusion module
        try:
            self.fusion = CrossModalFusion(
                mol_feat_dim=512,  # Molecule encoder output dimension
                prot_feat_dim=512,  # Protein encoder output dimension
                hidden_dim=512  # Uniformly use hidden_dim
            )
        except Exception as e:
            logging.error(f"Failed to initialize CrossModalFusion: {str(e)}")
            raise
        
        # Initialize task heads and hooks
        self._init_task_heads()
        self._register_hooks()
        self.scaler = torch.amp.GradScaler('cuda:6', enabled=ModelConfig.use_amp)

    def _init_task_heads(self):
        """Initialize task-specific heads"""
        self.group_layers = nn.ModuleDict()
        self.heads = nn.ModuleDict()
        
        # Sample statistics and capacity calculation (unchanged)
        group_counts = {
            'high_volume': 40471,
            'mid_volume': 8067,
            'low_volume': 2793,
            'tiny_volume': 682
        }
        max_count = max(group_counts.values())

        # Unified initial feature projection (shared by all groups)
        self.base_proj = nn.Sequential(
            SafeResidualBlock(ModelConfig.hidden_dim, ModelConfig.hidden_dim),
            nn.LayerNorm(ModelConfig.hidden_dim),
            nn.GELU()
        )

        # Group-specific processing
        for group, config in self.task_groups.items():
            count = group_counts[group]
            log_ratio = math.log10(count + 1) / math.log10(max_count + 1)
            capacity = 0.3 + 0.7 * log_ratio
            
            # Fix dim1 as 512, only dynamically calculate dim2
            dim1 = ModelConfig.hidden_dim  
            dim2 = int(dim1 * (0.3 + 0.2*capacity))  # 153~256 (when capacity=0.3~1.0)

            # Group-specific layers
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
                layers = [nn.Linear(ModelConfig.hidden_dim, dim1)]  # 512->512
                
                # Add group-specific constraints
                if group == 'mid_volume':
                    layers.extend([
                        nn.LayerNorm(dim1),
                        nn.Dropout(0.4)
                    ])
                elif group == 'low_volume':
                    layers.extend([
                        spectral_norm(nn.Linear(dim1, dim1)),  # 512->512
                        nn.Dropout(0.6)
                    ])
                else:  # tiny_volume
                    layers.extend([
                        weight_norm(nn.Linear(dim1, dim1)),  # 512->512
                        nn.Dropout(0.8)
                    ])
                
                self.group_layers[group] = nn.Sequential(*layers)

            # Task-specific head
            for task in config['tasks']:
                self.heads[task] = nn.Sequential(
                    nn.Linear(dim1, dim2),  # 512->dim2
                    nn.LayerNorm(dim2),
                    nn.GELU() if count > 1000 else nn.SELU(),
                    weight_norm(nn.Linear(dim2, ModelConfig.num_classes)) if count < 5000 else 
                    nn.Linear(dim2, ModelConfig.num_classes)
                )

        # Regularization configuration
        for group in self.task_groups:
            count = group_counts[group]
            for name, param in self.named_parameters():
                if group in name:
                    param.reg_lambda = 0.1 * (1 - math.log10(count+1)/math.log10(max_count+1))
                    param.sample_weight = count / max_count

    def _register_hooks(self):
        """Register shape checking hooks"""
        def shape_hook(module, input, output):
            logging.debug(f"{module.__class__.__name__} output shape: {output.shape}")
        
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.LayerNorm, nn.AdaptiveAvgPool1d)):
                module.register_forward_hook(shape_hook)

    def forward(self, mol_data, prot_data, task, group):
        with torch.amp.autocast('cuda:6', enabled=ModelConfig.use_amp):
            # Encode features
            mol_feat = self.mol_encoder(mol_data)
            prot_feat = self.prot_encoder(prot_data)

            # Feature fusion
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

# === Trainer Classes ===
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
    """Focal Loss for handling class imbalance"""
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
    """Weighted cross entropy loss"""
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
        
        # Ensure ModelConfig is properly initialized
        if not hasattr(ModelConfig, 'learning_rates') or ModelConfig.learning_rates is None:
            ModelConfig.__post_init__(ModelConfig)
            
        self.optimizer = self._create_optimizer()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='max',
            factor=0.5,
            patience=5,
            verbose=True
        )

        self.scaler = torch.amp.GradScaler('cuda:6', enabled=ModelConfig.use_amp)
        if ModelConfig.use_amp:
            logging.info("AMP GradScaler initialized")
        
        # Training state
        self.best_metrics = {'balanced_accuracy': 0.0}
        self.early_stop_counter = 0
        self.history = defaultdict(list)
        
        # Create data loaders
        self.train_loaders = self._create_data_loaders(train=True)
        self.val_loader = self._create_data_loaders(train=False)
        logging.info(f"Created {len(self.train_loaders)} training loaders and {len(self.val_loader)} validation loaders")
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer with parameter groups"""
        param_groups = [
            {'params': [], 'lr': lr, 'name': name}
            for name, lr in ModelConfig.learning_rates.items()
        ]
        
        # Group parameters by type
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
        """Create data loaders for training or validation"""
        datasets = self.train_datasets if train else self.val_datasets
        loaders = {}
        
        for group, dataset in datasets.items():
            try:
                if train:
                    # Create weighted sampler for training
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
                # Debug: Print the problematic sample
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
        """Custom collate function for batch processing"""
        try:
            # Filter out None samples
            batch = [b for b in batch if b is not None]
            
            # Stack features and labels
            mol_feats = [b['mol_data'].cpu() if b['mol_data'].is_cuda else b['mol_data'] for b in batch]
            prot_feats = [b['prot_data'].cpu() if b['prot_data'].is_cuda else b['prot_data'] for b in batch]
            
            # Stack labels
            labels = torch.stack([b['label'] for b in batch])
            
            # Get task and group info
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
        # Input validation
        assert isinstance(outputs, torch.Tensor), "Output must be Tensor"
        assert isinstance(labels, torch.Tensor), "Labels must be Tensor"
        assert isinstance(tasks, list), "Tasks must be list"
        
        # Device consistency check
        assert labels.device == outputs.device, "Device mismatch"
        
        # Task loss configuration
        TASK_LOSS_CONFIG = {
            'P2X': {'type': 'focal', 'alpha': [0.1, 0.5, 3.0], 'gamma': 2.0},
            'Kir': {'type': 'weighted', 'weight': [1.0, 0.5, 2.0]},
            'VGSC': {'type': 'focal', 'alpha': [0.2, 0.5, 5.0], 'gamma': 2.0},
            'default': {'type': 'ce'}
        }
        
        # Task group weights
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
            
            # Get task group weight
            group = next(
                g for g in self.model.task_groups 
                if task in self.model.task_groups[g]['tasks']
            )
            group_weight = task_weights[group]
            
            try:
                # Dynamic loss function selection
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
                        labels[i].unsqueeze(0)
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
        with torch.amp.autocast('cuda:6', enabled=ModelConfig.use_amp):
            # Forward propagation
            outputs = []
            for i in range(len(batch['tasks'])):
                # Add input protection
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
            
            # Check if loss is nan
            if torch.isnan(loss):
                logging.warning("Detected NaN loss, skipping this update")
                return None
        
        # Backward propagation
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=ModelConfig.grad_clip,
            norm_type=2.0
        )
        
        # Parameter update
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
                    # Try to recover model state
                    self._recover_from_nan()
                    continue
                    
        return total_loss / valid_steps if valid_steps > 0 else float('nan')

    def _recover_from_nan(self):
        """Recover from NaN state"""
        logging.warning("Attempting to recover from NaN state...")
        # 1. Reload last checkpoint
        checkpoint = torch.load(os.path.join(ModelConfig.save_dir, 'best_model_1230.pth'))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # 2. Reduce learning rate
        for g in self.optimizer.param_groups:
            g['lr'] *= 0.5
        
        # 3. Reset monitoring state
        self.monitor.nan_count = 0  
    
    def evaluate(self, dataloader: DataLoader = None) -> Dict[str, Any]:
        """Evaluate model performance"""
        
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
                            
                            # Device transfer validation
                            mol_feats = batch['mol_feat'][i].to(self.device)
                            prot_feats = batch['prot_feat'][i].to(self.device)

                            # Safe forward propagation
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
                        
                        # Result collection
                        all_preds.extend(batch_outputs)
                        all_labels.extend(batch['label'].cpu().numpy().ravel())
                        all_tasks.extend(batch['tasks'])
                        
                    except Exception as e:
                        logging.error(f"Evaluation error - task {batch['tasks'][i] if 'tasks' in batch else 'N/A'}:\n"
                                    f"- Error details: {str(e)}")
                        continue
                        
        # Compute metrics
        y_true = np.array(all_labels)
    
        y_pred_uniform = [pred.squeeze() if pred.ndim == 2 else pred for pred in all_preds]
        y_pred_cpu = [pred.cpu() for pred in y_pred_uniform]
        y_pred = np.array(y_pred_cpu)
        
        global_metrics = self._compute_metrics(y_true, y_pred)
        
        # Compute per-task metrics
        task_metrics = {}
        for task in set(all_tasks):
            task_mask = np.array(all_tasks) == task
            if sum(task_mask) == 0:
                continue
                
            task_y_true = y_true[task_mask]
            task_y_pred = y_pred[task_mask]
                        
            if len(np.unique(task_y_true)) < 2:
                continue  # Skip single-class tasks
                
            task_metrics[task] = self._compute_metrics(task_y_true, task_y_pred)
            task_metrics[task]['support'] = len(task_y_true)
        
        return {
            'global': global_metrics,
            'tasks': task_metrics
        }
    
    def _compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Three-class model evaluation function"""
        metrics = {}
        unique_classes = np.unique(y_true)
        num_classes = len(unique_classes)
        
        # Convert labels to continuous indices
        label_map = {cls:i for i, cls in enumerate(sorted(unique_classes))}
        y_true_mapped = np.array([label_map[cls] for cls in y_true])
        
        # Calculate basic metrics
        y_pred_class = np.argmax(y_pred, axis=1)
        metrics.update({
            'accuracy': accuracy_score(y_true_mapped, y_pred_class),
            'balanced_accuracy': balanced_accuracy_score(y_true, y_pred_class),
            'macro_f1': f1_score(y_true, y_pred_class, average='macro'),
            'confusion_matrix': confusion_matrix(y_true_mapped, y_pred_class)
        })
        
        # Handle F1-score
        f1_scores = f1_score(y_true_mapped, y_pred_class, average=None)
        for i, cls in enumerate(unique_classes):
            metrics[f'f1_class{cls}'] = float(f1_scores[i])  
        metrics['f1_macro'] = float(np.mean(f1_scores))  
        
        # Multi-class AUC calculation
        try:
            # 1. Check and record NaN samples
            valid_mask = ~(np.isnan(y_pred).any(axis=1) | np.isnan(y_true_mapped))
            if not valid_mask.all():
                invalid_indices = np.where(~valid_mask)[0]
                
                # Prepare NaN sample recording
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
                
                # Filter invalid samples
                y_true_mapped = y_true_mapped[valid_mask]
                y_pred = y_pred[valid_mask]
                y_pred_class = y_pred_class[valid_mask]
                
                if len(y_true_mapped) == 0:
                    raise ValueError("No valid samples after filtering NaN")
            
            # 2. Calculate AUC
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
            # Record detailed debug information
            debug_info = {
                'y_true_sample': [float(x) for x in y_true[:3]] if len(y_true) > 0 else [],
                'y_pred_sample': [list(map(float, x)) for x in y_pred[:3]] if len(y_pred) > 0 else [],
                'num_classes': num_classes,
                'error': str(e)
            }
            logging.info(f"Debug information: {debug_info}")
            metrics['roc_auc'] = float('nan')
        
        # Add class distribution information (ensure serializable)
        metrics['class_distribution'] = {
            str(cls): int(sum(y_true == cls)) for cls in unique_classes
        }
        
        return metrics
    
    def save_model(self, epoch: int, metrics: Dict[str, Any]):
        """Improved model saving method, handling metric structure changes"""
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

        torch.save(checkpoint, os.path.join(ModelConfig.save_dir, 'best_model.pth'))

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
                        os.path.join(ModelConfig.log_dir, 'task_metrics.csv'),
                        index=False
                    )
            
            if hasattr(self, 'history') and self.history:
                history_df = pd.DataFrame(self.history)

                for col in ['val_balanced_accuracy', 'val_macro_f1']:
                    if col not in history_df.columns:
                        history_df[col] = None
                history_df.to_csv(
                    os.path.join(ModelConfig.log_dir, 'training_history.csv'),
                    index=False
                )

        except Exception as e:
            logging.error(f"Failed to save metric data: {str(e)}")

            with open(os.path.join(ModelConfig.log_dir, 'save_error.log'), 'a') as f:
                f.write(f"{datetime.now()} - {str(e)}\n")
                f.write(f"Metrics structure: {str(metrics.keys())}\n")

    def _get_serializable_config(self):
        """Get fully serializable configuration"""
        return {
            k: v for k, v in ModelConfig.__dict__.items()
            if not k.startswith('__')
            and not isinstance(v, (type, classmethod, staticmethod, property))
            and isinstance(v, (int, float, str, bool, list, dict, tuple, type(None)))
        }
    
    def plot_training_curves(self):
        """Plot training and validation curves"""
        plt.figure(figsize=(15, 5))
        
        # Loss curve
        plt.subplot(1, 3, 1)
        plt.plot(self.history['train_loss'], label='Train Loss')
        plt.title("Training Loss")
        plt.xlabel("Epoch")
        plt.legend()
        
        # Balanced accuracy
        plt.subplot(1, 3, 2)
        plt.plot(self.history['val_balanced_accuracy'], 
                color='orange', label='Val Balanced Accuracy')
        plt.title("Validation Accuracy")
        plt.xlabel("Epoch")
        plt.legend()
        
        # Macro F1
        plt.subplot(1, 3, 3)
        plt.plot(self.history['val_macro_f1'], 
                color='green', label='Val Macro F1')
        plt.title("Validation F1 Score")
        plt.xlabel("Epoch")
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(
            os.path.join(ModelConfig.plot_dir, 'training_curves.png'),
            bbox_inches='tight'
        )
        plt.close()
    
    def plot_confusion_matrix(self, cm: np.ndarray):
        """Plot normalized confusion matrix"""
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
            os.path.join(ModelConfig.plot_dir, 'confusion_matrix.png'),
            bbox_inches='tight'
        )
        plt.close()
    
    def train(self):
        """Full training loop"""
        ModelConfig.setup_dirs()
        
        for epoch in range(ModelConfig.epochs):
            logging.info(f"\nEpoch {epoch+1}/{ModelConfig.epochs}")
            start_time = time.time()
            
            # Training phase
            train_loss = self.train_epoch()
            self.history['train_loss'].append(train_loss)
            
            # Validation phase
            val_metrics = self.evaluate()
            self.history['val_balanced_accuracy'].append(
                val_metrics['global']['balanced_accuracy']
            )
            self.history['val_macro_f1'].append(
                val_metrics['global']['macro_f1']
            )
            
            # Print metrics
            logging.info(f"Time: {time.time()-start_time:.1f}s | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val BAcc: {val_metrics['global']['balanced_accuracy']:.4f} | "
                  f"Val Macro F1: {val_metrics['global']['macro_f1']:.4f}")
            
            # Learning rate scheduling
            self.scheduler.step(val_metrics['global']['balanced_accuracy'])
            
            # Early stopping and model saving
            if val_metrics['global']['balanced_accuracy'] > self.best_metrics['balanced_accuracy']:
                self.best_metrics = val_metrics['global']
                self.early_stop_counter = 0
                self.save_model(epoch, val_metrics)
                self.plot_training_curves()
                self.plot_confusion_matrix(val_metrics['global']['confusion_matrix'])
                logging.info("🌟 New best model saved!")
            else:
                self.early_stop_counter += 1
                if self.early_stop_counter >= ModelConfig.early_stop_patient:
                    logging.info(f"🛑 Early stopping at epoch {epoch+1}")
                    break

def main():
    # Initialize logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(ModelConfig.log_dir, 'training.log')),
            logging.StreamHandler()
        ]
    )
    
    try:
        # Data paths
        train_root = 'feature_cache_ESM2_train_New'
        val_root = 'feature_cache_ESM2_val_New'
        
        # Load datasets
        logging.info("Loading datasets...")
        train_datasets = {
            group: PKLGroupDataset(train_root, group)
            for group in ['high_volume', 'mid_volume', 'low_volume', 'tiny_volume']
        }
        val_datasets = {
            group: PKLGroupDataset(val_root, group)
            for group in ['high_volume', 'mid_volume', 'low_volume', 'tiny_volume']
        }
        
        # Build task groups
        task_groups = {}
        for group, dataset in train_datasets.items():
            # Get unique tasks
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
        
        # Initialize model
        logging.info("Initializing model...")
        model = MTLModel(task_groups)
        
        # Initialize trainer
        logging.info("Initializing trainer...")
        trainer = Trainer(model, train_datasets, val_datasets)
        
        # Start training
        logging.info("Starting training...")
        start_time = time.time()
        try:
            trainer.train()
        except Exception as e:
            logging.error("Training interrupted!", exc_info=True)
            
        training_time = time.time() - start_time
        
        # Generate final report
        logging.info("Generating final report...")
        report = generate_report(trainer)
        logging.info("\n=== Training Report ===\n" + report)
        
        # Save final model and metrics
        logging.info("Saving final model and metrics...")
        trainer.save_model(ModelConfig.epochs, trainer.best_metrics)
        
        # Log training summary
        logging.info(f"\n=== Training Summary ===")
        logging.info(f"Total training time: {training_time/60:.2f} minutes")
        logging.info(f"Best balanced accuracy: {trainer.best_metrics['balanced_accuracy']:.4f}")
        logging.info(f"Best macro F1: {trainer.best_metrics['macro_f1']:.4f}")
        
    except Exception as e:
        logging.error("Training failed!", exc_info=True)
        sys.exit(1)


def generate_report(trainer: Trainer) -> str:
    """Generate training report"""
    # Load metrics
    try:
        metrics_df = pd.read_csv(os.path.join(ModelConfig.log_dir, 'task_metrics.csv'))
        history_df = pd.read_csv(os.path.join(ModelConfig.log_dir, 'training_history.csv'))
    except Exception as e:
        logging.error(f"Failed to load metrics: {str(e)}")
        return "Error generating report"
    
    # Generate report sections
    report = [
        "="*50,
        "Training Report",
        f"Best Balanced Accuracy: {trainer.best_metrics['balanced_accuracy']:.4f}",
        f"Final Training Loss: {history_df['train_loss'].iloc[-1]:.4f}",
        "="*50,
        "\nTop Performing Tasks:"
    ]
    
    # Add top performing tasks
    top_tasks = metrics_df.sort_values('balanced_accuracy', ascending=False).head(5)
    for _, row in top_tasks.iterrows():
        report.append(
            f"{row['task']}: "
            f"Acc={row['accuracy']:.3f}, "
            f"BAcc={row['balanced_accuracy']:.3f}, "
            f"F1={row['macro_f1']:.3f}"
        )
    
    # Add training history summary
    report.extend([
        "\nTraining History:",
        f"Epochs completed: {len(history_df)}",
        f"Final validation F1: {history_df['val_macro_f1'].iloc[-1]:.4f}",
        f"Best validation F1: {history_df['val_macro_f1'].max():.4f}"
    ])    
    return "\n".join(report)


if __name__ == "__main__":
    mp.set_start_method('fork', force=True)
    
    # Run main function
    main()