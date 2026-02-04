import os
import pickle
from src.config import GlobalConfig
import torch
import logging
from torch_geometric.data import Data
import hashlib
from esm.pretrained import esm2_t33_650M_UR50D
from torch_scatter import scatter_min
from rdkit import Chem
from rdkit.Chem import AllChem
from gvp import data
from Bio.PDB import PDBParser
import dgl
import networkx as nx
import math
from rdkit.Chem import BondType
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
from rdkit.Chem.MolStandardize.rdMolStandardize import TautomerEnumerator
from rdkit.Chem import HybridizationType
import time
from transformers import AutoModel, AutoTokenizer
import pandas as pd
from typing import Dict
from tqdm import tqdm
import traceback
from datetime import datetime
import sys

# ==================== Error Handling and Validation Classes ====================

class FeatureErrorHandler:
    """Class for handling feature generation errors"""
    
    def __init__(self, error_log_path):
        self.error_log_path = error_log_path
        self.failed_samples = []
        self._init_error_log()
    
    def _init_error_log(self):
        """Initialize error log file"""
        if not os.path.exists(self.error_log_path):
            error_df = pd.DataFrame(columns=[
                'timestamp', 'index', 'smiles', 'uniprot_id', 
                'task', 'group', 'error_type', 'error_message'
            ])
            error_df.to_csv(self.error_log_path, index=False)
    
    def record_failure(self, index, row, error_type, error_msg):
        """Record failed sample"""
        error_record = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'index': index,
            'smiles': row.get('SMILES', ''),
            'uniprot_id': row.get('UniProt ID', ''),
            'task': row.get('ABB', ''),
            'group': '',
            'error_type': error_type,
            'error_message': str(error_msg)[:500]
        }
        self.failed_samples.append(error_record)
        
        # Periodically save to file
        if len(self.failed_samples) >= 10:
            self.save_errors()
    
    def save_errors(self):
        """Save error records to file"""
        if self.failed_samples:
            error_df = pd.DataFrame(self.failed_samples)
            if os.path.exists(self.error_log_path):
                existing_df = pd.read_csv(self.error_log_path)
                combined_df = pd.concat([existing_df, error_df], ignore_index=True)
            else:
                combined_df = error_df
            
            combined_df.to_csv(self.error_log_path, index=False)
            self.failed_samples = []
            logging.info(f"Saved {len(error_df)} error records to {self.error_log_path}")

class FeatureValidator:
    """Validate if generated features are valid"""
    
    @staticmethod
    def validate_mol_feature(mol_fea):
        """Validate molecular features"""
        errors = []
        
        # Check if required features exist
        required_attrs = [
            'mol_atoms_feats', 'mol_edges_feats', 'mol_edge_index',
            'atoinmol_index', 'mol_coords_feats', 'mol_embedding',
            'mol_graphs', 'mol_seq'
        ]
        
        for attr in required_attrs:
            if not hasattr(mol_fea, attr):
                errors.append(f"Missing molecular feature: {attr}")
            else:
                val = getattr(mol_fea, attr)
                if val is None:
                    errors.append(f"Molecular feature {attr} is None")
                elif isinstance(val, torch.Tensor) and torch.isnan(val).any():
                    errors.append(f"Molecular feature {attr} contains NaN values")
                elif isinstance(val, torch.Tensor) and torch.isinf(val).any():
                    errors.append(f"Molecular feature {attr} contains Inf values")
        
        return errors
    
    @staticmethod
    def validate_prot_feature(prot_fea):
        """Validate protein features"""
        errors = []
        
        required_attrs = [
            'pro_atoms_feats_s', 'pro_atoms_feats_v', 'pro_coords_feats',
            'pro_edges_feats_s', 'pro_edges_feats_v', 'pro_edge_index',
            'pro_index', 'pro_token_repre', 'pro_fp'
        ]
        
        for attr in required_attrs:
            if not hasattr(prot_fea, attr):
                errors.append(f"Missing protein feature: {attr}")
            else:
                val = getattr(prot_fea, attr)
                if val is None:
                    errors.append(f"Protein feature {attr} is None")
                elif isinstance(val, torch.Tensor) and torch.isnan(val).any():
                    errors.append(f"Protein feature {attr} contains NaN values")
                elif isinstance(val, torch.Tensor) and torch.isinf(val).any():
                    errors.append(f"Protein feature {attr} contains Inf values")
        
        return errors
    
    @staticmethod
    def validate_feature_dimensions(mol_fea, prot_fea):
        """Validate feature dimensions"""
        errors = []
        
        # Check molecular embedding dimensions
        if hasattr(mol_fea, 'mol_embedding'):
            emb_shape = mol_fea.mol_embedding.shape
            if len(emb_shape) != 3:
                errors.append(f"Molecular embedding dimension error: {emb_shape}")
        
        # Check protein embedding dimensions
        if hasattr(prot_fea, 'pro_token_repre'):
            emb_shape = prot_fea.pro_token_repre.shape
            if len(emb_shape) != 3:
                errors.append(f"Protein embedding dimension error: {emb_shape}")
        
        return errors


def _initialize_models():
    start_time = time.time()            
    GPU_number = torch.cuda.device_count()
    print(f"GPU_number:{GPU_number}")
    
    torch.cuda.set_device(2)
    device = torch.device(f'cuda:{3}')

    print(f"Current device: {torch.cuda.get_device_name(2)}")
    print(f"Available GPU memory: {torch.cuda.mem_get_info()[1]/1024**3:.2f}GB")
    logging.info(f"Using device: {device}")
    
    # initialize ESM2
    _esm_model, _esm_alphabet = esm2_t33_650M_UR50D()
    _esm_model = _esm_model.to(device)
    _batch_converter = DeviceAwareBatchConverter(_esm_alphabet, device) 
    
    # initialize ChemBERTa
    model_name = "DeepChem/ChemBERTa-10M-MLM"
    _chembert_model = AutoModel.from_pretrained(model_name).to(device)
    chembert_tokenizer = AutoTokenizer.from_pretrained(model_name)
    _device_aware_tokenizer = DeviceAwareChemTokenizer(chembert_tokenizer, device)
    
    logging.info(f"LLM model initialization completed, time: {time.time() - start_time:.2f} seconds")
    return _esm_model, _chembert_model, _device_aware_tokenizer, _batch_converter

class DeviceAwareBatchConverter:
    """Batch processor that automatically transfers data to model device"""
    def __init__(self, alphabet, device):
        self.alphabet = alphabet
        self.device = device
        self._base_converter = alphabet.get_batch_converter()
        
    def __call__(self, data):
        labels, strs, tokens = self._base_converter(data)
        return labels, strs, tokens.to(self.device)

class DeviceAwareChemTokenizer:
    """Tokenizer with fixed parameter conflicts"""
    def __init__(self, tokenizer, device):
        self.tokenizer = tokenizer
        self.device = device
        
    def __call__(self, text, **kwargs):
        kwargs.pop('return_tensors', None)  
        return self.tokenize(text, **kwargs)  
        
    def tokenize(self, text, padding=True, truncation=True, max_length=512, **kwargs):
        encoding = self.tokenizer(
            text,
            return_tensors="pt",  
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            **kwargs
        )
        return {k: v.to(self.device) for k, v in encoding.items()} 

def gen_seq_list(pdb):
    parser = PDBParser(QUIET=True)
    s = parser.get_structure(pdb, pdb)
    res_list = get_clean_res_list(s.get_residues(), verbose=False, ensure_ca_exist=True)
    res_list = [res for res in res_list if (('N' in res) and ('CA' in res) and ('C' in res) and ('O' in res))]
    seq_list = [f'{res.id[1]}{res.id[2]}' for res in res_list]
    return seq_list

def generate_graph_feature(pdb):
    parser = PDBParser(QUIET=True)
    s = parser.get_structure(f'{pdb}', pdb)
    res_list = get_clean_res_list(s.get_residues(), verbose=False, ensure_ca_exist=True)
    res_list = [res for res in res_list if (('N' in res) and ('CA' in res) and ('C' in res) and ('O' in res))]
    structure = {}
    structure['name'] = "placeholder"
    structure['seq'] = "".join([three_to_one.get(res.resname) for res in res_list])  
    coords = []
    for res in res_list:
        res_coords = []
        for atom in [res['N'], res['CA'], res['C'], res['O']]:
            res_coords.append(list(atom.coord))
        coords.append(res_coords) 
    structure['coords'] = coords 
    torch.set_num_threads(1)    
    dataset = data.ProteinGraphDataset([structure])
    protein = dataset[0]
    for i, edge_index in enumerate(protein.edge_index.tolist()):
        if i == 0:
            s_edge = torch.LongTensor(edge_index)
        elif i == 1:
            t_edge = torch.LongTensor(edge_index)
    
    protein_graph = dgl.graph((s_edge, t_edge))
    protein_graph.ndata['h'] = torch.FloatTensor(protein.node_s)
    
    return protein, protein_graph, coords

three_to_one = {'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 
                'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 
                'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'}

def get_clean_res_list(res_list, verbose=False, ensure_ca_exist=False, bfactor_cutoff=None):
    clean_res_list = []
    for res in res_list:
        hetero, resid, insertion = res.full_id[-1]
        if hetero == ' ':
            a = res.resname
            if res.resname not in three_to_one:
                if verbose:
                    print(res, "has non-standard resname")
                continue
            if (not ensure_ca_exist) or ('CA' in res):
                if bfactor_cutoff is not None:
                    ca_bfactor = float(res['CA'].bfactor)
                    if ca_bfactor < bfactor_cutoff:
                        continue
                clean_res_list.append(res)
        else:
            if verbose:
                print(res, res.full_id, "is hetero")
    return clean_res_list

def generate_protein_sequence(pdb_path):
    aa_codes = {'ALA':'A', 'CYS':'C', 'ASP':'D', 'GLU':'E', 'PHE':'F', 'GLY':'G', 'HIS':'H', 'LYS':'K',
    'ILE':'I', 'LEU':'L', 'MET':'M', 'ASN':'N', 'PRO':'P', 'GLN':'Q', 'ARG':'R', 'SER':'S', 'THR':'T',
    'VAL':'V', 'TYR':'Y', 'TRP':'W'}
    amino_acid_sequence = ''
    chain_num = 0
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)
    for model in structure:
        for chain in model:
            chain_num = chain_num + 1
            for residue in chain:
                if residue.get_id()[0] == " " and residue.get_resname() != "HOH":
                    if (('N' in residue) and ('CA' in residue) and ('C' in residue) and ('O' in residue)):
                        amino_acid_sequence += aa_codes[residue.get_resname()]
    length = len(amino_acid_sequence)
    return amino_acid_sequence, chain_num, length

def generate_inner_coor(pos, atom_feats, edge_index):
    cutoff = 8.0
    num_nodes = atom_feats.size(0)
    j, i = edge_index
    vecs = pos[j] - pos[i]
    dist = vecs.norm(dim=-1)

    # Calculate distances.
    _, argmin0 = scatter_min(dist, i, dim_size=num_nodes)
    argmin0[argmin0 >= len(i)] = 0
    n0 = j[argmin0]
    add = torch.zeros_like(dist).to(dist.device)
    add[argmin0] = cutoff
    dist1 = dist + add

    _, argmin1 = scatter_min(dist1, i, dim_size=num_nodes)
    argmin1[argmin1 >= len(i)] = 0
    n1 = j[argmin1]

    _, argmin0_j = scatter_min(dist, j, dim_size=num_nodes)
    argmin0_j[argmin0_j >= len(j)] = 0
    n0_j = i[argmin0_j]

    add_j = torch.zeros_like(dist).to(dist.device)
    add_j[argmin0_j] = cutoff
    dist1_j = dist + add_j

    # i[argmin] = range(0, num_nodes)
    _, argmin1_j = scatter_min(dist1_j, j, dim_size=num_nodes)
    argmin1_j[argmin1_j >= len(j)] = 0
    n1_j = i[argmin1_j]

    # n0, n1 for i
    n0 = n0[i]
    n1 = n1[i]

    # n0, n1 for j
    n0_j = n0_j[j]
    n1_j = n1_j[j]

    mask_iref = n0 == j
    iref = torch.clone(n0)
    iref[mask_iref] = n1[mask_iref]
    idx_iref = argmin0[i]
    idx_iref[mask_iref] = argmin1[i][mask_iref]

    mask_jref = n0_j == i
    jref = torch.clone(n0_j)
    jref[mask_jref] = n1_j[mask_jref]
    idx_jref = argmin0_j[j]
    idx_jref[mask_jref] = argmin1_j[j][mask_jref]

    pos_ji, pos_in0, pos_in1, pos_iref, pos_jref_j = (
        vecs,
        vecs[argmin0][i],
        vecs[argmin1][i],
        vecs[idx_iref],
        vecs[idx_jref]
    )

    # Calculate angles.
    a = ((-pos_ji) * pos_in0).sum(dim=-1)
    b = torch.cross(-pos_ji, pos_in0, dim=-1).norm(dim=-1)
    theta = torch.atan2(b, a)
    theta[theta < 0] = theta[theta < 0] + math.pi

    # Calculate torsions.
    dist_ji = pos_ji.pow(2).sum(dim=-1).sqrt()
    plane1 = torch.cross(-pos_ji, pos_in0, dim=-1)
    plane2 = torch.cross(-pos_ji, pos_in1, dim=-1)
    a = (plane1 * plane2).sum(dim=-1)  # cos_angle * |plane1| * |plane2|
    b = (torch.cross(plane1, plane2, dim=-1) * pos_ji).sum(dim=-1) / dist_ji
    phi = torch.atan2(b, a)
    phi[phi < 0] = phi[phi < 0] + math.pi

    # Calculate right torsions.
    plane1 = torch.cross(pos_ji, pos_jref_j, dim=-1)
    plane2 = torch.cross(pos_ji, pos_iref, dim=-1)
    a = (plane1 * plane2).sum(dim=-1)  # cos_angle * |plane1| * |plane2|
    b = (torch.cross(plane1, plane2, dim=-1) * pos_ji).sum(dim=-1) / dist_ji
    tau = torch.atan2(b, a)
    tau[tau < 0] = tau[tau < 0] + math.pi            
    return dist, theta, phi, tau

def get_activity_features(activity):    
    return [activity]

def get_function_features(function):
    function_mapping = {
        'Inactive': 0,
        'Agonist': 1,
        'Antagonist': 2
    }
    return [function_mapping.get(function, None)]

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        pass
    return list(map(lambda s: x == s, allowable_set))

def generate_smiles_nodes_edges_coords_graph_index_features(smile):
    enumerator = TautomerEnumerator()
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        raise ValueError(f"Cannot parse SMILES: {smile}")
    canonical_smiles = Chem.MolToSmiles(mol, 
                                    isomericSmiles=True,  
                                    kekuleSmiles=False) 
    mol = Chem.MolFromSmiles(canonical_smiles)
    mol = enumerator.Canonicalize(mol)
    Chem.SanitizeMol(mol)
    mol = Chem.AddHs(mol)
    
    logging.debug(f"Processing SMILES: {smile}")
    logging.debug(f"Number of atoms (including H): {mol.GetNumAtoms()}")
    
    # Node features
    atom_type = []
    atom_number = []
    atom_hybridization = []
    atomHs = []
    atom_charge = []
    atom_imvalence = []
    atom_aromatic = []
    atom_explicit = []
    atoms = mol.GetAtoms()
    num_atoms = mol.GetNumAtoms()
    
    for i, atom in enumerate(atoms):
        atom_type.append(atom.GetSymbol())
        atom_number.append(atom.GetAtomicNum())
        atom_hybridization.append(atom.GetHybridization())
        atomHs.append(atom.GetTotalNumHs())
        atom_charge.append(atom.GetFormalCharge())
        atom_imvalence.append(atom.GetImplicitValence())
        atom_aromatic.append(1 if atom.GetIsAromatic() else 0)
        atom_explicit.append(atom.GetExplicitValence())
    
    nodes_hybridization = [one_of_k_encoding(h, [HybridizationType.SP, HybridizationType.SP2, HybridizationType.SP3, HybridizationType.SP3D, HybridizationType.SP3D2]) for h in atom_hybridization]
    nodes_type = [one_of_k_encoding(t[0], ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) for t in atom_type]
    atom_Hss = [one_of_k_encoding(t, [0, 1, 2, 3, 4]) for t in atomHs]
    atom_imvalence = [one_of_k_encoding(t, [0, 1, 2, 3, 4, 5, 6]) for t in atom_imvalence]
    atom_charge = [one_of_k_encoding(t, [0, 1, 2, 3, 4, 5, 6]) for t in atom_charge]
    atom_explicit = [one_of_k_encoding(t, [0, 1, 2, 3, 4]) for t in atom_explicit]
    
    # Edge features
    bond_type = []
    bond_conj = []
    bond_ring = []
    row = []
    col = []
    bonds = mol.GetBonds()
    
    for bond in bonds:
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        bond_type += 2 * [bond.GetBondType()]
        bond_ring += 2 * [bond.IsInRing()]
        bond_conj += 2 * [bond.GetIsConjugated()]
    
    edge_conj = [one_of_k_encoding(t, [True, False]) for t in bond_conj]
    edge_ring = [one_of_k_encoding(t, [True, False]) for t in bond_ring]
    bond_index = torch.LongTensor([row, col])
    edge_type = [one_of_k_encoding(t, [BondType.SINGLE, BondType.DOUBLE, BondType.TRIPLE, BondType.AROMATIC]) for t in bond_type]
    perm = (bond_index[0] * num_atoms + bond_index[1]).argsort()
    edge_index = bond_index[:, perm]
    atom_f1 = torch.tensor([atom_number, atom_aromatic, atomHs], dtype=torch.float).t().contiguous()
    
    # Combine node features
    mol_all_nodes_feature = torch.cat([torch.FloatTensor(nodes_hybridization), torch.FloatTensor(nodes_type), 
                                       torch.FloatTensor(atom_Hss), torch.FloatTensor(atom_imvalence), torch.FloatTensor(atom_charge), torch.FloatTensor(atom_explicit), atom_f1], dim=-1)    
    # Combine edge features
    mol_all_edges_feature = torch.cat([torch.FloatTensor(edge_conj), torch.FloatTensor(edge_ring), torch.FloatTensor(edge_type)], dim=-1)    
    
    try:
        # Generate 3D conformation using ETKDG method
        AllChem.EmbedMolecule(mol, randomSeed=42, useRandomCoords=True)
        
        # Optimize using MMFF94s force field
        if AllChem.MMFFHasAllMoleculeParams(mol):
            AllChem.MMFFOptimizeMolecule(mol)
        else:
            # Try UFF force field if MMFF94s is not applicable
            AllChem.UFFOptimizeMolecule(mol)
        
        mol_all_coords_feature = torch.FloatTensor(mol.GetConformer().GetPositions())
        logging.debug(f"Successfully generated 3D coordinates, shape: {mol_all_coords_feature.shape}")
    except Exception as e:
        logging.warning(f"Cannot generate 3D coordinates for molecule '{smile}': {str(e)}")
        mol_all_coords_feature = torch.zeros((num_atoms, 3), dtype=torch.float)
    
    # Generate graph structure
    atom_matrix = GetAdjacencyMatrix(mol)
    graph = nx.from_numpy_array(atom_matrix)
    mol_dgl_graph = dgl.from_networkx(graph)
    mol_dgl_graph.ndata['h'] = torch.FloatTensor(mol_all_nodes_feature)
    
    return mol_all_nodes_feature, mol_all_edges_feature, mol_all_coords_feature, mol_dgl_graph, edge_index, canonical_smiles, atom_type, mol

def get_pdb_features(PDB_file, batch_converter, model1):
    pro_index = gen_seq_list(PDB_file)

    protein, protein_graph, coords = generate_graph_feature(PDB_file)
    protein_sequence, chain_num, length = generate_protein_sequence(PDB_file)
    pr_dist, pr_theta, pr_phi, pr_tau = generate_inner_coor(protein.x, protein.node_s, protein.edge_index)
    prot = [(0, protein_sequence[:2024])]
    batch_labels, batch_strs, batch_tokens = batch_converter(prot)
    with torch.no_grad(): 
        results = model1(batch_tokens, repr_layers=[33], return_contacts=True)
    pro_token_repre = results["representations"][33]
    pro_token_repre = pro_token_repre.cpu()       
    ionchannel_pdb_fea = Data(pro_graphs=protein_graph,
                    pro_index=pro_index,
                    batch_labels=batch_labels,
                    batch_strs=batch_strs,
                    pro_atoms_feats_s=protein.node_s,
                    pro_atoms_feats_v=protein.node_v,
                    pro_coords_feats=protein.x,
                    pro_edges_feats_s=protein.edge_s,
                    pro_edges_feats_v=protein.edge_v,
                    pro_edge_index=protein.edge_index,
                    pro_coords = coords,
                    pro_token_repre=pro_token_repre,
                    pro_fp=protein_sequence,
                    pr_dist=pr_dist,
                    pr_theta=pr_theta,
                    pr_phi=pr_phi,
                    pr_tau=pr_tau)            
    return ionchannel_pdb_fea

def generate_indexes(model, tokenizer, atom_type, mol_smiles):
    # Chemical element whitelist
    VALID_ELEMENTS = {'H', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I', 'Hg', 'B', 'Si', 'Na', 'K', 'Ca', 'Mg', 'Fe', 'Zn', 'Cu', 'Mn', 'Mo', 'Ni', 'Se', 'As', 'V', 'Al', 'Ti', 'Ge', 'Sn', 'Pb', 'Cd', 'W', 'Cr', 'Co', 'Li', 'Y', 'Zr', 'Nb', 'Ta', 'Re', 'Pd', 'Ag', 'Au', 'Pt', 'Rh', 'Ir'}
    
    # Generate valid atom indices
    mol_index = [
        idx for idx, atom in enumerate(atom_type)
        if any(elem in atom for elem in VALID_ELEMENTS)
    ]
    
    # Chemical-specific tokenization
    subwords = chemical_tokenize(mol_smiles)
    
    # Generate subword indices
    subinmol_index = [
        idx for idx, token in enumerate(subwords)
        if any(c.isalpha() for c in token)
    ]
    subinmol_words = [subwords[idx] for idx in subinmol_index]
    
    # Extract atomic base symbols (ignore modifiers)
    atom_symbols = [atom.split('_')[0].strip('@[]') for atom in atom_type]
    
    # Dynamic alignment
    aligned_mol_idx, aligned_submol_idx = align_elements(atom_symbols, subinmol_words)
    
    # Generate embeddings
    tokenized = tokenizer(mol_smiles, return_tensors="pt")
    with torch.no_grad():
        embeddings = model(**tokenized).last_hidden_state
    
    # Validate dimensions
    if len(aligned_mol_idx) != len(aligned_submol_idx):
        logging.warning(f"Dimension mismatch: {mol_smiles}")
        return [], [], embeddings, mol_index
    
    return aligned_mol_idx, aligned_submol_idx, embeddings, mol_index

def chemical_tokenize(smiles):
    """Chemical-specific tokenizer"""
    elements = ['Fe', 'Cl', 'Br', 'Na', 'Mg']  # Prioritize two-character elements
    tokens = []
    i = 0
    while i < len(smiles):
        matched = False
        # Match two-character elements
        for elem in elements:
            if smiles[i:i+2] == elem:
                tokens.append(elem)
                i += 2
                matched = True
                break
        if not matched:
            # Process special symbols
            if smiles[i] in ['@', '[', ']']:
                tokens.append(smiles[i])
                i +=1
            # Merge digits
            elif smiles[i].isdigit():
                num = []
                while i < len(smiles) and smiles[i].isdigit():
                    num.append(smiles[i])
                    i +=1
                tokens.append(''.join(num))
            else:
                tokens.append(smiles[i])
                i +=1
    return tokens

from difflib import SequenceMatcher

def align_elements(atom_list, subword_list):
    """Align based on longest common subsequence"""
    matcher = SequenceMatcher(None, atom_list, subword_list)
    matches = matcher.get_matching_blocks()
    
    aligned_atom = []
    aligned_subword = []
    for match in matches:
        a_start = match.a
        b_start = match.b
        size = match.size
        
        aligned_atom.extend(range(a_start, a_start+size))
        aligned_subword.extend(range(b_start, b_start+size))
    
    return aligned_atom, aligned_subword

from typing import Union
def get_cache_path(smiles: str, uniprot_id: str, index: str, cache_dir) -> str:
    """Generate safe cache file path"""
    safe_smiles = hashlib.md5(smiles.encode()).hexdigest()[:16]
    safe_uniprot = uniprot_id.replace("/", "_")[:16]
    modified_index = index
    return os.path.join(cache_dir, f"{modified_index}_{safe_uniprot}_{safe_smiles}.pkl")

# ==================== Channel Classification ====================

def assign_task_group(task_name: str, task_groups: dict, df: pd.DataFrame, 
                     q75: float = None, q50: float = None, q25: float = None) -> str:
    """Directly assign task group according to specified classification method"""
    
    # 1. First check predefined groups
    for group, config in task_groups.items():
        if task_name in config['tasks']:
            return group
    
    # 2. Get task data volume (for logging)
    task_size = len(df[df['ABB'] == task_name])
    if task_size == 0:
        raise ValueError(f"Task '{task_name}' does not exist in data")
    
    # 3. Use explicit mapping for classification
    explicit_mapping = {
        # tiny_volume (< 1000)
        'CNG': 'tiny_volume',
        'CaCC': 'tiny_volume',
        'GlyRs': 'tiny_volume',
        'KCa': 'tiny_volume',
        
        # low_volume (1000-3162)
        'CFTR': 'low_volume',
        'γ-GABA-ARs': 'low_volume',
        '5-HT3R': 'low_volume',
        'K2P': 'low_volume',
        
        # mid_volume (3162-10000)
        'Kir': 'mid_volume',
        'nAChRs': 'mid_volume',
        'VGCCs': 'mid_volume',
        'i-GluRs': 'mid_volume',
        
        # high_volume (≥10000)
        'TRP': 'high_volume',
        'P2X': 'high_volume',
        'VGSCs': 'high_volume',
        'VGKCs': 'high_volume'
    }
    
    # 4. If task is in mapping, directly return corresponding group
    if task_name in explicit_mapping:
        return explicit_mapping[task_name]
    
    # 5. For new tasks not in mapping, use logarithmic thresholds as fallback
    if task_size >= 10000:
        return 'high_volume'
    elif task_size >= 3162:
        return 'mid_volume'
    elif task_size >= 1000:
        return 'low_volume'
    else:
        return 'tiny_volume'


def build_task_groups(df: pd.DataFrame) -> Dict:
    """Build task group configuration - using direct specified classification method"""
    
    # Calculate sample count for each channel
    abb_counts = df['ABB'].value_counts().to_dict()
    
    # Explicitly specify classification for each channel
    explicit_mapping = {
        # tiny_volume (< 1000)
        'CNG': 'tiny_volume',
        'CaCC': 'tiny_volume',
        'GlyRs': 'tiny_volume',
        'KCa': 'tiny_volume',
        
        # low_volume (1000-3162)
        'CFTR': 'low_volume',
        'γ-GABA-ARs': 'low_volume',
        '5-HT3R': 'low_volume',
        'K2P': 'low_volume',
        
        # mid_volume (3162-10000)
        'Kir': 'mid_volume',
        'nAChRs': 'mid_volume',
        'VGCCs': 'mid_volume',
        'i-GluRs': 'mid_volume',
        
        # high_volume (≥10000)
        'TRP': 'high_volume',
        'P2X': 'high_volume',
        'VGSCs': 'high_volume',
        'VGKCs': 'high_volume'
    }
    
    # Initialize group structure
    groups = {
        'high_volume': {'tasks': [], 'range': (10000, float('inf'))},
        'mid_volume': {'tasks': [], 'range': (3162, 10000)},
        'low_volume': {'tasks': [], 'range': (1000, 3162)},
        'tiny_volume': {'tasks': [], 'range': (0, 1000)}
    }
    
    # Assign tasks to groups according to explicit mapping
    for abb, count in abb_counts.items():
        if abb in explicit_mapping:
            group = explicit_mapping[abb]
            groups[group]['tasks'].append(abb)
        else:
            # For tasks not in mapping, classify by logarithmic thresholds
            if count >= 10000:
                groups['high_volume']['tasks'].append(abb)
            elif count >= 3162:
                groups['mid_volume']['tasks'].append(abb)
            elif count >= 1000:
                groups['low_volume']['tasks'].append(abb)
            else:
                groups['tiny_volume']['tasks'].append(abb)
    
    # Add statistics for each group
    for group, config in groups.items():
        if config['tasks']:
            config['sample_count'] = sum(abb_counts[abb] for abb in config['tasks'])
            config['min_samples'] = min(abb_counts[abb] for abb in config['tasks'])
            config['max_samples'] = max(abb_counts[abb] for abb in config['tasks'])
            config['avg_samples'] = config['sample_count'] / len(config['tasks'])
            config['task_count'] = len(config['tasks'])
        else:
            config['sample_count'] = 0
            config['min_samples'] = 0
            config['max_samples'] = 0
            config['avg_samples'] = 0
            config['task_count'] = 0
    
    return groups


def build_test_task_groups(df: pd.DataFrame) -> dict:
    """Build test task group configuration - using same direct specified classification method"""
    
    # Calculate sample count for each channel
    abb_counts = df['ABB'].value_counts().to_dict()
    
    # Use same explicit mapping as training set
    explicit_mapping = {
        'CNG': 'tiny_volume',
        'CaCC': 'tiny_volume',
        'GlyRs': 'tiny_volume',
        'KCa': 'tiny_volume',
        'CFTR': 'low_volume',
        'γ-GABA-ARs': 'low_volume',
        '5-HT3R': 'low_volume',
        'K2P': 'low_volume',
        'Kir': 'mid_volume',
        'nAChRs': 'mid_volume',
        'VGCCs': 'mid_volume',
        'i-GluRs': 'mid_volume',
        'TRP': 'high_volume',
        'P2X': 'high_volume',
        'VGSCs': 'high_volume',
        'VGKCs': 'high_volume'
    }
    
    # Initialize group structure
    groups = {
        'high_volume': {'tasks': [], 'range': (10000, float('inf'))},
        'mid_volume': {'tasks': [], 'range': (3162, 10000)},
        'low_volume': {'tasks': [], 'range': (1000, 3162)},
        'tiny_volume': {'tasks': [], 'range': (0, 1000)}
    }
    
    # Assign tasks to groups according to explicit mapping
    for abb, count in abb_counts.items():
        if abb in explicit_mapping:
            group = explicit_mapping[abb]
            groups[group]['tasks'].append(abb)
        else:
            # For tasks not in mapping, classify by logarithmic thresholds
            if count >= 10000:
                groups['high_volume']['tasks'].append(abb)
            elif count >= 3162:
                groups['mid_volume']['tasks'].append(abb)
            elif count >= 1000:
                groups['low_volume']['tasks'].append(abb)
            else:
                groups['tiny_volume']['tasks'].append(abb)
    
    # Add statistics for each group
    for group, config in groups.items():
        if config['tasks']:
            config['sample_count'] = sum(abb_counts[abb] for abb in config['tasks'])
            config['min_samples'] = min(abb_counts[abb] for abb in config['tasks'])
            config['max_samples'] = max(abb_counts[abb] for abb in config['tasks'])
            config['avg_samples'] = config['sample_count'] / len(config['tasks'])
            config['task_count'] = len(config['tasks'])
        else:
            config['sample_count'] = 0
            config['min_samples'] = 0
            config['max_samples'] = 0
            config['avg_samples'] = 0
            config['task_count'] = 0
    
    return groups


def generate_feature_report(cache_dir):
    """Generate feature validation report"""
    report_path = os.path.join(cache_dir, 'feature_generation_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("Feature Generation Validation Report\n")
        f.write("="*50 + "\n")
        f.write(f"Generation time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Count feature files for each group
        for group in ['high_volume', 'mid_volume', 'low_volume', 'tiny_volume']:
            group_dir = os.path.join(cache_dir, group)
            if os.path.exists(group_dir):
                files = [f for f in os.listdir(group_dir) if f.endswith('.pkl')]
                f.write(f"{group} group: {len(files)} feature files\n")
        
        # Failed samples
        failed_dir = os.path.join(cache_dir, 'failed')
        if os.path.exists(failed_dir):
            failed_files = [f for f in os.listdir(failed_dir) if f.endswith('.pkl')]
            f.write(f"\nFailed samples: {len(failed_files)}\n")
        
        f.write("\nFeature validation conclusion:\n")
        f.write("1. All successfully generated features have been saved to respective group directories\n")
        f.write("2. Failed features have been recorded to failed_features.csv\n")
        f.write("3. Original data of failed features have been saved to failed/ directory\n")
        f.write("4. Generated features can be directly used for training code\n")


# ==================== Main Function ====================

if __name__ == "__main__": 
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'feature_generation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Initialize error handler
    error_handler = FeatureErrorHandler('failed_features.csv')
    
    file_path = 'data/train_set.csv' 
    df = pd.read_csv(file_path)
    
    # Statistics
    success_count = 0
    failure_count = 0
    
    # Initialize models
    logging.info("Initializing models...")
    try:
        _esm_model, _chembert_model, _device_aware_tokenizer, _batch_converter = _initialize_models()
    except Exception as e:
        logging.error(f"Model initialization failed: {str(e)}")
        sys.exit(1)
    
    # Create progress bar
    progress_bar = tqdm(total=len(df), desc="Generating features")
    
    # Ensure cache directory exists
    os.makedirs(GlobalConfig.CACHE_DIR_train, exist_ok=True)
    for group in ['high_volume', 'mid_volume', 'low_volume', 'tiny_volume']:
        os.makedirs(os.path.join(GlobalConfig.CACHE_DIR_train, group), exist_ok=True)
    
    # Build task groups (using direct specified classification method)
    task_groups = build_task_groups(df)
    
    # Validate classification results
    logging.info("Task group results:")
    for group_name, config in task_groups.items():
        logging.info(f"  {group_name}: {len(config['tasks'])} tasks, total samples: {config['sample_count']}")
        logging.info(f"    Task list: {config['tasks']}")
    
    # Process each sample
    for index, row in df.iterrows():
        try:
            counts = df['ABB'].value_counts()
            
            # Get task group (using new classification method)
            try:
                task_group = assign_task_group(row['ABB'], task_groups, df)
            except Exception as e:
                logging.warning(f"Cannot determine task group: {str(e)}")
                task_group = 'unknown'
            
            logging.info(f"Processing row {index}: SMILES={row['SMILES']}, UniProt ID={row['UniProt ID']}, task group={task_group}") 
            
            # 1. Generate molecular features
            try:
                mol_result = generate_smiles_nodes_edges_coords_graph_index_features(row['SMILES'])
                (mol_all_nodes_feature, mol_all_edges_feature, 
                 mol_all_coords_feature, mol_dgl_graph, 
                 edge_index, mol_smiles, atom_types, mol) = mol_result
                
                # Generate internal coordinates
                mol_dist, mol_theta, mol_phi, mol_tau = generate_inner_coor(
                    mol_all_coords_feature, mol_all_nodes_feature, edge_index)
                
                # Generate indices and embeddings
                aligned_mol_idx, subinmol_index, embeddings, mol_index = generate_indexes(
                    _chembert_model, _device_aware_tokenizer, atom_types, mol_smiles)
                
                # Move to CPU
                embeddings = embeddings.cpu()
                
                # Build molecular feature object
                mol_fea = Data(
                    mol_atoms_feats=mol_all_nodes_feature,
                    mol_edges_feats=mol_all_edges_feature,
                    mol_coords_feats=mol_all_coords_feature,
                    atoinmol_index=mol_index,
                    subwinsmi_index=subinmol_index,
                    mol_embedding=embeddings,
                    mol_graphs=mol_dgl_graph,
                    mol_edge_index=edge_index,
                    mol_seq=mol_smiles,
                    mol_dist=mol_dist,
                    mol_theta=mol_theta,
                    mol_phi=mol_phi,
                    mol_tau=mol_tau,
                    atom_types=atom_types,
                    mol=mol
                )
            except Exception as e:
                raise ValueError(f"Molecular feature generation failed: {str(e)}")
            
            # 2. Generate protein features
            try:
                pdb_path = os.path.join(GlobalConfig.PDB_file_path, f"{row['UniProt ID']}.pdb")
                if not os.path.exists(pdb_path):
                    raise FileNotFoundError(f"PDB file does not exist: {pdb_path}")
                
                prot_fea = get_pdb_features(pdb_path, _batch_converter, _esm_model)
            except Exception as e:
                raise ValueError(f"Protein feature generation failed: {str(e)}")
            
            # 3. Validate features
            validator = FeatureValidator()
            mol_errors = validator.validate_mol_feature(mol_fea)
            prot_errors = validator.validate_prot_feature(prot_fea)
            dim_errors = validator.validate_feature_dimensions(mol_fea, prot_fea)
            
            all_errors = mol_errors + prot_errors + dim_errors
            
            if all_errors:
                error_msg = "; ".join(all_errors)
                raise ValueError(f"Feature validation failed: {error_msg}")
            
            # 4. Build feature data
            fea_data = {
                'mol_fea': mol_fea,
                'prot_fea': prot_fea,
                'label': torch.tensor(row['Label'], dtype=torch.long),
                'task': row['ABB'],
                'group': task_group,
                'smiles': row['SMILES'],
                'uniprot_id': row['UniProt ID']
            }
            
            # 5. Save features to cache
            cache_dir_train = os.path.join(GlobalConfig.CACHE_DIR_train, task_group)
            os.makedirs(cache_dir_train, exist_ok=True)
            
            index_str = str(index)
            mindex = 'TW' + index_str
            
            try:
                cache_path = get_cache_path(row['SMILES'], mindex, row['UniProt ID'], cache_dir_train)
            except Exception as e:
                raise ValueError(f"Cache path generation failed: {str(e)}")
            
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump(fea_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            success_count += 1
            logging.info(f"Feature generation and save successful (row {index}, SMILES: {row['SMILES']}, UniProt ID: {row['UniProt ID']})")
            
        except Exception as e:
            # Record failure
            failure_count += 1
            error_type = type(e).__name__
            error_msg = str(e)
            
            # Record error
            error_handler.record_failure(index, row, error_type, error_msg)
            
            # Detailed logging
            logging.error(f"Error processing row {index}: {error_msg}")
            logging.debug(traceback.format_exc())
            
            # Save failed features (for debugging)
            try:
                failed_data = {
                    'index': index,
                    'row': row.to_dict(),
                    'error_type': error_type,
                    'error_message': error_msg,
                    'traceback': traceback.format_exc()
                }
                
                failed_dir = os.path.join(GlobalConfig.CACHE_DIR_train, 'failed')
                os.makedirs(failed_dir, exist_ok=True)
                failed_path = os.path.join(failed_dir, f"failed_{index}_{row['UniProt ID']}.pkl")
                
                with open(failed_path, 'wb') as f:
                    pickle.dump(failed_data, f)
                    
            except Exception as save_error:
                logging.error(f"Error saving failed feature: {str(save_error)}")
        
        finally:
            progress_bar.update(1)
            progress_bar.set_postfix({
                'Success': success_count,
                'Failure': failure_count
            })
            
            # Periodically clean GPU cache
            if index % 50 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # Complete progress bar
    progress_bar.close()
    
    # Save remaining error records
    error_handler.save_errors()
    
    # Print classification statistics
    logging.info("\nTask classification statistics:")
    for group_name, config in task_groups.items():
        logging.info(f"  {group_name}: {config['task_count']} tasks, {config['sample_count']} samples")
    
    # Generate feature validation report
    generate_feature_report(GlobalConfig.CACHE_DIR_train)