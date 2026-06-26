import os
import pickle
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
import random 
from rdkit.Chem import BondType
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
from rdkit.Chem.MolStandardize.rdMolStandardize import TautomerEnumerator
from rdkit.Chem import HybridizationType
import time
from transformers import AutoModel, AutoTokenizer
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from tqdm import tqdm
import traceback
from datetime import datetime
import sys


class GlobalConfig:
    REQUIRED_COLUMNS = ['SMILES', 'Sequence', 'ABB', 'Label']
    DATA_CACHE_DIR = "9CRS_SDF/"
    PDB_file_path = "9CRS_SDF/"
    PDB_cache_path = "9CRS_SDF/"
    CACHE_DIR = "9CRS_SDF/"
    CACHE_DIR_train = "9CRS_SDF/"

    FP_SIZE = 2048
    LABEL_MAPPING = {0: "inactive", 1: "inhibitor", 2: "activator"}
    NUM_CLASSES = len(LABEL_MAPPING)
    MOL_FEAT_DIM = 2048
    PROT_FEAT_DIM = 4096
    HIDDEN_DIM = 2048
    DROPOUT_RATE = 0.3
    SEED = 42
    TEST_SIZE = 0.2
    BATCH_SIZES = {
        'high_volume': 256,
        'mid_volume': 128,
        'low_volume': 64,
        'tiny_volume': 32
    }
    LEARNING_RATES = {
        'base': 1e-4,
        'high': 3e-4,
        'mid': 1e-4,
        'low': 5e-5,
        'tiny': 1e-5
    }
    WEIGHT_DECAY = 1e-5
    EPOCHS = 150
    EARLY_STOP_PATIENCE = 15
    GRAD_CLIP = 1.0

    DEVICE = torch.device("cuda:7")
    NUM_GPUS = 0
    NUM_WORKERS = min(4, os.cpu_count())


    @classmethod
    def setup_dirs(cls):
        os.makedirs(cls.DATA_CACHE_DIR, exist_ok=True)

    @classmethod
    def set_seed(cls, seed=None):
        seed = seed or cls.SEED
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class FeatureErrorHandler:
    def __init__(self, error_log_path):
        self.error_log_path = error_log_path
        self.failed_samples = []
        self._init_error_log()

    def _init_error_log(self):
        if not os.path.exists(self.error_log_path):
            pd.DataFrame(columns=['timestamp', 'index', 'smiles', 'uniprot_id',
                                  'task', 'group', 'error_type', 'error_message']).to_csv(self.error_log_path, index=False)

    def record_failure(self, index, row, error_type, error_msg):
        error_record = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'index': index,
            'smiles': row.get('SMILES', ''),
            'uniprot_id': row.get('PDB_ID', ''),
            'task': row.get('ABB', ''),
            'group': '',
            'error_type': error_type,
            'error_message': str(error_msg)[:500]
        }
        self.failed_samples.append(error_record)
        if len(self.failed_samples) >= 10:
            self.save_errors()

    def save_errors(self):
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
    @staticmethod
    def validate_mol_feature(mol_fea):
        errors = []
        required_attrs = ['mol_atoms_feats', 'mol_edges_feats', 'mol_edge_index',
                          'atoinmol_index', 'mol_coords_feats', 'mol_embedding',
                          'mol_graphs', 'mol_seq']
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
        errors = []
        required_attrs = ['pro_atoms_feats_s', 'pro_atoms_feats_v', 'pro_coords_feats',
                          'pro_edges_feats_s', 'pro_edges_feats_v', 'pro_edge_index',
                          'pro_index', 'pro_token_repre', 'pro_fp']
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
        errors = []
        if hasattr(mol_fea, 'mol_embedding') and len(mol_fea.mol_embedding.shape) != 3:
            errors.append(f"Molecular embedding dimension error: {mol_fea.mol_embedding.shape}")
        if hasattr(prot_fea, 'pro_token_repre') and len(prot_fea.pro_token_repre.shape) != 3:
            errors.append(f"Protein embedding dimension error: {prot_fea.pro_token_repre.shape}")
        return errors



def _initialize_models():
    start_time = time.time()
    GPU_number = torch.cuda.device_count()
    torch.cuda.set_device(7)
    device = torch.device(f'cuda:{7}')
    logging.info(f"Using device: {device}")

    _esm_model, _esm_alphabet = esm2_t33_650M_UR50D()
    _esm_model = _esm_model.to(device)
    _batch_converter = DeviceAwareBatchConverter(_esm_alphabet, device)

    model_name = "DeepChem/ChemBERTa-10M-MLM"
    _chembert_model = AutoModel.from_pretrained(model_name).to(device)
    chembert_tokenizer = AutoTokenizer.from_pretrained(model_name)
    _device_aware_tokenizer = DeviceAwareChemTokenizer(chembert_tokenizer, device)

    logging.info(f"LLM models initialization completed, took {time.time() - start_time:.2f} seconds")
    return _esm_model, _chembert_model, _device_aware_tokenizer, _batch_converter


class DeviceAwareBatchConverter:
    def __init__(self, alphabet, device):
        self.alphabet = alphabet
        self.device = device
        self._base_converter = alphabet.get_batch_converter()

    def __call__(self, data):
        labels, strs, tokens = self._base_converter(data)
        return labels, strs, tokens.to(self.device)


class DeviceAwareChemTokenizer:
    def __init__(self, tokenizer, device):
        self.tokenizer = tokenizer
        self.device = device

    def __call__(self, text, **kwargs):
        kwargs.pop('return_tensors', None)
        return self.tokenize(text, **kwargs)

    def tokenize(self, text, padding=True, truncation=True, max_length=512, **kwargs):
        encoding = self.tokenizer(
            text, return_tensors="pt", padding=padding,
            truncation=truncation, max_length=max_length, **kwargs
        )
        return {k: v.to(self.device) for k, v in encoding.items()}



three_to_one = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
}


def get_clean_res_list(res_list, verbose=False, ensure_ca_exist=False, bfactor_cutoff=None):
    clean_res_list = []
    for res in res_list:
        hetero, resid, insertion = res.full_id[-1]
        if hetero == ' ':
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


def gen_seq_list(pdb):

    parser = PDBParser(QUIET=True)
    s = parser.get_structure(pdb, pdb)
    res_list = get_clean_res_list(s.get_residues(), verbose=False, ensure_ca_exist=True)
    res_list = [res for res in res_list if ('N' in res and 'CA' in res and 'C' in res and 'O' in res)]
    seq_list = [f'{res.id[1]}{res.id[2]}' for res in res_list]
    return seq_list

def build_global_residue_map_from_res_list(res_list: List) -> Tuple[Dict[Tuple[str, int, str], int], str]:

    res_map = {}
    global_seq = []
    for idx, res in enumerate(res_list):
        _, res_num, ins_code = res.full_id[-1]
        chain_id = res.full_id[2]  
        key = (chain_id, res_num, ins_code)
        if key in res_map:

            raise ValueError(f"Duplicate residue key: {key}")
        res_map[key] = idx
        global_seq.append(three_to_one[res.resname])
    return res_map, ''.join(global_seq)


def generate_graph_feature(pdb):
    parser = PDBParser(QUIET=True)
    s = parser.get_structure(f'{pdb}', pdb)
    res_list = get_clean_res_list(s.get_residues(), verbose=False, ensure_ca_exist=True)
    res_list = [res for res in res_list if ('N' in res and 'CA' in res and 'C' in res and 'O' in res)]
    
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
    
    s_edge = torch.LongTensor(protein.edge_index[0].tolist())
    t_edge = torch.LongTensor(protein.edge_index[1].tolist())
    protein_graph = dgl.graph((s_edge, t_edge))
    protein_graph.ndata['h'] = torch.FloatTensor(protein.node_s)
    
    return protein, protein_graph, coords, res_list

def generate_protein_sequence(pdb_path):
    aa_codes = {'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H',
                'LYS': 'K', 'ILE': 'I', 'LEU': 'L', 'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q',
                'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TYR': 'Y', 'TRP': 'W'}
    amino_acid_sequence = ''
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_id()[0] == " " and residue.get_resname() != "HOH":
                    if 'N' in residue and 'CA' in residue and 'C' in residue and 'O' in residue:
                        amino_acid_sequence += aa_codes[residue.get_resname()]
    length = len(amino_acid_sequence)
    return amino_acid_sequence, 1, length


def generate_inner_coor(pos, atom_feats, edge_index):
    cutoff = 8.0
    num_nodes = atom_feats.size(0)
    j, i = edge_index
    vecs = pos[j] - pos[i]
    dist = vecs.norm(dim=-1)

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
    _, argmin1_j = scatter_min(dist1_j, j, dim_size=num_nodes)
    argmin1_j[argmin1_j >= len(j)] = 0
    n1_j = i[argmin1_j]

    n0 = n0[i]
    n1 = n1[i]
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

    a = ((-pos_ji) * pos_in0).sum(dim=-1)
    b = torch.cross(-pos_ji, pos_in0, dim=-1).norm(dim=-1)
    theta = torch.atan2(b, a)
    theta[theta < 0] = theta[theta < 0] + math.pi

    dist_ji = pos_ji.pow(2).sum(dim=-1).sqrt()
    plane1 = torch.cross(-pos_ji, pos_in0, dim=-1)
    plane2 = torch.cross(-pos_ji, pos_in1, dim=-1)
    a = (plane1 * plane2).sum(dim=-1)
    b = (torch.cross(plane1, plane2, dim=-1) * pos_ji).sum(dim=-1) / dist_ji
    phi = torch.atan2(b, a)
    phi[phi < 0] = phi[phi < 0] + math.pi

    plane1 = torch.cross(pos_ji, pos_jref_j, dim=-1)
    plane2 = torch.cross(pos_ji, pos_iref, dim=-1)
    a = (plane1 * plane2).sum(dim=-1)
    b = (torch.cross(plane1, plane2, dim=-1) * pos_ji).sum(dim=-1) / dist_ji
    tau = torch.atan2(b, a)
    tau[tau < 0] = tau[tau < 0] + math.pi
    return dist, theta, phi, tau


def generate_smiles_nodes_edges_coords_graph_index_features(sdf_input, sanitize=True, add_hs=True):
    if isinstance(sdf_input, str):
        if sdf_input.endswith('.sdf') and os.path.isfile(sdf_input):
            supplier = Chem.SDMolSupplier(sdf_input, sanitize=False)
            mol = supplier[0] if supplier else None
        else:
            mol = Chem.MolFromMolBlock(sdf_input, sanitize=False)
    elif isinstance(sdf_input, Chem.rdchem.Mol):
        mol = sdf_input
    else:
        raise ValueError("sdf_input must be a file path, MolBlock string, or RDKit Mol object")
    if mol is None:
        raise ValueError("Failed to read molecule from input")
    if sanitize:
        try:
            Chem.SanitizeMol(mol)
        except Exception as e:
            logging.warning(f"Sanitization failed: {e}")
    if add_hs:
        mol = Chem.AddHs(mol)
    canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=True, kekuleSmiles=False)
    atoms = mol.GetAtoms()
    num_atoms = mol.GetNumAtoms()
    atom_type = []
    atom_number = []
    atom_hybridization = []
    atomHs = []
    atom_charge = []
    atom_imvalence = []
    atom_aromatic = []
    atom_explicit = []
    for atom in atoms:
        atom_type.append(atom.GetSymbol())
        atom_number.append(atom.GetAtomicNum())
        atom_hybridization.append(atom.GetHybridization())
        atomHs.append(atom.GetTotalNumHs())
        atom_charge.append(atom.GetFormalCharge())
        atom_imvalence.append(atom.GetImplicitValence())
        atom_aromatic.append(1 if atom.GetIsAromatic() else 0)
        atom_explicit.append(atom.GetExplicitValence())
    nodes_hybridization = [one_of_k_encoding(h, [HybridizationType.SP, HybridizationType.SP2,
                                                 HybridizationType.SP3, HybridizationType.SP3D,
                                                 HybridizationType.SP3D2]) for h in atom_hybridization]
    nodes_type = [one_of_k_encoding(t[0], ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na',
                                           'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb',
                                           'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'Li', 'Ge', 'Cu',
                                           'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb',
                                           'Unknown']) for t in atom_type]
    atom_Hss = [one_of_k_encoding(t, [0, 1, 2, 3, 4]) for t in atomHs]
    atom_imvalence = [one_of_k_encoding(t, [0, 1, 2, 3, 4, 5, 6]) for t in atom_imvalence]
    atom_charge = [one_of_k_encoding(t, [0, 1, 2, 3, 4, 5, 6]) for t in atom_charge]
    atom_explicit = [one_of_k_encoding(t, [0, 1, 2, 3, 4]) for t in atom_explicit]
    atom_f1 = torch.tensor([atom_number, atom_aromatic, atomHs], dtype=torch.float).t().contiguous()
    mol_all_nodes_feature = torch.cat([
        torch.FloatTensor(nodes_hybridization),
        torch.FloatTensor(nodes_type),
        torch.FloatTensor(atom_Hss),
        torch.FloatTensor(atom_imvalence),
        torch.FloatTensor(atom_charge),
        torch.FloatTensor(atom_explicit),
        atom_f1
    ], dim=-1)
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
    edge_type = [one_of_k_encoding(t, [BondType.SINGLE, BondType.DOUBLE,
                                       BondType.TRIPLE, BondType.AROMATIC]) for t in bond_type]
    edge_index = torch.LongTensor([row, col])
    perm = (edge_index[0] * num_atoms + edge_index[1]).argsort()
    edge_index = edge_index[:, perm]
    mol_all_edges_feature = torch.cat([
        torch.FloatTensor(edge_conj),
        torch.FloatTensor(edge_ring),
        torch.FloatTensor(edge_type)
    ], dim=-1)
    try:
        conf = mol.GetConformer()
        coords = conf.GetPositions()
        mol_all_coords_feature = torch.FloatTensor(coords)
    except (ValueError, AttributeError):
        try:
            AllChem.EmbedMolecule(mol, randomSeed=42, useRandomCoords=True)
            if AllChem.MMFFHasAllMoleculeParams(mol):
                AllChem.MMFFOptimizeMolecule(mol)
            else:
                AllChem.UFFOptimizeMolecule(mol)
            mol_all_coords_feature = torch.FloatTensor(mol.GetConformer().GetPositions())
        except:
            mol_all_coords_feature = torch.zeros((num_atoms, 3), dtype=torch.float)
    atom_matrix = GetAdjacencyMatrix(mol)
    graph = nx.from_numpy_array(atom_matrix)
    mol_dgl_graph = dgl.from_networkx(graph)
    mol_dgl_graph.ndata['h'] = torch.FloatTensor(mol_all_nodes_feature)
    return (mol_all_nodes_feature, mol_all_edges_feature, mol_all_coords_feature,
            mol_dgl_graph, edge_index, canonical_smiles, atom_type, mol)


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        pass
    return list(map(lambda s: x == s, allowable_set))


def generate_indexes(model, tokenizer, atom_type, mol_smiles):
    VALID_ELEMENTS = {'H', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I', 'Hg', 'B', 'Si', 'Na', 'K',
                      'Ca', 'Mg', 'Fe', 'Zn', 'Cu', 'Mn', 'Mo', 'Ni', 'Se', 'As', 'V', 'Al', 'Ti',
                      'Ge', 'Sn', 'Pb', 'Cd', 'W', 'Cr', 'Co', 'Li', 'Y', 'Zr', 'Nb', 'Ta', 'Re',
                      'Pd', 'Ag', 'Au', 'Pt', 'Rh', 'Ir'}
    mol_index = [idx for idx, atom in enumerate(atom_type)
                 if any(elem in atom for elem in VALID_ELEMENTS)]
    subwords = chemical_tokenize(mol_smiles)
    subinmol_index = [idx for idx, token in enumerate(subwords)
                      if any(c.isalpha() for c in token)]
    subinmol_words = [subwords[idx] for idx in subinmol_index]
    atom_symbols = [atom.split('_')[0].strip('@[]') for atom in atom_type]
    aligned_mol_idx, aligned_submol_idx = align_elements(atom_symbols, subinmol_words)
    tokenized = tokenizer(mol_smiles, return_tensors="pt")
    with torch.no_grad():
        embeddings = model(**tokenized).last_hidden_state
    if len(aligned_mol_idx) != len(aligned_submol_idx):
        logging.warning(f"Dimension mismatch: {mol_smiles}")
        return [], [], embeddings, mol_index
    return aligned_mol_idx, aligned_submol_idx, embeddings, mol_index


def chemical_tokenize(smiles):
    elements = ['Fe', 'Cl', 'Br', 'Na', 'Mg']
    tokens = []
    i = 0
    while i < len(smiles):
        matched = False
        for elem in elements:
            if smiles[i:i+2] == elem:
                tokens.append(elem)
                i += 2
                matched = True
                break
        if not matched:
            if smiles[i] in ['@', '[', ']']:
                tokens.append(smiles[i])
                i += 1
            elif smiles[i].isdigit():
                num = []
                while i < len(smiles) and smiles[i].isdigit():
                    num.append(smiles[i])
                    i += 1
                tokens.append(''.join(num))
            else:
                tokens.append(smiles[i])
                i += 1
    return tokens


from difflib import SequenceMatcher

def align_elements(atom_list, subword_list):
    matcher = SequenceMatcher(None, atom_list, subword_list)
    matches = matcher.get_matching_blocks()
    aligned_atom = []
    aligned_subword = []
    for match in matches:
        aligned_atom.extend(range(match.a, match.a + match.size))
        aligned_subword.extend(range(match.b, match.b + match.size))
    return aligned_atom, aligned_subword


def get_pdb_features(PDB_file, batch_converter, model1):

    protein, protein_graph, coords, res_list = generate_graph_feature(PDB_file)

    res_map, global_seq = build_global_residue_map_from_res_list(res_list)

    pr_dist, pr_theta, pr_phi, pr_tau = generate_inner_coor(
        protein.x, protein.node_s, protein.edge_index)

    global_pro_index = []
    for res in res_list:
        chain_id = res.full_id[2]
        _, res_num, ins_code = res.full_id[-1]
        key = (chain_id, res_num, ins_code)
        global_id = res_map[key]
        global_pro_index.append(global_id)    

    if len(global_pro_index) != protein.node_s.size(0):
        raise RuntimeError(f"Internal error: residue count {len(global_pro_index)} != node count {protein.node_s.size(0)}")

    prot_seq_for_esm = global_seq[:2024]
    prot = [(0, prot_seq_for_esm)]
    batch_labels, batch_strs, batch_tokens = batch_converter(prot)
    with torch.no_grad():
        results = model1(batch_tokens, repr_layers=[33], return_contacts=True)
    pro_token_repre = results["representations"][33].cpu()


    ionchannel_pdb_fea = Data(
        pro_graphs=protein_graph,
        pro_index=global_pro_index,
        batch_labels=batch_labels,
        batch_strs=batch_strs,
        pro_atoms_feats_s=protein.node_s,
        pro_atoms_feats_v=protein.node_v,
        pro_coords_feats=protein.x,
        pro_edges_feats_s=protein.edge_s,
        pro_edges_feats_v=protein.edge_v,
        pro_edge_index=protein.edge_index,
        pro_coords=coords,
        pro_token_repre=pro_token_repre,
        pro_fp=global_seq,
        pr_dist=pr_dist,
        pr_theta=pr_theta,
        pr_phi=pr_phi,
        pr_tau=pr_tau
    )
    return ionchannel_pdb_fea

def get_cache_path(smiles: str, uniprot_id: str, index: str, cache_dir) -> str:
    safe_smiles = hashlib.md5(smiles.encode()).hexdigest()[:16]
    safe_uniprot = uniprot_id.replace("/", "_")[:16]
    modified_index = index
    return os.path.join(cache_dir, f"{modified_index}_{safe_uniprot}_{safe_smiles}.pkl")


def assign_task_group(task_name: str, task_groups: dict, df: pd.DataFrame,
                     q75: float = None, q50: float = None, q25: float = None) -> str:
    for group, config in task_groups.items():
        if task_name in config['tasks']:
            return group
    task_size = len(df[df['ABB'] == task_name])
    explicit_mapping = {
        'KCa_CNG': 'tiny_volume', 'CaCC': 'tiny_volume', 'AQP': 'tiny_volume',
        'CFTR': 'low_volume', '5-HT3R': 'low_volume',
        'GABA_Gly': 'low_volume', 'VGKCs': 'low_volume',
        'Kir': 'mid_volume', 'K2P': 'mid_volume',
        'nAChRs': 'mid_volume', 'VGCCs': 'mid_volume', 'i-GluRs': 'mid_volume',
        'TRP': 'high_volume', 'P2X': 'high_volume', 'VGSCs': 'high_volume', 'Herg': 'high_volume'
    }
    if task_name in explicit_mapping:
        return explicit_mapping[task_name]
    raise ValueError(f"Unable to determine task group: {task_name}")


def build_task_groups(df: pd.DataFrame) -> Dict:
    abb_counts = df['ABB'].value_counts().to_dict()
    explicit_mapping = {
        'KCa_CNG': 'tiny_volume', 'CaCC': 'tiny_volume', 'AQP': 'tiny_volume',
        'CFTR': 'low_volume', '5-HT3R': 'low_volume',
        'GABA_Gly': 'low_volume', 'VGKCs': 'low_volume',
        'Kir': 'mid_volume', 'K2P': 'mid_volume',
        'nAChRs': 'mid_volume', 'VGCCs': 'mid_volume', 'i-GluRs': 'mid_volume',
        'TRP': 'high_volume', 'P2X': 'high_volume', 'VGSCs': 'high_volume', 'Herg': 'high_volume'
    }
    groups = {
        'high_volume': {'tasks': [], 'range': (10000, float('inf'))},
        'mid_volume': {'tasks': [], 'range': (3162, 10000)},
        'low_volume': {'tasks': [], 'range': (1000, 3162)},
        'tiny_volume': {'tasks': [], 'range': (0, 1000)}
    }
    for abb, count in abb_counts.items():
        if abb in explicit_mapping:
            group = explicit_mapping[abb]
        elif count >= 10000:
            group = 'high_volume'
        elif count >= 1700:
            group = 'mid_volume'
        elif count >= 400:
            group = 'low_volume'
        else:
            group = 'tiny_volume'
        groups[group]['tasks'].append(abb)
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

    return build_task_groups(df)  


def generate_feature_report(cache_dir):
    report_path = os.path.join(cache_dir, 'feature_generation_report.txt')
    with open(report_path, 'w') as f:
        f.write("Feature Generation Verification Report\n")
        f.write("=" * 50 + "\n")
        f.write(f"Generation time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        for group in ['high_volume', 'mid_volume', 'low_volume', 'tiny_volume']:
            group_dir = os.path.join(cache_dir, group)
            if os.path.exists(group_dir):
                files = [f for f in os.listdir(group_dir) if f.endswith('.pkl')]
                f.write(f"{group} group: {len(files)} feature files\n")
        failed_dir = os.path.join(cache_dir, 'failed')
        if os.path.exists(failed_dir):
            failed_files = [f for f in os.listdir(failed_dir) if f.endswith('.pkl')]
            f.write(f"\nFailed samples: {len(failed_files)}\n")
        f.write("\nFeature verification conclusion:\n")
        f.write("1. All successfully generated features have been saved to the corresponding group directory\n")
        f.write("2. Failed features have been recorded in failed_features.csv\n")
        f.write("3. Failed feature raw data have been saved in the failed/ directory\n")
        f.write("4. The generated features can be directly used for training code\n")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'feature_generation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    error_handler = FeatureErrorHandler('failed_features.csv')
    # file_path = '9CRS_SDF/GABAA9CRS.csv'
    file_path = '9CRS_SDF/9IXY.csv'
    df = pd.read_csv(file_path)
    success_count = 0
    failure_count = 0
    logging.info("Initializing models...")
    try:
        _esm_model, _chembert_model, _device_aware_tokenizer, _batch_converter = _initialize_models()
    except Exception as e:
        logging.error(f"Model initialization failed: {str(e)}")
        sys.exit(1)

    progress_bar = tqdm(total=len(df), desc="Generating features")
    os.makedirs(GlobalConfig.CACHE_DIR_train, exist_ok=True)
    for group in ['high_volume', 'mid_volume', 'low_volume', 'tiny_volume']:
        os.makedirs(os.path.join(GlobalConfig.CACHE_DIR_train, group), exist_ok=True)

    task_groups = build_task_groups(df)
    logging.info("Task grouping results:")
    for group_name, config in task_groups.items():
        logging.info(f"  {group_name}: {len(config['tasks'])} tasks, total samples: {config['sample_count']}")

    for index, row in df.iterrows():
        try:
            task_group = assign_task_group(row['ABB'], task_groups, df)
            logging.info(f"Processing row {index}: SMILES={row['SMILES']}, PDB_ID={row['PDB_ID']}, group={task_group}")

            # Molecular features
            try:
                sdf_filename = f"{row['SMILES_ID']}.sdf"
                search_dirs = ['9CRS_SDF']
                sdf_path = None
                for d in search_dirs:
                    cand = os.path.join(d, sdf_filename)
                    if os.path.exists(cand):
                        sdf_path = cand
                        break
                if sdf_path is None:
                    raise FileNotFoundError(f"SDF file not found: {sdf_filename}")
                mol_result = generate_smiles_nodes_edges_coords_graph_index_features(sdf_path)
                (mol_all_nodes_feature, mol_all_edges_feature,
                 mol_all_coords_feature, mol_dgl_graph,
                 edge_index, mol_smiles, atom_types, mol) = mol_result
                mol_dist, mol_theta, mol_phi, mol_tau = generate_inner_coor(
                    mol_all_coords_feature, mol_all_nodes_feature, edge_index)
                aligned_mol_idx, subinmol_index, embeddings, mol_index = generate_indexes(
                    _chembert_model, _device_aware_tokenizer, atom_types, mol_smiles)
                embeddings = embeddings.cpu()
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

            # Protein features
            try:
                pdb_path = os.path.join(GlobalConfig.PDB_file_path, f"{row['PDB_ID']}.pdb")
                if not os.path.exists(pdb_path):
                    raise FileNotFoundError(f"PDB file does not exist: {pdb_path}")
                prot_fea = get_pdb_features(pdb_path, _batch_converter, _esm_model)
            except Exception as e:
                raise ValueError(f"Protein feature generation failed: {str(e)}")

            validator = FeatureValidator()
            mol_errors = validator.validate_mol_feature(mol_fea)
            prot_errors = validator.validate_prot_feature(prot_fea)
            dim_errors = validator.validate_feature_dimensions(mol_fea, prot_fea)
            all_errors = mol_errors + prot_errors + dim_errors
            if all_errors:
                raise ValueError(f"Feature validation failed: {'; '.join(all_errors)}")

            fea_data = {
                'mol_fea': mol_fea,
                'prot_fea': prot_fea,
                'label': torch.tensor(row['Label'], dtype=torch.long),
                'task': row['ABB'],
                'group': task_group,
                'smiles': row['SMILES'],
                'uniprot_id': row['PDB_ID']
            }

            cache_dir_train = os.path.join(GlobalConfig.CACHE_DIR_train, task_group)
            os.makedirs(cache_dir_train, exist_ok=True)
            index_str = str(index)
            mindex = 'TW' + index_str
            cache_path = get_cache_path(row['SMILES'], mindex, row['PDB_ID'], cache_dir_train)
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump(fea_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            success_count += 1

        except Exception as e:
            failure_count += 1
            error_type = type(e).__name__
            error_msg = str(e)
            error_handler.record_failure(index, row, error_type, error_msg)
            logging.error(f"Error processing row {index}: {error_msg}")
            logging.debug(traceback.format_exc())
            try:
                failed_data = {
                    'index': index, 'row': row.to_dict(),
                    'error_type': error_type, 'error_message': error_msg,
                    'traceback': traceback.format_exc()
                }
                failed_dir = os.path.join(GlobalConfig.CACHE_DIR_train, 'failed')
                os.makedirs(failed_dir, exist_ok=True)
                failed_path = os.path.join(failed_dir, f"failed_{index}_{row['PDB_ID']}.pkl")
                with open(failed_path, 'wb') as f:
                    pickle.dump(failed_data, f)
            except Exception as save_error:
                logging.error(f"Error saving failed feature: {str(save_error)}")
        finally:
            progress_bar.update(1)
            progress_bar.set_postfix({'Success': success_count, 'Failed': failure_count})
            if index % 50 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

    progress_bar.close()
    error_handler.save_errors()
    logging.info("\n" + "=" * 50)
    logging.info(f"Feature generation completed")
    logging.info(f"Success: {success_count}/{len(df)} ({success_count/len(df)*100:.1f}%)")
    logging.info(f"Failed: {failure_count}")
    generate_feature_report(GlobalConfig.CACHE_DIR_train)
