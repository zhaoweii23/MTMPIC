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
from typing import Dict, Optional
from torch.utils.data import Dataset, DataLoader, Subset

from model_train import MTLModel, Trainer, ModelConfig


FEATURE_CACHE_DIR = 'feature_cache/high_volume' 
SPLIT_DIR = 'split_datasets'
RESULT_DIR = 'experiment_results'
os.makedirs(RESULT_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ABB 到 volume group 的映射
ABB_TO_GROUP = {
    'KCa_CNG': 'tiny_volume',
    'CaCC': 'tiny_volume',
    'AQP': 'tiny_volume',
    'CFTR': 'low_volume',
    '5-HT3R': 'low_volume',
    'GABA_Gly': 'low_volume',
    'VGKCs': 'low_volume',
    'Kir': 'mid_volume',
    'K2P': 'mid_volume',
    'nAChRs': 'mid_volume',
    'VGCCs': 'mid_volume',
    'i-GluRs': 'mid_volume',
    'TRP': 'high_volume',
    'P2X': 'high_volume',
    'VGSCs': 'high_volume',
    'Herg': 'high_volume'
}


class ExtendedTrainer(Trainer):
    def __init__(self, model, train_datasets, val_datasets=None, val_ratio=0.1):
        if val_datasets is None:
            train_datasets, val_datasets = self._split_train_val(train_datasets, val_ratio)
        super().__init__(model, train_datasets, val_datasets)

    def _split_train_val(self, datasets, val_ratio):
        train_ds, val_ds = {}, {}
        for group, dataset in datasets.items():
            n_val = max(1, int(len(dataset) * val_ratio))
            indices = np.random.permutation(len(dataset))
            val_idx = indices[:n_val]
            train_idx = indices[n_val:]
            train_ds[group] = Subset(dataset, train_idx)
            val_ds[group] = Subset(dataset, val_idx)
        return train_ds, val_ds

    def evaluate_on_loaders(self, loaders):
        original_val_loader = self.val_loader
        self.val_loader = loaders
        metrics = self.evaluate()
        self.val_loader = original_val_loader
        return metrics

class CSVSubsetDataset(Dataset):
    def __init__(self, csv_path: str, cache_root: str, group: Optional[str] = None,
                 df_with_group: Optional[pd.DataFrame] = None):
        self.cache_root = cache_root
        self.group = group

        if df_with_group is not None:
            self.df = df_with_group.copy()
        else:
            self.df = pd.read_csv(csv_path)


        if 'group' not in self.df.columns:
            if 'ABB' not in self.df.columns:
                raise KeyError("CSV must contain 'ABB' column for group mapping")
            self.df['group'] = self.df['ABB'].map(ABB_TO_GROUP).fillna('unknown')

        self.file_mapping = self._build_file_mapping(cache_root)
        self._add_match_keys()


        if len(self.df) > 0:
            sample_key = (self.df.iloc[0]['_target_id'], self.df.iloc[0]['_drug_id'])
            logging.info(f"Sample generated key: {sample_key}")
        if self.file_mapping:
            sample_map_key = list(self.file_mapping.keys())[0]
            logging.info(f"Sample file mapping key: {sample_map_key}")

        self._filter_missing_files()

        if group is not None:
            self.df = self.df[self.df['group'] == group].reset_index(drop=True)

        self.task_label_counts = self._scan_task_distribution()
        self._scanned_tasks = set(self.task_label_counts.keys())
        logging.info(f"CSVSubsetDataset (group={group}): {len(self.df)} samples")

    def _build_file_mapping(self, cache_root: str) -> Dict[tuple, str]:
        mapping = {}
        for root, _, files in os.walk(cache_root):
            for f in files:
                if not f.endswith('.pkl'):
                    continue
                name = f[:-4]
                if '_' not in name:
                    continue
                target_id = name.split('_', 1)[0]
                drug_id = name.rsplit('_', 1)[-1]
                mapping[(target_id, drug_id)] = os.path.join(root, f)
        logging.info(f"Found {len(mapping)} pkl files in {cache_root}")
        return mapping

    def _add_match_keys(self):
        target_col = 'Target_ID' if 'Target_ID' in self.df.columns else 'UniProt ID'
        drug_col = 'Drug_ID' if 'Drug_ID' in self.df.columns else 'SMILES'
        self.df['_target_id'] = self.df[target_col].astype(str)
        self.df['_drug_id'] = self.df[drug_col].astype(str)

    def _filter_missing_files(self):
        def exists(row):
            key = (row['_target_id'], row['_drug_id'])
            return key in self.file_mapping
        mask = self.df.apply(exists, axis=1)
        dropped = (~mask).sum()
        if dropped > 0:
            logging.warning(f"Dropping {dropped} samples")
            self.df = self.df[mask].reset_index(drop=True)

    def _scan_task_distribution(self):
        task_counts = defaultdict(lambda: defaultdict(int))
        for _, row in self.df.iterrows():
            task = row['ABB']
            label = int(row['Label'])
            task_counts[task][label] += 1
        return task_counts

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        key = (row['_target_id'], row['_drug_id'])
        with open(self.file_mapping[key], 'rb') as f:
            sample = pickle.load(f)

        mol_data = sample['mol_fea']
        prot_data = sample['prot_fea']

        from torch_geometric.data import Data
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
            mol_smiles=sample.get('smiles', '')
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
            pro_seq=prot_data.pro_fp
        )
        return {
            'mol_data': mol_obj,
            'prot_data': prot_obj,
            'label': torch.tensor(sample['label'], dtype=torch.long),
            'task': sample.get('task', row['ABB']),
            'group': sample.get('group', row['group'])
        }


def create_datasets_from_csv(csv_path: str, cache_root: str) -> Dict[str, CSVSubsetDataset]:
    full_dataset = CSVSubsetDataset(csv_path, cache_root, group=None)
    groups = full_dataset.df['group'].unique()
    datasets = {}
    for grp in groups:
        datasets[grp] = CSVSubsetDataset(
            csv_path, cache_root, group=grp,
            df_with_group=full_dataset.df
        )
    return datasets



def run_single_experiment(scenario: str, seed: int, train_csv: str,
                          val_csv: Optional[str], test_csv: str) -> Dict[str, float]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(7)
        torch.cuda.manual_seed_all(seed)

    train_datasets = create_datasets_from_csv(train_csv, FEATURE_CACHE_DIR)
    val_datasets = create_datasets_from_csv(val_csv, FEATURE_CACHE_DIR) if val_csv else None
    test_datasets = create_datasets_from_csv(test_csv, FEATURE_CACHE_DIR)

    task_groups = {}
    for group, dataset in train_datasets.items():
        tasks = list(dataset._scanned_tasks)
        if not tasks:
            tasks = [f'default_{group}']
        task_groups[group] = {'tasks': tasks, 'class_weights': dataset.task_label_counts}

    model = MTLModel(task_groups).to(ModelConfig.device)
    trainer = ExtendedTrainer(model, train_datasets, val_datasets, val_ratio=0.1)
    trainer.train()

    test_loaders = {}
    for group, ds in test_datasets.items():
        test_loaders[group] = DataLoader(
            ds, batch_size=ModelConfig.batch_sizes.get(group, 16),
            shuffle=False, collate_fn=trainer._collate_fn, num_workers=0
        )
    test_metrics = trainer.evaluate_on_loaders(test_loaders)

    return {
        'balanced_accuracy': test_metrics['global']['balanced_accuracy'],
        'macro_f1': test_metrics['global']['macro_f1'],
        'macro_auc': test_metrics['global'].get('roc_auc_ovr', float('nan'))
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenarios', nargs='+', default=['random', 'drug_cs', 'target_cs', 'pair_cs'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 456, 789, 1024])
    args = parser.parse_args()

    all_results = {}
    for scenario in args.scenarios:
        scenario_results = []
        for seed in args.seeds:
            logging.info(f"Running {scenario} seed {seed}")

            if scenario == 'random':
                train_csv = os.path.join(SPLIT_DIR, 'train_random.csv')
                val_csv = os.path.join(SPLIT_DIR, 'validation_random.csv')
                test_csv = os.path.join(SPLIT_DIR, 'test_random.csv')
            else:
                scenario_name_map = {
                    'drug_cs': 'drug_cold_start',
                    'target_cs': 'target_cold_start',
                    'pair_cs': 'pair_cold_start'
                }
                base_name = scenario_name_map.get(scenario, scenario)
                train_csv = os.path.join(SPLIT_DIR, f'train_{base_name}.csv')
                val_csv = None
                test_csv = os.path.join(SPLIT_DIR, f'test_{base_name}.csv')

            metrics = run_single_experiment(scenario, seed, train_csv, val_csv, test_csv)
            metrics['seed'] = seed
            scenario_results.append(metrics)

            with open(os.path.join(RESULT_DIR, f'{scenario}_seed{seed}_metrics.json'), 'w') as f:
                json.dump(metrics, f, indent=2)

        baccs = [r['balanced_accuracy'] for r in scenario_results]
        f1s = [r['macro_f1'] for r in scenario_results]
        aucs = [r['macro_auc'] for r in scenario_results]
        all_results[scenario] = {
            'balanced_accuracy_mean': np.mean(baccs), 'balanced_accuracy_std': np.std(baccs),
            'macro_f1_mean': np.mean(f1s), 'macro_f1_std': np.std(f1s),
            'macro_auc_mean': np.mean(aucs), 'macro_auc_std': np.std(aucs),
            'per_seed': scenario_results
        }
        logging.info(f"{scenario} summary: BAcc = {np.mean(baccs):.4f} ± {np.std(baccs):.4f}")

    with open(os.path.join(RESULT_DIR, 'all_scenarios_summary.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    generate_latex_table(all_results)


def generate_latex_table(results):
    latex = [
        "\\begin{table}[htbp]", "\\centering",
        "\\caption{Performance across different data split scenarios (mean ± std).}",
        "\\label{tab:cold_start_results}", "\\begin{tabular}{lccc}", "\\hline",
        "Scenario & Balanced Accuracy & Macro F1 & Macro AUC \\\\", "\\hline"
    ]
    names = {'random': 'Random', 'drug_cs': 'Drug Cold-Start',
             'target_cs': 'Target Cold-Start', 'pair_cs': 'Pair Cold-Start'}
    for key, name in names.items():
        if key in results:
            r = results[key]
            latex.append(f"{name} & ${r['balanced_accuracy_mean']:.3f} \\pm {r['balanced_accuracy_std']:.3f}$ & "
                         f"${r['macro_f1_mean']:.3f} \\pm {r['macro_f1_std']:.3f}$ & "
                         f"${r['macro_auc_mean']:.3f} \\pm {r['macro_auc_std']:.3f}$ \\\\")
    latex.extend(["\\hline", "\\end{tabular}", "\\end{table}"])
    with open(os.path.join(RESULT_DIR, 'results_table.tex'), 'w') as f:
        f.write('\n'.join(latex))


if __name__ == "__main__":
    main()
