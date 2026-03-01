"""
Neural ODE用データローダー
"""
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.data_utils import extract_composition_from_row
from utils.element_properties import ELEMENT_LIST


class NeuralODEDataset(Dataset):
    """
    Neural ODE用データセット
    """
    def __init__(self, data_path: str, grid_size: int = 64):
        """
        Args:
            data_path: CSVファイルのパス
            grid_size: 入力グリッドサイズ
        """
        self.df = pd.read_csv(data_path)
        self.df = self.df.dropna(subset=['elastic_modulus'])
        self.grid_size = grid_size
        
        self.comp_cols = [col for col in self.df.columns if col.startswith('comp_')]
        self.descriptor_cols = [
            'mixing_entropy', 'mixing_enthalpy', 'vec', 'delta_r',
            'delta_chi', 'mean_atomic_radius', 'mean_electronegativity', 'density'
        ]
        
        print(f"✅ Neural ODE Dataset: {len(self.df)}サンプルを読み込みました")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 組成を空間化
        compositions = extract_composition_from_row(row)
        composition_grid = np.zeros(self.grid_size)
        
        sorted_elements = sorted(compositions.items(),
                                key=lambda x: ELEMENT_LIST.index(x[0]) if x[0] in ELEMENT_LIST else 999)
        
        for i, (elem, comp) in enumerate(sorted_elements[:self.grid_size]):
            if i < self.grid_size:
                composition_grid[i] = comp
        
        # 材料記述子
        descriptors = []
        for col in self.descriptor_cols:
            val = row[col] if col in row and pd.notna(row[col]) else 0.0
            descriptors.append(float(val))
        
        return {
            'input': torch.tensor(composition_grid, dtype=torch.float32),
            'additional_features': torch.tensor(descriptors, dtype=torch.float32),
            'target': torch.tensor([row['elastic_modulus']], dtype=torch.float32),
        }


def collate_neural_ode(batch):
    """Neural ODE用バッチコレーター"""
    return {
        'input': torch.stack([item['input'] for item in batch]),
        'additional_features': torch.stack([item['additional_features'] for item in batch]),
        'target': torch.stack([item['target'] for item in batch]),
    }
