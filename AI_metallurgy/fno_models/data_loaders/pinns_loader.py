"""
PINNs用データローダー
"""
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.data_utils import extract_composition_from_row
from utils.element_properties import ELEMENT_LIST


class PINNsDataset(Dataset):
    """
    PINNs用データセット
    組成と材料記述子を結合した特徴量ベクトル
    """
    def __init__(self, data_path: str, max_elements: int = 17):
        """
        Args:
            data_path: CSVファイルのパス
            max_elements: 最大元素数
        """
        self.df = pd.read_csv(data_path)
        self.df = self.df.dropna(subset=['elastic_modulus'])
        self.max_elements = max_elements
        
        self.comp_cols = [col for col in self.df.columns if col.startswith('comp_')]
        self.descriptor_cols = [
            'mixing_entropy', 'mixing_enthalpy', 'vec', 'delta_r',
            'delta_chi', 'mean_atomic_radius', 'mean_electronegativity', 'density'
        ]
        
        print(f"✅ PINNs Dataset: {len(self.df)}サンプルを読み込みました")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 組成を抽出
        compositions = extract_composition_from_row(row)
        
        # 組成ベクトル（固定サイズ）
        composition_vector = np.zeros(self.max_elements)
        for i, elem in enumerate(ELEMENT_LIST[:self.max_elements]):
            composition_vector[i] = compositions.get(elem, 0.0)
        
        # 材料記述子
        descriptors = []
        for col in self.descriptor_cols:
            val = row[col] if col in row and pd.notna(row[col]) else 0.0
            descriptors.append(float(val))
        
        # 結合
        features = np.concatenate([composition_vector, descriptors])
        
        return {
            'input': torch.tensor(features, dtype=torch.float32),
            'target': torch.tensor([row['elastic_modulus']], dtype=torch.float32),
        }


def collate_pinns(batch):
    """PINNs用バッチコレーター"""
    return {
        'input': torch.stack([item['input'] for item in batch]),
        'target': torch.stack([item['target'] for item in batch]),
    }
