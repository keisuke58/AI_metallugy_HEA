"""
FNO用データローダー
組成データを空間化してFNOで処理できる形式に変換
"""
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List
from pathlib import Path
from sklearn.preprocessing import RobustScaler
import pickle

from utils.data_utils import extract_composition_from_row, get_material_descriptors
from utils.element_properties import ELEMENT_LIST


class FNODataset(Dataset):
    """
    FNO用データセット
    組成データを1次元グリッドに空間化
    """
    
    def __init__(
        self,
        data_path: str,
        grid_size: int = 64,
        fit_scaler: bool = True,
        scaler_path: str = None
    ):
        """
        Args:
            data_path: CSVファイルのパス
            grid_size: グリッドサイズ（FNOの解像度）
            fit_scaler: スケーラーをフィットするか
            scaler_path: スケーラーの保存/読み込みパス
        """
        self.df = pd.read_csv(data_path)
        self.df = self.df.dropna(subset=['elastic_modulus'])
        self.grid_size = grid_size
        
        # 組成カラムを取得（無い場合もある）
        self.comp_cols = [col for col in self.df.columns if col.startswith('comp_')]
        
        # 材料記述子カラム（ベース）
        self.descriptor_cols = [
            'mixing_entropy', 'mixing_enthalpy', 'vec', 'delta_r',
            'delta_chi', 'mean_atomic_radius', 'mean_electronegativity', 'density'
        ]

        # 統合データセットには vec, delta_r などが無い場合があるので、無ければ0で追加
        for col in self.descriptor_cols:
            if col not in self.df.columns:
                # 欠損している記述子カラムは0で埋める
                self.df[col] = 0.0
        
        # スケーラーの設定
        self.scaler = None
        if fit_scaler:
            self.scaler = RobustScaler()
            descriptor_data = self.df[self.descriptor_cols].fillna(0).values
            self.scaler.fit(descriptor_data)
            
            if scaler_path:
                Path(scaler_path).parent.mkdir(parents=True, exist_ok=True)
                with open(scaler_path, 'wb') as f:
                    pickle.dump(self.scaler, f)
        else:
            if scaler_path and Path(scaler_path).exists():
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
        
        print(f"✅ FNO Dataset: {len(self.df)}サンプルを読み込みました")
        print(f"📊 グリッドサイズ: {grid_size}")
        print(f"📊 組成特徴量: {len(self.comp_cols)}個")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 組成を抽出
        compositions = extract_composition_from_row(row)
        
        # 組成を空間化（1次元グリッド）
        # 各元素をグリッド上の位置に配置
        composition_grid = np.zeros(self.grid_size)
        
        # 元素を原子番号順にソートして配置
        sorted_elements = sorted(compositions.items(), 
                                key=lambda x: ELEMENT_LIST.index(x[0]) if x[0] in ELEMENT_LIST else 999)
        
        for i, (elem, comp) in enumerate(sorted_elements[:self.grid_size]):
            if i < self.grid_size:
                composition_grid[i] = comp
        
        # 材料記述子を取得
        descriptors = []
        for col in self.descriptor_cols:
            val = row[col] if col in row and pd.notna(row[col]) else 0.0
            descriptors.append(float(val))
        
        # 正規化
        if self.scaler is not None:
            descriptors = self.scaler.transform([descriptors])[0]
        
        # マルチチャンネル入力: [組成グリッド, 材料記述子を拡張]
        # チャンネル1: 組成グリッド
        # チャンネル2: 材料記述子をグリッドに拡張
        descriptor_grid = np.tile(descriptors[:8], self.grid_size // 8 + 1)[:self.grid_size]
        
        # 入力: [grid_size, 2] (組成 + 記述子)
        input_grid = np.stack([composition_grid, descriptor_grid], axis=0)
        
        return {
            'input': torch.tensor(input_grid, dtype=torch.float32),  # [2, grid_size]
            'target': torch.tensor([row['elastic_modulus']], dtype=torch.float32),
            'descriptors': torch.tensor(descriptors, dtype=torch.float32),
        }


def collate_fno(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """FNO用バッチコレーター"""
    return {
        'input': torch.stack([item['input'] for item in batch]),  # [batch, 2, grid_size]
        'target': torch.stack([item['target'] for item in batch]),  # [batch, 1]
        'descriptors': torch.stack([item['descriptors'] for item in batch]),  # [batch, 8]
        'additional_features': torch.stack([item['descriptors'] for item in batch]),  # [batch, 8]
    }
