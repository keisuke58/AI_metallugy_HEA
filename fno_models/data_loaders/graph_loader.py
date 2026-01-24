"""
グラフベースモデル（MEGNet, CGCNN）用データローダー
"""
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data, Batch
from torch.utils.data import Dataset
from typing import List

from utils.data_utils import extract_composition_from_row, get_material_descriptors
from utils.element_properties import ELEMENT_PROPERTIES


class GraphDataset(Dataset):
    """
    グラフベースモデル用データセット
    MEGNetとCGCNNで共通使用
    """
    def __init__(self, data_path: str):
        """
        Args:
            data_path: CSVファイルのパス
        """
        self.df = pd.read_csv(data_path)
        self.df = self.df.dropna(subset=['elastic_modulus'])
        
        # 組成カラムを取得
        self.comp_cols = [col for col in self.df.columns if col.startswith('comp_')]
        
        print(f"✅ Graph Dataset: {len(self.df)}サンプルを読み込みました")
        print(f"📊 組成特徴量: {len(self.comp_cols)}個")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 組成を抽出
        compositions = extract_composition_from_row(row)
        
        # 元素リストと組成リストを作成
        elements = list(compositions.keys())
        comp_values = [compositions[elem] for elem in elements]
        
        if len(elements) == 0:
            # フォールバック
            elements = ['Ti', 'Zr', 'Nb', 'Ta']
            comp_values = [0.25, 0.25, 0.25, 0.25]
        
        num_nodes = len(elements)
        
        # ノード特徴量（元素の物理特性 + 組成）
        node_features = []
        for elem in elements:
            props = ELEMENT_PROPERTIES.get(elem, {
                'atomic_num': 0, 'radius': 0, 'electronegativity': 0, 'vec': 0
            })
            node_feat = [
                props['atomic_num'] / 100.0,
                props['radius'] / 200.0,
                props['electronegativity'] / 3.0,
                props['vec'] / 12.0,
                comp_values[elements.index(elem)]
            ]
            node_features.append(node_feat)
        
        # エッジを作成（完全グラフ）
        edge_index = []
        edge_attr = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    edge_index.append([i, j])
                    comp_prod = comp_values[i] * comp_values[j]
                    radius_diff = abs(
                        ELEMENT_PROPERTIES.get(elements[i], {}).get('radius', 0) -
                        ELEMENT_PROPERTIES.get(elements[j], {}).get('radius', 0)
                    ) / 200.0
                    chi_diff = abs(
                        ELEMENT_PROPERTIES.get(elements[i], {}).get('electronegativity', 0) -
                        ELEMENT_PROPERTIES.get(elements[j], {}).get('electronegativity', 0)
                    ) / 3.0
                    edge_attr.append([comp_prod, radius_diff, chi_diff])
        
        if len(edge_index) == 0:
            for i in range(num_nodes):
                edge_index.append([i, i])
                edge_attr.append([1.0, 0.0, 0.0])
        
        # 追加特徴量（29個全てを使用）
        additional_features = []
        
        # 材料記述子を計算（存在しない場合は計算）
        descriptors = get_material_descriptors(compositions)
        
        # 1. 組成特徴量（17元素）
        comp_elements = ['Ti', 'Zr', 'Hf', 'Nb', 'Ta', 'V', 'Cr', 'Mo', 'W', 
                        'Fe', 'Co', 'Ni', 'Cu', 'Al', 'Mn', 'Si', 'Sn']
        for elem in comp_elements:
            col = f'comp_{elem}'
            # データフレームから取得、なければ組成辞書から、それもなければ0
            if col in row and pd.notna(row[col]):
                val = float(row[col])
            elif elem in compositions:
                val = float(compositions[elem])
            else:
                val = 0.0
            additional_features.append(val)
        
        # 2. 原子半径関連（5個）
        # mean_atomic_radius, std_atomic_radius, max_atomic_radius, min_atomic_radius, delta_r
        if elements and comp_values:
            radii_list = [ELEMENT_PROPERTIES.get(elem, {}).get('radius', 0) for elem in elements]
            radii_with_comp = [(r, c) for r, c in zip(radii_list, comp_values) if r > 0]
            if radii_with_comp:
                radii_only = [r for r, _ in radii_with_comp]
                comp_weights = [c for _, c in radii_with_comp]
                # 重み付き平均
                mean_radius = np.average(radii_only, weights=comp_weights) if sum(comp_weights) > 0 else np.mean(radii_only)
                # 重み付き標準偏差
                if len(radii_only) > 1:
                    variance = np.average([(r - mean_radius)**2 for r in radii_only], weights=comp_weights)
                    std_radius = np.sqrt(variance) if variance > 0 else 0.0
                else:
                    std_radius = 0.0
                max_radius = np.max(radii_only)
                min_radius = np.min(radii_only)
                delta_r = max_radius - min_radius
            else:
                mean_radius = descriptors.get('mean_atomic_radius', 0.0)
                std_radius = 0.0
                max_radius = 0.0
                min_radius = 0.0
                delta_r = 0.0
        else:
            # データフレームから取得を試みる
            mean_radius = float(row['mean_atomic_radius']) if 'mean_atomic_radius' in row and pd.notna(row['mean_atomic_radius']) else descriptors.get('mean_atomic_radius', 0.0)
            std_radius = float(row['std_atomic_radius']) if 'std_atomic_radius' in row and pd.notna(row['std_atomic_radius']) else 0.0
            max_radius = float(row['max_atomic_radius']) if 'max_atomic_radius' in row and pd.notna(row['max_atomic_radius']) else 0.0
            min_radius = float(row['min_atomic_radius']) if 'min_atomic_radius' in row and pd.notna(row['min_atomic_radius']) else 0.0
            delta_r = float(row['delta_r']) if 'delta_r' in row and pd.notna(row['delta_r']) else descriptors.get('delta_r', 0.0)
        
        additional_features.extend([mean_radius, std_radius, max_radius, min_radius, delta_r])
        
        # 3. 電気陰性度関連（3個）
        # mean_electronegativity, std_electronegativity, delta_chi
        if elements and comp_values:
            chi_list = [ELEMENT_PROPERTIES.get(elem, {}).get('electronegativity', 0) for elem in elements]
            chi_with_comp = [(c, comp) for c, comp in zip(chi_list, comp_values) if c > 0]
            if chi_with_comp:
                chi_only = [c for c, _ in chi_with_comp]
                comp_weights = [comp for _, comp in chi_with_comp]
                # 重み付き平均
                mean_chi = np.average(chi_only, weights=comp_weights) if sum(comp_weights) > 0 else np.mean(chi_only)
                # 重み付き標準偏差
                if len(chi_only) > 1:
                    variance = np.average([(c - mean_chi)**2 for c in chi_only], weights=comp_weights)
                    std_chi = np.sqrt(variance) if variance > 0 else 0.0
                else:
                    std_chi = 0.0
                delta_chi = np.max(chi_only) - np.min(chi_only)
            else:
                mean_chi = descriptors.get('mean_electronegativity', 0.0)
                std_chi = 0.0
                delta_chi = 0.0
        else:
            # データフレームから取得を試みる
            mean_chi = float(row['mean_electronegativity']) if 'mean_electronegativity' in row and pd.notna(row['mean_electronegativity']) else descriptors.get('mean_electronegativity', 0.0)
            std_chi = float(row['std_electronegativity']) if 'std_electronegativity' in row and pd.notna(row['std_electronegativity']) else 0.0
            delta_chi = float(row['delta_chi']) if 'delta_chi' in row and pd.notna(row['delta_chi']) else descriptors.get('delta_chi', 0.0)
        
        additional_features.extend([mean_chi, std_chi, delta_chi])
        
        # 4. 電子・熱力学的記述子（3個）
        # vec, mixing_entropy, num_elements
        # VECは組成比で重み付けした平均
        vec_val = descriptors.get('vec', 0.0)
        # データフレームから取得を試みる（valence_electronカラムもチェック）
        if 'vec' in row and pd.notna(row['vec']):
            vec_val = float(row['vec'])
        elif 'valence_electron' in row and pd.notna(row['valence_electron']):
            vec_val = float(row['valence_electron'])
        elif elements and comp_values:
            # 組成から計算
            vec_list = [ELEMENT_PROPERTIES.get(elem, {}).get('vec', 0) for elem in elements]
            vec_val = np.average(vec_list, weights=comp_values) if sum(comp_values) > 0 else np.mean(vec_list)
        
        mixing_entropy_val = descriptors.get('mixing_entropy', 0.0)
        if 'mixing_entropy' in row and pd.notna(row['mixing_entropy']):
            mixing_entropy_val = float(row['mixing_entropy'])
        
        num_elements_val = len(compositions) if compositions else 0.0
        
        additional_features.extend([vec_val, mixing_entropy_val, num_elements_val])
        
        # 5. その他（1個）
        # density
        density_val = 0.0
        if 'density' in row and pd.notna(row['density']):
            density_val = float(row['density'])
        
        additional_features.append(density_val)
        
        # 合計29個の特徴量を確認
        assert len(additional_features) == 29, f"Expected 29 features, got {len(additional_features)}"
        
        # PyTorch Geometric Dataオブジェクト
        data = Data(
            x=torch.tensor(node_features, dtype=torch.float32),
            edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
            edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
            y=torch.tensor([row['elastic_modulus']], dtype=torch.float32),
            num_nodes=num_nodes,
            additional_features=torch.tensor(additional_features, dtype=torch.float32)
        )
        
        return data


def collate_graph(batch: List[Data]) -> Batch:
    """グラフ用バッチコレーター"""
    return Batch.from_data_list(batch)
