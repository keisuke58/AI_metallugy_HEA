"""
データローダー: HEAデータをGNNとTransformer用に前処理
"""
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data, Batch
from torch.utils.data import Dataset
from typing import List, Tuple, Dict
import json
from pathlib import Path
from sklearn.preprocessing import RobustScaler
import pickle

# 元素の物理特性データベース（原子番号、原子半径、電気陰性度など）
ELEMENT_PROPERTIES = {
    'Ti': {'atomic_num': 22, 'radius': 147, 'electronegativity': 1.54, 'vec': 4},
    'Zr': {'atomic_num': 40, 'radius': 160, 'electronegativity': 1.33, 'vec': 4},
    'Hf': {'atomic_num': 72, 'radius': 159, 'electronegativity': 1.3, 'vec': 4},
    'Nb': {'atomic_num': 41, 'radius': 146, 'electronegativity': 1.6, 'vec': 5},
    'Ta': {'atomic_num': 73, 'radius': 146, 'electronegativity': 1.5, 'vec': 5},
    'V': {'atomic_num': 23, 'radius': 134, 'electronegativity': 1.63, 'vec': 5},
    'Cr': {'atomic_num': 24, 'radius': 128, 'electronegativity': 1.66, 'vec': 6},
    'Mo': {'atomic_num': 42, 'radius': 139, 'electronegativity': 2.16, 'vec': 6},
    'W': {'atomic_num': 74, 'radius': 139, 'electronegativity': 2.36, 'vec': 6},
    'Fe': {'atomic_num': 26, 'radius': 126, 'electronegativity': 1.83, 'vec': 8},
    'Co': {'atomic_num': 27, 'radius': 125, 'electronegativity': 1.88, 'vec': 9},
    'Ni': {'atomic_num': 28, 'radius': 124, 'electronegativity': 1.91, 'vec': 10},
    'Cu': {'atomic_num': 29, 'radius': 128, 'electronegativity': 1.9, 'vec': 11},
    'Al': {'atomic_num': 13, 'radius': 143, 'electronegativity': 1.61, 'vec': 3},
    'Mn': {'atomic_num': 25, 'radius': 127, 'electronegativity': 1.55, 'vec': 7},
    'Si': {'atomic_num': 14, 'radius': 111, 'electronegativity': 1.9, 'vec': 4},
    'Sn': {'atomic_num': 50, 'radius': 145, 'electronegativity': 1.96, 'vec': 4},
}

ELEMENT_LIST = sorted(ELEMENT_PROPERTIES.keys())
ELEMENT_TO_IDX = {elem: idx for idx, elem in enumerate(ELEMENT_LIST)}


class HEADataset(Dataset):
    """HEAデータセット（GNN用）"""
    
    def __init__(self, data_path: str, max_elements: int = 10, normalize_target: bool = False, 
                 target_mean: float = None, target_std: float = None, 
                 normalize_features: bool = True, feature_scaler=None):
        """
        Args:
            data_path: CSVファイルのパス
            max_elements: 最大元素数
            normalize_target: ターゲット変数を正規化するか
            target_mean: 正規化用の平均値（推論時用）
            target_std: 正規化用の標準偏差（推論時用）
            normalize_features: 追加特徴量を正規化するか
            feature_scaler: 特徴量スケーラー（推論時用）
        """
        print(f"📂 データファイルを読み込み中: {data_path}")
        self.df = pd.read_csv(data_path)
        original_size = len(self.df)
        print(f"📊 読み込み前のデータ数: {original_size}")
        
        # elastic_modulusがNaNの行を削除
        self.df = self.df.dropna(subset=['elastic_modulus'])
        dropped_count = original_size - len(self.df)
        if dropped_count > 0:
            print(f"⚠️  elastic_modulusがNaNのため {dropped_count} サンプルを削除しました")
        
        self.max_elements = max_elements
        self.normalize_target = normalize_target
        self.normalize_features = normalize_features
        
        # ターゲット変数の正規化
        if normalize_target:
            if target_mean is None or target_std is None:
                # 訓練時: データから計算
                self.target_mean = self.df['elastic_modulus'].mean()
                self.target_std = self.df['elastic_modulus'].std()
                print(f"📊 ターゲット正規化: mean={self.target_mean:.2f}, std={self.target_std:.2f}")
            else:
                # 推論時: 指定された値を使用
                self.target_mean = target_mean
                self.target_std = target_std
                print(f"📊 ターゲット正規化（推論用）: mean={self.target_mean:.2f}, std={self.target_std:.2f}")
        else:
            self.target_mean = None
            self.target_std = None
        
        # 追加特徴量の正規化（重要！）
        feature_cols = ['mixing_entropy', 'mixing_enthalpy', 'vec', 'delta_r', 
                       'delta_chi', 'mean_atomic_radius', 'mean_electronegativity', 'density']
        
        if normalize_features:
            if feature_scaler is None:
                # 訓練時: スケーラーをフィット
                from sklearn.preprocessing import RobustScaler
                self.feature_scaler = RobustScaler()
                feature_data = self.df[feature_cols].fillna(0).values
                self.feature_scaler.fit(feature_data)
                print(f"📊 追加特徴量の正規化（RobustScaler）を適用しました")
                
                # 特徴量の統計情報を表示
                print(f"📈 追加特徴量の統計（正規化前）:")
                for col in feature_cols:
                    if col in self.df.columns:
                        print(f"   {col}: min={self.df[col].min():.4f}, max={self.df[col].max():.4f}, "
                              f"mean={self.df[col].mean():.4f}, std={self.df[col].std():.4f}")
            else:
                # 推論時: 指定されたスケーラーを使用
                self.feature_scaler = feature_scaler
                print(f"📊 追加特徴量の正規化（推論用）を適用しました")
        else:
            self.feature_scaler = None
        
        # 組成カラムを取得
        self.comp_cols = [col for col in self.df.columns if col.startswith('comp_')]
        
        print(f"✅ 最終的に {len(self.df)} サンプルを読み込みました")
        print(f"📊 組成特徴量: {len(self.comp_cols)}個")
        
        # データの統計情報を表示
        if len(self.df) > 0:
            print(f"📈 elastic_modulus統計: min={self.df['elastic_modulus'].min():.2f}, "
                  f"max={self.df['elastic_modulus'].max():.2f}, "
                  f"mean={self.df['elastic_modulus'].mean():.2f}, "
                  f"std={self.df['elastic_modulus'].std():.2f}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 組成を取得
        compositions = {}
        for col in self.comp_cols:
            element = col.replace('comp_', '')
            comp_value = row[col]
            if pd.notna(comp_value) and comp_value > 0:
                compositions[element] = comp_value
        
        # 元素リストと組成リストを作成
        elements = list(compositions.keys())
        comp_values = [compositions[elem] for elem in elements]
        
        # 正規化
        total_comp = sum(comp_values)
        if total_comp > 0:
            comp_values = [c / total_comp for c in comp_values]
        
        # グラフデータを作成
        num_nodes = len(elements)
        if num_nodes == 0:
            # フォールバック: デフォルトの元素を使用
            elements = ['Ti', 'Zr', 'Nb', 'Ta']
            comp_values = [0.25, 0.25, 0.25, 0.25]
            num_nodes = 4
        
        # ノード特徴量（元素の物理特性 + 組成）
        node_features = []
        for elem in elements:
            props = ELEMENT_PROPERTIES.get(elem, {
                'atomic_num': 0, 'radius': 0, 'electronegativity': 0, 'vec': 0
            })
            # 特徴量: [原子番号, 原子半径, 電気陰性度, VEC, 組成比]
            node_feat = [
                props['atomic_num'] / 100.0,  # 正規化
                props['radius'] / 200.0,
                props['electronegativity'] / 3.0,
                props['vec'] / 12.0,
                comp_values[elements.index(elem)]
            ]
            node_features.append(node_feat)
        
        # エッジを作成（完全グラフ: すべての元素間の相互作用）
        edge_index = []
        edge_attr = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    edge_index.append([i, j])
                    # エッジ特徴量: [組成積, 原子半径差, 電気陰性度差]
                    comp_prod = comp_values[i] * comp_values[j]
                    radius_diff = abs(ELEMENT_PROPERTIES.get(elements[i], {}).get('radius', 0) - 
                                     ELEMENT_PROPERTIES.get(elements[j], {}).get('radius', 0)) / 200.0
                    chi_diff = abs(ELEMENT_PROPERTIES.get(elements[i], {}).get('electronegativity', 0) - 
                                  ELEMENT_PROPERTIES.get(elements[j], {}).get('electronegativity', 0)) / 3.0
                    edge_attr.append([comp_prod, radius_diff, chi_diff])
        
        if len(edge_index) == 0:
            # 自己ループを追加（すべてのノードに）
            for i in range(num_nodes):
                edge_index.append([i, i])
                edge_attr.append([1.0, 0.0, 0.0])  # 自己ループの特徴量
        
        # edge_indexとedge_attrのサイズを確認
        if len(edge_index) != len(edge_attr):
            # サイズが一致しない場合、edge_attrを調整
            if len(edge_attr) < len(edge_index):
                # 不足分をゼロで埋める
                edge_attr.extend([[0.0, 0.0, 0.0]] * (len(edge_index) - len(edge_attr)))
            else:
                # 余分なものを削除
                edge_attr = edge_attr[:len(edge_index)]
        
        # 追加の特徴量（材料記述子）
        additional_features = []
        feature_cols = ['mixing_entropy', 'mixing_enthalpy', 'vec', 'delta_r', 
                       'delta_chi', 'mean_atomic_radius', 'mean_electronegativity', 'density']
        for col in feature_cols:
            val = row[col] if col in row and pd.notna(row[col]) else 0.0
            additional_features.append(float(val))
        
        # 追加特徴量の正規化（重要！）
        if self.normalize_features and self.feature_scaler is not None:
            features_array = np.array(additional_features).reshape(1, -1)
            additional_features = self.feature_scaler.transform(features_array).flatten().tolist()
        
        # ターゲット変数の処理
        target_value = row['elastic_modulus']
        if self.normalize_target and self.target_mean is not None and self.target_std is not None:
            # 正規化: (x - mean) / std
            target_value = (target_value - self.target_mean) / self.target_std
        
        # PyTorch Geometric Dataオブジェクト
        data = Data(
            x=torch.tensor(node_features, dtype=torch.float32),
            edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
            edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
            y=torch.tensor([target_value], dtype=torch.float32),
            num_nodes=num_nodes,
            additional_features=torch.tensor(additional_features, dtype=torch.float32)
        )
        
        return data


class TransformerDataset(Dataset):
    """Transformer用データセット（改善版：特徴量正規化付き）"""
    
    def __init__(self, data_path: str, max_length: int = 20, fit_scaler: bool = True, scaler_path: str = None,
                 normalize_target: bool = False, target_mean: float = None, target_std: float = None):
        """
        Args:
            data_path: CSVファイルのパス
            max_length: 最大シーケンス長
            fit_scaler: スケーラーをフィットするか（訓練時はTrue、推論時はFalse）
            scaler_path: スケーラーの保存パス
            normalize_target: ターゲット変数を正規化するか
            target_mean: 正規化用の平均値（推論時用）
            target_std: 正規化用の標準偏差（推論時用）
        """
        self.df = pd.read_csv(data_path)
        self.df = self.df.dropna(subset=['elastic_modulus'])
        self.max_length = max_length
        
        # 組成カラムを取得
        self.comp_cols = [col for col in self.df.columns if col.startswith('comp_')]
        
        # 語彙サイズ（元素数 + 特殊トークン）
        self.vocab_size = len(ELEMENT_LIST) + 3  # [PAD], [CLS], [SEP]
        self.pad_idx = 0
        self.cls_idx = 1
        self.sep_idx = 2
        self.elem_to_idx = {elem: idx + 3 for idx, elem in enumerate(ELEMENT_LIST)}
        
        # ターゲット変数の正規化
        self.normalize_target = normalize_target
        if normalize_target:
            if target_mean is None or target_std is None:
                # 訓練時: データから計算
                self.target_mean = self.df['elastic_modulus'].mean()
                self.target_std = self.df['elastic_modulus'].std()
            else:
                # 推論時: 指定された値を使用
                self.target_mean = target_mean
                self.target_std = target_std
        else:
            self.target_mean = None
            self.target_std = None
        
        # 特徴量の正規化（RobustScaler: 外れ値に頑健）
        feature_cols = ['mixing_entropy', 'mixing_enthalpy', 'vec', 'delta_r', 
                       'delta_chi', 'mean_atomic_radius', 'mean_electronegativity', 'density']
        
        # スケーラーの初期化
        self.scaler = None
        
        if fit_scaler:
            # 訓練時：スケーラーをフィット
            self.scaler = RobustScaler()
            feature_data = self.df[feature_cols].fillna(0).values
            self.scaler.fit(feature_data)
            
            # スケーラーを保存
            if scaler_path:
                Path(scaler_path).parent.mkdir(parents=True, exist_ok=True)
                with open(scaler_path, 'wb') as f:
                    pickle.dump(self.scaler, f)
        else:
            # 推論時：保存されたスケーラーを読み込み
            if scaler_path and Path(scaler_path).exists():
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
        
        print(f"✅ {len(self.df)}サンプル, 語彙:{self.vocab_size}, 特徴量正規化:{'有効' if self.scaler else '無効'}, ターゲット正規化:{'有効' if normalize_target else '無効'}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 組成を取得してシーケンスを作成
        compositions = {}
        for col in self.comp_cols:
            element = col.replace('comp_', '')
            comp_value = row[col]
            if pd.notna(comp_value) and comp_value > 0:
                compositions[element] = comp_value
        
        # 組成比でソート（降順）
        sorted_elements = sorted(compositions.items(), key=lambda x: x[1], reverse=True)
        
        # トークンIDシーケンスを作成
        token_ids = [self.cls_idx]  # [CLS]トークン
        comp_values = []
        
        for elem, comp in sorted_elements[:self.max_length - 2]:  # -2 for [CLS] and [SEP]
            if elem in self.elem_to_idx:
                token_ids.append(self.elem_to_idx[elem])
                comp_values.append(comp)
        
        token_ids.append(self.sep_idx)  # [SEP]トークン
        
        # パディング
        padding_length = self.max_length - len(token_ids)
        token_ids = token_ids + [self.pad_idx] * padding_length
        comp_values = comp_values + [0.0] * (self.max_length - len(comp_values) - 2)
        
        # 追加の特徴量（材料記述子）
        feature_cols = ['mixing_entropy', 'mixing_enthalpy', 'vec', 'delta_r', 
                       'delta_chi', 'mean_atomic_radius', 'mean_electronegativity', 'density']
        additional_features = []
        for col in feature_cols:
            val = row[col] if col in row and pd.notna(row[col]) else 0.0
            additional_features.append(float(val))
        
        # 特徴量を正規化
        if self.scaler is not None:
            features_array = np.array(additional_features).reshape(1, -1)
            additional_features = self.scaler.transform(features_array).flatten().tolist()
        
        # ターゲット変数の処理
        target_value = row['elastic_modulus']
        if self.normalize_target and self.target_mean is not None and self.target_std is not None:
            # 正規化: (x - mean) / std
            target_value = (target_value - self.target_mean) / self.target_std
        
        return {
            'token_ids': torch.tensor(token_ids, dtype=torch.long),
            'comp_values': torch.tensor(comp_values + [0.0, 0.0], dtype=torch.float32),  # [CLS]と[SEP]用
            'attention_mask': torch.tensor([1 if tid != self.pad_idx else 0 for tid in token_ids], dtype=torch.long),
            'additional_features': torch.tensor(additional_features, dtype=torch.float32),
            'target': torch.tensor([target_value], dtype=torch.float32)
        }


def collate_gnn(batch: List[Data]) -> Batch:
    """GNN用バッチコレーター"""
    # Batch.from_data_listはadditional_featuresを連結するので、
    # 手動でスタックしてからBatchを作成する方が安全
    batched = Batch.from_data_list(batch)
    
    # additional_featuresを正しくスタック
    if hasattr(batch[0], 'additional_features'):
        additional_features_list = [data.additional_features for data in batch]
        # 各サンプルのadditional_featuresは(8,)の形状
        # これを(batch_size, 8)にスタック
        batched.additional_features = torch.stack(additional_features_list, dim=0)
    
    return batched


def collate_transformer(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Transformer用バッチコレーター"""
    return {
        'token_ids': torch.stack([item['token_ids'] for item in batch]),
        'comp_values': torch.stack([item['comp_values'] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
        'additional_features': torch.stack([item['additional_features'] for item in batch]),
        'target': torch.stack([item['target'] for item in batch])
    }
