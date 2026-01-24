#!/usr/bin/env python3
"""
特徴量エンジニアリングスクリプト

組成情報から材料記述子を計算
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re

# 設定
BASE_DIR = Path(__file__).parent.parent
PROCESSED_DATA_DIR = BASE_DIR / "processed_data"

# 元素の物理的性質（原子半径、電気陰性度、価電子数）
ELEMENT_PROPERTIES = {
    # 原子半径 (pm), 電気陰性度, 価電子数
    'Ti': {'radius': 147, 'electronegativity': 1.54, 'valence': 4},
    'Zr': {'radius': 160, 'electronegativity': 1.33, 'valence': 4},
    'Hf': {'radius': 159, 'electronegativity': 1.3, 'valence': 4},
    'Nb': {'radius': 146, 'electronegativity': 1.6, 'valence': 5},
    'Ta': {'radius': 146, 'electronegativity': 1.5, 'valence': 5},
    'V': {'radius': 134, 'electronegativity': 1.63, 'valence': 5},
    'Cr': {'radius': 128, 'electronegativity': 1.66, 'valence': 6},
    'Mo': {'radius': 139, 'electronegativity': 2.16, 'valence': 6},
    'W': {'radius': 139, 'electronegativity': 2.36, 'valence': 6},
    'Fe': {'radius': 126, 'electronegativity': 1.83, 'valence': 8},
    'Co': {'radius': 125, 'electronegativity': 1.88, 'valence': 9},
    'Ni': {'radius': 124, 'electronegativity': 1.91, 'valence': 10},
    'Cu': {'radius': 128, 'electronegativity': 1.9, 'valence': 11},
    'Al': {'radius': 143, 'electronegativity': 1.61, 'valence': 3},
    'Mn': {'radius': 127, 'electronegativity': 1.55, 'valence': 7},
    'Si': {'radius': 111, 'electronegativity': 1.9, 'valence': 4},
    'Sn': {'radius': 145, 'electronegativity': 1.96, 'valence': 4},
}

def parse_composition(composition_str):
    """
    合金組成文字列を解析して元素と組成比を抽出
    
    例: "CoFeNi" -> {'Co': 0.33, 'Fe': 0.33, 'Ni': 0.33}
    例: "Al0.5CoCrCuFeNi" -> {'Al': 0.5, 'Co': 1, 'Cr': 1, 'Cu': 1, 'Fe': 1, 'Ni': 1}
    """
    if pd.isna(composition_str) or not isinstance(composition_str, str):
        return {}
    
    # 元素記号と数値を抽出
    pattern = r'([A-Z][a-z]?)(\d*\.?\d*)'
    matches = re.findall(pattern, composition_str)
    
    elements = {}
    total = 0
    
    for element, value_str in matches:
        if element in ELEMENT_PROPERTIES:
            if value_str == '':
                value = 1.0
            else:
                value = float(value_str)
            elements[element] = value
            total += value
    
    # 正規化（合計が1になるように）
    if total > 0:
        elements = {k: v/total for k, v in elements.items()}
    
    return elements

def calculate_material_descriptors(composition_dict):
    """
    組成から材料記述子を計算
    """
    if not composition_dict:
        return {}
    
    descriptors = {}
    
    # 原子半径関連
    radii = [ELEMENT_PROPERTIES[elem]['radius'] * comp 
             for elem, comp in composition_dict.items() 
             if elem in ELEMENT_PROPERTIES]
    
    if radii:
        descriptors['mean_atomic_radius'] = np.mean(radii)
        descriptors['std_atomic_radius'] = np.std(radii) if len(radii) > 1 else 0
        descriptors['max_atomic_radius'] = np.max(radii)
        descriptors['min_atomic_radius'] = np.min(radii)
        descriptors['delta_r'] = descriptors['max_atomic_radius'] - descriptors['min_atomic_radius']
    
    # 電気陰性度関連
    electronegativities = [ELEMENT_PROPERTIES[elem]['electronegativity'] * comp 
                          for elem, comp in composition_dict.items() 
                          if elem in ELEMENT_PROPERTIES]
    
    if electronegativities:
        descriptors['mean_electronegativity'] = np.mean(electronegativities)
        descriptors['std_electronegativity'] = np.std(electronegativities) if len(electronegativities) > 1 else 0
        descriptors['delta_chi'] = np.max(electronegativities) - np.min(electronegativities)
    
    # 価電子濃度（VEC）
    vec_values = [ELEMENT_PROPERTIES[elem]['valence'] * comp 
                 for elem, comp in composition_dict.items() 
                 if elem in ELEMENT_PROPERTIES]
    
    if vec_values:
        descriptors['vec'] = np.sum(vec_values)
    
    # 混合エントロピー（configurational entropy）
    n = len(composition_dict)
    if n > 1:
        # ΔS_mix = -R * Σ(x_i * ln(x_i))
        R = 8.314  # J/(mol·K)
        entropy = -R * sum(comp * np.log(comp) for comp in composition_dict.values() if comp > 0)
        descriptors['mixing_entropy'] = entropy
    else:
        descriptors['mixing_entropy'] = 0
    
    # 元素数
    descriptors['num_elements'] = len(composition_dict)
    
    # 各元素の組成比（主要元素のみ）
    for elem in ['Ti', 'Zr', 'Hf', 'Nb', 'Ta', 'V', 'Cr', 'Mo', 'W', 
                 'Fe', 'Co', 'Ni', 'Cu', 'Al', 'Mn', 'Si', 'Sn']:
        descriptors[f'comp_{elem}'] = composition_dict.get(elem, 0.0)
    
    return descriptors

def engineer_features(df):
    """
    データフレームに特徴量を追加
    """
    print("=" * 60)
    print("特徴量エンジニアリング")
    print("=" * 60)
    
    # 組成を解析
    print("\n📊 組成を解析中...")
    compositions = []
    for idx, row in df.iterrows():
        comp = parse_composition(str(row['alloy_name']))
        compositions.append(comp)
    
    # 材料記述子を計算
    print("📊 材料記述子を計算中...")
    descriptors_list = []
    for comp in compositions:
        desc = calculate_material_descriptors(comp)
        descriptors_list.append(desc)
    
    # 記述子をDataFrameに変換
    descriptors_df = pd.DataFrame(descriptors_list)
    
    # 元のデータフレームと結合
    result_df = pd.concat([df.reset_index(drop=True), descriptors_df], axis=1)
    
    print(f"\n✅ 特徴量エンジニアリング完了")
    print(f"📊 元の特徴量数: {len(df.columns)}")
    print(f"📊 追加後の特徴量数: {len(result_df.columns)}")
    print(f"📊 追加された特徴量: {len(descriptors_df.columns)}個")
    
    # 欠損値の確認
    missing = result_df.isnull().sum()
    if missing.sum() > 0:
        print(f"\n⚠️  欠損値:")
        print(missing[missing > 0])
    
    return result_df

if __name__ == "__main__":
    # 統合データを読み込む
    input_file = PROCESSED_DATA_DIR / "integrated_data.csv"
    
    if not input_file.exists():
        print(f"❌ ファイルが見つかりません: {input_file}")
        print("   先にデータ統合を実行してください: python scripts/integrate_data.py")
    else:
        df = pd.read_csv(input_file)
        print(f"✅ {len(df)}行のデータを読み込みました")
        
        # 特徴量エンジニアリング
        df_features = engineer_features(df)
        
        # 保存
        output_file = PROCESSED_DATA_DIR / "data_with_features.csv"
        df_features.to_csv(output_file, index=False)
        print(f"\n✅ 特徴量付きデータを保存しました: {output_file}")
