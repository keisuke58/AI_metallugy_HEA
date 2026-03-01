#!/usr/bin/env python3
"""
unified_datasetに特徴量を追加するスクリプト
5340サンプルのデータセットに必要な特徴量を追加
"""
import pandas as pd
import numpy as np
from pathlib import Path
import re
import sys

# 設定
BASE_DIR = Path(__file__).parent.parent
FINAL_DATA_DIR = BASE_DIR / "final_data"
PROCESSED_DATA_DIR = BASE_DIR / "processed_data"

# 元素の物理的性質
ELEMENT_PROPERTIES = {
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
    """組成文字列を解析"""
    if pd.isna(composition_str) or not isinstance(composition_str, str):
        return {}
    
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
    
    if total > 0:
        elements = {k: v/total for k, v in elements.items()}
    
    return elements

def calculate_features(composition_dict):
    """組成から特徴量を計算"""
    if not composition_dict:
        return {}
    
    descriptors = {}
    
    # 原子半径関連
    radii = [ELEMENT_PROPERTIES[elem]['radius'] * comp 
             for elem, comp in composition_dict.items() 
             if elem in ELEMENT_PROPERTIES]
    
    if radii:
        descriptors['mean_atomic_radius'] = np.mean(radii)
        descriptors['delta_r'] = np.max(radii) - np.min(radii) if len(radii) > 1 else 0
    
    # 電気陰性度関連
    electronegativities = [ELEMENT_PROPERTIES[elem]['electronegativity'] * comp 
                          for elem, comp in composition_dict.items() 
                          if elem in ELEMENT_PROPERTIES]
    
    if electronegativities:
        descriptors['mean_electronegativity'] = np.mean(electronegativities)
        descriptors['delta_chi'] = np.max(electronegativities) - np.min(electronegativities) if len(electronegativities) > 1 else 0
    
    # VEC (Valence Electron Concentration)
    vec_values = [ELEMENT_PROPERTIES[elem]['valence'] * comp 
                  for elem, comp in composition_dict.items() 
                  if elem in ELEMENT_PROPERTIES]
    if vec_values:
        descriptors['vec'] = np.sum(vec_values)
    
    # 組成カラムを追加
    for elem in ELEMENT_PROPERTIES.keys():
        descriptors[f'comp_{elem}'] = composition_dict.get(elem, 0.0)
    
    return descriptors

def main():
    """メイン関数"""
    input_file = FINAL_DATA_DIR / "unified_dataset_with_mpea_20260123_175026.csv"
    output_file = PROCESSED_DATA_DIR / "data_with_features_5340.csv"
    
    if not input_file.exists():
        print(f"❌ ファイルが見つかりません: {input_file}")
        return
    
    print("=" * 80)
    print("unified_datasetに特徴量を追加")
    print("=" * 80)
    print(f"📊 入力ファイル: {input_file}")
    
    df = pd.read_csv(input_file)
    print(f"✅ {len(df)}行のデータを読み込みました")
    
    # 組成を解析
    print("\n📊 組成を解析中...")
    compositions = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="組成解析"):
        comp_str = str(row.get('composition', row.get('alloy_name', '')))
        comp = parse_composition(comp_str)
        compositions.append(comp)
    
    # 特徴量を計算
    print("\n📊 特徴量を計算中...")
    features_list = []
    for comp in tqdm(compositions, desc="特徴量計算"):
        features = calculate_features(comp)
        features_list.append(features)
    
    # 特徴量をDataFrameに変換
    features_df = pd.DataFrame(features_list)
    
    # 元のデータと結合
    result_df = pd.concat([df.reset_index(drop=True), features_df], axis=1)
    
    # 重複列を除去
    result_df = result_df.loc[:, ~result_df.columns.duplicated()]
    
    # 欠損値を0で埋める（組成関連の特徴量）
    feature_cols = ['mixing_entropy', 'mixing_enthalpy', 'vec', 'delta_r', 
                   'delta_chi', 'mean_atomic_radius', 'mean_electronegativity', 'density']
    for col in feature_cols:
        if col in result_df.columns:
            result_df[col] = result_df[col].fillna(0)
    
    # 保存
    output_file.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_file, index=False)
    
    print(f"\n✅ 特徴量付きデータを保存しました: {output_file}")
    print(f"📊 最終データ数: {len(result_df)}行")
    print(f"📊 特徴量数: {len(result_df.columns)}個")
    print(f"📊 必要な特徴量の有無:")
    for col in feature_cols:
        if col in result_df.columns:
            missing = result_df[col].isna().sum()
            print(f"   {col}: ✅ (欠損: {missing})")
        else:
            print(f"   {col}: ❌ なし")

if __name__ == "__main__":
    from tqdm import tqdm
    main()
