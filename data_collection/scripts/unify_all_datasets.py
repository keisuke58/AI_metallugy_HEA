#!/usr/bin/env python3
"""
すべてのデータセットを統一フォーマットに変換・統合するスクリプト
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).parent.parent
RAW_DATA_DIR = BASE_DIR / "raw_data"
PROCESSED_DATA_DIR = BASE_DIR / "processed_data"
COLLECTED_DATA_DIR = BASE_DIR / "collected_data"
FINAL_DATA_DIR = BASE_DIR / "final_data"
FINAL_DATA_DIR.mkdir(parents=True, exist_ok=True)

# 統一フォーマットのカラム定義
STANDARD_COLUMNS = [
    'alloy_name',           # 必須
    'elastic_modulus',      # 必須
    'source',              # 必須
    'composition',         # 推奨
    'phases',             # 推奨
    'yield_strength',      # 補助
    'ultimate_strength',   # 補助
    'hardness',           # 補助
    'elongation',         # 補助
    'density',            # 補助
    'mixing_entropy',     # 特徴量
    'mixing_enthalpy',    # 特徴量
    'valence_electron',   # 特徴量
    'year',               # メタデータ
    'notes'               # メタデータ
]

def standardize_dataset(df, source_name, column_mapping=None):
    """データセットを統一フォーマットに変換"""
    if df.empty:
        return pd.DataFrame()
    
    standardized = pd.DataFrame()
    
    # デフォルトのカラムマッピング
    default_mapping = {
        'alloy_name': ['alloy_name', 'Alloy', 'Material', 'Name', 'material_id', 'formula_pretty'],
        'elastic_modulus': ['elastic_modulus', 'Young\'s Modulus', 'Young\'s Modulus (GPa)', 
                          'E', 'E (GPa)', 'elastic_modulus_gpa', 'Young\'s Mod (GPa)'],
        'composition': ['composition', 'Composition', 'Formula', 'formula_pretty'],
        'phases': ['phases', 'Phases', 'Phases present', 'phase'],
        'yield_strength': ['yield_strength', 'Yield Strength', 'Yield Strength (MPa)', 'YS'],
        'ultimate_strength': ['ultimate_strength', 'Ultimate Tensile Strength', 'Ultimate Tensile Strength (MPa)', 'UTS'],
        'hardness': ['hardness', 'Hardness', 'Hardness (HVN)', 'HVN'],
        'elongation': ['elongation', 'Elongation', 'Elongation (%)'],
        'density': ['density', 'Density', 'Density (g/cm³)'],
        'mixing_entropy': ['mixing_entropy', 'Mixing Entropy', 'mixing_entropy.1'],
        'mixing_enthalpy': ['mixing_enthalpy', 'Mixing Enthalpy'],
        'valence_electron': ['valence_electron', 'Valence electron'],
        'year': ['year', 'Year'],
        'notes': ['notes', 'Notes', 'Note', 'Note on any unqiue processing or data features']
    }
    
    # カスタムマッピングがあればマージ
    if column_mapping:
        for key, value in column_mapping.items():
            if key in default_mapping:
                default_mapping[key].extend(value if isinstance(value, list) else [value])
    
    # カラムマッピングを適用
    for standard_col, possible_cols in default_mapping.items():
        found = False
        for col in possible_cols:
            if col in df.columns:
                standardized[standard_col] = df[col]
                found = True
                break
        
        # 見つからない場合はNaN
        if not found:
            standardized[standard_col] = np.nan
    
    # ソース情報を追加
    standardized['source'] = source_name
    
    # 弾性率の単位をGPaに統一
    if 'elastic_modulus' in standardized.columns:
        # 既にGPa単位と仮定（必要に応じて変換処理を追加）
        standardized['elastic_modulus'] = pd.to_numeric(
            standardized['elastic_modulus'], errors='coerce'
        )
    
    return standardized

def load_doe_osti():
    """DOE/OSTI Datasetを読み込んで標準化"""
    print("📊 DOE/OSTI Dataset読み込み中...")
    
    file_path = RAW_DATA_DIR / "doe_osti_dataset" / "youngsdata.xlsx"
    if not file_path.exists():
        print("   ⚠️  ファイルが見つかりません")
        return pd.DataFrame()
    
    try:
        df = pd.read_excel(file_path)
        standardized = standardize_dataset(df, 'DOE/OSTI')
        print(f"   ✅ {len(standardized)}サンプル")
        return standardized
    except Exception as e:
        print(f"   ❌ エラー: {e}")
        return pd.DataFrame()

def load_gorsse():
    """Gorsse Datasetを読み込んで標準化"""
    print("📊 Gorsse Dataset読み込み中...")
    
    # 抽出済みファイルを確認
    extracted_file = COLLECTED_DATA_DIR / "gorsse_elastic_modulus.csv"
    if extracted_file.exists():
        try:
            df = pd.read_csv(extracted_file)
            standardized = standardize_dataset(df, 'Gorsse')
            print(f"   ✅ {len(standardized)}サンプル")
            return standardized
        except Exception as e:
            print(f"   ❌ エラー: {e}")
    
    # 元ファイルから読み込み
    file_path = RAW_DATA_DIR / "gorsse_dataset" / "1-s2.0-S2352340920311100-mmc1.xlsx"
    if file_path.exists():
        try:
            df = pd.read_excel(file_path)
            standardized = standardize_dataset(df, 'Gorsse')
            print(f"   ✅ {len(standardized)}サンプル")
            return standardized
        except Exception as e:
            print(f"   ❌ エラー: {e}")
    
    print("   ⚠️  ファイルが見つかりません")
    return pd.DataFrame()

def load_latest_research():
    """最新研究データを読み込んで標準化"""
    print("📊 最新研究データ読み込み中...")
    
    file_path = RAW_DATA_DIR / "latest_research" / "latest_research.csv"
    if not file_path.exists():
        print("   ⚠️  ファイルが見つかりません")
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(file_path)
        standardized = standardize_dataset(df, 'Latest Research')
        print(f"   ✅ {len(standardized)}サンプル")
        return standardized
    except Exception as e:
        print(f"   ❌ エラー: {e}")
        return pd.DataFrame()

def load_materials_project():
    """Materials Projectデータを読み込んで標準化"""
    print("📊 Materials Projectデータ読み込み中...")
    
    mp_files = list(COLLECTED_DATA_DIR.glob("materials_project_*.csv"))
    if not mp_files:
        print("   ⚠️  ファイルが見つかりません")
        return pd.DataFrame()
    
    latest_file = max(mp_files, key=lambda p: p.stat().st_mtime)
    try:
        df = pd.read_csv(latest_file)
        standardized = standardize_dataset(df, 'Materials Project')
        print(f"   ✅ {len(standardized)}サンプル")
        return standardized
    except Exception as e:
        print(f"   ❌ エラー: {e}")
        return pd.DataFrame()

def load_refractory_hea():
    """Refractory HEA Elastic Constantsを読み込んで標準化"""
    print("📊 Refractory HEA Elastic Constants読み込み中...")
    
    file_path = COLLECTED_DATA_DIR / "refractory_hea_elastic_modulus.csv"
    if not file_path.exists():
        print("   ⚠️  ファイルが見つかりません")
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(file_path)
        standardized = standardize_dataset(df, 'Refractory HEA Elastic Constants')
        print(f"   ✅ {len(standardized)}サンプル")
        return standardized
    except Exception as e:
        print(f"   ❌ エラー: {e}")
        return pd.DataFrame()

def load_all_datasets():
    """すべてのデータセットを読み込む"""
    print("=" * 60)
    print("すべてのデータセットの読み込み")
    print("=" * 60)
    
    all_datasets = []
    
    # 各データセットを読み込み
    datasets = [
        load_doe_osti(),
        load_gorsse(),
        load_latest_research(),
        load_materials_project(),
        load_refractory_hea()
    ]
    
    for df in datasets:
        if not df.empty:
            all_datasets.append(df)
    
    return all_datasets

def remove_duplicates(df):
    """重複データを除去"""
    if df.empty:
        return df
    
    initial_count = len(df)
    
    # 合金名と弾性率で重複を判定
    if 'alloy_name' in df.columns and 'elastic_modulus' in df.columns:
        # 合金名が同じで弾性率が近い（±5 GPa以内）ものを重複とみなす
        df_sorted = df.sort_values('elastic_modulus')
        df_unique = df_sorted.drop_duplicates(subset=['alloy_name'], keep='first')
        
        removed = initial_count - len(df_unique)
        print(f"   重複除去: {removed}サンプル除去")
        
        return df_unique
    
    return df

def clean_data(df):
    """データをクリーニング"""
    if df.empty:
        return df
    
    initial_count = len(df)
    
    # 弾性率データの検証
    if 'elastic_modulus' in df.columns:
        df = df[df['elastic_modulus'].notna()].copy()
        df = df[df['elastic_modulus'] > 0].copy()
        df = df[df['elastic_modulus'] < 1000].copy()  # 異常値除去
    
    removed = initial_count - len(df)
    if removed > 0:
        print(f"   クリーニング: {removed}サンプル除去")
    
    return df

def main():
    """メイン関数"""
    print("=" * 60)
    print("データセット統一化スクリプト")
    print("=" * 60)
    print(f"開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # すべてのデータセットを読み込み
    all_datasets = load_all_datasets()
    
    if not all_datasets:
        print("\n❌ データセットが見つかりませんでした")
        return
    
    # 統合
    print("\n" + "=" * 60)
    print("データ統合")
    print("=" * 60)
    
    combined = pd.concat(all_datasets, ignore_index=True, sort=False)
    print(f"統合前: {len(combined)}サンプル")
    
    # 重複除去
    combined = remove_duplicates(combined)
    
    # クリーニング
    combined = clean_data(combined)
    
    print(f"最終データ数: {len(combined)}サンプル")
    
    # 統計情報
    print("\n" + "=" * 60)
    print("データ統計")
    print("=" * 60)
    
    if 'elastic_modulus' in combined.columns:
        print(f"弾性率範囲: {combined['elastic_modulus'].min():.2f} - {combined['elastic_modulus'].max():.2f} GPa")
        print(f"平均: {combined['elastic_modulus'].mean():.2f} GPa")
        print(f"中央値: {combined['elastic_modulus'].median():.2f} GPa")
    
    if 'source' in combined.columns:
        print("\nデータソース別:")
        source_counts = combined['source'].value_counts()
        for source, count in source_counts.items():
            print(f"   {source}: {count}サンプル")
    
    # 保存
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = FINAL_DATA_DIR / f"unified_dataset_{timestamp}.csv"
    combined.to_csv(output_file, index=False)
    print(f"\n✅ 統一データセットを保存しました: {output_file}")
    
    print("\n" + "=" * 60)
    print("✅ 統一化完了")
    print("=" * 60)
    print(f"終了時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
