#!/usr/bin/env python3
"""
データの統合とクリーニングスクリプト

すべてのデータセットを統合し、重複を除去してクリーニング
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re

# 設定
BASE_DIR = Path(__file__).parent.parent
RAW_DATA_DIR = BASE_DIR / "raw_data"
PROCESSED_DATA_DIR = BASE_DIR / "processed_data"
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

def load_doe_osti_data():
    """
    DOE/OSTI Datasetを読み込む
    """
    print("=" * 60)
    print("DOE/OSTI Dataset 読み込み")
    print("=" * 60)
    
    excel_file = RAW_DATA_DIR / "doe_osti_dataset" / "youngsdata.xlsx"
    
    if not excel_file.exists():
        print(f"❌ ファイルが見つかりません: {excel_file}")
        return None
    
    df = pd.read_excel(excel_file)
    print(f"✅ {len(df)}行のデータを読み込みました")
    print(f"📋 カラム: {list(df.columns)}")
    
    # 弾性率カラムを確認
    elastic_col = 'Young\'s Mod (GPa)'
    if elastic_col in df.columns:
        valid_data = df[elastic_col].notna().sum()
        print(f"✅ 弾性率データ: {valid_data}個")
    
    # 必要なカラムを選択
    result_df = pd.DataFrame({
        'alloy_name': df['Alloy'],
        'elastic_modulus': df[elastic_col],
        'source': 'DOE/OSTI'
    })
    
    # 追加の特徴量があれば追加
    feature_cols = ['Mixing Entropy', 'Mixing Enthalpy', 'Valence electron', 
                   'Diff. in atomic radii', 'Diff. Electronegativity', 
                   'Omega', 'Lambda']
    for col in feature_cols:
        if col in df.columns:
            result_df[col.lower().replace(' ', '_').replace('.', '')] = df[col]
    
    return result_df

def load_gorsse_data():
    """
    Gorsse Datasetを読み込む
    """
    print("\n" + "=" * 60)
    print("Gorsse Dataset 読み込み")
    print("=" * 60)
    
    excel_file = RAW_DATA_DIR / "gorsse_dataset" / "1-s2.0-S2352340920311100-mmc1.xlsx"
    
    if not excel_file.exists():
        print(f"❌ ファイルが見つかりません: {excel_file}")
        return None
    
    # Table 1を読み込む
    df = pd.read_excel(excel_file, sheet_name='Table 1', header=6)
    df = df.dropna(how='all')
    
    # 最初の説明行を削除
    if '3d TM HEAs' in str(df.iloc[0, 2]):
        df = df.iloc[1:].reset_index(drop=True)
    
    print(f"✅ {len(df)}行のデータを読み込みました")
    
    # 弾性率カラムを確認
    elastic_col = 'E# (GPa)'
    if elastic_col in df.columns:
        valid_data = df[elastic_col].notna().sum()
        print(f"✅ 弾性率データ: {valid_data}個")
    
    # 必要なカラムを選択
    result_df = pd.DataFrame({
        'alloy_name': df['Composition (atomic)'],
        'elastic_modulus': df[elastic_col],
        'source': 'Gorsse'
    })
    
    # 追加の特徴量があれば追加
    if 'Type of phases' in df.columns:
        result_df['phases'] = df['Type of phases']
    if 'r# (g/cm3)' in df.columns:
        result_df['density'] = df['r# (g/cm3)']
    if 'HV' in df.columns:
        result_df['hardness'] = df['HV']
    if 'sy (MPa)' in df.columns:
        result_df['yield_strength'] = df['sy (MPa)']
    
    return result_df

def load_latest_research_data():
    """
    最新研究データを読み込む
    """
    print("\n" + "=" * 60)
    print("最新研究データ 読み込み")
    print("=" * 60)
    
    csv_file = RAW_DATA_DIR / "latest_research" / "latest_research.csv"
    
    if not csv_file.exists():
        print(f"❌ ファイルが見つかりません: {csv_file}")
        return None
    
    df = pd.read_csv(csv_file)
    print(f"✅ {len(df)}行のデータを読み込みました")
    
    # 弾性率カラムを確認
    elastic_col = None
    for col in df.columns:
        if any(kw in str(col).lower() for kw in ['modulus', 'elastic', 'young', 'E', 'GPa']):
            elastic_col = col
            break
    
    if elastic_col:
        result_df = pd.DataFrame({
            'alloy_name': df.get('alloy', df.get('composition', df.index)),
            'elastic_modulus': df[elastic_col],
            'source': 'Latest Research'
        })
        return result_df
    else:
        print("⚠️  弾性率カラムが見つかりませんでした")
        return None

def integrate_all_data():
    """
    すべてのデータセットを統合
    """
    print("\n" + "=" * 60)
    print("データ統合")
    print("=" * 60)
    
    all_dataframes = []
    
    # 各データセットを読み込む
    doe_osti_df = load_doe_osti_data()
    if doe_osti_df is not None:
        all_dataframes.append(doe_osti_df)
    
    gorsse_df = load_gorsse_data()
    if gorsse_df is not None:
        all_dataframes.append(gorsse_df)
    
    latest_df = load_latest_research_data()
    if latest_df is not None:
        all_dataframes.append(latest_df)
    
    if not all_dataframes:
        print("❌ データが読み込めませんでした")
        return None
    
    # データを統合
    integrated_df = pd.concat(all_dataframes, ignore_index=True)
    print(f"\n✅ 統合完了: {len(integrated_df)}行")
    
    # 重複を除去（合金名と弾性率が同じ場合）
    before_dedup = len(integrated_df)
    integrated_df = integrated_df.drop_duplicates(subset=['alloy_name', 'elastic_modulus'], keep='first')
    after_dedup = len(integrated_df)
    print(f"📊 重複除去: {before_dedup} → {after_dedup}行（{before_dedup - after_dedup}個の重複を除去）")
    
    # 弾性率がNaNの行を除去
    before_clean = len(integrated_df)
    integrated_df = integrated_df.dropna(subset=['elastic_modulus'])
    after_clean = len(integrated_df)
    print(f"📊 NaN除去: {before_clean} → {after_clean}行（{before_clean - after_clean}個のNaNを除去）")
    
    # 統計情報
    print(f"\n📊 最終データ数: {len(integrated_df)}行")
    print(f"📊 弾性率範囲: {integrated_df['elastic_modulus'].min():.2f} - {integrated_df['elastic_modulus'].max():.2f} GPa")
    print(f"📊 平均: {integrated_df['elastic_modulus'].mean():.2f} GPa")
    
    # 目標範囲（30-90 GPa）内のデータ
    target_range = integrated_df[(integrated_df['elastic_modulus'] >= 30) & 
                                 (integrated_df['elastic_modulus'] <= 90)]
    print(f"⭐ 目標範囲（30-90 GPa）内: {len(target_range)}個")
    
    # データソース別の統計
    print(f"\n📊 データソース別:")
    for source in integrated_df['source'].unique():
        count = len(integrated_df[integrated_df['source'] == source])
        print(f"   - {source}: {count}個")
    
    return integrated_df

def save_integrated_data(df):
    """
    統合されたデータを保存
    """
    output_file = PROCESSED_DATA_DIR / "integrated_data.csv"
    df.to_csv(output_file, index=False)
    print(f"\n✅ 統合データを保存しました: {output_file}")
    return output_file

if __name__ == "__main__":
    integrated_df = integrate_all_data()
    if integrated_df is not None:
        save_integrated_data(integrated_df)
        print("\n" + "=" * 60)
        print("データ統合完了")
        print("=" * 60)
