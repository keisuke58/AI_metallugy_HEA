#!/usr/bin/env python3
"""
ダウンロードしたデータを分析するスクリプト
"""

import pandas as pd
import numpy as np
from pathlib import Path
import zipfile

# 設定
BASE_DIR = Path(__file__).parent.parent
RAW_DATA_DIR = BASE_DIR / "raw_data"

def analyze_disma_dataset():
    """
    DISMA Research HEA Datasetを分析
    """
    print("=" * 60)
    print("DISMA Research HEA Dataset 分析")
    print("=" * 60)
    
    disma_dir = RAW_DATA_DIR / "disma_hea_dataset"
    
    if not disma_dir.exists():
        print("❌ ディレクトリが存在しません")
        return None
    
    # ファイルを探す
    data_files = []
    for ext in ['*.csv', '*.xlsx', '*.xls', '*.txt', '*.json']:
        data_files.extend(list(disma_dir.glob(ext)))
        data_files.extend(list(disma_dir.rglob(ext)))
    
    if not data_files:
        print("❌ データファイルが見つかりませんでした")
        return None
    
    print(f"\n✅ {len(data_files)}個のファイルが見つかりました:")
    for f in data_files:
        size = f.stat().st_size / 1024  # KB
        print(f"   - {f.name} ({size:.2f} KB)")
    
    # 最初のファイルを読み込んで分析
    for file_path in data_files:
        try:
            if file_path.suffix.lower() == '.csv':
                df = pd.read_csv(file_path)
            elif file_path.suffix.lower() in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
            else:
                continue
            
            print(f"\n📁 ファイル: {file_path.name}")
            print(f"📊 データ形状: {df.shape}")
            print(f"📋 カラム: {list(df.columns)}")
            
            # 弾性率データの有無を確認
            elastic_modulus_keywords = ['modulus', 'elastic', 'young', 'E', 'GPa', 'youngs']
            has_elastic_modulus = any(
                keyword.lower() in str(col).lower() 
                for col in df.columns 
                for keyword in elastic_modulus_keywords
            )
            
            if has_elastic_modulus:
                print("✅ 弾性率データが見つかりました")
                # 弾性率カラムを探す
                elastic_cols = [col for col in df.columns 
                               if any(kw in str(col).lower() for kw in elastic_modulus_keywords)]
                print(f"   弾性率カラム: {elastic_cols}")
                if elastic_cols:
                    print(f"   データ数: {df[elastic_cols[0]].notna().sum()}")
                    print(f"   統計: {df[elastic_cols[0]].describe()}")
            else:
                print("⚠️  弾性率データが見つかりませんでした")
            
            # 最初の数行を表示
            print(f"\n📄 データの最初の5行:")
            print(df.head())
            
            return df
            
        except Exception as e:
            print(f"⚠️  {file_path.name}の読み込みに失敗: {e}")
            continue
    
    return None

def analyze_mpea_dataset():
    """
    MPEA Mechanical Properties Databaseを分析
    """
    print("\n" + "=" * 60)
    print("MPEA Mechanical Properties Database 分析")
    print("=" * 60)
    
    mpea_dir = RAW_DATA_DIR / "mpea_mechanical_properties"
    
    if not mpea_dir.exists():
        print("❌ ディレクトリが存在しません")
        return None
    
    # ファイルを探す
    data_files = []
    for ext in ['*.csv', '*.xlsx', '*.xls', '*.txt', '*.json']:
        data_files.extend(list(mpea_dir.glob(ext)))
        data_files.extend(list(mpea_dir.rglob(ext)))
    
    if not data_files:
        print("❌ データファイルが見つかりませんでした")
        return None
    
    print(f"\n✅ {len(data_files)}個のファイルが見つかりました:")
    for f in data_files:
        size = f.stat().st_size / 1024  # KB
        print(f"   - {f.name} ({size:.2f} KB)")
    
    # 最初のファイルを読み込んで分析
    for file_path in data_files:
        try:
            if file_path.suffix.lower() == '.csv':
                df = pd.read_csv(file_path)
            elif file_path.suffix.lower() in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
            else:
                continue
            
            print(f"\n📁 ファイル: {file_path.name}")
            print(f"📊 データ形状: {df.shape}")
            print(f"📋 カラム: {list(df.columns)}")
            
            # 弾性率データの有無を確認
            elastic_modulus_keywords = ['modulus', 'elastic', 'young', 'E', 'GPa', 'youngs']
            has_elastic_modulus = any(
                keyword.lower() in str(col).lower() 
                for col in df.columns 
                for keyword in elastic_modulus_keywords
            )
            
            if has_elastic_modulus:
                print("✅ 弾性率データが見つかりました")
                # 弾性率カラムを探す
                elastic_cols = [col for col in df.columns 
                               if any(kw in str(col).lower() for kw in elastic_modulus_keywords)]
                print(f"   弾性率カラム: {elastic_cols}")
                if elastic_cols:
                    print(f"   データ数: {df[elastic_cols[0]].notna().sum()}")
                    print(f"   統計: {df[elastic_cols[0]].describe()}")
            else:
                print("⚠️  弾性率データが見つかりませんでした")
            
            # 最初の数行を表示
            print(f"\n📄 データの最初の5行:")
            print(df.head())
            
            return df
            
        except Exception as e:
            print(f"⚠️  {file_path.name}の読み込みに失敗: {e}")
            continue
    
    return None

if __name__ == "__main__":
    disma_df = analyze_disma_dataset()
    mpea_df = analyze_mpea_dataset()
    
    print("\n" + "=" * 60)
    print("分析結果サマリー")
    print("=" * 60)
    
    if disma_df is not None:
        print(f"\n✅ DISMA Dataset: {disma_df.shape[0]}行, {disma_df.shape[1]}列")
    else:
        print("\n❌ DISMA Dataset: 分析できませんでした")
    
    if mpea_df is not None:
        print(f"\n✅ MPEA Dataset: {mpea_df.shape[0]}行, {mpea_df.shape[1]}列")
    else:
        print("\n❌ MPEA Dataset: 分析できませんでした")
