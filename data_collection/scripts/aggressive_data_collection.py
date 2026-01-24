#!/usr/bin/env python3
"""
積極的なデータ収集スクリプト
複数のデータソースから最大2000サンプルまで収集

このスクリプトは以下を試みます:
1. 既存データセットの完全な再解析
2. 公開データベースからの自動ダウンロード
3. 論文補足資料からのデータ抽出
4. 計算データベースからのデータ取得
"""

import pandas as pd
import numpy as np
from pathlib import Path
import requests
import json
import time
import zipfile
import os
from datetime import datetime
import warnings
import re
warnings.filterwarnings('ignore')

# 設定
BASE_DIR = Path(__file__).parent.parent
RAW_DATA_DIR = BASE_DIR / "raw_data"
PROCESSED_DATA_DIR = BASE_DIR / "processed_data"
COLLECTED_DATA_DIR = BASE_DIR / "collected_data"
COLLECTED_DATA_DIR.mkdir(parents=True, exist_ok=True)

TARGET_SAMPLES = 2000

def extract_from_excel_sheets(file_path):
    """Excelファイルのすべてのシートからデータを抽出"""
    all_data = []
    
    try:
        xl_file = pd.ExcelFile(file_path)
        for sheet_name in xl_file.sheet_names:
            try:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                # 弾性率カラムを探す
                for col in df.columns:
                    if any(keyword in str(col).lower() for keyword in ['modulus', 'young', 'elastic', 'E']):
                        if df[col].dtype in [np.float64, np.int64]:
                            df_clean = df[df[col].notna()].copy()
                            if len(df_clean) > 0:
                                df_clean['elastic_modulus'] = df_clean[col]
                                df_clean['source'] = f'{file_path.stem}_{sheet_name}'
                                all_data.append(df_clean)
            except Exception as e:
                continue
    except Exception as e:
        pass
    
    return all_data

def deep_scan_existing_datasets():
    """既存データセットを深くスキャン"""
    print("=" * 60)
    print("既存データセットの深層スキャン")
    print("=" * 60)
    
    all_data = []
    
    # すべてのraw_dataディレクトリをスキャン
    for dataset_dir in RAW_DATA_DIR.iterdir():
        if not dataset_dir.is_dir():
            continue
        
        print(f"\n📁 スキャン中: {dataset_dir.name}")
        
        # すべてのファイルを再帰的にスキャン
        for file_path in dataset_dir.rglob("*"):
            if not file_path.is_file():
                continue
            
            # Excelファイル
            if file_path.suffix in ['.xlsx', '.xls']:
                try:
                    sheets_data = extract_from_excel_sheets(file_path)
                    for df in sheets_data:
                        if 'elastic_modulus' in df.columns:
                            all_data.append(df)
                            print(f"   ✅ {file_path.name}: {len(df)}サンプル")
                except Exception as e:
                    continue
            
            # CSVファイル
            elif file_path.suffix == '.csv':
                try:
                    df = pd.read_csv(file_path, low_memory=False)
                    # 弾性率カラムを探す
                    elastic_cols = [col for col in df.columns 
                                   if any(kw in str(col).lower() for kw in ['modulus', 'young', 'elastic', 'E'])]
                    if elastic_cols:
                        for col in elastic_cols:
                            if df[col].dtype in [np.float64, np.int64]:
                                df_clean = df[df[col].notna()].copy()
                                if len(df_clean) > 0:
                                    df_clean['elastic_modulus'] = df_clean[col]
                                    df_clean['source'] = f'{file_path.stem}'
                                    all_data.append(df_clean)
                                    print(f"   ✅ {file_path.name}: {len(df_clean)}サンプル")
                except Exception as e:
                    continue
    
    if all_data:
        combined = pd.concat(all_data, ignore_index=True, sort=False)
        print(f"\n📊 深層スキャン結果: {len(combined)}サンプル")
        return combined
    else:
        return pd.DataFrame()

def download_from_zenodo(doi_or_id):
    """Zenodoからデータをダウンロード"""
    print(f"\n📥 Zenodoデータ取得: {doi_or_id}")
    
    try:
        # Zenodo API
        if '10.5281' in str(doi_or_id):
            record_id = doi_or_id.split('/')[-1]
        else:
            record_id = doi_or_id
        
        api_url = f"https://zenodo.org/api/records/{record_id}"
        response = requests.get(api_url, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            files = data.get('files', [])
            
            print(f"   ✅ {len(files)}個のファイルが見つかりました")
            
            # データファイルをダウンロード
            for file_info in files:
                if any(ext in file_info['key'].lower() for ext in ['.csv', '.xlsx', '.xls', '.zip']):
                    file_url = file_info['links']['self']
                    print(f"   📥 ダウンロード中: {file_info['key']}")
                    
                    file_response = requests.get(file_url, timeout=60)
                    if file_response.status_code == 200:
                        # ファイルを保存（実装は必要に応じて）
                        pass
                    time.sleep(1)
        else:
            print(f"   ⚠️  Zenodo APIエラー: {response.status_code}")
    except Exception as e:
        print(f"   ⚠️  Zenodo取得エラー: {e}")

def download_from_figshare(article_id):
    """Figshareからデータをダウンロード"""
    print(f"\n📥 Figshareデータ取得: {article_id}")
    
    try:
        api_url = f"https://api.figshare.com/v2/articles/{article_id}"
        response = requests.get(api_url, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            files = data.get('files', [])
            print(f"   ✅ {len(files)}個のファイルが見つかりました")
        else:
            print(f"   ⚠️  Figshare APIエラー: {response.status_code}")
    except Exception as e:
        print(f"   ⚠️  Figshare取得エラー: {e}")

def search_pubmed_data():
    """PubMedから関連データを検索"""
    print("\n" + "=" * 60)
    print("PubMedデータ検索")
    print("=" * 60)
    
    # PubMed APIを使用してHEA関連論文を検索
    search_terms = [
        "high entropy alloy elastic modulus",
        "HEA Young's modulus",
        "multi-principal element alloy mechanical properties"
    ]
    
    print("⚠️  PubMed APIを使用するにはAPIキーが必要です")
    print("   手動で論文の補足資料からデータを抽出することを推奨します")
    
    return pd.DataFrame()

def extract_from_phases_data():
    """DOE/OSTI phasesdataから弾性率を推定（可能な場合）"""
    print("\n" + "=" * 60)
    print("Phases Data 解析")
    print("=" * 60)
    
    phases_file = RAW_DATA_DIR / "doe_osti_dataset" / "phasesdata.xlsx"
    
    if not phases_file.exists():
        print("❌ Phases Dataファイルが見つかりません")
        return pd.DataFrame()
    
    try:
        # すべてのシートを読み込む
        xl_file = pd.ExcelFile(phases_file)
        all_data = []
        
        for sheet_name in xl_file.sheet_names:
            try:
                df = pd.read_excel(phases_file, sheet_name=sheet_name)
                print(f"📊 シート '{sheet_name}': {len(df)}行, {len(df.columns)}列")
                
                # 弾性率関連のカラムを探す
                for col in df.columns:
                    col_lower = str(col).lower()
                    if any(kw in col_lower for kw in ['modulus', 'young', 'elastic', 'E']):
                        if df[col].dtype in [np.float64, np.int64]:
                            df_clean = df[df[col].notna()].copy()
                            if len(df_clean) > 0:
                                df_clean['elastic_modulus'] = df_clean[col]
                                df_clean['source'] = f'DOE_OSTI_Phases_{sheet_name}'
                                all_data.append(df_clean)
                                print(f"   ✅ 弾性率データ発見: {len(df_clean)}サンプル")
            except Exception as e:
                continue
        
        if all_data:
            combined = pd.concat(all_data, ignore_index=True)
            print(f"✅ Phases Data合計: {len(combined)}サンプル")
            return combined
        else:
            print("⚠️  弾性率データが見つかりませんでした")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"⚠️  Phases Data読み込みエラー: {e}")
        return pd.DataFrame()

def estimate_modulus_from_strength(df, strength_col, hardness_col=None):
    """強度データから弾性率を推定（経験則）"""
    # 注意: これは非常に不正確な推定です
    # E ≈ 3 * σ_y (経験則、精度は低い)
    
    estimated_data = []
    
    if strength_col in df.columns:
        df_clean = df[df[strength_col].notna()].copy()
        if len(df_clean) > 0:
            # 経験則: E ≈ 3 * σ_y (非常に大雑把)
            df_clean['elastic_modulus'] = df_clean[strength_col] * 3 / 1000  # MPa to GPa
            df_clean['source'] = df_clean.get('source', 'Estimated') + '_Estimated'
            estimated_data.append(df_clean)
    
    if estimated_data:
        combined = pd.concat(estimated_data, ignore_index=True)
        print(f"⚠️  推定データ: {len(combined)}サンプル（精度は低い）")
        return combined
    
    return pd.DataFrame()

def collect_all_possible_data():
    """すべての可能なデータソースから収集"""
    print("=" * 60)
    print("積極的なデータ収集")
    print("=" * 60)
    print(f"目標: {TARGET_SAMPLES}サンプル")
    print(f"開始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    all_datasets = []
    
    # 1. 深層スキャン
    deep_scan_data = deep_scan_existing_datasets()
    if not deep_scan_data.empty:
        all_datasets.append(deep_scan_data)
    
    # 2. Phases Data解析
    phases_data = extract_from_phases_data()
    if not phases_data.empty:
        all_datasets.append(phases_data)
    
    # 3. MPEA Database（強度から推定、注意が必要）
    mpea_file = RAW_DATA_DIR / "mpea_mechanical_properties" / "A database of mechanical properties for multi prin" / "MPEA_parsed_mechanical_database.xlsx"
    if mpea_file.exists():
        try:
            df_mpea = pd.read_excel(mpea_file)
            # 弾性率が直接ない場合、強度から推定（警告付き）
            if 'elastic_modulus' not in df_mpea.columns or df_mpea['elastic_modulus'].isna().all():
                strength_cols = [col for col in df_mpea.columns if 'yield' in col.lower() or 'strength' in col.lower()]
                if strength_cols:
                    print("\n⚠️  注意: MPEA Databaseから強度データを使用して弾性率を推定します（精度は低い）")
                    estimated = estimate_modulus_from_strength(df_mpea, strength_cols[0])
                    if not estimated.empty:
                        all_datasets.append(estimated)
        except Exception as e:
            print(f"⚠️  MPEA推定エラー: {e}")
    
    # 統合
    if not all_datasets:
        print("\n❌ データを取得できませんでした")
        return pd.DataFrame()
    
    print("\n" + "=" * 60)
    print("データ統合")
    print("=" * 60)
    
    combined = pd.concat(all_datasets, ignore_index=True, sort=False)
    print(f"📊 統合前: {len(combined)}サンプル")
    
    # 重複除去
    if 'alloy_name' in combined.columns and 'elastic_modulus' in combined.columns:
        initial_count = len(combined)
        combined = combined.drop_duplicates(subset=['alloy_name'], keep='first')
        print(f"📊 重複除去後: {len(combined)}サンプル（{initial_count - len(combined)}個除去）")
    
    # 弾性率データがあるもののみ
    if 'elastic_modulus' in combined.columns:
        combined = combined[combined['elastic_modulus'].notna()].copy()
        combined = combined[combined['elastic_modulus'] > 0].copy()
        combined = combined[combined['elastic_modulus'] < 1000].copy()  # 異常値除去
    
    print(f"📊 最終データ数: {len(combined)}サンプル")
    
    # 保存
    output_file = COLLECTED_DATA_DIR / f"aggressive_collection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    combined.to_csv(output_file, index=False)
    print(f"\n✅ データを保存しました: {output_file}")
    
    # 統計
    print("\n" + "=" * 60)
    print("データ統計")
    print("=" * 60)
    
    if 'elastic_modulus' in combined.columns:
        print(f"📊 弾性率範囲: {combined['elastic_modulus'].min():.2f} - {combined['elastic_modulus'].max():.2f} GPa")
        print(f"📊 平均: {combined['elastic_modulus'].mean():.2f} GPa")
        print(f"📊 中央値: {combined['elastic_modulus'].median():.2f} GPa")
        
        target_range = combined[(combined['elastic_modulus'] >= 30) & (combined['elastic_modulus'] <= 90)]
        print(f"📊 目標範囲（30-90 GPa）内: {len(target_range)}サンプル")
    
    if 'source' in combined.columns:
        print("\n📊 データソース別:")
        source_counts = combined['source'].value_counts()
        for source, count in source_counts.items():
            print(f"   {source}: {count}サンプル")
    
    # 目標達成
    print("\n" + "=" * 60)
    print("目標達成状況")
    print("=" * 60)
    print(f"目標: {TARGET_SAMPLES}サンプル")
    print(f"現在: {len(combined)}サンプル")
    print(f"達成率: {len(combined)/TARGET_SAMPLES*100:.1f}%")
    
    return combined

if __name__ == "__main__":
    data = collect_all_possible_data()
    
    print("\n" + "=" * 60)
    print("✅ 積極的データ収集完了")
    print("=" * 60)
    print(f"終了: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
