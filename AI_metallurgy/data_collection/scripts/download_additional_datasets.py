#!/usr/bin/env python3
"""
追加データセットのダウンロードスクリプト
新たに発見したデータセットを自動的にダウンロード
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
import subprocess
warnings.filterwarnings('ignore')

# 設定
BASE_DIR = Path(__file__).parent.parent
RAW_DATA_DIR = BASE_DIR / "raw_data"
COLLECTED_DATA_DIR = BASE_DIR / "collected_data"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
COLLECTED_DATA_DIR.mkdir(parents=True, exist_ok=True)

def download_refractory_hea_github():
    """Refractory HEA Elastic ConstantsをGitHubからダウンロード"""
    print("=" * 60)
    print("Refractory HEA Elastic Constants (GitHub) ダウンロード")
    print("=" * 60)
    
    repo_url = "https://github.com/uttambhandari91/Elastic-constant-DFT-data"
    file_url = "https://github.com/uttambhandari91/Elastic-constant-DFT-data/raw/main/materials_journal_elastic_constant_data.xlsx"
    
    output_dir = RAW_DATA_DIR / "refractory_hea_elastic_constants"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "materials_journal_elastic_constant_data.xlsx"
    
    if output_file.exists():
        print(f"✅ ファイルは既に存在します: {output_file}")
        return output_file
    
    try:
        print(f"📥 ダウンロード中: {file_url}")
        response = requests.get(file_url, timeout=60, stream=True)
        
        if response.status_code == 200:
            with open(output_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"✅ ダウンロード完了: {output_file}")
            
            # データを確認
            try:
                df = pd.read_excel(output_file)
                print(f"📊 データ数: {len(df)}サンプル")
                print(f"📊 カラム: {list(df.columns)[:10]}...")
            except Exception as e:
                print(f"⚠️  データ読み込みエラー: {e}")
            
            return output_file
        else:
            print(f"❌ ダウンロード失敗: HTTP {response.status_code}")
            print(f"   手動ダウンロード: {repo_url}")
            return None
            
    except Exception as e:
        print(f"❌ エラー: {e}")
        print(f"   手動ダウンロード: {repo_url}")
        return None

def download_dryad_elastic_properties():
    """Dryad Elastic Properties Databaseをダウンロード"""
    print("\n" + "=" * 60)
    print("Dryad Elastic Properties Database ダウンロード")
    print("=" * 60)
    
    doi = "10.5061/dryad.h505v"
    dryad_url = f"https://datadryad.org/stash/dataset/doi:{doi}"
    
    output_dir = RAW_DATA_DIR / "dryad_elastic_properties"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"⚠️  Dryadデータは手動ダウンロードが必要です")
    print(f"   URL: {dryad_url}")
    print(f"   ダウンロード後、{output_dir} に保存してください")
    
    return None

def search_chemdataextractor():
    """ChemDataExtractor Databaseの情報を検索"""
    print("\n" + "=" * 60)
    print("ChemDataExtractor Database 情報")
    print("=" * 60)
    
    print("📚 論文情報:")
    print("   - Nature Scientific Data (2024)")
    print("   - URL: https://www.nature.com/articles/s41597-024-03979-6")
    print("   - データ: Cambridge Repository")
    print("   - データ数: 720,308 records")
    print("   - Young's modulusを含む")
    print("\n⚠️  手動で論文からデータをダウンロードする必要があります")

def download_oqmd_data():
    """OQMD Databaseからデータを取得"""
    print("\n" + "=" * 60)
    print("OQMD Database データ取得")
    print("=" * 60)
    
    try:
        # OQMD APIの使用例
        api_url = "https://oqmd.org/api/optimade/v1/structures"
        
        # HEA関連元素で検索
        hea_elements = ["Ti", "Zr", "Hf", "Nb", "Ta", "V", "Cr", "Mo", "W", 
                       "Fe", "Co", "Ni", "Cu", "Al", "Mn"]
        
        print("⚠️  OQMD APIの実装には詳細な仕様確認が必要です")
        print("   API URL: https://oqmd.org/api/")
        print("   Python package: qmpy-rester")
        print("   インストール: pip install qmpy-rester")
        
        return None
        
    except Exception as e:
        print(f"❌ エラー: {e}")
        return None

def download_aflow_data():
    """AFLOW Databaseからデータを取得"""
    print("\n" + "=" * 60)
    print("AFLOW Database データ取得")
    print("=" * 60)
    
    api_url = "https://aflowlib.duke.edu/AFLOWDATA/LIB2_WEB/"
    
    print("⚠️  AFLOW APIの実装には詳細な仕様確認が必要です")
    print("   API URL: https://aflowlib.duke.edu/AFLOWDATA/LIB2_WEB/")
    print("   ドキュメント: http://aflow.org")
    
    return None

def extract_refractory_hea_elastic_modulus(file_path):
    """Refractory HEA Elastic Constantsから弾性率を抽出"""
    if file_path is None or not file_path.exists():
        return pd.DataFrame()
    
    print("\n" + "=" * 60)
    print("Refractory HEA Elastic Constants データ抽出")
    print("=" * 60)
    
    try:
        # 最初の行がヘッダーなので、それを読み込む
        df_header = pd.read_excel(file_path, nrows=1)
        df_data = pd.read_excel(file_path, skiprows=1)
        
        print(f"📊 元データ: {len(df_data)}サンプル")
        
        # ヘッダー行からカラム名を取得
        header_row = df_header.iloc[0].values
        
        # 弾性定数カラムを探す（C11, C12, C44, E, Young's modulus等）
        elastic_cols = []
        alloy_col = None
        
        for i, col_name in enumerate(header_row):
            col_str = str(col_name).lower()
            if 'alloy' in col_str or 'name' in col_str:
                alloy_col = i
            elif any(kw in col_str for kw in ['c11', 'c12', 'c44', 'modulus', 'young', 'elastic', 'e']):
                elastic_cols.append((i, col_name))
        
        if not elastic_cols:
            print("⚠️  弾性定数カラムが見つかりませんでした")
            print(f"   ヘッダー: {list(header_row)}")
            return pd.DataFrame()
        
        print(f"✅ 弾性定数カラムが見つかりました: {[c[1] for c in elastic_cols]}")
        
        # データを整理
        result_data = []
        
        for idx, row in df_data.iterrows():
            alloy_name = None
            if alloy_col is not None:
                alloy_name = row.iloc[alloy_col] if alloy_col < len(row) else None
            
            # 弾性定数からYoung's modulusを計算
            # 等方性材料の場合: E = C11 - C12 (簡易版)
            # または Voigt平均: E = (C11 - C12) * (C11 + 2*C12) / (C11 + C12)
            
            elastic_modulus = None
            
            # C11, C12, C44から計算を試みる
            c11 = None
            c12 = None
            c44 = None
            
            for col_idx, col_name in elastic_cols:
                val = row.iloc[col_idx] if col_idx < len(row) else None
                col_str = str(col_name).lower()
                
                if 'c11' in col_str:
                    c11 = pd.to_numeric(val, errors='coerce')
                elif 'c12' in col_str:
                    c12 = pd.to_numeric(val, errors='coerce')
                elif 'c44' in col_str:
                    c44 = pd.to_numeric(val, errors='coerce')
                elif 'young' in col_str or 'modulus' in col_str or 'e' == col_str.lower():
                    elastic_modulus = pd.to_numeric(val, errors='coerce')
            
            # 弾性定数から計算
            if elastic_modulus is None and c11 is not None and c12 is not None:
                # 簡易計算: E ≈ C11 - C12 (等方性近似)
                elastic_modulus = c11 - c12 if pd.notna(c11) and pd.notna(c12) else None
            
            if pd.notna(elastic_modulus) and elastic_modulus > 0:
                result_data.append({
                    'alloy_name': alloy_name if alloy_name else f'Alloy_{idx}',
                    'elastic_modulus': elastic_modulus,
                    'source': 'Refractory HEA Elastic Constants (GitHub)',
                    'C11': c11,
                    'C12': c12,
                    'C44': c44
                })
        
        if result_data:
            df_result = pd.DataFrame(result_data)
            
            # 保存
            output_file = COLLECTED_DATA_DIR / "refractory_hea_elastic_modulus.csv"
            df_result.to_csv(output_file, index=False)
            print(f"✅ データを保存しました: {output_file}")
            print(f"📊 抽出データ数: {len(df_result)}サンプル")
            print(f"📊 弾性率範囲: {df_result['elastic_modulus'].min():.2f} - {df_result['elastic_modulus'].max():.2f} GPa")
            
            return df_result
        else:
            print("⚠️  有効なデータが見つかりませんでした")
            return pd.DataFrame()
        
    except Exception as e:
        print(f"❌ エラー: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

def main():
    """メイン関数"""
    print("=" * 60)
    print("追加データセットのダウンロード")
    print("=" * 60)
    print(f"開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    all_datasets = []
    
    # 1. Refractory HEA Elastic Constants (GitHub)
    refractory_file = download_refractory_hea_github()
    if refractory_file:
        df_refractory = extract_refractory_hea_elastic_modulus(refractory_file)
        if not df_refractory.empty:
            all_datasets.append(df_refractory)
    
    # 2. Dryad Elastic Properties
    download_dryad_elastic_properties()
    
    # 3. ChemDataExtractor Database
    search_chemdataextractor()
    
    # 4. OQMD Database
    download_oqmd_data()
    
    # 5. AFLOW Database
    download_aflow_data()
    
    # 統合
    if all_datasets:
        combined = pd.concat(all_datasets, ignore_index=True, sort=False)
        output_file = COLLECTED_DATA_DIR / f"additional_datasets_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        combined.to_csv(output_file, index=False)
        print(f"\n✅ 統合データを保存しました: {output_file}")
        print(f"📊 総データ数: {len(combined)}サンプル")
    
    print("\n" + "=" * 60)
    print("✅ ダウンロード処理完了")
    print("=" * 60)
    print(f"終了時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\n💡 手動ダウンロードが必要なデータセット:")
    print("   1. Dryad Elastic Properties: https://datadryad.org/stash/dataset/doi:10.5061/dryad.h505v")
    print("   2. ChemDataExtractor: https://www.nature.com/articles/s41597-024-03979-6")
    print("   3. MPEA Nano-indentation: PubMedから検索")

if __name__ == "__main__":
    main()
