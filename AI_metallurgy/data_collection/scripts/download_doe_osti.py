#!/usr/bin/env python3
"""
DOE/OSTI Dataset ダウンロードスクリプト

出典: Balasubramanian, G. & Ganesh, S.
"Phases and Young's Modulus Dataset for High Entropy Alloys"
DOE/OSTI (2020)

URL: https://www.osti.gov/dataexplorer/biblio/dataset/1644295
"""

import os
import requests
import pandas as pd
from pathlib import Path

# 設定
BASE_DIR = Path(__file__).parent.parent
RAW_DATA_DIR = BASE_DIR / "raw_data" / "doe_osti_dataset"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

# OSTI Data ExplorerのURL
OSTI_URL = "https://www.osti.gov/dataexplorer/biblio/dataset/1644295"

def download_doe_osti_dataset():
    """
    DOE/OSTI Datasetをダウンロード
    
    注意: このデータセットはOSTI Data Explorerから手動でダウンロードする必要があります。
    """
    print("=" * 60)
    print("DOE/OSTI Dataset ダウンロード")
    print("=" * 60)
    print(f"\nURL: {OSTI_URL}")
    print("\n⚠️  注意: このデータセットはOSTI Data Explorerから手動でダウンロードする必要があります。")
    print("\n手動ダウンロード手順:")
    print("1. 上記のURLにアクセス")
    print("2. データセットをダウンロード（通常はCSVまたはJSON形式）")
    print(f"3. ダウンロードしたファイルを以下のディレクトリに保存:")
    print(f"   {RAW_DATA_DIR}")
    print("\nデータファイルを保存したら、以下のコマンドでデータを確認できます:")
    print("   python scripts/check_doe_osti_data.py")
    
    # データファイルが存在するか確認
    data_files = list(RAW_DATA_DIR.glob("*.csv")) + list(RAW_DATA_DIR.glob("*.json"))
    if data_files:
        print(f"\n✅ データファイルが見つかりました: {len(data_files)}個")
        for file in data_files:
            print(f"   - {file.name}")
    else:
        print(f"\n❌ データファイルが見つかりませんでした。")
        print(f"   上記の手順に従って、データファイルを {RAW_DATA_DIR} に保存してください。")

def check_doe_osti_data():
    """
    ダウンロードしたDOE/OSTI Datasetを確認
    """
    print("\n" + "=" * 60)
    print("DOE/OSTI Dataset データ確認")
    print("=" * 60)
    
    # データファイルを探す（CSV, JSON, Excel）
    csv_files = list(RAW_DATA_DIR.glob("*.csv"))
    json_files = list(RAW_DATA_DIR.glob("*.json"))
    xlsx_files = list(RAW_DATA_DIR.glob("*.xlsx"))
    xls_files = list(RAW_DATA_DIR.glob("*.xls"))
    
    if not csv_files and not json_files and not xlsx_files and not xls_files:
        print("\n❌ データファイルが見つかりませんでした。")
        return
    
    # 最初のファイルを読み込む（優先順位: CSV > Excel > JSON）
    if csv_files:
        file_path = csv_files[0]
        df = pd.read_csv(file_path)
    elif xlsx_files:
        file_path = xlsx_files[0]
        df = pd.read_excel(file_path)
    elif xls_files:
        file_path = xls_files[0]
        df = pd.read_excel(file_path)
    else:
        import json
        file_path = json_files[0]
        with open(file_path, 'r') as f:
            data = json.load(f)
        # JSONデータをDataFrameに変換（データ構造に応じて調整が必要）
        df = pd.DataFrame(data)
    
    print(f"\n📁 ファイル: {file_path.name}")
    print(f"📊 データ形状: {df.shape}")
    print(f"📋 カラム: {list(df.columns)}")
    
    # 弾性率データの有無を確認
    elastic_modulus_keywords = ['modulus', 'elastic', 'young', 'E', 'GPa']
    has_elastic_modulus = any(
        keyword.lower() in col.lower() 
        for col in df.columns 
        for keyword in elastic_modulus_keywords
    )
    
    if has_elastic_modulus:
        print("\n✅ 弾性率データが見つかりました")
    else:
        print("\n⚠️  弾性率データが見つかりませんでした（カラム名を確認してください）")
    
    # 最初の数行を表示
    print("\n📄 データの最初の5行:")
    print(df.head())
    
    return df

if __name__ == "__main__":
    download_doe_osti_dataset()
    check_doe_osti_data()
