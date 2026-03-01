#!/usr/bin/env python3
"""
Gorsse Dataset ダウンロードスクリプト

出典: Gorsse, S., Nguyen, M. H., Senkov, O. N., & Miracle, D. B. (2018)
"Database on the mechanical properties of high entropy alloys and complex concentrated alloys"
Data in Brief, 2018

URL: https://pubmed.ncbi.nlm.nih.gov/30761350/
"""

import os
import requests
import pandas as pd
from pathlib import Path

# 設定
BASE_DIR = Path(__file__).parent.parent
RAW_DATA_DIR = BASE_DIR / "raw_data" / "gorsse_dataset"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

# PubMed論文のURL
PUBMED_URL = "https://pubmed.ncbi.nlm.nih.gov/30761350/"

def download_gorsse_dataset():
    """
    Gorsse Datasetをダウンロード
    
    注意: このデータセットは通常、論文の補足資料として提供されます。
    直接ダウンロードできない場合は、手動でダウンロードしてください。
    """
    print("=" * 60)
    print("Gorsse Dataset ダウンロード")
    print("=" * 60)
    print(f"\n論文URL: {PUBMED_URL}")
    print("\n⚠️  注意: このデータセットは通常、論文の補足資料から手動でダウンロードする必要があります。")
    print("\n手動ダウンロード手順:")
    print("1. 上記のURLにアクセス")
    print("2. 論文の補足資料（Supplementary Material）を確認")
    print("3. データファイル（通常はCSVまたはExcel形式）をダウンロード")
    print(f"4. ダウンロードしたファイルを以下のディレクトリに保存:")
    print(f"   {RAW_DATA_DIR}")
    print("\nデータファイルを保存したら、以下のコマンドでデータを確認できます:")
    print("   python scripts/check_gorsse_data.py")
    
    # データファイルが存在するか確認
    data_files = list(RAW_DATA_DIR.glob("*.csv")) + list(RAW_DATA_DIR.glob("*.xlsx"))
    if data_files:
        print(f"\n✅ データファイルが見つかりました: {len(data_files)}個")
        for file in data_files:
            print(f"   - {file.name}")
    else:
        print(f"\n❌ データファイルが見つかりませんでした。")
        print(f"   上記の手順に従って、データファイルを {RAW_DATA_DIR} に保存してください。")

def check_gorsse_data():
    """
    ダウンロードしたGorsse Datasetを確認
    """
    print("\n" + "=" * 60)
    print("Gorsse Dataset データ確認")
    print("=" * 60)
    
    # CSVファイルを探す
    csv_files = list(RAW_DATA_DIR.glob("*.csv"))
    xlsx_files = list(RAW_DATA_DIR.glob("*.xlsx"))
    
    if not csv_files and not xlsx_files:
        print("\n❌ データファイルが見つかりませんでした。")
        return
    
    # 最初のファイルを読み込む
    if csv_files:
        file_path = csv_files[0]
        df = pd.read_csv(file_path)
    else:
        file_path = xlsx_files[0]
        df = pd.read_excel(file_path)
    
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
    download_gorsse_dataset()
    check_gorsse_data()
