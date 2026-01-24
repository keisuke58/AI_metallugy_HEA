#!/usr/bin/env python3
"""
代替データセットのダウンロードスクリプト

Gorsse Datasetが取得できない場合の代替データソース
"""

import os
import requests
import pandas as pd
from pathlib import Path
import zipfile
import json

# 設定
BASE_DIR = Path(__file__).parent.parent
RAW_DATA_DIR = BASE_DIR / "raw_data"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

def download_mendeley_dataset(dataset_id, dataset_name, target_dir):
    """
    Mendeley Dataからデータセットをダウンロード
    
    Args:
        dataset_id: Mendeley DataのDOIまたはID
        dataset_name: データセット名
        target_dir: 保存先ディレクトリ
    """
    print(f"\n{'='*60}")
    print(f"{dataset_name} ダウンロード")
    print(f"{'='*60}")
    
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Mendeley DataのURL
    mendeley_url = f"https://data.mendeley.com/datasets/{dataset_id}"
    print(f"\nURL: {mendeley_url}")
    print(f"\n⚠️  注意: Mendeley Dataから手動でダウンロードする必要があります。")
    print(f"\n手動ダウンロード手順:")
    print(f"1. 上記のURLにアクセス")
    print(f"2. 「Download All」ボタンをクリック")
    print(f"3. ダウンロードしたファイルを以下のディレクトリに保存:")
    print(f"   {target_dir}")
    
    # 既存のファイルを確認
    existing_files = list(target_dir.glob("*"))
    data_files = [f for f in existing_files if f.is_file() and not f.name.startswith('.')]
    
    if data_files:
        print(f"\n✅ データファイルが見つかりました: {len(data_files)}個")
        for file in data_files:
            size = file.stat().st_size / 1024  # KB
            print(f"   - {file.name} ({size:.2f} KB)")
        return True
    else:
        print(f"\n❌ データファイルが見つかりませんでした。")
        return False

def download_disma_hea_dataset():
    """
    DISMA Research HEA Dataset
    DOI: 10.17632/p3txdrdth7.1
    URL: https://data.mendeley.com/datasets/p3txdrdth7/1
    """
    target_dir = RAW_DATA_DIR / "disma_hea_dataset"
    return download_mendeley_dataset("p3txdrdth7/1", "DISMA Research HEA Dataset", target_dir)

def download_mpea_mechanical_properties():
    """
    A database of mechanical properties for multi-principal element alloys
    DOI: 10.17632/4d4kpfwpf6
    URL: https://data.mendeley.com/datasets/4d4kpfwpf6
    """
    target_dir = RAW_DATA_DIR / "mpea_mechanical_properties"
    return download_mendeley_dataset("4d4kpfwpf6", "MPEA Mechanical Properties Database", target_dir)

def download_fracture_toughness_dataset():
    """
    Dataset for Fracture and Impact Toughness of High-Entropy Alloys
    Nature Scientific Data, Materials Cloud
    """
    print(f"\n{'='*60}")
    print("Fracture and Impact Toughness Dataset")
    print(f"{'='*60}")
    
    target_dir = RAW_DATA_DIR / "fracture_toughness_dataset"
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Nature Scientific DataのURL
    nature_url = "https://www.nature.com/articles/s41597-022-01911-4"
    materials_cloud_url = "https://www.materialscloud.org/"
    
    print(f"\n論文URL: {nature_url}")
    print(f"Materials Cloud: {materials_cloud_url}")
    print(f"\n⚠️  注意: Materials Cloudから手動でダウンロードする必要があります。")
    print(f"\n手動ダウンロード手順:")
    print(f"1. 論文ページにアクセス: {nature_url}")
    print(f"2. 「Data Availability」セクションを確認")
    print(f"3. Materials Cloudのリンクからデータをダウンロード")
    print(f"4. ダウンロードしたファイルを以下のディレクトリに保存:")
    print(f"   {target_dir}")
    
    # 既存のファイルを確認
    existing_files = list(target_dir.glob("*"))
    data_files = [f for f in existing_files if f.is_file() and not f.name.startswith('.')]
    
    if data_files:
        print(f"\n✅ データファイルが見つかりました: {len(data_files)}個")
        for file in data_files:
            size = file.stat().st_size / 1024  # KB
            print(f"   - {file.name} ({size:.2f} KB)")
        return True
    else:
        print(f"\n❌ データファイルが見つかりませんでした。")
        return False

def check_all_alternative_datasets():
    """
    すべての代替データセットの状況を確認
    """
    print("=" * 60)
    print("代替データセットの確認")
    print("=" * 60)
    
    datasets = [
        ("DISMA Research HEA Dataset", download_disma_hea_dataset),
        ("MPEA Mechanical Properties", download_mpea_mechanical_properties),
        ("Fracture and Impact Toughness", download_fracture_toughness_dataset),
    ]
    
    results = {}
    for name, download_func in datasets:
        try:
            result = download_func()
            results[name] = result
        except Exception as e:
            print(f"\n❌ {name}の確認中にエラーが発生: {e}")
            results[name] = False
    
    print("\n" + "=" * 60)
    print("確認結果サマリー")
    print("=" * 60)
    
    for name, result in results.items():
        status = "✅ データあり" if result else "❌ データなし"
        print(f"{name}: {status}")
    
    return results

if __name__ == "__main__":
    check_all_alternative_datasets()
