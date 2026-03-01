#!/usr/bin/env python3
"""
直接ダウンロードを試みるスクリプト

様々なデータソースから直接ダウンロードを試みます
"""

import os
import requests
import pandas as pd
from pathlib import Path
import zipfile
import time

# 設定
BASE_DIR = Path(__file__).parent.parent
RAW_DATA_DIR = BASE_DIR / "raw_data"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

def download_file(url, output_path, description=""):
    """
    ファイルをダウンロード
    
    Args:
        url: ダウンロードURL
        output_path: 保存先パス
        description: 説明
    """
    print(f"\n{'='*60}")
    print(f"ダウンロード試行: {description}")
    print(f"{'='*60}")
    print(f"URL: {url}")
    print(f"保存先: {output_path}")
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, stream=True, timeout=30)
        response.raise_for_status()
        
        # ファイルサイズを確認
        total_size = int(response.headers.get('content-length', 0))
        print(f"ファイルサイズ: {total_size / 1024:.2f} KB")
        
        # ダウンロード
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        print(f"✅ ダウンロード成功: {output_path}")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"❌ ダウンロード失敗: {e}")
        return False
    except Exception as e:
        print(f"❌ エラー: {e}")
        return False

def try_download_mendeley(dataset_id, dataset_name, target_dir):
    """
    Mendeley Dataから直接ダウンロードを試みる
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Mendeley Dataの直接ダウンロードURL（試行）
    # 注意: 実際のURLは動的に生成される可能性がある
    possible_urls = [
        f"https://data.mendeley.com/public-files/datasets/{dataset_id}/files/",
        f"https://prod-dcd-datasets-cache-zipfiles.s3.amazonaws.com/{dataset_id}.zip",
    ]
    
    print(f"\n{'='*60}")
    print(f"{dataset_name} 直接ダウンロード試行")
    print(f"{'='*60}")
    print(f"\n⚠️  注意: Mendeley Dataは通常、ブラウザからの手動ダウンロードが必要です。")
    print(f"   以下のURLにアクセスして手動でダウンロードしてください:")
    print(f"   https://data.mendeley.com/datasets/{dataset_id}")
    
    return False

def try_download_nature_materials_cloud():
    """
    Nature Scientific Data / Materials Cloudから直接ダウンロードを試みる
    """
    print(f"\n{'='*60}")
    print("Nature Scientific Data / Materials Cloud ダウンロード試行")
    print(f"{'='*60}")
    
    target_dir = RAW_DATA_DIR / "fracture_toughness_dataset"
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Materials CloudのURL（試行）
    # 注意: 実際のURLは論文のData Availabilityセクションに記載されている
    print(f"\n⚠️  注意: Materials Cloudからは通常、手動ダウンロードが必要です。")
    print(f"   以下のURLにアクセスして手動でダウンロードしてください:")
    print(f"   論文: https://www.nature.com/articles/s41597-022-01911-4")
    print(f"   Materials Cloud: https://www.materialscloud.org/")
    
    return False

def try_download_google_sheets(sheet_id, sheet_name, target_dir):
    """
    Google Sheetsから直接ダウンロードを試みる
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Google SheetsのCSVエクスポートURL
    csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid=0"
    
    output_path = target_dir / f"{sheet_name}.csv"
    
    print(f"\n{'='*60}")
    print(f"Google Sheets ダウンロード試行: {sheet_name}")
    print(f"{'='*60}")
    print(f"URL: {csv_url}")
    
    return download_file(csv_url, output_path, f"Google Sheets: {sheet_name}")

def try_all_direct_downloads():
    """
    すべての直接ダウンロードを試行
    """
    print("=" * 60)
    print("直接ダウンロード試行")
    print("=" * 60)
    
    results = {}
    
    # 1. DISMA Research HEA Dataset
    results["DISMA HEA"] = try_download_mendeley(
        "p3txdrdth7/1",
        "DISMA Research HEA Dataset",
        RAW_DATA_DIR / "disma_hea_dataset"
    )
    
    # 2. MPEA Mechanical Properties
    results["MPEA Properties"] = try_download_mendeley(
        "4d4kpfwpf6",
        "MPEA Mechanical Properties",
        RAW_DATA_DIR / "mpea_mechanical_properties"
    )
    
    # 3. Fracture and Impact Toughness
    results["Fracture Toughness"] = try_download_nature_materials_cloud()
    
    # 4. Google Sheets（MPEA DatabaseがGoogle Sheetsで利用可能な場合）
    # 注意: 実際のSheet IDが必要
    # results["MPEA Google Sheets"] = try_download_google_sheets(
    #     "SHEET_ID_HERE",
    #     "mpea_mechanical_properties",
    #     RAW_DATA_DIR / "mpea_mechanical_properties"
    # )
    
    print("\n" + "=" * 60)
    print("ダウンロード試行結果")
    print("=" * 60)
    
    for name, result in results.items():
        status = "✅ 成功" if result else "❌ 失敗（手動ダウンロードが必要）"
        print(f"{name}: {status}")
    
    return results

if __name__ == "__main__":
    try_all_direct_downloads()
