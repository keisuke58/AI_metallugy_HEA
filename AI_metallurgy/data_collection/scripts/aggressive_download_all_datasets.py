#!/usr/bin/env python3
"""
積極的なデータセットダウンロードスクリプト
発見したすべての追加データセットをダウンロード
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

def download_dryad_elastic_properties():
    """Dryad Elastic Properties Databaseをダウンロード"""
    print("=" * 60)
    print("Dryad Elastic Properties Database ダウンロード")
    print("=" * 60)
    
    doi = "10.5061/dryad.h505v"
    dryad_url = f"https://datadryad.org/stash/dataset/doi:{doi}"
    
    output_dir = RAW_DATA_DIR / "dryad_elastic_properties"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📥 Dryadデータセットにアクセス中...")
    print(f"   URL: {dryad_url}")
    
    try:
        # Dryad APIを使用してデータを取得
        api_url = f"https://datadryad.org/api/v2/datasets/{doi.replace('/', '%2F')}"
        response = requests.get(api_url, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            files = data.get('_embedded', {}).get('stash:files', [])
            
            if files:
                print(f"✅ {len(files)}個のファイルが見つかりました")
                for file_info in files:
                    file_name = file_info.get('path', 'unknown')
                    file_url = file_info.get('_links', {}).get('stash:download', {}).get('href', '')
                    
                    if file_url:
                        print(f"   📥 ダウンロード中: {file_name}")
                        try:
                            file_response = requests.get(file_url, timeout=120, stream=True)
                            if file_response.status_code == 200:
                                output_file = output_dir / file_name
                                with open(output_file, 'wb') as f:
                                    for chunk in file_response.iter_content(chunk_size=8192):
                                        f.write(chunk)
                                print(f"   ✅ ダウンロード完了: {output_file}")
                            else:
                                print(f"   ⚠️  ダウンロード失敗: HTTP {file_response.status_code}")
                        except Exception as e:
                            print(f"   ⚠️  エラー: {e}")
            else:
                print("⚠️  ファイル情報を取得できませんでした")
                print("   手動ダウンロード: https://datadryad.org/stash/dataset/doi:10.5061/dryad.h505v")
        else:
            print(f"⚠️  APIエラー: HTTP {response.status_code}")
            print(f"   手動ダウンロード: {dryad_url}")
            
    except Exception as e:
        print(f"❌ エラー: {e}")
        print(f"   手動ダウンロード: {dryad_url}")
    
    return output_dir

def download_chemdataextractor_database():
    """ChemDataExtractor Databaseをダウンロード"""
    print("\n" + "=" * 60)
    print("ChemDataExtractor Database ダウンロード")
    print("=" * 60)
    
    output_dir = RAW_DATA_DIR / "chemdataextractor_database"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Cambridge RepositoryのURL
    repo_url = "https://www.repository.cam.ac.uk/items/398d4b36-80e6-46f4-93d6-5f441e7af63b"
    download_url = "https://www.repository.cam.ac.uk/bitstreams/fd6d8f15-592b-4426-8d73-937bb5feba0c/download"
    
    print(f"📥 ChemDataExtractor Databaseにアクセス中...")
    print(f"   URL: {repo_url}")
    
    try:
        print(f"   📥 ダウンロード中...")
        response = requests.get(download_url, timeout=300, stream=True)
        
        if response.status_code == 200:
            # ファイル名を取得
            content_disposition = response.headers.get('Content-Disposition', '')
            if 'filename=' in content_disposition:
                filename = content_disposition.split('filename=')[1].strip('"')
            else:
                filename = "chemdataextractor_database.zip"
            
            output_file = output_dir / filename
            
            with open(output_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"   ✅ ダウンロード完了: {output_file}")
            print(f"   📊 ファイルサイズ: {output_file.stat().st_size / 1024 / 1024:.2f} MB")
            
            # ZIPファイルの場合は展開
            if filename.endswith('.zip'):
                print(f"   📦 ZIPファイルを展開中...")
                with zipfile.ZipFile(output_file, 'r') as zip_ref:
                    zip_ref.extractall(output_dir)
                print(f"   ✅ 展開完了")
            
            return output_dir
        else:
            print(f"   ⚠️  ダウンロード失敗: HTTP {response.status_code}")
            print(f"   手動ダウンロード: {repo_url}")
            
    except Exception as e:
        print(f"   ❌ エラー: {e}")
        print(f"   手動ダウンロード: {repo_url}")
    
    return None

def download_mpea_nanoindentation():
    """MPEA Nano-indentation Databaseをダウンロード"""
    print("\n" + "=" * 60)
    print("MPEA Nano-indentation Database ダウンロード")
    print("=" * 60)
    
    output_dir = RAW_DATA_DIR / "mpea_nanoindentation"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # PMC11298849の論文ページ
    pmc_url = "https://pmc.ncbi.nlm.nih.gov/articles/PMC11298849/"
    
    # GitHubリポジトリ（CitrineInformatics）
    github_repo = "https://github.com/CitrineInformatics/MPEA_dataset"
    
    print(f"📥 MPEA Nano-indentation Databaseにアクセス中...")
    print(f"   PMC URL: {pmc_url}")
    print(f"   GitHub: {github_repo}")
    
    # GitHubからデータを取得を試みる
    try:
        # GitHub APIを使用
        api_url = "https://api.github.com/repos/CitrineInformatics/MPEA_dataset/contents"
        response = requests.get(api_url, timeout=30)
        
        if response.status_code == 200:
            files = response.json()
            print(f"   ✅ {len(files)}個のファイルが見つかりました")
            
            for file_info in files:
                if file_info['type'] == 'file':
                    file_name = file_info['name']
                    download_url = file_info['download_url']
                    
                    # データファイルのみダウンロード
                    if any(ext in file_name.lower() for ext in ['.csv', '.xlsx', '.xls', '.json', '.zip']):
                        print(f"   📥 ダウンロード中: {file_name}")
                        try:
                            file_response = requests.get(download_url, timeout=60)
                            if file_response.status_code == 200:
                                output_file = output_dir / file_name
                                with open(output_file, 'wb') as f:
                                    f.write(file_response.content)
                                print(f"   ✅ ダウンロード完了: {output_file}")
                        except Exception as e:
                            print(f"   ⚠️  エラー: {e}")
        else:
            print(f"   ⚠️  GitHub APIエラー: HTTP {response.status_code}")
            print(f"   手動ダウンロード: {pmc_url}")
            
    except Exception as e:
        print(f"   ❌ エラー: {e}")
        print(f"   手動ダウンロード: {pmc_url}")
    
    return output_dir

def download_oqmd_data():
    """OQMD Databaseからデータを取得"""
    print("\n" + "=" * 60)
    print("OQMD Database データ取得")
    print("=" * 60)
    
    output_dir = RAW_DATA_DIR / "oqmd_database"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("📥 OQMD APIからデータを取得中...")
    
    try:
        # OQMD OPTiMaDe APIを使用
        api_url = "https://oqmd.org/api/optimade/v1/structures"
        
        # HEA関連元素で検索
        hea_elements = ["Ti", "Zr", "Hf", "Nb", "Ta", "V", "Cr", "Mo", "W", 
                       "Fe", "Co", "Ni", "Cu", "Al", "Mn"]
        
        # 最初の数個の元素で試す
        for elem in hea_elements[:5]:
            try:
                params = {
                    'filter': f'elements HAS "{elem}"',
                    'page_limit': 100
                }
                response = requests.get(api_url, params=params, timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    structures = data.get('data', [])
                    print(f"   ✅ {elem}を含む構造: {len(structures)}個")
                    
                    if structures:
                        # データを保存
                        output_file = output_dir / f"oqmd_{elem}_structures.json"
                        with open(output_file, 'w') as f:
                            json.dump(structures, f, indent=2)
                        print(f"   💾 保存: {output_file}")
                    
                    time.sleep(1)  # レート制限対策
                else:
                    print(f"   ⚠️  APIエラー ({elem}): HTTP {response.status_code}")
                    
            except Exception as e:
                print(f"   ⚠️  エラー ({elem}): {e}")
                continue
        
        print("⚠️  OQMD APIの詳細な実装にはqmpy-resterパッケージの使用を推奨します")
        print("   インストール: pip install qmpy-rester")
        
    except Exception as e:
        print(f"❌ エラー: {e}")
    
    return output_dir

def download_aflow_data():
    """AFLOW Databaseからデータを取得"""
    print("\n" + "=" * 60)
    print("AFLOW Database データ取得")
    print("=" * 60)
    
    output_dir = RAW_DATA_DIR / "aflow_database"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("📥 AFLOW APIからデータを取得中...")
    
    try:
        # AFLOW APIを使用
        api_url = "https://aflowlib.duke.edu/AFLOWDATA/LIB2_WEB/"
        
        # 簡単なクエリを試す
        response = requests.get(api_url, timeout=30)
        
        if response.status_code == 200:
            print("   ✅ AFLOW APIにアクセス可能")
            print("   ⚠️  詳細なデータ取得にはAFLOW APIの仕様確認が必要です")
            print("   ドキュメント: http://aflow.org")
        else:
            print(f"   ⚠️  APIエラー: HTTP {response.status_code}")
            
    except Exception as e:
        print(f"   ❌ エラー: {e}")
    
    return output_dir

def main():
    """メイン関数"""
    print("=" * 60)
    print("積極的なデータセットダウンロード")
    print("=" * 60)
    print(f"開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    downloaded_datasets = []
    
    # 1. Dryad Elastic Properties
    dryad_dir = download_dryad_elastic_properties()
    if dryad_dir and any(dryad_dir.iterdir()):
        downloaded_datasets.append(('Dryad Elastic Properties', dryad_dir))
    
    # 2. ChemDataExtractor Database
    chem_dir = download_chemdataextractor_database()
    if chem_dir and any(chem_dir.iterdir()):
        downloaded_datasets.append(('ChemDataExtractor Database', chem_dir))
    
    # 3. MPEA Nano-indentation
    mpea_dir = download_mpea_nanoindentation()
    if mpea_dir and any(mpea_dir.iterdir()):
        downloaded_datasets.append(('MPEA Nano-indentation', mpea_dir))
    
    # 4. OQMD Database
    oqmd_dir = download_oqmd_data()
    if oqmd_dir and any(oqmd_dir.iterdir()):
        downloaded_datasets.append(('OQMD Database', oqmd_dir))
    
    # 5. AFLOW Database
    aflow_dir = download_aflow_data()
    
    # 結果サマリー
    print("\n" + "=" * 60)
    print("ダウンロード結果サマリー")
    print("=" * 60)
    
    if downloaded_datasets:
        print(f"✅ ダウンロード成功: {len(downloaded_datasets)}個のデータセット")
        for name, dir_path in downloaded_datasets:
            file_count = len(list(dir_path.rglob('*'))) - len(list(dir_path.rglob('*/')))
            print(f"   - {name}: {file_count}ファイル ({dir_path})")
    else:
        print("⚠️  自動ダウンロードできたデータセットはありませんでした")
    
    print("\n💡 手動ダウンロードが必要なデータセット:")
    print("   1. Dryad Elastic Properties: https://datadryad.org/stash/dataset/doi:10.5061/dryad.h505v")
    print("   2. ChemDataExtractor: https://www.repository.cam.ac.uk/items/398d4b36-80e6-46f4-93d6-5f441e7af63b")
    print("   3. MPEA Nano-indentation: https://pmc.ncbi.nlm.nih.gov/articles/PMC11298849/")
    
    print("\n" + "=" * 60)
    print("✅ ダウンロード処理完了")
    print("=" * 60)
    print(f"終了時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
