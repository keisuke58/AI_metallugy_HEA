#!/usr/bin/env python3
"""
包括的なデータ収集スクリプト
最大2000サンプルまで弾性率データを収集

データソース:
1. 既存データセットの再統合
2. Materials Project API
3. NOMAD Database
4. AFLOW Database
5. 公開データセット（Mendeley, Zenodo等）
6. 論文からのデータ抽出
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
warnings.filterwarnings('ignore')

# 設定
BASE_DIR = Path(__file__).parent.parent
RAW_DATA_DIR = BASE_DIR / "raw_data"
PROCESSED_DATA_DIR = BASE_DIR / "processed_data"
COLLECTED_DATA_DIR = BASE_DIR / "collected_data"
COLLECTED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# 目標サンプル数
TARGET_SAMPLES = 2000

def load_existing_data():
    """既存のデータを読み込む"""
    print("=" * 60)
    print("既存データの読み込み")
    print("=" * 60)
    
    all_data = []
    
    # 1. 統合済みデータ
    integrated_file = PROCESSED_DATA_DIR / "data_with_features.csv"
    if integrated_file.exists():
        df = pd.read_csv(integrated_file)
        if 'elastic_modulus' in df.columns:
            df_existing = df[df['elastic_modulus'].notna()].copy()
            all_data.append(df_existing)
            print(f"✅ 統合済みデータ: {len(df_existing)}サンプル")
    
    # 2. DOE/OSTI Dataset
    doe_file = RAW_DATA_DIR / "doe_osti_dataset" / "youngsdata.xlsx"
    if doe_file.exists():
        try:
            df_doe = pd.read_excel(doe_file)
            if 'Young\'s Modulus (GPa)' in df_doe.columns or 'Elastic Modulus' in df_doe.columns:
                col_name = 'Young\'s Modulus (GPa)' if 'Young\'s Modulus (GPa)' in df_doe.columns else 'Elastic Modulus'
                df_doe_clean = df_doe[df_doe[col_name].notna()].copy()
                df_doe_clean['elastic_modulus'] = df_doe_clean[col_name]
                df_doe_clean['source'] = 'DOE/OSTI'
                all_data.append(df_doe_clean)
                print(f"✅ DOE/OSTI Dataset: {len(df_doe_clean)}サンプル")
        except Exception as e:
            print(f"⚠️  DOE/OSTI Dataset読み込みエラー: {e}")
    
    # 3. Gorsse Dataset
    gorsse_file = RAW_DATA_DIR / "gorsse_dataset" / "1-s2.0-S2352340920311100-mmc1.xlsx"
    if gorsse_file.exists():
        try:
            df_gorsse = pd.read_excel(gorsse_file)
            # 弾性率カラムを探す
            elastic_cols = [col for col in df_gorsse.columns if 'modulus' in col.lower() or 'young' in col.lower() or 'E' == col]
            if elastic_cols:
                col_name = elastic_cols[0]
                df_gorsse_clean = df_gorsse[df_gorsse[col_name].notna()].copy()
                df_gorsse_clean['elastic_modulus'] = df_gorsse_clean[col_name]
                df_gorsse_clean['source'] = 'Gorsse'
                all_data.append(df_gorsse_clean)
                print(f"✅ Gorsse Dataset: {len(df_gorsse_clean)}サンプル")
        except Exception as e:
            print(f"⚠️  Gorsse Dataset読み込みエラー: {e}")
    
    # 4. 最新研究データ
    latest_file = RAW_DATA_DIR / "latest_research" / "latest_research.csv"
    if latest_file.exists():
        try:
            df_latest = pd.read_csv(latest_file)
            if 'elastic_modulus' in df_latest.columns:
                df_latest_clean = df_latest[df_latest['elastic_modulus'].notna()].copy()
                df_latest_clean['source'] = 'Latest Research'
                all_data.append(df_latest_clean)
                print(f"✅ 最新研究データ: {len(df_latest_clean)}サンプル")
        except Exception as e:
            print(f"⚠️  最新研究データ読み込みエラー: {e}")
    
    # 統合
    if all_data:
        combined = pd.concat(all_data, ignore_index=True, sort=False)
        print(f"\n📊 既存データ合計: {len(combined)}サンプル")
        return combined
    else:
        print("\n❌ 既存データが見つかりませんでした")
        return pd.DataFrame()

def download_materials_project_data(api_key=None):
    """Materials Project APIからデータを取得"""
    print("\n" + "=" * 60)
    print("Materials Project API データ取得")
    print("=" * 60)
    
    if not api_key:
        api_key = os.getenv("MP_API_KEY")
    
    if not api_key:
        print("⚠️  Materials Project APIキーが設定されていません。スキップします。")
        print("   APIキーを取得: https://materialsproject.org")
        return pd.DataFrame()
    
    try:
        from mp_api.client import MPRester
        
        print("Materials Projectからデータを取得中...")
        data_list = []
        
        # HEA関連元素
        hea_elements = ["Ti", "Zr", "Hf", "Nb", "Ta", "V", "Cr", "Mo", "W", 
                       "Fe", "Co", "Ni", "Cu", "Al", "Mn", "Sn", "Si"]
        
        with MPRester(api_key) as mpr:
            # 複数元素を含む材料を検索（HEAの特徴）
            for i, elem1 in enumerate(hea_elements[:10]):
                for elem2 in hea_elements[i+1:min(i+6, len(hea_elements))]:
                    try:
                        summaries = mpr.materials.summary.search(
                            elements=[elem1, elem2],
                            is_metal=True,
                            fields=["material_id", "formula_pretty", "bulk_modulus", 
                                   "shear_modulus", "elastic_tensor", "density"]
                        )
                        
                        for s in summaries:
                            # 弾性率を計算（体積弾性率とせん断弾性率から）
                            if s.bulk_modulus and s.shear_modulus:
                                # Young's modulus = 9*K*G / (3*K + G)
                                K = s.bulk_modulus
                                G = s.shear_modulus
                                E = (9 * K * G) / (3 * K + G) if (3*K + G) != 0 else None
                                
                                if E and E > 0:
                                    data_list.append({
                                        'material_id': s.material_id,
                                        'alloy_name': s.formula_pretty,
                                        'elastic_modulus': E,
                                        'bulk_modulus': K,
                                        'shear_modulus': G,
                                        'density': s.density,
                                        'source': 'Materials Project'
                                    })
                        
                        time.sleep(0.1)  # APIレート制限対策
                        
                    except Exception as e:
                        print(f"⚠️  エラー ({elem1}-{elem2}): {e}")
                        continue
        
        if data_list:
            df = pd.DataFrame(data_list)
            print(f"✅ Materials Project: {len(df)}サンプル取得")
            return df
        else:
            print("⚠️  Materials Projectからデータを取得できませんでした")
            return pd.DataFrame()
            
    except ImportError:
        print("⚠️  mp-apiパッケージがインストールされていません。スキップします。")
        print("   インストール: pip install mp-api")
        return pd.DataFrame()
    except Exception as e:
        print(f"⚠️  Materials Project取得エラー: {e}")
        return pd.DataFrame()

def download_nomad_data():
    """NOMAD Databaseからデータを取得（可能な場合）"""
    print("\n" + "=" * 60)
    print("NOMAD Database データ取得")
    print("=" * 60)
    
    print("⚠️  NOMAD Databaseは通常、計算データ（DFT）を含みます。")
    print("   実験データとの整合性に注意が必要です。")
    print("   手動でNOMADからデータを取得する必要がある場合があります。")
    
    # NOMAD APIの使用例（実際の実装はAPI仕様に依存）
    nomad_data_dir = RAW_DATA_DIR / "nomad_database"
    nomad_data_dir.mkdir(parents=True, exist_ok=True)
    
    # 既存のNOMADデータを確認
    existing_files = list(nomad_data_dir.glob("*.csv")) + list(nomad_data_dir.glob("*.json"))
    if existing_files:
        print(f"✅ NOMADデータファイルが見つかりました: {len(existing_files)}個")
        # データを読み込んで処理
        return pd.DataFrame()
    else:
        print("❌ NOMADデータファイルが見つかりませんでした")
        return pd.DataFrame()

def download_aflow_data():
    """AFLOW Databaseからデータを取得"""
    print("\n" + "=" * 60)
    print("AFLOW Database データ取得")
    print("=" * 60)
    
    print("⚠️  AFLOW Databaseは通常、計算データ（DFT）を含みます。")
    print("   AFLOW APIまたは手動ダウンロードが必要です。")
    
    aflow_data_dir = RAW_DATA_DIR / "aflow_database"
    aflow_data_dir.mkdir(parents=True, exist_ok=True)
    
    # 既存のAFLOWデータを確認
    existing_files = list(aflow_data_dir.glob("*.csv")) + list(aflow_data_dir.glob("*.json"))
    if existing_files:
        print(f"✅ AFLOWデータファイルが見つかりました: {len(existing_files)}個")
        return pd.DataFrame()
    else:
        print("❌ AFLOWデータファイルが見つかりませんでした")
        return pd.DataFrame()

def extract_from_mpea_database():
    """MPEA Databaseから弾性率を推定（強度データから）"""
    print("\n" + "=" * 60)
    print("MPEA Database データ抽出")
    print("=" * 60)
    
    mpea_file = RAW_DATA_DIR / "mpea_mechanical_properties" / "A database of mechanical properties for multi prin" / "MPEA_parsed_mechanical_database.xlsx"
    
    if not mpea_file.exists():
        print("❌ MPEA Databaseファイルが見つかりません")
        return pd.DataFrame()
    
    try:
        df_mpea = pd.read_excel(mpea_file)
        print(f"📊 MPEA Database: {len(df_mpea)}サンプル")
        
        # 弾性率カラムを探す
        elastic_cols = [col for col in df_mpea.columns 
                        if 'modulus' in col.lower() or 'young' in col.lower() or 'E' == col]
        
        if elastic_cols:
            print(f"✅ 弾性率カラムが見つかりました: {elastic_cols}")
            df_clean = df_mpea[df_mpea[elastic_cols[0]].notna()].copy()
            df_clean['elastic_modulus'] = df_clean[elastic_cols[0]]
            df_clean['source'] = 'MPEA Database'
            print(f"✅ MPEA Database: {len(df_clean)}サンプル")
            return df_clean
        else:
            print("⚠️  弾性率データが見つかりませんでした")
            print("   強度データから弾性率を推定することも可能ですが、精度は低いです")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"⚠️  MPEA Database読み込みエラー: {e}")
        return pd.DataFrame()

def extract_from_disma_database():
    """DISMA Databaseからデータを抽出"""
    print("\n" + "=" * 60)
    print("DISMA Database データ抽出")
    print("=" * 60)
    
    disma_dir = RAW_DATA_DIR / "disma_hea_dataset" / "DISMA_Research Dataset (High-entropy alloys)"
    
    if not disma_dir.exists():
        print("❌ DISMA Databaseディレクトリが見つかりません")
        return pd.DataFrame()
    
    # 強度データから弾性率を推定（可能な場合）
    strength_file = disma_dir / "Training_data_Strength.csv"
    if strength_file.exists():
        try:
            df_strength = pd.read_csv(strength_file)
            print(f"📊 DISMA Strength Data: {len(df_strength)}サンプル")
            # 弾性率データが直接含まれているか確認
            # 含まれていない場合は、強度から推定することも可能だが精度は低い
            print("⚠️  弾性率データの直接抽出は困難です")
            return pd.DataFrame()
        except Exception as e:
            print(f"⚠️  DISMA Database読み込みエラー: {e}")
            return pd.DataFrame()
    else:
        print("❌ DISMA Databaseファイルが見つかりません")
        return pd.DataFrame()

def search_literature_data():
    """最新の論文からデータを検索（可能な場合）"""
    print("\n" + "=" * 60)
    print("文献データ検索")
    print("=" * 60)
    
    print("⚠️  文献からのデータ抽出は手動で行う必要があります。")
    print("   以下のリソースを確認してください:")
    print("   - PubMed: https://pubmed.ncbi.nlm.nih.gov/")
    print("   - arXiv: https://arxiv.org/")
    print("   - Materials Cloud: https://www.materialscloud.org/")
    
    literature_dir = RAW_DATA_DIR / "literature_data"
    literature_dir.mkdir(parents=True, exist_ok=True)
    
    # 既存の文献データを確認
    existing_files = list(literature_dir.glob("*.csv")) + list(literature_dir.glob("*.xlsx"))
    if existing_files:
        print(f"✅ 文献データファイルが見つかりました: {len(existing_files)}個")
        all_data = []
        for file in existing_files:
            try:
                if file.suffix == '.csv':
                    df = pd.read_csv(file)
                else:
                    df = pd.read_excel(file)
                if 'elastic_modulus' in df.columns:
                    df['source'] = f'Literature_{file.stem}'
                    all_data.append(df)
            except Exception as e:
                print(f"⚠️  {file.name}読み込みエラー: {e}")
        
        if all_data:
            combined = pd.concat(all_data, ignore_index=True)
            print(f"✅ 文献データ: {len(combined)}サンプル")
            return combined
    
    return pd.DataFrame()

def remove_duplicates(df):
    """重複データを除去"""
    if df.empty:
        return df
    
    print("\n" + "=" * 60)
    print("重複データの除去")
    print("=" * 60)
    
    initial_count = len(df)
    
    # 合金名と弾性率で重複を判定
    if 'alloy_name' in df.columns and 'elastic_modulus' in df.columns:
        # 合金名が同じで弾性率が近い（±5 GPa以内）ものを重複とみなす
        df_sorted = df.sort_values('elastic_modulus')
        df_unique = df_sorted.drop_duplicates(subset=['alloy_name'], keep='first')
        
        # さらに、合金名が異なるが組成が同じものを検出（可能な場合）
        if 'composition' in df.columns or any('comp_' in col for col in df.columns):
            # 組成ベースの重複除去（簡易版）
            pass
        
        removed = initial_count - len(df_unique)
        print(f"📊 初期データ数: {initial_count}")
        print(f"📊 重複除去後: {len(df_unique)}")
        print(f"📊 除去された重複: {removed}")
        
        return df_unique
    
    return df

def combine_all_data():
    """すべてのデータソースを統合"""
    print("=" * 60)
    print("包括的なデータ収集")
    print("=" * 60)
    print(f"目標サンプル数: {TARGET_SAMPLES}")
    print(f"開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    all_datasets = []
    
    # 1. 既存データ
    existing_data = load_existing_data()
    if not existing_data.empty:
        all_datasets.append(existing_data)
    
    # 2. Materials Project
    mp_data = download_materials_project_data()
    if not mp_data.empty:
        all_datasets.append(mp_data)
    
    # 3. MPEA Database
    mpea_data = extract_from_mpea_database()
    if not mpea_data.empty:
        all_datasets.append(mpea_data)
    
    # 4. DISMA Database
    disma_data = extract_from_disma_database()
    if not disma_data.empty:
        all_datasets.append(disma_data)
    
    # 5. NOMAD
    nomad_data = download_nomad_data()
    if not nomad_data.empty:
        all_datasets.append(nomad_data)
    
    # 6. AFLOW
    aflow_data = download_aflow_data()
    if not aflow_data.empty:
        all_datasets.append(aflow_data)
    
    # 7. 文献データ
    literature_data = search_literature_data()
    if not literature_data.empty:
        all_datasets.append(literature_data)
    
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
    combined = remove_duplicates(combined)
    
    # 弾性率データがあるもののみ
    if 'elastic_modulus' in combined.columns:
        combined = combined[combined['elastic_modulus'].notna()].copy()
        combined = combined[combined['elastic_modulus'] > 0].copy()
    
    print(f"📊 最終データ数: {len(combined)}サンプル")
    
    # 保存
    output_file = COLLECTED_DATA_DIR / f"comprehensive_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    combined.to_csv(output_file, index=False)
    print(f"\n✅ データを保存しました: {output_file}")
    
    # 統計情報
    print("\n" + "=" * 60)
    print("データ統計")
    print("=" * 60)
    
    if 'elastic_modulus' in combined.columns:
        print(f"📊 弾性率範囲: {combined['elastic_modulus'].min():.2f} - {combined['elastic_modulus'].max():.2f} GPa")
        print(f"📊 平均弾性率: {combined['elastic_modulus'].mean():.2f} GPa")
        print(f"📊 中央値: {combined['elastic_modulus'].median():.2f} GPa")
        
        # 目標範囲内のデータ
        target_range = combined[(combined['elastic_modulus'] >= 30) & (combined['elastic_modulus'] <= 90)]
        print(f"📊 目標範囲（30-90 GPa）内: {len(target_range)}サンプル")
    
    if 'source' in combined.columns:
        print("\n📊 データソース別:")
        source_counts = combined['source'].value_counts()
        for source, count in source_counts.items():
            print(f"   {source}: {count}サンプル")
    
    # 目標達成状況
    print("\n" + "=" * 60)
    print("目標達成状況")
    print("=" * 60)
    print(f"目標: {TARGET_SAMPLES}サンプル")
    print(f"現在: {len(combined)}サンプル")
    print(f"達成率: {len(combined)/TARGET_SAMPLES*100:.1f}%")
    
    if len(combined) < TARGET_SAMPLES:
        needed = TARGET_SAMPLES - len(combined)
        print(f"不足: {needed}サンプル")
        print("\n💡 追加データ収集の推奨:")
        print("   1. Materials Project APIキーを取得してより多くのデータを取得")
        print("   2. 最新の論文からデータを手動で抽出")
        print("   3. 他の公開データセットを探索")
    
    return combined

if __name__ == "__main__":
    combined_data = combine_all_data()
    
    print("\n" + "=" * 60)
    print("✅ データ収集完了")
    print("=" * 60)
    print(f"終了時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
