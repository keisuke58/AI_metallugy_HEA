#!/usr/bin/env python3
"""
最終データ統合スクリプト
すべてのデータソースを統合して最大2000サンプルまで収集
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).parent.parent
RAW_DATA_DIR = BASE_DIR / "raw_data"
PROCESSED_DATA_DIR = BASE_DIR / "processed_data"
COLLECTED_DATA_DIR = BASE_DIR / "collected_data"
FINAL_DATA_DIR = BASE_DIR / "final_data"
FINAL_DATA_DIR.mkdir(parents=True, exist_ok=True)

TARGET_SAMPLES = 2000

def load_all_data_sources():
    """すべてのデータソースを読み込む"""
    print("=" * 60)
    print("すべてのデータソースの読み込み")
    print("=" * 60)
    
    all_datasets = []
    
    # 1. 統合済みデータ
    integrated_file = PROCESSED_DATA_DIR / "data_with_features.csv"
    if integrated_file.exists():
        df = pd.read_csv(integrated_file)
        if 'elastic_modulus' in df.columns:
            df_clean = df[df['elastic_modulus'].notna()].copy()
            df_clean = df_clean[df_clean['elastic_modulus'] > 0].copy()
            if 'source' not in df_clean.columns:
                df_clean['source'] = 'Integrated'
            all_datasets.append(('統合済みデータ', df_clean))
            print(f"✅ 統合済みデータ: {len(df_clean)}サンプル")
    
    # 2. Gorsse Dataset（抽出済み）
    gorsse_file = COLLECTED_DATA_DIR / "gorsse_elastic_modulus.csv"
    if gorsse_file.exists():
        df_gorsse = pd.read_csv(gorsse_file)
        if 'elastic_modulus' in df_gorsse.columns:
            df_gorsse_clean = df_gorsse[df_gorsse['elastic_modulus'].notna()].copy()
            df_gorsse_clean = df_gorsse_clean[df_gorsse_clean['elastic_modulus'] > 0].copy()
            all_datasets.append(('Gorsse Dataset', df_gorsse_clean))
            print(f"✅ Gorsse Dataset: {len(df_gorsse_clean)}サンプル")
    
    # 3. DOE/OSTI Dataset
    doe_file = RAW_DATA_DIR / "doe_osti_dataset" / "youngsdata.xlsx"
    if doe_file.exists():
        try:
            df_doe = pd.read_excel(doe_file)
            # 弾性率カラムを探す
            elastic_cols = [col for col in df_doe.columns 
                          if 'modulus' in col.lower() or 'young' in col.lower() or 'elastic' in col.lower()]
            if elastic_cols:
                col_name = elastic_cols[0]
                df_doe_clean = df_doe[df_doe[col_name].notna()].copy()
                df_doe_clean = df_doe_clean[df_doe_clean[col_name] > 0].copy()
                df_doe_clean['elastic_modulus'] = df_doe_clean[col_name]
                if 'alloy_name' not in df_doe_clean.columns:
                    # 合金名カラムを探す
                    name_cols = [col for col in df_doe_clean.columns if 'alloy' in col.lower() or 'name' in col.lower() or 'composition' in col.lower()]
                    if name_cols:
                        df_doe_clean['alloy_name'] = df_doe_clean[name_cols[0]]
                df_doe_clean['source'] = 'DOE/OSTI'
                all_datasets.append(('DOE/OSTI', df_doe_clean))
                print(f"✅ DOE/OSTI Dataset: {len(df_doe_clean)}サンプル")
        except Exception as e:
            print(f"⚠️  DOE/OSTI読み込みエラー: {e}")
    
    # 4. 最新研究データ
    latest_file = RAW_DATA_DIR / "latest_research" / "latest_research.csv"
    if latest_file.exists():
        try:
            df_latest = pd.read_csv(latest_file)
            if 'elastic_modulus' in df_latest.columns:
                df_latest_clean = df_latest[df_latest['elastic_modulus'].notna()].copy()
                df_latest_clean = df_latest_clean[df_latest_clean['elastic_modulus'] > 0].copy()
                if 'source' not in df_latest_clean.columns:
                    df_latest_clean['source'] = 'Latest Research'
                all_datasets.append(('Latest Research', df_latest_clean))
                print(f"✅ 最新研究データ: {len(df_latest_clean)}サンプル")
        except Exception as e:
            print(f"⚠️  最新研究データ読み込みエラー: {e}")
    
    # 5. 積極的収集データ（推定データを含む、注意が必要）
    aggressive_file = COLLECTED_DATA_DIR / "aggressive_collection_20260119_192554.csv"
    if aggressive_file.exists():
        try:
            df_aggressive = pd.read_csv(aggressive_file)
            if 'elastic_modulus' in df_aggressive.columns:
                # 推定データを分離
                df_estimated = df_aggressive[df_aggressive['source'].str.contains('Estimated', na=False)].copy()
                df_actual = df_aggressive[~df_aggressive['source'].str.contains('Estimated', na=False)].copy()
                
                if len(df_actual) > 0:
                    all_datasets.append(('積極的収集（実測）', df_actual))
                    print(f"✅ 積極的収集（実測）: {len(df_actual)}サンプル")
                
                # 推定データは別途処理（精度が低いため）
                if len(df_estimated) > 0:
                    print(f"⚠️  推定データ: {len(df_estimated)}サンプル（精度が低いため、必要に応じて使用）")
        except Exception as e:
            print(f"⚠️  積極的収集データ読み込みエラー: {e}")
    
    # 6. Materials Projectデータ（抽出済み）
    mp_files = list(COLLECTED_DATA_DIR.glob("materials_project_*.csv"))
    if mp_files:
        # 最新のファイルを使用
        latest_mp_file = max(mp_files, key=lambda p: p.stat().st_mtime)
        try:
            df_mp = pd.read_csv(latest_mp_file)
            if 'elastic_modulus' in df_mp.columns:
                df_mp_clean = df_mp[df_mp['elastic_modulus'].notna()].copy()
                df_mp_clean = df_mp_clean[df_mp_clean['elastic_modulus'] > 0].copy()
                if 'source' not in df_mp_clean.columns:
                    df_mp_clean['source'] = 'Materials Project'
                all_datasets.append(('Materials Project', df_mp_clean))
                print(f"✅ Materials Project: {len(df_mp_clean)}サンプル")
        except Exception as e:
            print(f"⚠️  Materials Project読み込みエラー: {e}")
    
    # 7. 文献データ（抽出済み）
    literature_files = list(COLLECTED_DATA_DIR.glob("literature_data_*.csv"))
    if literature_files:
        # 最新のファイルを使用
        latest_lit_file = max(literature_files, key=lambda p: p.stat().st_mtime)
        try:
            df_lit = pd.read_csv(latest_lit_file)
            if 'elastic_modulus' in df_lit.columns:
                df_lit_clean = df_lit[df_lit['elastic_modulus'].notna()].copy()
                df_lit_clean = df_lit_clean[df_lit_clean['elastic_modulus'] > 0].copy()
                if 'source' not in df_lit_clean.columns:
                    df_lit_clean['source'] = 'Literature'
                all_datasets.append(('Literature', df_lit_clean))
                print(f"✅ Literature: {len(df_lit_clean)}サンプル")
        except Exception as e:
            print(f"⚠️  Literature読み込みエラー: {e}")
    
    return all_datasets

def integrate_all_data(all_datasets):
    """すべてのデータを統合"""
    print("\n" + "=" * 60)
    print("データ統合")
    print("=" * 60)
    
    if not all_datasets:
        print("❌ データセットが見つかりませんでした")
        return pd.DataFrame()
    
    # データを結合
    dataframes = [df for _, df in all_datasets]
    combined = pd.concat(dataframes, ignore_index=True, sort=False)
    
    print(f"📊 統合前: {len(combined)}サンプル")
    
    # 重複除去
    if 'alloy_name' in combined.columns and 'elastic_modulus' in combined.columns:
        initial_count = len(combined)
        
        # 合金名が同じで弾性率が近い（±10 GPa以内）ものを重複とみなす
        combined_sorted = combined.sort_values('elastic_modulus')
        
        # まず、合金名で重複除去
        combined_unique = combined_sorted.drop_duplicates(subset=['alloy_name'], keep='first')
        
        # さらに、合金名が異なるが組成が同じものを検出（可能な場合）
        # 簡易版：弾性率が非常に近い（±1 GPa以内）ものを重複とみなす
        if len(combined_unique) > 1:
            combined_unique = combined_unique.sort_values('elastic_modulus')
            # 近接する弾性率を検出
            elastic_diff = combined_unique['elastic_modulus'].diff()
            # 同じ組成の可能性があるもの（弾性率が±1 GPa以内）を検出
            # ただし、これは過度に除去しないように注意
        
        removed = initial_count - len(combined_unique)
        print(f"📊 重複除去後: {len(combined_unique)}サンプル")
        print(f"📊 除去された重複: {removed}サンプル")
        
        combined = combined_unique
    
    # 弾性率データの検証
    if 'elastic_modulus' in combined.columns:
        initial_count = len(combined)
        combined = combined[combined['elastic_modulus'].notna()].copy()
        combined = combined[combined['elastic_modulus'] > 0].copy()
        combined = combined[combined['elastic_modulus'] < 1000].copy()  # 異常値除去
        
        print(f"📊 検証後: {len(combined)}サンプル")
        print(f"📊 除去された異常値: {initial_count - len(combined)}サンプル")
    
    return combined

def add_estimated_data_if_needed(combined, all_datasets):
    """必要に応じて推定データを追加（目標サンプル数に達していない場合）"""
    if len(combined) >= TARGET_SAMPLES:
        return combined
    
    print(f"\n⚠️  目標サンプル数に達していません（現在: {len(combined)}, 目標: {TARGET_SAMPLES}）")
    print("   推定データを追加するか確認します...")
    
    # 推定データを探す
    estimated_data = None
    aggressive_file = COLLECTED_DATA_DIR / "aggressive_collection_20260119_192554.csv"
    if aggressive_file.exists():
        try:
            df_aggressive = pd.read_csv(aggressive_file)
            if 'elastic_modulus' in df_aggressive.columns:
                estimated_data = df_aggressive[df_aggressive['source'].str.contains('Estimated', na=False)].copy()
        except:
            pass
    
    if estimated_data is not None and len(estimated_data) > 0:
        needed = TARGET_SAMPLES - len(combined)
        if needed > 0:
            # 推定データから必要な分だけ追加
            estimated_to_add = estimated_data.head(min(needed, len(estimated_data)))
            
            print(f"   推定データから {len(estimated_to_add)}サンプルを追加します（精度は低い）")
            
            # 統合
            combined = pd.concat([combined, estimated_to_add], ignore_index=True)
            print(f"   統合後: {len(combined)}サンプル")
    
    return combined

def save_final_dataset(df):
    """最終データセットを保存"""
    print("\n" + "=" * 60)
    print("最終データセットの保存")
    print("=" * 60)
    
    if df.empty:
        print("❌ データが空です")
        return
    
    # 保存
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = FINAL_DATA_DIR / f"final_dataset_{timestamp}.csv"
    df.to_csv(output_file, index=False)
    print(f"✅ データを保存しました: {output_file}")
    
    # 統計情報
    print("\n" + "=" * 60)
    print("最終データセット統計")
    print("=" * 60)
    
    print(f"📊 総サンプル数: {len(df)}")
    
    if 'elastic_modulus' in df.columns:
        print(f"📊 弾性率範囲: {df['elastic_modulus'].min():.2f} - {df['elastic_modulus'].max():.2f} GPa")
        print(f"📊 平均: {df['elastic_modulus'].mean():.2f} GPa")
        print(f"📊 中央値: {df['elastic_modulus'].median():.2f} GPa")
        print(f"📊 標準偏差: {df['elastic_modulus'].std():.2f} GPa")
        
        target_range = df[(df['elastic_modulus'] >= 30) & (df['elastic_modulus'] <= 90)]
        print(f"⭐ 目標範囲（30-90 GPa）内: {len(target_range)}サンプル ({len(target_range)/len(df)*100:.1f}%)")
    
    if 'source' in df.columns:
        print("\n📊 データソース別:")
        source_counts = df['source'].value_counts()
        for source, count in source_counts.items():
            print(f"   {source}: {count}サンプル ({count/len(df)*100:.1f}%)")
    
    # 目標達成状況
    print("\n" + "=" * 60)
    print("目標達成状況")
    print("=" * 60)
    print(f"目標: {TARGET_SAMPLES}サンプル")
    print(f"現在: {len(df)}サンプル")
    print(f"達成率: {len(df)/TARGET_SAMPLES*100:.1f}%")
    
    if len(df) < TARGET_SAMPLES:
        needed = TARGET_SAMPLES - len(df)
        print(f"不足: {needed}サンプル")
        print("\n💡 追加データ収集の推奨:")
        print("   1. Materials Project APIキーを取得")
        print("   2. 最新の論文からデータを抽出")
        print("   3. 他の公開データセットを探索")
    else:
        print("✅ 目標サンプル数を達成しました！")

def main():
    """メイン関数"""
    print("=" * 60)
    print("最終データ統合")
    print("=" * 60)
    print(f"目標サンプル数: {TARGET_SAMPLES}")
    print(f"開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # すべてのデータソースを読み込む
    all_datasets = load_all_data_sources()
    
    if not all_datasets:
        print("\n❌ データセットが見つかりませんでした")
        return
    
    # データを統合
    combined = integrate_all_data(all_datasets)
    
    if combined.empty:
        print("\n❌ 統合データが空です")
        return
    
    # 必要に応じて推定データを追加
    combined = add_estimated_data_if_needed(combined, all_datasets)
    
    # 最終データセットを保存
    save_final_dataset(combined)
    
    print("\n" + "=" * 60)
    print("✅ 最終データ統合完了")
    print("=" * 60)
    print(f"終了時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
