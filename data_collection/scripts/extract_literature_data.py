#!/usr/bin/env python3
"""
文献データ抽出スクリプト
raw_data/literature_data/ 内のCSVファイルを統合
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).parent.parent
RAW_DATA_DIR = BASE_DIR / "raw_data" / "literature_data"
COLLECTED_DATA_DIR = BASE_DIR / "collected_data"
COLLECTED_DATA_DIR.mkdir(parents=True, exist_ok=True)

def extract_literature_data():
    """文献データを抽出・統合"""
    print("=" * 60)
    print("文献データ抽出")
    print("=" * 60)
    
    if not RAW_DATA_DIR.exists():
        print(f"❌ ディレクトリが見つかりません: {RAW_DATA_DIR}")
        print("   文献データを抽出して、このディレクトリに保存してください")
        return pd.DataFrame()
    
    # すべてのCSVファイルを検索
    csv_files = list(RAW_DATA_DIR.rglob("*.csv"))
    
    if not csv_files:
        print(f"❌ CSVファイルが見つかりませんでした: {RAW_DATA_DIR}")
        print("\n💡 文献データを抽出する手順:")
        print("   1. 論文からデータを抽出")
        print("   2. CSVファイルに保存")
        print(f"   3. {RAW_DATA_DIR} に配置")
        return pd.DataFrame()
    
    print(f"📁 {len(csv_files)}個のCSVファイルが見つかりました")
    
    all_data = []
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            
            # 弾性率カラムを探す
            elastic_cols = [col for col in df.columns 
                          if any(kw in str(col).lower() for kw in 
                                ['modulus', 'young', 'elastic', 'E', 'E#'])]
            
            if elastic_cols:
                col_name = elastic_cols[0]
                df_clean = df[df[col_name].notna()].copy()
                df_clean = df_clean[df_clean[col_name] > 0].copy()
                
                if len(df_clean) > 0:
                    # カラム名を統一
                    df_clean['elastic_modulus'] = df_clean[col_name]
                    
                    # 合金名を探す
                    name_cols = [col for col in df_clean.columns 
                               if any(kw in str(col).lower() for kw in 
                                     ['alloy', 'name', 'composition', 'formula'])]
                    if name_cols:
                        df_clean['alloy_name'] = df_clean[name_cols[0]]
                    
                    # ソース情報
                    if 'source' not in df_clean.columns:
                        df_clean['source'] = f'Literature_{csv_file.stem}'
                    if 'reference' not in df_clean.columns:
                        df_clean['reference'] = csv_file.stem
                    if 'year' not in df_clean.columns:
                        # ファイル名やパスから年を推測
                        year_match = None
                        for part in csv_file.parts:
                            if part.isdigit() and len(part) == 4:
                                year_match = int(part)
                                break
                        df_clean['year'] = year_match if year_match else None
                    
                    all_data.append(df_clean)
                    print(f"   ✅ {csv_file.name}: {len(df_clean)}サンプル")
            else:
                print(f"   ⚠️  {csv_file.name}: 弾性率カラムが見つかりませんでした")
                
        except Exception as e:
            print(f"   ❌ {csv_file.name}: エラー - {e}")
            continue
    
    if all_data:
        combined = pd.concat(all_data, ignore_index=True, sort=False)
        
        # 重複除去
        if 'alloy_name' in combined.columns:
            initial_count = len(combined)
            combined = combined.drop_duplicates(subset=['alloy_name'], keep='first')
            removed = initial_count - len(combined)
            if removed > 0:
                print(f"\n📊 重複除去: {removed}サンプル")
        
        # 弾性率データの検証
        if 'elastic_modulus' in combined.columns:
            initial_count = len(combined)
            combined = combined[combined['elastic_modulus'].notna()].copy()
            combined = combined[combined['elastic_modulus'] > 0].copy()
            combined = combined[combined['elastic_modulus'] < 1000].copy()
            removed = initial_count - len(combined)
            if removed > 0:
                print(f"📊 異常値除去: {removed}サンプル")
        
        print(f"\n📊 統合結果: {len(combined)}サンプル")
        
        if 'elastic_modulus' in combined.columns:
            print(f"📊 弾性率範囲: {combined['elastic_modulus'].min():.2f} - {combined['elastic_modulus'].max():.2f} GPa")
            print(f"📊 平均: {combined['elastic_modulus'].mean():.2f} GPa")
            
            target_range = combined[(combined['elastic_modulus'] >= 30) & (combined['elastic_modulus'] <= 90)]
            print(f"⭐ 目標範囲（30-90 GPa）内: {len(target_range)}サンプル")
        
        # 保存
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = COLLECTED_DATA_DIR / f"literature_data_{timestamp}.csv"
        combined.to_csv(output_file, index=False)
        print(f"\n✅ データを保存しました: {output_file}")
        
        return combined
    else:
        print("\n❌ データを抽出できませんでした")
        return pd.DataFrame()

if __name__ == "__main__":
    extract_literature_data()
