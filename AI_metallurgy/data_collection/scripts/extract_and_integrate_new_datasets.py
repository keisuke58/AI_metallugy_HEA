#!/usr/bin/env python3
"""
新しくダウンロードしたデータセットから弾性率データを抽出して統合
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).parent.parent
RAW_DATA_DIR = BASE_DIR / "raw_data"
COLLECTED_DATA_DIR = BASE_DIR / "collected_data"
FINAL_DATA_DIR = BASE_DIR / "final_data"
FINAL_DATA_DIR.mkdir(parents=True, exist_ok=True)

def extract_mpea_nanoindentation():
    """MPEA Nano-indentationデータから弾性率を抽出"""
    print("=" * 60)
    print("MPEA Nano-indentation データ抽出")
    print("=" * 60)
    
    file_path = RAW_DATA_DIR / "mpea_nanoindentation" / "MPEA_dataset.csv"
    if not file_path.exists():
        print("⚠️  ファイルが見つかりません")
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(file_path)
        print(f"📊 元データ: {len(df)}サンプル")
        
        # 弾性率カラムを探す
        exp_modulus_col = 'PROPERTY: Exp. Young modulus (GPa)'
        calc_modulus_col = 'PROPERTY: Calculated Young modulus (GPa)'
        formula_col = 'FORMULA'
        
        result_data = []
        
        for idx, row in df.iterrows():
            # 実験値の弾性率を優先
            elastic_modulus = None
            
            if exp_modulus_col in df.columns:
                exp_val = pd.to_numeric(row[exp_modulus_col], errors='coerce')
                if pd.notna(exp_val) and exp_val > 0:
                    elastic_modulus = exp_val
                    data_type = 'experimental'
            
            # 実験値がない場合は計算値を使用
            if elastic_modulus is None and calc_modulus_col in df.columns:
                calc_val = pd.to_numeric(row[calc_modulus_col], errors='coerce')
                if pd.notna(calc_val) and calc_val > 0:
                    elastic_modulus = calc_val
                    data_type = 'calculated'
            
            if elastic_modulus is not None:
                alloy_name = row[formula_col] if formula_col in df.columns else f'MPEA_{idx}'
                
                result_data.append({
                    'alloy_name': alloy_name,
                    'elastic_modulus': elastic_modulus,
                    'source': f'MPEA Nano-indentation ({data_type})',
                    'data_type': data_type,
                    'formula': row[formula_col] if formula_col in df.columns else None,
                    'reference_id': row.get('IDENTIFIER: Reference ID', None)
                })
        
        if result_data:
            df_result = pd.DataFrame(result_data)
            
            # 保存
            output_file = COLLECTED_DATA_DIR / "mpea_nanoindentation_elastic_modulus.csv"
            df_result.to_csv(output_file, index=False)
            print(f"✅ データを保存しました: {output_file}")
            print(f"📊 抽出データ数: {len(df_result)}サンプル")
            
            # 実験値と計算値の内訳
            exp_count = len(df_result[df_result['data_type'] == 'experimental'])
            calc_count = len(df_result[df_result['data_type'] == 'calculated'])
            print(f"   - 実験値: {exp_count}サンプル")
            print(f"   - 計算値: {calc_count}サンプル")
            
            if 'elastic_modulus' in df_result.columns:
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

def check_combined_data():
    """combined_data.csvも確認"""
    print("\n" + "=" * 60)
    print("Combined Data 確認")
    print("=" * 60)
    
    file_path = RAW_DATA_DIR / "mpea_nanoindentation" / "combined_data.csv"
    if not file_path.exists():
        print("⚠️  ファイルが見つかりません")
        return
    
    try:
        df = pd.read_csv(file_path, nrows=10)
        print(f"📊 ファイル: {file_path}")
        print(f"📊 カラム数: {len(df.columns)}")
        print(f"📊 カラム名（最初の10個）:")
        for col in df.columns[:10]:
            print(f"   - {col}")
        
        # 弾性率関連のカラムを探す
        elastic_cols = [col for col in df.columns if any(kw in str(col).lower() for kw in ['modulus', 'young', 'elastic', 'E'])]
        if elastic_cols:
            print(f"\n✅ 弾性率関連カラム: {elastic_cols}")
        else:
            print(f"\n⚠️  弾性率関連カラムが見つかりませんでした")
            
    except Exception as e:
        print(f"❌ エラー: {e}")

def integrate_with_existing():
    """既存の統合データセットに新しいデータを追加"""
    print("\n" + "=" * 60)
    print("既存データセットとの統合")
    print("=" * 60)
    
    # 既存の統合データセットを読み込む
    existing_file = FINAL_DATA_DIR / "unified_dataset_20260123_174509.csv"
    if not existing_file.exists():
        print("⚠️  既存の統合データセットが見つかりません")
        return
    
    try:
        df_existing = pd.read_csv(existing_file)
        print(f"📊 既存データ: {len(df_existing)}サンプル")
        
        # 新しいデータを読み込む
        new_file = COLLECTED_DATA_DIR / "mpea_nanoindentation_elastic_modulus.csv"
        if not new_file.exists():
            print("⚠️  新しいデータファイルが見つかりません")
            return
        
        df_new = pd.read_csv(new_file)
        print(f"📊 新規データ: {len(df_new)}サンプル")
        
        # カラムを統一
        # 既存データのカラムに合わせる
        df_new_standardized = pd.DataFrame()
        df_new_standardized['alloy_name'] = df_new['alloy_name']
        df_new_standardized['elastic_modulus'] = df_new['elastic_modulus']
        df_new_standardized['source'] = df_new['source']
        
        # 統合
        df_combined = pd.concat([df_existing, df_new_standardized], ignore_index=True, sort=False)
        print(f"📊 統合後: {len(df_combined)}サンプル")
        
        # 重複除去（合金名で）
        initial_count = len(df_combined)
        df_combined = df_combined.drop_duplicates(subset=['alloy_name'], keep='first')
        removed = initial_count - len(df_combined)
        print(f"📊 重複除去後: {len(df_combined)}サンプル（{removed}サンプル除去）")
        
        # 保存
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = FINAL_DATA_DIR / f"unified_dataset_with_mpea_{timestamp}.csv"
        df_combined.to_csv(output_file, index=False)
        print(f"✅ 統合データを保存しました: {output_file}")
        
        # 統計情報
        print("\n" + "=" * 60)
        print("統合後の統計")
        print("=" * 60)
        
        if 'elastic_modulus' in df_combined.columns:
            print(f"弾性率範囲: {df_combined['elastic_modulus'].min():.2f} - {df_combined['elastic_modulus'].max():.2f} GPa")
            print(f"平均: {df_combined['elastic_modulus'].mean():.2f} GPa")
            print(f"中央値: {df_combined['elastic_modulus'].median():.2f} GPa")
        
        if 'source' in df_combined.columns:
            print("\nデータソース別:")
            source_counts = df_combined['source'].value_counts()
            for source, count in source_counts.items():
                print(f"   {source}: {count:,}サンプル")
        
    except Exception as e:
        print(f"❌ エラー: {e}")
        import traceback
        traceback.print_exc()

def main():
    """メイン関数"""
    print("=" * 60)
    print("新規データセットの抽出と統合")
    print("=" * 60)
    print(f"開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # MPEA Nano-indentationデータを抽出
    df_mpea = extract_mpea_nanoindentation()
    
    # Combined Dataも確認
    check_combined_data()
    
    # 既存データセットと統合
    if not df_mpea.empty:
        integrate_with_existing()
    
    print("\n" + "=" * 60)
    print("✅ 抽出・統合完了")
    print("=" * 60)
    print(f"終了時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
