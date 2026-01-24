#!/usr/bin/env python3
"""
データセットのクリーンアップスクリプト
検証で見つかった問題を修正
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).parent.parent
FINAL_DATA_DIR = BASE_DIR / "final_data"
FINAL_DATA_DIR.mkdir(parents=True, exist_ok=True)

def clean_dataset(input_file, output_file=None):
    """データセットをクリーンアップ"""
    print("=" * 60)
    print("データセットクリーンアップ")
    print("=" * 60)
    
    if not input_file.exists():
        print(f"❌ ファイルが見つかりません: {input_file}")
        return None
    
    df = pd.read_csv(input_file)
    initial_count = len(df)
    print(f"📊 初期データ数: {initial_count:,}サンプル")
    
    # 1. 欠損値の処理
    print("\n1. 欠損値の処理")
    print("-" * 60)
    
    # 合金名が欠損しているサンプルを削除
    before = len(df)
    df = df[df['alloy_name'].notna() & (df['alloy_name'] != '')].copy()
    removed = before - len(df)
    if removed > 0:
        print(f"   ✅ 合金名が欠損している{removed}サンプルを削除")
    
    # 弾性率が欠損しているサンプルを削除
    before = len(df)
    df = df[df['elastic_modulus'].notna()].copy()
    removed = before - len(df)
    if removed > 0:
        print(f"   ✅ 弾性率が欠損している{removed}サンプルを削除")
    
    # 2. 無効な値の処理
    print("\n2. 無効な値の処理")
    print("-" * 60)
    
    # 弾性率を数値に変換
    df['elastic_modulus'] = pd.to_numeric(df['elastic_modulus'], errors='coerce')
    
    # 負の値やゼロを削除
    before = len(df)
    df = df[df['elastic_modulus'] > 0].copy()
    removed = before - len(df)
    if removed > 0:
        print(f"   ✅ 無効な値（≤0）の{removed}サンプルを削除")
    
    # 3. 極端な値の確認と処理
    print("\n3. 極端な値の確認")
    print("-" * 60)
    
    # 一般的な金属材料の範囲: 10-500 GPa
    extreme_low = df[df['elastic_modulus'] < 10]
    extreme_high = df[df['elastic_modulus'] > 500]
    
    print(f"   📊 <10 GPa: {len(extreme_low):,}サンプル")
    if len(extreme_low) > 0:
        print(f"      例: {extreme_low[['alloy_name', 'elastic_modulus', 'source']].head().to_string()}")
    
    print(f"   📊 >500 GPa: {len(extreme_high):,}サンプル")
    if len(extreme_high) > 0:
        print(f"      例: {extreme_high[['alloy_name', 'elastic_modulus', 'source']].head().to_string()}")
    
    # 極端な値は警告のみ（削除しない）- データソースに依存するため
    print(f"   ⚠️  極端な値は保持します（データソースに依存）")
    
    # 4. 重複の最終確認
    print("\n4. 重複の最終確認")
    print("-" * 60)
    
    before = len(df)
    df = df.drop_duplicates(subset=['alloy_name'], keep='first')
    removed = before - len(df)
    if removed > 0:
        print(f"   ✅ {removed}サンプルの重複を削除")
    else:
        print(f"   ✅ 重複はありません")
    
    # 5. データ型の最適化
    print("\n5. データ型の最適化")
    print("-" * 60)
    
    # 弾性率をfloat32に（メモリ節約）
    df['elastic_modulus'] = df['elastic_modulus'].astype('float32')
    print(f"   ✅ データ型を最適化")
    
    # 6. 最終統計
    print("\n" + "=" * 60)
    print("クリーンアップ結果")
    print("=" * 60)
    
    final_count = len(df)
    removed_total = initial_count - final_count
    
    print(f"📊 初期データ数: {initial_count:,}サンプル")
    print(f"📊 削除されたデータ: {removed_total:,}サンプル")
    print(f"📊 最終データ数: {final_count:,}サンプル")
    print(f"📊 保持率: {final_count/initial_count*100:.2f}%")
    
    if 'elastic_modulus' in df.columns:
        valid_data = df['elastic_modulus'].dropna()
        print(f"\n📊 弾性率統計:")
        print(f"   - 最小値: {valid_data.min():.2f} GPa")
        print(f"   - 最大値: {valid_data.max():.2f} GPa")
        print(f"   - 平均: {valid_data.mean():.2f} GPa")
        print(f"   - 中央値: {valid_data.median():.2f} GPa")
    
    # 7. 保存
    if output_file is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = FINAL_DATA_DIR / f"unified_dataset_cleaned_{timestamp}.csv"
    
    # タイムスタンプ付きのファイルとして保存
    df.to_csv(output_file, index=False)
    print(f"\n✅ クリーンアップ済みデータを保存しました: {output_file}")

    # 安定したパス（最新ファイル）としても保存
    latest_file = FINAL_DATA_DIR / "unified_dataset_latest.csv"
    df.to_csv(latest_file, index=False)
    print(f"✅ 最新データセットとして保存しました: {latest_file}")
    
    return df

def main():
    """メイン関数"""
    # 最新の統合データセットを探す
    unified_files = list(FINAL_DATA_DIR.glob("unified_dataset*.csv"))
    
    if not unified_files:
        print("❌ 統合データセットが見つかりません")
        return
    
    # 最新のファイルを使用
    latest_file = max(unified_files, key=lambda p: p.stat().st_mtime)
    
    print(f"📁 入力ファイル: {latest_file.name}")
    print()
    
    # クリーンアップ
    cleaned_df = clean_dataset(latest_file)
    
    if cleaned_df is not None:
        print("\n" + "=" * 60)
        print("✅ クリーンアップ完了")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("❌ クリーンアップ失敗")
        print("=" * 60)

if __name__ == "__main__":
    main()
