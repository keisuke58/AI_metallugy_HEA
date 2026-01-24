#!/usr/bin/env python3
"""
Gorsse Datasetの詳細分析スクリプト
"""

import pandas as pd
import numpy as np
from pathlib import Path

# 設定
BASE_DIR = Path(__file__).parent.parent
RAW_DATA_DIR = BASE_DIR / "raw_data" / "gorsse_dataset"

def analyze_gorsse_excel():
    """
    Gorsse DatasetのExcelファイルを分析
    """
    print("=" * 60)
    print("Gorsse Dataset Excel 分析")
    print("=" * 60)
    
    excel_file = RAW_DATA_DIR / "1-s2.0-S2352340920311100-mmc1.xlsx"
    
    if not excel_file.exists():
        print(f"❌ ファイルが見つかりません: {excel_file}")
        return None
    
    print(f"\n📁 ファイル: {excel_file.name}")
    
    # Excelファイルのシート名を確認
    try:
        xl_file = pd.ExcelFile(excel_file)
        print(f"\n📋 シート一覧: {xl_file.sheet_names}")
        
        # 各シートを分析
        all_data = {}
        for sheet_name in xl_file.sheet_names:
            # Table 1の場合はヘッダー行が6行目（0-indexedで6）
            if sheet_name == 'Table 1':
                df = pd.read_excel(excel_file, sheet_name=sheet_name, header=6)
                # 最初の空行を削除
                df = df.dropna(how='all')
                # 最初の行が説明行の場合は削除
                if df.iloc[0, 0] == '3d TM HEAs and CCAs in the Al-Co-Cr-Fe-Mn-Ni system and derivates':
                    df = df.iloc[1:].reset_index(drop=True)
            elif sheet_name == 'Table 2':
                df = pd.read_excel(excel_file, sheet_name=sheet_name, header=4)
                df = df.dropna(how='all')
            else:
                df = pd.read_excel(excel_file, sheet_name=sheet_name)
            
            all_data[sheet_name] = df
            
            print(f"\n{'='*60}")
            print(f"シート: {sheet_name}")
            print(f"{'='*60}")
            print(f"📊 データ形状: {df.shape}")
            print(f"📋 カラム: {list(df.columns)}")
            
            # 弾性率データの有無を確認
            elastic_modulus_keywords = ['modulus', 'elastic', 'young', 'E', 'GPa', 'youngs', 'Young', 'E#']
            elastic_cols = [col for col in df.columns 
                          if any(kw in str(col).lower() or str(col).startswith('E') for kw in elastic_modulus_keywords)]
            
            if elastic_cols:
                print(f"\n✅ 弾性率データが見つかりました:")
                for col in elastic_cols:
                    valid_data = df[col].notna().sum()
                    print(f"   - {col}: {valid_data}個のデータ")
                    if valid_data > 0:
                        print(f"     統計: min={df[col].min():.2f}, max={df[col].max():.2f}, mean={df[col].mean():.2f}")
                        
                        # 目標範囲（30-90 GPa）内のデータを確認
                        target_range = df[(df[col] >= 30) & (df[col] <= 90)]
                        if len(target_range) > 0:
                            print(f"     ⭐ 目標範囲（30-90 GPa）内: {len(target_range)}個")
            else:
                print("\n⚠️  弾性率データが見つかりませんでした")
            
            # 最初の数行を表示
            print(f"\n📄 データの最初の5行:")
            print(df.head())
        
        return all_data
        
    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
        return None

def summarize_gorsse_data():
    """
    Gorsse Datasetのサマリーを作成
    """
    print("\n" + "=" * 60)
    print("Gorsse Dataset サマリー")
    print("=" * 60)
    
    excel_file = RAW_DATA_DIR / "1-s2.0-S2352340920311100-mmc1.xlsx"
    
    if not excel_file.exists():
        print("❌ Excelファイルが見つかりません")
        return
    
    try:
        xl_file = pd.ExcelFile(excel_file)
        
        total_rows = 0
        elastic_modulus_data = []
        
        for sheet_name in xl_file.sheet_names:
            df = pd.read_excel(excel_file, sheet_name=sheet_name)
            total_rows += len(df)
            
            # 弾性率データを探す
            elastic_modulus_keywords = ['modulus', 'elastic', 'young', 'E', 'GPa', 'youngs', 'Young']
            for col in df.columns:
                if any(kw in str(col).lower() for kw in elastic_modulus_keywords):
                    valid_data = df[col].notna().sum()
                    if valid_data > 0:
                        elastic_modulus_data.append({
                            'sheet': sheet_name,
                            'column': col,
                            'count': valid_data,
                            'min': df[col].min(),
                            'max': df[col].max(),
                            'mean': df[col].mean()
                        })
        
        print(f"\n📊 総データ数: {total_rows}行")
        print(f"📋 シート数: {len(xl_file.sheet_names)}")
        
        if elastic_modulus_data:
            print(f"\n✅ 弾性率データ:")
            total_elastic = sum(d['count'] for d in elastic_modulus_data)
            print(f"   総データ数: {total_elastic}個")
            for d in elastic_modulus_data:
                print(f"   - {d['sheet']} / {d['column']}: {d['count']}個")
                print(f"     範囲: {d['min']:.2f} - {d['max']:.2f} GPa, 平均: {d['mean']:.2f} GPa")
        else:
            print("\n⚠️  弾性率データが見つかりませんでした")
        
    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")

if __name__ == "__main__":
    data = analyze_gorsse_excel()
    summarize_gorsse_data()
