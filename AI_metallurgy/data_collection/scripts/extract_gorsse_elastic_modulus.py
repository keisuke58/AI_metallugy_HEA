#!/usr/bin/env python3
"""
Gorsse Datasetから弾性率データを抽出
"""

import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
RAW_DATA_DIR = BASE_DIR / "raw_data" / "gorsse_dataset"
COLLECTED_DATA_DIR = BASE_DIR / "collected_data"
COLLECTED_DATA_DIR.mkdir(parents=True, exist_ok=True)

def extract_gorsse_elastic_modulus():
    """Gorsse Datasetから弾性率データを抽出"""
    print("=" * 60)
    print("Gorsse Dataset 弾性率データ抽出")
    print("=" * 60)
    
    excel_file = RAW_DATA_DIR / "1-s2.0-S2352340920311100-mmc1.xlsx"
    
    if not excel_file.exists():
        print(f"❌ ファイルが見つかりません: {excel_file}")
        return pd.DataFrame()
    
    try:
        # Table 1を読み込む（ヘッダー行6）
        df = pd.read_excel(excel_file, sheet_name='Table 1', header=6)
        df = df.dropna(how='all')
        
        # 弾性率カラムを確認
        if 'E# (GPa)' not in df.columns:
            print("❌ 弾性率カラムが見つかりません")
            return pd.DataFrame()
        
        # 弾性率データがある行を抽出
        df_clean = df[df['E# (GPa)'].notna()].copy()
        df_clean = df_clean[df_clean['E# (GPa)'] > 0].copy()
        
        # カラム名を整理
        df_clean['elastic_modulus'] = df_clean['E# (GPa)']
        df_clean['alloy_name'] = df_clean.get('Composition (atomic)', df_clean.get('Composition', ''))
        df_clean['density'] = df_clean.get('r# (g/cm3)', df_clean.get('r (g/cm3)', np.nan))
        df_clean['hardness'] = df_clean.get('HV', np.nan)
        df_clean['yield_strength'] = df_clean.get('sy (MPa)', np.nan)
        df_clean['ultimate_strength'] = df_clean.get('smax (MPa)', np.nan)
        df_clean['elongation'] = df_clean.get('e (%)', np.nan)
        df_clean['phases'] = df_clean.get('Type of phases', np.nan)
        df_clean['source'] = 'Gorsse Dataset'
        
        # 必要なカラムのみ選択
        output_cols = ['alloy_name', 'elastic_modulus', 'density', 'hardness', 
                      'yield_strength', 'ultimate_strength', 'elongation', 
                      'phases', 'source']
        available_cols = [col for col in output_cols if col in df_clean.columns]
        df_output = df_clean[available_cols].copy()
        
        print(f"✅ {len(df_output)}サンプルを抽出しました")
        print(f"📊 弾性率範囲: {df_output['elastic_modulus'].min():.2f} - {df_output['elastic_modulus'].max():.2f} GPa")
        print(f"📊 平均: {df_output['elastic_modulus'].mean():.2f} GPa")
        
        target_range = df_output[(df_output['elastic_modulus'] >= 30) & (df_output['elastic_modulus'] <= 90)]
        print(f"⭐ 目標範囲（30-90 GPa）内: {len(target_range)}サンプル")
        
        # 保存
        output_file = COLLECTED_DATA_DIR / "gorsse_elastic_modulus.csv"
        df_output.to_csv(output_file, index=False)
        print(f"\n✅ データを保存しました: {output_file}")
        
        return df_output
        
    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

if __name__ == "__main__":
    extract_gorsse_elastic_modulus()
