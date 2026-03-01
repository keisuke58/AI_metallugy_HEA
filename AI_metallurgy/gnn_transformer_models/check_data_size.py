#!/usr/bin/env python3
"""
データセットのサイズを確認するスクリプト
"""
import pandas as pd
from pathlib import Path
import sys

_script_dir = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = _script_dir.parent / "data_collection" / "processed_data" / "data_with_features.csv"

def check_data_size(data_path):
    """データセットのサイズを確認"""
    data_path = Path(data_path)
    
    if not data_path.exists():
        print(f"❌ データファイルが見つかりません: {data_path}")
        return False
    
    print("=" * 80)
    print("データセットサイズ確認")
    print("=" * 80)
    print(f"📊 データファイル: {data_path}")
    
    try:
        df = pd.read_csv(data_path)
        total_rows = len(df)
        
        # elastic_modulusが存在するか確認
        if 'elastic_modulus' in df.columns:
            valid_rows = len(df[df['elastic_modulus'].notna()])
            print(f"📊 総行数: {total_rows}")
            print(f"📊 有効なデータ数（elastic_modulusあり）: {valid_rows}")
            
            if valid_rows >= 5000:
                print(f"\n✅ データ数が{valid_rows}で5000以上です！")
                print("   → train_large_dataset.py の使用を推奨します")
                return True
            else:
                print(f"\n⚠️  データ数が{valid_rows}で5000未満です")
                print("   → train.py の使用を推奨します")
                return False
        else:
            print(f"📊 総行数: {total_rows}")
            print("⚠️  'elastic_modulus'カラムが見つかりません")
            return False
            
    except Exception as e:
        print(f"❌ エラー: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    else:
        data_path = DEFAULT_DATA_PATH
    
    check_data_size(data_path)
