#!/usr/bin/env python3
"""
データ収集状況の確認スクリプト
"""

from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).parent.parent
RAW_DATA_DIR = BASE_DIR / "raw_data"

def check_data_status():
    """
    データ収集の状況を確認
    """
    print("=" * 60)
    print("データ収集状況の確認")
    print("=" * 60)
    
    datasets = {
        "Gorsse Dataset": RAW_DATA_DIR / "gorsse_dataset",
        "DOE/OSTI Dataset": RAW_DATA_DIR / "doe_osti_dataset",
        "Materials Project": RAW_DATA_DIR / "materials_project",
        "最新研究データ": RAW_DATA_DIR / "latest_research",
    }
    
    total_files = 0
    total_size = 0
    
    for name, dir_path in datasets.items():
        print(f"\n📊 {name}:")
        if dir_path.exists():
            files = list(dir_path.glob("*"))
            data_files = [f for f in files if f.is_file() and not f.name.startswith('.')]
            
            if data_files:
                print(f"   ✅ {len(data_files)}個のファイルが見つかりました")
                for file in data_files:
                    size = file.stat().st_size / 1024  # KB
                    total_size += size
                    print(f"      - {file.name} ({size:.2f} KB)")
                total_files += len(data_files)
            else:
                print(f"   ❌ データファイルが見つかりませんでした")
        else:
            print(f"   ❌ ディレクトリが存在しません")
    
    print("\n" + "=" * 60)
    print("📈 総合状況")
    print("=" * 60)
    print(f"総ファイル数: {total_files}")
    print(f"総データサイズ: {total_size:.2f} KB ({total_size/1024:.2f} MB)")
    
    if total_files == 0:
        print("\n⚠️  データファイルが見つかりませんでした。")
        print("   データ収集を開始してください。")
    elif total_files < 2:
        print("\n⚠️  データファイルが少ないです。")
        print("   複数のデータセットを収集することを推奨します。")
    else:
        print("\n✅ データ収集が進行中です。")
    
    return total_files, total_size

if __name__ == "__main__":
    check_data_status()
