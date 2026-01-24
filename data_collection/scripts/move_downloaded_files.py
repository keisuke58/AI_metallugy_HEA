#!/usr/bin/env python3
"""
ダウンロードしたファイルを適切なディレクトリに移動するスクリプト
"""

import os
import shutil
from pathlib import Path
import zipfile

# 設定
BASE_DIR = Path(__file__).parent.parent
RAW_DATA_DIR = BASE_DIR / "raw_data"
DOWNLOADS_DIR = Path.home() / "Downloads"

# ターゲットディレクトリ
TARGET_DIRS = {
    "disma": RAW_DATA_DIR / "disma_hea_dataset",
    "mpea": RAW_DATA_DIR / "mpea_mechanical_properties",
    "fracture": RAW_DATA_DIR / "fracture_toughness_dataset",
}

def find_and_move_files():
    """
    ダウンロードフォルダからファイルを検索して移動
    """
    print("=" * 60)
    print("ダウンロードファイルの検索と移動")
    print("=" * 60)
    
    # ターゲットディレクトリを作成
    for target_dir in TARGET_DIRS.values():
        target_dir.mkdir(parents=True, exist_ok=True)
    
    # 検索パターン
    search_patterns = {
        "disma": ["disma", "p3txdrdth7", "hea", "high-entropy"],
        "mpea": ["mpea", "4d4kpfwpf6", "multi-principal", "mechanical-properties"],
        "fracture": ["fracture", "toughness", "impact"],
    }
    
    moved_files = []
    
    # ダウンロードフォルダ内のファイルを検索
    if not DOWNLOADS_DIR.exists():
        print(f"\n❌ ダウンロードフォルダが見つかりません: {DOWNLOADS_DIR}")
        return moved_files
    
    print(f"\n📁 検索中: {DOWNLOADS_DIR}")
    
    for file_path in DOWNLOADS_DIR.iterdir():
        if not file_path.is_file():
            continue
        
        file_name_lower = file_path.name.lower()
        
        # 各パターンでマッチング
        for category, patterns in search_patterns.items():
            for pattern in patterns:
                if pattern.lower() in file_name_lower:
                    target_dir = TARGET_DIRS[category]
                    target_path = target_dir / file_path.name
                    
                    try:
                        # ファイルを移動
                        if not target_path.exists():
                            shutil.move(str(file_path), str(target_path))
                            print(f"✅ 移動: {file_path.name} → {target_dir}")
                            moved_files.append((file_path.name, target_dir))
                            
                            # ZIPファイルの場合は解凍も試行
                            if file_path.suffix.lower() == '.zip':
                                try:
                                    with zipfile.ZipFile(target_path, 'r') as zip_ref:
                                        zip_ref.extractall(target_dir)
                                    print(f"   📦 解凍完了: {target_dir}")
                                except Exception as e:
                                    print(f"   ⚠️  解凍失敗: {e}")
                        else:
                            print(f"⚠️  既に存在: {target_path.name}")
                    except Exception as e:
                        print(f"❌ 移動失敗: {file_path.name} - {e}")
                    break
    
    return moved_files

def check_existing_files():
    """
    既存のファイルを確認
    """
    print("\n" + "=" * 60)
    print("既存ファイルの確認")
    print("=" * 60)
    
    for category, target_dir in TARGET_DIRS.items():
        print(f"\n📁 {category.upper()}: {target_dir}")
        
        if not target_dir.exists():
            print("   ❌ ディレクトリが存在しません")
            continue
        
        files = list(target_dir.glob("*"))
        data_files = [f for f in files if f.is_file() and not f.name.startswith('.')]
        
        if data_files:
            print(f"   ✅ {len(data_files)}個のファイルが見つかりました:")
            for f in data_files:
                size = f.stat().st_size / 1024  # KB
                print(f"      - {f.name} ({size:.2f} KB)")
        else:
            print("   ❌ データファイルが見つかりませんでした")

if __name__ == "__main__":
    # 既存ファイルを確認
    check_existing_files()
    
    # ファイルを検索して移動
    moved = find_and_move_files()
    
    if moved:
        print(f"\n✅ {len(moved)}個のファイルを移動しました")
        print("\n再度確認:")
        check_existing_files()
    else:
        print("\n⚠️  移動するファイルが見つかりませんでした")
        print("\n手動で移動する場合:")
        print("1. ダウンロードしたファイルを確認")
        print("2. 以下のコマンドで移動:")
        print(f"   mv ~/Downloads/*disma* {TARGET_DIRS['disma']}/")
        print(f"   mv ~/Downloads/*mpea* {TARGET_DIRS['mpea']}/")
