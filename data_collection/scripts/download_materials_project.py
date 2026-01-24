#!/usr/bin/env python3
"""
Materials Project APIを使用してHEAデータを取得

注意: APIキーが必要です
"""

import os
import pandas as pd
from pathlib import Path

# 設定
BASE_DIR = Path(__file__).parent.parent
RAW_DATA_DIR = BASE_DIR / "raw_data" / "materials_project"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

def download_materials_project_data():
    """
    Materials ProjectからHEAデータを取得
    
    注意: APIキーが必要です
    """
    print("=" * 60)
    print("Materials Project データ取得")
    print("=" * 60)
    
    print("\n⚠️  注意: Materials Project APIを使用するにはAPIキーが必要です。")
    print("\n手順:")
    print("1. Materials Projectにアカウント作成:")
    print("   https://materialsproject.org")
    print("2. APIキーを取得（Dashboard > API Keys）")
    print("3. 環境変数に設定:")
    print("   export MP_API_KEY='your_api_key_here'")
    print("   または、.envファイルに保存")
    print("\n4. 以下のコマンドでデータを取得:")
    print("   python scripts/download_materials_project.py --api-key YOUR_KEY")
    
    # APIキーの確認
    api_key = os.getenv("MP_API_KEY")
    if not api_key:
        print("\n❌ APIキーが設定されていません。")
        print("   環境変数 MP_API_KEY を設定するか、--api-key オプションを使用してください。")
        return
    
    try:
        from mp_api.client import MPRester
        
        print(f"\n✅ APIキーが見つかりました")
        print("Materials Projectからデータを取得中...")
        
        # HEAに関連する元素で検索
        hea_elements = ["Ti", "Zr", "Hf", "Nb", "Ta", "V", "Cr", "Mo", "W", 
                       "Fe", "Co", "Ni", "Cu", "Al", "Mn"]
        
        with MPRester(api_key) as mpr:
            # 金属材料を検索
            summaries = mpr.materials.summary.search(
                elements=hea_elements[:5],  # 最初の5元素で検索
                is_metal=True,
                fields=["material_id", "formula_pretty", "bulk_modulus", "shear_modulus"]
            )
            
            print(f"\n✅ {len(summaries)}個の材料が見つかりました")
            
            # データをDataFrameに変換
            data = []
            for s in summaries:
                data.append({
                    'material_id': s.material_id,
                    'formula': s.formula_pretty,
                    'bulk_modulus': s.bulk_modulus,
                    'shear_modulus': s.shear_modulus,
                })
            
            df = pd.DataFrame(data)
            
            # ファイルに保存
            output_file = RAW_DATA_DIR / "materials_project_hea.csv"
            df.to_csv(output_file, index=False)
            print(f"\n✅ データを保存しました: {output_file}")
            print(f"   データ数: {len(df)}")
            
            return df
            
    except ImportError:
        print("\n❌ mp-apiパッケージがインストールされていません。")
        print("   インストール: pip install mp-api")
        return
    except Exception as e:
        print(f"\n❌ エラーが発生しました: {e}")
        return

if __name__ == "__main__":
    import sys
    
    # コマンドライン引数からAPIキーを取得
    if len(sys.argv) > 1 and sys.argv[1] == "--api-key" and len(sys.argv) > 2:
        os.environ["MP_API_KEY"] = sys.argv[2]
    
    download_materials_project_data()
