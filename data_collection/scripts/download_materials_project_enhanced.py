#!/usr/bin/env python3
"""
Materials Project APIを使用してHEA弾性率データを取得（改善版）

特徴:
- 弾性率データを適切に計算（Young's modulus）
- 複数の元素組み合わせで検索
- 大量データの取得に対応
- 進捗表示
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 設定
BASE_DIR = Path(__file__).parent.parent
RAW_DATA_DIR = BASE_DIR / "raw_data" / "materials_project"
COLLECTED_DATA_DIR = BASE_DIR / "collected_data"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
COLLECTED_DATA_DIR.mkdir(parents=True, exist_ok=True)

def calculate_youngs_modulus(bulk_modulus, shear_modulus):
    """
    バルク弾性率とせん断弾性率からYoung's modulusを計算
    
    E = 9*K*G / (3*K + G)
    
    Args:
        bulk_modulus: バルク弾性率（GPa）
        shear_modulus: せん断弾性率（GPa）
    
    Returns:
        Young's modulus（GPa）
    """
    if bulk_modulus is None or shear_modulus is None:
        return None
    
    if isinstance(bulk_modulus, dict):
        # VRH平均を使用
        K = bulk_modulus.get('vrh', bulk_modulus.get('voigt', None))
    else:
        K = bulk_modulus
    
    if isinstance(shear_modulus, dict):
        # VRH平均を使用
        G = shear_modulus.get('vrh', shear_modulus.get('voigt', None))
    else:
        G = shear_modulus
    
    if K is None or G is None:
        return None
    
    if K <= 0 or G <= 0:
        return None
    
    denominator = 3 * K + G
    if denominator == 0:
        return None
    
    E = (9 * K * G) / denominator
    return E

def download_materials_project_data(api_key=None, max_materials=5000):
    """
    Materials ProjectからHEA弾性率データを取得
    
    Args:
        api_key: Materials Project APIキー
        max_materials: 取得する最大材料数
    """
    print("=" * 60)
    print("Materials Project データ取得（改善版）")
    print("=" * 60)
    
    # APIキーの確認
    if not api_key:
        api_key = os.getenv("MP_API_KEY")
    
    if not api_key:
        print("\n❌ APIキーが設定されていません。")
        print("\n手順:")
        print("1. Materials Projectにアカウント作成: https://materialsproject.org")
        print("2. Dashboard > API Keys からAPIキーを取得")
        print("3. 環境変数に設定: export MP_API_KEY='your_api_key_here'")
        print("   または、--api-key オプションで指定")
        return pd.DataFrame()
    
    try:
        from mp_api.client import MPRester
        
        print(f"\n✅ APIキーが見つかりました")
        print(f"📊 最大取得数: {max_materials}材料")
        print("Materials Projectからデータを取得中...\n")
        
        # HEAに関連する元素
        hea_elements = ["Ti", "Zr", "Hf", "Nb", "Ta", "V", "Cr", "Mo", "W", 
                       "Fe", "Co", "Ni", "Cu", "Al", "Mn", "Sn", "Si"]
        
        all_data = []
        total_found = 0
        
        with MPRester(api_key) as mpr:
            # 方法1: 弾性率データがある材料を直接検索（より広範囲）
            print("📥 方法1: 弾性率データがある材料を検索中...")
            try:
                # より広範囲の検索: 金属材料で弾性率データがあるもの
                summaries = mpr.materials.summary.search(
                    has_props=["elasticity"],
                    is_metal=True,
                    fields=["material_id", "formula_pretty", "bulk_modulus", 
                           "shear_modulus", "density", "formation_energy_per_atom",
                           "elements", "nsites"]
                )
                
                print(f"   ✅ {len(summaries)}個の材料が見つかりました")
                
                for i, s in enumerate(summaries):
                    if len(all_data) >= max_materials:
                        break
                    
                    try:
                        # より柔軟なフィルタリング: HEA関連元素を含む、または多元素系（3元素以上）
                        is_hea_related = False
                        element_strs = []
                        
                        if s.elements:
                            # Elementオブジェクトを文字列に変換
                            try:
                                element_strs = [str(e) if not isinstance(e, str) else e for e in s.elements]
                            except:
                                try:
                                    element_strs = [str(e) for e in s.elements]
                                except:
                                    continue
                            
                            hea_elements_in_material = set(element_strs) & set(hea_elements)
                            # HEA関連元素が2つ以上含まれる、または3元素以上の多元素系
                            if len(hea_elements_in_material) >= 2 or len(element_strs) >= 3:
                                is_hea_related = True
                        
                        if is_hea_related:
                            # Young's modulusを計算
                            E = calculate_youngs_modulus(s.bulk_modulus, s.shear_modulus)
                            
                            if E and E > 0 and E < 1000:  # 異常値除去
                                all_data.append({
                                    'material_id': s.material_id,
                                    'alloy_name': s.formula_pretty,
                                    'elastic_modulus': E,
                                    'bulk_modulus': s.bulk_modulus.get('vrh', None) if isinstance(s.bulk_modulus, dict) else s.bulk_modulus,
                                    'shear_modulus': s.shear_modulus.get('vrh', None) if isinstance(s.shear_modulus, dict) else s.shear_modulus,
                                    'density': s.density,
                                    'formation_energy': s.formation_energy_per_atom,
                                    'elements': ','.join(element_strs) if element_strs else '',
                                    'num_elements': len(element_strs) if element_strs else 0,
                                    'nsites': s.nsites,
                                    'source': 'Materials Project'
                                })
                    except Exception as e:
                        # エラーを無視して続行（デバッグ用に最初の数個だけ表示）
                        if i < 5:
                            print(f"   ⚠️  エラー (材料 {i}): {type(e).__name__}")
                        continue
                    
                    if (i + 1) % 100 == 0:
                        print(f"   処理中: {i+1}/{len(summaries)} (取得済み: {len(all_data)})")
                    
                    # レート制限対策
                    if (i + 1) % 100 == 0:
                        time.sleep(0.5)
                
                print(f"   ✅ HEA関連材料: {len(all_data)}個")
                
            except Exception as e:
                print(f"   ⚠️  エラー: {e}")
            
            # 方法2: 特定の元素組み合わせで検索（補完）
            if len(all_data) < max_materials:
                print(f"\n📥 方法2: 特定の元素組み合わせで検索中...")
                
                # 主要なHEA元素の組み合わせ
                element_combinations = [
                    ["Ti", "Zr", "Hf", "Nb", "Ta"],
                    ["Cr", "Mo", "W", "V", "Nb"],
                    ["Fe", "Co", "Ni", "Cr", "Mn"],
                    ["Al", "Ti", "Cr", "Fe", "Ni"],
                    ["Ti", "V", "Cr", "Nb", "Mo"],
                ]
                
                for combo in element_combinations:
                    if len(all_data) >= max_materials:
                        break
                    
                    try:
                        summaries = mpr.materials.summary.search(
                            elements=combo,
                            has_props=["elasticity"],
                            is_metal=True,
                            fields=["material_id", "formula_pretty", "bulk_modulus", 
                                   "shear_modulus", "density", "formation_energy_per_atom",
                                   "elements", "nsites"]
                        )
                        
                        combo_count = 0
                        for s in summaries:
                            if len(all_data) >= max_materials:
                                break
                            
                            # 重複チェック
                            if any(d['material_id'] == s.material_id for d in all_data):
                                continue
                            
                            E = calculate_youngs_modulus(s.bulk_modulus, s.shear_modulus)
                            
                            if E and E > 0 and E < 1000:  # 異常値除去
                                all_data.append({
                                    'material_id': s.material_id,
                                    'alloy_name': s.formula_pretty,
                                    'elastic_modulus': E,
                                    'bulk_modulus': s.bulk_modulus.get('vrh', None) if isinstance(s.bulk_modulus, dict) else s.bulk_modulus,
                                    'shear_modulus': s.shear_modulus.get('vrh', None) if isinstance(s.shear_modulus, dict) else s.shear_modulus,
                                    'density': s.density,
                                    'formation_energy': s.formation_energy_per_atom,
                                    'elements': ','.join(s.elements) if s.elements else '',
                                    'num_elements': len(s.elements) if s.elements else 0,
                                    'nsites': s.nsites,
                                    'source': 'Materials Project'
                                })
                                combo_count += 1
                        
                        print(f"   ✅ {combo}: {len(summaries)}個の材料を検索、{combo_count}個を追加")
                        time.sleep(0.5)  # レート制限対策
                        
                    except Exception as e:
                        print(f"   ⚠️  {combo}の検索エラー: {e}")
                        continue
        
        if all_data:
            df = pd.DataFrame(all_data)
            
            # 重複除去
            initial_count = len(df)
            df = df.drop_duplicates(subset=['material_id'], keep='first')
            removed = initial_count - len(df)
            
            print(f"\n📊 データ統計:")
            print(f"   取得数: {len(df)}サンプル")
            if removed > 0:
                print(f"   重複除去: {removed}サンプル")
            
            if 'elastic_modulus' in df.columns:
                print(f"   弾性率範囲: {df['elastic_modulus'].min():.2f} - {df['elastic_modulus'].max():.2f} GPa")
                print(f"   平均: {df['elastic_modulus'].mean():.2f} GPa")
                print(f"   中央値: {df['elastic_modulus'].median():.2f} GPa")
                
                target_range = df[(df['elastic_modulus'] >= 30) & (df['elastic_modulus'] <= 90)]
                print(f"   ⭐ 目標範囲（30-90 GPa）内: {len(target_range)}サンプル")
            
            # 保存
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = COLLECTED_DATA_DIR / f"materials_project_{timestamp}.csv"
            df.to_csv(output_file, index=False)
            print(f"\n✅ データを保存しました: {output_file}")
            
            return df
        else:
            print("\n⚠️  データを取得できませんでした")
            return pd.DataFrame()
            
    except ImportError:
        print("\n❌ mp-apiパッケージがインストールされていません。")
        print("   インストール: pip install mp-api")
        return pd.DataFrame()
    except Exception as e:
        print(f"\n❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Materials ProjectからHEA弾性率データを取得')
    parser.add_argument('--api-key', type=str, help='Materials Project APIキー')
    parser.add_argument('--max-materials', type=int, default=5000, help='最大取得材料数（デフォルト: 5000）')
    
    args = parser.parse_args()
    
    download_materials_project_data(api_key=args.api_key, max_materials=args.max_materials)
