#!/usr/bin/env python3
"""
データセットの品質と整合性を詳細に確認するスクリプト
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).parent.parent
FINAL_DATA_DIR = BASE_DIR / "final_data"

def validate_dataset(file_path):
    """データセットの詳細な検証"""
    print("=" * 60)
    print("データセット品質検証")
    print("=" * 60)
    
    if not file_path.exists():
        print(f"❌ ファイルが見つかりません: {file_path}")
        return False
    
    try:
        df = pd.read_csv(file_path)
        print(f"✅ ファイル読み込み成功: {file_path}")
        print(f"📊 総サンプル数: {len(df):,}サンプル")
        print(f"📊 総カラム数: {len(df.columns)}カラム")
        
        # 1. 必須カラムの確認
        print("\n" + "=" * 60)
        print("1. 必須カラムの確認")
        print("=" * 60)
        
        required_columns = ['alloy_name', 'elastic_modulus', 'source']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"❌ 必須カラムが不足しています: {missing_columns}")
            return False
        else:
            print("✅ 必須カラムはすべて存在します")
            for col in required_columns:
                print(f"   - {col}: {df[col].dtype}")
        
        # 2. 欠損値の確認
        print("\n" + "=" * 60)
        print("2. 欠損値の確認")
        print("=" * 60)
        
        missing_stats = df[required_columns].isnull().sum()
        has_missing = missing_stats.sum() > 0
        
        if has_missing:
            print("⚠️  欠損値が見つかりました:")
            for col, count in missing_stats.items():
                if count > 0:
                    percentage = count / len(df) * 100
                    print(f"   - {col}: {count:,} ({percentage:.2f}%)")
        else:
            print("✅ 必須カラムに欠損値はありません")
        
        # 3. 弾性率データの検証
        print("\n" + "=" * 60)
        print("3. 弾性率データの検証")
        print("=" * 60)
        
        if 'elastic_modulus' in df.columns:
            # データ型確認
            print(f"📊 データ型: {df['elastic_modulus'].dtype}")
            
            # 数値変換
            df['elastic_modulus'] = pd.to_numeric(df['elastic_modulus'], errors='coerce')
            
            # 統計情報
            valid_data = df['elastic_modulus'].dropna()
            print(f"📊 有効データ数: {len(valid_data):,}サンプル")
            print(f"📊 欠損データ数: {df['elastic_modulus'].isnull().sum():,}サンプル")
            
            if len(valid_data) > 0:
                print(f"\n📊 統計情報:")
                print(f"   - 最小値: {valid_data.min():.2f} GPa")
                print(f"   - 最大値: {valid_data.max():.2f} GPa")
                print(f"   - 平均: {valid_data.mean():.2f} GPa")
                print(f"   - 中央値: {valid_data.median():.2f} GPa")
                print(f"   - 標準偏差: {valid_data.std():.2f} GPa")
                print(f"   - 25%分位: {valid_data.quantile(0.25):.2f} GPa")
                print(f"   - 75%分位: {valid_data.quantile(0.75):.2f} GPa")
                
                # 異常値の検出
                Q1 = valid_data.quantile(0.25)
                Q3 = valid_data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = valid_data[(valid_data < lower_bound) | (valid_data > upper_bound)]
                print(f"\n📊 異常値（IQR法）: {len(outliers):,}サンプル ({len(outliers)/len(valid_data)*100:.2f}%)")
                
                # 物理的に妥当な範囲の確認（一般的な金属材料: 10-500 GPa）
                reasonable_range = valid_data[(valid_data >= 10) & (valid_data <= 500)]
                print(f"📊 妥当な範囲（10-500 GPa）内: {len(reasonable_range):,}サンプル ({len(reasonable_range)/len(valid_data)*100:.2f}%)")
                
                # 負の値やゼロの確認
                invalid_values = valid_data[valid_data <= 0]
                if len(invalid_values) > 0:
                    print(f"⚠️  無効な値（≤0）: {len(invalid_values):,}サンプル")
                else:
                    print("✅ すべての値が正の値です")
        
        # 4. 重複データの確認
        print("\n" + "=" * 60)
        print("4. 重複データの確認")
        print("=" * 60)
        
        # 合金名での重複
        duplicate_names = df[df.duplicated(subset=['alloy_name'], keep=False)]
        if len(duplicate_names) > 0:
            print(f"⚠️  合金名の重複: {len(duplicate_names):,}サンプル")
            print(f"   重複している合金名の例（最初の10個）:")
            duplicate_names_sorted = duplicate_names.sort_values('alloy_name')
            for name in duplicate_names_sorted['alloy_name'].unique()[:10]:
                count = len(duplicate_names_sorted[duplicate_names_sorted['alloy_name'] == name])
                print(f"   - {name}: {count}回")
        else:
            print("✅ 合金名の重複はありません")
        
        # 完全重複
        complete_duplicates = df[df.duplicated(keep=False)]
        if len(complete_duplicates) > 0:
            print(f"⚠️  完全重複: {len(complete_duplicates):,}サンプル")
        else:
            print("✅ 完全重複はありません")
        
        # 5. データソースの確認
        print("\n" + "=" * 60)
        print("5. データソースの確認")
        print("=" * 60)
        
        if 'source' in df.columns:
            source_counts = df['source'].value_counts()
            print(f"📊 データソース数: {len(source_counts)}個")
            print(f"\n📊 データソース別内訳:")
            for source, count in source_counts.items():
                percentage = count / len(df) * 100
                print(f"   - {source}: {count:,}サンプル ({percentage:.1f}%)")
        
        # 6. 合金名の確認
        print("\n" + "=" * 60)
        print("6. 合金名の確認")
        print("=" * 60)
        
        if 'alloy_name' in df.columns:
            # 空の合金名
            empty_names = df[df['alloy_name'].isnull() | (df['alloy_name'] == '')]
            if len(empty_names) > 0:
                print(f"⚠️  空の合金名: {len(empty_names):,}サンプル")
            else:
                print("✅ すべてのサンプルに合金名があります")
            
            # ユニークな合金数
            unique_alloys = df['alloy_name'].nunique()
            print(f"📊 ユニークな合金数: {unique_alloys:,}個")
            print(f"📊 重複率: {(len(df) - unique_alloys) / len(df) * 100:.2f}%")
        
        # 7. データ分布の確認
        print("\n" + "=" * 60)
        print("7. データ分布の確認")
        print("=" * 60)
        
        if 'elastic_modulus' in df.columns:
            valid_data = df['elastic_modulus'].dropna()
            
            # ヒストグラム用のビン
            bins = [0, 50, 100, 150, 200, 300, 500, float('inf')]
            bin_labels = ['0-50', '50-100', '100-150', '150-200', '200-300', '300-500', '500+']
            
            print("📊 弾性率の分布:")
            for i, (bin_min, bin_max) in enumerate(zip(bins[:-1], bins[1:])):
                count = len(valid_data[(valid_data >= bin_min) & (valid_data < bin_max)])
                percentage = count / len(valid_data) * 100
                print(f"   - {bin_labels[i]} GPa: {count:,}サンプル ({percentage:.1f}%)")
        
        # 8. データ品質スコア
        print("\n" + "=" * 60)
        print("8. データ品質スコア")
        print("=" * 60)
        
        quality_score = 100
        
        # 欠損値ペナルティ
        if has_missing:
            missing_penalty = missing_stats.sum() / len(df) * 100
            quality_score -= min(missing_penalty, 20)
            print(f"   欠損値ペナルティ: -{min(missing_penalty, 20):.1f}点")
        
        # 重複ペナルティ
        if len(duplicate_names) > 0:
            duplicate_penalty = len(duplicate_names) / len(df) * 100
            quality_score -= min(duplicate_penalty, 10)
            print(f"   重複ペナルティ: -{min(duplicate_penalty, 10):.1f}点")
        
        # 異常値ペナルティ
        if len(outliers) > 0:
            outlier_penalty = len(outliers) / len(valid_data) * 100
            quality_score -= min(outlier_penalty * 0.1, 5)
            print(f"   異常値ペナルティ: -{min(outlier_penalty * 0.1, 5):.1f}点")
        
        quality_score = max(quality_score, 0)
        
        print(f"\n📊 総合品質スコア: {quality_score:.1f}/100")
        
        if quality_score >= 90:
            print("✅ データ品質: 優秀")
        elif quality_score >= 80:
            print("✅ データ品質: 良好")
        elif quality_score >= 70:
            print("⚠️  データ品質: 普通（改善の余地あり）")
        else:
            print("❌ データ品質: 要改善")
        
        # 9. 推奨事項
        print("\n" + "=" * 60)
        print("9. 推奨事項")
        print("=" * 60)
        
        recommendations = []
        
        if has_missing:
            recommendations.append("欠損値を処理する（削除または補完）")
        
        if len(duplicate_names) > 0:
            recommendations.append("重複データを確認し、必要に応じて統合または削除")
        
        if len(outliers) > 0:
            recommendations.append("異常値を確認し、物理的に妥当か検証")
        
        if len(valid_data[(valid_data < 10) | (valid_data > 500)]) > 0:
            recommendations.append("極端な値（<10 GPa または >500 GPa）を確認")
        
        if recommendations:
            print("以下の改善を推奨します:")
            for i, rec in enumerate(recommendations, 1):
                print(f"   {i}. {rec}")
        else:
            print("✅ 特に問題は見つかりませんでした")
        
        return quality_score >= 70
        
    except Exception as e:
        print(f"❌ エラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """メイン関数"""
    # 最新の統合データセットを探す
    unified_files = list(FINAL_DATA_DIR.glob("unified_dataset*.csv"))
    
    if not unified_files:
        print("❌ 統合データセットが見つかりません")
        return
    
    # 最新のファイルを使用
    latest_file = max(unified_files, key=lambda p: p.stat().st_mtime)
    
    print(f"📁 検証対象ファイル: {latest_file.name}")
    print(f"📁 パス: {latest_file}")
    print()
    
    is_valid = validate_dataset(latest_file)
    
    print("\n" + "=" * 60)
    if is_valid:
        print("✅ データセット検証完了: 問題なし")
    else:
        print("⚠️  データセット検証完了: 改善が必要")
    print("=" * 60)

if __name__ == "__main__":
    main()
