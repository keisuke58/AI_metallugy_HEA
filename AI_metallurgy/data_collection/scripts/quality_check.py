#!/usr/bin/env python3
"""
完璧かどうかを調査する包括的な品質チェックスクリプト
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import glob
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).parent.parent
PROCESSED_DATA_DIR = BASE_DIR / "processed_data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures"
SCRIPTS_DIR = BASE_DIR / "scripts"

def check_data_quality():
    """データ品質のチェック"""
    print("=" * 80)
    print("1. データ品質チェック")
    print("=" * 80)
    
    issues = []
    warnings_list = []
    
    # データファイルの存在確認
    data_files = [
        PROCESSED_DATA_DIR / "data_preprocessed.csv",
        PROCESSED_DATA_DIR / "data_with_features.csv",
        PROCESSED_DATA_DIR / "integrated_data.csv"
    ]
    
    existing_files = [f for f in data_files if f.exists()]
    if not existing_files:
        issues.append("❌ データファイルが見つかりません")
        return issues, warnings_list
    
    print(f"✅ データファイル: {existing_files[0].name}")
    
    # データを読み込む
    df = pd.read_csv(existing_files[0])
    
    # 実測データのみを抽出
    measured_sources = ['DOE/OSTI', 'Gorsse Dataset', 'Latest Research']
    df_measured = df[df['source'].isin(measured_sources)].copy()
    df_measured = df_measured.dropna(subset=['elastic_modulus'])
    
    print(f"📊 実測データ: {len(df_measured)}行")
    
    # データ品質チェック
    if len(df_measured) < 50:
        warnings_list.append(f"⚠️  データ数が少ない: {len(df_measured)}行（推奨: 100行以上）")
    elif len(df_measured) < 100:
        warnings_list.append(f"⚠️  データ数がやや少ない: {len(df_measured)}行（推奨: 100行以上）")
    else:
        print(f"✅ データ数: {len(df_measured)}行（十分）")
    
    # 弾性率の範囲チェック
    y = df_measured['elastic_modulus'].values
    if y.min() < 0:
        issues.append(f"❌ 負の弾性率が存在: {y.min()}")
    if y.max() > 1000:
        warnings_list.append(f"⚠️  異常に大きい弾性率: {y.max()} GPa")
    
    print(f"📊 弾性率範囲: {y.min():.2f} - {y.max():.2f} GPa")
    print(f"📊 平均: {y.mean():.2f} GPa, 標準偏差: {y.std():.2f} GPa")
    
    # 欠損値チェック
    exclude_cols = ['alloy_name', 'elastic_modulus', 'source', 'phases']
    feature_cols = [col for col in df_measured.columns 
                    if col not in exclude_cols and df_measured[col].dtype in [np.float64, np.int64]]
    
    missing_ratios = {}
    for col in feature_cols:
        missing_ratio = df_measured[col].isna().sum() / len(df_measured)
        if missing_ratio > 0.5:
            warnings_list.append(f"⚠️  特徴量 '{col}' の欠損率が高い: {missing_ratio*100:.1f}%")
        missing_ratios[col] = missing_ratio
    
    high_missing = sum(1 for r in missing_ratios.values() if r > 0.3)
    if high_missing > 0:
        print(f"⚠️  欠損率30%以上の特徴量: {high_missing}個")
    else:
        print(f"✅ 欠損値: 問題なし")
    
    return issues, warnings_list

def check_model_quality():
    """モデル品質のチェック"""
    print("\n" + "=" * 80)
    print("2. モデル品質チェック")
    print("=" * 80)
    
    issues = []
    warnings_list = []
    
    # モデルファイルの存在確認
    model_dirs = sorted(glob.glob(str(MODELS_DIR / "*measured*")))
    if not model_dirs:
        issues.append("❌ モデルファイルが見つかりません")
        return issues, warnings_list
    
    print(f"✅ モデルディレクトリ: {len(model_dirs)}個")
    
    # 最新のモデルを読み込む
    latest_model_dir = Path(model_dirs[-1])
    model_file = latest_model_dir / "best_model.pkl"
    
    if not model_file.exists():
        issues.append(f"❌ モデルファイルが見つかりません: {model_file}")
        return issues, warnings_list
    
    print(f"✅ 最新モデル: {latest_model_dir.name}")
    
    # モデルを読み込む
    try:
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
        print(f"✅ モデル読み込み成功: {type(model).__name__}")
    except Exception as e:
        issues.append(f"❌ モデル読み込みエラー: {e}")
        return issues, warnings_list
    
    # データを読み込んで評価
    data_file = PROCESSED_DATA_DIR / "data_preprocessed.csv"
    if not data_file.exists():
        data_file = PROCESSED_DATA_DIR / "data_with_features.csv"
    
    df = pd.read_csv(data_file)
    measured_sources = ['DOE/OSTI', 'Gorsse Dataset', 'Latest Research']
    df_measured = df[df['source'].isin(measured_sources)].copy()
    df_measured = df_measured.dropna(subset=['elastic_modulus'])
    
    exclude_cols = ['alloy_name', 'elastic_modulus', 'source', 'phases']
    feature_cols = [col for col in df_measured.columns 
                    if col not in exclude_cols and df_measured[col].dtype in [np.float64, np.int64]
                    and df_measured[col].notna().sum() / len(df_measured) >= 0.7]
    
    X = df_measured[feature_cols].copy()
    y = df_measured['elastic_modulus'].values
    
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=feature_cols)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y, test_size=0.2, random_state=42
    )
    
    # 予測
    try:
        y_pred_test = model.predict(X_test)
        r2 = r2_score(y_test, y_pred_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        mae = mean_absolute_error(y_test, y_pred_test)
        
        print(f"📊 Test R²: {r2:.4f}")
        print(f"📊 Test RMSE: {rmse:.2f} GPa")
        print(f"📊 Test MAE: {mae:.2f} GPa")
        
        # 性能評価
        if r2 < 0:
            issues.append(f"❌ R²が負: {r2:.4f}（モデルが不適切）")
        elif r2 < 0.3:
            warnings_list.append(f"⚠️  R²が低い: {r2:.4f}（推奨: >0.5）")
        elif r2 < 0.5:
            warnings_list.append(f"⚠️  R²が中程度: {r2:.4f}（推奨: >0.6）")
        elif r2 >= 0.7:
            print(f"✅ 優秀な性能: R² = {r2:.4f}")
        else:
            print(f"✅ 良好な性能: R² = {r2:.4f}")
        
        # 過学習チェック
        y_pred_train = model.predict(X_train)
        r2_train = r2_score(y_train, y_pred_train)
        
        if r2_train - r2 > 0.3:
            warnings_list.append(f"⚠️  過学習の可能性: Train R²={r2_train:.4f}, Test R²={r2:.4f} (差={r2_train-r2:.4f})")
        else:
            print(f"✅ 過学習チェック: 問題なし (Train R²={r2_train:.4f}, Test R²={r2:.4f})")
        
    except Exception as e:
        issues.append(f"❌ 予測エラー: {e}")
    
    return issues, warnings_list

def check_results_quality():
    """結果ファイルの品質チェック"""
    print("\n" + "=" * 80)
    print("3. 結果ファイル品質チェック")
    print("=" * 80)
    
    issues = []
    warnings_list = []
    
    # 結果ファイルの存在確認
    result_files = sorted(glob.glob(str(RESULTS_DIR / "*measured*.json")))
    if not result_files:
        issues.append("❌ 結果ファイルが見つかりません")
        return issues, warnings_list
    
    print(f"✅ 結果ファイル: {len(result_files)}個")
    
    # 最新の結果を読み込む
    latest_result_file = result_files[-1]
    print(f"📊 最新結果: {Path(latest_result_file).name}")
    
    try:
        with open(latest_result_file, 'r') as f:
            results = json.load(f)
        
        # 結果の構造チェック
        if 'results' in results:
            models = results['results']
            print(f"✅ モデル数: {len(models)}個")
            
            best_r2 = -np.inf
            best_model = None
            
            for name, result in models.items():
                r2 = result.get('test_r2', -np.inf)
                if r2 > best_r2:
                    best_r2 = r2
                    best_model = name
            
            if best_model:
                print(f"⭐ 最良モデル: {best_model} (R² = {best_r2:.4f})")
                
                if best_r2 >= 0.7:
                    print(f"✅ 目標達成: R² >= 0.70")
                elif best_r2 >= 0.65:
                    print(f"✅ 良好: R² >= 0.65")
                elif best_r2 < 0.5:
                    warnings_list.append(f"⚠️  最良R²が低い: {best_r2:.4f}")
        elif 'optimized_r2' in results:
            r2 = results.get('optimized_r2', 0)
            print(f"📊 最適化R²: {r2:.4f}")
            
            if r2 >= 0.7:
                print(f"✅ 目標達成: R² >= 0.70")
            elif r2 < 0.5:
                warnings_list.append(f"⚠️  最適化R²が低い: {r2:.4f}")
        
    except Exception as e:
        issues.append(f"❌ 結果ファイル読み込みエラー: {e}")
    
    return issues, warnings_list

def check_visualizations():
    """可視化ファイルのチェック"""
    print("\n" + "=" * 80)
    print("4. 可視化ファイルチェック")
    print("=" * 80)
    
    issues = []
    warnings_list = []
    
    # 可視化ファイルの存在確認
    figure_files = sorted(glob.glob(str(FIGURES_DIR / "*.png")))
    
    if not figure_files:
        issues.append("❌ 可視化ファイルが見つかりません")
        return issues, warnings_list
    
    print(f"✅ 可視化ファイル: {len(figure_files)}個")
    
    # 必須の可視化ファイル
    required_plots = [
        'predicted_vs_actual',
        'residuals_analysis',
        'feature_importance',
        'model_comparison',
        'error_distribution',
        'data_distribution'
    ]
    
    existing_plots = [Path(f).stem for f in figure_files]
    missing_plots = []
    
    for required in required_plots:
        found = any(required in plot for plot in existing_plots)
        if not found:
            missing_plots.append(required)
    
    if missing_plots:
        warnings_list.append(f"⚠️  推奨プロットが不足: {', '.join(missing_plots)}")
    else:
        print(f"✅ 必須プロット: すべて存在")
    
    # ファイルサイズチェック
    large_files = []
    for f in figure_files:
        size_mb = Path(f).stat().st_size / (1024 * 1024)
        if size_mb > 5:
            large_files.append(f"{Path(f).name}: {size_mb:.1f} MB")
    
    if large_files:
        warnings_list.append(f"⚠️  大きなファイル: {len(large_files)}個")
        for f in large_files[:3]:
            print(f"   {f}")
    
    return issues, warnings_list

def check_code_quality():
    """コード品質のチェック"""
    print("\n" + "=" * 80)
    print("5. コード品質チェック")
    print("=" * 80)
    
    issues = []
    warnings_list = []
    
    # 主要スクリプトの存在確認
    required_scripts = [
        'comprehensive_preprocessing_training.py',
        'comprehensive_visualization.py',
        'final_optimization.py'
    ]
    
    existing_scripts = []
    for script in required_scripts:
        script_path = SCRIPTS_DIR / script
        if script_path.exists():
            existing_scripts.append(script)
            print(f"✅ {script}")
        else:
            warnings_list.append(f"⚠️  スクリプトが見つかりません: {script}")
    
    return issues, warnings_list

def check_consistency():
    """一貫性チェック"""
    print("\n" + "=" * 80)
    print("6. 一貫性チェック")
    print("=" * 80)
    
    issues = []
    warnings_list = []
    
    # 結果ファイル間の一貫性
    result_files = sorted(glob.glob(str(RESULTS_DIR / "*measured*.json")))
    
    if len(result_files) >= 2:
        r2_scores = []
        for result_file in result_files:
            try:
                with open(result_file, 'r') as f:
                    result = json.load(f)
                
                if 'results' in result:
                    for name, model_result in result['results'].items():
                        r2 = model_result.get('test_r2', 0)
                        if r2 > 0:
                            r2_scores.append(r2)
                elif 'optimized_r2' in result:
                    r2 = result.get('optimized_r2', 0)
                    if r2 > 0:
                        r2_scores.append(r2)
            except:
                pass
        
        if r2_scores:
            print(f"📊 R²スコア範囲: {min(r2_scores):.4f} - {max(r2_scores):.4f}")
            print(f"📊 平均: {np.mean(r2_scores):.4f}, 標準偏差: {np.std(r2_scores):.4f}")
            
            if max(r2_scores) - min(r2_scores) > 0.5:
                warnings_list.append("⚠️  R²スコアのばらつきが大きい（最適化の効果が不明確）")
            else:
                print(f"✅ 一貫性: 問題なし")
    
    return issues, warnings_list

def generate_quality_report(issues, warnings_list):
    """品質レポートを生成"""
    print("\n" + "=" * 80)
    print("品質チェック結果サマリー")
    print("=" * 80)
    
    total_issues = len(issues)
    total_warnings = len(warnings_list)
    
    print(f"\n❌ 重大な問題: {total_issues}個")
    if issues:
        for issue in issues:
            print(f"   {issue}")
    else:
        print("   ✅ 重大な問題はありません")
    
    print(f"\n⚠️  警告: {total_warnings}個")
    if warnings_list:
        for warning in warnings_list:
            print(f"   {warning}")
    else:
        print("   ✅ 警告はありません")
    
    # 総合評価
    print(f"\n{'='*80}")
    print("総合評価")
    print(f"{'='*80}")
    
    if total_issues == 0 and total_warnings == 0:
        print("🎉🎉🎉 完璧です！すべてのチェックをパスしました！ 🎉🎉🎉")
        score = 100
    elif total_issues == 0 and total_warnings <= 3:
        print("✨ 優秀です！軽微な警告のみです。")
        score = 90 - total_warnings * 5
    elif total_issues == 0:
        print("✅ 良好です。いくつかの改善点があります。")
        score = 80 - total_warnings * 3
    elif total_issues <= 2:
        print("⚠️  注意が必要です。いくつかの問題があります。")
        score = 60 - total_issues * 10 - total_warnings * 2
    else:
        print("❌ 改善が必要です。複数の問題があります。")
        score = max(0, 40 - total_issues * 10 - total_warnings * 2)
    
    print(f"\nスコア: {score}/100")
    
    # 改善提案
    if total_issues > 0 or total_warnings > 0:
        print(f"\n改善提案:")
        if total_issues > 0:
            print("   1. 重大な問題を優先的に解決してください")
        if total_warnings > 0:
            print("   2. 警告事項を確認し、可能な限り改善してください")
        print("   3. データの品質向上を検討してください")
        print("   4. モデルのハイパーパラメータをさらに最適化してください")
    
    return score, issues, warnings_list

def main():
    """メイン関数"""
    print("=" * 80)
    print("完璧かどうかを調査する包括的な品質チェック")
    print("=" * 80)
    
    all_issues = []
    all_warnings = []
    
    # 各チェックを実行
    issues, warnings = check_data_quality()
    all_issues.extend(issues)
    all_warnings.extend(warnings)
    
    issues, warnings = check_model_quality()
    all_issues.extend(issues)
    all_warnings.extend(warnings)
    
    issues, warnings = check_results_quality()
    all_issues.extend(issues)
    all_warnings.extend(warnings)
    
    issues, warnings = check_visualizations()
    all_issues.extend(issues)
    all_warnings.extend(warnings)
    
    issues, warnings = check_code_quality()
    all_issues.extend(issues)
    all_warnings.extend(warnings)
    
    issues, warnings = check_consistency()
    all_issues.extend(issues)
    all_warnings.extend(warnings)
    
    # レポート生成
    score, issues, warnings = generate_quality_report(all_issues, all_warnings)
    
    # 結果を保存
    report = {
        'score': score,
        'total_issues': len(all_issues),
        'total_warnings': len(all_warnings),
        'issues': all_issues,
        'warnings': all_warnings,
        'status': 'perfect' if score >= 95 else 'excellent' if score >= 85 else 'good' if score >= 70 else 'needs_improvement'
    }
    
    report_file = RESULTS_DIR / "quality_check_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 品質チェックレポートを保存しました: {report_file}")
    
    print("\n" + "=" * 80)
    print("✅ 品質チェックが完了しました！")
    print("=" * 80)
    
    return score

if __name__ == "__main__":
    main()
