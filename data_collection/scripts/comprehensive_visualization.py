#!/usr/bin/env python3
"""
実測データのみの最適化結果を包括的に可視化
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import glob
import pickle
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import learning_curve
import warnings
warnings.filterwarnings('ignore')

# 日本語フォント設定
plt.rcParams['font.family'] = 'DejaVu Sans'
sns.set_style("whitegrid")
sns.set_palette("husl")

BASE_DIR = Path(__file__).parent.parent
PROCESSED_DATA_DIR = BASE_DIR / "processed_data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def load_latest_model_and_data():
    """最新のモデルとデータを読み込む"""
    # 最新の結果ファイルを探す
    result_files = sorted(glob.glob(str(RESULTS_DIR / "final_measured_*.json")))
    if not result_files:
        result_files = sorted(glob.glob(str(RESULTS_DIR / "*measured*.json")))
    
    if not result_files:
        print("❌ 結果ファイルが見つかりません")
        return None, None, None, None
    
    latest_result_file = result_files[-1]
    print(f"📊 最新の結果ファイル: {latest_result_file}")
    
    with open(latest_result_file, 'r') as f:
        results = json.load(f)
    
    # データを読み込む
    data_file = PROCESSED_DATA_DIR / "data_preprocessed.csv"
    if not data_file.exists():
        data_file = PROCESSED_DATA_DIR / "data_with_features.csv"
    
    df = pd.read_csv(data_file)
    measured_sources = ['DOE/OSTI', 'Gorsse Dataset', 'Latest Research']
    df_measured = df[df['source'].isin(measured_sources)].copy()
    df_measured = df_measured.dropna(subset=['elastic_modulus'])
    
    # 特徴量を準備
    exclude_cols = ['alloy_name', 'elastic_modulus', 'source', 'phases']
    feature_cols = [col for col in df_measured.columns 
                    if col not in exclude_cols and df_measured[col].dtype in [np.float64, np.int64]
                    and df_measured[col].notna().sum() / len(df_measured) >= 0.7]
    
    X = df_measured[feature_cols].copy()
    y = df_measured['elastic_modulus'].values
    
    # 欠損値処理
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=feature_cols)
    
    # 最新のモデルを読み込む
    model_dirs = sorted(glob.glob(str(MODELS_DIR / "final_measured_*")))
    if not model_dirs:
        model_dirs = sorted(glob.glob(str(MODELS_DIR / "*measured*")))
    
    best_model = None
    if model_dirs:
        latest_model_dir = Path(model_dirs[-1])
        model_file = latest_model_dir / "best_model.pkl"
        if model_file.exists():
            with open(model_file, 'rb') as f:
                best_model = pickle.load(f)
            print(f"✅ モデルを読み込みました: {latest_model_dir}")
    
    return X_imputed, y, best_model, results

def plot_predicted_vs_actual(X, y, model, title_suffix=""):
    """予測値 vs 実測値のプロット"""
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    if model is None:
        print("⚠️  モデルが読み込めませんでした。スキップします。")
        return
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 訓練データ
    axes[0].scatter(y_train, y_pred_train, alpha=0.6, s=50)
    axes[0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 
                 'r--', lw=2, label='Perfect Prediction')
    r2_train = r2_score(y_train, y_pred_train)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    axes[0].set_xlabel('Actual Elastic Modulus (GPa)', fontsize=12)
    axes[0].set_ylabel('Predicted Elastic Modulus (GPa)', fontsize=12)
    axes[0].set_title(f'Training Data{title_suffix}\nR² = {r2_train:.4f}, RMSE = {rmse_train:.2f} GPa', 
                      fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # テストデータ
    axes[1].scatter(y_test, y_pred_test, alpha=0.6, s=50, color='orange')
    axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                 'r--', lw=2, label='Perfect Prediction')
    r2_test = r2_score(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    axes[1].set_xlabel('Actual Elastic Modulus (GPa)', fontsize=12)
    axes[1].set_ylabel('Predicted Elastic Modulus (GPa)', fontsize=12)
    axes[1].set_title(f'Test Data{title_suffix}\nR² = {r2_test:.4f}, RMSE = {rmse_test:.2f} GPa', 
                      fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f'predicted_vs_actual{title_suffix.lower().replace(" ", "_")}.png', 
                dpi=300, bbox_inches='tight')
    print(f"✅ 予測値 vs 実測値プロットを保存: predicted_vs_actual{title_suffix.lower().replace(' ', '_')}.png")
    plt.close()

def plot_residuals(X, y, model, title_suffix=""):
    """残差プロット"""
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    if model is None:
        return
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    residuals_train = y_train - y_pred_train
    residuals_test = y_test - y_pred_test
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 訓練データの残差 vs 予測値
    axes[0, 0].scatter(y_pred_train, residuals_train, alpha=0.6, s=50)
    axes[0, 0].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0, 0].set_xlabel('Predicted Elastic Modulus (GPa)', fontsize=12)
    axes[0, 0].set_ylabel('Residuals (GPa)', fontsize=12)
    axes[0, 0].set_title('Training Data: Residuals vs Predicted', fontsize=14, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # テストデータの残差 vs 予測値
    axes[0, 1].scatter(y_pred_test, residuals_test, alpha=0.6, s=50, color='orange')
    axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0, 1].set_xlabel('Predicted Elastic Modulus (GPa)', fontsize=12)
    axes[0, 1].set_ylabel('Residuals (GPa)', fontsize=12)
    axes[0, 1].set_title('Test Data: Residuals vs Predicted', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 残差の分布（訓練データ）
    axes[1, 0].hist(residuals_train, bins=20, alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(x=0, color='r', linestyle='--', lw=2)
    axes[1, 0].set_xlabel('Residuals (GPa)', fontsize=12)
    axes[1, 0].set_ylabel('Frequency', fontsize=12)
    axes[1, 0].set_title('Training Data: Residual Distribution', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # 残差の分布（テストデータ）
    axes[1, 1].hist(residuals_test, bins=20, alpha=0.7, edgecolor='black', color='orange')
    axes[1, 1].axvline(x=0, color='r', linestyle='--', lw=2)
    axes[1, 1].set_xlabel('Residuals (GPa)', fontsize=12)
    axes[1, 1].set_ylabel('Frequency', fontsize=12)
    axes[1, 1].set_title('Test Data: Residual Distribution', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f'residuals_analysis{title_suffix.lower().replace(" ", "_")}.png', 
                dpi=300, bbox_inches='tight')
    print(f"✅ 残差分析プロットを保存: residuals_analysis{title_suffix.lower().replace(' ', '_')}.png")
    plt.close()

def plot_feature_importance(model, feature_cols, top_n=20):
    """特徴量重要度のプロット"""
    if model is None:
        return
    
    if not hasattr(model, 'feature_importances_'):
        print("⚠️  このモデルには特徴量重要度がありません。")
        return
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    features = [feature_cols[i] for i in indices]
    values = importances[indices]
    
    bars = ax.barh(range(len(features)), values, color='steelblue', alpha=0.7)
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features, fontsize=10)
    ax.set_xlabel('Feature Importance', fontsize=12)
    ax.set_title(f'Top {top_n} Feature Importances', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # 値をバーの上に表示
    for i, (bar, val) in enumerate(zip(bars, values)):
        ax.text(val + 0.001, i, f'{val:.4f}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'feature_importance.png', dpi=300, bbox_inches='tight')
    print(f"✅ 特徴量重要度プロットを保存: feature_importance.png")
    plt.close()

def plot_model_comparison(results):
    """モデル性能比較"""
    if not results or 'results' not in results:
        return
    
    models_data = []
    for name, result in results['results'].items():
        models_data.append({
            'Model': name,
            'Test R²': result.get('test_r2', 0),
            'Test RMSE': result.get('test_rmse', 0),
            'Test MAE': result.get('test_mae', 0),
        })
    
    df = pd.DataFrame(models_data)
    df = df.sort_values('Test R²', ascending=False)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # R²比較
    axes[0].barh(df['Model'], df['Test R²'], color='steelblue', alpha=0.7)
    axes[0].set_xlabel('Test R² Score', fontsize=12)
    axes[0].set_title('Model Comparison: Test R²', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='x')
    for i, (idx, row) in enumerate(df.iterrows()):
        axes[0].text(row['Test R²'] + 0.01, i, f"{row['Test R²']:.4f}", 
                    va='center', fontsize=10)
    
    # RMSE比較
    axes[1].barh(df['Model'], df['Test RMSE'], color='coral', alpha=0.7)
    axes[1].set_xlabel('Test RMSE (GPa)', fontsize=12)
    axes[1].set_title('Model Comparison: Test RMSE', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='x')
    for i, (idx, row) in enumerate(df.iterrows()):
        axes[1].text(row['Test RMSE'] + 1, i, f"{row['Test RMSE']:.2f}", 
                    va='center', fontsize=10)
    
    # MAE比較
    axes[2].barh(df['Model'], df['Test MAE'], color='lightgreen', alpha=0.7)
    axes[2].set_xlabel('Test MAE (GPa)', fontsize=12)
    axes[2].set_title('Model Comparison: Test MAE', fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3, axis='x')
    for i, (idx, row) in enumerate(df.iterrows()):
        axes[2].text(row['Test MAE'] + 1, i, f"{row['Test MAE']:.2f}", 
                    va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'model_comparison_all.png', dpi=300, bbox_inches='tight')
    print(f"✅ モデル比較プロットを保存: model_comparison_all.png")
    plt.close()

def plot_optimization_history():
    """最適化の軌跡を可視化"""
    result_files = sorted(glob.glob(str(RESULTS_DIR / "*measured*.json")))
    
    if len(result_files) < 2:
        print("⚠️  最適化履歴が不十分です。")
        return
    
    optimization_data = []
    for result_file in result_files:
        with open(result_file, 'r') as f:
            result = json.load(f)
        
        timestamp = result.get('timestamp', '')
        if 'results' in result:
            for name, model_result in result['results'].items():
                optimization_data.append({
                    'Timestamp': timestamp,
                    'Model': name,
                    'Test R²': model_result.get('test_r2', 0),
                    'Test RMSE': model_result.get('test_rmse', 0),
                })
        elif 'optimized_r2' in result:
            optimization_data.append({
                'Timestamp': timestamp,
                'Model': 'Optimized',
                'Test R²': result.get('optimized_r2', 0),
                'Test RMSE': result.get('optimized_rmse', 0),
            })
    
    if not optimization_data:
        return
    
    df = pd.DataFrame(optimization_data)
    df = df.sort_values('Timestamp')
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # R²の推移
    for model in df['Model'].unique():
        model_data = df[df['Model'] == model]
        axes[0].plot(range(len(model_data)), model_data['Test R²'], 
                   marker='o', label=model, linewidth=2, markersize=8)
    
    axes[0].set_xlabel('Optimization Step', fontsize=12)
    axes[0].set_ylabel('Test R² Score', fontsize=12)
    axes[0].set_title('Optimization History: Test R²', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # RMSEの推移
    for model in df['Model'].unique():
        model_data = df[df['Model'] == model]
        axes[1].plot(range(len(model_data)), model_data['Test RMSE'], 
                   marker='s', label=model, linewidth=2, markersize=8)
    
    axes[1].set_xlabel('Optimization Step', fontsize=12)
    axes[1].set_ylabel('Test RMSE (GPa)', fontsize=12)
    axes[1].set_title('Optimization History: Test RMSE', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'optimization_history.png', dpi=300, bbox_inches='tight')
    print(f"✅ 最適化履歴プロットを保存: optimization_history.png")
    plt.close()

def plot_error_distribution(X, y, model):
    """誤差分布の可視化"""
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    if model is None:
        return
    
    y_pred_test = model.predict(X_test)
    errors = y_test - y_pred_test
    abs_errors = np.abs(errors)
    relative_errors = np.abs(errors / (y_test + 1e-10)) * 100
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 絶対誤差の分布
    axes[0, 0].hist(abs_errors, bins=20, alpha=0.7, edgecolor='black', color='steelblue')
    axes[0, 0].axvline(x=np.mean(abs_errors), color='r', linestyle='--', lw=2, 
                      label=f'Mean: {np.mean(abs_errors):.2f} GPa')
    axes[0, 0].set_xlabel('Absolute Error (GPa)', fontsize=12)
    axes[0, 0].set_ylabel('Frequency', fontsize=12)
    axes[0, 0].set_title('Absolute Error Distribution', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # 相対誤差の分布
    axes[0, 1].hist(relative_errors, bins=20, alpha=0.7, edgecolor='black', color='coral')
    axes[0, 1].axvline(x=np.mean(relative_errors), color='r', linestyle='--', lw=2,
                      label=f'Mean: {np.mean(relative_errors):.2f}%')
    axes[0, 1].set_xlabel('Relative Error (%)', fontsize=12)
    axes[0, 1].set_ylabel('Frequency', fontsize=12)
    axes[0, 1].set_title('Relative Error Distribution', fontsize=14, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # 誤差 vs 実測値
    axes[1, 0].scatter(y_test, errors, alpha=0.6, s=50, color='steelblue')
    axes[1, 0].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1, 0].set_xlabel('Actual Elastic Modulus (GPa)', fontsize=12)
    axes[1, 0].set_ylabel('Error (GPa)', fontsize=12)
    axes[1, 0].set_title('Error vs Actual Value', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 箱ひげ図
    axes[1, 1].boxplot([errors, abs_errors], labels=['Error', 'Absolute Error'])
    axes[1, 1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1, 1].set_ylabel('Error (GPa)', fontsize=12)
    axes[1, 1].set_title('Error Statistics', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'error_distribution.png', dpi=300, bbox_inches='tight')
    print(f"✅ 誤差分布プロットを保存: error_distribution.png")
    plt.close()

def plot_data_distribution(y):
    """データ分布の可視化"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # ヒストグラム
    axes[0].hist(y, bins=30, alpha=0.7, edgecolor='black', color='steelblue')
    axes[0].axvline(x=np.mean(y), color='r', linestyle='--', lw=2, 
                   label=f'Mean: {np.mean(y):.2f} GPa')
    axes[0].axvline(x=np.median(y), color='g', linestyle='--', lw=2,
                   label=f'Median: {np.median(y):.2f} GPa')
    axes[0].set_xlabel('Elastic Modulus (GPa)', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Elastic Modulus Distribution', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # 箱ひげ図
    axes[1].boxplot(y, vert=True)
    axes[1].set_ylabel('Elastic Modulus (GPa)', fontsize=12)
    axes[1].set_title('Elastic Modulus Box Plot', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # 統計情報を表示
    stats_text = f"Mean: {np.mean(y):.2f} GPa\n"
    stats_text += f"Median: {np.median(y):.2f} GPa\n"
    stats_text += f"Std: {np.std(y):.2f} GPa\n"
    stats_text += f"Min: {np.min(y):.2f} GPa\n"
    stats_text += f"Max: {np.max(y):.2f} GPa"
    axes[1].text(1.1, np.median(y), stats_text, fontsize=10, 
                verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'data_distribution.png', dpi=300, bbox_inches='tight')
    print(f"✅ データ分布プロットを保存: data_distribution.png")
    plt.close()

def plot_performance_metrics_comparison(results):
    """性能指標の総合比較"""
    if not results or 'results' not in results:
        return
    
    models_data = []
    for name, result in results['results'].items():
        models_data.append({
            'Model': name,
            'R²': result.get('test_r2', 0),
            'RMSE': result.get('test_rmse', 0),
            'MAE': result.get('test_mae', 0),
        })
    
    df = pd.DataFrame(models_data)
    df = df.sort_values('R²', ascending=False)
    
    # 正規化（0-1スケール）
    df_norm = df.copy()
    for col in ['R²', 'RMSE', 'MAE']:
        if col == 'R²':
            df_norm[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min() + 1e-10)
        else:
            df_norm[col] = 1 - (df[col] - df[col].min()) / (df[col].max() - df[col].min() + 1e-10)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = np.arange(len(df))
    width = 0.25
    
    bars1 = ax.bar(x - width, df_norm['R²'], width, label='R² (normalized)', alpha=0.8)
    bars2 = ax.bar(x, df_norm['RMSE'], width, label='RMSE (normalized, inverted)', alpha=0.8)
    bars3 = ax.bar(x + width, df_norm['MAE'], width, label='MAE (normalized, inverted)', alpha=0.8)
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Normalized Score', fontsize=12)
    ax.set_title('Comprehensive Performance Comparison (Normalized)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df['Model'], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'performance_metrics_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✅ 性能指標比較プロットを保存: performance_metrics_comparison.png")
    plt.close()

def main():
    """メイン関数"""
    print("=" * 80)
    print("実測データのみの最適化結果を包括的に可視化")
    print("=" * 80)
    
    # データとモデルを読み込む
    X, y, model, results = load_latest_model_and_data()
    
    if X is None:
        return
    
    print(f"\n📊 データ統計:")
    print(f"   サンプル数: {len(X)}")
    print(f"   特徴量数: {len(X.columns)}")
    print(f"   弾性率範囲: {y.min():.2f} - {y.max():.2f} GPa")
    print(f"   平均: {y.mean():.2f} GPa")
    
    # 可視化を実行
    print("\n" + "=" * 80)
    print("可視化を実行中...")
    print("=" * 80)
    
    # 1. 予測値 vs 実測値
    print("\n1. 予測値 vs 実測値プロット作成中...")
    plot_predicted_vs_actual(X, y, model, " (Measured Data Only)")
    
    # 2. 残差分析
    print("\n2. 残差分析プロット作成中...")
    plot_residuals(X, y, model, " (Measured Data Only)")
    
    # 3. 特徴量重要度
    print("\n3. 特徴量重要度プロット作成中...")
    plot_feature_importance(model, X.columns)
    
    # 4. モデル比較
    print("\n4. モデル比較プロット作成中...")
    plot_model_comparison(results)
    
    # 5. 最適化履歴
    print("\n5. 最適化履歴プロット作成中...")
    plot_optimization_history()
    
    # 6. 誤差分布
    print("\n6. 誤差分布プロット作成中...")
    plot_error_distribution(X, y, model)
    
    # 7. データ分布
    print("\n7. データ分布プロット作成中...")
    plot_data_distribution(y)
    
    # 8. 性能指標比較
    print("\n8. 性能指標比較プロット作成中...")
    plot_performance_metrics_comparison(results)
    
    print("\n" + "=" * 80)
    print("✅ すべての可視化が完了しました！")
    print(f"📁 保存先: {FIGURES_DIR}")
    print("=" * 80)
    
    # 生成されたファイル一覧
    figure_files = sorted(glob.glob(str(FIGURES_DIR / "*.png")))
    print(f"\n📊 生成されたプロット ({len(figure_files)}個):")
    for f in figure_files:
        print(f"   - {Path(f).name}")

if __name__ == "__main__":
    main()
