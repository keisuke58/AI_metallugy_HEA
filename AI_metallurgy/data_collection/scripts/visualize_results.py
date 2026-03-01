#!/usr/bin/env python3
"""
結果の可視化スクリプト

1. 弾性率比較図（骨 vs 従来インプラント vs 目標）
2. モデル性能比較（R²、RMSEのバーグラフ）
3. 予測値 vs 実験値（散布図）
4. 残差分析（残差の分布）
5. 特徴重要度（Random Forest）
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import pickle

# 設定
BASE_DIR = Path(__file__).parent.parent
PROCESSED_DATA_DIR = BASE_DIR / "processed_data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
FIGS_DIR = BASE_DIR / "figures"
FIGS_DIR.mkdir(parents=True, exist_ok=True)

# 日本語フォント設定（必要に応じて）
plt.rcParams['font.size'] = 12
sns.set_style("whitegrid")

def load_data_and_models():
    """
    データとモデルを読み込む
    """
    # データを読み込む
    data_file = PROCESSED_DATA_DIR / "data_with_features.csv"
    df = pd.read_csv(data_file)
    df = df.dropna(subset=['elastic_modulus'])
    
    # 特徴量を選択
    feature_cols = []
    for col in df.columns:
        if col in ['alloy_name', 'elastic_modulus', 'source']:
            continue
        if df[col].dtype in [np.float64, np.int64]:
            if df[col].notna().sum() / len(df) >= 0.5:
                feature_cols.append(col)
    
    X = df[feature_cols].fillna(df[feature_cols].median())
    y = df['elastic_modulus'].values
    
    # 結果を読み込む
    results_file = RESULTS_DIR / "model_results.json"
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # モデルを読み込む（予測用）
    models = {}
    for name in ['LIN', 'L', 'R', 'RF', 'SVR']:
        try:
            with open(MODELS_DIR / f"model_{name}.pkl", 'rb') as f:
                models[name] = pickle.load(f)
        except:
            pass
    
    return df, X, y, results, models

def plot_elastic_modulus_comparison():
    """
    弾性率比較図（骨 vs 従来インプラント vs 目標）
    """
    print("📊 弾性率比較図を作成中...")
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # データ
    # 骨のヤング率: 海綿骨 0.05-2 GPa, 皮質骨 7-30 GPa, 骨全体（平均的扱い）10-30 GPa
    categories = ['Human Bone\n(Cortical: 7-30 GPa)', 'Target Range\n(30-90 GPa)', 'Conventional\nImplants', 'HEA Data\n(Min-Max)']
    values_min = [7, 30, 200, 27]
    values_max = [30, 90, 220, 466]
    colors = ['green', 'blue', 'red', 'orange']
    
    # バーを描画
    for i, (cat, v_min, v_max, color) in enumerate(zip(categories, values_min, values_max, colors)):
        ax.barh(i, v_max - v_min, left=v_min, height=0.6, color=color, alpha=0.7, label=cat)
        ax.text((v_min + v_max) / 2, i, f'{v_min}-{v_max} GPa', 
                ha='center', va='center', fontweight='bold')
    
    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels(categories)
    ax.set_xlabel('Elastic Modulus (GPa)', fontsize=14)
    ax.set_title('Elastic Modulus Comparison: Bone vs Implants vs Target', fontsize=16, fontweight='bold')
    ax.set_xlim(0, 500)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGS_DIR / "elastic_modulus_comparison.png", dpi=300, bbox_inches='tight')
    print(f"   ✅ 保存: {FIGS_DIR / 'elastic_modulus_comparison.png'}")
    plt.close()

def plot_model_performance(results):
    """
    モデル性能比較（R²、RMSEのバーグラフ）
    """
    print("📊 モデル性能比較図を作成中...")
    
    model_names = []
    r2_scores = []
    rmse_scores = []
    
    for name, result in results.items():
        if result is None or result['test_r2'] < -100:  # 異常値を除外
            continue
        model_names.append(name)
        r2_scores.append(result['test_r2'])
        rmse_scores.append(result['test_rmse'])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # R²スコア
    bars1 = ax1.bar(model_names, r2_scores, color='steelblue', alpha=0.7)
    ax1.set_ylabel('R² Score', fontsize=14)
    ax1.set_title('Model Performance: R² Score', fontsize=16, fontweight='bold')
    ax1.set_ylim(0, max(r2_scores) * 1.2)
    ax1.grid(axis='y', alpha=0.3)
    
    # 値を表示
    for bar, score in zip(bars1, r2_scores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.3f}', ha='center', va='bottom')
    
    # RMSE
    bars2 = ax2.bar(model_names, rmse_scores, color='coral', alpha=0.7)
    ax2.set_ylabel('RMSE (GPa)', fontsize=14)
    ax2.set_title('Model Performance: RMSE', fontsize=16, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # 値を表示
    for bar, score in zip(bars2, rmse_scores):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.1f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(FIGS_DIR / "model_performance_comparison.png", dpi=300, bbox_inches='tight')
    print(f"   ✅ 保存: {FIGS_DIR / 'model_performance_comparison.png'}")
    plt.close()

def plot_predicted_vs_actual(df, X, y, models):
    """
    予測値 vs 実験値（散布図）
    """
    print("📊 予測値 vs 実験値の散布図を作成中...")
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    n_models = len([m for m in models.values() if m is not None])
    if n_models == 0:
        print("   ⚠️  モデルが読み込めませんでした")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    plot_idx = 0
    for name, model in models.items():
        if model is None or plot_idx >= len(axes):
            continue
        
        ax = axes[plot_idx]
        
        # 予測
        if name == 'P':
            # Polynomial Regressionは特別な処理が必要
            continue
        elif name == 'SVR' and isinstance(model, dict):
            y_pred = model['model'].predict(model['scaler'].transform(X_test))
        else:
            y_pred = model.predict(X_test)
        
        # 散布図
        ax.scatter(y_test, y_pred, alpha=0.6, s=50)
        
        # 理想線（y=x）
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal')
        
        # R²を計算
        from sklearn.metrics import r2_score
        r2 = r2_score(y_test, y_pred)
        
        ax.set_xlabel('Actual Elastic Modulus (GPa)', fontsize=12)
        ax.set_ylabel('Predicted Elastic Modulus (GPa)', fontsize=12)
        ax.set_title(f'{name} (R² = {r2:.3f})', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.legend()
        
        plot_idx += 1
    
    # 未使用のサブプロットを非表示
    for idx in range(plot_idx, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(FIGS_DIR / "predicted_vs_actual.png", dpi=300, bbox_inches='tight')
    print(f"   ✅ 保存: {FIGS_DIR / 'predicted_vs_actual.png'}")
    plt.close()

def plot_feature_importance(results):
    """
    特徴重要度（Random Forest）
    """
    print("📊 特徴重要度図を作成中...")
    
    if 'RF' not in results or results['RF'] is None:
        print("   ⚠️  Random Forestの結果が見つかりませんでした")
        return
    
    if 'feature_importance' not in results['RF']:
        print("   ⚠️  特徴重要度データが見つかりませんでした")
        return
    
    importance = results['RF']['feature_importance']
    
    # 重要度でソート
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:15]
    
    features = [f[0] for f in sorted_importance]
    values = [f[1] for f in sorted_importance]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    bars = ax.barh(range(len(features)), values, color='steelblue', alpha=0.7)
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features)
    ax.set_xlabel('Feature Importance', fontsize=14)
    ax.set_title('Random Forest: Top 15 Feature Importance', fontsize=16, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    
    # 値を表示
    for i, (bar, val) in enumerate(zip(bars, values)):
        ax.text(val, i, f'{val:.3f}', va='center', ha='left', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(FIGS_DIR / "feature_importance.png", dpi=300, bbox_inches='tight')
    print(f"   ✅ 保存: {FIGS_DIR / 'feature_importance.png'}")
    plt.close()

def plot_residuals(df, X, y, models):
    """
    残差分析（残差の分布）
    """
    print("📊 残差分析図を作成中...")
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    plot_idx = 0
    for name, model in models.items():
        if model is None or plot_idx >= len(axes):
            continue
        
        ax = axes[plot_idx]
        
        # 予測
        if name == 'SVR' and isinstance(model, dict):
            y_pred = model['model'].predict(model['scaler'].transform(X_test))
        else:
            y_pred = model.predict(X_test)
        
        # 残差
        residuals = y_test - y_pred
        
        # 残差の散布図
        ax.scatter(y_pred, residuals, alpha=0.6, s=50)
        ax.axhline(y=0, color='r', linestyle='--', lw=2)
        ax.set_xlabel('Predicted Elastic Modulus (GPa)', fontsize=12)
        ax.set_ylabel('Residuals (GPa)', fontsize=12)
        ax.set_title(f'{name} Residuals', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        
        plot_idx += 1
    
    # 未使用のサブプロットを非表示
    for idx in range(plot_idx, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(FIGS_DIR / "residuals_analysis.png", dpi=300, bbox_inches='tight')
    print(f"   ✅ 保存: {FIGS_DIR / 'residuals_analysis.png'}")
    plt.close()

def create_all_visualizations():
    """
    すべての可視化を作成
    """
    print("=" * 60)
    print("結果の可視化")
    print("=" * 60)
    
    df, X, y, results, models = load_data_and_models()
    
    # 各可視化を作成
    plot_elastic_modulus_comparison()
    plot_model_performance(results)
    plot_predicted_vs_actual(df, X, y, models)
    plot_feature_importance(results)
    plot_residuals(df, X, y, models)
    
    print("\n" + "=" * 60)
    print("可視化完了")
    print("=" * 60)
    print(f"✅ すべての図を保存しました: {FIGS_DIR}")

if __name__ == "__main__":
    create_all_visualizations()
