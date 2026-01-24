#!/usr/bin/env python3
"""
学習曲線分析スクリプト
データセットサイズとモデル性能の関係を可視化
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# 日本語フォント設定
plt.rcParams['font.family'] = 'DejaVu Sans'
sns.set_style("whitegrid")

# 設定
BASE_DIR = Path(__file__).parent.parent
PROCESSED_DATA_DIR = BASE_DIR / "processed_data"
FIGURES_DIR = BASE_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    """特徴量付きデータを読み込む"""
    data_file = PROCESSED_DATA_DIR / "data_with_features.csv"
    
    if not data_file.exists():
        print(f"❌ ファイルが見つかりません: {data_file}")
        return None, None, None
    
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
    
    return X, y, feature_cols

def plot_learning_curves(X, y, models, model_names):
    """
    学習曲線を描画
    """
    print("=" * 60)
    print("学習曲線分析")
    print("=" * 60)
    
    n_samples = len(X)
    print(f"📊 総データ数: {n_samples}")
    print(f"📊 特徴量数: {X.shape[1]}")
    
    # データを分割（テストセットは固定）
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 異なる訓練データサイズで評価
    train_sizes = np.linspace(0.1, 1.0, 10) * len(X_train_full)
    train_sizes = train_sizes.astype(int)
    train_sizes = np.unique(train_sizes)  # 重複除去
    
    results = {}
    
    for model, name in zip(models, model_names):
        print(f"\n📈 {name} の学習曲線を計算中...")
        
        train_scores = []
        test_scores = []
        train_rmse = []
        test_rmse = []
        
        for size in train_sizes:
            # 指定サイズの訓練データをサンプリング
            indices = np.random.choice(len(X_train_full), size=size, replace=False)
            X_train_subset = X_train_full.iloc[indices]
            y_train_subset = y_train_full[indices]
            
            # モデルを訓練
            model.fit(X_train_subset, y_train_subset)
            
            # 訓練データでの評価
            y_pred_train = model.predict(X_train_subset)
            train_r2 = r2_score(y_train_subset, y_pred_train)
            train_scores.append(train_r2)
            train_rmse.append(np.sqrt(mean_squared_error(y_train_subset, y_pred_train)))
            
            # テストデータでの評価
            y_pred_test = model.predict(X_test)
            test_r2 = r2_score(y_test, y_pred_test)
            test_r2 = r2_score(y_test, y_pred_test)
            test_scores.append(test_r2)
            test_rmse.append(np.sqrt(mean_squared_error(y_test, y_pred_test)))
        
        results[name] = {
            'train_sizes': train_sizes,
            'train_scores': train_scores,
            'test_scores': test_scores,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse
        }
        
        print(f"   完了 - 最終 Test R²: {test_scores[-1]:.4f}")
    
    # 可視化
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Learning Curves: Effect of Dataset Size on Model Performance', 
                 fontsize=16, fontweight='bold')
    
    # 1. R² Score
    ax1 = axes[0, 0]
    for name in model_names:
        data = results[name]
        ax1.plot(data['train_sizes'], data['train_scores'], 
                'o-', label=f'{name} (Train)', alpha=0.7, linewidth=2)
        ax1.plot(data['train_sizes'], data['test_scores'], 
                's--', label=f'{name} (Test)', alpha=0.7, linewidth=2)
    ax1.set_xlabel('Training Set Size', fontsize=12)
    ax1.set_ylabel('R² Score', fontsize=12)
    ax1.set_title('R² Score vs Training Set Size', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    # 2. RMSE
    ax2 = axes[0, 1]
    for name in model_names:
        data = results[name]
        ax2.plot(data['train_sizes'], data['train_rmse'], 
                'o-', label=f'{name} (Train)', alpha=0.7, linewidth=2)
        ax2.plot(data['train_sizes'], data['test_rmse'], 
                's--', label=f'{name} (Test)', alpha=0.7, linewidth=2)
    ax2.set_xlabel('Training Set Size', fontsize=12)
    ax2.set_ylabel('RMSE (GPa)', fontsize=12)
    ax2.set_title('RMSE vs Training Set Size', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # 3. Test R² の改善率
    ax3 = axes[1, 0]
    for name in model_names:
        data = results[name]
        baseline = data['test_scores'][0]  # 最小データサイズでの性能
        improvement = [(score - baseline) / abs(baseline) * 100 
                      if baseline != 0 else 0 
                      for score in data['test_scores']]
        ax3.plot(data['train_sizes'], improvement, 
                'o-', label=name, linewidth=2, markersize=6)
    ax3.set_xlabel('Training Set Size', fontsize=12)
    ax3.set_ylabel('Improvement in Test R² (%)', fontsize=12)
    ax3.set_title('Performance Improvement vs Dataset Size', fontsize=14, fontweight='bold')
    ax3.legend(loc='best', fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    # 4. Train-Test Gap (過学習の指標)
    ax4 = axes[1, 1]
    for name in model_names:
        data = results[name]
        gap = [train - test for train, test in 
               zip(data['train_scores'], data['test_scores'])]
        ax4.plot(data['train_sizes'], gap, 
                'o-', label=name, linewidth=2, markersize=6)
    ax4.set_xlabel('Training Set Size', fontsize=12)
    ax4.set_ylabel('Train R² - Test R² (Overfitting Gap)', fontsize=12)
    ax4.set_title('Overfitting Analysis', fontsize=14, fontweight='bold')
    ax4.legend(loc='best', fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    
    # 保存
    output_file = FIGURES_DIR / "learning_curves.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✅ 学習曲線を保存しました: {output_file}")
    
    return results

def analyze_data_requirements(results, current_size, n_features):
    """
    データセットサイズの推奨値を分析
    """
    print("\n" + "=" * 60)
    print("データセットサイズ分析")
    print("=" * 60)
    
    print(f"\n📊 現在の状況:")
    print(f"   データ数: {current_size}")
    print(f"   特徴量数: {n_features}")
    print(f"   サンプル/特徴量比: {current_size/n_features:.1f}")
    
    print(f"\n📈 推奨値:")
    print(f"   最低限: {n_features * 10}サンプル (10サンプル/特徴量)")
    print(f"   推奨: {n_features * 20}サンプル (20サンプル/特徴量)")
    print(f"   理想: {n_features * 37}サンプル (37サンプル/特徴量)")
    
    # 各モデルの性能改善を分析
    print(f"\n📊 モデル別の性能改善予測:")
    for name, data in results.items():
        current_r2 = data['test_scores'][-1]  # 現在のデータサイズでの性能
        max_r2 = max(data['test_scores'])
        improvement = (max_r2 - current_r2) / abs(current_r2) * 100 if current_r2 != 0 else 0
        
        print(f"\n   {name}:")
        print(f"     現在の Test R²: {current_r2:.4f}")
        print(f"     最大 Test R²: {max_r2:.4f}")
        print(f"     改善余地: {improvement:.1f}%")
        
        # 性能が飽和するポイントを探す
        if len(data['test_scores']) > 3:
            recent_improvement = (data['test_scores'][-1] - data['test_scores'][-3]) / abs(data['test_scores'][-3]) * 100
            if recent_improvement < 1.0:
                print(f"     → 性能はほぼ飽和しています")
            else:
                print(f"     → データを増やすと改善が見込めます")

def main():
    """メイン関数"""
    # データを読み込む
    X, y, feature_cols = load_data()
    if X is None:
        return
    
    # モデルを定義
    models = [
        RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        Lasso(alpha=0.1, max_iter=10000),
        Ridge(alpha=1.0, max_iter=10000)
    ]
    model_names = ['Random Forest', 'Lasso', 'Ridge']
    
    # 学習曲線を計算・可視化
    results = plot_learning_curves(X, y, models, model_names)
    
    # データ要件を分析
    analyze_data_requirements(results, len(X), len(feature_cols))
    
    print("\n" + "=" * 60)
    print("✅ 分析完了")
    print("=" * 60)

if __name__ == "__main__":
    main()
