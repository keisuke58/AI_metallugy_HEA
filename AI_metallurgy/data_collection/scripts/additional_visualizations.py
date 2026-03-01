#!/usr/bin/env python3
"""
追加の可視化を作成
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import glob
import pickle
from sklearn.model_selection import train_test_split, learning_curve, validation_curve
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'DejaVu Sans'
sns.set_style("whitegrid")

BASE_DIR = Path(__file__).parent.parent
PROCESSED_DATA_DIR = BASE_DIR / "processed_data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures"

def load_model_and_data():
    """モデルとデータを読み込む"""
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
    
    model_dirs = sorted(glob.glob(str(MODELS_DIR / "final_measured_*")))
    if not model_dirs:
        model_dirs = sorted(glob.glob(str(MODELS_DIR / "*measured*")))
    
    model = None
    if model_dirs:
        latest_model_dir = Path(model_dirs[-1])
        model_file = latest_model_dir / "best_model.pkl"
        if model_file.exists():
            with open(model_file, 'rb') as f:
                model = pickle.load(f)
    
    return X_imputed, y, model

def plot_learning_curves(X, y, model):
    """学習曲線"""
    if model is None:
        return
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_train, y_train, cv=5, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='r2'
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score', linewidth=2)
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
    ax.plot(train_sizes, val_mean, 'o-', color='red', label='Validation Score', linewidth=2)
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
    
    ax.set_xlabel('Training Set Size', fontsize=12)
    ax.set_ylabel('R² Score', fontsize=12)
    ax.set_title('Learning Curves', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'learning_curves_detailed.png', dpi=300, bbox_inches='tight')
    print(f"✅ 学習曲線を保存: learning_curves_detailed.png")
    plt.close()

def plot_feature_correlation(X, top_n=20):
    """特徴量相関マトリックス（上位特徴量のみ）"""
    if hasattr(X, 'columns'):
        # 相関が高い特徴量を選択
        corr_matrix = X.corr().abs()
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # 高相関の特徴量ペアを取得
        high_corr_pairs = []
        for col in upper_triangle.columns:
            for idx in upper_triangle.index:
                if pd.notna(upper_triangle.loc[idx, col]) and upper_triangle.loc[idx, col] > 0.5:
                    high_corr_pairs.append((idx, col, upper_triangle.loc[idx, col]))
        
        # 上位N個の特徴量を選択
        if len(high_corr_pairs) > 0:
            high_corr_pairs.sort(key=lambda x: x[2], reverse=True)
            selected_features = set()
            for feat1, feat2, _ in high_corr_pairs[:top_n*2]:
                selected_features.add(feat1)
                selected_features.add(feat2)
            
            if len(selected_features) > 1:
                selected_features = list(selected_features)[:top_n]
                X_selected = X[selected_features]
                
                corr = X_selected.corr()
                
                fig, ax = plt.subplots(figsize=(12, 10))
                sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                           square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
                ax.set_title(f'Feature Correlation Matrix (Top {len(selected_features)} Features)', 
                           fontsize=14, fontweight='bold')
                plt.xticks(rotation=45, ha='right')
                plt.yticks(rotation=0)
                plt.tight_layout()
                plt.savefig(FIGURES_DIR / 'feature_correlation.png', dpi=300, bbox_inches='tight')
                print(f"✅ 特徴量相関マトリックスを保存: feature_correlation.png")
                plt.close()

def plot_prediction_intervals(X, y, model):
    """予測区間の可視化"""
    if model is None:
        return
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    y_pred_test = model.predict(X_test)
    residuals = y_test - y_pred_test
    std_residual = np.std(residuals)
    
    # 95%予測区間
    lower_bound = y_pred_test - 1.96 * std_residual
    upper_bound = y_pred_test + 1.96 * std_residual
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 予測区間
    ax.fill_between(range(len(y_test)), lower_bound, upper_bound, 
                    alpha=0.3, color='lightblue', label='95% Prediction Interval')
    
    # 予測値
    ax.plot(range(len(y_test)), y_pred_test, 'o-', color='blue', 
           label='Predicted', linewidth=2, markersize=6)
    
    # 実測値
    ax.plot(range(len(y_test)), y_test, 's-', color='red', 
           label='Actual', linewidth=2, markersize=6)
    
    # 区間内の点をカウント
    in_interval = np.sum((y_test >= lower_bound) & (y_test <= upper_bound))
    coverage = in_interval / len(y_test) * 100
    
    ax.set_xlabel('Sample Index', fontsize=12)
    ax.set_ylabel('Elastic Modulus (GPa)', fontsize=12)
    ax.set_title(f'Prediction Intervals (Coverage: {coverage:.1f}%)', 
                fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'prediction_intervals.png', dpi=300, bbox_inches='tight')
    print(f"✅ 予測区間プロットを保存: prediction_intervals.png")
    plt.close()

def plot_feature_vs_target(X, y, model, top_features=6):
    """主要特徴量とターゲットの関係"""
    if model is None:
        return
    
    # 特徴量重要度から上位特徴量を選択
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_features]
        selected_features = [X.columns[i] for i in indices]
    else:
        # 相関が高い特徴量を選択
        correlations = [abs(np.corrcoef(X[col].values, y)[0, 1]) 
                       for col in X.columns]
        indices = np.argsort(correlations)[::-1][:top_features]
        selected_features = [X.columns[i] for i in indices]
    
    n_cols = 3
    n_rows = (top_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten() if top_features > 1 else [axes]
    
    for idx, feat in enumerate(selected_features):
        ax = axes[idx]
        ax.scatter(X[feat], y, alpha=0.6, s=50)
        
        # 回帰直線
        z = np.polyfit(X[feat], y, 1)
        p = np.poly1d(z)
        ax.plot(X[feat], p(X[feat]), "r--", alpha=0.8, linewidth=2)
        
        corr = np.corrcoef(X[feat], y)[0, 1]
        ax.set_xlabel(feat, fontsize=10)
        ax.set_ylabel('Elastic Modulus (GPa)', fontsize=10)
        ax.set_title(f'{feat}\n(Correlation: {corr:.3f})', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    # 余分なサブプロットを非表示
    for idx in range(len(selected_features), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'feature_vs_target.png', dpi=300, bbox_inches='tight')
    print(f"✅ 特徴量 vs ターゲットプロットを保存: feature_vs_target.png")
    plt.close()

def plot_all_results_summary():
    """すべての結果のサマリー"""
    result_files = sorted(glob.glob(str(RESULTS_DIR / "*measured*.json")))
    
    all_results = []
    for result_file in result_files:
        with open(result_file, 'r') as f:
            result = json.load(f)
        
        if 'results' in result:
            for name, model_result in result['results'].items():
                all_results.append({
                    'File': Path(result_file).stem,
                    'Model': name,
                    'Test R²': model_result.get('test_r2', 0),
                    'Test RMSE': model_result.get('test_rmse', 0),
                    'Test MAE': model_result.get('test_mae', 0),
                })
        elif 'optimized_r2' in result:
            all_results.append({
                'File': Path(result_file).stem,
                'Model': 'Optimized',
                'Test R²': result.get('optimized_r2', 0),
                'Test RMSE': result.get('optimized_rmse', 0),
                'Test MAE': result.get('optimized_mae', 0),
            })
    
    if not all_results:
        return
    
    df = pd.DataFrame(all_results)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # R²の分布
    axes[0, 0].hist(df['Test R²'], bins=20, alpha=0.7, edgecolor='black', color='steelblue')
    axes[0, 0].axvline(x=df['Test R²'].mean(), color='r', linestyle='--', lw=2,
                      label=f'Mean: {df["Test R²"].mean():.4f}')
    axes[0, 0].set_xlabel('Test R² Score', fontsize=12)
    axes[0, 0].set_ylabel('Frequency', fontsize=12)
    axes[0, 0].set_title('Distribution of Test R² Scores', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # RMSEの分布
    axes[0, 1].hist(df['Test RMSE'], bins=20, alpha=0.7, edgecolor='black', color='coral')
    axes[0, 1].axvline(x=df['Test RMSE'].mean(), color='r', linestyle='--', lw=2,
                      label=f'Mean: {df["Test RMSE"].mean():.2f} GPa')
    axes[0, 1].set_xlabel('Test RMSE (GPa)', fontsize=12)
    axes[0, 1].set_ylabel('Frequency', fontsize=12)
    axes[0, 1].set_title('Distribution of Test RMSE', fontsize=14, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # R² vs RMSE
    axes[1, 0].scatter(df['Test RMSE'], df['Test R²'], alpha=0.6, s=100, c=df['Test MAE'], 
                      cmap='viridis', edgecolors='black', linewidth=1)
    axes[1, 0].set_xlabel('Test RMSE (GPa)', fontsize=12)
    axes[1, 0].set_ylabel('Test R² Score', fontsize=12)
    axes[1, 0].set_title('Test R² vs Test RMSE (colored by MAE)', fontsize=14, fontweight='bold')
    cbar = plt.colorbar(axes[1, 0].collections[0], ax=axes[1, 0])
    cbar.set_label('Test MAE (GPa)', fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    
    # 最良モデルの比較
    best_models = df.nlargest(5, 'Test R²')
    axes[1, 1].barh(range(len(best_models)), best_models['Test R²'], alpha=0.7, color='steelblue')
    axes[1, 1].set_yticks(range(len(best_models)))
    axes[1, 1].set_yticklabels([f"{row['Model']}\n({Path(row['File']).stem})" 
                               for _, row in best_models.iterrows()], fontsize=9)
    axes[1, 1].set_xlabel('Test R² Score', fontsize=12)
    axes[1, 1].set_title('Top 5 Models', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='x')
    
    for i, (_, row) in enumerate(best_models.iterrows()):
        axes[1, 1].text(row['Test R²'] + 0.01, i, f"{row['Test R²']:.4f}", 
                       va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'all_results_summary.png', dpi=300, bbox_inches='tight')
    print(f"✅ 全結果サマリーを保存: all_results_summary.png")
    plt.close()

def main():
    """メイン関数"""
    print("=" * 80)
    print("追加の可視化を作成")
    print("=" * 80)
    
    X, y, model = load_model_and_data()
    
    if X is None:
        return
    
    print("\n1. 学習曲線作成中...")
    plot_learning_curves(X, y, model)
    
    print("\n2. 特徴量相関マトリックス作成中...")
    plot_feature_correlation(X)
    
    print("\n3. 予測区間プロット作成中...")
    plot_prediction_intervals(X, y, model)
    
    print("\n4. 特徴量 vs ターゲットプロット作成中...")
    plot_feature_vs_target(X, y, model)
    
    print("\n5. 全結果サマリー作成中...")
    plot_all_results_summary()
    
    print("\n" + "=" * 80)
    print("✅ 追加の可視化が完了しました！")
    print("=" * 80)

if __name__ == "__main__":
    main()
