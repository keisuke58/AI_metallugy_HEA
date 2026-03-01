#!/usr/bin/env python3
"""
実測データのみで圧倒的な成果（R² > 0.70）を達成
元の0.6992を超えることを目標に、より詳細な最適化を実施
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.impute import SimpleImputer

BASE_DIR = Path(__file__).parent.parent
PROCESSED_DATA_DIR = BASE_DIR / "processed_data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

def load_data():
    """データを読み込む"""
    data_file = PROCESSED_DATA_DIR / "data_preprocessed.csv"
    df = pd.read_csv(data_file)
    
    measured_sources = ['DOE/OSTI', 'Gorsse Dataset', 'Latest Research']
    df_measured = df[df['source'].isin(measured_sources)].copy()
    
    exclude_cols = ['alloy_name', 'elastic_modulus', 'source', 'phases']
    feature_cols = [col for col in df_measured.columns 
                    if col not in exclude_cols and df_measured[col].dtype in [np.float64, np.int64]
                    and df_measured[col].notna().sum() / len(df_measured) >= 0.7]
    
    X = df_measured[feature_cols].copy()
    y = df_measured['elastic_modulus'].values
    
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=feature_cols)
    
    return X_imputed, y, imputer, feature_cols

def main():
    """メイン関数"""
    print("=" * 80)
    print("実測データのみで圧倒的な成果を達成するための究極の最適化")
    print("=" * 80)
    
    X, y, imputer, feature_cols = load_data()
    
    # データ分割（元の設定と同じ）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\n📊 データ分割:")
    print(f"   訓練データ: {len(X_train)}サンプル")
    print(f"   テストデータ: {len(X_test)}サンプル")
    print(f"   特徴量数: {len(feature_cols)}個")
    
    # 元の設定を再現（n_estimators=100, max_depth=10）
    print("\n" + "=" * 80)
    print("元の設定での再現")
    print("=" * 80)
    
    gb_original = GradientBoostingRegressor(n_estimators=100, max_depth=10, random_state=42)
    gb_original.fit(X_train, y_train)
    
    y_pred_train_orig = gb_original.predict(X_train)
    y_pred_test_orig = gb_original.predict(X_test)
    
    r2_orig = r2_score(y_test, y_pred_test_orig)
    rmse_orig = np.sqrt(mean_squared_error(y_test, y_pred_test_orig))
    mae_orig = mean_absolute_error(y_test, y_pred_test_orig)
    
    print(f"\n📊 元の設定（n_estimators=100, max_depth=10）:")
    print(f"   Test R²: {r2_orig:.4f}")
    print(f"   Test RMSE: {rmse_orig:.2f} GPa")
    print(f"   Test MAE: {mae_orig:.2f} GPa")
    
    # 詳細なランダムサーチ最適化
    print("\n" + "=" * 80)
    print("詳細なランダムサーチ最適化")
    print("=" * 80)
    
    from sklearn.model_selection import RandomizedSearchCV
    from scipy.stats import randint, uniform
    
    # 元の結果に近いパラメータ範囲で詳細に探索
    param_distributions = {
        'n_estimators': randint(80, 250),
        'max_depth': randint(8, 15),
        'learning_rate': uniform(0.05, 0.15),
        'min_samples_split': randint(2, 15),
        'min_samples_leaf': randint(1, 8),
        'subsample': uniform(0.8, 0.2),
        'max_features': ['sqrt', 'log2', None]
    }
    
    print("\n🔧 ランダムサーチ実行中（300回試行）...")
    
    gb = GradientBoostingRegressor(random_state=42)
    
    # ランダムサーチ（より多くの試行回数）
    random_search = RandomizedSearchCV(
        gb, param_distributions, n_iter=300, cv=5, scoring='r2',
        n_jobs=-1, random_state=42, verbose=1
    )
    
    random_search.fit(X_train, y_train)
    
    grid_search = random_search
    
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)
    
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='r2', n_jobs=-1)
    
    r2_test = r2_score(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mae_test = mean_absolute_error(y_test, y_pred_test)
    
    print(f"\n✅ 最適パラメータ:")
    for key, value in best_params.items():
        print(f"   {key}: {value}")
    
    print(f"\n📊 最適化後の結果:")
    print(f"   CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"   Test R²: {r2_test:.4f} ⭐")
    print(f"   Test RMSE: {rmse_test:.2f} GPa")
    print(f"   Test MAE: {mae_test:.2f} GPa")
    
    # 比較
    print(f"\n{'='*80}")
    print("結果比較")
    print(f"{'='*80}")
    print(f"{'設定':<30} {'Test R²':<12} {'Test RMSE':<12} {'Test MAE':<12}")
    print("-" * 80)
    print(f"{'元の設定 (n=100, d=10)':<30} {r2_orig:>10.4f}   {rmse_orig:>10.2f} GPa   {mae_orig:>10.2f} GPa")
    print(f"{'最適化後':<30} {r2_test:>10.4f}   {rmse_test:>10.2f} GPa   {mae_test:>10.2f} GPa")
    improvement = r2_test - r2_orig
    print(f"\n改善: {improvement:+.4f} ({improvement/r2_orig*100:+.2f}%)")
    
    if r2_test >= 0.70:
        print(f"\n🎉🎉🎉 圧倒的な成果！ R² >= 0.70 を達成しました！ 🎉🎉🎉")
    elif r2_test >= r2_orig:
        print(f"\n✨ 改善達成！ 元の結果を上回りました！")
    print(f"{'='*80}")
    
    # 結果を保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = MODELS_DIR / f"ultimate_final_{timestamp}"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    with open(model_dir / "best_model.pkl", 'wb') as f:
        pickle.dump(best_model, f)
    
    with open(model_dir / "imputer.pkl", 'wb') as f:
        pickle.dump(imputer, f)
    
    results_summary = {
        'timestamp': timestamp,
        'original_r2': float(r2_orig),
        'original_rmse': float(rmse_orig),
        'original_mae': float(mae_orig),
        'optimized_r2': float(r2_test),
        'optimized_rmse': float(rmse_test),
        'optimized_mae': float(mae_test),
        'improvement': float(improvement),
        'improvement_percent': float(improvement/r2_orig*100),
        'best_params': best_params,
        'cv_r2_mean': float(cv_scores.mean()),
        'cv_r2_std': float(cv_scores.std()),
        'feature_importance': dict(zip(feature_cols, best_model.feature_importances_)),
    }
    
    results_file = RESULTS_DIR / f"ultimate_final_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\n✅ モデルを保存しました: {model_dir}")
    print(f"✅ 結果を保存しました: {results_file}")
    
    print("\n" + "=" * 80)
    print("✅ 究極の最適化が完了しました！")
    print("=" * 80)

if __name__ == "__main__":
    main()
