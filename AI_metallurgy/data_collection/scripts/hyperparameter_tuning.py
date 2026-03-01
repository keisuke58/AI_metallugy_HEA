#!/usr/bin/env python3
"""
すべてのモデルのハイパーパラメータ調整スクリプト

グリッドサーチと交差検証を使用
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import json
from datetime import datetime

from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 設定
BASE_DIR = Path(__file__).parent.parent
PROCESSED_DATA_DIR = BASE_DIR / "processed_data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

def load_data():
    """
    特徴量付きデータを読み込む
    """
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
    
    return X, y, feature_cols

def tune_lasso(X_train, y_train):
    """Lasso Regressionの最適化"""
    param_grid = {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}
    model = Lasso(max_iter=10000)
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_

def tune_ridge(X_train, y_train):
    """Ridge Regressionの最適化"""
    param_grid = {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]}
    model = Ridge(max_iter=10000)
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_

def tune_knn(X_train, y_train):
    """KNNの最適化"""
    param_grid = {
        'n_neighbors': [3, 5, 7, 10, 15, 20],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski']
    }
    model = KNeighborsRegressor()
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_

def tune_svr(X_train, y_train):
    """SVRの最適化"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    param_grid = {
        'C': [0.1, 1, 10, 100, 1000],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
        'epsilon': [0.01, 0.1, 0.5, 1.0]
    }
    model = SVR(kernel='rbf')
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=1)
    grid_search.fit(X_train_scaled, y_train)
    
    return grid_search.best_estimator_, grid_search.best_params_, scaler

def tune_random_forest(X_train, y_train):
    """Random Forestの最適化（簡易版）"""
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [5, 7, 10],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    model = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_

def tune_all_models():
    """
    すべてのモデルを最適化
    """
    print("=" * 60)
    print("すべてのモデルのハイパーパラメータ調整")
    print("=" * 60)
    
    # データを読み込む
    X, y, feature_cols = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"✅ {len(X_train)}行の訓練データ")
    print(f"✅ {len(X_test)}行のテストデータ")
    
    results = {}
    
    # 1. Lasso
    print("\n1. Lasso Regression 最適化中...")
    model, params = tune_lasso(X_train, y_train)
    y_pred = model.predict(X_test)
    results['L'] = {
        'params': params,
        'test_r2': float(r2_score(y_test, y_pred)),
        'test_rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
        'test_mae': float(mean_absolute_error(y_test, y_pred))
    }
    with open(MODELS_DIR / "model_L_optimized.pkl", 'wb') as f:
        pickle.dump(model, f)
    print(f"   ✅ 完了 - Test R²: {results['L']['test_r2']:.4f}")
    
    # 2. Ridge
    print("\n2. Ridge Regression 最適化中...")
    model, params = tune_ridge(X_train, y_train)
    y_pred = model.predict(X_test)
    results['R'] = {
        'params': params,
        'test_r2': float(r2_score(y_test, y_pred)),
        'test_rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
        'test_mae': float(mean_absolute_error(y_test, y_pred))
    }
    with open(MODELS_DIR / "model_R_optimized.pkl", 'wb') as f:
        pickle.dump(model, f)
    print(f"   ✅ 完了 - Test R²: {results['R']['test_r2']:.4f}")
    
    # 3. KNN
    print("\n3. K-Nearest Neighbors 最適化中...")
    model, params = tune_knn(X_train, y_train)
    y_pred = model.predict(X_test)
    results['KNN'] = {
        'params': params,
        'test_r2': float(r2_score(y_test, y_pred)),
        'test_rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
        'test_mae': float(mean_absolute_error(y_test, y_pred))
    }
    with open(MODELS_DIR / "model_KNN_optimized.pkl", 'wb') as f:
        pickle.dump(model, f)
    print(f"   ✅ 完了 - Test R²: {results['KNN']['test_r2']:.4f}")
    
    # 4. SVR
    print("\n4. Support Vector Regression 最適化中...")
    print("   ⚠️  これは時間がかかります...")
    model, params, scaler = tune_svr(X_train, y_train)
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    results['SVR'] = {
        'params': params,
        'test_r2': float(r2_score(y_test, y_pred)),
        'test_rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
        'test_mae': float(mean_absolute_error(y_test, y_pred))
    }
    with open(MODELS_DIR / "model_SVR_optimized.pkl", 'wb') as f:
        pickle.dump(model, f)
    with open(MODELS_DIR / "scaler_SVR_optimized.pkl", 'wb') as f:
        pickle.dump(scaler, f)
    print(f"   ✅ 完了 - Test R²: {results['SVR']['test_r2']:.4f}")
    
    # 5. Random Forest
    print("\n5. Random Forest 最適化中...")
    print("   ⚠️  これは時間がかかります...")
    model, params = tune_random_forest(X_train, y_train)
    y_pred = model.predict(X_test)
    results['RF'] = {
        'params': params,
        'test_r2': float(r2_score(y_test, y_pred)),
        'test_rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
        'test_mae': float(mean_absolute_error(y_test, y_pred))
    }
    with open(MODELS_DIR / "model_RF_optimized.pkl", 'wb') as f:
        pickle.dump(model, f)
    print(f"   ✅ 完了 - Test R²: {results['RF']['test_r2']:.4f}")
    
    # 結果を保存
    results_file = RESULTS_DIR / "hyperparameter_tuning_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ 結果を保存しました: {results_file}")
    
    # 結果を表示
    print("\n" + "=" * 60)
    print("最適化後の性能")
    print("=" * 60)
    print(f"\n{'モデル':<10} {'Test R²':<12} {'Test RMSE':<12}")
    print("-" * 60)
    for name, result in results.items():
        print(f"{name:<10} {result['test_r2']:>10.4f}   {result['test_rmse']:>10.2f} GPa")
    
    best_model = max(results.items(), key=lambda x: x[1]['test_r2'])
    print(f"\n⭐ 最良モデル: {best_model[0]}")
    print(f"   Test R²: {best_model[1]['test_r2']:.4f}")
    print(f"   Test RMSE: {best_model[1]['test_rmse']:.2f} GPa")
    
    return results

if __name__ == "__main__":
    tune_all_models()
