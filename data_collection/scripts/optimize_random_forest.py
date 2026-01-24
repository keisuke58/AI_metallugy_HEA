#!/usr/bin/env python3
"""
Random Forestモデルの最適化スクリプト

過学習を抑制し、性能を向上させる
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import json
from datetime import datetime

from sklearn.ensemble import RandomForestRegressor
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

def optimize_random_forest():
    """
    Random Forestモデルを最適化
    """
    print("=" * 60)
    print("Random Forest 最適化")
    print("=" * 60)
    
    # データを読み込む
    X, y, feature_cols = load_data()
    print(f"✅ {len(X)}行のデータを読み込みました")
    print(f"📊 特徴量数: {len(feature_cols)}個")
    
    # データを分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\n📊 データ分割:")
    print(f"   訓練データ: {len(X_train)}サンプル")
    print(f"   テストデータ: {len(X_test)}サンプル")
    
    # グリッドサーチのパラメータ
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 7, 10, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }
    
    print("\n" + "=" * 60)
    print("グリッドサーチ開始")
    print("=" * 60)
    print("⚠️  これは時間がかかります（10-30分程度）...")
    
    # ベースモデル
    base_model = RandomForestRegressor(random_state=42)
    
    # グリッドサーチ（交差検証付き）
    grid_search = GridSearchCV(
        base_model,
        param_grid,
        cv=5,
        scoring='r2',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print("\n" + "=" * 60)
    print("最適化完了")
    print("=" * 60)
    
    # 最適なパラメータ
    print(f"\n⭐ 最適なパラメータ:")
    for param, value in grid_search.best_params_.items():
        print(f"   {param}: {value}")
    
    # 最適モデルで予測
    best_model = grid_search.best_estimator_
    
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)
    
    # 性能評価
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    
    print(f"\n📊 性能評価:")
    print(f"   訓練データ:")
    print(f"     R²: {train_r2:.4f}")
    print(f"     RMSE: {train_rmse:.2f} GPa")
    print(f"     MAE: {train_mae:.2f} GPa")
    print(f"   テストデータ:")
    print(f"     R²: {test_r2:.4f}")
    print(f"     RMSE: {test_rmse:.2f} GPa")
    print(f"     MAE: {test_mae:.2f} GPa")
    
    # 過学習の確認
    overfitting = train_r2 - test_r2
    print(f"\n📊 過学習の指標:")
    print(f"   Train R² - Test R²: {overfitting:.4f}")
    if overfitting > 0.2:
        print("   ⚠️  過学習の可能性があります")
    else:
        print("   ✅ 過学習は抑制されています")
    
    # 交差検証スコア
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='r2')
    print(f"\n📊 交差検証スコア（5-fold CV）:")
    print(f"   平均 R²: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # 特徴量重要度
    feature_importance = dict(zip(feature_cols, best_model.feature_importances_))
    sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n📊 特徴量重要度（上位10）:")
    for i, (feature, importance) in enumerate(sorted_importance[:10], 1):
        print(f"   {i}. {feature}: {importance:.4f}")
    
    # モデルを保存
    model_file = MODELS_DIR / "model_RF_optimized.pkl"
    with open(model_file, 'wb') as f:
        pickle.dump(best_model, f)
    print(f"\n✅ 最適化されたモデルを保存しました: {model_file}")
    
    # 結果を保存
    results = {
        'best_params': grid_search.best_params_,
        'train_r2': float(train_r2),
        'test_r2': float(test_r2),
        'train_rmse': float(train_rmse),
        'test_rmse': float(test_rmse),
        'train_mae': float(train_mae),
        'test_mae': float(test_mae),
        'cv_mean_r2': float(cv_scores.mean()),
        'cv_std_r2': float(cv_scores.std()),
        'overfitting': float(overfitting),
        'feature_importance': feature_importance
    }
    
    results_file = RESULTS_DIR / "rf_optimization_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✅ 結果を保存しました: {results_file}")
    
    # 元のモデルと比較
    try:
        with open(MODELS_DIR / "model_RF.pkl", 'rb') as f:
            original_model = pickle.load(f)
        
        original_y_pred_test = original_model.predict(X_test)
        original_test_r2 = r2_score(y_test, original_y_pred_test)
        original_test_rmse = np.sqrt(mean_squared_error(y_test, original_y_pred_test))
        
        print(f"\n" + "=" * 60)
        print("改善の比較")
        print("=" * 60)
        print(f"元のモデル:")
        print(f"   Test R²: {original_test_r2:.4f}")
        print(f"   Test RMSE: {original_test_rmse:.2f} GPa")
        print(f"最適化後:")
        print(f"   Test R²: {test_r2:.4f} ({test_r2 - original_test_r2:+.4f})")
        print(f"   Test RMSE: {test_rmse:.2f} GPa ({test_rmse - original_test_rmse:+.2f} GPa)")
        
        improvement_r2 = test_r2 - original_test_r2
        improvement_rmse = original_test_rmse - test_rmse
        
        if improvement_r2 > 0:
            print(f"\n✅ R²が {improvement_r2:.4f} 改善しました！")
        if improvement_rmse > 0:
            print(f"✅ RMSEが {improvement_rmse:.2f} GPa 改善しました！")
        
    except FileNotFoundError:
        print("\n⚠️  元のモデルが見つかりませんでした（比較スキップ）")
    
    return best_model, results

if __name__ == "__main__":
    optimize_random_forest()
