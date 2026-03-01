#!/usr/bin/env python3
"""
8つの機械学習モデルの訓練スクリプト

1. LIN - Linear Regression
2. L - Lasso Regression
3. R - Ridge Regression
4. P - Polynomial Regression
5. KNN - K-Nearest Neighbors
6. RF - Random Forest
7. SVR - Support Vector Regression
8. MLFFNN - Multi-Layer Feedforward Neural Network
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import json
from datetime import datetime

# 機械学習
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 深層学習
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠️  TensorFlowがインストールされていません。MLFFNNモデルはスキップされます。")

# 設定
BASE_DIR = Path(__file__).parent.parent
PROCESSED_DATA_DIR = BASE_DIR / "processed_data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    """
    特徴量付きデータを読み込む
    """
    data_file = PROCESSED_DATA_DIR / "data_with_features.csv"
    
    if not data_file.exists():
        print(f"❌ ファイルが見つかりません: {data_file}")
        print("   先に特徴量エンジニアリングを実行してください: python scripts/feature_engineering.py")
        return None, None
    
    df = pd.read_csv(data_file)
    print(f"✅ {len(df)}行のデータを読み込みました")
    
    # 弾性率がNaNの行を除去
    df = df.dropna(subset=['elastic_modulus'])
    print(f"📊 弾性率データあり: {len(df)}行")
    
    # 特徴量を選択（数値型のみ、NaNが少ないもの）
    feature_cols = []
    for col in df.columns:
        if col in ['alloy_name', 'elastic_modulus', 'source']:
            continue
        if df[col].dtype in [np.float64, np.int64]:
            # NaNが50%以下の特徴量のみ使用
            if df[col].notna().sum() / len(df) >= 0.5:
                feature_cols.append(col)
    
    print(f"📊 使用する特徴量: {len(feature_cols)}個")
    print(f"   特徴量: {feature_cols[:10]}..." if len(feature_cols) > 10 else f"   特徴量: {feature_cols}")
    
    # データを準備
    X = df[feature_cols].fillna(df[feature_cols].median())
    y = df['elastic_modulus'].values
    
    return X, y, feature_cols

def train_linear_regression(X_train, y_train, X_test, y_test):
    """Linear Regression"""
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    return {
        'model': model,
        'train_r2': r2_score(y_train, y_pred_train),
        'test_r2': r2_score(y_test, y_pred_test),
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'train_mae': mean_absolute_error(y_train, y_pred_train),
        'test_mae': mean_absolute_error(y_test, y_pred_test),
    }

def train_lasso_regression(X_train, y_train, X_test, y_test):
    """Lasso Regression"""
    model = Lasso(alpha=0.1, max_iter=10000)
    model.fit(X_train, y_train)
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    return {
        'model': model,
        'train_r2': r2_score(y_train, y_pred_train),
        'test_r2': r2_score(y_test, y_pred_test),
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'train_mae': mean_absolute_error(y_train, y_pred_train),
        'test_mae': mean_absolute_error(y_test, y_pred_test),
    }

def train_ridge_regression(X_train, y_train, X_test, y_test):
    """Ridge Regression"""
    model = Ridge(alpha=1.0, max_iter=10000)
    model.fit(X_train, y_train)
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    return {
        'model': model,
        'train_r2': r2_score(y_train, y_pred_train),
        'test_r2': r2_score(y_test, y_pred_test),
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'train_mae': mean_absolute_error(y_train, y_pred_train),
        'test_mae': mean_absolute_error(y_test, y_pred_test),
    }

def train_polynomial_regression(X_train, y_train, X_test, y_test, degree=2):
    """Polynomial Regression"""
    # データをスケーリング（過学習を防ぐため）
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    poly_features = PolynomialFeatures(degree=degree, include_bias=False)
    X_train_poly = poly_features.fit_transform(X_train_scaled)
    X_test_poly = poly_features.transform(X_test_scaled)
    
    # Ridge正則化を使用して過学習を防ぐ
    model = Ridge(alpha=1.0, max_iter=10000)
    model.fit(X_train_poly, y_train)
    
    y_pred_train = model.predict(X_train_poly)
    y_pred_test = model.predict(X_test_poly)
    
    return {
        'model': model,
        'poly_features': poly_features,
        'scaler': scaler,
        'train_r2': r2_score(y_train, y_pred_train),
        'test_r2': r2_score(y_test, y_pred_test),
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'train_mae': mean_absolute_error(y_train, y_pred_train),
        'test_mae': mean_absolute_error(y_test, y_pred_test),
    }

def train_knn(X_train, y_train, X_test, y_test, n_neighbors=5):
    """K-Nearest Neighbors"""
    model = KNeighborsRegressor(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    return {
        'model': model,
        'train_r2': r2_score(y_train, y_pred_train),
        'test_r2': r2_score(y_test, y_pred_test),
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'train_mae': mean_absolute_error(y_train, y_pred_train),
        'test_mae': mean_absolute_error(y_test, y_pred_test),
    }

def train_random_forest(X_train, y_train, X_test, y_test, n_estimators=100, max_depth=10):
    """Random Forest"""
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    return {
        'model': model,
        'train_r2': r2_score(y_train, y_pred_train),
        'test_r2': r2_score(y_test, y_pred_test),
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'train_mae': mean_absolute_error(y_train, y_pred_train),
        'test_mae': mean_absolute_error(y_test, y_pred_test),
        'feature_importance': dict(zip(X_train.columns, model.feature_importances_)),
    }

def train_svr(X_train, y_train, X_test, y_test):
    """Support Vector Regression"""
    # データをスケーリング
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = SVR(kernel='rbf', C=100, gamma='scale', epsilon=0.1)
    model.fit(X_train_scaled, y_train)
    
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    
    return {
        'model': model,
        'scaler': scaler,
        'train_r2': r2_score(y_train, y_pred_train),
        'test_r2': r2_score(y_test, y_pred_test),
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'train_mae': mean_absolute_error(y_train, y_pred_train),
        'test_mae': mean_absolute_error(y_test, y_pred_test),
    }

def train_mlffnn(X_train, y_train, X_test, y_test):
    """Multi-Layer Feedforward Neural Network"""
    if not TENSORFLOW_AVAILABLE:
        return None
    
    # データをスケーリング
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # モデル構築
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(16, activation='relu'),
        layers.Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    # 訓練
    history = model.fit(
        X_train_scaled, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        verbose=0
    )
    
    y_pred_train = model.predict(X_train_scaled, verbose=0).flatten()
    y_pred_test = model.predict(X_test_scaled, verbose=0).flatten()
    
    return {
        'model': model,
        'scaler': scaler,
        'history': history.history,
        'train_r2': r2_score(y_train, y_pred_train),
        'test_r2': r2_score(y_test, y_pred_test),
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'train_mae': mean_absolute_error(y_train, y_pred_train),
        'test_mae': mean_absolute_error(y_test, y_pred_test),
    }

def train_all_models():
    """
    すべてのモデルを訓練
    """
    print("=" * 60)
    print("機械学習モデルの訓練")
    print("=" * 60)
    
    # データを読み込む
    X, y, feature_cols = load_data()
    if X is None:
        return
    
    # データを分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\n📊 データ分割:")
    print(f"   訓練データ: {len(X_train)}サンプル")
    print(f"   テストデータ: {len(X_test)}サンプル")
    
    # 各モデルを訓練
    models = {}
    results = {}
    
    print("\n" + "=" * 60)
    print("モデル訓練開始")
    print("=" * 60)
    
    # 1. Linear Regression
    print("\n1. Linear Regression 訓練中...")
    results['LIN'] = train_linear_regression(X_train, y_train, X_test, y_test)
    models['LIN'] = results['LIN']['model']
    print(f"   ✅ 完了 - Test R²: {results['LIN']['test_r2']:.4f}, Test RMSE: {results['LIN']['test_rmse']:.2f} GPa")
    
    # 2. Lasso Regression
    print("\n2. Lasso Regression 訓練中...")
    results['L'] = train_lasso_regression(X_train, y_train, X_test, y_test)
    models['L'] = results['L']['model']
    print(f"   ✅ 完了 - Test R²: {results['L']['test_r2']:.4f}, Test RMSE: {results['L']['test_rmse']:.2f} GPa")
    
    # 3. Ridge Regression
    print("\n3. Ridge Regression 訓練中...")
    results['R'] = train_ridge_regression(X_train, y_train, X_test, y_test)
    models['R'] = results['R']['model']
    print(f"   ✅ 完了 - Test R²: {results['R']['test_r2']:.4f}, Test RMSE: {results['R']['test_rmse']:.2f} GPa")
    
    # 4. Polynomial Regression
    print("\n4. Polynomial Regression 訓練中...")
    results['P'] = train_polynomial_regression(X_train, y_train, X_test, y_test, degree=2)
    models['P'] = results['P']
    print(f"   ✅ 完了 - Test R²: {results['P']['test_r2']:.4f}, Test RMSE: {results['P']['test_rmse']:.2f} GPa")
    
    # 5. KNN
    print("\n5. K-Nearest Neighbors 訓練中...")
    results['KNN'] = train_knn(X_train, y_train, X_test, y_test, n_neighbors=5)
    models['KNN'] = results['KNN']['model']
    print(f"   ✅ 完了 - Test R²: {results['KNN']['test_r2']:.4f}, Test RMSE: {results['KNN']['test_rmse']:.2f} GPa")
    
    # 6. Random Forest
    print("\n6. Random Forest 訓練中...")
    results['RF'] = train_random_forest(X_train, y_train, X_test, y_test, n_estimators=100, max_depth=10)
    models['RF'] = results['RF']['model']
    print(f"   ✅ 完了 - Test R²: {results['RF']['test_r2']:.4f}, Test RMSE: {results['RF']['test_rmse']:.2f} GPa")
    
    # 7. SVR
    print("\n7. Support Vector Regression 訓練中...")
    results['SVR'] = train_svr(X_train, y_train, X_test, y_test)
    models['SVR'] = results['SVR']
    print(f"   ✅ 完了 - Test R²: {results['SVR']['test_r2']:.4f}, Test RMSE: {results['SVR']['test_rmse']:.2f} GPa")
    
    # 8. MLFFNN
    if TENSORFLOW_AVAILABLE:
        print("\n8. Multi-Layer Feedforward Neural Network 訓練中...")
        results['MLFFNN'] = train_mlffnn(X_train, y_train, X_test, y_test)
        if results['MLFFNN'] is not None:
            models['MLFFNN'] = results['MLFFNN']
            print(f"   ✅ 完了 - Test R²: {results['MLFFNN']['test_r2']:.4f}, Test RMSE: {results['MLFFNN']['test_rmse']:.2f} GPa")
    else:
        print("\n8. Multi-Layer Feedforward Neural Network スキップ（TensorFlow未インストール）")
        results['MLFFNN'] = None
    
    # 結果を保存
    save_results(models, results, feature_cols, X_test, y_test)
    
    # 結果を表示
    print_results(results)
    
    return models, results

def save_results(models, results, feature_cols, X_test, y_test):
    """
    モデルと結果を保存
    """
    print("\n" + "=" * 60)
    print("結果の保存")
    print("=" * 60)
    
    # モデルを保存
    for name, model_data in models.items():
        if model_data is None:
            continue
        
        if name == 'P':
            # Polynomial Regressionは特別な処理
            with open(MODELS_DIR / f"model_{name}_poly.pkl", 'wb') as f:
                pickle.dump(model_data['poly_features'], f)
            with open(MODELS_DIR / f"model_{name}.pkl", 'wb') as f:
                pickle.dump(model_data['model'], f)
        elif name == 'SVR':
            with open(MODELS_DIR / f"model_{name}.pkl", 'wb') as f:
                pickle.dump(model_data['model'], f)
            with open(MODELS_DIR / f"scaler_{name}.pkl", 'wb') as f:
                pickle.dump(model_data['scaler'], f)
        elif name == 'MLFFNN':
            if model_data is not None:
                model_data['model'].save(MODELS_DIR / f"model_{name}.h5")
                with open(MODELS_DIR / f"scaler_{name}.pkl", 'wb') as f:
                    pickle.dump(model_data['scaler'], f)
        else:
            with open(MODELS_DIR / f"model_{name}.pkl", 'wb') as f:
                pickle.dump(model_data, f)
    
    # 結果をJSONで保存
    results_summary = {}
    for name, result in results.items():
        if result is None:
            continue
        results_summary[name] = {
            'test_r2': float(result['test_r2']),
            'test_rmse': float(result['test_rmse']),
            'test_mae': float(result['test_mae']),
            'train_r2': float(result['train_r2']),
            'train_rmse': float(result['train_rmse']),
        }
        if 'feature_importance' in result:
            results_summary[name]['feature_importance'] = result['feature_importance']
    
    with open(RESULTS_DIR / "model_results.json", 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"✅ モデルを保存しました: {MODELS_DIR}")
    print(f"✅ 結果を保存しました: {RESULTS_DIR / 'model_results.json'}")

def print_results(results):
    """
    結果を表示
    """
    print("\n" + "=" * 60)
    print("モデル性能比較")
    print("=" * 60)
    
    print(f"\n{'モデル':<10} {'Test R²':<12} {'Test RMSE':<12} {'Test MAE':<12}")
    print("-" * 60)
    
    for name, result in results.items():
        if result is None:
            continue
        print(f"{name:<10} {result['test_r2']:>10.4f}   {result['test_rmse']:>10.2f} GPa   {result['test_mae']:>10.2f} GPa")
    
    # 最良モデルを特定
    best_model = max(
        [(name, result) for name, result in results.items() if result is not None],
        key=lambda x: x[1]['test_r2']
    )
    
    print(f"\n⭐ 最良モデル: {best_model[0]}")
    print(f"   Test R²: {best_model[1]['test_r2']:.4f}")
    print(f"   Test RMSE: {best_model[1]['test_rmse']:.2f} GPa")
    print(f"   Test MAE: {best_model[1]['test_mae']:.2f} GPa")

if __name__ == "__main__":
    train_all_models()
