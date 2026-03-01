#!/usr/bin/env python3
"""
包括的なデータ前処理、特徴量エンジニアリング、モデル訓練、性能評価スクリプト

1. データ前処理と特徴量エンジニアリング
2. 実測データのみでモデル訓練
3. 実測+計算データでモデル訓練
4. 性能評価と比較
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re
import pickle
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 機械学習
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, RobustScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.impute import SimpleImputer

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

# 元素の物理的性質（拡張版）
ELEMENT_PROPERTIES = {
    'Ti': {'radius': 147, 'electronegativity': 1.54, 'valence': 4, 'mass': 47.87, 'density': 4.54},
    'Zr': {'radius': 160, 'electronegativity': 1.33, 'valence': 4, 'mass': 91.22, 'density': 6.52},
    'Hf': {'radius': 159, 'electronegativity': 1.3, 'valence': 4, 'mass': 178.49, 'density': 13.31},
    'Nb': {'radius': 146, 'electronegativity': 1.6, 'valence': 5, 'mass': 92.91, 'density': 8.57},
    'Ta': {'radius': 146, 'electronegativity': 1.5, 'valence': 5, 'mass': 180.95, 'density': 16.65},
    'V': {'radius': 134, 'electronegativity': 1.63, 'valence': 5, 'mass': 50.94, 'density': 6.11},
    'Cr': {'radius': 128, 'electronegativity': 1.66, 'valence': 6, 'mass': 52.00, 'density': 7.19},
    'Mo': {'radius': 139, 'electronegativity': 2.16, 'valence': 6, 'mass': 95.94, 'density': 10.22},
    'W': {'radius': 139, 'electronegativity': 2.36, 'valence': 6, 'mass': 183.84, 'density': 19.25},
    'Fe': {'radius': 126, 'electronegativity': 1.83, 'valence': 8, 'mass': 55.85, 'density': 7.87},
    'Co': {'radius': 125, 'electronegativity': 1.88, 'valence': 9, 'mass': 58.93, 'density': 8.90},
    'Ni': {'radius': 124, 'electronegativity': 1.91, 'valence': 10, 'mass': 58.69, 'density': 8.91},
    'Cu': {'radius': 128, 'electronegativity': 1.9, 'valence': 11, 'mass': 63.55, 'density': 8.96},
    'Al': {'radius': 143, 'electronegativity': 1.61, 'valence': 3, 'mass': 26.98, 'density': 2.70},
    'Mn': {'radius': 127, 'electronegativity': 1.55, 'valence': 7, 'mass': 54.94, 'density': 7.21},
    'Si': {'radius': 111, 'electronegativity': 1.9, 'valence': 4, 'mass': 28.09, 'density': 2.33},
    'Sn': {'radius': 145, 'electronegativity': 1.96, 'valence': 4, 'mass': 118.71, 'density': 7.31},
    'Re': {'radius': 137, 'electronegativity': 1.9, 'valence': 7, 'mass': 186.21, 'density': 21.02},
    'Ru': {'radius': 134, 'electronegativity': 2.2, 'valence': 8, 'mass': 101.07, 'density': 12.45},
    'Pd': {'radius': 137, 'electronegativity': 2.2, 'valence': 10, 'mass': 106.42, 'density': 12.02},
}

def parse_composition(composition_str):
    """合金組成文字列を解析して元素と組成比を抽出"""
    if pd.isna(composition_str) or not isinstance(composition_str, str):
        return {}
    
    # 元素記号と数値を抽出
    pattern = r'([A-Z][a-z]?)(\d*\.?\d*)'
    matches = re.findall(pattern, composition_str)
    
    elements = {}
    total = 0
    
    for element, value_str in matches:
        if element in ELEMENT_PROPERTIES:
            if value_str == '':
                value = 1.0
            else:
                value = float(value_str)
            elements[element] = value
            total += value
    
    # 正規化（合計が1になるように）
    if total > 0:
        elements = {k: v/total for k, v in elements.items()}
    
    return elements

def calculate_advanced_features(composition_dict):
    """高度な材料記述子を計算"""
    if not composition_dict:
        return {}
    
    descriptors = {}
    
    # 原子半径関連
    radii = [ELEMENT_PROPERTIES[elem]['radius'] * comp 
             for elem, comp in composition_dict.items() 
             if elem in ELEMENT_PROPERTIES]
    
    if radii:
        descriptors['mean_atomic_radius'] = np.mean(radii)
        descriptors['std_atomic_radius'] = np.std(radii) if len(radii) > 1 else 0
        descriptors['max_atomic_radius'] = np.max(radii)
        descriptors['min_atomic_radius'] = np.min(radii)
        descriptors['delta_r'] = descriptors['max_atomic_radius'] - descriptors['min_atomic_radius']
        descriptors['radius_range_ratio'] = descriptors['delta_r'] / descriptors['mean_atomic_radius'] if descriptors['mean_atomic_radius'] > 0 else 0
    
    # 電気陰性度関連
    electronegativities = [ELEMENT_PROPERTIES[elem]['electronegativity'] * comp 
                          for elem, comp in composition_dict.items() 
                          if elem in ELEMENT_PROPERTIES]
    
    if electronegativities:
        descriptors['mean_electronegativity'] = np.mean(electronegativities)
        descriptors['std_electronegativity'] = np.std(electronegativities) if len(electronegativities) > 1 else 0
        descriptors['delta_chi'] = np.max(electronegativities) - np.min(electronegativities)
    
    # 価電子濃度（VEC）
    vec_values = [ELEMENT_PROPERTIES[elem]['valence'] * comp 
                 for elem, comp in composition_dict.items() 
                 if elem in ELEMENT_PROPERTIES]
    
    if vec_values:
        descriptors['vec'] = np.sum(vec_values)
    
    # 混合エントロピー（configurational entropy）
    n = len(composition_dict)
    if n > 1:
        R = 8.314  # J/(mol·K)
        entropy = -R * sum(comp * np.log(comp) for comp in composition_dict.values() if comp > 0)
        descriptors['mixing_entropy_calc'] = entropy
    else:
        descriptors['mixing_entropy_calc'] = 0
    
    # 原子質量関連
    masses = [ELEMENT_PROPERTIES[elem]['mass'] * comp 
             for elem, comp in composition_dict.items() 
             if elem in ELEMENT_PROPERTIES]
    
    if masses:
        descriptors['mean_atomic_mass'] = np.mean(masses)
        descriptors['std_atomic_mass'] = np.std(masses) if len(masses) > 1 else 0
    
    # 密度関連（Vegard's law近似）
    densities = [ELEMENT_PROPERTIES[elem]['density'] * comp 
                for elem, comp in composition_dict.items() 
                if elem in ELEMENT_PROPERTIES]
    
    if densities:
        descriptors['estimated_density'] = np.sum(densities)
    
    # 元素数
    descriptors['num_elements'] = len(composition_dict)
    
    # 各元素の組成比
    for elem in ['Ti', 'Zr', 'Hf', 'Nb', 'Ta', 'V', 'Cr', 'Mo', 'W', 
                 'Fe', 'Co', 'Ni', 'Cu', 'Al', 'Mn', 'Si', 'Sn', 'Re', 'Ru', 'Pd']:
        descriptors[f'comp_{elem}'] = composition_dict.get(elem, 0.0)
    
    return descriptors

def preprocess_data(df):
    """データ前処理"""
    print("=" * 60)
    print("データ前処理")
    print("=" * 60)
    
    original_len = len(df)
    print(f"📊 元のデータ数: {original_len}")
    
    # 弾性率がNaNの行を除去
    df = df.dropna(subset=['elastic_modulus'])
    print(f"📊 弾性率データあり: {len(df)}行 ({original_len - len(df)}行除去)")
    
    # 異常値の除去（弾性率が0以下または極端に大きい値）
    df = df[(df['elastic_modulus'] > 0) & (df['elastic_modulus'] < 1000)]
    print(f"📊 異常値除去後: {len(df)}行")
    
    # 重複データの除去
    df = df.drop_duplicates(subset=['alloy_name'], keep='first')
    print(f"📊 重複除去後: {len(df)}行")
    
    # 組成を解析して特徴量を追加
    print("\n📊 組成を解析中...")
    compositions = []
    for idx, row in df.iterrows():
        comp = parse_composition(str(row['alloy_name']))
        compositions.append(comp)
    
    # 高度な特徴量を計算
    print("📊 高度な特徴量を計算中...")
    descriptors_list = []
    for comp in compositions:
        desc = calculate_advanced_features(comp)
        descriptors_list.append(desc)
    
    # 記述子をDataFrameに変換
    descriptors_df = pd.DataFrame(descriptors_list)
    
    # 元のデータフレームと結合
    result_df = pd.concat([df.reset_index(drop=True), descriptors_df], axis=1)
    
    # 重複列の除去（mixing_entropyなど）
    result_df = result_df.loc[:, ~result_df.columns.duplicated()]
    
    print(f"\n✅ 前処理完了")
    print(f"📊 最終データ数: {len(result_df)}行")
    print(f"📊 特徴量数: {len(result_df.columns)}個")
    
    return result_df

def select_features(df):
    """特徴量を選択"""
    # 除外する列
    exclude_cols = ['alloy_name', 'elastic_modulus', 'source', 'phases']
    
    # 特徴量を選択（数値型のみ、NaNが少ないもの）
    feature_cols = []
    for col in df.columns:
        if col in exclude_cols:
            continue
        if df[col].dtype in [np.float64, np.int64]:
            # NaNが30%以下の特徴量のみ使用
            if df[col].notna().sum() / len(df) >= 0.7:
                feature_cols.append(col)
    
    return feature_cols

def prepare_data(df, feature_cols):
    """データを準備（欠損値処理、スケーリング）"""
    X = df[feature_cols].copy()
    y = df['elastic_modulus'].values
    
    # 欠損値処理（中央値で補完）
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=feature_cols)
    
    return X_imputed, y, imputer

def train_model(model_name, model_func, X_train, y_train, X_test, y_test):
    """モデルを訓練"""
    try:
        result = model_func(X_train, y_train, X_test, y_test)
        if result is None:
            return None
        
        # 交差検証スコアを計算（MLFFNNはスキップ）
        if model_name != 'MLFFNN':
            try:
                model_for_cv = result['model'] if 'model' in result else result
                if hasattr(model_for_cv, 'predict'):
                    cv_scores = cross_val_score(
                        model_for_cv,
                        X_train, y_train, 
                        cv=min(5, len(X_train) // 3), scoring='r2', n_jobs=-1
                    )
                    result['cv_r2_mean'] = cv_scores.mean()
                    result['cv_r2_std'] = cv_scores.std()
                else:
                    result['cv_r2_mean'] = 0
                    result['cv_r2_std'] = 0
            except Exception as cv_error:
                result['cv_r2_mean'] = 0
                result['cv_r2_std'] = 0
        else:
            # MLFFNNの場合は手動で交差検証
            kf = KFold(n_splits=min(5, len(X_train) // 3), shuffle=True, random_state=42)
            cv_scores = []
            for train_idx, val_idx in kf.split(X_train):
                X_train_cv, X_val_cv = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_train_cv, y_val_cv = y_train[train_idx], y_train[val_idx]
                
                # スケーラーを適用
                scaler = RobustScaler()
                X_train_cv_scaled = scaler.fit_transform(X_train_cv)
                X_val_cv_scaled = scaler.transform(X_val_cv)
                
                # モデルを訓練
                model_cv = keras.Sequential([
                    layers.Dense(128, activation='relu', input_shape=(X_train_cv_scaled.shape[1],)),
                    layers.Dropout(0.3),
                    layers.Dense(64, activation='relu'),
                    layers.Dropout(0.3),
                    layers.Dense(32, activation='relu'),
                    layers.Dense(1)
                ])
                model_cv.compile(optimizer='adam', loss='mse', metrics=['mae'])
                model_cv.fit(X_train_cv_scaled, y_train_cv, epochs=50, batch_size=32, verbose=0)
                
                y_pred_cv = model_cv.predict(X_val_cv_scaled, verbose=0).flatten()
                cv_scores.append(r2_score(y_val_cv, y_pred_cv))
            
            result['cv_r2_mean'] = np.mean(cv_scores) if cv_scores else 0
            result['cv_r2_std'] = np.std(cv_scores) if cv_scores else 0
        
        return result
    except Exception as e:
        print(f"   ⚠️  エラー: {e}")
        return None

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
    model = Lasso(alpha=0.1, max_iter=10000, random_state=42)
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
    model = Ridge(alpha=1.0, max_iter=10000, random_state=42)
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

def train_random_forest(X_train, y_train, X_test, y_test):
    """Random Forest"""
    model = RandomForestRegressor(
        n_estimators=200, 
        max_depth=15, 
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
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

def train_gradient_boosting(X_train, y_train, X_test, y_test):
    """Gradient Boosting"""
    model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
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
    scaler = RobustScaler()
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
    
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # モデル構築
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(16, activation='relu'),
        layers.Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    # Early stopping
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=20, restore_best_weights=True
    )
    
    # 訓練
    history = model.fit(
        X_train_scaled, y_train,
        epochs=200,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
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

def train_all_models(X_train, y_train, X_test, y_test):
    """すべてのモデルを訓練"""
    models = {}
    results = {}
    
    model_configs = [
        ('LIN', train_linear_regression),
        ('L', train_lasso_regression),
        ('R', train_ridge_regression),
        ('RF', train_random_forest),
        ('GB', train_gradient_boosting),
        ('SVR', train_svr),
    ]
    
    if TENSORFLOW_AVAILABLE:
        model_configs.append(('MLFFNN', train_mlffnn))
    
    for name, model_func in model_configs:
        print(f"\n{name} 訓練中...")
        result = train_model(name, model_func, X_train, y_train, X_test, y_test)
        if result is not None:
            results[name] = result
            models[name] = result
            print(f"   ✅ 完了 - Test R²: {result['test_r2']:.4f}, Test RMSE: {result['test_rmse']:.2f} GPa")
        else:
            print(f"   ❌ 失敗")
    
    return models, results

def save_results(models, results, feature_cols, data_type, imputer):
    """結果を保存"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # モデルを保存
    model_dir = MODELS_DIR / f"{data_type}_{timestamp}"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    for name, model_data in models.items():
        if model_data is None:
            continue
        
        if name == 'SVR' or name == 'MLFFNN':
            with open(model_dir / f"model_{name}.pkl", 'wb') as f:
                pickle.dump(model_data['model'], f)
            with open(model_dir / f"scaler_{name}.pkl", 'wb') as f:
                pickle.dump(model_data['scaler'], f)
        elif name == 'MLFFNN':
            model_data['model'].save(model_dir / f"model_{name}.h5")
            with open(model_dir / f"scaler_{name}.pkl", 'wb') as f:
                pickle.dump(model_data['scaler'], f)
        else:
            with open(model_dir / f"model_{name}.pkl", 'wb') as f:
                pickle.dump(model_data['model'], f)
    
    # Imputerを保存
    with open(model_dir / "imputer.pkl", 'wb') as f:
        pickle.dump(imputer, f)
    
    # 結果をJSONで保存
    results_summary = {
        'data_type': data_type,
        'timestamp': timestamp,
        'feature_count': len(feature_cols),
        'features': feature_cols,
        'models': {}
    }
    
    for name, result in results.items():
        if result is None:
            continue
        results_summary['models'][name] = {
            'test_r2': float(result['test_r2']),
            'test_rmse': float(result['test_rmse']),
            'test_mae': float(result['test_mae']),
            'train_r2': float(result['train_r2']),
            'train_rmse': float(result['train_rmse']),
            'train_mae': float(result['train_mae']),
            'cv_r2_mean': float(result.get('cv_r2_mean', 0)),
            'cv_r2_std': float(result.get('cv_r2_std', 0)),
        }
        if 'feature_importance' in result:
            results_summary['models'][name]['feature_importance'] = result['feature_importance']
    
    results_file = RESULTS_DIR / f"results_{data_type}_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\n✅ モデルを保存しました: {model_dir}")
    print(f"✅ 結果を保存しました: {results_file}")
    
    return results_file

def print_results(results, data_type):
    """結果を表示"""
    print("\n" + "=" * 60)
    print(f"モデル性能比較 ({data_type})")
    print("=" * 60)
    
    print(f"\n{'モデル':<10} {'Test R²':<12} {'Test RMSE':<12} {'Test MAE':<12} {'CV R²':<12}")
    print("-" * 70)
    
    for name, result in results.items():
        if result is None:
            continue
        cv_r2 = f"{result.get('cv_r2_mean', 0):.4f}±{result.get('cv_r2_std', 0):.4f}"
        print(f"{name:<10} {result['test_r2']:>10.4f}   {result['test_rmse']:>10.2f} GPa   {result['test_mae']:>10.2f} GPa   {cv_r2:>12}")
    
    # 最良モデルを特定
    valid_results = [(name, result) for name, result in results.items() if result is not None]
    if valid_results:
        best_model = max(valid_results, key=lambda x: x[1]['test_r2'])
        
        print(f"\n⭐ 最良モデル: {best_model[0]}")
        print(f"   Test R²: {best_model[1]['test_r2']:.4f}")
        print(f"   Test RMSE: {best_model[1]['test_rmse']:.2f} GPa")
        print(f"   Test MAE: {best_model[1]['test_mae']:.2f} GPa")
        print(f"   CV R²: {best_model[1].get('cv_r2_mean', 0):.4f} ± {best_model[1].get('cv_r2_std', 0):.4f}")

def compare_results(results_measured, results_combined):
    """実測データのみと実測+計算データの結果を比較"""
    print("\n" + "=" * 80)
    print("実測データのみ vs 実測+計算データ 比較")
    print("=" * 80)
    
    # 共通のモデルを比較
    common_models = set(results_measured.keys()) & set(results_combined.keys())
    
    print(f"\n{'モデル':<10} {'実測のみ R²':<15} {'実測+計算 R²':<15} {'改善':<10}")
    print("-" * 60)
    
    for model in sorted(common_models):
        if results_measured[model] is None or results_combined[model] is None:
            continue
        
        r2_measured = results_measured[model]['test_r2']
        r2_combined = results_combined[model]['test_r2']
        improvement = r2_combined - r2_measured
        
        print(f"{model:<10} {r2_measured:>13.4f}   {r2_combined:>13.4f}   {improvement:>+8.4f}")
    
    # 最良モデルを比較
    valid_measured = [(n, r) for n, r in results_measured.items() if r is not None]
    valid_combined = [(n, r) for n, r in results_combined.items() if r is not None]
    
    if valid_measured and valid_combined:
        best_measured = max(valid_measured, key=lambda x: x[1]['test_r2'])
        best_combined = max(valid_combined, key=lambda x: x[1]['test_r2'])
        
        print(f"\n最良モデル（実測のみ）: {best_measured[0]} (R² = {best_measured[1]['test_r2']:.4f})")
        print(f"最良モデル（実測+計算）: {best_combined[0]} (R² = {best_combined[1]['test_r2']:.4f})")

def main():
    """メイン関数"""
    print("=" * 80)
    print("包括的なデータ前処理、特徴量エンジニアリング、モデル訓練、性能評価")
    print("=" * 80)
    
    # データを読み込む
    input_file = PROCESSED_DATA_DIR / "integrated_data.csv"
    if not input_file.exists():
        print(f"❌ ファイルが見つかりません: {input_file}")
        print("   先にデータ統合を実行してください: python scripts/integrate_data.py")
        return
    
    df = pd.read_csv(input_file)
    print(f"\n✅ {len(df)}行のデータを読み込みました")
    
    # データ前処理と特徴量エンジニアリング
    df_processed = preprocess_data(df)
    
    # 処理済みデータを保存
    output_file = PROCESSED_DATA_DIR / "data_preprocessed.csv"
    df_processed.to_csv(output_file, index=False)
    print(f"\n✅ 処理済みデータを保存しました: {output_file}")
    
    # データソースで分類
    measured_sources = ['DOE/OSTI', 'Gorsse Dataset', 'Latest Research']
    df_measured = df_processed[df_processed['source'].isin(measured_sources)].copy()
    df_combined = df_processed.copy()
    
    print(f"\n📊 データ分類:")
    print(f"   実測データのみ: {len(df_measured)}行")
    print(f"   実測+計算データ: {len(df_combined)}行")
    
    # 特徴量を選択
    feature_cols_measured = select_features(df_measured)
    feature_cols_combined = select_features(df_combined)
    
    print(f"\n📊 特徴量数:")
    print(f"   実測データのみ: {len(feature_cols_measured)}個")
    print(f"   実測+計算データ: {len(feature_cols_combined)}個")
    
    # 実測データのみでモデル訓練
    print("\n" + "=" * 80)
    print("実測データのみでモデル訓練")
    print("=" * 80)
    
    if len(df_measured) >= 20:  # 最小データ数チェック
        X_measured, y_measured, imputer_measured = prepare_data(df_measured, feature_cols_measured)
        X_train_measured, X_test_measured, y_train_measured, y_test_measured = train_test_split(
            X_measured, y_measured, test_size=0.2, random_state=42
        )
        
        print(f"\n📊 データ分割:")
        print(f"   訓練データ: {len(X_train_measured)}サンプル")
        print(f"   テストデータ: {len(X_test_measured)}サンプル")
        
        models_measured, results_measured = train_all_models(
            X_train_measured, y_train_measured, X_test_measured, y_test_measured
        )
        print_results(results_measured, "実測データのみ")
        save_results(models_measured, results_measured, feature_cols_measured, "measured", imputer_measured)
    else:
        print(f"⚠️  実測データが少なすぎます（{len(df_measured)}行）。モデル訓練をスキップします。")
        results_measured = {}
    
    # 実測+計算データでモデル訓練
    print("\n" + "=" * 80)
    print("実測+計算データでモデル訓練")
    print("=" * 80)
    
    if len(df_combined) >= 20:  # 最小データ数チェック
        X_combined, y_combined, imputer_combined = prepare_data(df_combined, feature_cols_combined)
        X_train_combined, X_test_combined, y_train_combined, y_test_combined = train_test_split(
            X_combined, y_combined, test_size=0.2, random_state=42
        )
        
        print(f"\n📊 データ分割:")
        print(f"   訓練データ: {len(X_train_combined)}サンプル")
        print(f"   テストデータ: {len(X_test_combined)}サンプル")
        
        models_combined, results_combined = train_all_models(
            X_train_combined, y_train_combined, X_test_combined, y_test_combined
        )
        print_results(results_combined, "実測+計算データ")
        save_results(models_combined, results_combined, feature_cols_combined, "combined", imputer_combined)
    else:
        print(f"⚠️  データが少なすぎます（{len(df_combined)}行）。モデル訓練をスキップします。")
        results_combined = {}
    
    # 結果を比較
    if results_measured and results_combined:
        compare_results(results_measured, results_combined)
    
    print("\n" + "=" * 80)
    print("✅ すべての処理が完了しました！")
    print("=" * 80)

if __name__ == "__main__":
    main()
