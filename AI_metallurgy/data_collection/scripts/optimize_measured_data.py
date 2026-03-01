#!/usr/bin/env python3
"""
実測データのみで最高性能を達成するための最適化スクリプト

1. ハイパーパラメータ最適化
2. 特徴量選択
3. アンサンブル手法
4. 高度な前処理
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 機械学習
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.preprocessing import RobustScaler, StandardScaler, PowerTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, StackingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression, RFE, SelectFromModel
import xgboost as xgb
from scipy.stats import randint, uniform

# 設定
BASE_DIR = Path(__file__).parent.parent
PROCESSED_DATA_DIR = BASE_DIR / "processed_data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def load_preprocessed_data():
    """前処理済みデータを読み込む"""
    data_file = PROCESSED_DATA_DIR / "data_preprocessed.csv"
    
    if not data_file.exists():
        print(f"❌ ファイルが見つかりません: {data_file}")
        print("   先に前処理を実行してください: python scripts/comprehensive_preprocessing_training.py")
        return None
    
    df = pd.read_csv(data_file)
    print(f"✅ {len(df)}行のデータを読み込みました")
    
    # 実測データのみを抽出
    measured_sources = ['DOE/OSTI', 'Gorsse Dataset', 'Latest Research']
    df_measured = df[df['source'].isin(measured_sources)].copy()
    
    print(f"📊 実測データ: {len(df_measured)}行")
    
    return df_measured

def select_best_features(X, y, method='rf', k=None):
    """最適な特徴量を選択"""
    print(f"\n📊 特徴量選択: {method}")
    
    if method == 'rf':
        # Random Forestベースの特徴量選択
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        
        # 重要度で選択
        importances = rf.feature_importances_
        threshold = np.percentile(importances, 75)  # 上位25%
        selector = SelectFromModel(rf, threshold=threshold, prefit=True)
        X_selected = selector.transform(X)
        selected_features = X.columns[selector.get_support()].tolist()
        
    elif method == 'lasso':
        # Lassoベースの特徴量選択
        lasso = Lasso(alpha=0.1, max_iter=10000, random_state=42)
        lasso.fit(X, y)
        
        selector = SelectFromModel(lasso, prefit=True)
        X_selected = selector.transform(X)
        selected_features = X.columns[selector.get_support()].tolist()
        
    elif method == 'kbest':
        # KBest特徴量選択
        if k is None:
            k = min(30, X.shape[1])
        selector = SelectKBest(f_regression, k=k)
        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()].tolist()
        
    elif method == 'rfe':
        # Recursive Feature Elimination
        if k is None:
            k = min(30, X.shape[1])
        rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        selector = RFE(rf, n_features_to_select=k, step=1)
        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()].tolist()
    
    else:
        # すべての特徴量を使用
        X_selected = X.values
        selected_features = X.columns.tolist()
    
    print(f"   ✅ 選択された特徴量: {len(selected_features)}個 / {len(X.columns)}個")
    
    return pd.DataFrame(X_selected, columns=selected_features), selected_features

def optimize_gradient_boosting(X_train, y_train, X_test, y_test):
    """Gradient Boostingの最適化"""
    print("\n🔧 Gradient Boosting 最適化中...")
    
    param_grid = {
        'n_estimators': [200, 300, 500],
        'max_depth': [5, 7, 10, 15],
        'learning_rate': [0.01, 0.05, 0.1],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'subsample': [0.8, 0.9, 1.0]
    }
    
    gb = GradientBoostingRegressor(random_state=42)
    
    # RandomizedSearchCVで最適化
    random_search = RandomizedSearchCV(
        gb, param_grid, n_iter=50, cv=5, scoring='r2',
        n_jobs=-1, random_state=42, verbose=0
    )
    
    random_search.fit(X_train, y_train)
    
    best_model = random_search.best_estimator_
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)
    
    result = {
        'model': best_model,
        'best_params': random_search.best_params_,
        'train_r2': r2_score(y_train, y_pred_train),
        'test_r2': r2_score(y_test, y_pred_test),
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'train_mae': mean_absolute_error(y_train, y_pred_train),
        'test_mae': mean_absolute_error(y_test, y_pred_test),
        'feature_importance': dict(zip(X_train.columns, best_model.feature_importances_)),
    }
    
    print(f"   ✅ 最適パラメータ: {random_search.best_params_}")
    print(f"   ✅ Test R²: {result['test_r2']:.4f}, Test RMSE: {result['test_rmse']:.2f} GPa")
    
    return result

def optimize_xgboost(X_train, y_train, X_test, y_test):
    """XGBoostの最適化"""
    print("\n🔧 XGBoost 最適化中...")
    
    param_grid = {
        'n_estimators': [200, 300, 500],
        'max_depth': [5, 7, 10],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'reg_alpha': [0, 0.1, 0.5],
        'reg_lambda': [1, 1.5, 2]
    }
    
    xgb_model = xgb.XGBRegressor(random_state=42, n_jobs=-1)
    
    random_search = RandomizedSearchCV(
        xgb_model, param_grid, n_iter=50, cv=5, scoring='r2',
        n_jobs=-1, random_state=42, verbose=0
    )
    
    random_search.fit(X_train, y_train)
    
    best_model = random_search.best_estimator_
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)
    
    result = {
        'model': best_model,
        'best_params': random_search.best_params_,
        'train_r2': r2_score(y_train, y_pred_train),
        'test_r2': r2_score(y_test, y_pred_test),
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'train_mae': mean_absolute_error(y_train, y_pred_train),
        'test_mae': mean_absolute_error(y_test, y_pred_test),
        'feature_importance': dict(zip(X_train.columns, best_model.feature_importances_)),
    }
    
    print(f"   ✅ 最適パラメータ: {random_search.best_params_}")
    print(f"   ✅ Test R²: {result['test_r2']:.4f}, Test RMSE: {result['test_rmse']:.2f} GPa")
    
    return result

def optimize_random_forest(X_train, y_train, X_test, y_test):
    """Random Forestの最適化"""
    print("\n🔧 Random Forest 最適化中...")
    
    param_grid = {
        'n_estimators': [200, 300, 500],
        'max_depth': [10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }
    
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    
    random_search = RandomizedSearchCV(
        rf, param_grid, n_iter=50, cv=5, scoring='r2',
        n_jobs=-1, random_state=42, verbose=0
    )
    
    random_search.fit(X_train, y_train)
    
    best_model = random_search.best_estimator_
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)
    
    result = {
        'model': best_model,
        'best_params': random_search.best_params_,
        'train_r2': r2_score(y_train, y_pred_train),
        'test_r2': r2_score(y_test, y_pred_test),
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'train_mae': mean_absolute_error(y_train, y_pred_train),
        'test_mae': mean_absolute_error(y_test, y_pred_test),
        'feature_importance': dict(zip(X_train.columns, best_model.feature_importances_)),
    }
    
    print(f"   ✅ 最適パラメータ: {random_search.best_params_}")
    print(f"   ✅ Test R²: {result['test_r2']:.4f}, Test RMSE: {result['test_rmse']:.2f} GPa")
    
    return result

def optimize_svr(X_train, y_train, X_test, y_test):
    """SVRの最適化"""
    print("\n🔧 SVR 最適化中...")
    
    # データをスケーリング
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    param_grid = {
        'C': [0.1, 1, 10, 100, 1000],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
        'epsilon': [0.01, 0.1, 0.5, 1.0]
    }
    
    svr = SVR(kernel='rbf')
    
    random_search = RandomizedSearchCV(
        svr, param_grid, n_iter=30, cv=5, scoring='r2',
        n_jobs=-1, random_state=42, verbose=0
    )
    
    random_search.fit(X_train_scaled, y_train)
    
    best_model = random_search.best_estimator_
    y_pred_train = best_model.predict(X_train_scaled)
    y_pred_test = best_model.predict(X_test_scaled)
    
    result = {
        'model': best_model,
        'scaler': scaler,
        'best_params': random_search.best_params_,
        'train_r2': r2_score(y_train, y_pred_train),
        'test_r2': r2_score(y_test, y_pred_test),
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'train_mae': mean_absolute_error(y_train, y_pred_train),
        'test_mae': mean_absolute_error(y_test, y_pred_test),
    }
    
    print(f"   ✅ 最適パラメータ: {random_search.best_params_}")
    print(f"   ✅ Test R²: {result['test_r2']:.4f}, Test RMSE: {result['test_rmse']:.2f} GPa")
    
    return result

def create_ensemble(models_dict, X_train, y_train, X_test, y_test):
    """アンサンブルモデルを作成"""
    print("\n🔧 アンサンブルモデル作成中...")
    
    # 最良の3つのモデルを選択
    sorted_models = sorted(models_dict.items(), key=lambda x: x[1]['test_r2'], reverse=True)
    top_3 = sorted_models[:3]
    
    print(f"   📊 使用モデル: {[name for name, _ in top_3]}")
    
    estimators = []
    for name, result in top_3:
        if 'scaler' in result:
            # SVRなどスケーラーが必要なモデルはラッパーで処理
            class ScaledModel:
                def __init__(self, model, scaler):
                    self.model = model
                    self.scaler = scaler
                def predict(self, X):
                    return self.model.predict(self.scaler.transform(X))
            
            estimators.append((name, ScaledModel(result['model'], result['scaler'])))
        else:
            estimators.append((name, result['model']))
    
    # Voting Regressor
    voting = VotingRegressor(estimators=estimators)
    voting.fit(X_train, y_train)
    
    y_pred_train = voting.predict(X_train)
    y_pred_test = voting.predict(X_test)
    
    result = {
        'model': voting,
        'train_r2': r2_score(y_train, y_pred_train),
        'test_r2': r2_score(y_test, y_pred_test),
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'train_mae': mean_absolute_error(y_train, y_pred_train),
        'test_mae': mean_absolute_error(y_test, y_pred_test),
        'base_models': [name for name, _ in top_3],
    }
    
    print(f"   ✅ Test R²: {result['test_r2']:.4f}, Test RMSE: {result['test_rmse']:.2f} GPa")
    
    return result

def main():
    """メイン関数"""
    print("=" * 80)
    print("実測データのみで最高性能を達成するための最適化")
    print("=" * 80)
    
    # データを読み込む
    df = load_preprocessed_data()
    if df is None:
        return
    
    # 特徴量を選択
    exclude_cols = ['alloy_name', 'elastic_modulus', 'source', 'phases']
    feature_cols = []
    for col in df.columns:
        if col in exclude_cols:
            continue
        if df[col].dtype in [np.float64, np.int64]:
            if df[col].notna().sum() / len(df) >= 0.7:
                feature_cols.append(col)
    
    # データを準備
    X = df[feature_cols].copy()
    y = df['elastic_modulus'].values
    
    # 欠損値処理
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=feature_cols)
    
    # データ分割
    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y, test_size=0.2, random_state=42
    )
    
    print(f"\n📊 データ分割:")
    print(f"   訓練データ: {len(X_train)}サンプル")
    print(f"   テストデータ: {len(X_test)}サンプル")
    print(f"   特徴量数: {len(feature_cols)}個")
    
    # 特徴量選択を試行
    feature_selection_methods = ['rf', 'lasso', 'kbest']
    best_features = None
    best_score = -np.inf
    best_method = None
    
    print("\n" + "=" * 80)
    print("特徴量選択の最適化")
    print("=" * 80)
    
    for method in feature_selection_methods:
        try:
            X_selected, selected_features = select_best_features(X_train, y_train, method=method)
            
            # 簡単なモデルで評価
            rf_temp = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            scores = cross_val_score(rf_temp, X_selected, y_train, cv=5, scoring='r2', n_jobs=-1)
            mean_score = scores.mean()
            
            print(f"   {method}: CV R² = {mean_score:.4f}")
            
            if mean_score > best_score:
                best_score = mean_score
                best_features = selected_features
                best_method = method
        except Exception as e:
            print(f"   ⚠️  {method}でエラー: {e}")
    
    if best_features:
        print(f"\n✅ 最良の特徴量選択方法: {best_method} ({len(best_features)}個の特徴量)")
        X_train_selected, _ = select_best_features(X_train, y_train, method=best_method)
        X_test_selected, _ = select_best_features(X_test, y_test, method=best_method)
        # X_test_selectedは同じ特徴量を使用する必要があるため、手動で選択
        X_test_selected = X_test[best_features]
        X_train_selected = X_train[best_features]
    else:
        X_train_selected = X_train
        X_test_selected = X_test
        best_features = feature_cols
    
    # モデル最適化
    print("\n" + "=" * 80)
    print("モデル最適化")
    print("=" * 80)
    
    models = {}
    results = {}
    
    # 1. Gradient Boosting
    try:
        results['GB_optimized'] = optimize_gradient_boosting(X_train_selected, y_train, X_test_selected, y_test)
        models['GB_optimized'] = results['GB_optimized']
    except Exception as e:
        print(f"   ⚠️  Gradient Boosting最適化でエラー: {e}")
    
    # 2. XGBoost
    try:
        results['XGB_optimized'] = optimize_xgboost(X_train_selected, y_train, X_test_selected, y_test)
        models['XGB_optimized'] = results['XGB_optimized']
    except Exception as e:
        print(f"   ⚠️  XGBoost最適化でエラー: {e}")
    
    # 3. Random Forest
    try:
        results['RF_optimized'] = optimize_random_forest(X_train_selected, y_train, X_test_selected, y_test)
        models['RF_optimized'] = results['RF_optimized']
    except Exception as e:
        print(f"   ⚠️  Random Forest最適化でエラー: {e}")
    
    # 4. SVR
    try:
        results['SVR_optimized'] = optimize_svr(X_train_selected, y_train, X_test_selected, y_test)
        models['SVR_optimized'] = results['SVR_optimized']
    except Exception as e:
        print(f"   ⚠️  SVR最適化でエラー: {e}")
    
    # アンサンブル
    if len(models) >= 2:
        try:
            results['Ensemble'] = create_ensemble(models, X_train_selected, y_train, X_test_selected, y_test)
            models['Ensemble'] = results['Ensemble']
        except Exception as e:
            print(f"   ⚠️  アンサンブル作成でエラー: {e}")
    
    # 結果表示
    print("\n" + "=" * 80)
    print("最適化結果")
    print("=" * 80)
    
    print(f"\n{'モデル':<20} {'Test R²':<12} {'Test RMSE':<12} {'Test MAE':<12}")
    print("-" * 70)
    
    for name, result in sorted(results.items(), key=lambda x: x[1]['test_r2'], reverse=True):
        print(f"{name:<20} {result['test_r2']:>10.4f}   {result['test_rmse']:>10.2f} GPa   {result['test_mae']:>10.2f} GPa")
    
    # 最良モデル
    best_model_name = max(results.items(), key=lambda x: x[1]['test_r2'])[0]
    best_result = results[best_model_name]
    
    print(f"\n⭐ 最良モデル: {best_model_name}")
    print(f"   Test R²: {best_result['test_r2']:.4f}")
    print(f"   Test RMSE: {best_result['test_rmse']:.2f} GPa")
    print(f"   Test MAE: {best_result['test_mae']:.2f} GPa")
    
    # 結果を保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = MODELS_DIR / f"optimized_measured_{timestamp}"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # 最良モデルを保存
    best_model_data = models[best_model_name]
    if 'scaler' in best_model_data:
        with open(model_dir / "best_model.pkl", 'wb') as f:
            pickle.dump(best_model_data['model'], f)
        with open(model_dir / "scaler.pkl", 'wb') as f:
            pickle.dump(best_model_data['scaler'], f)
    else:
        with open(model_dir / "best_model.pkl", 'wb') as f:
            pickle.dump(best_model_data['model'], f)
    
    with open(model_dir / "imputer.pkl", 'wb') as f:
        pickle.dump(imputer, f)
    
    # 結果をJSONで保存
    results_summary = {
        'timestamp': timestamp,
        'best_model': best_model_name,
        'feature_selection_method': best_method,
        'selected_features': best_features,
        'feature_count': len(best_features),
        'results': {}
    }
    
    for name, result in results.items():
        results_summary['results'][name] = {
            'test_r2': float(result['test_r2']),
            'test_rmse': float(result['test_rmse']),
            'test_mae': float(result['test_mae']),
            'train_r2': float(result['train_r2']),
            'train_rmse': float(result['train_rmse']),
            'train_mae': float(result['train_mae']),
        }
        if 'best_params' in result:
            results_summary['results'][name]['best_params'] = result['best_params']
        if 'feature_importance' in result:
            # 上位10個のみ保存
            importance = result['feature_importance']
            top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
            results_summary['results'][name]['top_features'] = {k: float(v) for k, v in top_features}
    
    results_file = RESULTS_DIR / f"optimized_measured_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\n✅ モデルを保存しました: {model_dir}")
    print(f"✅ 結果を保存しました: {results_file}")
    
    print("\n" + "=" * 80)
    print("✅ 最適化が完了しました！")
    print("=" * 80)

if __name__ == "__main__":
    main()
