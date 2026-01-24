#!/usr/bin/env python3
"""
実測データのみで圧倒的な成果を達成するための究極の最適化スクリプト

1. より詳細なハイパーパラメータ探索
2. 高度な特徴量エンジニアリング
3. スタッキングアンサンブル
4. 交差検証ベースの最適化
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
from sklearn.preprocessing import RobustScaler, StandardScaler, PowerTransformer, QuantileTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, StackingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, make_scorer
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression, RFE, SelectFromModel
import xgboost as xgb
from scipy.stats import randint, uniform
from scipy import stats

# 設定
BASE_DIR = Path(__file__).parent.parent
PROCESSED_DATA_DIR = BASE_DIR / "processed_data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def load_and_prepare_data():
    """データを読み込んで準備"""
    data_file = PROCESSED_DATA_DIR / "data_preprocessed.csv"
    
    if not data_file.exists():
        print(f"❌ ファイルが見つかりません: {data_file}")
        return None, None, None
    
    df = pd.read_csv(data_file)
    
    # 実測データのみを抽出
    measured_sources = ['DOE/OSTI', 'Gorsse Dataset', 'Latest Research']
    df_measured = df[df['source'].isin(measured_sources)].copy()
    
    print(f"✅ 実測データ: {len(df_measured)}行")
    
    # 特徴量を選択
    exclude_cols = ['alloy_name', 'elastic_modulus', 'source', 'phases']
    feature_cols = []
    for col in df_measured.columns:
        if col in exclude_cols:
            continue
        if df_measured[col].dtype in [np.float64, np.int64]:
            if df_measured[col].notna().sum() / len(df_measured) >= 0.7:
                feature_cols.append(col)
    
    X = df_measured[feature_cols].copy()
    y = df_measured['elastic_modulus'].values
    
    # 欠損値処理
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=feature_cols)
    
    return X_imputed, y, imputer, feature_cols

def create_advanced_features(X):
    """高度な特徴量を作成"""
    X_advanced = X.copy()
    
    # 既存の特徴量から新しい特徴量を作成
    # 1. 比率特徴量
    if 'mean_atomic_radius' in X.columns and 'std_atomic_radius' in X.columns:
        X_advanced['radius_cv'] = X['std_atomic_radius'] / (X['mean_atomic_radius'] + 1e-10)
    
    if 'mean_electronegativity' in X.columns and 'std_electronegativity' in X.columns:
        X_advanced['electronegativity_cv'] = X['std_electronegativity'] / (X['mean_electronegativity'] + 1e-10)
    
    # 2. 相互作用特徴量
    if 'mixing_entropy' in X.columns and 'mixing_enthalpy' in X.columns:
        X_advanced['entropy_enthalpy_ratio'] = X['mixing_entropy'] / (np.abs(X['mixing_enthalpy']) + 1e-10)
    
    if 'vec' in X.columns and 'num_elements' in X.columns:
        X_advanced['vec_per_element'] = X['vec'] / (X['num_elements'] + 1e-10)
    
    # 3. 多項式特徴量（重要な特徴量のみ）
    important_features = ['mixing_entropy', 'vec', 'mean_atomic_radius', 'delta_r']
    for feat in important_features:
        if feat in X.columns:
            X_advanced[f'{feat}_squared'] = X[feat] ** 2
            X_advanced[f'{feat}_sqrt'] = np.sqrt(np.abs(X[feat]))
    
    return X_advanced

def optimize_with_cv(X, y, model_class, param_distributions, n_iter=100, cv=5):
    """交差検証ベースの最適化"""
    random_search = RandomizedSearchCV(
        model_class, param_distributions, n_iter=n_iter, cv=cv, 
        scoring='r2', n_jobs=-1, random_state=42, verbose=0
    )
    random_search.fit(X, y)
    return random_search.best_estimator_, random_search.best_params_, random_search.best_score_

def optimize_gradient_boosting_advanced(X_train, y_train, X_test, y_test):
    """高度に最適化されたGradient Boosting"""
    print("\n🔧 Gradient Boosting 高度最適化中...")
    
    param_distributions = {
        'n_estimators': randint(300, 1000),
        'max_depth': randint(3, 10),
        'learning_rate': uniform(0.005, 0.05),
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 10),
        'subsample': uniform(0.6, 0.4),
        'max_features': ['sqrt', 'log2', None]
    }
    
    gb = GradientBoostingRegressor(random_state=42)
    best_model, best_params, best_cv_score = optimize_with_cv(
        X_train, y_train, gb, param_distributions, n_iter=100, cv=5
    )
    
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)
    
    result = {
        'model': best_model,
        'best_params': best_params,
        'best_cv_score': best_cv_score,
        'train_r2': r2_score(y_train, y_pred_train),
        'test_r2': r2_score(y_test, y_pred_test),
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'train_mae': mean_absolute_error(y_train, y_pred_train),
        'test_mae': mean_absolute_error(y_test, y_pred_test),
        'feature_importance': dict(zip(X_train.columns, best_model.feature_importances_)),
    }
    
    print(f"   ✅ CV R²: {best_cv_score:.4f}")
    print(f"   ✅ Test R²: {result['test_r2']:.4f}, Test RMSE: {result['test_rmse']:.2f} GPa")
    
    return result

def optimize_xgboost_advanced(X_train, y_train, X_test, y_test):
    """高度に最適化されたXGBoost"""
    print("\n🔧 XGBoost 高度最適化中...")
    
    param_distributions = {
        'n_estimators': randint(300, 1000),
        'max_depth': randint(3, 10),
        'learning_rate': uniform(0.005, 0.05),
        'subsample': uniform(0.6, 0.4),
        'colsample_bytree': uniform(0.6, 0.4),
        'reg_alpha': uniform(0, 1),
        'reg_lambda': uniform(0.5, 2),
        'gamma': uniform(0, 1)
    }
    
    xgb_model = xgb.XGBRegressor(random_state=42, n_jobs=-1)
    best_model, best_params, best_cv_score = optimize_with_cv(
        X_train, y_train, xgb_model, param_distributions, n_iter=100, cv=5
    )
    
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)
    
    result = {
        'model': best_model,
        'best_params': best_params,
        'best_cv_score': best_cv_score,
        'train_r2': r2_score(y_train, y_pred_train),
        'test_r2': r2_score(y_test, y_pred_test),
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'train_mae': mean_absolute_error(y_train, y_pred_train),
        'test_mae': mean_absolute_error(y_test, y_pred_test),
        'feature_importance': dict(zip(X_train.columns, best_model.feature_importances_)),
    }
    
    print(f"   ✅ CV R²: {best_cv_score:.4f}")
    print(f"   ✅ Test R²: {result['test_r2']:.4f}, Test RMSE: {result['test_rmse']:.2f} GPa")
    
    return result

def optimize_random_forest_advanced(X_train, y_train, X_test, y_test):
    """高度に最適化されたRandom Forest"""
    print("\n🔧 Random Forest 高度最適化中...")
    
    param_distributions = {
        'n_estimators': randint(300, 1000),
        'max_depth': [10, 15, 20, 25, None],
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 10),
        'max_features': ['sqrt', 'log2', None],
        'bootstrap': [True, False]
    }
    
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    best_model, best_params, best_cv_score = optimize_with_cv(
        X_train, y_train, rf, param_distributions, n_iter=100, cv=5
    )
    
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)
    
    result = {
        'model': best_model,
        'best_params': best_params,
        'best_cv_score': best_cv_score,
        'train_r2': r2_score(y_train, y_pred_train),
        'test_r2': r2_score(y_test, y_pred_test),
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'train_mae': mean_absolute_error(y_train, y_pred_train),
        'test_mae': mean_absolute_error(y_test, y_pred_test),
        'feature_importance': dict(zip(X_train.columns, best_model.feature_importances_)),
    }
    
    print(f"   ✅ CV R²: {best_cv_score:.4f}")
    print(f"   ✅ Test R²: {result['test_r2']:.4f}, Test RMSE: {result['test_rmse']:.2f} GPa")
    
    return result

def optimize_extra_trees(X_train, y_train, X_test, y_test):
    """Extra Trees Regressorの最適化"""
    print("\n🔧 Extra Trees 最適化中...")
    
    param_distributions = {
        'n_estimators': randint(300, 1000),
        'max_depth': [10, 15, 20, 25, None],
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 10),
        'max_features': ['sqrt', 'log2', None]
    }
    
    et = ExtraTreesRegressor(random_state=42, n_jobs=-1)
    best_model, best_params, best_cv_score = optimize_with_cv(
        X_train, y_train, et, param_distributions, n_iter=50, cv=5
    )
    
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)
    
    result = {
        'model': best_model,
        'best_params': best_params,
        'best_cv_score': best_cv_score,
        'train_r2': r2_score(y_train, y_pred_train),
        'test_r2': r2_score(y_test, y_pred_test),
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'train_mae': mean_absolute_error(y_train, y_pred_train),
        'test_mae': mean_absolute_error(y_test, y_pred_test),
        'feature_importance': dict(zip(X_train.columns, best_model.feature_importances_)),
    }
    
    print(f"   ✅ CV R²: {best_cv_score:.4f}")
    print(f"   ✅ Test R²: {result['test_r2']:.4f}, Test RMSE: {result['test_rmse']:.2f} GPa")
    
    return result

def create_stacking_ensemble(models_dict, X_train, y_train, X_test, y_test):
    """スタッキングアンサンブルを作成"""
    print("\n🔧 スタッキングアンサンブル作成中...")
    
    # 最良の3-4つのモデルを選択
    sorted_models = sorted(models_dict.items(), key=lambda x: x[1]['test_r2'], reverse=True)
    top_models = sorted_models[:min(4, len(sorted_models))]
    
    print(f"   📊 使用モデル: {[name for name, _ in top_models]}")
    
    estimators = []
    for name, result in top_models:
        if 'scaler' in result:
            class ScaledModel:
                def __init__(self, model, scaler):
                    self.model = model
                    self.scaler = scaler
                def predict(self, X):
                    return self.model.predict(self.scaler.transform(X))
            estimators.append((name, ScaledModel(result['model'], result['scaler'])))
        else:
            estimators.append((name, result['model']))
    
    # メタ学習器としてElasticNetを使用
    meta_learner = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000)
    
    stacking = StackingRegressor(
        estimators=estimators,
        final_estimator=meta_learner,
        cv=5,
        n_jobs=-1
    )
    
    stacking.fit(X_train, y_train)
    
    y_pred_train = stacking.predict(X_train)
    y_pred_test = stacking.predict(X_test)
    
    result = {
        'model': stacking,
        'train_r2': r2_score(y_train, y_pred_train),
        'test_r2': r2_score(y_test, y_pred_test),
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'train_mae': mean_absolute_error(y_train, y_pred_train),
        'test_mae': mean_absolute_error(y_test, y_pred_test),
        'base_models': [name for name, _ in top_models],
    }
    
    print(f"   ✅ Test R²: {result['test_r2']:.4f}, Test RMSE: {result['test_rmse']:.2f} GPa")
    
    return result

def main():
    """メイン関数"""
    print("=" * 80)
    print("実測データのみで圧倒的な成果を達成するための究極の最適化")
    print("=" * 80)
    
    # データを読み込む
    X, y, imputer, feature_cols = load_and_prepare_data()
    if X is None:
        return
    
    # 高度な特徴量を作成
    print("\n📊 高度な特徴量を作成中...")
    X_advanced = create_advanced_features(X)
    print(f"   ✅ 特徴量数: {len(X.columns)} → {len(X_advanced.columns)}")
    
    # データ分割
    X_train, X_test, y_train, y_test = train_test_split(
        X_advanced, y, test_size=0.2, random_state=42
    )
    
    print(f"\n📊 データ分割:")
    print(f"   訓練データ: {len(X_train)}サンプル")
    print(f"   テストデータ: {len(X_test)}サンプル")
    print(f"   特徴量数: {len(X_advanced.columns)}個")
    
    # 特徴量選択（重要度ベース）
    print("\n📊 特徴量選択中...")
    rf_selector = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf_selector.fit(X_train, y_train)
    
    importances = rf_selector.feature_importances_
    threshold = np.percentile(importances, 70)  # 上位30%
    selector = SelectFromModel(rf_selector, threshold=threshold, prefit=True)
    X_train_selected = pd.DataFrame(
        selector.transform(X_train), 
        columns=X_train.columns[selector.get_support()]
    )
    X_test_selected = pd.DataFrame(
        selector.transform(X_test),
        columns=X_test.columns[selector.get_support()]
    )
    
    print(f"   ✅ 選択された特徴量: {len(X_train_selected.columns)}個 / {len(X_train.columns)}個")
    
    # モデル最適化
    print("\n" + "=" * 80)
    print("高度なモデル最適化")
    print("=" * 80)
    
    models = {}
    results = {}
    
    # 1. Gradient Boosting
    try:
        results['GB_advanced'] = optimize_gradient_boosting_advanced(
            X_train_selected, y_train, X_test_selected, y_test
        )
        models['GB_advanced'] = results['GB_advanced']
    except Exception as e:
        print(f"   ⚠️  Gradient Boosting最適化でエラー: {e}")
    
    # 2. XGBoost
    try:
        results['XGB_advanced'] = optimize_xgboost_advanced(
            X_train_selected, y_train, X_test_selected, y_test
        )
        models['XGB_advanced'] = results['XGB_advanced']
    except Exception as e:
        print(f"   ⚠️  XGBoost最適化でエラー: {e}")
    
    # 3. Random Forest
    try:
        results['RF_advanced'] = optimize_random_forest_advanced(
            X_train_selected, y_train, X_test_selected, y_test
        )
        models['RF_advanced'] = results['RF_advanced']
    except Exception as e:
        print(f"   ⚠️  Random Forest最適化でエラー: {e}")
    
    # 4. Extra Trees
    try:
        results['ET_advanced'] = optimize_extra_trees(
            X_train_selected, y_train, X_test_selected, y_test
        )
        models['ET_advanced'] = results['ET_advanced']
    except Exception as e:
        print(f"   ⚠️  Extra Trees最適化でエラー: {e}")
    
    # スタッキングアンサンブル
    if len(models) >= 2:
        try:
            results['Stacking'] = create_stacking_ensemble(
                models, X_train_selected, y_train, X_test_selected, y_test
            )
            models['Stacking'] = results['Stacking']
        except Exception as e:
            print(f"   ⚠️  スタッキングアンサンブル作成でエラー: {e}")
    
    # 結果表示
    print("\n" + "=" * 80)
    print("究極の最適化結果")
    print("=" * 80)
    
    print(f"\n{'モデル':<20} {'CV R²':<12} {'Test R²':<12} {'Test RMSE':<12} {'Test MAE':<12}")
    print("-" * 80)
    
    for name, result in sorted(results.items(), key=lambda x: x[1]['test_r2'], reverse=True):
        cv_r2 = result.get('best_cv_score', 0)
        print(f"{name:<20} {cv_r2:>10.4f}   {result['test_r2']:>10.4f}   {result['test_rmse']:>10.2f} GPa   {result['test_mae']:>10.2f} GPa")
    
    # 最良モデル
    best_model_name = max(results.items(), key=lambda x: x[1]['test_r2'])[0]
    best_result = results[best_model_name]
    
    print(f"\n⭐ 最良モデル: {best_model_name}")
    print(f"   CV R²: {best_result.get('best_cv_score', 'N/A')}")
    print(f"   Test R²: {best_result['test_r2']:.4f}")
    print(f"   Test RMSE: {best_result['test_rmse']:.2f} GPa")
    print(f"   Test MAE: {best_result['test_mae']:.2f} GPa")
    
    # 結果を保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = MODELS_DIR / f"ultimate_measured_{timestamp}"
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
    
    with open(model_dir / "feature_selector.pkl", 'wb') as f:
        pickle.dump(selector, f)
    
    # 結果をJSONで保存
    results_summary = {
        'timestamp': timestamp,
        'best_model': best_model_name,
        'feature_count_original': len(X.columns),
        'feature_count_advanced': len(X_advanced.columns),
        'feature_count_selected': len(X_train_selected.columns),
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
        if 'best_cv_score' in result:
            results_summary['results'][name]['cv_r2'] = float(result['best_cv_score'])
        if 'best_params' in result:
            results_summary['results'][name]['best_params'] = result['best_params']
        if 'feature_importance' in result:
            importance = result['feature_importance']
            top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
            results_summary['results'][name]['top_features'] = {k: float(v) for k, v in top_features}
    
    results_file = RESULTS_DIR / f"ultimate_measured_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\n✅ モデルを保存しました: {model_dir}")
    print(f"✅ 結果を保存しました: {results_file}")
    
    print("\n" + "=" * 80)
    print("✅ 究極の最適化が完了しました！")
    print("=" * 80)

if __name__ == "__main__":
    main()
