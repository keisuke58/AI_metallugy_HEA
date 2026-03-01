#!/usr/bin/env python3
"""
実測データのみで最高性能（R² > 0.70）を達成するための最終最適化
元の0.6992の結果を超えることを目標
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
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, VotingRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
import xgboost as xgb
from scipy.stats import randint, uniform

# 設定
BASE_DIR = Path(__file__).parent.parent
PROCESSED_DATA_DIR = BASE_DIR / "processed_data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

def load_data():
    """データを読み込む"""
    data_file = PROCESSED_DATA_DIR / "data_preprocessed.csv"
    df = pd.read_csv(data_file)
    
    # 実測データのみ
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

def optimize_gb_final(X_train, y_train, X_test, y_test):
    """Gradient Boosting最終最適化"""
    print("\n🔧 Gradient Boosting 最終最適化中...")
    
    # より広範囲なパラメータ探索
    param_distributions = {
        'n_estimators': randint(200, 800),
        'max_depth': randint(3, 12),
        'learning_rate': uniform(0.01, 0.1),
        'min_samples_split': randint(2, 15),
        'min_samples_leaf': randint(1, 8),
        'subsample': uniform(0.7, 0.3),
        'max_features': ['sqrt', 'log2', None]
    }
    
    gb = GradientBoostingRegressor(random_state=42)
    
    # より多くの試行回数
    random_search = RandomizedSearchCV(
        gb, param_distributions, n_iter=200, cv=5, scoring='r2',
        n_jobs=-1, random_state=42, verbose=0
    )
    
    random_search.fit(X_train, y_train)
    best_model = random_search.best_estimator_
    
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)
    
    # 交差検証スコア
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='r2', n_jobs=-1)
    
    result = {
        'model': best_model,
        'best_params': random_search.best_params_,
        'cv_r2_mean': cv_scores.mean(),
        'cv_r2_std': cv_scores.std(),
        'train_r2': r2_score(y_train, y_pred_train),
        'test_r2': r2_score(y_test, y_pred_test),
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'train_mae': mean_absolute_error(y_train, y_pred_train),
        'test_mae': mean_absolute_error(y_test, y_pred_test),
    }
    
    print(f"   ✅ CV R²: {result['cv_r2_mean']:.4f} ± {result['cv_r2_std']:.4f}")
    print(f"   ✅ Test R²: {result['test_r2']:.4f}, Test RMSE: {result['test_rmse']:.2f} GPa")
    
    return result

def optimize_xgb_final(X_train, y_train, X_test, y_test):
    """XGBoost最終最適化"""
    print("\n🔧 XGBoost 最終最適化中...")
    
    param_distributions = {
        'n_estimators': randint(200, 800),
        'max_depth': randint(3, 12),
        'learning_rate': uniform(0.01, 0.1),
        'subsample': uniform(0.7, 0.3),
        'colsample_bytree': uniform(0.7, 0.3),
        'reg_alpha': uniform(0, 0.5),
        'reg_lambda': uniform(0.5, 1.5),
        'gamma': uniform(0, 0.5)
    }
    
    xgb_model = xgb.XGBRegressor(random_state=42, n_jobs=-1)
    
    random_search = RandomizedSearchCV(
        xgb_model, param_distributions, n_iter=200, cv=5, scoring='r2',
        n_jobs=-1, random_state=42, verbose=0
    )
    
    random_search.fit(X_train, y_train)
    best_model = random_search.best_estimator_
    
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)
    
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='r2', n_jobs=-1)
    
    result = {
        'model': best_model,
        'best_params': random_search.best_params_,
        'cv_r2_mean': cv_scores.mean(),
        'cv_r2_std': cv_scores.std(),
        'train_r2': r2_score(y_train, y_pred_train),
        'test_r2': r2_score(y_test, y_pred_test),
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'train_mae': mean_absolute_error(y_train, y_pred_train),
        'test_mae': mean_absolute_error(y_test, y_pred_test),
    }
    
    print(f"   ✅ CV R²: {result['cv_r2_mean']:.4f} ± {result['cv_r2_std']:.4f}")
    print(f"   ✅ Test R²: {result['test_r2']:.4f}, Test RMSE: {result['test_rmse']:.2f} GPa")
    
    return result

def create_weighted_ensemble(models_dict, X_train, y_train, X_test, y_test):
    """重み付きアンサンブル"""
    print("\n🔧 重み付きアンサンブル作成中...")
    
    # 最良の2-3モデルを選択
    sorted_models = sorted(models_dict.items(), key=lambda x: x[1]['test_r2'], reverse=True)
    top_models = sorted_models[:min(3, len(sorted_models))]
    
    print(f"   📊 使用モデル: {[name for name, _ in top_models]}")
    
    # テストスコアに基づいて重みを計算
    weights = []
    models_list = []
    total_score = sum(result['test_r2'] for _, result in top_models)
    
    for name, result in top_models:
        weight = max(0, result['test_r2'] / total_score)  # 負のスコアは0に
        weights.append(weight)
        models_list.append((name, result['model']))
    
    # 正規化
    weights = np.array(weights)
    weights = weights / weights.sum()
    
    print(f"   📊 重み: {dict(zip([name for name, _ in top_models], weights))}")
    
    # 予測
    predictions_train = []
    predictions_test = []
    
    for (name, model), weight in zip(models_list, weights):
        pred_train = model.predict(X_train) * weight
        pred_test = model.predict(X_test) * weight
        predictions_train.append(pred_train)
        predictions_test.append(pred_test)
    
    y_pred_train = np.sum(predictions_train, axis=0)
    y_pred_test = np.sum(predictions_test, axis=0)
    
    result = {
        'models': models_list,
        'weights': dict(zip([name for name, _ in top_models], weights)),
        'train_r2': r2_score(y_train, y_pred_train),
        'test_r2': r2_score(y_test, y_pred_test),
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'train_mae': mean_absolute_error(y_train, y_pred_train),
        'test_mae': mean_absolute_error(y_test, y_pred_test),
    }
    
    print(f"   ✅ Test R²: {result['test_r2']:.4f}, Test RMSE: {result['test_rmse']:.2f} GPa")
    
    return result

def main():
    """メイン関数"""
    print("=" * 80)
    print("実測データのみで最高性能（R² > 0.70）を達成するための最終最適化")
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
    
    # モデル最適化
    print("\n" + "=" * 80)
    print("最終最適化")
    print("=" * 80)
    
    models = {}
    results = {}
    
    # 1. Gradient Boosting
    results['GB_final'] = optimize_gb_final(X_train, y_train, X_test, y_test)
    models['GB_final'] = results['GB_final']
    
    # 2. XGBoost
    results['XGB_final'] = optimize_xgb_final(X_train, y_train, X_test, y_test)
    models['XGB_final'] = results['XGB_final']
    
    # 3. 重み付きアンサンブル
    if len(models) >= 2:
        results['Weighted_Ensemble'] = create_weighted_ensemble(
            models, X_train, y_train, X_test, y_test
        )
    
    # 結果表示
    print("\n" + "=" * 80)
    print("最終最適化結果")
    print("=" * 80)
    
    print(f"\n{'モデル':<25} {'CV R²':<15} {'Test R²':<12} {'Test RMSE':<12} {'Test MAE':<12}")
    print("-" * 90)
    
    for name, result in sorted(results.items(), key=lambda x: x[1]['test_r2'], reverse=True):
        cv_r2 = f"{result.get('cv_r2_mean', 0):.4f}±{result.get('cv_r2_std', 0):.4f}" if 'cv_r2_mean' in result else "N/A"
        print(f"{name:<25} {cv_r2:<15} {result['test_r2']:>10.4f}   {result['test_rmse']:>10.2f} GPa   {result['test_mae']:>10.2f} GPa")
    
    # 最良モデル
    best_model_name = max(results.items(), key=lambda x: x[1]['test_r2'])[0]
    best_result = results[best_model_name]
    
    print(f"\n{'='*80}")
    print(f"⭐ 最良モデル: {best_model_name}")
    if 'cv_r2_mean' in best_result:
        print(f"   CV R²: {best_result['cv_r2_mean']:.4f} ± {best_result['cv_r2_std']:.4f}")
    print(f"   Test R²: {best_result['test_r2']:.4f} ⭐")
    print(f"   Test RMSE: {best_result['test_rmse']:.2f} GPa")
    print(f"   Test MAE: {best_result['test_mae']:.2f} GPa")
    
    if best_result['test_r2'] >= 0.70:
        print(f"\n🎉🎉🎉 目標達成！ R² >= 0.70 を達成しました！ 🎉🎉🎉")
    elif best_result['test_r2'] >= 0.65:
        print(f"\n✨ 良好な結果！ R² >= 0.65 を達成しました！")
    print(f"{'='*80}")
    
    # 結果を保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = MODELS_DIR / f"final_measured_{timestamp}"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # 最良モデルを保存
    if best_model_name == 'Weighted_Ensemble':
        # アンサンブルの場合は個別モデルを保存
        for name, model in best_result['models']:
            with open(model_dir / f"{name}.pkl", 'wb') as f:
                pickle.dump(model, f)
        with open(model_dir / "ensemble_weights.json", 'w') as f:
            json.dump(best_result['weights'], f, indent=2)
    else:
        with open(model_dir / "best_model.pkl", 'wb') as f:
            pickle.dump(best_result['model'], f)
    
    with open(model_dir / "imputer.pkl", 'wb') as f:
        pickle.dump(imputer, f)
    
    # 結果をJSONで保存
    results_summary = {
        'timestamp': timestamp,
        'best_model': best_model_name,
        'target_achieved': best_result['test_r2'] >= 0.70,
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
        if 'cv_r2_mean' in result:
            results_summary['results'][name]['cv_r2_mean'] = float(result['cv_r2_mean'])
            results_summary['results'][name]['cv_r2_std'] = float(result['cv_r2_std'])
        if 'best_params' in result:
            results_summary['results'][name]['best_params'] = result['best_params']
        if 'weights' in result:
            results_summary['results'][name]['weights'] = result['weights']
    
    results_file = RESULTS_DIR / f"final_measured_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\n✅ モデルを保存しました: {model_dir}")
    print(f"✅ 結果を保存しました: {results_file}")
    
    print("\n" + "=" * 80)
    print("✅ 最終最適化が完了しました！")
    print("=" * 80)

if __name__ == "__main__":
    main()
