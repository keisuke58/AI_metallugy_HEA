#!/usr/bin/env python3
"""
推奨モデル訓練スクリプト（順次実行）

実行順序:
1. Gradient Boostingのハイパーパラメータ最適化
2. アンサンブル: Gradient Boosting + SVR + Random Forest
3. MEGNet/CGCNNでグラフ構造を活用
4. Transformerの改善

作成日: 2026-01-25
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
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, StackingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import RobustScaler, StandardScaler
from scipy.stats import randint, uniform

# 設定
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "final_data"
PROCESSED_DATA_DIR = BASE_DIR / "processed_data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# データパス（優先順位順）
DATA_FILES = [
    PROCESSED_DATA_DIR / "data_with_features.csv",
    DATA_DIR / "unified_dataset_latest.csv",
    DATA_DIR / "unified_dataset_cleaned_20260123_175245.csv"
]

DATA_FILE = None
for file in DATA_FILES:
    if file.exists():
        DATA_FILE = file
        break

print("=" * 80)
print("推奨モデル訓練スクリプト")
print("=" * 80)
print(f"📊 データファイル: {DATA_FILE}")
print(f"💾 モデル保存先: {MODELS_DIR}")
print(f"📈 結果保存先: {RESULTS_DIR}")
print("=" * 80)

# ============================================================================
# Phase 1: Gradient Boostingのハイパーパラメータ最適化
# ============================================================================

def optimize_gradient_boosting(X_train, y_train, X_test, y_test):
    """Gradient Boostingの高度なハイパーパラメータ最適化"""
    print("\n" + "=" * 80)
    print("Phase 1: Gradient Boosting ハイパーパラメータ最適化")
    print("=" * 80)
    
    # より広範囲なパラメータ探索空間
    param_distributions = {
        'n_estimators': randint(300, 1200),
        'max_depth': randint(3, 15),
        'learning_rate': uniform(0.005, 0.1),
        'min_samples_split': randint(2, 30),
        'min_samples_leaf': randint(1, 15),
        'subsample': uniform(0.6, 0.4),
        'max_features': ['sqrt', 'log2', None],
        'loss': ['squared_error', 'absolute_error', 'huber']
    }
    
    gb = GradientBoostingRegressor(random_state=42)
    
    # RandomizedSearchCVで最適化（より多くの試行回数）
    print("🔧 ハイパーパラメータ探索中...")
    random_search = RandomizedSearchCV(
        gb, param_distributions, n_iter=200, cv=5, scoring='r2',
        n_jobs=-1, random_state=42, verbose=1
    )
    
    random_search.fit(X_train, y_train)
    best_model = random_search.best_estimator_
    
    # 予測
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
        'feature_importance': dict(zip(X_train.columns, best_model.feature_importances_))
    }
    
    print(f"\n✅ 最適パラメータ:")
    for key, value in random_search.best_params_.items():
        print(f"   {key}: {value}")
    print(f"\n✅ CV R²: {result['cv_r2_mean']:.4f} ± {result['cv_r2_std']:.4f}")
    print(f"✅ Train R²: {result['train_r2']:.4f}, RMSE: {result['train_rmse']:.2f} GPa")
    print(f"✅ Test R²: {result['test_r2']:.4f}, RMSE: {result['test_rmse']:.2f} GPa")
    
    # モデル保存
    model_path = MODELS_DIR / "gradient_boosting_optimized.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)
    print(f"💾 モデルを保存しました: {model_path}")
    
    return result, best_model

# ============================================================================
# Phase 2: アンサンブルモデル（Gradient Boosting + SVR + Random Forest）
# ============================================================================

def create_ensemble_model(gb_model, X_train, y_train, X_test, y_test):
    """アンサンブルモデルの作成と訓練"""
    print("\n" + "=" * 80)
    print("Phase 2: アンサンブルモデル（GB + SVR + RF）")
    print("=" * 80)
    
    # SVRの最適化
    print("🔧 SVR最適化中...")
    svr_param_dist = {
        'C': uniform(0.1, 100),
        'epsilon': uniform(0.001, 0.5),
        'gamma': ['scale', 'auto'] + list(uniform(0.0001, 0.1).rvs(10))
    }
    svr = SVR(kernel='rbf')
    svr_search = RandomizedSearchCV(
        svr, svr_param_dist, n_iter=50, cv=5, scoring='r2',
        n_jobs=-1, random_state=42, verbose=0
    )
    svr_search.fit(X_train, y_train)
    svr_best = svr_search.best_estimator_
    print(f"   ✅ SVR最適化完了: C={svr_search.best_params_['C']:.2f}")
    
    # Random Forestの最適化
    print("🔧 Random Forest最適化中...")
    rf_param_dist = {
        'n_estimators': randint(200, 800),
        'max_depth': randint(5, 30),
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 10),
        'max_features': ['sqrt', 'log2', None]
    }
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    rf_search = RandomizedSearchCV(
        rf, rf_param_dist, n_iter=50, cv=5, scoring='r2',
        n_jobs=-1, random_state=42, verbose=0
    )
    rf_search.fit(X_train, y_train)
    rf_best = rf_search.best_estimator_
    print(f"   ✅ Random Forest最適化完了: n_estimators={rf_search.best_params_['n_estimators']}")
    
    # Voting Regressor（平均）
    print("🔧 Voting Regressor作成中...")
    voting_regressor = VotingRegressor([
        ('gb', gb_model),
        ('svr', svr_best),
        ('rf', rf_best)
    ], weights=[2, 1, 1])  # GBに重みを大きく
    
    voting_regressor.fit(X_train, y_train)
    
    # 予測
    y_pred_train = voting_regressor.predict(X_train)
    y_pred_test = voting_regressor.predict(X_test)
    
    result = {
        'model': voting_regressor,
        'base_models': {
            'gb': gb_model,
            'svr': svr_best,
            'rf': rf_best
        },
        'train_r2': r2_score(y_train, y_pred_train),
        'test_r2': r2_score(y_test, y_pred_test),
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'train_mae': mean_absolute_error(y_train, y_pred_train),
        'test_mae': mean_absolute_error(y_test, y_pred_test)
    }
    
    print(f"\n✅ Ensemble Train R²: {result['train_r2']:.4f}, RMSE: {result['train_rmse']:.2f} GPa")
    print(f"✅ Ensemble Test R²: {result['test_r2']:.4f}, RMSE: {result['test_rmse']:.2f} GPa")
    
    # モデル保存
    model_path = MODELS_DIR / "ensemble_voting.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(voting_regressor, f)
    print(f"💾 アンサンブルモデルを保存しました: {model_path}")
    
    return result, voting_regressor

# ============================================================================
# Phase 3: MEGNet/CGCNNの訓練（グラフ構造を活用）
# ============================================================================

def train_megnet_cgcnn(data_path):
    """MEGNetとCGCNNの訓練"""
    print("\n" + "=" * 80)
    print("Phase 3: MEGNet/CGCNN訓練（グラフ構造を活用）")
    print("=" * 80)
    
    import subprocess
    import sys
    
    fno_models_dir = BASE_DIR.parent / "fno_models"
    train_script = fno_models_dir / "train.py"
    
    if not train_script.exists():
        print("⚠️  fno_models/train.pyが見つかりません。")
        print("   手動で以下を実行してください:")
        print(f"   cd {fno_models_dir}")
        print(f"   python train.py --model megnet --data_path {data_path}")
        print(f"   python train.py --model cgcnn --data_path {data_path}")
        return None
    
    # MEGNet訓練
    print("🔧 MEGNet訓練中...")
    try:
        result = subprocess.run(
            [sys.executable, str(train_script), "--model", "megnet", "--data_path", str(data_path)],
            cwd=str(fno_models_dir),
            capture_output=True,
            text=True,
            timeout=3600  # 1時間タイムアウト
        )
        if result.returncode == 0:
            print("✅ MEGNet訓練完了")
        else:
            print(f"⚠️  MEGNet訓練でエラー: {result.stderr}")
    except subprocess.TimeoutExpired:
        print("⚠️  MEGNet訓練がタイムアウトしました")
    except Exception as e:
        print(f"⚠️  MEGNet訓練でエラー: {e}")
    
    # CGCNN訓練
    print("🔧 CGCNN訓練中...")
    try:
        result = subprocess.run(
            [sys.executable, str(train_script), "--model", "cgcnn", "--data_path", str(data_path)],
            cwd=str(fno_models_dir),
            capture_output=True,
            text=True,
            timeout=3600
        )
        if result.returncode == 0:
            print("✅ CGCNN訓練完了")
        else:
            print(f"⚠️  CGCNN訓練でエラー: {result.stderr}")
    except subprocess.TimeoutExpired:
        print("⚠️  CGCNN訓練がタイムアウトしました")
    except Exception as e:
        print(f"⚠️  CGCNN訓練でエラー: {e}")
    
    return None

# ============================================================================
# Phase 4: Transformerの改善
# ============================================================================

def improve_transformer(data_path):
    """Transformerモデルの改善"""
    print("\n" + "=" * 80)
    print("Phase 4: Transformerモデルの改善")
    print("=" * 80)
    
    import subprocess
    import sys
    
    gnn_transformer_dir = BASE_DIR.parent / "gnn_transformer_models"
    train_script = gnn_transformer_dir / "train.py"
    
    if not train_script.exists():
        print("⚠️  gnn_transformer_models/train.pyが見つかりません。")
        print("   手動で以下を実行してください:")
        print(f"   cd {gnn_transformer_dir}")
        print(f"   python train.py --model transformer --data_path {data_path} --batch_size 16 --learning_rate 1e-4")
        return None
    
    print("🔧 Transformer改善訓練中...")
    print("   ハイパーパラメータ: batch_size=16, learning_rate=1e-4, num_epochs=800, early_stopping_patience=100")
    
    try:
        result = subprocess.run(
            [sys.executable, str(train_script), 
             "--model", "transformer",
             "--data_path", str(data_path),
             "--batch_size", "16",
             "--learning_rate", "1e-4",
             "--num_epochs", "800",  # 300から800に大幅増加
             "--early_stopping_patience", "100"],  # より長い忍耐で早期停止を抑制
            cwd=str(gnn_transformer_dir),
            capture_output=True,
            text=True,
            timeout=18000  # 5時間タイムアウト（エポック数増加に対応）
        )
        if result.returncode == 0:
            print("✅ Transformer改善訓練完了")
            print("   出力:")
            print(result.stdout[-500:])  # 最後の500文字を表示
        else:
            print(f"⚠️  Transformer訓練でエラー: {result.stderr}")
    except subprocess.TimeoutExpired:
        print("⚠️  Transformer訓練がタイムアウトしました")
    except Exception as e:
        print(f"⚠️  Transformer訓練でエラー: {e}")
    
    return None

# ============================================================================
# メイン関数
# ============================================================================

def load_and_prepare_data():
    """データの読み込みと準備"""
    print("\n📊 データ読み込み中...")
    
    # データファイルを検索
    data_files = [
        PROCESSED_DATA_DIR / "data_with_features.csv",
        DATA_DIR / "unified_dataset_latest.csv",
        DATA_DIR / "unified_dataset_cleaned_20260123_175245.csv"
    ]
    
    data_file = None
    for file in data_files:
        print(f"   確認中: {file} (存在: {file.exists()})")
        if file.exists():
            data_file = file
            print(f"   ✅ データファイルを見つけました: {data_file}")
            break
    
    if data_file is None or not data_file.exists():
        raise FileNotFoundError(f"データファイルが見つかりません。確認したパス: {[str(f) for f in data_files]}")
    
    df = pd.read_csv(data_file)
    print(f"   データ数: {len(df)}")
    
    # ターゲット変数
    if 'elastic_modulus' not in df.columns:
        raise ValueError("'elastic_modulus'カラムが見つかりません")
    
    # 特徴量カラム（数値カラムからターゲットを除外）
    feature_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                    if col != 'elastic_modulus' and not col.startswith('Unnamed')]
    
    # 欠損値処理
    df = df.dropna(subset=['elastic_modulus'])
    df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())
    
    X = df[feature_cols]
    y = df['elastic_modulus']
    
    print(f"   特徴量数: {len(feature_cols)}")
    print(f"   有効データ数: {len(X)}")
    
    # データ分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # スケーリング
    scaler = RobustScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    
    # スケーラー保存
    scaler_path = MODELS_DIR / "scaler_robust.pkl"
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"💾 スケーラーを保存しました: {scaler_path}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, feature_cols

def main():
    """メイン実行関数"""
    try:
        # データ準備
        X_train, X_test, y_train, y_test, feature_cols = load_and_prepare_data()
        
        all_results = {}
        
        # Phase 1: Gradient Boosting最適化
        gb_result, gb_model = optimize_gradient_boosting(X_train, y_train, X_test, y_test)
        all_results['gradient_boosting'] = {
            'best_params': gb_result['best_params'],
            'cv_r2': gb_result['cv_r2_mean'],
            'test_r2': gb_result['test_r2'],
            'test_rmse': gb_result['test_rmse'],
            'test_mae': gb_result['test_mae']
        }
        
        # Phase 2: アンサンブル
        ensemble_result, ensemble_model = create_ensemble_model(
            gb_model, X_train, y_train, X_test, y_test
        )
        all_results['ensemble'] = {
            'test_r2': ensemble_result['test_r2'],
            'test_rmse': ensemble_result['test_rmse'],
            'test_mae': ensemble_result['test_mae']
        }
        
        # Phase 3: MEGNet/CGCNN
        data_files = [
            PROCESSED_DATA_DIR.resolve() / "data_with_features.csv",
            DATA_DIR.resolve() / "unified_dataset_latest.csv",
            DATA_DIR.resolve() / "unified_dataset_cleaned_20260123_175245.csv"
        ]
        data_file = None
        for file in data_files:
            if file.exists():
                data_file = file
                break
        
        if data_file:
            train_megnet_cgcnn(data_file)
            # Phase 4: Transformer改善
            improve_transformer(data_file)
        
        # 結果保存
        results_path = RESULTS_DIR / f"recommended_models_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\n💾 結果を保存しました: {results_path}")
        
        # サマリー表示
        print("\n" + "=" * 80)
        print("結果サマリー")
        print("=" * 80)
        print(f"\n{'モデル':<30} {'Test R²':<12} {'Test RMSE':<12} {'Test MAE':<12}")
        print("-" * 80)
        print(f"{'Gradient Boosting':<30} {all_results['gradient_boosting']['test_r2']:<12.4f} "
              f"{all_results['gradient_boosting']['test_rmse']:<12.2f} "
              f"{all_results['gradient_boosting']['test_mae']:<12.2f}")
        print(f"{'Ensemble (GB+SVR+RF)':<30} {all_results['ensemble']['test_r2']:<12.4f} "
              f"{all_results['ensemble']['test_rmse']:<12.2f} "
              f"{all_results['ensemble']['test_mae']:<12.2f}")
        print("=" * 80)
        
        print("\n✅ Phase 1-2が完了しました。")
        print("📝 Phase 3-4は手動で実行してください（上記の指示を参照）。")
        
    except Exception as e:
        print(f"\n❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
