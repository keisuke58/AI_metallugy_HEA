#!/usr/bin/env python3
"""
問題を修正して再検証
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import glob
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).parent.parent
PROCESSED_DATA_DIR = BASE_DIR / "processed_data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

def find_best_model():
    """最良のモデルを見つける"""
    result_files = sorted(glob.glob(str(RESULTS_DIR / "*measured*.json")))
    
    best_r2 = -np.inf
    best_result_file = None
    best_model_name = None
    
    for result_file in result_files:
        try:
            with open(result_file, 'r') as f:
                result = json.load(f)
            
            if 'results' in result:
                for name, model_result in result['results'].items():
                    r2 = model_result.get('test_r2', -np.inf)
                    if r2 > best_r2:
                        best_r2 = r2
                        best_result_file = result_file
                        best_model_name = name
            elif 'optimized_r2' in result:
                r2 = result.get('optimized_r2', -np.inf)
                if r2 > best_r2:
                    best_r2 = r2
                    best_result_file = result_file
                    best_model_name = 'Optimized'
        except:
            pass
    
    return best_result_file, best_model_name, best_r2

def verify_model_with_correct_data():
    """正しいデータでモデルを検証"""
    print("=" * 80)
    print("モデル検証と修正")
    print("=" * 80)
    
    # 最良の結果ファイルを見つける
    best_result_file, best_model_name, best_r2 = find_best_model()
    
    if not best_result_file:
        print("❌ 結果ファイルが見つかりません")
        return
    
    print(f"📊 最良結果: {Path(best_result_file).name}")
    print(f"📊 最良モデル: {best_model_name}")
    print(f"📊 記録されたR²: {best_r2:.4f}")
    
    # final_optimization.pyで作成されたモデルを使用（最も信頼できる）
    model_dirs = sorted(glob.glob(str(MODELS_DIR / "final_measured_*")))
    if not model_dirs:
        model_dirs = sorted(glob.glob(str(MODELS_DIR / "*measured*")))
    
    if not model_dirs:
        print("❌ モデルディレクトリが見つかりません")
        return
    
    # final_measured_*を優先
    final_models = [d for d in model_dirs if 'final_measured' in d]
    if final_models:
        model_dir = Path(final_models[-1])
    else:
        model_dir = Path(model_dirs[-1])
    
    print(f"📊 使用モデルディレクトリ: {model_dir.name}")
    
    # モデルを読み込む
    model_file = model_dir / "best_model.pkl"
    if not model_file.exists():
        print(f"❌ モデルファイルが見つかりません: {model_file}")
        return
    
    try:
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
        print(f"✅ モデル読み込み成功: {type(model).__name__}")
    except Exception as e:
        print(f"❌ モデル読み込みエラー: {e}")
        return
    
    # データを読み込む（final_optimization.pyと同じデータソース）
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
    
    # データ分割（同じrandom_state）
    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y, test_size=0.2, random_state=42
    )
    
    print(f"\n📊 データ統計:")
    print(f"   訓練データ: {len(X_train)}サンプル")
    print(f"   テストデータ: {len(X_test)}サンプル")
    print(f"   特徴量数: {len(feature_cols)}個")
    
    # 予測
    try:
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        r2_train = r2_score(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)
        rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
        mae_test = mean_absolute_error(y_test, y_pred_test)
        
        print(f"\n📊 検証結果:")
        print(f"   Train R²: {r2_train:.4f}")
        print(f"   Test R²: {r2_test:.4f}")
        print(f"   Test RMSE: {rmse_test:.2f} GPa")
        print(f"   Test MAE: {mae_test:.2f} GPa")
        
        # 結果を比較
        print(f"\n📊 結果比較:")
        print(f"   記録されたR²: {best_r2:.4f}")
        print(f"   検証R²: {r2_test:.4f}")
        print(f"   差: {abs(best_r2 - r2_test):.4f}")
        
        if abs(best_r2 - r2_test) < 0.01:
            print(f"   ✅ 結果が一致しています（完璧！）")
        elif abs(best_r2 - r2_test) < 0.05:
            print(f"   ✅ 結果がほぼ一致しています（良好）")
        else:
            print(f"   ⚠️  結果に差があります（要確認）")
        
        # 最終評価
        print(f"\n{'='*80}")
        print("最終評価")
        print(f"{'='*80}")
        
        if r2_test >= 0.70:
            print("🎉🎉🎉 完璧です！ R² >= 0.70 を達成！ 🎉🎉🎉")
            status = "perfect"
        elif r2_test >= 0.65:
            print("✨ 優秀です！ R² >= 0.65 を達成！")
            status = "excellent"
        elif r2_test >= 0.60:
            print("✅ 良好です！ R² >= 0.60 を達成！")
            status = "good"
        elif r2_test >= 0.50:
            print("⚠️  中程度です。改善の余地があります。")
            status = "fair"
        else:
            print("❌ 改善が必要です。")
            status = "needs_improvement"
        
        # 結果を保存
        verification_result = {
            'model_dir': str(model_dir),
            'model_type': type(model).__name__,
            'recorded_r2': float(best_r2),
            'verified_r2_train': float(r2_train),
            'verified_r2_test': float(r2_test),
            'verified_rmse': float(rmse_test),
            'verified_mae': float(mae_test),
            'difference': float(abs(best_r2 - r2_test)),
            'status': status,
            'data_samples': {
                'train': len(X_train),
                'test': len(X_test),
                'features': len(feature_cols)
            }
        }
        
        verification_file = RESULTS_DIR / "model_verification.json"
        with open(verification_file, 'w') as f:
            json.dump(verification_result, f, indent=2)
        
        print(f"\n✅ 検証結果を保存しました: {verification_file}")
        
        return verification_result
        
    except Exception as e:
        print(f"❌ 予測エラー: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """メイン関数"""
    result = verify_model_with_correct_data()
    
    if result:
        print(f"\n{'='*80}")
        print("✅ 検証が完了しました！")
        print(f"{'='*80}")
    else:
        print(f"\n{'='*80}")
        print("❌ 検証に失敗しました")
        print(f"{'='*80}")

if __name__ == "__main__":
    main()
