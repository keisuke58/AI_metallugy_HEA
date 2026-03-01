# 推奨モデル訓練スクリプト実行ガイド

**作成日**: 2026-01-25

## 📋 概要

このスクリプトは、推奨された順番でモデルを訓練します：

1. **Phase 1**: Gradient Boostingのハイパーパラメータ最適化
2. **Phase 2**: アンサンブルモデル（GB + SVR + RF）
3. **Phase 3**: MEGNet/CGCNNでグラフ構造を活用
4. **Phase 4**: Transformerモデルの改善

## 🚀 実行方法

### 基本的な実行

```bash
cd /home/nishioka/LUH/AI_metallurgy/data_collection/scripts
python recommended_model_training.py
```

### 必要なパッケージ

```bash
pip install pandas numpy scikit-learn scipy
```

## 📊 実行順序と詳細

### Phase 1: Gradient Boosting最適化

- **目的**: 最高性能を達成するためのハイパーパラメータ最適化
- **手法**: RandomizedSearchCV（200回試行、5-fold CV）
- **探索パラメータ**:
  - `n_estimators`: 300-1200
  - `max_depth`: 3-15
  - `learning_rate`: 0.005-0.1
  - `min_samples_split`: 2-30
  - `min_samples_leaf`: 1-15
  - `subsample`: 0.6-1.0
  - `max_features`: sqrt, log2, None
  - `loss`: squared_error, absolute_error, huber

**出力**:
- 最適化されたモデル: `models/gradient_boosting_optimized.pkl`
- 性能メトリクス（R², RMSE, MAE）

### Phase 2: アンサンブルモデル

- **目的**: 複数モデルの予測を統合して性能向上
- **構成**:
  - Gradient Boosting（Phase 1で最適化）
  - SVR（RBFカーネル、最適化）
  - Random Forest（最適化）
- **手法**: VotingRegressor（重み付き平均）
  - GB: 重み2
  - SVR: 重み1
  - RF: 重み1

**出力**:
- アンサンブルモデル: `models/ensemble_voting.pkl`
- 性能メトリクス

### Phase 3: MEGNet/CGCNN

- **目的**: グラフ構造を活用した材料特性予測
- **実装**: `fno_models/train.py`を使用
- **モデル**:
  - MEGNet: 材料科学に特化したGNN
  - CGCNN: 結晶構造に特化したGNN

**実行**:
```bash
cd /home/nishioka/LUH/AI_metallurgy/fno_models
python train.py --model megnet --data_path ../data_collection/final_data/unified_dataset_latest.csv
python train.py --model cgcnn --data_path ../data_collection/final_data/unified_dataset_latest.csv
```

### Phase 4: Transformer改善

- **目的**: Transformerモデルの性能改善
- **改善点**:
  - バッチサイズ: 16
  - 学習率: 1e-4
  - エポック数: 300
  - その他のハイパーパラメータ調整

**実行**:
```bash
cd /home/nishioka/LUH/AI_metallurgy/gnn_transformer_models
python train.py --model transformer --batch_size 16 --learning_rate 1e-4 --num_epochs 300
```

## 📁 出力ファイル

### モデルファイル
- `models/gradient_boosting_optimized.pkl`: 最適化されたGBモデル
- `models/ensemble_voting.pkl`: アンサンブルモデル
- `models/scaler_robust.pkl`: データスケーラー

### 結果ファイル
- `results/recommended_models_results_YYYYMMDD_HHMMSS.json`: 全結果のJSON

## ⚙️ カスタマイズ

### データファイルの変更

スクリプト内で以下の順序でデータファイルを検索します：
1. `processed_data/data_with_features.csv`
2. `final_data/unified_dataset_latest.csv`
3. `final_data/unified_dataset_cleaned_20260123_175245.csv`

### ハイパーパラメータの調整

スクリプト内の`param_distributions`を編集して調整可能です。

## 🔍 トラブルシューティング

### データファイルが見つからない

```bash
# データファイルの存在確認
ls -la /home/nishioka/LUH/AI_metallurgy/data_collection/processed_data/
ls -la /home/nishioka/LUH/AI_metallurgy/data_collection/final_data/
```

### Phase 3/4でエラーが発生する場合

Phase 3と4は別のディレクトリのスクリプトを呼び出すため、手動実行も可能です：

```bash
# MEGNet
cd /home/nishioka/LUH/AI_metallurgy/fno_models
python train.py --model megnet

# CGCNN
python train.py --model cgcnn

# Transformer
cd /home/nishioka/LUH/AI_metallurgy/gnn_transformer_models
python train.py --model transformer --batch_size 16 --learning_rate 1e-4
```

## 📈 期待される結果

### Phase 1-2（古典的ML）
- **Gradient Boosting**: R² ≈ 0.65-0.70
- **Ensemble**: R² ≈ 0.65-0.72（改善の可能性）

### Phase 3（GNN）
- **MEGNet**: R² ≈ 0.45-0.50
- **CGCNN**: R² ≈ 0.45-0.50

### Phase 4（Transformer）
- **Transformer**: R² > 0.30（改善後）

## 📝 注意事項

1. **実行時間**: Phase 1-2は数分、Phase 3-4は数時間かかる可能性があります
2. **メモリ**: 大規模データセット（5,339サンプル）を使用する場合、十分なメモリが必要です
3. **GPU**: Phase 3-4はGPUがあると高速化されます

## 🔗 関連ファイル

- `MODEL_RECOMMENDATION_ANALYSIS.md`: モデル推奨分析
- `FNO_IMPLEMENTATION_PLAN.md`: FNO実装計画
- `fno_models/train.py`: MEGNet/CGCNN訓練スクリプト
- `gnn_transformer_models/train.py`: Transformer訓練スクリプト
