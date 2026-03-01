# 推奨モデル訓練結果サマリー

**実行日**: 2026-01-25  
**環境**: hea_gnn  
**データセット**: 322サンプル（実験データ）

---

## 📊 実行結果サマリー

### Phase 1: Gradient Boosting ハイパーパラメータ最適化 ✅

**最適パラメータ**:
- `n_estimators`: 707
- `max_depth`: 12
- `learning_rate`: 0.0344
- `loss`: huber
- `min_samples_split`: 26
- `min_samples_leaf`: 2
- `subsample`: 0.965
- `max_features`: None

**性能**:
- **CV R²**: 0.4806 ± (std)
- **Test R²**: **0.4956** ⭐
- **Test RMSE**: **40.42 GPa**
- **Test MAE**: **25.14 GPa**

**評価**: ✅ 良好な性能を達成

---

### Phase 2: アンサンブルモデル（GB + SVR + RF） ✅

**構成**:
- Gradient Boosting（Phase 1で最適化、重み: 2）
- SVR（RBFカーネル、最適化、重み: 1）
- Random Forest（最適化、重み: 1）

**性能**:
- **Test R²**: **0.4797**
- **Test RMSE**: **41.05 GPa**
- **Test MAE**: **26.30 GPa**

**評価**: ⚠️ 単一のGBモデルとほぼ同等（アンサンブルの効果は限定的）

---

### Phase 3: MEGNet/CGCNN（グラフ構造を活用） ✅

#### MEGNet

**性能**:
- **Test R²**: **-0.0096** ❌
- **Test RMSE**: 57.27 GPa
- **Test MAE**: 41.39 GPa

**評価**: ❌ 性能不良（負のR²）

#### CGCNN

**性能**:
- **Test R²**: **0.2187** ⚠️
- **Test RMSE**: 50.38 GPa
- **Test MAE**: 29.90 GPa

**評価**: ⚠️ 中程度の性能（GBより低い）

---

### Phase 4: Transformer改善 ✅

**改善前**: R² = -0.238（性能不良）

**改善後**:
- **Test R²**: **0.3191** ✅
- **Test RMSE**: **47.03 GPa**
- **Test MAE**: **32.63 GPa**

**改善点**:
- バッチサイズ: 16
- 学習率: 1e-4
- エポック数: 300

**評価**: ✅ 大幅に改善（負のR²から0.32へ）

---

## 📈 全モデル性能比較

| モデル | Test R² | Test RMSE (GPa) | Test MAE (GPa) | 評価 |
|--------|---------|-----------------|----------------|------|
| **Gradient Boosting** | **0.4956** | **40.42** | **25.14** | ⭐⭐⭐⭐⭐ |
| **Ensemble (GB+SVR+RF)** | 0.4797 | 41.05 | 26.30 | ⭐⭐⭐⭐ |
| **Transformer (改善後)** | 0.3191 | 47.03 | 32.63 | ⭐⭐⭐ |
| **CGCNN** | 0.2187 | 50.38 | 29.90 | ⭐⭐ |
| **MEGNet** | -0.0096 | 57.27 | 41.39 | ❌ |

---

## 🎯 主要な発見

### 1. Gradient Boostingが最高性能

- **R² = 0.4956**で最高性能を達成
- ハイパーパラメータ最適化により良好な結果
- 実用的なアプリケーションに最適

### 2. アンサンブルの効果は限定的

- Ensembleは単一のGBモデルとほぼ同等
- 3モデルの予測が類似している可能性
- より多様なモデルの組み合わせが必要

### 3. Transformerは大幅に改善

- **改善前**: R² = -0.238 → **改善後**: R² = 0.3191
- ハイパーパラメータ調整により性能向上
- ただし、GBよりは低い性能

### 4. GNNモデル（MEGNet/CGCNN）の性能は低い

- **CGCNN**: R² = 0.22（中程度）
- **MEGNet**: R² = -0.01（性能不良）
- グラフ構造の活用が不十分な可能性

---

## 💡 推奨事項

### 最適なモデル: **Gradient Boosting**

**理由**:
1. ✅ 最高性能（R² = 0.50）
2. ✅ 実装が容易
3. ✅ 解釈可能性（特徴量重要度）
4. ✅ 計算効率が高い

### 今後の改善案

1. **Gradient Boostingのさらなる最適化**
   - より広範囲なパラメータ探索
   - 特徴量エンジニアリングの改善

2. **Transformerのさらなる改善**
   - アーキテクチャの調整
   - より長い訓練期間

3. **GNNモデルの改善**
   - グラフ構造の最適化
   - 特徴量の統合

4. **アンサンブルの改善**
   - より多様なモデルの組み合わせ
   - スタッキングアンサンブルの検討

---

## 📁 保存されたファイル

### モデルファイル
- `models/gradient_boosting_optimized.pkl`: 最適化されたGBモデル
- `models/ensemble_voting.pkl`: アンサンブルモデル
- `models/scaler_robust.pkl`: データスケーラー

### 結果ファイル
- `results/recommended_models_results_20260125_175726.json`: Phase 1-2の結果
- `fno_models/results/megnet_results.json`: MEGNet結果
- `fno_models/results/cgcnn_results.json`: CGCNN結果
- `gnn_transformer_models/results/model_comparison.json`: Transformer結果

---

## ✅ 実行完了

すべてのPhaseが正常に完了しました。

**次のステップ**:
1. 結果の詳細分析
2. モデルのさらなる最適化
3. 実用的なアプリケーションへの適用
