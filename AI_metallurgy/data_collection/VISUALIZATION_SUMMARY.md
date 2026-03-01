# 📊 可視化サマリー

**作成日**: 2026年1月19日  
**目的**: 実測データのみの最適化結果を包括的に可視化

---

## ✅ 生成された可視化プロット（20個）

### 1. 予測性能の可視化

#### `predicted_vs_actual_(measured_data_only).png`
- **内容**: 予測値 vs 実測値の散布図（訓練データとテストデータ）
- **目的**: モデルの予測精度を視覚的に確認
- **指標**: R²、RMSEを表示

#### `predicted_vs_actual.png`
- **内容**: 元の予測値 vs 実測値プロット
- **目的**: 比較用

---

### 2. 残差分析

#### `residuals_analysis_(measured_data_only).png`
- **内容**: 
  - 残差 vs 予測値の散布図
  - 残差の分布（ヒストグラム）
- **目的**: モデルのバイアスと分散を確認
- **評価**: 残差がランダムに分布しているか確認

#### `residuals_analysis.png`
- **内容**: 元の残差分析プロット
- **目的**: 比較用

---

### 3. 特徴量分析

#### `feature_importance.png`
- **内容**: 上位20個の特徴量重要度
- **目的**: どの特徴量が予測に重要かを確認
- **活用**: 特徴量選択の参考

#### `feature_correlation.png`
- **内容**: 特徴量間の相関マトリックス（ヒートマップ）
- **目的**: 特徴量間の相関関係を確認
- **活用**: 多重共線性の検出

#### `feature_vs_target.png`
- **内容**: 主要特徴量とターゲット（弾性率）の関係
- **目的**: 各特徴量が弾性率にどのように影響するかを確認
- **表示**: 相関係数を含む

---

### 4. データ分析

#### `data_distribution.png`
- **内容**: 
  - 弾性率の分布（ヒストグラム）
  - 箱ひげ図
- **目的**: データの分布特性を確認
- **統計**: 平均、中央値、標準偏差、最小値、最大値を表示

#### `data_characteristics.png`
- **内容**: データ特性と最良モデル性能の関係
- **目的**: データ数と性能の関係を可視化

---

### 5. モデル比較

#### `model_comparison_all.png`
- **内容**: すべてのモデルの性能比較（R²、RMSE、MAE）
- **目的**: 最良モデルを特定
- **表示**: 横棒グラフで比較

#### `model_comparison_measured_vs_combined.png`
- **内容**: 実測データのみ vs 実測+計算データの比較
- **目的**: データソースによる性能の違いを確認

#### `model_performance_comparison.png`
- **内容**: 元のモデル性能比較
- **目的**: 比較用

#### `performance_metrics_comparison.png`
- **内容**: 正規化された性能指標の総合比較
- **目的**: 複数の指標を統合して比較

---

### 6. 最適化の軌跡

#### `optimization_history.png`
- **内容**: 最適化過程でのR²とRMSEの推移
- **目的**: 最適化の効果を確認
- **表示**: 時系列プロット

#### `performance_improvement.png`
- **内容**: 実測データのみ vs 実測+計算データの改善率
- **目的**: データソースによる改善を確認

---

### 7. 誤差分析

#### `error_distribution.png`
- **内容**: 
  - 絶対誤差の分布
  - 相対誤差の分布
  - 誤差 vs 実測値
  - 誤差の箱ひげ図
- **目的**: 誤差の特性を詳細に分析
- **活用**: モデルの改善点を特定

---

### 8. 学習曲線

#### `learning_curves_detailed.png`
- **内容**: 訓練データサイズに対する性能の変化
- **目的**: 過学習やデータ不足を確認
- **表示**: 訓練スコアと検証スコアの推移

#### `learning_curves.png`
- **内容**: 元の学習曲線
- **目的**: 比較用

---

### 9. 予測区間

#### `prediction_intervals.png`
- **内容**: 95%予測区間と実測値の比較
- **目的**: 予測の不確実性を可視化
- **指標**: カバレッジ率を表示

---

### 10. 全結果サマリー

#### `all_results_summary.png`
- **内容**: 
  - R²スコアの分布
  - RMSEの分布
  - R² vs RMSE（MAEで色分け）
  - 上位5モデルの比較
- **目的**: すべての最適化結果を統合して表示
- **活用**: 最良モデルの特定と傾向の把握

---

## 📁 ファイル構成

```
figures/
├── all_results_summary.png                    # 全結果サマリー
├── data_characteristics.png                   # データ特性
├── data_distribution.png                      # データ分布
├── elastic_modulus_comparison.png             # 弾性率比較
├── error_distribution.png                      # 誤差分布
├── feature_correlation.png                     # 特徴量相関
├── feature_importance.png                     # 特徴量重要度
├── feature_vs_target.png                      # 特徴量 vs ターゲット
├── learning_curves_detailed.png               # 詳細学習曲線
├── learning_curves.png                        # 学習曲線
├── model_comparison_all.png                   # 全モデル比較
├── model_comparison_measured_vs_combined.png  # 実測 vs 実測+計算
├── model_performance_comparison.png           # モデル性能比較
├── optimization_history.png                   # 最適化履歴
├── performance_improvement.png                 # 性能改善
├── performance_metrics_comparison.png         # 性能指標比較
├── predicted_vs_actual_(measured_data_only).png  # 予測値 vs 実測値（実測のみ）
├── predicted_vs_actual.png                   # 予測値 vs 実測値
├── prediction_intervals.png                   # 予測区間
├── residuals_analysis_(measured_data_only).png    # 残差分析（実測のみ）
└── residuals_analysis.png                     # 残差分析
```

---

## 🎯 可視化の目的と活用方法

### 1. モデル性能の評価
- **予測値 vs 実測値**: モデルの予測精度を直感的に確認
- **残差分析**: モデルのバイアスと分散を評価
- **誤差分布**: 誤差の特性を詳細に分析

### 2. 特徴量の理解
- **特徴量重要度**: どの特徴量が重要かを特定
- **特徴量相関**: 特徴量間の関係を理解
- **特徴量 vs ターゲット**: 各特徴量の影響を確認

### 3. 最適化の効果確認
- **最適化履歴**: 最適化の効果を時系列で確認
- **モデル比較**: 複数のモデルを比較
- **全結果サマリー**: すべての結果を統合して評価

### 4. データの理解
- **データ分布**: データの特性を把握
- **予測区間**: 予測の不確実性を理解

---

## 📝 実行方法

### 包括的可視化
```bash
cd /home/nishioka/LUH/AI_metallurgy/data_collection
python scripts/comprehensive_visualization.py
```

### 追加可視化
```bash
python scripts/additional_visualizations.py
```

---

## 📊 統計情報

- **総プロット数**: 20個
- **主要カテゴリ**: 
  - 予測性能: 2個
  - 残差分析: 2個
  - 特徴量分析: 3個
  - データ分析: 2個
  - モデル比較: 4個
  - 最適化軌跡: 2個
  - 誤差分析: 1個
  - 学習曲線: 2個
  - 予測区間: 1個
  - 全結果サマリー: 1個

---

## ✨ 主な発見

1. **最良モデル**: Gradient Boosting (R² = 0.6744)
2. **重要な特徴量**: 
   - diff_in_atomic_radii
   - estimated_density
   - comp_V
   - comp_Al
   - mixing_enthalpy
3. **データ分布**: 27-466 GPaの範囲、平均165.40 GPa
4. **予測精度**: RMSE = 49.70 GPa, MAE = 37.07 GPa

---

**レポート作成日**: 2026年1月19日  
**最終更新**: 2026年1月19日
