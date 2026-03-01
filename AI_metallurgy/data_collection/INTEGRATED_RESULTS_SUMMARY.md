# 統合結果レポート - 追加内容サマリー

## 更新日: 2026-01-25

## 追加された内容

### 1. 新しいセクション: "Comprehensive Model Evaluation on Full Dataset"

全モデルの結果を統合した包括的な評価セクションを追加しました。

#### 追加されたテーブル: Table (tab:comprehensive_results)

全モデルの性能を比較する包括的なテーブルを追加：

**Classical Machine Learning Models:**
- Gradient Boosting (exp. only): R² = 0.67, RMSE = 49.7 GPa, MAE = 37.1 GPa
- SVR (optimized): R² = 0.59, RMSE = 36.2 GPa
- Random Forest: R² = 0.57, RMSE = 37.5 GPa, MAE = 25.9 GPa
- MLFFNN: R² = 0.49, RMSE = 40.5 GPa, MAE = 29.0 GPa

**Graph Neural Networks:**
- MEGNet: R² = 0.485, RMSE = 63.79 GPa, MAE = 43.59 GPa
- CGCNN: R² = 0.451, RMSE = 65.88 GPa, MAE = 44.50 GPa
- GNN (HEAGNN): R² = 0.259, RMSE = 49.08 GPa, MAE = 29.53 GPa

**Neural Operator and Physics-Informed Models:**
- PINNs: R² = 0.403, RMSE = 68.71 GPa, MAE = 45.78 GPa
- Neural ODE: R² = 0.254, RMSE = 76.77 GPa, MAE = 53.46 GPa
- DeepONet: R² = 0.275, RMSE = 75.68 GPa, MAE = 53.08 GPa
- FNO: R² = 0.243, RMSE = 77.33 GPa, MAE = 53.46 GPa

**Transformer-based Models:**
- Transformer (HEATransformer): R² = -0.238, RMSE = 63.43 GPa, MAE = 41.68 GPa

### 2. 主要な発見 (Key Findings)

- 古典的MLモデルが最高性能
- グラフニューラルネットワークは有望
- Neural Operator手法は性能が低い
- Transformerモデルは性能不良

### 3. モデル適合性分析 (Model Suitability Analysis)

- データ表現の違い
- データセットサイズと複雑さ
- 実験データと計算データのバイアス

### 4. モデル選択の推奨事項 (Recommendations for Model Selection)

- 実用的なアプリケーション: SVRまたはGradient Boosting
- 研究探索: MEGNetまたはCGCNN
- 将来の開発: ハイブリッドアプローチ

### 5. 結論セクションの更新

全モデルの評価結果を反映した包括的な結論に更新しました。

## ファイル場所

- **更新されたLaTeXファイル**: `/home/nishioka/LUH/AI_metallurgy/data_collection/PROJECT_FINAL_REPORT_EN.tex`

## 次のステップ

1. LaTeXファイルをコンパイルしてPDFを生成
2. 図表の参照を確認
3. 必要に応じて追加の可視化を作成
