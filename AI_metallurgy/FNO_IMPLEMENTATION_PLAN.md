# 🎯 Fourier Neural Operator (FNO) 実装計画

**作成日**: 2026年1月23日  
**プロジェクト**: AI Applications in Metallurgy  
**テーマ**: 生体医学用Fe合金の弾性率予測

---

## 📋 目次

1. [FNO概要](#fno概要)
2. [実装アプローチ](#実装アプローチ)
3. [実装計画](#実装計画)
4. [他のモデル案](#他のモデル案)
5. [実装スケジュール](#実装スケジュール)

---

## 🔬 FNO概要

### Fourier Neural Operatorとは

FNO（Fourier Neural Operator）は、偏微分方程式の解を学習するためのニューラルオペレーターです。

**特徴**:
- **関数間の写像を学習**: 入力関数から出力関数への写像を直接学習
- **Fourier変換を活用**: 周波数領域での効率的な処理
- **解像度不変**: 訓練時の解像度と異なる解像度でも推論可能
- **物理法則の学習**: 偏微分方程式の解を効率的に近似

**論文**: 
- Li et al. (2020) "Fourier Neural Operator for Parametric Partial Differential Equations"
- Li et al. (2021) "Fourier Neural Operator for Parametric Partial Differential Equations"

### 材料科学への応用

材料科学では、以下のような問題にFNOを適用できます：

1. **組成-特性マッピング**: 組成分布から材料特性を予測
2. **時系列予測**: 熱処理過程での特性変化を予測
3. **空間分布予測**: 材料内部の特性分布を予測

---

## 🎯 実装アプローチ

### アプローチ1: Compositional FNO（推奨）

**アイデア**: 組成データを空間的に配置し、FNOで処理

**実装方法**:
1. **組成の空間化**: 
   - 各元素を1次元グリッド上の位置として配置
   - 組成比を関数値として扱う
   - 例: `[Ti: 0.3, Zr: 0.2, Nb: 0.3, Ta: 0.2]` → `f(x) = [0.3, 0.2, 0.3, 0.2]` (xは元素位置)

2. **材料記述子の統合**:
   - 材料記述子（mixing entropy, enthalpy等）を追加チャンネルとして追加
   - または、条件付きFNOとして実装

3. **出力**:
   - 弾性率をスカラー値として予測

**メリット**:
- ✅ FNOの標準的な実装を活用可能
- ✅ 組成の連続的な変化を捉えられる
- ✅ 解像度不変性により、異なる元素数の合金にも対応可能

**デメリット**:
- ⚠️ 組成データの空間化が必要（人工的な変換）
- ⚠️ 元素の順序に依存する可能性

### アプローチ2: Feature-based FNO

**アイデア**: 材料記述子を関数として扱い、FNOで処理

**実装方法**:
1. **特徴量の関数化**:
   - 材料記述子を1次元関数として扱う
   - 例: `[mixing_entropy, mixing_enthalpy, vec, ...]` → `f(x)`

2. **FNO処理**:
   - 特徴量関数をFNOで変換
   - 出力関数から弾性率を予測

**メリット**:
- ✅ 材料記述子の連続的な関係を捉えられる
- ✅ 物理的な意味が明確

**デメリット**:
- ⚠️ 組成情報の直接的な利用が難しい

### アプローチ3: Hybrid FNO（推奨度: ⭐⭐⭐）

**アイデア**: 組成と材料記述子を組み合わせたハイブリッドアプローチ

**実装方法**:
1. **マルチチャンネル入力**:
   - チャンネル1: 組成データ（空間化）
   - チャンネル2: 材料記述子（空間化または条件付き）

2. **FNO処理**:
   - マルチチャンネルFNOで処理
   - 出力から弾性率を予測

**メリット**:
- ✅ 組成と材料記述子の両方を活用
- ✅ より豊富な情報を利用可能

---

## 📝 実装計画

### Phase 1: データ前処理とFNO用データローダー

**タスク**:
1. ✅ FNO用データローダーの実装
   - 組成データの空間化
   - 材料記述子の統合
   - バッチ処理の実装

2. ✅ データ検証
   - データ形状の確認
   - 正規化の実装

**ファイル**: `fno_data_loader.py`

### Phase 2: FNOモデルの実装

**タスク**:
1. ✅ 基本FNOレイヤーの実装
   - Fourier変換レイヤー
   - 線形変換レイヤー
   - 活性化関数

2. ✅ FNOモデルの実装
   - 入力埋め込み
   - FNOブロック（複数層）
   - 出力層（回帰）

3. ✅ 軽量版FNOの実装（オプション）
   - パラメータ数を削減したバージョン

**ファイル**: `fno_model.py`

**アーキテクチャ**:
```python
class FNO1d(nn.Module):
    """
    1次元FNO（組成データ用）
    """
    def __init__(
        self,
        modes: int = 16,  # Fourier modes数
        width: int = 64,  # チャンネル数
        layers: int = 4,  # FNO層数
        input_dim: int = 1,  # 入力次元（組成チャンネル数）
        output_dim: int = 1,  # 出力次元（弾性率）
        additional_feat_dim: int = 8  # 追加特徴量次元
    ):
        ...
```

### Phase 3: 訓練スクリプトの実装

**タスク**:
1. ✅ 訓練ループの実装
   - 損失関数（MSE, Huber Loss）
   - オプティマイザ（AdamW）
   - 学習率スケジューラー

2. ✅ 評価機能の実装
   - R², RMSE, MAEの計算
   - 可視化機能

3. ✅ Early stoppingの実装

**ファイル**: `train_fno.py`

### Phase 4: 推論と評価

**タスク**:
1. ✅ 推論スクリプトの実装
   - モデルの読み込み
   - バッチ推論
   - 結果の保存

2. ✅ モデル比較
   - GNN, Transformer, FNOの比較
   - アンサンブル（オプション）

**ファイル**: `inference_fno.py`

### Phase 5: ドキュメントとテスト

**タスク**:
1. ✅ READMEの作成
2. ✅ 使用例の作成
3. ✅ ハイパーパラメータの説明

---

## 🚀 他のモデル案

### 1. Neural ODE (Neural Ordinary Differential Equations) ⭐⭐⭐⭐

**概要**:
- 連続的な動的システムとして材料特性をモデル化
- 組成や処理条件の連続的な変化を捉える

**適用可能性**:
- ✅ 熱処理過程での特性変化の予測
- ✅ 組成の連続的な変化に対する特性予測
- ✅ 物理法則に基づいたモデリング

**実装難易度**: 中〜高

**参考**:
- Chen et al. (2018) "Neural Ordinary Differential Equations"
- 材料科学への応用: 相変態、拡散過程のモデリング

### 2. DeepONet (Deep Operator Network) ⭐⭐⭐⭐⭐

**概要**:
- 別のニューラルオペレーター手法
- BranchネットワークとTrunkネットワークの組み合わせ

**適用可能性**:
- ✅ FNOと同様に組成-特性マッピングに適用可能
- ✅ より柔軟なアーキテクチャ

**実装難易度**: 中

**参考**:
- Lu et al. (2021) "Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators"

### 3. Physics-Informed Neural Networks (PINNs) ⭐⭐⭐⭐

**概要**:
- 物理法則を損失関数に組み込む
- データが少ない場合でも物理的制約により予測精度向上

**適用可能性**:
- ✅ 材料科学の物理法則（弾性理論、熱力学等）を組み込める
- ✅ データ不足の問題を緩和

**実装難易度**: 中

**参考**:
- Raissi et al. (2019) "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations"

### 4. Graph Neural Operator ⭐⭐⭐⭐

**概要**:
- GNNとオペレーター学習を組み合わせ
- 既存のGNN実装を拡張

**適用可能性**:
- ✅ 既存のGNNコードを活用可能
- ✅ グラフ構造とオペレーター学習の両方の利点

**実装難易度**: 中

**参考**:
- Li et al. (2020) "Neural Operator: Graph Kernel Network for Partial Differential Equations"

### 5. Crystal Graph Convolutional Networks (CGCNN) ⭐⭐⭐⭐⭐

**概要**:
- 結晶構造に特化したGNN
- 材料科学で広く使用されている

**適用可能性**:
- ✅ 結晶構造情報がある場合に非常に有効
- ✅ 実績のある手法

**実装難易度**: 低〜中

**参考**:
- Xie & Grossman (2018) "Crystal Graph Convolutional Neural Networks for an Accurate and Interpretable Prediction of Material Properties"

### 6. MatErials Graph Networks (MEGNet) ⭐⭐⭐⭐⭐

**概要**:
- Materials Projectで開発されたGNN
- 材料科学に特化した設計

**適用可能性**:
- ✅ 材料特性予測に最適化されている
- ✅ 豊富なドキュメントと実装例

**実装難易度**: 低〜中

**参考**:
- Chen et al. (2019) "Graph Networks as a Universal Machine Learning Framework for Molecules and Crystals"

### 7. Transformer-based Operator Learning ⭐⭐⭐

**概要**:
- Transformerアーキテクチャをオペレーター学習に適用
- 既存のTransformer実装を拡張

**適用可能性**:
- ✅ 既存のTransformerコードを活用可能
- ✅ 注意機構による長距離依存関係の学習

**実装難易度**: 中

**参考**:
- Cao (2021) "Choose a Transformer: Fourier or Galerkin"

---

## 📊 モデル比較表

| モデル | 実装難易度 | データ要件 | 物理的制約 | 推奨度 |
|--------|-----------|-----------|-----------|--------|
| **FNO** | 中 | 中 | 低 | ⭐⭐⭐⭐ |
| **Neural ODE** | 高 | 低 | 高 | ⭐⭐⭐⭐ |
| **DeepONet** | 中 | 中 | 低 | ⭐⭐⭐⭐⭐ |
| **PINNs** | 中 | 低 | 高 | ⭐⭐⭐⭐ |
| **Graph Neural Operator** | 中 | 中 | 低 | ⭐⭐⭐⭐ |
| **CGCNN** | 低〜中 | 中 | 低 | ⭐⭐⭐⭐⭐ |
| **MEGNet** | 低〜中 | 中 | 低 | ⭐⭐⭐⭐⭐ |
| **Transformer Operator** | 中 | 中 | 低 | ⭐⭐⭐ |

---

## 🎯 推奨実装順序

### Phase 1: FNO実装（優先度: 高）
1. ✅ FNOの実装（この計画）
2. ✅ GNN, Transformerとの比較

### Phase 2: 追加モデル（優先度: 中）
1. **DeepONet**: FNOと比較可能なオペレーター手法
2. **MEGNet**: 材料科学に特化したGNN（既存GNNの改善）

### Phase 3: 高度なモデル（優先度: 低）
1. **Neural ODE**: 連続的な動的システムのモデリング
2. **PINNs**: 物理法則の組み込み

---

## 📅 実装スケジュール

### Week 1: FNO実装
- Day 1-2: データローダーの実装
- Day 3-4: FNOモデルの実装
- Day 5: 訓練スクリプトの実装

### Week 2: 評価と最適化
- Day 1-2: 訓練と評価
- Day 3-4: ハイパーパラメータ調整
- Day 5: モデル比較とドキュメント

### Week 3: 追加モデル（オプション）
- DeepONetまたはMEGNetの実装

---

## 🔧 技術スタック

- **フレームワーク**: PyTorch
- **Fourier変換**: `torch.fft`
- **データ処理**: NumPy, Pandas
- **可視化**: Matplotlib, Seaborn
- **評価**: scikit-learn

---

## 📚 参考資料

1. **FNO論文**:
   - Li et al. (2020) "Fourier Neural Operator for Parametric Partial Differential Equations"
   - Li et al. (2021) "Fourier Neural Operator for Parametric Partial Differential Equations"

2. **実装例**:
   - NeuralOperators.jl (Julia)
   - PyTorch実装例（GitHub）

3. **材料科学への応用**:
   - 材料特性予測へのFNO適用例を調査

---

## ✅ 次のステップ

1. ✅ この計画のレビューと承認
2. ✅ FNOデータローダーの実装開始
3. ✅ FNOモデルの実装開始

---

**作成者**: AI Assistant  
**最終更新**: 2026年1月23日
