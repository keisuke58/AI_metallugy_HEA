# GNN & Transformer Models for HEA Elastic Modulus Prediction

最新研究（2024-2025）に基づくGraph Neural Network (GNN)とTransformerモデルの実装です。

## 📋 概要

本実装は、高エントロピー合金（HEA）の弾性率予測のための深層学習モデルを提供します。

### 実装したモデル

1. **Graph Neural Network (GNN)**
   - **ベース**: ALIGNN/MatGNetスタイル
   - **特徴**:
     - エッジゲーティング畳み込み
     - 自己注意機構
     - 角度特徴量の考慮
     - 材料記述子の統合

2. **Transformer**
   - **ベース**: Crystalformer/AlloyBERTスタイル
   - **特徴**:
     - 組成シーケンスのエンコーディング
     - Multi-head attention
     - 位置エンコーディング
     - 材料記述子の統合

## 🚀 セットアップ

### 必要な環境

- Python 3.8以上
- CUDA対応GPU（推奨、CPUでも動作可能）

### インストール

```bash
cd /home/nishioka/LUH/AI_metallurgy/gnn_transformer_models
pip install -r requirements.txt
```

### PyTorch Geometricのインストール

PyTorch GeometricはPyTorchのバージョンに応じてインストールが必要です：

```bash
# CPU版
pip install torch-geometric

# GPU版（CUDA 11.8の場合）
pip install torch-geometric -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

詳細は[PyTorch Geometric公式ドキュメント](https://pytorch-geometric.readthedocs.io/)を参照してください。

## 📊 データ形式

データファイルは以下の形式である必要があります：

- パス: `../data_collection/processed_data/data_with_features.csv`
- 必須カラム:
  - `elastic_modulus`: 目的変数（弾性率、GPa）
  - `comp_*`: 組成カラム（例: `comp_Ti`, `comp_Zr`, ...）
  - 材料記述子: `mixing_entropy`, `mixing_enthalpy`, `vec`, `delta_r`, `delta_chi`, `mean_atomic_radius`, `mean_electronegativity`, `density`

## 🎯 使用方法

### 訓練

```bash
python train.py
```

このスクリプトは以下を実行します：
1. GNNモデルの訓練
2. Transformerモデルの訓練
3. 結果の可視化
4. モデル比較

### 出力

訓練後、以下のファイルが生成されます：

```
gnn_transformer_models/
├── models/
│   ├── gnn_best_model.pth          # 最良GNNモデル
│   └── transformer_best_model.pth  # 最良Transformerモデル
└── results/
    ├── gnn_results.json            # GNN結果
    ├── transformer_results.json    # Transformer結果
    ├── model_comparison.json       # モデル比較
    ├── gnn_results.png             # GNN可視化
    └── transformer_results.png     # Transformer可視化
```

## 🔧 ハイパーパラメータ設定

`train.py`の`CONFIG`辞書で設定を変更できます：

```python
CONFIG = {
    'batch_size': 16,              # バッチサイズ
    'learning_rate': 1e-3,         # 学習率
    'num_epochs': 200,             # エポック数
    'early_stopping_patience': 30, # Early stoppingのpatience
    'train_ratio': 0.8,            # 訓練データ比率
    'val_ratio': 0.1,              # 検証データ比率
    'test_ratio': 0.1,              # テストデータ比率
    'use_light_model': True,       # 軽量版モデルを使用（データ数が少ない場合）
}
```

## 📈 モデルアーキテクチャ

### GNNモデル

- **ノード特徴量**: [原子番号, 原子半径, 電気陰性度, VEC, 組成比]
- **エッジ特徴量**: [組成積, 原子半径差, 電気陰性度差]
- **グラフ構造**: 完全グラフ（すべての元素間の相互作用をモデル化）

### Transformerモデル

- **入力**: 組成シーケンス（元素トークン + 組成比）
- **エンコーディング**: 位置エンコーディング + 組成埋め込み
- **アーキテクチャ**: Multi-head self-attention + Feed-forward

## 📚 参考研究

### GNN関連
- **MatGNet** (2025): Mat2vec embeddings + angular features + gated convolution
- **ALIGNN**: Angle-aware graph neural networks for crystal property prediction
- **DenseGNN** (2025): Deeper GNNs with dense connectivity

### Transformer関連
- **Crystalformer** (2024): Transformer for periodic crystal structures
- **AlloyBERT** (2024): Transformer for alloy property prediction
- **CrysCo** (2025): Hybrid GNN-Transformer architecture

## 🐛 トラブルシューティング

### CUDA out of memory

バッチサイズを小さくしてください：

```python
CONFIG['batch_size'] = 8  # または 4
```

### データが見つからない

データファイルのパスを確認してください：

```python
DATA_PATH = Path(__file__).parent.parent / "data_collection" / "processed_data" / "data_with_features.csv"
```

### PyTorch Geometricのインストールエラー

PyTorchのバージョンに合わせてインストールしてください：

```bash
# PyTorchのバージョンを確認
python -c "import torch; print(torch.__version__)"

# 対応するPyTorch Geometricをインストール
pip install torch-geometric -f https://data.pyg.org/whl/torch-{version}.html
```

## 📝 ライセンス

この実装は研究目的で使用できます。

## 🙏 謝辞

最新の材料科学における深層学習研究に基づいて実装しました。
