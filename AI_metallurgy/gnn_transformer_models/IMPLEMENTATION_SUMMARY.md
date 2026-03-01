# GNN & Transformer実装サマリー

## 📋 実装概要

最新研究（2024-2025）に基づいて、HEA（高エントロピー合金）の弾性率予測のためのGNNとTransformerモデルを実装しました。

## 🎯 実装したモデル

### 1. Graph Neural Network (GNN)

#### アーキテクチャ
- **ベース研究**: MatGNet (2025), ALIGNN, DenseGNN (2025)
- **主要コンポーネント**:
  - **EdgeGatedConv**: エッジゲーティング畳み込み層
    - エッジ特徴量とノード特徴量を統合
    - ゲート機構で重要な相互作用を強調
  - **SelfAttentionLayer**: 自己注意層
    - グローバルな元素間相互作用を捉える
  - **Graph Pooling**: Mean, Max, Sum poolingの組み合わせ

#### グラフ構造
- **ノード**: 各元素（Ti, Zr, Nb, Ta, etc.）
- **ノード特徴量**: [原子番号, 原子半径, 電気陰性度, VEC, 組成比]
- **エッジ**: 完全グラフ（すべての元素間の相互作用）
- **エッジ特徴量**: [組成積, 原子半径差, 電気陰性度差]

#### モデルバリエーション
- **HEAGNN**: フルモデル（hidden_dim=128, num_layers=4）
- **HEAGNNLight**: 軽量版（hidden_dim=64, num_layers=3）

### 2. Transformer

#### アーキテクチャ
- **ベース研究**: Crystalformer (2024), AlloyBERT (2024), CrysCo (2025)
- **主要コンポーネント**:
  - **CompositionEmbedding**: 組成埋め込み
    - 元素トークン + 組成比の統合埋め込み
  - **PositionalEncoding**: 位置エンコーディング
    - シーケンス内の位置情報をエンコード
  - **TransformerEncoderLayer**: Multi-head self-attention + Feed-forward
    - 元素間の長距離依存関係を捉える

#### 入力形式
- **シーケンス**: 組成比の降順でソートされた元素トークン列
- **特殊トークン**: [CLS], [SEP], [PAD]
- **追加特徴量**: 材料記述子（mixing_entropy, mixing_enthalpy, etc.）

#### モデルバリエーション
- **HEATransformer**: フルモデル（d_model=256, num_layers=4）
- **HEATransformerLight**: 軽量版（d_model=128, num_layers=3）

## 📁 ファイル構造

```
gnn_transformer_models/
├── __init__.py
├── data_loader.py          # データローダー（GNN/Transformer用）
├── gnn_model.py           # GNNモデル実装
├── transformer_model.py   # Transformerモデル実装
├── train.py               # 訓練スクリプト
├── inference.py           # 推論スクリプト
├── requirements.txt       # 依存パッケージ
├── README.md              # 使用方法
├── IMPLEMENTATION_SUMMARY.md  # このファイル
├── models/                # 訓練済みモデル（生成される）
│   ├── gnn_best_model.pth
│   └── transformer_best_model.pth
└── results/               # 結果（生成される）
    ├── gnn_results.json
    ├── transformer_results.json
    ├── model_comparison.json
    ├── gnn_results.png
    └── transformer_results.png
```

## 🔬 技術的特徴

### 最新研究の取り入れ

1. **MatGNetスタイルのエッジゲーティング**
   - エッジ特徴量とノード特徴量の統合
   - ゲート機構による重要度の学習

2. **ALIGNNスタイルの角度特徴量**
   - エッジ特徴量に原子半径差、電気陰性度差を含める
   - 元素間の幾何学的関係を考慮

3. **Crystalformerスタイルの位置エンコーディング**
   - シーケンス内の位置情報を保持
   - 組成の順序を考慮

4. **AlloyBERTスタイルの組成エンコーディング**
   - 元素トークンと組成比の統合埋め込み
   - 材料記述子の追加

### データ効率性

- **軽量版モデル**: データ数が少ない場合（<500サンプル）に最適化
- **Early Stopping**: 過学習を防止
- **勾配クリッピング**: 訓練の安定化
- **学習率スケジューリング**: ReduceLROnPlateau

## 📊 期待される性能

既存の研究結果（R² = 0.59-0.67）と比較して、以下の改善が期待されます：

- **GNN**: グラフ構造により元素間の相互作用をより正確にモデル化
- **Transformer**: シーケンスモデルにより組成パターンを捉える

ただし、データ数が限られている（322サンプル）ため、以下の点に注意が必要です：

1. **過学習のリスク**: Early stoppingと正則化が重要
2. **データ拡張**: 可能であればデータ数を増やす
3. **転移学習**: 類似材料のデータを活用

## 🚀 使用方法

### 訓練

```bash
cd /home/nishioka/LUH/AI_metallurgy/gnn_transformer_models
python train.py
```

### 推論

```bash
python inference.py --model both
```

### 個別モデルの使用

```bash
# GNNのみ
python inference.py --model gnn

# Transformerのみ
python inference.py --model transformer
```

## 🔧 カスタマイズ

### ハイパーパラメータの調整

`train.py`の`CONFIG`を編集：

```python
CONFIG = {
    'batch_size': 16,
    'learning_rate': 1e-3,
    'num_epochs': 200,
    'use_light_model': True,  # Falseにするとフルモデルを使用
    ...
}
```

### モデルアーキテクチャの変更

`gnn_model.py`または`transformer_model.py`で直接編集：

```python
# GNNの隠れ層サイズを変更
model = HEAGNN(
    hidden_dim=256,  # 128から256に変更
    num_layers=6,    # 4から6に変更
    ...
)
```

## 📚 参考論文

1. **MatGNet** (2025): "MatGNet: A Graph Neural Network for Crystal Property Prediction"
2. **ALIGNN**: "Atomistic Line Graph Neural Network for improved materials property predictions"
3. **Crystalformer** (2024): "Crystalformer: Transformer for Periodic Crystal Structures"
4. **AlloyBERT** (2024): "AlloyBERT: Transformer-based Property Prediction for Alloys"
5. **CrysCo** (2025): "CrysCo: A Hybrid Graph-Transformer Architecture for Crystal Property Prediction"
6. **DenseGNN** (2025): "DenseGNN: Dense Graph Neural Networks for Materials Property Prediction"

## ⚠️ 注意事項

1. **データ要件**: データファイルは`data_with_features.csv`である必要があります
2. **GPU推奨**: 訓練時間を短縮するため、GPUの使用を推奨します
3. **メモリ**: バッチサイズを調整してメモリ使用量を制御してください
4. **PyTorch Geometric**: 正しいバージョンのPyTorch Geometricをインストールしてください

## 🐛 既知の問題と解決策

### 問題1: CUDA out of memory

**解決策**: バッチサイズを減らす

```python
CONFIG['batch_size'] = 8  # または 4
```

### 問題2: データが見つからない

**解決策**: データパスを確認

```python
DATA_PATH = Path(__file__).parent.parent / "data_collection" / "processed_data" / "data_with_features.csv"
```

### 問題3: PyTorch Geometricのインストールエラー

**解決策**: PyTorchのバージョンに合わせてインストール

```bash
pip install torch-geometric -f https://data.pyg.org/whl/torch-{version}.html
```

## 📈 今後の改善案

1. **アンサンブル学習**: GNNとTransformerの予測を統合
2. **転移学習**: 類似材料のデータで事前訓練
3. **不確実性定量化**: 予測区間の推定
4. **説明可能性**: Attention重みの可視化
5. **データ拡張**: SMOTEやGANによる合成データ生成

## 📝 ライセンス

研究目的で自由に使用できます。

## 🙏 謝辞

最新の材料科学における深層学習研究に基づいて実装しました。
