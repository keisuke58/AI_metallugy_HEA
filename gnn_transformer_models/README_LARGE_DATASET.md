# 大規模データセット用訓練スクリプト

## 概要

`train_large_dataset.py`は、**5000サンプル以上の大規模データセット**に最適化された訓練スクリプトです。

通常の`train.py`と比較して、以下の最適化が行われています：

## 主な違い

### 1. **より大きなモデル容量**
- **Transformer層数**: 5 → **6層**
- **GNN層数**: 4 → **6層**
- **GNN隠れ層**: 128 → **256**
- **Transformerフィードフォワード**: 512 → **1024**

### 2. **より大きなバッチサイズ**
- **デフォルトバッチサイズ**: 32 → **64**
- データ数に応じて自動調整（最大128）

### 3. **より長い訓練**
- **エポック数**: 400 → **500**
- **ウォームアップ**: 25 → **30エポック**

### 4. **より強い正則化**
- **Dropout率**: 0.25 → **0.3**
- **Weight decay**: 2e-4 → **3e-4**

### 5. **勾配累積の活用**
- **デフォルト**: 1 → **2ステップ**
- 実質的なバッチサイズを増やす

### 6. **より多くのワーカー**
- **num_workers**: 4 → **8**（大規模データ処理の高速化）

## 使用方法

### 基本的な使用

```bash
cd /home/nishioka/LUH/AI_metallurgy/gnn_transformer_models
python train_large_dataset.py
```

### オプション指定

```bash
# Transformerのみ訓練
python train_large_dataset.py --model transformer

# GNNのみ訓練
python train_large_dataset.py --model gnn

# カスタムデータパス
python train_large_dataset.py --data_path /path/to/your/data.csv

# カスタム出力ディレクトリ
python train_large_dataset.py --output_dir ./my_results --model_dir ./my_models
```

## 出力先

- **結果**: `results_large/`
- **モデル**: `models_large/`

通常の`train.py`とは別のディレクトリに保存されるため、既存の結果と混在しません。

## データ数の確認

スクリプトは自動的にデータ数を確認し、5000未満の場合は警告を表示します。

```bash
⚠️  警告: データ数が{count}で5000未満です。通常のtrain.pyを使用することを推奨します。
続行しますか？ (y/n):
```

## 期待される性能

大規模データセット（5000+サンプル）では、以下の性能向上が期待できます：

- **R²**: 0.75-0.85以上
- **RMSE**: 25-30 GPa以下
- **より安定した訓練**: 過学習の抑制

## 注意事項

1. **メモリ使用量**: より大きなモデルとバッチサイズのため、より多くのGPU/CPUメモリが必要です
2. **訓練時間**: より長いエポック数とより大きなモデルのため、訓練時間が長くなります
3. **データ数**: 5000サンプル未満の場合は、通常の`train.py`の使用を推奨します

## ハイパーパラメータのカスタマイズ

スクリプト内の`LARGE_DATASET_CONFIG`を編集することで、ハイパーパラメータを調整できます。

```python
LARGE_DATASET_CONFIG = {
    'batch_size': 64,  # バッチサイズ
    'learning_rate': 2e-4,  # 学習率
    'num_epochs': 500,  # エポック数
    'transformer_num_layers': 6,  # Transformer層数
    'transformer_dim_feedforward': 1024,  # フィードフォワード層サイズ
    # ... その他の設定
}
```
