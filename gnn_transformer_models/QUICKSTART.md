# クイックスタートガイド

## 🚀 5分で始める

### Step 1: 環境セットアップ

```bash
cd /home/nishioka/LUH/AI_metallurgy/gnn_transformer_models

# 依存パッケージのインストール
pip install -r requirements.txt

# PyTorch Geometricのインストール（必要に応じて）
pip install torch-geometric
```

### Step 2: データの確認

データファイルが存在することを確認：

```bash
ls ../data_collection/processed_data/data_with_features.csv
```

### Step 3: モデルの訓練

```bash
python train.py
```

これで以下が実行されます：
- GNNモデルの訓練
- Transformerモデルの訓練
- 結果の可視化と保存

### Step 4: 結果の確認

訓練後、以下のファイルが生成されます：

```bash
# 結果JSON
cat results/gnn_results.json
cat results/transformer_results.json
cat results/model_comparison.json

# 可視化画像
ls results/*.png
```

### Step 5: 推論（オプション）

訓練済みモデルで予測を実行：

```bash
python inference.py --model both
```

## 📊 期待される出力

訓練が成功すると、以下のような出力が表示されます：

```
================================================================================
GNN Model Training
================================================================================
✅ 322サンプルを読み込みました
📊 データ分割: Train=257, Val=32, Test=33

Epoch 1/200
Training: 100%|████████████| 17/17 [00:05<00:00,  3.21it/s]
Evaluating: 100%|██████████| 2/2 [00:00<00:00,  4.56it/s]
Train Loss: 1234.5678, R²: 0.1234, RMSE: 35.1234, MAE: 28.5678
Val Loss: 1456.7890, R²: 0.0987, RMSE: 38.1234, MAE: 30.1234
✅ Best model saved!

...

Test Evaluation
================================================================================
Test Loss: 1234.5678
Test R²: 0.6543
Test RMSE: 34.5678 GPa
Test MAE: 27.1234 GPa
```

## ⚙️ カスタマイズ

### 軽量版モデルの使用

デフォルトで軽量版が使用されます。フルモデルを使用する場合：

`train.py`を編集：

```python
CONFIG = {
    ...
    'use_light_model': False,  # True → False
    ...
}
```

### バッチサイズの調整

メモリ不足の場合：

```python
CONFIG = {
    ...
    'batch_size': 8,  # 16 → 8
    ...
}
```

## 🐛 トラブルシューティング

### エラー: "No module named 'torch_geometric'"

```bash
pip install torch-geometric
```

### エラー: "CUDA out of memory"

`train.py`でバッチサイズを減らす：

```python
CONFIG['batch_size'] = 4
```

### エラー: "File not found: data_with_features.csv"

データファイルのパスを確認：

```python
# train.py内で確認
DATA_PATH = Path(__file__).parent.parent / "data_collection" / "processed_data" / "data_with_features.csv"
print(DATA_PATH)  # パスを確認
```

## 📚 詳細情報

- 詳細な使用方法: `README.md`
- 実装の詳細: `IMPLEMENTATION_SUMMARY.md`

## 💡 ヒント

1. **最初の訓練**: 軽量版モデル（`use_light_model=True`）で開始することを推奨
2. **GPU使用**: CUDA対応GPUがある場合、自動的に使用されます
3. **Early Stopping**: 過学習を防ぐため、デフォルトで有効です
4. **結果の比較**: `results/model_comparison.json`でGNNとTransformerの性能を比較できます
