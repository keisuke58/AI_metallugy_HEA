# 使用方法ガイド

## 🚀 クイックスタート

### 方法1: 自動セットアップスクリプト（推奨）

```bash
cd /home/nishioka/LUH/AI_metallurgy/gnn_transformer_models
bash setup_and_run.sh
```

### 方法2: 手動セットアップ

#### Step 1: 環境セットアップ

```bash
cd /home/nishioka/LUH/AI_metallurgy/gnn_transformer_models

# 依存パッケージのインストール
pip install -r requirements.txt

# PyTorch Geometricのインストール
pip install torch-geometric
```

**注意**: PyTorch GeometricはPyTorchのバージョンに依存します。エラーが出る場合は：

```bash
# PyTorchのバージョンを確認
python -c "import torch; print(torch.__version__)"

# 対応するバージョンをインストール
pip install torch-geometric -f https://data.pyg.org/whl/torch-{version}.html
```

#### Step 2: データの確認

```bash
# データファイルの存在確認
ls ../data_collection/processed_data/data_with_features.csv

# データの行数を確認
wc -l ../data_collection/processed_data/data_with_features.csv
```

#### Step 3: モデルの訓練

```bash
python train.py
```

**実行内容**:
- GNNモデルの訓練
- Transformerモデルの訓練
- 結果の可視化と保存
- モデル性能の比較

**出力**:
- `models/gnn_best_model.pth`: 最良GNNモデル
- `models/transformer_best_model.pth`: 最良Transformerモデル
- `results/gnn_results.json`: GNN結果
- `results/transformer_results.json`: Transformer結果
- `results/model_comparison.json`: モデル比較
- `results/*.png`: 可視化画像

#### Step 4: 推論（訓練後）

```bash
# 両方のモデルで予測
python inference.py --model both

# GNNのみ
python inference.py --model gnn

# Transformerのみ
python inference.py --model transformer
```

## 📊 詳細な使用方法

### 訓練オプション

`train.py`の`CONFIG`を編集して設定を変更：

```python
CONFIG = {
    'batch_size': 16,              # バッチサイズ（メモリに応じて調整）
    'learning_rate': 1e-3,         # 学習率
    'num_epochs': 200,             # 最大エポック数
    'early_stopping_patience': 30,  # Early stoppingのpatience
    'train_ratio': 0.8,            # 訓練データ比率
    'val_ratio': 0.1,              # 検証データ比率
    'test_ratio': 0.1,             # テストデータ比率
    'device': 'cuda',              # 'cuda' or 'cpu' or 'auto'
    'use_light_model': True,       # 軽量版モデルを使用
}
```

### 推論オプション

```bash
# 基本使用
python inference.py --model both

# カスタムデータパス
python inference.py --data_path /path/to/data.csv --model both

# カスタムモデルパス
python inference.py --gnn_model models/custom_gnn.pth --model gnn

# CPU使用（GPUがない場合）
python inference.py --device cpu --model both

# フルモデルを使用（軽量版ではない）
python inference.py --use_light --model both  # 軽量版
python inference.py --model both  # デフォルト（軽量版）
```

## 🔧 トラブルシューティング

### エラー1: "No module named 'torch_geometric'"

**解決策**:
```bash
pip install torch-geometric
```

または、PyTorchのバージョンに合わせて：
```bash
pip install torch-geometric -f https://data.pyg.org/whl/torch-{version}.html
```

### エラー2: "CUDA out of memory"

**解決策**: バッチサイズを減らす

`train.py`を編集：
```python
CONFIG['batch_size'] = 8  # または 4
```

### エラー3: "File not found: data_with_features.csv"

**解決策**: データパスを確認

```python
# train.py内で確認
DATA_PATH = Path(__file__).parent.parent / "data_collection" / "processed_data" / "data_with_features.csv"
print(DATA_PATH)  # パスを確認
```

または、カスタムパスを指定：
```bash
python train.py --data_path /path/to/your/data.csv
```

### エラー4: "RuntimeError: Expected tensor for argument #1 'indices' to have one of the following scalar types: Long, Int"

**解決策**: PyTorchのバージョンを確認し、必要に応じて更新：
```bash
pip install --upgrade torch
```

### エラー5: 訓練が遅い

**解決策**:
1. GPUが使用されているか確認：
   ```python
   import torch
   print(torch.cuda.is_available())
   ```

2. バッチサイズを増やす（メモリに余裕がある場合）：
   ```python
   CONFIG['batch_size'] = 32
   ```

3. 軽量版モデルを使用（デフォルト）：
   ```python
   CONFIG['use_light_model'] = True
   ```

## 📈 性能最適化のヒント

### 1. データ数が少ない場合（<500サンプル）

- 軽量版モデルを使用（`use_light_model=True`）
- バッチサイズを小さく（8-16）
- Early stoppingのpatienceを大きく（30-50）
- データ拡張を検討

### 2. データ数が多い場合（>1000サンプル）

- フルモデルを使用（`use_light_model=False`）
- バッチサイズを大きく（32-64）
- エポック数を増やす（200-500）

### 3. GPUメモリが限られている場合

- バッチサイズを減らす（4-8）
- 軽量版モデルを使用
- 勾配累積を使用（実装が必要）

## 🎯 主な特徴の確認

実装されている主な特徴：

### GNNモデル
- ✅ エッジゲーティング畳み込み（MatGNetスタイル）
- ✅ 自己注意機構
- ✅ 角度特徴量（ALIGNNスタイル）
- ✅ 材料記述子の統合
- ✅ 残差接続
- ✅ ドロップアウト正則化

### Transformerモデル
- ✅ 組成シーケンスエンコーディング
- ✅ 位置エンコーディング
- ✅ Multi-head self-attention
- ✅ 材料記述子の統合
- ✅ 残差接続とLayer Normalization

### 訓練機能
- ✅ Early stopping
- ✅ 学習率スケジューリング
- ✅ 勾配クリッピング
- ✅ 自動可視化
- ✅ モデル保存

## 📚 参考資料

- 詳細な実装説明: `IMPLEMENTATION_SUMMARY.md`
- クイックスタート: `QUICKSTART.md`
- README: `README.md`

## 💡 よくある質問

### Q: どのモデルが良いですか？

A: データ数と計算リソースによります：
- **データ数 < 500**: 軽量版モデル（デフォルト）
- **データ数 > 1000**: フルモデル
- **計算リソースが限られている**: 軽量版モデル

### Q: 訓練にどのくらい時間がかかりますか？

A: データ数とハードウェアによります：
- **322サンプル、GPU**: 約5-10分
- **322サンプル、CPU**: 約30-60分
- **1000サンプル、GPU**: 約15-30分

### Q: 既存のモデルと比較してどうですか？

A: 最新研究（2024-2025）に基づく実装で、以下の改善が期待されます：
- グラフ構造による元素間相互作用のモデル化（GNN）
- シーケンスモデルによる組成パターンの捕捉（Transformer）

ただし、データ数が限られているため、過学習に注意が必要です。
