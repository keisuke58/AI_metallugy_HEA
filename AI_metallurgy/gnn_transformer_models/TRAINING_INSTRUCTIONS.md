# 訓練開始手順

## 現在の状況

必要なPythonパッケージがインストールされていません。以下の手順でインストールしてから訓練を開始してください。

## 方法1: 自動インストールスクリプト（推奨）

```bash
cd /home/nishioka/LUH/AI_metallurgy/gnn_transformer_models
bash install_and_train.sh
```

このスクリプトは以下を実行します：
1. pip3の確認
2. 必要なパッケージのインストール
3. データファイルの確認
4. 訓練の開始

## 方法2: 手動インストール

### Step 1: pip3のインストール（必要な場合）

```bash
sudo apt-get update
sudo apt-get install -y python3-pip
```

### Step 2: パッケージのインストール

```bash
cd /home/nishioka/LUH/AI_metallurgy/gnn_transformer_models

# PyTorchとその依存パッケージ
pip3 install --user torch torchvision torchaudio

# その他のパッケージ
pip3 install --user pandas numpy scikit-learn matplotlib seaborn tqdm

# PyTorch Geometric
pip3 install --user torch-geometric
```

### Step 3: インストール確認

```bash
python3 -c "import torch; import pandas as pd; print('✅ インストール成功')"
```

### Step 4: 訓練の開始

```bash
python3 train.py --model both --batch_size 8 --num_epochs 50 --early_stopping_patience 15
```

## 方法3: Conda環境を使用（推奨）

Condaが利用可能な場合：

```bash
# 新しい環境を作成
conda create -n hea_gnn python=3.8 -y
conda activate hea_gnn

# パッケージのインストール
conda install pytorch torchvision torchaudio -c pytorch -y
conda install pandas numpy scikit-learn matplotlib seaborn tqdm -y
pip install torch-geometric

# 訓練の開始
cd /home/nishioka/LUH/AI_metallurgy/gnn_transformer_models
python train.py --model both --batch_size 8 --num_epochs 50 --early_stopping_patience 15
```

## 訓練の監視

訓練が開始されると、以下の情報が表示されます：

1. **エポックごとの進捗**
   - Train Loss, R², RMSE, MAE
   - Val Loss, R², RMSE, MAE

2. **最良モデルの保存**
   - `models/gnn_best_model.pth`
   - `models/transformer_best_model.pth`

3. **結果の保存**
   - `results/gnn_results.json`
   - `results/transformer_results.json`
   - `results/model_comparison.json`
   - `results/*.png` (可視化画像)

## 訓練パラメータの調整

訓練スクリプトのオプション：

```bash
python3 train.py \
  --model both \                    # gnn, transformer, both
  --batch_size 8 \                  # バッチサイズ（メモリに応じて調整）
  --learning_rate 0.001 \           # 学習率
  --num_epochs 50 \                 # エポック数
  --early_stopping_patience 15 \    # Early stopping patience
  --use_light_model                 # 軽量版モデルを使用
```

## トラブルシューティング

### エラー: "No module named 'torch'"

→ パッケージがインストールされていません。上記の手順でインストールしてください。

### エラー: "CUDA out of memory"

→ バッチサイズを減らしてください：
```bash
python3 train.py --batch_size 4
```

### エラー: "File not found: data_with_features.csv"

→ データファイルのパスを確認してください：
```bash
ls ../data_collection/processed_data/data_with_features.csv
```

## 訓練時間の目安

- **322サンプル、CPU**: 約30-60分
- **322サンプル、GPU**: 約5-10分
- **軽量版モデル**: より高速

## 次のステップ

訓練が完了したら：

1. 結果の確認：
   ```bash
   cat results/model_comparison.json
   ```

2. 可視化画像の確認：
   ```bash
   ls results/*.png
   ```

3. 推論の実行：
   ```bash
   python3 inference.py --model both
   ```
