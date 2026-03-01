# Conda環境でのセットアップと訓練

## 🚀 クイックスタート

### 方法1: 自動セットアップスクリプト（推奨）

```bash
cd /home/nishioka/LUH/AI_metallurgy/gnn_transformer_models
bash setup_conda_and_train.sh
```

このスクリプトは以下を自動実行します：
1. Conda環境の作成（`hea_gnn`）
2. 必要なパッケージのインストール
3. データファイルの確認
4. 訓練の開始

### 方法2: 手動セットアップ

#### Step 1: Conda環境の作成

```bash
conda create -n hea_gnn python=3.8 -y
conda activate hea_gnn
```

#### Step 2: パッケージのインストール

```bash
# PyTorch（CPU版）
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y

# またはGPU版（CUDA 11.8の場合）
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# その他のパッケージ
conda install pandas numpy scikit-learn matplotlib seaborn tqdm -y

# PyTorch Geometric
pip install torch-geometric
```

#### Step 3: インストール確認

```bash
python -c "import torch; import pandas as pd; print('✅ インストール成功')"
```

#### Step 4: 訓練の開始

```bash
cd /home/nishioka/LUH/AI_metallurgy/gnn_transformer_models
python train.py --model both --batch_size 8 --num_epochs 50 --early_stopping_patience 15
```

## 📋 環境情報

- **環境名**: `hea_gnn`
- **Pythonバージョン**: 3.8
- **主要パッケージ**:
  - PyTorch
  - pandas, numpy, scikit-learn
  - matplotlib, seaborn, tqdm
  - torch-geometric

## 🔄 環境の管理

### 環境のアクティベート

```bash
conda activate hea_gnn
```

### 環境の非アクティベート

```bash
conda deactivate
```

### 環境の削除

```bash
conda env remove -n hea_gnn -y
```

### 環境の一覧表示

```bash
conda env list
```

## 🐛 トラブルシューティング

### エラー: "Could not find conda environment"

→ 環境が作成されていません。上記の手順で環境を作成してください。

### エラー: "CommandNotFoundError: Your shell has not been properly configured"

→ condaの初期化が必要です：

```bash
# Bashの場合
source $(conda info --base)/etc/profile.d/conda.sh
conda activate hea_gnn
```

### エラー: "CUDA out of memory"

→ バッチサイズを減らしてください：

```bash
python train.py --batch_size 4
```

### PyTorch Geometricのインストールエラー

→ PyTorchのバージョンに合わせてインストール：

```bash
pip install torch-geometric -f https://data.pyg.org/whl/torch-$(python -c "import torch; print(torch.__version__)").html
```

## 📊 訓練の監視

訓練が開始されたら、別のターミナルで監視：

```bash
# 環境をアクティベート
conda activate hea_gnn

# 監視スクリプトを実行
cd /home/nishioka/LUH/AI_metallurgy/gnn_transformer_models
./monitor_training.sh

# またはリアルタイム監視
watch -n 5 ./monitor_training.sh
```

## 💡 ヒント

1. **環境の再利用**: 一度作成した環境は、次回は `conda activate hea_gnn` だけで使用できます

2. **パッケージの更新**: 
   ```bash
   conda activate hea_gnn
   conda update --all
   ```

3. **環境のエクスポート**:
   ```bash
   conda activate hea_gnn
   conda env export > environment.yml
   ```

4. **環境のインポート**:
   ```bash
   conda env create -f environment.yml
   ```

## 🎯 次のステップ

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
   python inference.py --model both
   ```
