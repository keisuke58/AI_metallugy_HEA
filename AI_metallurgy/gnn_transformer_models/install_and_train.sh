#!/bin/bash
# パッケージインストールと訓練実行スクリプト

set -e

echo "=========================================="
echo "HEA GNN & Transformer 訓練セットアップ"
echo "=========================================="

cd "$(dirname "$0")"

# Python環境の確認
echo ""
echo "Python環境の確認..."
python3 --version

# pipのインストール確認とインストール
if ! command -v pip3 &> /dev/null; then
    echo ""
    echo "pip3が見つかりません。インストールを試みます..."
    if command -v apt-get &> /dev/null; then
        echo "sudo apt-get update && sudo apt-get install -y python3-pip"
        echo "上記コマンドを実行してください（sudo権限が必要です）"
        exit 1
    else
        echo "pip3を手動でインストールしてください"
        exit 1
    fi
fi

echo "✅ pip3が見つかりました: $(which pip3)"

# 必要なパッケージのインストール
echo ""
echo "必要なパッケージのインストール..."
echo "（これには数分かかる場合があります）"

pip3 install --user torch torchvision torchaudio || {
    echo "❌ PyTorchのインストールに失敗しました"
    exit 1
}

pip3 install --user pandas numpy scikit-learn matplotlib seaborn tqdm || {
    echo "❌ その他のパッケージのインストールに失敗しました"
    exit 1
}

pip3 install --user torch-geometric || {
    echo "⚠️  torch-geometricのインストールに失敗しました（後で再試行可能）"
}

# インストール確認
echo ""
echo "インストール確認..."
python3 -c "
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
print(f'✅ PyTorch: {torch.__version__}')
print(f'✅ CUDA available: {torch.cuda.is_available()}')
print(f'✅ Pandas: {pd.__version__}')
print(f'✅ NumPy: {np.__version__}')
"

# データファイルの確認
echo ""
echo "データファイルの確認..."
DATA_PATH="../data_collection/processed_data/data_with_features.csv"
if [ -f "$DATA_PATH" ]; then
    echo "✅ データファイルが見つかりました: $DATA_PATH"
    ROWS=$(wc -l < "$DATA_PATH" | tr -d ' ')
    echo "   データ数: $ROWS 行"
else
    echo "❌ データファイルが見つかりません: $DATA_PATH"
    exit 1
fi

# ディレクトリの作成
mkdir -p models results

# 訓練の開始
echo ""
echo "=========================================="
echo "訓練を開始します"
echo "=========================================="
echo ""

python3 train.py --model both --batch_size 8 --num_epochs 50 --early_stopping_patience 15

echo ""
echo "=========================================="
echo "訓練完了"
echo "=========================================="
