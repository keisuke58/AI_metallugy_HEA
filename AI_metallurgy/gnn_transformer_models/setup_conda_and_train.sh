#!/bin/bash
# Conda環境の作成と訓練実行スクリプト

set -e

echo "=========================================="
echo "HEA GNN & Transformer Conda環境セットアップ"
echo "=========================================="

cd "$(dirname "$0")"

# Condaの初期化（複数の一般的なパスを試行）
if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    source "/opt/conda/etc/profile.d/conda.sh"
elif [ -f "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" ]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
fi

# Condaの確認
if ! command -v conda &> /dev/null; then
    echo "❌ condaが見つかりません"
    echo "   Condaをインストールするか、PATHに追加してください"
    echo ""
    echo "手動でcondaを初期化する場合:"
    echo "   source ~/anaconda3/etc/profile.d/conda.sh"
    echo "   または"
    echo "   source ~/miniconda3/etc/profile.d/conda.sh"
    exit 1
fi

echo "✅ condaが見つかりました: $(which conda)"

# 環境名
ENV_NAME="hea_gnn"

# 既存の環境を確認
if conda env list | grep -q "^${ENV_NAME} "; then
    echo ""
    echo "⚠️  環境 '${ENV_NAME}' は既に存在します"
    read -p "削除して再作成しますか？ (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "環境を削除中..."
        conda env remove -n ${ENV_NAME} -y
    else
        echo "既存の環境を使用します"
        USE_EXISTING=true
    fi
else
    USE_EXISTING=false
fi

# 環境の作成（存在しない場合）
if [ "$USE_EXISTING" != "true" ]; then
    echo ""
    echo "新しいConda環境を作成中: ${ENV_NAME}"
    echo "（これには数分かかる場合があります）"
    conda create -n ${ENV_NAME} python=3.8 -y
fi

# 環境のアクティベート
echo ""
echo "環境をアクティベート中..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ${ENV_NAME}

# Pythonバージョンの確認
echo ""
echo "Python環境の確認..."
python --version
which python

# PyTorchのインストール
echo ""
echo "PyTorchをインストール中..."
echo "（これには数分かかる場合があります）"
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y || {
    echo "⚠️  CPU版のインストールに失敗。GPU版を試行します..."
    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y || {
        echo "❌ PyTorchのインストールに失敗しました"
        exit 1
    }
}

# その他のパッケージのインストール
echo ""
echo "その他のパッケージをインストール中..."
conda install pandas numpy scikit-learn matplotlib seaborn tqdm -y

# PyTorch Geometricのインストール
echo ""
echo "PyTorch Geometricをインストール中..."
pip install torch-geometric

# インストール確認
echo ""
echo "インストール確認..."
python -c "
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
print(f'✅ PyTorch: {torch.__version__}')
print(f'✅ CUDA available: {torch.cuda.is_available()}')
print(f'✅ Pandas: {pd.__version__}')
print(f'✅ NumPy: {np.__version__}')
print('✅ すべてのパッケージがインストールされました')
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
echo "環境: ${ENV_NAME}"
echo ""

python train.py --model both --batch_size 8 --num_epochs 50 --early_stopping_patience 15

echo ""
echo "=========================================="
echo "訓練完了"
echo "=========================================="
echo ""
echo "環境を終了するには: conda deactivate"
echo "環境を再度アクティベートするには: conda activate ${ENV_NAME}"
