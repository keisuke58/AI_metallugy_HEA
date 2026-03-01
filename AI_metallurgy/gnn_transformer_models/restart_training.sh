#!/bin/bash
# 訓練の再開スクリプト（argparse修正後）

cd "$(dirname "$0")"

# Condaの初期化
if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    source "/opt/conda/etc/profile.d/conda.sh"
elif [ -f "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" ]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
fi

# 環境の確認
if conda env list | grep -q "^hea_gnn "; then
    echo "✅ hea_gnn環境が見つかりました"
    conda activate hea_gnn
    
    echo ""
    echo "=========================================="
    echo "訓練を再開します"
    echo "=========================================="
    echo ""
    
    # データファイルの確認
    DATA_PATH="../data_collection/processed_data/data_with_features.csv"
    if [ -f "$DATA_PATH" ]; then
        echo "✅ データファイル: $DATA_PATH"
    else
        echo "❌ データファイルが見つかりません: $DATA_PATH"
        exit 1
    fi
    
    # ディレクトリの作成
    mkdir -p models results
    
    # 訓練の開始
    python train.py --model both --batch_size 8 --num_epochs 50 --early_stopping_patience 15 2>&1 | tee training_log.txt
    
    echo ""
    echo "=========================================="
    echo "訓練完了"
    echo "=========================================="
else
    echo "❌ hea_gnn環境が見つかりません"
    echo "   先に setup_conda_and_train.sh を実行してください"
    exit 1
fi
