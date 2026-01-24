#!/bin/bash
# セットアップと実行スクリプト

echo "=========================================="
echo "HEA GNN & Transformer セットアップ"
echo "=========================================="

# ディレクトリに移動
cd "$(dirname "$0")"

# Step 1: 依存パッケージのインストール
echo ""
echo "Step 1: 依存パッケージのインストール"
echo "----------------------------------------"
pip install -r requirements.txt

# PyTorch Geometricのインストール
echo ""
echo "PyTorch Geometricのインストール"
echo "----------------------------------------"
pip install torch-geometric

# Step 2: データファイルの確認
echo ""
echo "Step 2: データファイルの確認"
echo "----------------------------------------"
DATA_PATH="../data_collection/processed_data/data_with_features.csv"
if [ -f "$DATA_PATH" ]; then
    echo "✅ データファイルが見つかりました: $DATA_PATH"
    echo "   データ数: $(wc -l < "$DATA_PATH" | tr -d ' ') 行"
else
    echo "❌ データファイルが見つかりません: $DATA_PATH"
    echo "   先にデータ前処理を実行してください"
    exit 1
fi

# Step 3: ディレクトリの作成
echo ""
echo "Step 3: 出力ディレクトリの作成"
echo "----------------------------------------"
mkdir -p models results
echo "✅ ディレクトリを作成しました"

# Step 4: 訓練の実行（オプション）
echo ""
echo "=========================================="
echo "セットアップ完了！"
echo "=========================================="
echo ""
echo "次のコマンドで訓練を開始できます:"
echo "  python train.py"
echo ""
echo "または推論を実行:"
echo "  python inference.py --model both"
echo ""
