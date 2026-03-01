#!/bin/bash
# 訓練の監視スクリプト

cd "$(dirname "$0")"

echo "=========================================="
echo "訓練監視ツール"
echo "=========================================="
echo ""

# ログファイルの確認
if [ -f "training_log.txt" ]; then
    echo "📊 最新の訓練ログ（最後の30行）:"
    echo "----------------------------------------"
    tail -30 training_log.txt
    echo ""
else
    echo "⚠️  訓練ログファイルが見つかりません"
    echo ""
fi

# 結果ファイルの確認
echo "📁 結果ファイル:"
echo "----------------------------------------"
if [ -f "results/gnn_results.json" ]; then
    echo "✅ GNN結果: results/gnn_results.json"
    python3 -c "import json; d=json.load(open('results/gnn_results.json')); print(f'   R²: {d[\"test_r2\"]:.4f}, RMSE: {d[\"test_rmse\"]:.4f} GPa')" 2>/dev/null || echo "   （読み込み中...）"
else
    echo "⏳ GNN結果: まだ生成されていません"
fi

if [ -f "results/transformer_results.json" ]; then
    echo "✅ Transformer結果: results/transformer_results.json"
    python3 -c "import json; d=json.load(open('results/transformer_results.json')); print(f'   R²: {d[\"test_r2\"]:.4f}, RMSE: {d[\"test_rmse\"]:.4f} GPa')" 2>/dev/null || echo "   （読み込み中...）"
else
    echo "⏳ Transformer結果: まだ生成されていません"
fi

if [ -f "results/model_comparison.json" ]; then
    echo "✅ モデル比較: results/model_comparison.json"
else
    echo "⏳ モデル比較: まだ生成されていません"
fi

echo ""

# モデルファイルの確認
echo "💾 保存されたモデル:"
echo "----------------------------------------"
if [ -f "models/gnn_best_model.pth" ]; then
    SIZE=$(du -h models/gnn_best_model.pth | cut -f1)
    echo "✅ GNNモデル: models/gnn_best_model.pth ($SIZE)"
else
    echo "⏳ GNNモデル: まだ保存されていません"
fi

if [ -f "models/transformer_best_model.pth" ]; then
    SIZE=$(du -h models/transformer_best_model.pth | cut -f1)
    echo "✅ Transformerモデル: models/transformer_best_model.pth ($SIZE)"
else
    echo "⏳ Transformerモデル: まだ保存されていません"
fi

echo ""

# プロセスの確認
echo "🔄 実行中の訓練プロセス:"
echo "----------------------------------------"
if pgrep -f "train.py" > /dev/null; then
    echo "✅ 訓練プロセスが実行中です"
    ps aux | grep "train.py" | grep -v grep | head -1
else
    echo "⏸️  訓練プロセスは実行されていません"
fi

echo ""
echo "=========================================="
echo "監視を終了します（5秒後に自動更新するには watch -n 5 ./monitor_training.sh を実行）"
echo "=========================================="
