#!/bin/bash
# 訓練状況の確認スクリプト

cd "$(dirname "$0")"

echo "=========================================="
echo "訓練状況確認"
echo "=========================================="
echo ""

# Conda環境の確認
echo "🐍 Conda環境:"
echo "----------------------------------------"
if conda info --envs 2>/dev/null | grep -q "hea_gnn"; then
    echo "✅ hea_gnn 環境が存在します"
    conda info --envs 2>/dev/null | grep hea_gnn
else
    echo "⏳ hea_gnn 環境はまだ作成されていません"
fi

echo ""

# プロセスの確認
echo "🔄 実行中のプロセス:"
echo "----------------------------------------"
if pgrep -f "setup_conda_and_train.sh" > /dev/null; then
    echo "✅ セットアップスクリプトが実行中です"
    ps aux | grep "setup_conda_and_train.sh" | grep -v grep | head -1
elif pgrep -f "train.py" > /dev/null; then
    echo "✅ 訓練プロセスが実行中です"
    ps aux | grep "train.py" | grep -v grep | head -1
else
    echo "⏸️  実行中のプロセスはありません"
fi

echo ""

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

echo ""

# モデルファイルの確認
echo "💾 保存されたモデル:"
echo "----------------------------------------"
if [ -f "models/gnn_best_model.pth" ]; then
    SIZE=$(du -h models/gnn_best_model.pth 2>/dev/null | cut -f1)
    echo "✅ GNNモデル: models/gnn_best_model.pth ($SIZE)"
else
    echo "⏳ GNNモデル: まだ保存されていません"
fi

if [ -f "models/transformer_best_model.pth" ]; then
    SIZE=$(du -h models/transformer_best_model.pth 2>/dev/null | cut -f1)
    echo "✅ Transformerモデル: models/transformer_best_model.pth ($SIZE)"
else
    echo "⏳ Transformerモデル: まだ保存されていません"
fi

echo ""
echo "=========================================="
echo "監視を続けるには: watch -n 5 ./check_status.sh"
echo "=========================================="
