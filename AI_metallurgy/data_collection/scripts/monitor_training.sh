#!/bin/bash
# Transformer訓練の監視スクリプト

LOG_FILE="/home/nishioka/LUH/AI_metallurgy/data_collection/scripts/transformer_800epochs.log"

echo "=========================================="
echo "Transformer訓練監視"
echo "=========================================="
echo ""

# プロセス確認
echo "📊 実行中のプロセス:"
ps aux | grep "[p]ython.*train.py.*transformer" | awk '{print "  PID:", $2, "| CPU:", $3"%", "| MEM:", $4"%", "| 実行時間:", $10}'
echo ""

# ログファイルの最新行
if [ -f "$LOG_FILE" ]; then
    echo "📝 最新のログ（最後の30行）:"
    echo "----------------------------------------"
    tail -30 "$LOG_FILE"
    echo "----------------------------------------"
    echo ""
    echo "📈 ログファイルサイズ: $(wc -l < "$LOG_FILE") 行"
else
    echo "⚠️  ログファイルが見つかりません: $LOG_FILE"
    echo "   訓練がまだ開始されていない可能性があります"
fi

echo ""
echo "リアルタイム監視: tail -f $LOG_FILE"
