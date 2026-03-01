#!/bin/bash
# セットアップ進行状況の監視

cd "$(dirname "$0")"

echo "=========================================="
echo "セットアップ進行状況"
echo "=========================================="
echo ""

if [ -f "setup_log.txt" ]; then
    echo "📊 最新のログ（最後の20行）:"
    echo "----------------------------------------"
    tail -20 setup_log.txt
    echo ""
else
    echo "⏳ ログファイルがまだ作成されていません"
    echo ""
fi

# プロセスの確認
echo "🔄 実行中のプロセス:"
echo "----------------------------------------"
if pgrep -f "setup_conda_and_train.sh" > /dev/null; then
    echo "✅ セットアップが進行中です"
    ps aux | grep "setup_conda_and_train.sh" | grep -v grep | head -1
elif pgrep -f "train.py" > /dev/null; then
    echo "✅ 訓練が進行中です"
    ps aux | grep "train.py" | grep -v grep | head -1
else
    echo "⏸️  実行中のプロセスはありません"
    if [ -f "setup_log.txt" ]; then
        if grep -q "訓練完了" setup_log.txt; then
            echo "✅ セットアップと訓練が完了しました！"
        elif grep -q "エラー\|失敗\|Error\|Failed" setup_log.txt; then
            echo "⚠️  エラーが発生した可能性があります。ログを確認してください。"
        fi
    fi
fi

echo ""
echo "=========================================="
echo "リアルタイム監視: tail -f setup_log.txt"
echo "自動更新: watch -n 5 ./watch_setup.sh"
echo "=========================================="
