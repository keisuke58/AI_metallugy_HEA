#!/bin/bash
# バックグラウンドで訓練を実行するスクリプト
# SSH接続が切れても訓練を継続

# 使用方法:
#   bash train_background.sh cgcnn
#   bash train_background.sh megnet
#   bash train_background.sh cgcnn megnet  # 複数モデルを順次実行

set -e

# ログディレクトリを作成
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

# タイムスタンプ
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# モデルリスト（引数がない場合は全て）
if [ $# -eq 0 ]; then
    MODELS=("cgcnn" "megnet")
else
    MODELS=("$@")
fi

echo "=========================================="
echo "バックグラウンド訓練スクリプト"
echo "=========================================="
echo "実行時刻: $(date)"
echo "モデル: ${MODELS[@]}"
echo "ログディレクトリ: $LOG_DIR"
echo "=========================================="

# 各モデルを訓練
for MODEL in "${MODELS[@]}"; do
    echo ""
    echo "=========================================="
    echo "訓練開始: $MODEL"
    echo "=========================================="
    
    LOG_FILE="$LOG_DIR/${MODEL}_${TIMESTAMP}.log"
    
    # nohupでバックグラウンド実行
    nohup python train.py \
        --model "$MODEL" \
        --device cuda \
        --num_epochs 200 \
        --batch_size 32 \
        > "$LOG_FILE" 2>&1 &
    
    PID=$!
    echo "✅ $MODEL の訓練を開始しました (PID: $PID)"
    echo "📝 ログファイル: $LOG_FILE"
    echo ""
    echo "進捗確認コマンド:"
    echo "  tail -f $LOG_FILE"
    echo "  または"
    echo "  watch -n 5 'tail -20 $LOG_FILE'"
    echo ""
    
    # 少し待機（ログファイルが作成されるまで）
    sleep 2
    
    # プロセスが実行中か確認
    if ps -p $PID > /dev/null; then
        echo "✅ プロセスは正常に実行中です"
    else
        echo "❌ プロセスが終了しました。ログを確認してください:"
        echo "   tail -50 $LOG_FILE"
    fi
done

echo ""
echo "=========================================="
echo "全ての訓練を開始しました"
echo "=========================================="
echo ""
echo "実行中のプロセスを確認:"
echo "  ps aux | grep 'python train.py'"
echo ""
echo "ログファイル一覧:"
echo "  ls -lh $LOG_DIR/"
echo ""
echo "特定のログを監視:"
echo "  tail -f $LOG_DIR/${MODELS[0]}_${TIMESTAMP}.log"
echo ""
echo "訓練を停止する場合:"
echo "  pkill -f 'python train.py'"
echo ""
