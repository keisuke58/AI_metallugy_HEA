#!/bin/bash
# tmuxセッションで訓練を実行するスクリプト
# SSH接続が切れても訓練を継続（tmuxで再接続可能）

# 使用方法:
#   bash train_tmux.sh cgcnn
#   bash train_tmux.sh megnet
#   bash train_tmux.sh cgcnn megnet  # 複数モデルを順次実行

set -e

# tmuxがインストールされているか確認
if ! command -v tmux &> /dev/null; then
    echo "❌ tmuxがインストールされていません"
    echo "インストール: sudo apt-get install tmux  (Ubuntu/Debian)"
    echo "            brew install tmux  (macOS)"
    exit 1
fi

# モデルリスト（引数がない場合は全て）
if [ $# -eq 0 ]; then
    MODELS=("cgcnn" "megnet")
else
    MODELS=("$@")
fi

echo "=========================================="
echo "Tmuxセッションで訓練を実行"
echo "=========================================="
echo "実行時刻: $(date)"
echo "モデル: ${MODELS[@]}"
echo "=========================================="

# 各モデルをtmuxセッションで実行
for MODEL in "${MODELS[@]}"; do
    SESSION_NAME="train_${MODEL}"
    
    echo ""
    echo "=========================================="
    echo "Tmuxセッション作成: $SESSION_NAME"
    echo "=========================================="
    
    # 既存のセッションがあれば削除
    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        echo "⚠️  既存のセッション '$SESSION_NAME' を終了します"
        tmux kill-session -t "$SESSION_NAME" || true
        sleep 1
    fi
    
    # 新しいtmuxセッションを作成して訓練を開始
    tmux new-session -d -s "$SESSION_NAME" "
        echo '==========================================';
        echo '訓練開始: $MODEL';
        echo '実行時刻: \$(date)';
        echo '==========================================';
        python train.py --model $MODEL --device cuda --num_epochs 200 --batch_size 32;
        echo '';
        echo '==========================================';
        echo '訓練完了: $MODEL';
        echo '完了時刻: \$(date)';
        echo '==========================================';
        echo '';
        echo 'このウィンドウは10秒後に自動的に閉じます';
        sleep 10
    "
    
    sleep 1
    
    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        echo "✅ Tmuxセッション '$SESSION_NAME' を作成しました"
    else
        echo "❌ Tmuxセッションの作成に失敗しました"
        continue
    fi
done

echo ""
echo "=========================================="
echo "全てのTmuxセッションを作成しました"
echo "=========================================="
echo ""
echo "実行中のセッション一覧:"
tmux list-sessions
echo ""
echo "セッションに接続:"
for MODEL in "${MODELS[@]}"; do
    echo "  tmux attach -t train_${MODEL}"
done
echo ""
echo "セッションから切断（接続中に）: Ctrl+B, D"
echo ""
echo "セッションを終了:"
for MODEL in "${MODELS[@]}"; do
    echo "  tmux kill-session -t train_${MODEL}"
done
echo ""
echo "全てのセッションを一覧表示:"
echo "  tmux list-sessions"
echo ""
