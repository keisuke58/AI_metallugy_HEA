#!/bin/bash
# screenセッションで訓練を実行するスクリプト
# SSH接続が切れても訓練を継続（screenで再接続可能）

# 使用方法:
#   bash train_screen.sh cgcnn
#   bash train_screen.sh megnet
#   bash train_screen.sh cgcnn megnet  # 複数モデルを順次実行

set -e

# screenがインストールされているか確認
if ! command -v screen &> /dev/null; then
    echo "❌ screenがインストールされていません"
    echo "インストール: sudo apt-get install screen  (Ubuntu/Debian)"
    echo "            brew install screen  (macOS)"
    exit 1
fi

# モデルリスト（引数がない場合は全て）
if [ $# -eq 0 ]; then
    MODELS=("cgcnn" "megnet")
else
    MODELS=("$@")
fi

echo "=========================================="
echo "Screenセッションで訓練を実行"
echo "=========================================="
echo "実行時刻: $(date)"
echo "モデル: ${MODELS[@]}"
echo "=========================================="

# 各モデルをscreenセッションで実行
for MODEL in "${MODELS[@]}"; do
    SESSION_NAME="train_${MODEL}"
    
    echo ""
    echo "=========================================="
    echo "Screenセッション作成: $SESSION_NAME"
    echo "=========================================="
    
    # 既存のセッションがあれば削除
    if screen -list | grep -q "$SESSION_NAME"; then
        echo "⚠️  既存のセッション '$SESSION_NAME' を終了します"
        screen -S "$SESSION_NAME" -X quit || true
        sleep 1
    fi
    
    # 新しいscreenセッションを作成して訓練を開始
    screen -dmS "$SESSION_NAME" bash -c "
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
    
    if screen -list | grep -q "$SESSION_NAME"; then
        echo "✅ Screenセッション '$SESSION_NAME' を作成しました"
    else
        echo "❌ Screenセッションの作成に失敗しました"
        continue
    fi
done

echo ""
echo "=========================================="
echo "全てのScreenセッションを作成しました"
echo "=========================================="
echo ""
echo "実行中のセッション一覧:"
screen -list
echo ""
echo "セッションに接続:"
for MODEL in "${MODELS[@]}"; do
    echo "  screen -r train_${MODEL}"
done
echo ""
echo "セッションから切断（接続中に）: Ctrl+A, D"
echo ""
echo "セッションを終了:"
for MODEL in "${MODELS[@]}"; do
    echo "  screen -S train_${MODEL} -X quit"
done
echo ""
echo "全てのセッションを一覧表示:"
echo "  screen -list"
echo ""
