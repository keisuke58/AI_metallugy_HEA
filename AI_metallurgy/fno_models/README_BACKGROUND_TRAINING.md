# バックグラウンド訓練ガイド

SSH接続が切れても訓練を継続できるようにする方法を説明します。

## 方法1: nohupを使用（最も簡単）

### 実行方法

```bash
cd /home/nishioka/LUH/AI_metallurgy/fno_models
bash train_background.sh cgcnn
# または
bash train_background.sh megnet
# または両方
bash train_background.sh cgcnn megnet
```

### 進捗確認

```bash
# ログファイルを監視
tail -f logs/cgcnn_*.log

# または最新のログを確認
ls -lt logs/ | head -5
tail -50 logs/cgcnn_最新のログファイル名.log
```

### プロセス確認

```bash
# 実行中のプロセスを確認
ps aux | grep 'python train.py'

# 訓練を停止
pkill -f 'python train.py'
```

---

## 方法2: screenを使用（推奨）

### 実行方法

```bash
cd /home/nishioka/LUH/AI_metallurgy/fno_models
bash train_screen.sh cgcnn
```

### セッションに接続

```bash
# セッション一覧を表示
screen -list

# セッションに接続
screen -r train_cgcnn
```

### セッションから切断

接続中に `Ctrl+A` を押してから `D` を押すと、セッションから切断されます（訓練は継続）。

### セッションを終了

```bash
screen -S train_cgcnn -X quit
```

---

## 方法3: tmuxを使用（推奨）

### 実行方法

```bash
cd /home/nishioka/LUH/AI_metallurgy/fno_models
bash train_tmux.sh cgcnn
```

### セッションに接続

```bash
# セッション一覧を表示
tmux list-sessions

# セッションに接続
tmux attach -t train_cgcnn
```

### セッションから切断

接続中に `Ctrl+B` を押してから `D` を押すと、セッションから切断されます（訓練は継続）。

### セッションを終了

```bash
tmux kill-session -t train_cgcnn
```

---

## 推奨される使用方法

1. **初めて使用する場合**: `train_background.sh`（最も簡単）
2. **進捗をリアルタイムで確認したい場合**: `train_screen.sh` または `train_tmux.sh`
3. **複数のモデルを同時に訓練する場合**: 各モデルごとに別々のscreen/tmuxセッションを作成

---

## 注意事項

- 訓練中にSSH接続が切れても、訓練は継続されます
- screen/tmuxを使用する場合、再接続して進捗を確認できます
- nohupを使用する場合、ログファイルを確認して進捗を把握します
- GPUメモリが不足する場合は、`--batch_size`を小さくしてください

---

## トラブルシューティング

### プロセスが停止している場合

```bash
# エラーログを確認
tail -100 logs/cgcnn_*.log

# プロセスを再起動
bash train_background.sh cgcnn
```

### GPUメモリエラーの場合

```bash
# バッチサイズを小さくして実行
python train.py --model cgcnn --device cuda --num_epochs 200 --batch_size 16
```

### screen/tmuxがインストールされていない場合

```bash
# Ubuntu/Debian
sudo apt-get install screen tmux

# macOS
brew install screen tmux
```
