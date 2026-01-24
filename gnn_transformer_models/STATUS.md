# 訓練状況

## 現在の状態

**⚠️ 訓練を開始する前に、必要なパッケージをインストールしてください**

## 必要なパッケージ

以下のパッケージがインストールされていません：
- torch (PyTorch)
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- tqdm
- torch-geometric

## インストール方法

### クイックスタート

```bash
cd /home/nishioka/LUH/AI_metallurgy/gnn_transformer_models

# 方法1: 自動インストールスクリプト（推奨）
bash install_and_train.sh

# 方法2: 手動インストール
# まずpip3をインストール（必要な場合）
sudo apt-get install -y python3-pip

# パッケージをインストール
pip3 install --user torch torchvision torchaudio pandas numpy scikit-learn matplotlib seaborn tqdm torch-geometric

# 訓練を開始
python3 train.py --model both --batch_size 8 --num_epochs 50 --early_stopping_patience 15
```

## 訓練の監視

訓練が開始されたら、以下のコマンドで監視できます：

```bash
# リアルタイム監視（5秒ごとに更新）
watch -n 5 ./monitor_training.sh

# または手動で監視
./monitor_training.sh

# ログファイルを監視
tail -f training_log.txt
```

## データファイル

✅ データファイルは存在します：
- パス: `../data_collection/processed_data/data_with_features.csv`
- サイズ: 121KB

## 次のステップ

1. パッケージをインストール
2. 訓練を開始: `python3 train.py --model both`
3. 監視: `./monitor_training.sh` または `watch -n 5 ./monitor_training.sh`

詳細は `TRAINING_INSTRUCTIONS.md` を参照してください。
