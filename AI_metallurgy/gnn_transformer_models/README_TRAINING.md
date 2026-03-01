# 訓練開始ガイド

## 🚀 Conda環境での訓練（推奨）

### 自動セットアップと訓練

```bash
cd /home/nishioka/LUH/AI_metallurgy/gnn_transformer_models
bash setup_conda_and_train.sh
```

このスクリプトは以下を自動実行します：
1. ✅ Conda環境 `hea_gnn` の作成
2. ✅ 必要なパッケージのインストール
3. ✅ データファイルの確認
4. ✅ 訓練の開始

### 手動で環境を作成する場合

```bash
# 1. 環境の作成
conda create -n hea_gnn python=3.8 -y
conda activate hea_gnn

# 2. パッケージのインストール
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
conda install pandas numpy scikit-learn matplotlib seaborn tqdm -y
pip install torch-geometric

# 3. 訓練の開始
cd /home/nishioka/LUH/AI_metallurgy/gnn_transformer_models
python train.py --model both --batch_size 8 --num_epochs 50 --early_stopping_patience 15
```

## 📊 訓練の監視

### リアルタイム監視

```bash
# 状況確認（5秒ごとに自動更新）
watch -n 5 ./check_status.sh

# または手動で確認
./check_status.sh
```

### ログの監視

```bash
# 訓練ログをリアルタイムで表示
tail -f training_log.txt
```

## 📁 出力ファイル

訓練が完了すると、以下のファイルが生成されます：

### モデルファイル
- `models/gnn_best_model.pth` - 最良GNNモデル
- `models/transformer_best_model.pth` - 最良Transformerモデル

### 結果ファイル
- `results/gnn_results.json` - GNN結果
- `results/transformer_results.json` - Transformer結果
- `results/model_comparison.json` - モデル比較
- `results/gnn_results.png` - GNN可視化
- `results/transformer_results.png` - Transformer可視化

## ⏱️ 訓練時間の目安

- **322サンプル、CPU**: 約30-60分
- **322サンプル、GPU**: 約5-10分
- **軽量版モデル**: より高速

## 🔧 訓練パラメータの調整

```bash
python train.py \
  --model both \                    # gnn, transformer, both
  --batch_size 8 \                  # バッチサイズ
  --learning_rate 0.001 \           # 学習率
  --num_epochs 50 \                 # エポック数
  --early_stopping_patience 15 \    # Early stopping patience
  --use_light_model                 # 軽量版モデル
```

## 🐛 トラブルシューティング

### Conda環境が見つからない

```bash
# 環境の一覧を確認
conda env list

# 環境を再作成
conda env remove -n hea_gnn -y
conda create -n hea_gnn python=3.8 -y
```

### パッケージのインストールエラー

```bash
# 環境をアクティベート
conda activate hea_gnn

# パッケージを再インストール
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
pip install torch-geometric
```

### CUDA out of memory

バッチサイズを減らしてください：
```bash
python train.py --batch_size 4
```

## 📚 詳細ドキュメント

- `CONDA_SETUP.md` - Conda環境の詳細なセットアップ手順
- `TRAINING_INSTRUCTIONS.md` - 訓練の詳細な手順
- `USAGE_GUIDE.md` - 使用方法ガイド
