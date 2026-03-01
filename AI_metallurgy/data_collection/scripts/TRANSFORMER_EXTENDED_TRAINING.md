# Transformer拡張訓練（800エポック）

**実行日**: 2026-01-25  
**環境**: hea_gnn  
**データセット**: 322サンプル（実験データ）

---

## 🔧 訓練設定

### ハイパーパラメータ
- **エポック数**: 800（以前: 300）
- **早期停止忍耐**: 100（以前: 50）
- **バッチサイズ**: 16
- **学習率**: 1e-4
- **デバイス**: CUDA

### 目的
- より長い訓練により性能向上を目指す
- 早期停止を抑制して、より多くのエポックで学習

---

## 📊 実行コマンド

```bash
cd /home/nishioka/LUH/AI_metallurgy/gnn_transformer_models
python train.py \
  --model transformer \
  --data_path ../data_collection/processed_data/data_with_features.csv \
  --batch_size 16 \
  --learning_rate 1e-4 \
  --num_epochs 800 \
  --early_stopping_patience 100
```

---

## 📈 進捗監視

### 監視方法

```bash
# プロセス確認
ps aux | grep "[p]ython.*train.py.*transformer"

# ログファイル確認（存在する場合）
tail -f /home/nishioka/LUH/AI_metallurgy/data_collection/scripts/transformer_800epochs.log
```

### 期待される改善

- **以前の結果**: R² = 0.3191（300エポック）
- **目標**: R² > 0.35-0.40（800エポック）

---

## ⏱️ 実行時間の目安

- **300エポック**: 約1-2時間
- **800エポック**: 約3-5時間（推定）

---

## 📝 注意事項

1. **早期停止**: patience=100に設定されているため、100エポック改善がなければ停止
2. **GPU使用**: CUDAが利用可能な場合、自動的にGPUを使用
3. **メモリ**: 長時間の訓練により、メモリ使用量に注意

---

## ✅ 完了後の確認

訓練完了後、以下を確認：

1. **結果ファイル**: `gnn_transformer_models/results/transformer_results.json`
2. **モデルファイル**: `gnn_transformer_models/models/transformer_best_model.pth`
3. **性能メトリクス**: Test R², RMSE, MAE
