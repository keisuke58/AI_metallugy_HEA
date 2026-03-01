# 訓練監視ガイド

## 📊 Transformer訓練の監視方法

### リアルタイム監視

```bash
# ログファイルをリアルタイムで監視
tail -f /home/nishioka/LUH/AI_metallurgy/data_collection/scripts/transformer_1000epochs.log

# または、gnn_transformer_modelsディレクトリで直接実行している場合
cd /home/nishioka/LUH/AI_metallurgy/gnn_transformer_models
tail -f results/training.log  # 存在する場合
```

### プロセス状態確認

```bash
# 実行中のプロセスを確認
ps aux | grep "[p]ython.*train.py.*transformer"

# CPU/メモリ使用率を確認
ps aux | grep "[p]ython.*train.py.*transformer" | awk '{print "PID:", $2, "CPU:", $3"%", "MEM:", $4"%", "実行時間:", $10}'
```

### 進捗確認（エポック数）

```bash
# ログから最新のエポックを確認
tail -100 /home/nishioka/LUH/AI_metallurgy/data_collection/scripts/transformer_1000epochs.log | grep "Epoch" | tail -5

# 最新の性能を確認
tail -100 /home/nishioka/LUH/AI_metallurgy/data_collection/scripts/transformer_1000epochs.log | grep -E "(R²|RMSE|MAE)" | tail -5
```

### 結果ファイルの確認

```bash
# 最新の結果JSONを確認
ls -lht /home/nishioka/LUH/AI_metallurgy/gnn_transformer_models/results/transformer_results.json

# 結果の内容を確認
cat /home/nishioka/LUH/AI_metallurgy/gnn_transformer_models/results/transformer_results.json | python -m json.tool | grep -E "(test_r2|test_rmse|test_mae)"
```

## 🔧 現在の訓練設定

- **エポック数**: 1000
- **早期停止忍耐**: 200
- **バッチサイズ**: 16
- **学習率**: 1e-4

## ⏱️ 実行時間の目安

- **100エポック**: 約30-40分
- **300エポック**: 約1.5-2時間
- **1000エポック**: 約5-7時間（推定）

## 📝 注意事項

1. **早期停止**: 負のR²が発生するとpatienceが2倍カウントされるため、実際のpatienceは設定値より短くなる可能性があります
2. **GPU使用**: CUDAが利用可能な場合、自動的にGPUを使用します
3. **ログファイル**: 長時間の訓練により、ログファイルが大きくなる可能性があります
