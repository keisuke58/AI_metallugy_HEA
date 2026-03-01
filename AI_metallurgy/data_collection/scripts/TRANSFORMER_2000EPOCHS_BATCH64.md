# Transformer拡張訓練（2000エポック、batch_size=64）

**実行日**: 2026-01-25  
**環境**: hea_gnn  
**データセット**: 322サンプル（実験データ）

---

## 🔧 訓練設定

### ハイパーパラメータ
- **エポック数**: 2000（以前: 1000）
- **早期停止忍耐**: 300（以前: 200）
- **バッチサイズ**: 64（ユーザー指定）
- **学習率**: 5e-5（より低い学習率で安定化）
- **Weight Decay**: 5e-4（より強い正則化）
- **デバイス**: CUDA

### 目的
- より長い訓練により精度向上を目指す
- Gradient Boosting（R²=0.4956）に近い性能を達成
- 正則化を強化して過学習を抑制

---

## 📊 実行コマンド

```bash
cd /home/nishioka/LUH/AI_metallurgy/gnn_transformer_models
python train.py \
  --model transformer \
  --data_path ../data_collection/processed_data/data_with_features.csv \
  --batch_size 64 \
  --learning_rate 5e-5 \
  --num_epochs 2000 \
  --early_stopping_patience 300 \
  --weight_decay 5e-4
```

---

## 📈 期待される改善

### 以前の結果
- **300エポック**: R² = 0.3191
- **1000エポック（262で停止）**: R² = 0.4434

### 今回の目標
- **2000エポック**: R² = **0.48-0.50**（Gradient Boostingに近づく）
- **RMSE**: 42.52 → **40-41 GPa**
- **MAE**: 28.35 → **25-26 GPa**

---

## ⏱️ 実行時間の目安

- **1000エポック（262で停止）**: 約1時間30分
- **2000エポック**: 約10-15時間（推定、早期停止により短縮される可能性）

---

## 📝 注意事項

1. **早期停止**: patience=300に設定されているため、300エポック改善がなければ停止
2. **バッチサイズ**: 64は大きめのバッチサイズ（より安定した訓練）
3. **正則化**: weight_decay=5e-4で過学習を抑制
4. **学習率**: 5e-5でより安定した学習

---

## 📁 ログファイル

- `data_collection/scripts/transformer_2000epochs_batch64.log`: 訓練ログ

---

## ✅ 監視方法

```bash
# プロセス確認
ps aux | grep "[p]ython.*train.py.*transformer"

# ログファイルをリアルタイム監視
tail -f /home/nishioka/LUH/AI_metallurgy/data_collection/scripts/transformer_2000epochs_batch64.log

# 最新のエポックと性能を確認
tail -50 /home/nishioka/LUH/AI_metallurgy/data_collection/scripts/transformer_2000epochs_batch64.log | grep "Epoch"
```
