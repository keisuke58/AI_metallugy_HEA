# GNN & Transformer モデル - HEA弾性率予測

最新研究（2024-2025）に基づくGraph Neural Network (GNN)とTransformerモデルの実装です。

## 🚀 クイックスタート

### 1. セットアップ

```bash
cd /home/nishioka/LUH/AI_metallurgy/gnn_transformer_models
pip install -r requirements.txt
pip install torch-geometric
```

または、自動セットアップスクリプトを使用：

```bash
bash setup_and_run.sh
```

### 2. 訓練

```bash
python train.py
```

### 3. 推論

```bash
python inference.py --model both
```

## 📋 主な特徴

### ✅ 完全実装済み

#### GNNモデル
- ✅ エッジゲーティング畳み込み（MatGNet 2025スタイル）
- ✅ 自己注意機構
- ✅ 角度特徴量（ALIGNNスタイル）
- ✅ 材料記述子の統合
- ✅ 軽量版 + フル版

#### Transformerモデル
- ✅ 組成シーケンスエンコーディング（AlloyBERT 2024スタイル）
- ✅ 位置エンコーディング（Crystalformer 2024スタイル）
- ✅ Multi-head self-attention
- ✅ 材料記述子の統合
- ✅ 軽量版 + フル版

#### 訓練機能
- ✅ Early stopping
- ✅ 学習率スケジューリング
- ✅ 勾配クリッピング
- ✅ 自動評価（R², RMSE, MAE）
- ✅ 自動可視化

#### その他
- ✅ 柔軟なデータローダー
- ✅ 推論スクリプト
- ✅ 包括的なドキュメント
- ✅ エラーハンドリング

詳細は `FEATURES.md` を参照してください。

## 📁 ファイル構造

```
gnn_transformer_models/
├── data_loader.py          # データローダー
├── gnn_model.py           # GNNモデル
├── transformer_model.py  # Transformerモデル
├── train.py               # 訓練スクリプト
├── inference.py           # 推論スクリプト
├── setup_and_run.sh       # 自動セットアップスクリプト
├── requirements.txt       # 依存パッケージ
├── README.md              # 英語版README
├── README_JP.md           # このファイル
├── QUICKSTART.md          # クイックスタートガイド
├── USAGE_GUIDE.md         # 詳細な使用方法
├── IMPLEMENTATION_SUMMARY.md  # 実装の詳細
└── FEATURES.md            # 主な特徴
```

## 📚 ドキュメント

- **クイックスタート**: `QUICKSTART.md`
- **詳細な使用方法**: `USAGE_GUIDE.md`
- **実装の詳細**: `IMPLEMENTATION_SUMMARY.md`
- **主な特徴**: `FEATURES.md`

## 🔧 カスタマイズ

`train.py`の`CONFIG`を編集して設定を変更：

```python
CONFIG = {
    'batch_size': 16,
    'learning_rate': 1e-3,
    'num_epochs': 200,
    'use_light_model': True,  # 軽量版を使用
    ...
}
```

## 🐛 トラブルシューティング

詳細は `USAGE_GUIDE.md` の「トラブルシューティング」セクションを参照してください。

## 📊 期待される性能

既存の研究結果（R² = 0.59-0.67）と比較して、以下の改善が期待されます：

- **GNN**: グラフ構造により元素間の相互作用をより正確にモデル化
- **Transformer**: シーケンスモデルにより組成パターンを捉える

## 📝 ライセンス

研究目的で自由に使用できます。

## 🙏 謝辞

最新の材料科学における深層学習研究（2024-2025）に基づいて実装しました。
