# 🚀 Neural Operator Models for Material Property Prediction

複数のニューラルオペレーターと材料科学特化モデルの実装

## 📋 実装済みモデル

1. **FNO (Fourier Neural Operator)** ⭐⭐⭐⭐
   - Li et al. (2020) "Fourier Neural Operator for Parametric Partial Differential Equations"
   - 組成データを空間化してFourier変換で処理

2. **DeepONet (Deep Operator Network)** ⭐⭐⭐⭐⭐
   - Lu et al. (2021) "Learning nonlinear operators via DeepONet"
   - BranchネットワークとTrunkネットワークの組み合わせ

3. **MEGNet (Materials Graph Networks)** ⭐⭐⭐⭐⭐
   - Chen et al. (2019) "Graph Networks as a Universal Machine Learning Framework"
   - 材料科学に特化したGNN

4. **CGCNN (Crystal Graph Convolutional Networks)** ⭐⭐⭐⭐⭐
   - Xie & Grossman (2018) "Crystal Graph Convolutional Neural Networks"
   - 結晶構造に特化したGNN

5. **Neural ODE** ⭐⭐⭐⭐
   - Chen et al. (2018) "Neural Ordinary Differential Equations"
   - 連続的な動的システムとして材料特性をモデル化

6. **PINNs (Physics-Informed Neural Networks)** ⭐⭐⭐⭐
   - Raissi et al. (2019) "Physics-informed neural networks"
   - 物理法則を損失関数に組み込む

## 🚀 セットアップ

### 必要な環境

- Python 3.8以上
- CUDA対応GPU（推奨、CPUでも動作可能）

### インストール

```bash
cd /home/nishioka/LUH/AI_metallurgy/fno_models
pip install -r requirements.txt
```

### PyTorch Geometricのインストール

```bash
# CPU版
pip install torch-geometric

# GPU版（CUDA 11.8の場合）
pip install torch-geometric -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

### Neural ODE用のtorchdiffeq

```bash
pip install torchdiffeq
```

## 📊 データ準備

データファイルは以下のパスに配置してください：

```
/home/nishioka/LUH/AI_metallurgy/data_collection/processed_data/data_with_features.csv
```

必要なカラム：
- `comp_*`: 組成カラム（例: `comp_Ti`, `comp_Zr`など）
- `elastic_modulus`: 弾性率（目標変数）
- `mixing_entropy`, `mixing_enthalpy`, `vec`, `delta_r`, `delta_chi`, `mean_atomic_radius`, `mean_electronegativity`, `density`: 材料記述子

## 🎯 使用方法

### 訓練

#### すべてのモデルを訓練

```bash
python train.py --model all --data_path /path/to/data.csv
```

#### 特定のモデルを訓練

```bash
# FNO
python train.py --model fno --data_path /path/to/data.csv

# DeepONet
python train.py --model deeponet --data_path /path/to/data.csv

# MEGNet
python train.py --model megnet --data_path /path/to/data.csv

# CGCNN
python train.py --model cgcnn --data_path /path/to/data.csv

# Neural ODE
python train.py --model neural_ode --data_path /path/to/data.csv

# PINNs
python train.py --model pinns --data_path /path/to/data.csv
```

#### ハイパーパラメータの調整

```bash
python train.py --model fno \
    --batch_size 64 \
    --learning_rate 0.001 \
    --num_epochs 300 \
    --data_path /path/to/data.csv
```

### 推論

#### すべてのモデルで推論

```bash
python inference.py --model all --data_path /path/to/data.csv
```

#### 特定のモデルで推論

```bash
python inference.py --model fno --data_path /path/to/data.csv --output_path results.csv
```

## 📁 ディレクトリ構造

```
fno_models/
├── models/              # モデル実装
│   ├── fno.py          # FNO
│   ├── deeponet.py     # DeepONet
│   ├── megnet.py       # MEGNet
│   ├── cgcnn.py        # CGCNN
│   ├── neural_ode.py   # Neural ODE
│   └── pinns.py        # PINNs
├── data_loaders/        # データローダー
│   ├── fno_loader.py
│   ├── deeponet_loader.py
│   ├── graph_loader.py
│   ├── neural_ode_loader.py
│   └── pinns_loader.py
├── utils/               # ユーティリティ
│   ├── element_properties.py
│   └── data_utils.py
├── checkpoints/         # 訓練済みモデル
├── results/             # 結果（JSON, PNG）
├── train.py            # 訓練スクリプト
├── inference.py        # 推論スクリプト
└── requirements.txt    # 依存パッケージ
```

## 📊 結果の確認

訓練後、以下のファイルが生成されます：

- `results/{model}_results.json`: 評価結果（R², RMSE, MAE）
- `results/{model}_results.png`: 可視化結果
- `checkpoints/{model}_best_model.pth`: 最良モデル

## 🔧 モデル比較

訓練後、`results/all_results.json`に全モデルの比較結果が保存されます。

推論スクリプトでも比較結果が表示されます：

```bash
python inference.py --model all
```

## 📚 参考資料

1. **FNO**: Li et al. (2020) "Fourier Neural Operator for Parametric Partial Differential Equations"
2. **DeepONet**: Lu et al. (2021) "Learning nonlinear operators via DeepONet"
3. **MEGNet**: Chen et al. (2019) "Graph Networks as a Universal Machine Learning Framework"
4. **CGCNN**: Xie & Grossman (2018) "Crystal Graph Convolutional Neural Networks"
5. **Neural ODE**: Chen et al. (2018) "Neural Ordinary Differential Equations"
6. **PINNs**: Raissi et al. (2019) "Physics-informed neural networks"

## ⚠️ 注意事項

- Neural ODEは`torchdiffeq`パッケージが必要です
- MEGNetとCGCNNは`torch-geometric`が必要です
- GPUメモリが不足する場合は`batch_size`を小さくしてください

## 🐛 トラブルシューティング

### ImportError: No module named 'torchdiffeq'

```bash
pip install torchdiffeq
```

### CUDA out of memory

`batch_size`を小さくしてください（例: `--batch_size 16`）

### データファイルが見つからない

`--data_path`オプションで正しいパスを指定してください。

## 📝 ライセンス

このプロジェクトは研究・教育目的で使用できます。

## 👥 作成者

AI Assistant

## 📅 最終更新

2026年1月23日
