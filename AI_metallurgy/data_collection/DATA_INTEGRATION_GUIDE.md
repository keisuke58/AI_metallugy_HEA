# 📥 データ統合ガイド

**作成日**: 2026年1月20日

---

## 🎯 現在の状況

- ✅ **DOE/OSTI Dataset**: 107合金（完了）
- ✅ **最新研究データ**: 4合金（完了）
- ⏳ **DISMA Research HEA Dataset**: ダウンロード済み（確認必要）
- ⏳ **MPEA Mechanical Properties Database**: ダウンロード済み（確認必要）

---

## 📋 ダウンロードしたファイルの確認と移動

### ステップ1: ダウンロードしたファイルを探す

```bash
# 一般的なダウンロードフォルダを確認
ls -lh ~/Downloads/
ls -lh ~/ダウンロード/
ls -lh ~/Desktop/

# または、最近ダウンロードしたファイルを検索
find ~ -type f -mtime -1 -iname "*.zip" -o -iname "*.csv" -o -iname "*.xlsx" 2>/dev/null | head -20
```

### ステップ2: ファイルを適切な場所に移動

#### DISMA Research HEA Dataset

```bash
cd /home/nishioka/LUH/AI_metallurgy/data_collection

# ファイルを見つけたら移動
# 例: ファイル名が "disma_hea_dataset.zip" の場合
mv ~/Downloads/disma_hea_dataset.zip raw_data/disma_hea_dataset/
# または
mv ~/Downloads/*p3txdrdth7* raw_data/disma_hea_dataset/

# ZIPファイルの場合は解凍
cd raw_data/disma_hea_dataset/
unzip *.zip
```

#### MPEA Mechanical Properties Database

```bash
cd /home/nishioka/LUH/AI_metallurgy/data_collection

# ファイルを見つけたら移動
# 例: ファイル名が "mpea_mechanical_properties.zip" の場合
mv ~/Downloads/mpea_mechanical_properties.zip raw_data/mpea_mechanical_properties/
# または
mv ~/Downloads/*4d4kpfwpf6* raw_data/mpea_mechanical_properties/

# ZIPファイルの場合は解凍
cd raw_data/mpea_mechanical_properties/
unzip *.zip
```

### ステップ3: データを確認

```bash
cd /home/nishioka/LUH/AI_metallurgy/data_collection

# データ収集状況の確認
python scripts/check_data_status.py

# 代替データセットの確認
python scripts/download_alternative_datasets.py
```

---

## 📊 最低限必要なデータ量

### 結論

**最低限**: **100-200サンプル**で基本的なモデル訓練は可能

**現在の状況**:
- ✅ **111サンプル**: 基本的なモデル訓練は可能
- ⏳ **200+サンプル**: より高精度なモデルが可能（DISMA + MPEA追加後）

### 詳細

| データ数 | 可能なこと | 期待性能（R²） |
|---------|----------|--------------|
| **50-100** | 概念実証のみ | 0.3-0.5 |
| **100-200** | 基本的なモデル訓練 | 0.5-0.7 |
| **200-300** | 高精度モデル | 0.7-0.8 |
| **300-500** | 最適化まで可能 | 0.8-0.9 |

---

## 🔄 Gorsse Datasetについて

### 代替案

Gorsse Datasetが取得できない場合、以下の代替データセットで十分です：

1. **MPEA Mechanical Properties Database** ⭐⭐⭐⭐⭐
   - Gorsse Datasetより新しい（2023年）
   - より多くのメタデータ
   - 直接ダウンロード可能

2. **DISMA Research HEA Dataset** ⭐⭐⭐⭐⭐
   - 機械学習用に整理されている
   - 比較的新しい（2023年）

3. **DOE/OSTI Dataset** ⭐⭐⭐⭐⭐（既に取得済み）
   - 107合金の弾性率データ
   - 材料記述子が豊富

### Gorsse Datasetの再試行方法

1. **論文の補足資料から直接ダウンロード**
   - URL: https://www.sciencedirect.com/science/article/pii/S235234091831504X
   - 「Supplementary Material」セクションを確認

2. **著者に連絡**
   - 研究目的を説明してデータの提供をリクエスト

---

## ✅ 次のステップ

### 最優先（今すぐ）

1. **ダウンロードしたファイルを確認・移動**
   ```bash
   # ファイルを探す
   find ~ -type f -mtime -1 \( -iname "*disma*" -o -iname "*mpea*" -o -iname "*hea*" \) 2>/dev/null
   
   # 見つけたファイルを移動
   mv [見つけたファイル] /home/nishioka/LUH/AI_metallurgy/data_collection/raw_data/[適切なディレクトリ]/
   ```

2. **データを確認・分析**
   ```bash
   cd /home/nishioka/LUH/AI_metallurgy/data_collection
   python scripts/check_data_status.py
   ```

3. **データ数を集計**
   - DISMA + MPEA + DOE/OSTI + 最新研究
   - 合計が200以上なら十分

### 次優先

4. **データの統合とクリーニング**
   - すべてのデータセットを統合
   - 重複を除去
   - データの正規化

5. **データ拡張の検討**（データ数が200未満の場合）
   - ノイズ追加
   - 補間手法

---

## 📋 チェックリスト

- [ ] ダウンロードしたファイルを探す
- [ ] ファイルを適切なディレクトリに移動
- [ ] ZIPファイルを解凍（必要に応じて）
- [ ] データ内容を確認（弾性率データの有無）
- [ ] データ数を集計
- [ ] データ数が200以上か確認
- [ ] 200未満の場合はデータ拡張を検討

---

## 💡 重要なポイント

1. **現在111サンプルで基本的なモデル訓練は可能**
   - Linear Regression, Ridge/Lasso, KNNなど

2. **DISMA + MPEAを追加すれば200+サンプルになる可能性が高い**
   - より高精度なモデルが可能
   - Random Forest, SVR, MLFFNNなど

3. **Gorsse Datasetは必須ではない**
   - 代替データセットで十分
   - より新しいデータが利用可能

---

**最終更新**: 2026年1月20日
