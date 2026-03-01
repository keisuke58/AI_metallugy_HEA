# 📊 データ収集ガイド

**作成日**: 2026年1月20日  
**目的**: HEA弾性率予測プロジェクト用のデータ収集

---

## 🎯 データ収集の目標

- **Gorsse Dataset**: 約370合金（弾性率データ含む）
- **DOE/OSTI Dataset**: 107合金の弾性率データ、340合金の相データ
- **最新研究データ**: 2024-2025年の文献から抽出
- **統合データ数**: 400-500合金

---

## 📋 データセット一覧

### 1. Gorsse Dataset ⭐⭐⭐⭐⭐（最優先）

**出典**: 
- Gorsse, S., Nguyen, M. H., Senkov, O. N., & Miracle, D. B. (2018)
- "Database on the mechanical properties of high entropy alloys and complex concentrated alloys"
- Data in Brief, 2018

**データ内容**:
- 約370合金（2004-2016年）
- Young's modulus（弾性率）を含む
- 降伏強度、引張強度、硬度

**ダウンロード方法**:
1. PubMedで論文を検索: https://pubmed.ncbi.nlm.nih.gov/30761350/
2. 論文の補足資料（Supplementary Material）を確認
3. データファイルをダウンロード（CSVまたはExcel形式）
4. `raw_data/gorsse_dataset/` に保存

**URL**:
- PubMed: https://pubmed.ncbi.nlm.nih.gov/30761350/
- Data in Brief: 論文の補足資料

---

### 2. DOE/OSTI Dataset ⭐⭐⭐⭐⭐

**出典**:
- Balasubramanian, G. & Ganesh, S.
- "Phases and Young's Modulus Dataset for High Entropy Alloys"
- DOE/OSTI (2020)

**データ内容**:
- 107合金: Young's modulusデータ
- 340合金: 相データ
- 計算された特徴量を含む

**ダウンロード方法**:
1. URLにアクセス: https://www.osti.gov/dataexplorer/biblio/dataset/1644295
2. データセットをダウンロード
3. データ形式を確認（通常はCSVまたはJSON）
4. `raw_data/doe_osti_dataset/` に保存

**URL**:
- OSTI Data Explorer: https://www.osti.gov/dataexplorer/biblio/dataset/1644295

---

### 3. Materials Project ⭐⭐⭐（補完用）

**出典**: Materials Project Database

**データ内容**:
- 第一原理計算による弾性テンソル
- 1,100以上の無機結晶化合物

**アクセス方法**:
1. Materials Projectにアカウント作成: https://materialsproject.org
2. APIキーを取得
3. Python APIを使用してデータを取得（`scripts/download_materials_project.py`を参照）

**制限**:
- ⚠️ HEAのデータは限定的
- ⚠️ 計算値（実験値ではない）

---

### 4. 最新研究データ（2024-2025）⭐⭐⭐⭐

**データ内容**:
- Ti-Zr-Nb系MEA: 62-66 GPa
- Ti-Zr-Hf-Nb-Ta HEA: 69 GPa
- Ti40Zr25Nb25Ta5Mo5 HEA: 86 GPa
- Ti-Nb-Ta-Cr-Co HEA: 82 GPa

**収集方法**:
- 論文の補足資料から抽出
- 論文の表から手動で抽出
- `raw_data/latest_research/` に保存

---

## 📁 ディレクトリ構造

```
data_collection/
├── README.md (このファイル)
├── raw_data/
│   ├── gorsse_dataset/
│   │   └── (Gorsse Datasetのファイル)
│   ├── doe_osti_dataset/
│   │   └── (DOE/OSTI Datasetのファイル)
│   ├── materials_project/
│   │   └── (Materials Projectのデータ)
│   └── latest_research/
│       └── (最新研究データ)
├── processed_data/
│   └── (統合・クリーニング後のデータ)
└── scripts/
    ├── download_gorsse.py
    ├── download_doe_osti.py
    ├── download_materials_project.py
    ├── merge_datasets.py
    └── clean_data.py
```

---

## 🚀 データ収集の手順

### Step 1: Gorsse Datasetのダウンロード

```bash
# 手動でダウンロード
# 1. https://pubmed.ncbi.nlm.nih.gov/30761350/ にアクセス
# 2. 補足資料からデータファイルをダウンロード
# 3. raw_data/gorsse_dataset/ に保存
```

または、スクリプトを使用（作成予定）:
```bash
python scripts/download_gorsse.py
```

---

### Step 2: DOE/OSTI Datasetのダウンロード

```bash
# 手動でダウンロード
# 1. https://www.osti.gov/dataexplorer/biblio/dataset/1644295 にアクセス
# 2. データセットをダウンロード
# 3. raw_data/doe_osti_dataset/ に保存
```

または、スクリプトを使用（作成予定）:
```bash
python scripts/download_doe_osti.py
```

---

### Step 3: データの統合とクリーニング

```bash
python scripts/merge_datasets.py
python scripts/clean_data.py
```

---

## ✅ チェックリスト

### データ収集
- [ ] Gorsse Datasetをダウンロード
- [ ] DOE/OSTI Datasetをダウンロード
- [ ] 最新研究データ（2024-2025）を抽出
- [ ] Materials Projectデータを取得（オプション）

### データ処理
- [ ] データを統合
- [ ] データクリーニング
- [ ] 重複を除去
- [ ] 外れ値を処理

### データ検証
- [ ] データ数の確認（目標: 400-500合金）
- [ ] 弾性率データの有無を確認
- [ ] 組成データの有無を確認
- [ ] データ品質の確認

---

## 📝 注意事項

1. **データの出典を明記**: 使用したデータセットの出典を必ず記録
2. **ライセンスの確認**: 各データセットのライセンスを確認
3. **データのバックアップ**: ダウンロードしたデータはバックアップを取る
4. **バージョン管理**: データのバージョンを記録

---

## 🔗 参考リンク

- Gorsse Dataset: https://pubmed.ncbi.nlm.nih.gov/30761350/
- DOE/OSTI Dataset: https://www.osti.gov/dataexplorer/biblio/dataset/1644295
- Materials Project: https://materialsproject.org
- AFLOW: http://aflow.org

---

**最終更新**: 2026年1月20日
