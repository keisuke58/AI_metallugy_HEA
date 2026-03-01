# 📚 最新論文からのデータ抽出ガイド

**作成日**: 2026年1月20日  
**目標**: 2020-2025年の論文から100-300サンプルの弾性率データを抽出

---

## 🎯 抽出対象

### 検索キーワード

- "high entropy alloy" + "elastic modulus"
- "HEA" + "Young's modulus"
- "multi-principal element alloy" + "elastic modulus"
- "MPEA" + "Young's modulus"
- "refractory high entropy alloy" + "elastic modulus"
- "Ti-Zr-Nb" + "elastic modulus"
- "CoCrFeNi" + "elastic modulus"

### 対象期間

- **2020年 - 2025年**の論文
- 特に2022-2025年の最新研究を優先

---

## 📊 データソース

### 1. PubMed / PubMed Central

**URL**: https://pubmed.ncbi.nlm.nih.gov/

**検索例**:
```
("high entropy alloy"[Title/Abstract] OR "HEA"[Title/Abstract]) 
AND ("elastic modulus"[Title/Abstract] OR "Young's modulus"[Title/Abstract])
AND ("2020"[Publication Date] : "2025"[Publication Date])
```

**手順**:
1. PubMedで上記の検索クエリを実行
2. 関連論文を確認
3. 論文の表や補足資料からデータを抽出
4. CSVファイルに整理

---

### 2. arXiv

**URL**: https://arxiv.org/

**検索例**:
- "high entropy alloy elastic modulus"
- "HEA Young's modulus"

**手順**:
1. arXivで検索
2. 関連論文をダウンロード
3. 表や図からデータを抽出

---

### 3. Materials Cloud

**URL**: https://www.materialscloud.org/

**特徴**:
- 材料科学のデータリポジトリ
- 多くの論文の補足データが公開されている

**手順**:
1. Materials Cloudで検索
2. データセットをダウンロード
3. 弾性率データを抽出

---

### 4. 主要ジャーナル

#### Nature系
- Nature Materials
- Nature Communications
- Scientific Reports

#### Science系
- Science
- Science Advances

#### 材料科学専門誌
- Acta Materialia
- Scripta Materialia
- Materials Science and Engineering: A
- Journal of Alloys and Compounds
- Intermetallics

**手順**:
1. 各ジャーナルのウェブサイトで検索
2. 論文の補足資料を確認
3. データを抽出

---

## 📝 データ抽出方法

### 手動抽出

1. **論文の表から抽出**
   - 表に弾性率データが記載されている場合
   - ExcelまたはCSVに手動で入力

2. **図から抽出**
   - グラフから値を読み取る（精度は低い）
   - 可能な限り表データを優先

3. **補足資料から抽出**
   - 多くの論文で補足資料に詳細データが含まれる
   - Excel、CSV、JSON形式が多い

### 自動抽出（可能な場合）

- PDFから表を抽出するツール（Tabula、Camelot等）
- ただし、精度は限定的

---

## 📋 データ形式

抽出したデータは以下の形式で保存してください：

```csv
alloy_name,elastic_modulus,composition,density,source,reference,year
Ti-Zr-Nb,64.0,"Ti33.3,Zr33.3,Nb33.3",6.2,Latest Research,"Author et al. 2024",2024
```

### 必須カラム

- `alloy_name`: 合金名
- `elastic_modulus`: 弾性率（GPa）
- `source`: データソース（例: "Latest Research"）
- `reference`: 論文の引用情報
- `year`: 論文の発表年

### 推奨カラム

- `composition`: 組成（原子%または重量%）
- `density`: 密度（g/cm³）
- `phases`: 相情報
- `processing`: 処理条件
- `temperature`: 測定温度（℃）

---

## 🔍 推奨検索戦略

### ステップ1: 系統的検索

1. **PubMedで検索**
   - 上記の検索クエリを使用
   - 2020-2025年の論文をフィルタ

2. **主要ジャーナルを直接検索**
   - Acta Materialia
   - Scripta Materialia
   - Materials Science and Engineering: A

3. **arXivで検索**
   - プレプリントも含める

### ステップ2: データ抽出

1. **論文をダウンロード**
2. **表や補足資料を確認**
3. **データをCSVに整理**

### ステップ3: データ統合

1. 抽出したデータを `raw_data/literature_data/` に保存
2. `scripts/final_data_integration.py` で統合

---

## 📁 ファイル構造

```
raw_data/
└── literature_data/
    ├── 2020/
    │   ├── paper1_data.csv
    │   └── paper2_data.csv
    ├── 2021/
    ├── 2022/
    ├── 2023/
    ├── 2024/
    └── 2025/
```

---

## 🛠️ 便利なツール

### PDFから表を抽出

1. **Tabula**
   - URL: https://tabula.technology/
   - PDFから表を抽出

2. **Camelot**
   - Pythonライブラリ
   - `pip install camelot-py[cv]`

3. **PDFPlumber**
   - Pythonライブラリ
   - `pip install pdfplumber`

---

## 📊 期待される結果

- **目標サンプル数**: 100-300サンプル
- **期間**: 2020-2025年
- **品質**: 高品質（実測データ）

---

## ✅ チェックリスト

- [ ] PubMedで検索
- [ ] arXivで検索
- [ ] Materials Cloudで検索
- [ ] 主要ジャーナルで検索
- [ ] データを抽出
- [ ] CSVファイルに整理
- [ ] `raw_data/literature_data/` に保存
- [ ] データ統合スクリプトで統合

---

## 🔗 参考リンク

- **PubMed**: https://pubmed.ncbi.nlm.nih.gov/
- **arXiv**: https://arxiv.org/
- **Materials Cloud**: https://www.materialscloud.org/
- **Tabula**: https://tabula.technology/
- **Camelot**: https://camelot-py.readthedocs.io/

---

**最終更新**: 2026年1月20日
