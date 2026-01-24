# 📥 Gorsse Dataset ダウンロードガイド

**更新日**: 2026年1月20日

---

## 🎯 論文情報

**タイトル**: Database on the mechanical properties of high entropy alloys and complex concentrated alloys

**著者**: Stéphane Gorsse, M. H. Nguyen, O. N. Senkov, D. B. Miracle

**ジャーナル**: Data in Brief, Volume 21, December 2018, Pages 2664-2678

**DOI**: 10.1016/j.dib.2018.11.111

**PubMed ID**: 30761350

**データ内容**:
- 約370合金（2004-2016年）
- 組成、微細組織、密度、硬度、降伏強度、引張強度、伸び、**弾性率（Young's modulus）**
- 27個の難融HEAについて温度依存データ

---

## 📥 ダウンロード方法

### 方法1: ScienceDirectから補足資料をダウンロード（推奨）⭐⭐⭐⭐⭐

**URL**: https://www.sciencedirect.com/science/article/pii/S235234091831504X

**手順**:
1. 上記のURLにアクセス
2. 論文ページの「Supplementary Material」セクションを確認
3. **mmc1.pdf** (~81.6 KB) をダウンロード
   - このPDFにはデータベースエントリが含まれています
4. PDFからデータを抽出するか、Google Sheets版を使用

**注意**: PDF形式なので、CSV/Excelに変換する必要があります。

---

### 方法2: Google Sheets版を使用（最推奨）⭐⭐⭐⭐⭐

**利点**:
- ✅ 直接編集可能
- ✅ CSV/Excel形式でエクスポート可能
- ✅ データが構造化されている

**手順**:
1. ScienceDirectの論文ページにアクセス
2. 著者が共有しているGoogle Sheetsのリンクを確認
3. Google Sheetsからデータをダウンロード
   - 「ファイル」→「ダウンロード」→「CSV」または「Excel」

---

### 方法3: 訂正版（Corrigendum）から取得（推奨）⭐⭐⭐⭐⭐

**重要**: 2020年10月に訂正版が公開されています！

**URL**: https://www.sciencedirect.com/science/article/pii/S2352340920311100

**訂正内容**:
- 一部の合金の相識別の誤りを修正
- 一部の合金組成の誤りを修正
- CrHfNbTiZrの硬度値の修正
- 元のデータセットで省略されていた2つの合金のデータを追加

**手順**:
1. 上記のURLにアクセス
2. 訂正版の補足資料をダウンロード
3. **訂正版のデータを使用**（より正確）

---

### 方法4: PubMedからアクセス

**URL**: https://pubmed.ncbi.nlm.nih.gov/30761350/

**手順**:
1. 上記のURLにアクセス
2. 「Full text links」セクションからScienceDirectにアクセス
3. 補足資料をダウンロード

---

## 📋 ダウンロード後の手順

### ステップ1: ファイルを保存

```bash
cd /home/nishioka/LUH/AI_metallurgy/data_collection
mkdir -p raw_data/gorsse_dataset

# ダウンロードしたファイルを移動
mv ~/Downloads/gorsse_data.* raw_data/gorsse_dataset/
# または
mv ~/Downloads/mmc1.* raw_data/gorsse_dataset/
```

### ステップ2: PDFからデータを抽出（PDFの場合）

PDF形式の場合は、以下の方法でデータを抽出できます：

1. **手動でコピー&ペースト**
   - PDFから表をコピー
   - ExcelまたはCSVに貼り付け

2. **Pythonスクリプトで抽出**
   ```python
   import tabula
   df = tabula.read_pdf("mmc1.pdf", pages="all")
   df.to_csv("gorsse_dataset.csv", index=False)
   ```

3. **オンラインツールを使用**
   - PDF to Excel/CSV変換ツール

### ステップ3: データを確認

```bash
cd /home/nishioka/LUH/AI_metallurgy/data_collection
python scripts/download_gorsse.py
```

---

## ✅ 推奨アクション

### 最優先（今すぐ）

1. **訂正版（Corrigendum）からデータを取得**
   - URL: https://www.sciencedirect.com/science/article/pii/S2352340920311100
   - より正確なデータが含まれている

2. **Google Sheets版を使用**（可能な場合）
   - 最も簡単にデータを取得できる
   - 直接CSV/Excel形式でダウンロード可能

3. **データを保存**
   ```bash
   cd /home/nishioka/LUH/AI_metallurgy/data_collection
   mkdir -p raw_data/gorsse_dataset
   mv ~/Downloads/*gorsse* raw_data/gorsse_dataset/
   ```

---

## 📊 期待されるデータ内容

### データ構造

- **合金数**: 約370合金
- **カラム**:
  - 組成（Composition）
  - 微細組織（Microstructure）
  - 密度（Density）
  - 硬度（Hardness, Vickers）
  - 降伏強度（Yield Strength）
  - 引張強度（Ultimate Tensile Strength）
  - 伸び（Elongation）
  - **弾性率（Young's Modulus）** ⭐

### データ範囲

- **期間**: 2004-2016年
- **合金タイプ**: HEA/CCA
- **難融HEA**: 27合金について温度依存データあり

---

## 🔗 すべてのリンク（まとめ）

| 方法 | URL | 優先度 |
|------|-----|--------|
| **訂正版（Corrigendum）** | https://www.sciencedirect.com/science/article/pii/S2352340920311100 | ⭐⭐⭐⭐⭐ |
| **元の論文** | https://www.sciencedirect.com/science/article/pii/S235234091831504X | ⭐⭐⭐⭐ |
| **PubMed** | https://pubmed.ncbi.nlm.nih.gov/30761350/ | ⭐⭐⭐ |

---

## 💡 重要なポイント

1. **訂正版を使用することを強く推奨**
   - 2020年に公開された訂正版には誤りが修正されている
   - より正確なデータが含まれている

2. **Google Sheets版が最も便利**
   - 直接CSV/Excel形式でダウンロード可能
   - データが構造化されている

3. **PDF形式の場合は変換が必要**
   - PDFからデータを抽出する必要がある
   - オンラインツールやPythonスクリプトを使用可能

---

## 📋 チェックリスト

- [ ] 訂正版（Corrigendum）のURLにアクセス
- [ ] 補足資料をダウンロード
- [ ] データファイルを `raw_data/gorsse_dataset/` に保存
- [ ] PDFの場合はCSV/Excelに変換
- [ ] データ内容を確認（弾性率データの有無）
- [ ] データ数を確認（約370合金）

---

**最終更新**: 2026年1月20日
