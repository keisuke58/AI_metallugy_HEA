# 🔍 Gorsse Dataset - PubMed検索結果の分析

**更新日**: 2026年1月20日

---

## 📊 PubMed検索結果

**検索語**: "Gorsse"  
**結果数**: 41件

### 重要な年

- **2018年**: 2件 ⭐⭐⭐（Gorsse Datasetの公開年）
- **2020年**: 3件
- **2023年**: 4件
- **2024年**: 1件
- **2025年**: 1件

---

## 🎯 ターゲット論文

### 2018年の論文（最有力候補）

**タイトル**: "Database on the mechanical properties of high entropy alloys and complex concentrated alloys"

**著者**: Gorsse, S., Nguyen, M. H., Senkov, O. N., & Miracle, D. B.

**ジャーナル**: Data in Brief (2018)

**PubMed ID**: 確認必要

**データ内容**:
- 約370合金（2004-2016年）
- 降伏強度、引張強度、伸び、弾性率、密度、硬度

---

## 📥 データ取得方法

### 方法1: PubMedから直接アクセス

1. **PubMedで論文を検索**
   - URL: https://pubmed.ncbi.nlm.nih.gov/
   - 検索語: "Gorsse" AND "high entropy alloy" AND "2018"
   - または: "Database on the mechanical properties of high entropy alloys"

2. **論文ページを開く**
   - 2018年の論文を選択
   - 特に "Data in Brief" ジャーナルの論文

3. **Supplementary Materialを確認**
   - 論文ページの「Supplementary Material」セクション
   - または「Data Availability」セクション
   - 通常はCSVまたはExcel形式

4. **データファイルをダウンロード**
   - ファイル名: "Supplementary Material" または "Data"
   - ダウンロード後、`raw_data/gorsse_dataset/` に保存

### 方法2: ScienceDirectからアクセス

1. **ScienceDirectで論文を検索**
   - URL: https://www.sciencedirect.com/science/article/pii/S235234091831504X
   - または: "Database on the mechanical properties of high entropy alloys" で検索

2. **補足資料を確認**
   - 論文ページの「Supplementary Material」セクション
   - データファイルをダウンロード

### 方法3: 著者に連絡

**連絡先の確認方法**:
1. 論文の著者情報を確認
2. 所属機関のウェブサイトから連絡先を取得
3. 研究目的を説明してデータの提供をリクエスト

---

## 🔗 直接リンク

### ScienceDirect（推奨）

**URL**: https://www.sciencedirect.com/science/article/pii/S235234091831504X

**手順**:
1. 上記のURLにアクセス
2. 「Supplementary Material」セクションを確認
3. データファイルをダウンロード
4. ダウンロードしたファイルを `raw_data/gorsse_dataset/` に保存

---

## 📋 チェックリスト

- [ ] PubMedで2018年の論文を確認
- [ ] ScienceDirectで論文を確認
- [ ] 補足資料（Supplementary Material）をダウンロード
- [ ] データファイルを `raw_data/gorsse_dataset/` に保存
- [ ] データ内容を確認（弾性率データの有無）

---

## 💡 重要なポイント

1. **2018年の論文が最有力候補**
   - Gorsse Datasetの公開年
   - "Data in Brief" ジャーナル

2. **補足資料から直接ダウンロード可能**
   - 通常はCSVまたはExcel形式
   - 論文ページから直接アクセス可能

3. **ScienceDirectが最も確実**
   - 論文の完全な補足資料が利用可能
   - 直接ダウンロード可能

---

## ✅ 次のステップ

### 最優先（今すぐ）

1. **ScienceDirectで論文を確認**
   - URL: https://www.sciencedirect.com/science/article/pii/S235234091831504X
   - 補足資料をダウンロード

2. **データファイルを保存**
   ```bash
   cd /home/nishioka/LUH/AI_metallurgy/data_collection
   mkdir -p raw_data/gorsse_dataset
   mv ~/Downloads/gorsse_data.* raw_data/gorsse_dataset/
   ```

3. **データを確認**
   ```bash
   python scripts/download_gorsse.py
   ```

---

**最終更新**: 2026年1月20日
