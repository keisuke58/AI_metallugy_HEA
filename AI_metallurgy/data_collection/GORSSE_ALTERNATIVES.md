# 🔄 Gorsse Datasetの代替案

**作成日**: 2026年1月20日

---

## 🎯 Gorsse Datasetについて

### 元のデータセット

- **論文**: Gorsse, S., Nguyen, M. H., Senkov, O. N., & Miracle, D. B. (2018)
- **タイトル**: "Database on the mechanical properties of high entropy alloys and complex concentrated alloys"
- **データ数**: 約370合金（2004-2016年）
- **データ内容**: 降伏強度、引張強度、伸び、弾性率、密度、硬度

### アクセス方法

**元の論文**: https://www.sciencedirect.com/science/article/pii/S235234091831504X

**補足資料**: 論文のSupplementary Materialからダウンロード可能

---

## ✅ 代替データセット

### 1. MPEA Mechanical Properties Database ⭐⭐⭐⭐⭐（最推奨）

**出典**: Li, Zeng, Taheri, Birbilis et al.
- **URL**: https://data.mendeley.com/datasets/4d4kpfwpf6
- **データ内容**: 降伏強度、引張強度、伸び、硬度、相、処理条件
- **範囲**: 多主元素合金（HEAを含む）
- **公開日**: 2023年7月
- **ライセンス**: CC-BY 4.0

**利点**:
- ✅ Gorsse Datasetより新しい
- ✅ より多くのメタデータ（処理条件、微細組織）
- ✅ 直接ダウンロード可能

**注意**: 弾性率データが含まれているか確認が必要

---

### 2. DISMA Research HEA Dataset ⭐⭐⭐⭐⭐

**出典**: DISMA Research
- **URL**: https://data.mendeley.com/datasets/p3txdrdth7/1
- **データ内容**: 機械的特性と構造的特徴
- **公開日**: 2023年8月
- **特徴**: 機械学習用に整理されている

**利点**:
- ✅ 機械学習用に整理されている
- ✅ 比較的新しいデータ
- ✅ 直接ダウンロード可能

---

### 3. DOE/OSTI Dataset ⭐⭐⭐⭐⭐（既に取得済み）

**出典**: Balasubramanian, G. & Ganesh, S. (2020)
- **URL**: https://www.osti.gov/dataexplorer/biblio/dataset/1644295
- **データ数**: 107合金（弾性率）、340合金（相）
- **データ内容**: 弾性率、相構造、材料記述子

**利点**:
- ✅ 既に取得済み
- ✅ 弾性率データが豊富
- ✅ 材料記述子が含まれている

---

### 4. Fracture and Impact Toughness Dataset ⭐⭐⭐

**出典**: Nature Scientific Data
- **URL**: https://www.nature.com/articles/s41597-022-01911-4
- **データ内容**: 破壊靭性、衝撃靭性、引張特性
- **データ数**: 約153サンプル（破壊靭性）

**注意**: 弾性率データは含まれない可能性が高い

---

## 🔍 Gorsse Datasetの取得方法（再試行）

### 方法1: 論文の補足資料から直接ダウンロード

1. **論文ページにアクセス**:
   - https://www.sciencedirect.com/science/article/pii/S235234091831504X

2. **Supplementary Materialを確認**:
   - 論文ページの「Supplementary Material」セクション
   - または「Data Availability」セクション

3. **データファイルをダウンロード**:
   - 通常はCSVまたはExcel形式
   - ファイル名: "Supplementary Material" または "Data"

### 方法2: 論文の著者に連絡

**連絡先の確認方法**:
1. 論文の著者情報を確認
2. 所属機関のウェブサイトから連絡先を取得
3. 研究目的を説明してデータの提供をリクエスト

**推奨メール文例**:
```
件名: Request for HEA Database Data

Dear Dr. Gorsse,

I am working on a research project on machine learning 
for high entropy alloy elastic modulus prediction. 
I would like to request access to the database 
published in your 2018 Data in Brief paper.

[研究目的の説明]

I would be very grateful if you could provide 
access to the supplementary data files.

Best regards,
[Your Name]
```

### 方法3: データベースの再検索

**検索キーワード**:
- "Gorsse HEA database"
- "high entropy alloy mechanical properties database"
- "HEA database 2018"

**検索先**:
- Google Scholar
- ResearchGate
- Zenodo
- Figshare

---

## 📊 データセットの比較

| データセット | データ数 | 弾性率 | 公開年 | アクセス |
|------------|---------|--------|--------|---------|
| **Gorsse Dataset** | ~370 | ✅ | 2018 | ⚠️ 困難 |
| **MPEA Properties** | 不明 | ? | 2023 | ✅ 容易 |
| **DISMA HEA** | 不明 | ? | 2023 | ✅ 容易 |
| **DOE/OSTI** | 107 | ✅ | 2020 | ✅ 完了 |
| **Fracture Toughness** | ~153 | ❌ | 2022 | ✅ 容易 |

---

## 💡 推奨戦略

### 戦略1: 代替データセットで進める（推奨）

1. **MPEA Mechanical Properties Databaseを使用**
   - Gorsse Datasetより新しい
   - より多くのメタデータ
   - 直接ダウンロード可能

2. **DISMA Research HEA Datasetを使用**
   - 機械学習用に整理されている
   - 比較的新しい

3. **既存のDOE/OSTI Datasetと統合**
   - 107合金の弾性率データ
   - 材料記述子が豊富

### 戦略2: Gorsse Datasetを再試行

1. **論文の補足資料を確認**
   - Supplementary Materialセクション
   - Data Availabilityセクション

2. **著者に連絡**
   - 研究目的を説明
   - データの提供をリクエスト

---

## ✅ 次のステップ

### 最優先

1. **ダウンロードしたデータを確認**
   - DISMA Research HEA Dataset
   - MPEA Mechanical Properties Database
   - データ内容とデータ数を確認

2. **データの統合**
   - すべてのデータセットを統合
   - 重複を除去
   - データ数を集計

### 次優先

3. **Gorsse Datasetの再試行**
   - 論文の補足資料を確認
   - 著者に連絡（必要に応じて）

---

## 📋 チェックリスト

- [ ] ダウンロードしたデータを確認
- [ ] データ内容を分析（弾性率データの有無）
- [ ] データ数を集計
- [ ] Gorsse Datasetの補足資料を確認
- [ ] 必要に応じて著者に連絡

---

**最終更新**: 2026年1月20日
