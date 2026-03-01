# 🔄 代替データセット一覧

**作成日**: 2026年1月20日  
**目的**: Gorsse Datasetが取得できない場合の代替データソース

---

## 📊 利用可能な代替データセット

### 1. DISMA Research HEA Dataset ⭐⭐⭐⭐⭐

**出典**: 
- Mendeley Data
- DOI: 10.17632/p3txdrdth7.1
- 公開日: 2023年8月

**データ内容**:
- 機械的特性と構造的特徴
- 機械学習モデル用に整理
- 弾性率データが含まれる可能性

**アクセス方法**:
1. URLにアクセス: https://data.mendeley.com/datasets/p3txdrdth7/1
2. 「Download All」ボタンをクリック
3. ダウンロードしたファイルを `raw_data/disma_hea_dataset/` に保存

**URL**: https://data.mendeley.com/datasets/p3txdrdth7/1

---

### 2. MPEA Mechanical Properties Database ⭐⭐⭐⭐⭐

**出典**:
- Li, Zeng, Taheri, Birbilis et al.
- "A database of mechanical properties for multi-principal element alloys"
- Mendeley Data
- DOI: 10.17632/4d4kpfwpf6

**データ内容**:
- 引張/圧縮強度、硬度、伸び
- 多主元素合金（MPEA）のデータ
- HEAを含む
- 微細組織と処理条件のメタデータ

**アクセス方法**:
1. URLにアクセス: https://data.mendeley.com/datasets/4d4kpfwpf6
2. Google Sheetsリポジトリからダウンロード可能
3. ダウンロードしたファイルを `raw_data/mpea_mechanical_properties/` に保存

**URL**: https://data.mendeley.com/datasets/4d4kpfwpf6

---

### 3. Fracture and Impact Toughness Dataset ⭐⭐⭐⭐

**出典**:
- Nature Scientific Data
- "Dataset for Fracture and Impact Toughness of High-Entropy Alloys"
- 公開日: 2022年

**データ内容**:
- 破壊靭性、衝撃靭性、衝撃エネルギー
- 2022年までのデータ
- 微細組織、結晶粒径、処理履歴
- 引張特性（降伏強度、伸び）

**アクセス方法**:
1. 論文ページにアクセス: https://www.nature.com/articles/s41597-022-01911-4
2. 「Data Availability」セクションを確認
3. Materials Cloudからダウンロード
4. ダウンロードしたファイルを `raw_data/fracture_toughness_dataset/` に保存

**URL**: 
- 論文: https://www.nature.com/articles/s41597-022-01911-4
- Materials Cloud: https://www.materialscloud.org/

---

### 4. Materials Project API ⭐⭐⭐

**出典**: Materials Project Database

**データ内容**:
- 第一原理計算による弾性テンソル
- バルク弾性率、せん断弾性率
- Young's modulusは計算可能

**アクセス方法**:
1. Materials Projectにアカウント作成: https://materialsproject.org
2. APIキーを取得
3. Python APIを使用してデータを取得

**制限**:
- ⚠️ HEAのデータは限定的
- ⚠️ 計算値（実験値ではない）
- ⚠️ APIキーが必要

---

## 🚀 ダウンロード手順

### DISMA Research HEA Dataset

```bash
cd /home/nishioka/LUH/AI_metallurgy/data_collection

# 1. ブラウザで以下にアクセス
# https://data.mendeley.com/datasets/p3txdrdth7/1

# 2. 「Download All」ボタンをクリック

# 3. ダウンロードしたファイルを移動
mv ~/Downloads/disma_hea_dataset.* raw_data/disma_hea_dataset/

# 4. データを確認
python scripts/download_alternative_datasets.py
```

---

### MPEA Mechanical Properties Database

```bash
# 1. ブラウザで以下にアクセス
# https://data.mendeley.com/datasets/4d4kpfwpf6

# 2. Google Sheetsからダウンロード、または直接ダウンロード

# 3. ダウンロードしたファイルを移動
mv ~/Downloads/mpea_mechanical_properties.* raw_data/mpea_mechanical_properties/

# 4. データを確認
python scripts/download_alternative_datasets.py
```

---

### Fracture and Impact Toughness Dataset

```bash
# 1. ブラウザで以下にアクセス
# https://www.nature.com/articles/s41597-022-01911-4

# 2. 「Data Availability」セクションからMaterials Cloudのリンクにアクセス

# 3. データをダウンロード

# 4. ダウンロードしたファイルを移動
mv ~/Downloads/fracture_toughness.* raw_data/fracture_toughness_dataset/

# 5. データを確認
python scripts/download_alternative_datasets.py
```

---

## ✅ 推奨順序

### 最優先

1. **DISMA Research HEA Dataset** ⭐⭐⭐⭐⭐
   - 2023年公開、比較的新しい
   - 機械学習用に整理されている
   - Mendeley Dataから直接ダウンロード可能

2. **MPEA Mechanical Properties Database** ⭐⭐⭐⭐⭐
   - 多主元素合金の包括的なデータ
   - Google Sheetsからアクセス可能
   - 弾性率データが含まれる可能性

### 次優先

3. **Fracture and Impact Toughness Dataset** ⭐⭐⭐⭐
   - 2022年までのデータ
   - 破壊靭性データ（弾性率は含まれない可能性）

4. **Materials Project API** ⭐⭐⭐
   - 計算値のみ
   - APIキーが必要

---

## 📋 チェックリスト

- [ ] DISMA Research HEA Datasetをダウンロード
- [ ] MPEA Mechanical Properties Databaseをダウンロード
- [ ] Fracture and Impact Toughness Datasetをダウンロード（オプション）
- [ ] Materials Project APIでデータを取得（オプション）
- [ ] データを確認・分析
- [ ] DOE/OSTI Datasetと統合

---

## 🔗 参考リンク

- **DISMA Research HEA Dataset**: https://data.mendeley.com/datasets/p3txdrdth7/1
- **MPEA Mechanical Properties**: https://data.mendeley.com/datasets/4d4kpfwpf6
- **Fracture and Impact Toughness**: https://www.nature.com/articles/s41597-022-01911-4
- **Materials Project**: https://materialsproject.org

---

**最終更新**: 2026年1月20日
