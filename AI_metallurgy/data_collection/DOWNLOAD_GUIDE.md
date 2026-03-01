# 📥 データダウンロードガイド（手動ダウンロード用）

**作成日**: 2026年1月20日

---

## 🎯 現在の状況

- ✅ **DOE/OSTI Dataset**: ダウンロード完了（107合金）
- ❌ **Gorsse Dataset**: ウェブサイトから取得できなかった
- ⏳ **代替データセット**: 手動ダウンロードが必要

---

## 📋 代替データセットのダウンロード手順

### 1. DISMA Research HEA Dataset ⭐⭐⭐⭐⭐（最推奨）

**URL**: https://data.mendeley.com/datasets/p3txdrdth7/1

**手順**:
1. ブラウザで上記のURLにアクセス
2. 「Download All」ボタンをクリック
3. ダウンロードしたZIPファイルを解凍
4. ファイルを以下のディレクトリに移動:
   ```bash
   mv ~/Downloads/disma_hea_dataset.* /home/nishioka/LUH/AI_metallurgy/data_collection/raw_data/disma_hea_dataset/
   ```

**データ確認**:
```bash
cd /home/nishioka/LUH/AI_metallurgy/data_collection
python scripts/download_alternative_datasets.py
```

---

### 2. MPEA Mechanical Properties Database ⭐⭐⭐⭐⭐

**URL**: https://data.mendeley.com/datasets/4d4kpfwpf6

**手順**:
1. ブラウザで上記のURLにアクセス
2. 「Download All」ボタンをクリック、または
3. Google Sheetsリポジトリから直接ダウンロード
4. ファイルを以下のディレクトリに移動:
   ```bash
   mv ~/Downloads/mpea_mechanical_properties.* /home/nishioka/LUH/AI_metallurgy/data_collection/raw_data/mpea_mechanical_properties/
   ```

**データ確認**:
```bash
cd /home/nishioka/LUH/AI_metallurgy/data_collection
python scripts/download_alternative_datasets.py
```

---

### 3. Fracture and Impact Toughness Dataset ⭐⭐⭐⭐

**論文URL**: https://www.nature.com/articles/s41597-022-01911-4

**手順**:
1. ブラウザで上記のURLにアクセス
2. 「Data Availability」セクションを確認
3. Materials Cloudのリンクからデータをダウンロード
4. ファイルを以下のディレクトリに移動:
   ```bash
   mv ~/Downloads/fracture_toughness.* /home/nishioka/LUH/AI_metallurgy/data_collection/raw_data/fracture_toughness_dataset/
   ```

**注意**: このデータセットは破壊靭性データで、弾性率データは含まれない可能性があります。

---

## 🔄 Gorsse Datasetの代替案

### オプション1: 論文の著者に連絡

**連絡先**:
- 論文: Gorsse, S., Nguyen, M. H., Senkov, O. N., & Miracle, D. B. (2018)
- "Database on the mechanical properties of high entropy alloys and complex concentrated alloys"
- Data in Brief, 2018

**連絡方法**:
1. 論文の著者の連絡先を確認
2. データファイルの提供をリクエスト
3. 研究目的を説明

---

### オプション2: 既存のデータで進める

**現在のデータ**:
- ✅ DOE/OSTI Dataset: 107合金
- ✅ 最新研究データ: 4合金
- **合計: 111合金**

**推奨**:
- 111合金でも機械学習モデルの訓練は可能
- データが少ない場合は、データ拡張や転移学習を検討
- 目標400-500合金には届かないが、プロジェクトは進められる

---

## 📊 データ収集の優先順位

### 最優先（今すぐ）

1. **DISMA Research HEA Dataset**
   - URL: https://data.mendeley.com/datasets/p3txdrdth7/1
   - 比較的新しいデータ（2023年）
   - 機械学習用に整理されている

2. **MPEA Mechanical Properties Database**
   - URL: https://data.mendeley.com/datasets/4d4kpfwpf6
   - 多主元素合金の包括的なデータ
   - Google Sheetsからアクセス可能

### 次優先

3. **Fracture and Impact Toughness Dataset**
   - 破壊靭性データ（弾性率は含まれない可能性）

4. **Gorsse Datasetの再試行**
   - 論文の著者に連絡
   - または、論文の補足資料を再度確認

---

## ✅ ダウンロード後の確認コマンド

```bash
cd /home/nishioka/LUH/AI_metallurgy/data_collection

# すべてのデータセットの状況を確認
python scripts/check_data_status.py

# 代替データセットの確認
python scripts/download_alternative_datasets.py

# DOE/OSTI Datasetの確認
python scripts/download_doe_osti.py
```

---

## 📝 メモ

- 現在111合金のデータがある（DOE/OSTI + 最新研究）
- 目標400-500合金には届かないが、プロジェクトは進められる
- 代替データセットを追加することで、データ数を増やせる可能性がある

---

**最終更新**: 2026年1月20日
