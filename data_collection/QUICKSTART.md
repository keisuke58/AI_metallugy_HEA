# 🚀 データ収集クイックスタートガイド

**作成日**: 2026年1月20日

---

## 📋 ステップ1: 環境のセットアップ

### Pythonパッケージのインストール

```bash
cd /home/nishioka/LUH/AI_metallurgy/data_collection
pip install -r scripts/requirements.txt
```

---

## 📥 ステップ2: データセットのダウンロード

### 2.1 Gorsse Dataset（最優先）⭐⭐⭐⭐⭐

**手動ダウンロード**（推奨）:

1. **論文ページにアクセス**:
   - URL: https://pubmed.ncbi.nlm.nih.gov/30761350/
   - または検索: "Gorsse HEA database"

2. **補足資料を確認**:
   - 論文ページの「Supplementary Material」または「Data Availability」セクションを確認
   - データファイル（通常はCSVまたはExcel形式）をダウンロード

3. **ファイルを保存**:
   ```bash
   # ダウンロードしたファイルを以下のディレクトリに移動
   mv ~/Downloads/gorsse_data.* data_collection/raw_data/gorsse_dataset/
   ```

**データ確認**:
```bash
cd data_collection
python scripts/download_gorsse.py
```

---

### 2.2 DOE/OSTI Dataset ⭐⭐⭐⭐⭐

**手動ダウンロード**:

1. **OSTI Data Explorerにアクセス**:
   - URL: https://www.osti.gov/dataexplorer/biblio/dataset/1644295

2. **データセットをダウンロード**:
   - 「Download」ボタンをクリック
   - データファイル（通常はCSVまたはJSON形式）をダウンロード

3. **ファイルを保存**:
   ```bash
   # ダウンロードしたファイルを以下のディレクトリに移動
   mv ~/Downloads/doe_osti_data.* data_collection/raw_data/doe_osti_dataset/
   ```

**データ確認**:
```bash
cd data_collection
python scripts/download_doe_osti.py
```

---

### 2.3 最新研究データ（2024-2025）⭐⭐⭐⭐

**手動で収集**:

以下のデータを `data_collection/raw_data/latest_research/` にCSVファイルとして保存:

```csv
alloy_system,elastic_modulus_gpa,source,year
Ti-Zr-Nb MEA,62-66,論文,2025
Ti-Zr-Hf-Nb-Ta HEA,69,論文,2025
Ti40Zr25Nb25Ta5Mo5 HEA,86,論文,2024
Ti-Nb-Ta-Cr-Co HEA,82,論文,2024
```

または、`latest_research.csv` ファイルを作成:

```bash
cat > data_collection/raw_data/latest_research/latest_research.csv << EOF
alloy_system,elastic_modulus_gpa,source,year
Ti-Zr-Nb MEA,64,論文,2025
Ti-Zr-Hf-Nb-Ta HEA,69,論文,2025
Ti40Zr25Nb25Ta5Mo5 HEA,86,論文,2024
Ti-Nb-Ta-Cr-Co HEA,82,論文,2024
EOF
```

---

## ✅ ステップ3: データ収集状況の確認

```bash
cd data_collection
python scripts/check_data_status.py
```

**期待される出力**:
- Gorsse Dataset: ✅ 1個以上のファイル
- DOE/OSTI Dataset: ✅ 1個以上のファイル
- 最新研究データ: ✅ 1個のCSVファイル

---

## 📊 ステップ4: 次のステップ

データ収集が完了したら:

1. **データの統合**: `scripts/merge_datasets.py`（作成予定）
2. **データのクリーニング**: `scripts/clean_data.py`（作成予定）
3. **特徴量エンジニアリング**: プロジェクトの次のフェーズ

---

## 🔗 参考リンク

- **Gorsse Dataset**: https://pubmed.ncbi.nlm.nih.gov/30761350/
- **DOE/OSTI Dataset**: https://www.osti.gov/dataexplorer/biblio/dataset/1644295
- **Materials Project**: https://materialsproject.org
- **AFLOW**: http://aflow.org

---

## ❓ よくある質問

### Q: データファイルが見つからない場合は？

A: 以下の方法を試してください:
1. 論文の補足資料を再度確認
2. 論文の著者に連絡してデータをリクエスト
3. 代替データソースを検討（Materials Projectなど）

### Q: データの形式が異なる場合は？

A: データの形式に応じて、スクリプトを調整してください。または、手動でデータを前処理してください。

### Q: データが少ない場合は？

A: 複数のデータソースを組み合わせて、データ数を増やしてください。目標は400-500合金です。

---

**最終更新**: 2026年1月20日
