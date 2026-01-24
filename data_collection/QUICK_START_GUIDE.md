# 🚀 クイックスタートガイド

**作成日**: 2026年1月20日

---

## 📋 概要

このガイドでは、Materials Project APIキーを取得してデータを収集し、最新論文からデータを抽出する手順を説明します。

---

## 🔑 ステップ1: Materials Project APIキーを取得

### 1.1 アカウント作成

1. https://materialsproject.org にアクセス
2. 「Sign Up」をクリック
3. アカウント情報を入力して登録
4. メール認証を完了

### 1.2 APIキーを取得

1. ログイン後、Dashboardにアクセス: https://materialsproject.org/dashboard
2. 「API Keys」セクションを確認
3. APIキーをコピー

### 1.3 APIキーを設定

```bash
# 環境変数に設定
export MP_API_KEY='your_api_key_here'

# 確認
echo $MP_API_KEY
```

詳細は `MATERIALS_PROJECT_API_GUIDE.md` を参照してください。

---

## 📦 ステップ2: 必要なパッケージをインストール

```bash
cd /home/nishioka/LUH/AI_metallurgy/data_collection
pip install mp-api pandas numpy openpyxl
```

---

## 🔍 ステップ3: Materials Projectからデータを取得

```bash
# APIキーが環境変数に設定されている場合
python scripts/download_materials_project_enhanced.py

# または、コマンドライン引数で指定
python scripts/download_materials_project_enhanced.py --api-key YOUR_KEY

# 最大取得数を指定（デフォルト: 5000）
python scripts/download_materials_project_enhanced.py --api-key YOUR_KEY --max-materials 10000
```

**期待される結果**: 500-1000サンプル

---

## 📚 ステップ4: 最新論文からデータを抽出

### 4.1 論文を検索

1. **PubMed**: https://pubmed.ncbi.nlm.nih.gov/
   - 検索クエリ: `("high entropy alloy" OR "HEA") AND ("elastic modulus" OR "Young's modulus") AND ("2020"[Publication Date] : "2025"[Publication Date])`

2. **arXiv**: https://arxiv.org/
   - 検索: "high entropy alloy elastic modulus"

3. **Materials Cloud**: https://www.materialscloud.org/

### 4.2 データを抽出

1. 論文の表や補足資料からデータを抽出
2. CSVファイルに整理
3. `raw_data/literature_data/` に保存

### 4.3 データを統合

```bash
python scripts/extract_literature_data.py
```

**期待される結果**: 100-300サンプル

詳細は `LITERATURE_DATA_EXTRACTION_GUIDE.md` を参照してください。

---

## 🔄 ステップ5: すべてのデータを統合

```bash
python scripts/final_data_integration.py
```

このスクリプトは以下を統合します：
- 既存データ（DOE/OSTI、Gorsse等）
- Materials Projectデータ
- 文献データ
- その他のデータソース

**期待される結果**: 最大2000サンプル

---

## 📊 ステップ6: 結果を確認

```bash
# 最終データセットを確認
ls -lh final_data/

# データ統計を確認
python -c "
import pandas as pd
df = pd.read_csv('final_data/final_dataset_YYYYMMDD_HHMMSS.csv')
print(f'総サンプル数: {len(df)}')
print(f'弾性率範囲: {df[\"elastic_modulus\"].min():.2f} - {df[\"elastic_modulus\"].max():.2f} GPa')
print(f'平均: {df[\"elastic_modulus\"].mean():.2f} GPa')
"
```

---

## ✅ チェックリスト

- [ ] Materials Project APIキーを取得
- [ ] APIキーを環境変数に設定
- [ ] 必要なパッケージをインストール
- [ ] Materials Projectからデータを取得
- [ ] 最新論文からデータを抽出
- [ ] 文献データをCSVに整理
- [ ] 文献データを統合
- [ ] すべてのデータを統合
- [ ] 結果を確認

---

## 🆘 トラブルシューティング

### APIキーが認識されない

```bash
# 環境変数を確認
echo $MP_API_KEY

# 再設定
export MP_API_KEY='your_api_key_here'
```

### mp-apiパッケージのエラー

```bash
# 再インストール
pip install --upgrade mp-api
```

### データが取得できない

- APIキーが正しいか確認
- インターネット接続を確認
- Materials Projectのサーバー状態を確認

---

## 📁 作成されるファイル

- `collected_data/materials_project_YYYYMMDD_HHMMSS.csv` - Materials Projectデータ
- `collected_data/literature_data_YYYYMMDD_HHMMSS.csv` - 文献データ
- `final_data/final_dataset_YYYYMMDD_HHMMSS.csv` - 最終統合データ

---

## 🔗 参考資料

- `MATERIALS_PROJECT_API_GUIDE.md` - Materials Project API詳細ガイド
- `LITERATURE_DATA_EXTRACTION_GUIDE.md` - 文献データ抽出詳細ガイド
- `DATA_COLLECTION_FINAL_REPORT.md` - データ収集最終レポート

---

**最終更新**: 2026年1月20日
