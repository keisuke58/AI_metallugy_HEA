# 🚀 Gorsse Dataset クイックスタート

**更新日**: 2026年1月20日

---

## 🎯 最速でデータを取得する方法

### ステップ1: 訂正版（Corrigendum）にアクセス

**URL**: https://www.sciencedirect.com/science/article/pii/S2352340920311100

**理由**: 2020年に公開された訂正版には誤りが修正されており、より正確なデータが含まれています。

---

### ステップ2: 補足資料をダウンロード

1. 上記のURLにアクセス
2. 「Supplementary Material」セクションを確認
3. データファイルをダウンロード
   - PDF形式（mmc1.pdf）の場合: CSV/Excelに変換が必要
   - Google Sheets版の場合: 直接CSV/Excel形式でダウンロード可能

---

### ステップ3: ファイルを保存

```bash
cd /home/nishioka/LUH/AI_metallurgy/data_collection
mkdir -p raw_data/gorsse_dataset

# ダウンロードしたファイルを移動
mv ~/Downloads/*gorsse* raw_data/gorsse_dataset/
# または
mv ~/Downloads/mmc1.* raw_data/gorsse_dataset/
```

---

### ステップ4: データを確認

```bash
cd /home/nishioka/LUH/AI_metallurgy/data_collection
python scripts/download_gorsse.py
```

---

## 📊 期待される結果

- **データ数**: 約370合金
- **弾性率データ**: ✅ あり
- **データ範囲**: 2004-2016年

---

## 🔗 クイックリンク

- **訂正版（推奨）**: https://www.sciencedirect.com/science/article/pii/S2352340920311100
- **元の論文**: https://www.sciencedirect.com/science/article/pii/S235234091831504X
- **PubMed**: https://pubmed.ncbi.nlm.nih.gov/30761350/

---

**最終更新**: 2026年1月20日
