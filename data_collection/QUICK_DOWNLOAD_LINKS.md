# 🔗 クイックダウンロードリンク

**作成日**: 2026年1月20日

---

## 🎯 現在のデータ状況

- ✅ **DOE/OSTI Dataset**: 107合金（完了）
- ✅ **最新研究データ**: 4合金（完了）
- **合計: 111合金**

---

## 📥 手動ダウンロードリンク（クリックしてダウンロード）

### 1. DISMA Research HEA Dataset ⭐⭐⭐⭐⭐（最推奨）

**直接リンク**: https://data.mendeley.com/datasets/p3txdrdth7/1

**手順**:
1. 上記のリンクをクリック
2. ページが開いたら「Download All」ボタンをクリック
3. ダウンロードしたZIPファイルを解凍
4. 以下のコマンドで移動:
   ```bash
   cd /home/nishioka/LUH/AI_metallurgy/data_collection
   mv ~/Downloads/*disma* raw_data/disma_hea_dataset/ 2>/dev/null || true
   mv ~/Downloads/*p3txdrdth7* raw_data/disma_hea_dataset/ 2>/dev/null || true
   ```

---

### 2. MPEA Mechanical Properties Database ⭐⭐⭐⭐⭐

**直接リンク**: https://data.mendeley.com/datasets/4d4kpfwpf6

**手順**:
1. 上記のリンクをクリック
2. ページが開いたら「Download All」ボタンをクリック
3. ダウンロードしたファイルを解凍
4. 以下のコマンドで移動:
   ```bash
   cd /home/nishioka/LUH/AI_metallurgy/data_collection
   mv ~/Downloads/*mpea* raw_data/mpea_mechanical_properties/ 2>/dev/null || true
   mv ~/Downloads/*4d4kpfwpf6* raw_data/mpea_mechanical_properties/ 2>/dev/null || true
   ```

---

### 3. Fracture and Impact Toughness Dataset ⭐⭐⭐⭐

**論文リンク**: https://www.nature.com/articles/s41597-022-01911-4

**手順**:
1. 上記のリンクをクリック
2. 「Data Availability」セクションを確認
3. Materials Cloudのリンクからデータをダウンロード
4. 以下のコマンドで移動:
   ```bash
   cd /home/nishioka/LUH/AI_metallurgy/data_collection
   mv ~/Downloads/*fracture* raw_data/fracture_toughness_dataset/ 2>/dev/null || true
   ```

---

## ✅ ダウンロード後の確認

すべてのファイルをダウンロードしたら、以下を実行:

```bash
cd /home/nishioka/LUH/AI_metallurgy/data_collection

# データ収集状況の確認
python scripts/check_data_status.py

# 代替データセットの確認
python scripts/download_alternative_datasets.py
```

---

## 📊 期待される結果

### DISMA Research HEA Datasetを追加した場合
- **追加データ数**: 不明（データセットの内容による）
- **合計**: 111 + α 合金

### MPEA Mechanical Properties Databaseを追加した場合
- **追加データ数**: 不明（データセットの内容による）
- **合計**: 111 + α 合金

### 両方を追加した場合
- **合計**: 111 + α + β 合金
- **目標**: 400-500合金に近づく可能性

---

## 💡 推奨アクション

1. **まずDISMA Research HEA Datasetをダウンロード**
   - 比較的新しいデータ（2023年）
   - 機械学習用に整理されている

2. **次にMPEA Mechanical Properties Databaseをダウンロード**
   - 多主元素合金の包括的なデータ

3. **データを確認・分析**
   - 弾性率データが含まれているか確認
   - データ数を確認

---

## 🔗 すべてのリンク（まとめ）

| データセット | リンク | 優先度 |
|------------|--------|--------|
| **DISMA Research HEA** | https://data.mendeley.com/datasets/p3txdrdth7/1 | ⭐⭐⭐⭐⭐ |
| **MPEA Mechanical Properties** | https://data.mendeley.com/datasets/4d4kpfwpf6 | ⭐⭐⭐⭐⭐ |
| **Fracture and Impact Toughness** | https://www.nature.com/articles/s41597-022-01911-4 | ⭐⭐⭐⭐ |

---

**最終更新**: 2026年1月20日
