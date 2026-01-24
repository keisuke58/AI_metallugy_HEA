# 📊 データ収集の現在の状況

**更新日**: 2026年1月20日

---

## ✅ 完了したデータセット

### 1. DOE/OSTI Dataset ✅

- **データ数**: 107合金
- **弾性率データ**: あり
- **弾性率範囲**: 27-466 GPa
- **目標範囲（30-90 GPa）内の合金**: 11個 ⭐
- **ファイル**: 
  - youngsdata.xlsx (107合金)
  - phasesdata.xlsx (340合金)

### 2. 最新研究データ ✅

- **データ数**: 4合金
- **弾性率データ**: あり
- **データ範囲**: 64-86 GPa

---

## ❌ 取得できなかったデータセット

### Gorsse Dataset

- **理由**: ウェブサイトから取得できなかった
- **代替案**: 
  1. 論文の著者に連絡
  2. 代替データセットを使用

---

## ⏳ 手動ダウンロードが必要なデータセット

### 1. DISMA Research HEA Dataset ⭐⭐⭐⭐⭐

- **URL**: https://data.mendeley.com/datasets/p3txdrdth7/1
- **ステータス**: 未ダウンロード
- **優先度**: 最高

### 2. MPEA Mechanical Properties Database ⭐⭐⭐⭐⭐

- **URL**: https://data.mendeley.com/datasets/4d4kpfwpf6
- **ステータス**: 未ダウンロード
- **優先度**: 最高

### 3. Fracture and Impact Toughness Dataset ⭐⭐⭐⭐

- **URL**: https://www.nature.com/articles/s41597-022-01911-4
- **ステータス**: 未ダウンロード
- **優先度**: 中

---

## 📈 現在のデータ収集状況

| データセット | ステータス | データ数 | 目標 |
|------------|----------|---------|------|
| Gorsse Dataset | ❌ 取得不可 | 0 | ~370 |
| **DOE/OSTI Dataset** | **✅ 完了** | **107** | **107** |
| Materials Project | ⏳ 未開始 | 0 | 補完用 |
| **最新研究データ** | **✅ 完了** | **4** | **10-20** |
| DISMA Research HEA | ⏳ 手動DL必要 | 0 | 不明 |
| MPEA Properties | ⏳ 手動DL必要 | 0 | 不明 |
| **合計** | **⏳ 進行中** | **111** | **400-500** |

### 進捗率

- **現在**: 111 / 500 = **22.2%** ⏳
- **目標達成まで**: 389合金が必要

---

## 🎯 次のアクション

### 最優先（今すぐ）

1. **DISMA Research HEA Datasetのダウンロード**
   - URL: https://data.mendeley.com/datasets/p3txdrdth7/1
   - 「Download All」ボタンをクリック
   - `raw_data/disma_hea_dataset/` に保存

2. **MPEA Mechanical Properties Databaseのダウンロード**
   - URL: https://data.mendeley.com/datasets/4d4kpfwpf6
   - 「Download All」ボタンをクリック
   - `raw_data/mpea_mechanical_properties/` に保存

### 次優先

3. **データの統合とクリーニング**
   - すべてのデータセットを統合
   - 重複データの除去
   - データの正規化

---

## 💡 重要なポイント

### 現在のデータで進められること

- ✅ **111合金のデータ**: 機械学習モデルの訓練は可能
- ✅ **目標範囲内の合金**: 11個の合金が目標範囲（30-90 GPa）内
- ✅ **データ品質**: DOE/OSTI Datasetは高品質

### データが少ない場合の対策

1. **データ拡張**: 既存データから新しいデータを生成
2. **転移学習**: 他の材料データで事前訓練
3. **アンサンブル手法**: 複数モデルの組み合わせ
4. **適応的実験設計**: 予測不確実性の高い領域で実験を追加

---

## 📋 チェックリスト

### データ収集
- [x] DOE/OSTI Datasetをダウンロード
- [x] 最新研究データを作成
- [ ] DISMA Research HEA Datasetをダウンロード
- [ ] MPEA Mechanical Properties Databaseをダウンロード
- [ ] Fracture and Impact Toughness Datasetをダウンロード（オプション）

### データ処理
- [ ] データを統合
- [ ] データクリーニング
- [ ] 重複を除去
- [ ] 外れ値を処理

---

## 🔗 クイックリンク

- **DISMA Research HEA**: https://data.mendeley.com/datasets/p3txdrdth7/1
- **MPEA Properties**: https://data.mendeley.com/datasets/4d4kpfwpf6
- **Fracture Toughness**: https://www.nature.com/articles/s41597-022-01911-4

---

**最終更新**: 2026年1月20日
