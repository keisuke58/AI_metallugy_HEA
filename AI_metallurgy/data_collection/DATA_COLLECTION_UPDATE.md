# 🎉 DOE/OSTI Dataset ダウンロード完了！

**更新日**: 2026年1月20日

---

## ✅ 完了したタスク

### DOE/OSTI Dataset ⭐⭐⭐⭐⭐

- [x] ダウンロード完了
- [x] 解凍完了
- [x] データ確認完了
- [x] データ分析完了

---

## 📊 データの詳細

### ファイル構成

1. **youngsdata.xlsx** (28.90 KB)
   - **107合金**の弾性率データ
   - 弾性率範囲: 27-466 GPa
   - 平均: 169.3 GPa
   - **目標範囲（30-90 GPa）内の合金: 11個** ⭐

2. **phasesdata.xlsx** (56.51 KB)
   - **340合金**の相データ
   - 相構造情報を含む

3. **submission-d24e1b31-41f2-40ed-b511-ad13e3f06b74_20260119_044328.zip** (80.56 KB)
   - 元のZIPファイル

---

## 🎯 重要な発見

### 目標範囲（30-90 GPa）内の合金

**11個の合金が目標範囲内！**

| 合金名 | 弾性率 (GPa) |
|--------|------------|
| AlCo0.5CrCuFeNi | 71.60 |
| CuNiCoFeCrAl0.5V0.4 | 71.90 |
| FeNiCrCuMo | 81.90 |
| CuNi2FeCrAl0.5 | 78.10 |
| Al1.25CoCrFeNi | 55.60 |
| Al0.5CoCrCuFeNiTi0.2 | 76.50 |
| AlCoCr1.6FeNi4 | 88.46 |
| CoCrFeNi2.1Nb0.2 | 47.80 |
| CoCrFeNi2.1Nb0.4 | 53.00 |
| TaNbHfZrTi | 49.88 |

**特に注目すべき合金**:
- **TaNbHfZrTi (49.9 GPa)**: Ti系HEA、生体医学用途に適している可能性
- **Al1.25CoCrFeNi (55.6 GPa)**: 目標範囲の中央値付近
- **CoCrFeNi2.1Nb0.2 (47.8 GPa)**: 目標に近い

---

## 📈 データ収集の進捗更新

### 更新前
- データ数: 4合金
- 進捗率: 0.8%

### 更新後
- **データ数: 111合金** (107 + 4)
- **進捗率: 22.2%** ⬆️
- **目標達成まで: 389合金**（Gorsse Datasetで約370合金追加予定）

---

## 🎯 次のステップ

### 最優先

1. **Gorsse Datasetのダウンロード**
   - URL: https://pubmed.ncbi.nlm.nih.gov/30761350/
   - 約370合金を追加予定
   - ダウンロード後、合計約481合金になる予定

### 次優先

2. **データの統合とクリーニング**
   - DOE/OSTI DatasetとGorsse Datasetの統合
   - 重複データの除去
   - データの正規化

3. **特徴量エンジニアリング**
   - 組成特徴量の抽出
   - 材料記述子の計算
   - 相情報の追加

---

## ✅ データ確認コマンド

```bash
cd /home/nishioka/LUH/AI_metallurgy/data_collection

# データ収集状況の確認
python scripts/check_data_status.py

# DOE/OSTI Datasetの詳細確認
python scripts/download_doe_osti.py

# 目標範囲内の合金を確認
python -c "
import pandas as pd
df = pd.read_excel('raw_data/doe_osti_dataset/youngsdata.xlsx')
target = df[(df['Young\'s Mod (GPa)'] >= 30) & (df['Young\'s Mod (GPa)'] <= 90)]
print(f'目標範囲（30-90 GPa）内の合金数: {len(target)}')
print(target[['Alloy', 'Young\'s Mod (GPa)']].to_string(index=False))
"
```

---

## 📝 メモ

- ✅ DOE/OSTI Datasetのダウンロードと確認が完了
- ✅ 目標範囲内の合金が11個見つかった（非常に良い結果！）
- ⏳ Gorsse Datasetのダウンロードが残っている（約370合金）
- 📊 データ収集は順調に進行中（22.2%完了）

---

**最終更新**: 2026年1月20日
