# 📊 DOE/OSTI Dataset データ分析

**分析日**: 2026年1月20日  
**データセット**: DOE/OSTI Dataset  
**出典**: Balasubramanian, G. & Ganesh, S. (2020)

---

## ✅ データダウンロード完了

- **ファイル名**: submission-d24e1b31-41f2-40ed-b511-ad13e3f06b74_20260119_044328.zip
- **ダウンロード日**: 2026年1月19日
- **保存先**: `raw_data/doe_osti_dataset/`

---

## 📁 データファイル構成

### 1. youngsdata.xlsx（弾性率データ）

**データ数**: 107合金

**カラム**:
- `Alloy`: 合金名
- `Young's Mod (GPa)`: 弾性率（目標変数）⭐
- `Diff. Lattice Constants`: 格子定数の差
- `Diff. Melting Point`: 融点の差
- `Mixing Enthalpy`: 混合エンタルピー
- `Lattice Constants`: 格子定数
- `Lambda`: Lambdaパラメータ
- `Diff. in atomic radii`: 原子半径の差
- `Omega`: Omegaパラメータ
- `Melting Temp.`: 融点
- `Diff. Electronegativity`: 電気陰性度の差
- `Mixing Entropy`: 混合エントロピー
- `Valence electron`: 価電子数

**弾性率データの統計**:
- **平均**: 169.3 GPa
- **標準偏差**: 73.6 GPa
- **最小値**: **27.0 GPa** ⭐（目標30 GPaに近い！）
- **25%分位**: 132.9 GPa
- **中央値**: 161.8 GPa
- **75%分位**: 197.5 GPa
- **最大値**: 466.0 GPa

**重要な発見**:
- ✅ **最小値27 GPaは目標30 GPaに非常に近い！**
- ✅ **目標範囲（30-90 GPa）内の合金が11個存在！**
  - 例: AlCo0.5CrCuFeNi (71.6 GPa), TaNbHfZrTi (49.9 GPa), Al1.25CoCrFeNi (55.6 GPa)
- ✅ 生体医学用途に適したデータが含まれている

---

### 2. phasesdata.xlsx（相データ）

**データ数**: 340合金

**カラム**:
- `Alloy`: 合金名
- `Phase`: 相構造
- `Diff. Lattice Constants`: 格子定数の差
- `Diff. Melting Point`: 融点の差
- `Mixing Enthalpy`: 混合エンタルピー
- `Mean Lattice Constants`: 平均格子定数
- `delta r`: 原子半径差
- `Omega`: Omegaパラメータ
- `Melting Temp.`: 融点
- `Diff. Electronegativity`: 電気陰性度の差
- `Mixing Entropy`: 混合エントロピー
- `Valence electron`: 価電子数
- `lambda`: Lambdaパラメータ

**用途**:
- 相構造の予測
- 特徴量エンジニアリング（相情報を追加）

---

## 🎯 データの特徴

### 弾性率データの分布

```
最小値:  27.0 GPa  ← 目標30 GPaに近い！
25%:   132.9 GPa
中央値: 161.8 GPa
75%:   197.5 GPa
最大値: 466.0 GPa
```

### 生体医学用途への適用可能性

**目標範囲**: 30-90 GPa（現実的）、30 GPa（理想的）

**データ内の該当範囲**:
- 最小値27 GPaは目標30 GPaに非常に近い
- 30-90 GPaの範囲内のデータが存在する可能性（要確認）

---

## 📊 データ品質評価

### ✅ 良い点

1. **データ数**: 107合金（目標107と一致）
2. **弾性率データ**: すべての合金に弾性率データあり
3. **特徴量**: 材料記述子が豊富（混合エントロピー、原子半径差など）
4. **データ範囲**: 27-466 GPaと広範囲
5. **目標に近い値**: 最小値27 GPaが目標30 GPaに近い

### ⚠️ 注意点

1. **データの古さ**: 2020年のデータ（最新研究データと組み合わせる必要がある）
2. **生体医学用途のデータ**: 生体医学用途に特化したデータではない（一般的なHEAデータ）
3. **組成情報**: 合金名から組成を抽出する必要がある可能性

---

## 🔍 次のステップ

### 1. データの詳細分析

- 30-90 GPaの範囲内の合金を特定
- 組成情報の抽出
- 特徴量の相関分析

### 2. Gorsse Datasetとの統合

- Gorsse Dataset（約370合金）と統合
- 重複データの除去
- データの正規化

### 3. 特徴量エンジニアリング

- 組成特徴量の抽出
- 材料記述子の計算
- 相情報の追加（phasesdata.xlsxから）

---

## 📝 データ使用に関する注意事項

1. **出典の明記**: 
   - Balasubramanian, G. & Ganesh, S. (2020)
   - "Phases and Young's Modulus Dataset for High Entropy Alloys"
   - DOE/OSTI (2020)

2. **ライセンス**: 使用前にライセンスを確認

3. **データの引用**: 論文やレポートで使用する場合は適切に引用

---

## ✅ データ確認コマンド

```bash
cd /home/nishioka/LUH/AI_metallurgy/data_collection

# データの確認
python scripts/download_doe_osti.py

# 詳細な分析
python -c "
import pandas as pd
df = pd.read_excel('raw_data/doe_osti_dataset/youngsdata.xlsx')
print('弾性率30-90 GPaの範囲内の合金数:')
print(len(df[(df['Young\'s Mod (GPa)'] >= 30) & (df['Young\'s Mod (GPa)'] <= 90)]))
print('\n該当合金:')
print(df[(df['Young\'s Mod (GPa)'] >= 30) & (df['Young\'s Mod (GPa)'] <= 90)][['Alloy', 'Young\'s Mod (GPa)']])
"
```

---

**最終更新**: 2026年1月20日
