# 📊 ダウンロードしたデータのサマリー

**更新日**: 2026年1月20日

---

## ✅ ダウンロード完了したデータセット

### 1. DISMA Research HEA Dataset ✅

**ファイル**: `DISMA_Research Dataset (High-entropy alloys).zip` (3.9 MB)

**内容**:
- **Training_data_independent_Predictor_phase_n=76.csv**: 76行、6列（相情報）
- **Training_data_independent_Predictor_phase_n=94.csv**: 94行、6列（相情報）
- **Training_data_Strength.csv**: 強度データ
- **Training_data_Elongation.csv**: 伸びデータ
- **Training_data_VAE.csv**: VAE用データ
- **Training_data_independent_Predictor_hardness-190521.csv**: 硬度データ
- **Generatic_latentSpace.csv**: 3.3 MB（潜在空間データ）
- **Prediction_data-latent-space-230317.csv**: 7.6 MB（予測用データ）

**特徴**:
- 機械学習用に整理されている
- 相情報、組成情報が含まれている
- 弾性率データは直接含まれていない可能性（要確認）

---

### 2. MPEA Mechanical Properties Database ✅

**ファイル**: `A database of mechanical properties for multi prin.zip` (201 KB)

**内容**:
- **MPEA_parsed_mechanical_database.xlsx**: 1713行、47列

**カラム**:
- 組成情報（Composition）
- 相情報（Phases present）
- 機械的特性:
  - Hardness (HVN)
  - Yield Strength (MPa)
  - Ultimate Tensile Strength (MPa)
  - Elongation (%)
  - Compressive strength (MPa)
  - Plasticity (%) - Compressive
- 元素組成（Ag, Al, B, C, Ca, Co, Cr, Cu, Fe, Ga, Ge, Hf, Li, Mg, Mn, Mo, N, Nb, Nd, Ni, Pd, Re, Sc, Si, Sn, Ta, Ti, V, W, Y, Zn, Zr）
- 処理条件、出典情報

**特徴**:
- ✅ **1713サンプル**の多主元素合金データ
- ✅ 降伏強度、引張強度、伸びなどの機械的特性データ
- ⚠️ 弾性率（Young's modulus）データは含まれていない可能性（要確認）

---

## 📊 現在のデータ収集状況

| データセット | データ数 | 弾性率データ | ステータス |
|------------|---------|------------|----------|
| **DOE/OSTI Dataset** | 107 | ✅ あり | ✅ 完了 |
| **最新研究データ** | 4 | ✅ あり | ✅ 完了 |
| **DISMA Research HEA** | 76-94 | ❓ 要確認 | ✅ ダウンロード完了 |
| **MPEA Properties** | 1713 | ❌ なし（強度データあり） | ✅ ダウンロード完了 |
| **合計（弾性率あり）** | **111+** | **✅** | **⏳ 進行中** |

---

## 🎯 重要な発見

### 1. MPEA Datasetについて

- ✅ **1713サンプル**の豊富なデータ
- ✅ 降伏強度、引張強度、伸びなどの機械的特性データ
- ❌ 弾性率データは含まれていない
- 💡 **活用方法**: 
  - 強度データから弾性率を推定するモデルを構築
  - または、強度データを補助特徴量として使用

### 2. DISMA Datasetについて

- ✅ 機械学習用に整理されている
- ✅ 相情報、組成情報が豊富
- ❓ 弾性率データの有無を確認中
- 💡 **活用方法**: 
  - 相情報を特徴量として使用
  - 組成情報から材料記述子を計算

---

## 📈 データ数の再評価

### 弾性率データが直接含まれているデータセット

- **DOE/OSTI Dataset**: 107サンプル ✅
- **最新研究データ**: 4サンプル ✅
- **合計**: 111サンプル

### 補助データとして活用可能

- **MPEA Dataset**: 1713サンプル（強度データ）
- **DISMA Dataset**: 76-94サンプル（相・組成データ）

---

## 💡 推奨戦略

### 戦略1: 現在の111サンプルで進める（推奨）

**理由**:
- 111サンプルで基本的なモデル訓練は可能
- 最低限のデータ数（100-200）を満たしている

**可能なモデル**:
- Linear Regression
- Ridge/Lasso Regression
- KNN（kを小さく設定）

### 戦略2: 強度データから弾性率を推定

**手法**:
- MPEA Datasetの強度データを補助特徴量として使用
- 強度と弾性率の相関を利用

### 戦略3: データ拡張

**手法**:
- 既存の111サンプルからデータ拡張
- ノイズ追加、補間手法

---

## ✅ 次のステップ

### 最優先

1. **DISMA Datasetの詳細確認**
   - 他のファイルに弾性率データが含まれているか確認
   - 特に `Training_data_Strength.csv` などを確認

2. **データの統合準備**
   - すべてのデータセットの構造を理解
   - 統合可能な形式に変換

### 次優先

3. **特徴量エンジニアリング**
   - 組成情報から材料記述子を計算
   - 相情報を特徴量として追加

4. **モデル訓練の開始**
   - 111サンプルで基本的なモデルを訓練
   - 性能を評価

---

## 📋 チェックリスト

- [x] DISMA Datasetをダウンロード・解凍
- [x] MPEA Datasetをダウンロード・解凍
- [ ] DISMA Datasetの詳細分析（弾性率データの有無）
- [ ] データの統合準備
- [ ] データ数を最終確認

---

**最終更新**: 2026年1月20日
