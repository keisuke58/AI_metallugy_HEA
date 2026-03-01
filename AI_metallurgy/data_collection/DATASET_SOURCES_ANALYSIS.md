# 📊 データセット出典分析と統一可能性レポート

**作成日**: 2026年1月23日  
**目的**: 現在使用されているデータセットの出典を特定し、統一可能性を確認。追加データセットを探索

---

## 📋 目次

1. [現在使用中のデータセット一覧](#現在使用中のデータセット一覧)
2. [各データセットの詳細分析](#各データセットの詳細分析)
3. [統一可能性の評価](#統一可能性の評価)
4. [追加データセット探索結果](#追加データセット探索結果)
5. [推奨事項](#推奨事項)

---

## 現在使用中のデータセット一覧

### ✅ 取得済みデータセット

| # | データセット名 | 出典 | データ数 | 弾性率データ | ステータス |
|---|--------------|------|---------|------------|----------|
| 1 | **DOE/OSTI Dataset** | DOE Energy Data eXchange | 107合金 | ✅ あり | ✅ 完了 |
| 2 | **Gorsse Dataset** | Data in Brief (2018) | ~370合金 | ✅ あり | ⚠️ 部分取得 |
| 3 | **最新研究データ** | 文献から手動抽出 | 4合金 | ✅ あり | ✅ 完了 |
| 4 | **DISMA Research HEA** | Mendeley Data | 76-94サンプル | ❓ 要確認 | ✅ ダウンロード済み |
| 5 | **MPEA Mechanical Properties** | Mendeley Data | 1,713サンプル | ❌ なし（強度のみ） | ✅ ダウンロード済み |
| 6 | **Materials Project** | Materials Project API | 変動 | ✅ 計算値 | ⏳ API必要 |

---

## 各データセットの詳細分析

### 1. DOE/OSTI Dataset ⭐⭐⭐⭐⭐

**出典情報**:
- **正式名称**: Phases and Young's Modulus Dataset for High Entropy Alloys
- **提供機関**: Lehigh University / DOE NETL
- **DOI**: 10.18141/1644295
- **URL**: https://edx.netl.doe.gov/dataset/phases-and-young-s-modulus-dataset-for-high-entropy-alloys
- **公開年**: 2020年

**データ内容**:
- **youngsdata.xlsx**: 107合金のYoung's modulusデータ
- **phasesdata.xlsx**: 340合金の相データ
- 11個の計算された特徴量を含む

**データ構造**:
- 合金名、組成、弾性率（GPa）、相情報
- 特徴量: mixing entropy, mixing enthalpy, valence electron等

**統一可能性**: ✅ **高**
- CSV/Excel形式で統一可能
- カラム名の標準化が必要

---

### 2. Gorsse Dataset ⭐⭐⭐⭐⭐

**出典情報**:
- **論文**: Gorsse, S., Nguyen, M. H., Senkov, O. N., & Miracle, D. B. (2018)
- **タイトル**: "Database on the mechanical properties of high entropy alloys and complex concentrated alloys"
- **ジャーナル**: Data in Brief, 2018
- **PubMed ID**: 30761350
- **URL**: https://pubmed.ncbi.nlm.nih.gov/30761350/
- **HAL Archive**: https://hal.science/hal-02156875/

**データ内容**:
- 約370合金（2004-2016年）
- Young's modulus（弾性率）
- 降伏強度、引張強度、硬度、伸び

**現在の状況**:
- ファイル: `raw_data/gorsse_dataset/1-s2.0-S2352340920311100-mmc1.xlsx`
- 抽出済み: `collected_data/gorsse_elastic_modulus.csv`

**統一可能性**: ✅ **高**
- Excel形式から抽出済み
- カラム名の標準化が必要

---

### 3. 最新研究データ ⭐⭐⭐

**出典情報**:
- **ソース**: 2024-2025年の最新論文から手動抽出
- **ファイル**: `raw_data/latest_research/latest_research.csv`

**データ内容**:
- Ti-Zr-Nb系MEA: 62-66 GPa
- Ti-Zr-Hf-Nb-Ta HEA: 69 GPa
- Ti40Zr25Nb25Ta5Mo5 HEA: 86 GPa
- Ti-Nb-Ta-Cr-Co HEA: 82 GPa

**統一可能性**: ✅ **高**
- 既にCSV形式
- 標準フォーマットに準拠

---

### 4. DISMA Research HEA Dataset ⭐⭐⭐⭐

**出典情報**:
- **プラットフォーム**: Mendeley Data
- **DOI**: 10.17632/p3txdrdth7.1
- **URL**: https://data.mendeley.com/datasets/p3txdrdth7/1
- **公開年**: 2023年8月

**データ内容**:
- 機械的特性と構造的特徴
- 機械学習モデル用に整理
- ファイル:
  - `Training_data_Strength.csv`: 強度データ
  - `Training_data_Elongation.csv`: 伸びデータ
  - `Training_data_independent_Predictor_phase_n=76.csv`: 相データ（76サンプル）
  - `Training_data_independent_Predictor_phase_n=94.csv`: 相データ（94サンプル）

**弾性率データ**: ❓ **要確認**
- 直接的な弾性率カラムは不明
- 強度データから推定可能かもしれない

**統一可能性**: ⚠️ **中**
- CSV形式で統一可能
- 弾性率データの有無を確認する必要がある

---

### 5. MPEA Mechanical Properties Database ⭐⭐⭐⭐

**出典情報**:
- **論文**: Li, Zeng, Taheri, Birbilis et al.
- **タイトル**: "A database of mechanical properties for multi-principal element alloys"
- **プラットフォーム**: Mendeley Data
- **DOI**: 10.17632/4d4kpfwpf6
- **URL**: https://data.mendeley.com/datasets/4d4kpfwpf6

**データ内容**:
- **データ数**: 1,713サンプル
- **プロパティ**: 引張/圧縮強度、硬度、伸び
- **対象**: 多主元素合金（MPEA）、HEAを含む
- **ファイル**: `MPEA_parsed_mechanical_database.xlsx`

**弾性率データ**: ❌ **なし**
- 強度、硬度、伸びデータのみ
- 補助特徴量として使用可能

**統一可能性**: ⚠️ **中**
- 弾性率データがないため、補助データとして使用
- 強度から弾性率を推定するモデルに活用可能

---

### 6. Materials Project ⭐⭐⭐

**出典情報**:
- **プラットフォーム**: Materials Project Database
- **URL**: https://materialsproject.org
- **API**: https://api.materialsproject.org/docs
- **アクセス**: APIキー必要（無料登録）

**データ内容**:
- 第一原理計算による弾性テンソル
- バルク弾性率、せん断弾性率
- Young's modulusは計算可能: E = 9KG/(3K+G)

**制限**:
- ⚠️ HEAのデータは限定的
- ⚠️ 計算値（実験値ではない）
- ⚠️ APIキーが必要

**統一可能性**: ✅ **高**
- API経由で自動取得可能
- 標準フォーマットで出力可能

---

## 統一可能性の評価

### ✅ 統一可能なデータセット

以下のデータセットは統一フォーマットに変換可能：

1. **DOE/OSTI Dataset** ✅
   - 標準カラム: `alloy_name`, `elastic_modulus`, `composition`, `phases`
   - 形式: Excel → CSV変換済み

2. **Gorsse Dataset** ✅
   - 標準カラム: `alloy_name`, `elastic_modulus`, `composition`
   - 形式: Excel → CSV抽出済み

3. **最新研究データ** ✅
   - 標準カラム: `alloy_name`, `elastic_modulus`
   - 形式: CSV（既に統一済み）

4. **Materials Project** ✅
   - 標準カラム: `material_id`, `alloy_name`, `elastic_modulus`（計算値）
   - 形式: API → CSV変換可能

### ⚠️ 要確認・補助データセット

5. **DISMA Research HEA** ⚠️
   - 弾性率データの有無を確認する必要がある
   - 強度データは補助特徴量として使用可能

6. **MPEA Mechanical Properties** ⚠️
   - 弾性率データなし
   - 補助特徴量（強度、硬度）として使用可能

### 📊 統一フォーマット提案

```csv
alloy_name,elastic_modulus,composition,phases,source,yield_strength,ultimate_strength,hardness,elongation,year,notes
```

**必須カラム**:
- `alloy_name`: 合金名
- `elastic_modulus`: 弾性率（GPa）
- `source`: データソース名

**推奨カラム**:
- `composition`: 組成情報
- `phases`: 相情報
- `yield_strength`, `ultimate_strength`, `hardness`, `elongation`: 補助プロパティ

---

## 追加データセット探索結果

### 🔍 新たに発見したデータセット

#### 1. Refractory HEA Elastic Constants (GitHub) ⭐⭐⭐⭐⭐

**出典情報**:
- **リポジトリ**: https://github.com/uttambhandari91/Elastic-constant-DFT-data
- **データ数**: 370 refractory high entropy alloys
- **形式**: Excel (materials_journal_elastic_constant_data.xlsx)
- **内容**: 弾性定数データ

**アクセス方法**:
```bash
# GitHubから直接ダウンロード可能
git clone https://github.com/uttambhandari91/Elastic-constant-DFT-data.git
# または
wget https://github.com/uttambhandari91/Elastic-constant-DFT-data/raw/main/materials_journal_elastic_constant_data.xlsx
```

**統一可能性**: ✅ **高**
- Excel形式、直接ダウンロード可能

---

#### 2. Multi-Principal Element Alloy Nano-indentation Database (2024) ⭐⭐⭐⭐⭐

**出典情報**:
- **論文**: Johns Hopkins University Applied Physics Laboratory
- **公開年**: 2024年7月
- **データ数**: 7,385 indentation tests on 19 different MPEAs
- **内容**: Phase-specific mechanical properties via nano-indentation
- **PubMed**: PMC11298849

**データ内容**:
- ナノインデンテーションによる機械的特性
- 相特異的な機械的特性
- 弾性率データが含まれる可能性が高い

**アクセス方法**:
- PubMedから論文を検索
- 補足資料からデータをダウンロード

**統一可能性**: ✅ **高**
- 最新データ（2024年）
- ナノインデンテーションデータは弾性率を含む可能性が高い

---

#### 3. ChemDataExtractor Database ⭐⭐⭐⭐

**出典情報**:
- **論文**: Nature Scientific Data (2024)
- **データ数**: 720,308 data records
- **内容**: 科学文献から自動抽出された材料データ
- **プロパティ**: Ultimate tensile strength, yield strength, fracture strength, **Young's modulus**, ductility
- **精度**: 82.03% precision, 92.13% recall

**URL**: 
- 論文: https://www.nature.com/articles/s41597-024-03979-6
- データ: Cambridge Repository

**統一可能性**: ✅ **高**
- 自動抽出データベース
- Young's modulusを含む
- 大量のデータ（720Kレコード）

---

#### 4. Dryad Elastic Properties Database ⭐⭐⭐⭐

**出典情報**:
- **プラットフォーム**: Dryad
- **DOI**: 10.5061/dryad.h505v
- **URL**: https://datadryad.org/stash/dataset/doi:10.5061/dryad.h505v
- **データ数**: 1,181 inorganic compounds
- **内容**: Complete elastic constant tensor

**データ内容**:
- 完全な弾性定数テンソル
- 無機結晶化合物
- HEAに直接適用可能かは要確認

**統一可能性**: ✅ **高**
- 標準フォーマットで提供
- 弾性定数テンソルからYoung's modulusを計算可能

---

#### 5. OQMD (Open Quantum Materials Database) ⭐⭐⭐

**出典情報**:
- **URL**: https://oqmd.org/
- **API**: https://oqmd.org/api/
- **データ数**: ~1.3 million DFT-calculated materials
- **アクセス**: RESTful API（認証不要）

**データ内容**:
- DFT計算による熱力学・構造特性
- 弾性特性が含まれる可能性

**統一可能性**: ✅ **高**
- API経由で自動取得可能
- Pythonパッケージ (`qmpy-rester`) 利用可能

---

#### 6. AFLOW Database ⭐⭐⭐

**出典情報**:
- **URL**: http://aflow.org
- **API**: https://aflowlib.duke.edu/AFLOWDATA/LIB2_WEB/
- **データ数**: 1,706 entries (binary/ternary compounds)
- **アクセス**: REST-API v1.0

**データ内容**:
- 計算材料データ
- 弾性特性が含まれる可能性

**統一可能性**: ✅ **高**
- API経由でアクセス可能
- 無料（学術・非営利用途）

---

#### 7. MatWeb Database ⭐⭐⭐

**出典情報**:
- **URL**: https://matweb.com
- **データ数**: 185,000+ materials
- **内容**: 金属、プラスチック、セラミック、複合材料

**データ内容**:
- 弾性率を含む機械的特性
- 登録ユーザーはフォルダ機能でデータ収集可能
- プレミアム会員はCAD/FEAソフトへのエクスポート可能

**統一可能性**: ⚠️ **中**
- 手動ダウンロードが必要
- スクレイピングは利用規約を確認する必要がある

---

### 📊 追加データセットまとめ表

| # | データセット名 | データ数 | 弾性率 | アクセス | 優先度 |
|---|--------------|---------|--------|---------|--------|
| 1 | Refractory HEA Elastic Constants (GitHub) | 370 | ✅ | 直接DL | ⭐⭐⭐⭐⭐ |
| 2 | MPEA Nano-indentation (2024) | 7,385 tests | ✅? | 論文補足 | ⭐⭐⭐⭐⭐ |
| 3 | ChemDataExtractor Database | 720K | ✅ | リポジトリ | ⭐⭐⭐⭐ |
| 4 | Dryad Elastic Properties | 1,181 | ✅ | 直接DL | ⭐⭐⭐⭐ |
| 5 | OQMD | 1.3M | ✅? | API | ⭐⭐⭐ |
| 6 | AFLOW | 1,706 | ✅? | API | ⭐⭐⭐ |
| 7 | MatWeb | 185K+ | ✅ | 手動 | ⭐⭐⭐ |

---

## 推奨事項

### 🎯 即座に取得すべきデータセット（最優先）

1. **Refractory HEA Elastic Constants (GitHub)** ⭐⭐⭐⭐⭐
   - **理由**: 370サンプルのrefractory HEA弾性定数データ、直接ダウンロード可能
   - **アクション**: GitHubから即座にダウンロード
   - **期待データ数**: +370サンプル

2. **MPEA Nano-indentation Database (2024)** ⭐⭐⭐⭐⭐
   - **理由**: 最新データ（2024年）、7,385テスト、19 MPEAs
   - **アクション**: PubMedから論文を検索し、補足資料をダウンロード
   - **期待データ数**: +19-100サンプル（弾性率データが含まれる場合）

### 📈 次優先データセット

3. **ChemDataExtractor Database** ⭐⭐⭐⭐
   - **理由**: 720Kレコード、Young's modulusを含む、自動抽出データ
   - **アクション**: Cambridge Repositoryからデータをダウンロード
   - **注意**: HEAに絞り込む必要がある

4. **Dryad Elastic Properties** ⭐⭐⭐⭐
   - **理由**: 1,181化合物の完全な弾性定数テンソル
   - **アクション**: Dryadから直接ダウンロード
   - **注意**: HEAに適用可能か確認が必要

### 🔧 統一化の実装

#### Step 1: データセットの標準化スクリプト作成

```python
# 統一フォーマットへの変換スクリプト
def standardize_dataset(df, source_name):
    """データセットを統一フォーマットに変換"""
    standardized = pd.DataFrame()
    
    # 必須カラムのマッピング
    column_mapping = {
        'alloy_name': ['alloy_name', 'Alloy', 'Material', 'Name'],
        'elastic_modulus': ['elastic_modulus', 'Young\'s Modulus', 'E', 'E (GPa)'],
        'composition': ['composition', 'Composition', 'Formula'],
        'phases': ['phases', 'Phases', 'Phase'],
        'source': ['source', 'Source']
    }
    
    # カラムマッピングを適用
    for standard_col, possible_cols in column_mapping.items():
        for col in possible_cols:
            if col in df.columns:
                standardized[standard_col] = df[col]
                break
    
    # ソース情報を追加
    standardized['source'] = source_name
    
    return standardized
```

#### Step 2: 統合スクリプトの実行

```bash
# すべてのデータセットを統合
python scripts/unify_all_datasets.py
```

### 📊 期待されるデータ数の増加

| データセット | 現在 | 追加後 | 増加 |
|------------|------|--------|------|
| DOE/OSTI | 107 | 107 | - |
| Gorsse | ~370 | ~370 | - |
| 最新研究 | 4 | 4 | - |
| **Refractory HEA (GitHub)** | **0** | **370** | **+370** |
| **MPEA Nano-indentation** | **0** | **19-100** | **+19-100** |
| **ChemDataExtractor (HEA抽出)** | **0** | **100-500** | **+100-500** |
| **合計** | **~481** | **~970-1,451** | **+489-970** |

**目標達成**: 現在の約481サンプルから、**970-1,451サンプル**に増加可能

---

## アクションプラン

### Phase 1: 即座に実行（今週中）

1. ✅ Refractory HEA Elastic ConstantsをGitHubからダウンロード
2. ✅ MPEA Nano-indentation DatabaseをPubMedから取得
3. ✅ 既存データセットの統一フォーマット変換スクリプト作成

### Phase 2: 短期（2週間以内）

4. ✅ ChemDataExtractor DatabaseからHEAデータを抽出
5. ✅ Dryad Elastic Properties Databaseをダウンロード
6. ✅ すべてのデータセットを統合

### Phase 3: 中期（1ヶ月以内）

7. ✅ OQMD APIからHEA関連データを取得
8. ✅ AFLOW APIからデータを取得
9. ✅ データ品質チェックとクリーニング

---

## まとめ

### ✅ 統一可能性の結論

**すべてのデータセットは統一可能**です。主な課題は：

1. **カラム名の標準化**: 各データセットで異なるカラム名を使用
2. **単位の統一**: GPaへの統一が必要
3. **データ品質**: 欠損値、異常値の処理が必要

### 📈 データ数の増加見込み

- **現在**: ~481サンプル
- **追加後**: ~970-1,451サンプル
- **増加率**: +102-202%

### 🎯 次のステップ

1. Refractory HEA Elastic Constantsを即座にダウンロード
2. 統一フォーマット変換スクリプトを作成・実行
3. すべてのデータセットを統合
4. データ品質チェックを実施

---

**最終更新**: 2026年1月23日  
**作成者**: AI Assistant  
**ステータス**: ✅ 完了
