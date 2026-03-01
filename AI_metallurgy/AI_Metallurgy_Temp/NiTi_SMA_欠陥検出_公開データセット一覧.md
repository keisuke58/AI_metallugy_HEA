# NiTi SMA欠陥検出：公開データセット一覧

**作成日**: 2026年1月20日  
**目的**: NiTi SMAの欠陥検出にGNNを適用するためのデータセット調査

---

## 📋 結論

### ⚠️ 直接的なNiTi SMA欠陥検出データセットは**限定的**

しかし、以下のデータセットを活用できます：
1. **NiTi SMA関連データセット**（材料特性、相変態データ）
2. **複合材料（CFRP）の欠陥検出データセット**（転移学習に使用可能）
3. **XCT（X-ray CT）データセット**（3D内部構造）
4. **材料科学の一般データベース**（Materials Project, AFLOW, NOMAD）

---

## ✅ NiTi SMA関連データセット

### 1. NASA Shape Memory Materials Database (SMMD)

**URL**: https://technology.nasa.gov/shape%20memory%20materials%20database

**内容**:
- 形状記憶合金、ポリマー、セラミックスのデータ
- アクチュエーション特性、構造、加工記録
- **NiTiを含む**

**データ形式**:
- オンライン対話型ツール
- ダウンロード可能
- 文献ソースにトレーサブル

**ライセンス**: パブリックアクセス（無料）

**用途**: 材料特性の参照、NiTiの基本特性の確認

---

### 2. Mendeley Data: NiTi Alloys Dataset

**URL**: https://data.mendeley.com/datasets/hcrgx28b9x

**内容**:
- 論文「Precipitate-Induced Stress Field Steering Martensite Nucleation in NiTi Alloys」のデータ
- 機械試験データ
- 超弾性回復ひずみ
- DSC（示差走査熱量測定）データ

**データ形式**:
- CSVファイル
- 生測定データ
- 複数のデータファイル

**ライセンス**: Creative Commons BY 4.0

**用途**: 
- 組成と相変態温度の関係
- 機械的特性の予測
- レーザー加工NiTiのデータ

---

### 3. Experimental & Numerical Data for Transformation Propagation in NiTi Structures

**URL**: https://www.sciencedirect.com/science/article/pii/S2352340919309217

**内容**:
- 応力誘起マルテンサイト相変態の伝播データ
- 様々な幾何形状でのデータ
- 熱場、応力・ひずみ場

**データ形式**:
- 数値データ（FEA）
- 熱画像データセット
- Data in Brief形式

**ライセンス**: Creative Commons（オープンアクセス）

**用途**:
- 相変態伝播のモデリング
- 熱機械挙動の理解
- FEAシミュレーションの検証

---

### 4. NIMS / JAXA Thermophysical Property Database

**URL**: https://mdr.nims.go.jp/datasets/a38c4c32-fafd-43d8-b0a2-5630b39c8fac

**内容**:
- NiTi合金の実験データ（Ni-50at%Ti, Ni-40at%Tiなど）
- 粘弾性特性
- 融解状態の特性
- 静電浮遊法による測定

**データ形式**:
- 生数値データ
- 画像
- 計器出力

**ライセンス**: Creative Commons BY 4.0

**用途**:
- 熱物理特性
- 高温挙動
- 融解、粘度データ

---

### 5. NIST Interatomic Potentials Repository - Ni-Ti Potentials

**URL**: https://www.ctcms.nist.gov/potentials/system/Ni-Ti/

**内容**:
- Ni-Tiの原子間ポテンシャル
- MEAM / NNIPポテンシャル
- 相変態、欠陥のモデリング用に最適化

**データ形式**:
- LAMMPSポテンシャルファイル
- 計算特性テーブル

**ライセンス**: 無料ダウンロード（引用必須）

**用途**:
- 原子スケールシミュレーション
- 分子動力学計算
- 欠陥の原子スケールモデリング

---

## 🔄 転移学習に使用可能なデータセット

### CFRP（複合材料）欠陥検出データセット

#### 1. Thermal Inspection Dataset for Defect Segmentation in CFRP Laminates

**URL**: https://data.mendeley.com/datasets/jrsb4b9yy5/1

**内容**:
- 熱画像シーケンス（パルスサーモグラフィ）
- セグメンテーションマスク
- 人工欠陥（Kaptonテープ）: 2×2, 3×3, 4×4 mm
- 深さ: 0.13-0.39 mm

**データ形式**:
- 1,034枚のMWIR画像（640×512 px）
- 55 Hz
- マスク整列済み
- ラミネートサンプル: 100×100 mm

**ライセンス**: CC BY 4.0

**用途**: 
- 転移学習の事前訓練
- 欠陥セグメンテーション手法の開発
- GNNモデルの初期化

---

#### 2. Dataset for Surface & Internal Damage after Impact on CFRP Laminates

**URL**: Mendeley Data / PubMed DOI

**内容**:
- 低速度衝撃試験後のデータ
- 表面損傷画像
- 深さ等高線マップ
- 超音波Cスキャン画像（内部損傷）

**データ形式**:
- 複数のセット（クロスプライ、準等方性ラミネート）
- 異なる層数（8, 16, 24層）
- PNG, JPG形式

**ライセンス**: パブリックアクセス

**用途**:
- 内部欠陥の検出
- 3D欠陥位置の特定
- 多モーダルデータの活用

---

#### 3. CFRP Defects Object Detection Dataset (Roboflow Universe)

**URL**: https://universe.roboflow.com/carbonfibredefect/cfrp-defects

**内容**:
- CFRP表面の欠陥画像
- バウンディングボックスアノテーション
- 物体検出用

**データ形式**:
- JPG/PNG画像
- アノテーションファイル

**ライセンス**: CC BY 4.0

**用途**:
- 物体検出モデルの訓練
- 欠陥分類の事前訓練

---

#### 4. Acoustic Emission Monitoring in CFRP Compression Tests

**URL**: https://4tu.edu.hpc.n-helix.com/datasets/a45beba4-30bd-4250-820d-0639c8d1566a

**内容**:
- 圧縮試験中のAE（アコースティックエミッション）信号
- 損傷モード分類データ（デラミネーションなど）
- 超音波Cスキャン画像

**データ形式**:
- CSVファイル（信号データ）
- 画像
- 多数のサンプル

**ライセンス**: CC0

**用途**:
- 時系列データの分析
- 損傷モードの分類
- 多モーダル学習

---

### 金属材料の欠陥検出データセット

#### 5. Image-based Data on Strain Fields of Microstructures with Porosity Defects

**URL**: https://data.mendeley.com/datasets/s4g76zd5ys/1

**内容**:
- 気孔欠陥を含む微細組織画像
- 対応するひずみ場（3成分）
- Al6061合金（SMAではないが構造が類似）

**データ形式**:
- `.mat`形式
- 1,000サンプル
- 100個の円形欠陥（半径0.1-0.5 mm）

**ライセンス**: CC BY 4.0

**用途**:
- 欠陥とひずみ場の関係の学習
- 微細組織認識MLモデルの開発
- NiTiへの転移学習

---

#### 6. Annotated Image Dataset for Defects Detection in Laser Powder Bed Fusion

**URL**: https://research.aalto.fi/en/datasets/annotated-image-dataset-for-defects-detection-in-laser-powder-bed/

**内容**:
- 316Lステンレス鋼のLPBFデータ
- ~2,638枚のパウダーベッド画像
- ~2,674枚の光学トモグラフィ画像
- 欠陥ラベル付き

**データ形式**:
- 露出前後のPB画像
- OT画像
- 手動アノテーション
- 二値分類（良好 vs 欠陥）
- 欠陥位置特定

**ライセンス**: パブリックアクセス

**用途**:
- 積層造形材料の欠陥検出
- 転移学習の事前訓練
- NiTiのAM（積層造形）欠陥への応用

---

#### 7. SEM Dataset of Additively Manufactured Ni-WC Metal Matrix Composites

**URL**: https://zenodo.org/records/17315241

**内容**:
- SEM画像 + ピクセル単位セグメンテーションマスク
- クラス: マトリックス、炭化物粒子、希釈バンド、再析出炭化物
- 後者2つは「劣化特徴」

**データ形式**:
- 405枚の画像クロップ
- 4つの倍率
- セグメンテーションマスク

**ライセンス**: オープンアクセス

**用途**:
- 微細組織特徴のセグメンテーション
- 欠陥/劣化クラスの分類
- NiTiへの適応

---

#### 8. Metal Parts Defect Detection Dataset (MPDD)

**URL**: https://paperswithcode.com/dataset/mpdd

**内容**:
- 製造金属部品の欠陥検出用ベンチマーク
- 1,000枚以上の画像
- ピクセル精度の欠陥マスク

**データ形式**:
- 画像 + アノテーション
- 表面欠陥（内部微細組織欠陥ではない）

**ライセンス**: パブリックアクセス

**用途**:
- アルゴリズムのベンチマーク
- セグメンテーションアーキテクチャの評価

---

#### 9. NIST Microstructure-sensitive Process Models in Near-alpha Titanium Alloys

**URL**: https://materialsdata.nist.gov/handle/11256/647

**内容**:
- SEM画像とEBSD画像（~1 GB+）
- 異なる加工条件での微細組織特性
- 近αチタン合金

**データ形式**:
- 微細組織領域データ
- 粒構造
- 鍛造変形データ

**ライセンス**: パブリックアクセス

**用途**:
- ソースドメインデータ
- NiTiへの転移学習
- 微細組織解析手法の開発

---

## 🔬 XCT（X-ray CT）データセット

### 1. CoCr AM XCT Data (NIST)

**URL**: https://www.nist.gov/el/intelligent-systems-division-73500/cocr-am-xct-data

**内容**:
- 積層造形CoCr部品の高解像度XCT再構成スライス
- LPBFの走査速度とハッチ間隔の変動

**データ形式**:
- ~2.5 µmボクセルサイズ
- 画像スタック: ~1000×1000×1000ボクセル/サンプル
- TIFF 16-bit

**ライセンス**: NIST Open License

**用途**:
- 欠陥検出
- 微細組織特性評価
- 3D内部構造の可視化

---

### 2. X-ray Computed Tomography Data of Dense Metallic Components (ORNL)

**URL**: https://www.osti.gov/dataexplorer/biblio/dataset/2568789

**内容**:
- 六角形燃料ノズルのXCTスキャン
- 3つの部分に分割

**データ形式**:
- HDF5形式の投影データ
- スキャン設定
- ボクセル/検出器ジオメトリメタデータ

**ライセンス**: DOE Data Explorer / ORNL（パブリック）

**用途**:
- トモグラフィアルゴリズム
- スパースビュー再構成
- 計算画像処理

---

### 3. In-situ 3D XCT during Polymer-to-Ceramic Conversion

**URL**: https://acdc.alcf.anl.gov/mdf/detail/uncoatedfiberbeds_insitupyrolysis_v1.1/

**内容**:
- ポリマー由来セラミック変換中の微細組織変化
- リアルタイム観察

**データ形式**:
- 3Dボリュームシリーズ（完全スタックXCT）
- in-situデータ

**ライセンス**: CC-BY 4.0

**用途**:
- 内部構造の進化追跡
- 動的プロセスの理解

---

### 4. XCT Dataset + Deep Learning Models for Fiber-Reinforced Composites

**URL**: https://acdc.alcf.anl.gov/mdf/detail/badran_deeplearning_supplementarymaterial_v1.1/

**内容**:
- 繊維複合材料のXCTスキャン
- 訓練済み深層学習モデル
- 自動セグメンテーション
- 負荷前後のデータ

**データ形式**:
- 複数のスキャン
- 高解像度（典型的なマイクロCT解像度）
- セグメンテーションラベル付き

**ライセンス**: CC-BY 4.0

**用途**:
- 複合材料の破壊、欠陥、き裂セグメンテーション
- 深層学習モデルの参考

---

### 5. Simulated XCT with Ground Truth Pores (NIST)

**URL**: https://catalog.data.gov/dataset/simulated-x-ray-computed-tomography-xct-and-ground-truth-images-of-cylindrical-sample-with

**内容**:
- 球状気孔を持つ円筒サンプルのシミュレーションXCTボリューム
- セグメンテーション/アルゴリズム評価用のグラウンドトゥルース

**データ形式**:
- 複数のSNR（信号対雑音比）レベル
- 生画像、グラウンドトゥルース、二値画像

**ライセンス**: NIST Open License

**用途**:
- 画像セグメンテーションの評価
- 検出アルゴリズムのベンチマーク
- 合成データの生成方法の参考

---

### 6. NASA X-ray Micro-Tomography (Micro-CT)

**URL**: https://data.nasa.gov/dataset/x-ray-micro-tomography-micro-ct

**内容**:
- 様々な先進材料: 複合材料、隕石、織物、熱保護材など
- シンクロトロンとラボソースで取得

**データ形式**:
- 数百nmからcmスケールまで
- 複数の材料と様々な解像度

**ライセンス**: パブリック（個別に確認が必要）

**用途**:
- 航空宇宙材料
- 多孔質材料
- 複合材料

---

## 📚 材料科学の一般データベース

### 1. Materials Project

**URL**: https://materialsproject.org

**内容**:
- 計算材料特性（主にDFT）
- 無機化合物の大規模データベース
- 構造、形成エネルギー、電子構造など

**アクセス方法**:
- REST API
- AWS OpenData（生データ、パース済み、ビルドデータセット）

**欠陥関連**:
- **pymatgen-analysis-defects**: 点欠陥（空孔、置換など）のフレームワーク
- **Screening of material defects using universal ML interatomic potentials** (Zenodo): ~86,000材料の空孔形成エネルギー

**用途**:
- 材料特性の参照
- 欠陥形成エネルギーの予測
- 材料探索

---

### 2. AFLOW

**URL**: http://aflow.org

**内容**:
- 高スループット第一原理計算材料リポジトリ
- 結晶構造、相図、熱力学、磁性など

**制限**:
- 欠陥データセットは主要な提供物ではない
- 主に無欠陥または合金化構造

**用途**:
- 結晶構造の参照
- 相図の確認

---

### 3. NOMAD

**URL**: https://nomad-lab.eu

**内容**:
- 材料シミュレーションの大規模リポジトリ
- 第一原理計算、分子動力学計算の結果
- メタデータ、入力/出力、軌跡など

**制限**:
- 欠陥中心のデータセットは第一級カテゴリではない
- 個別の計算セットの一部として欠陥が含まれる可能性

**用途**:
- シミュレーションデータの検索
- 計算結果の参照

---

### 4. 欠陥専用データセット

#### Screening of Material Defects using Universal ML Potentials (Zenodo)

**URL**: https://zenodo.org/records/15025795

**内容**:
- 空孔形成エネルギー
- ~86,259材料の空孔データ
- ~8,017の2D層（化学エッチング由来）

**用途**:
- 空孔型欠陥の広範囲なデータ
- MLモデルの訓練

---

#### Dataset of Double Defects in 2D Materials

**URL**: https://zenodo.org/records/15806884

**内容**:
- 2D材料の二重欠陥（欠落原子）のDFT緩和
- MoS₂, WSe₂など

**データ形式**:
- 数千の二重欠陥構造

**用途**:
- 2D材料の欠陥研究
- 欠陥間相互作用の理解

---

#### 2DMD (2D Materials Defect Dataset)

**URL**: https://airi.net.cdn.cloudflare.net/

**内容**:
- 2D材料の欠陥熱力学と電子特性
- 欠陥/置換/空孔
- MoS₂, WSe₂, hBN, GaSe, InSe, 黒リン

**用途**:
- 2D材料の点欠陥研究

---

#### QPOD (Quantum Point Defect in 2D materials database)

**URL**: arXiv:2110.01961

**内容**:
- >1900欠陥システム
- 82の2D絶縁体における503の固有点欠陥
- 形成エネルギー、電荷遷移レベルなど

**用途**:
- 2D材料の点欠陥の包括的研究

---

## 💡 データセットの活用戦略

### 戦略1: 転移学習（推奨）⭐⭐⭐⭐⭐

1. **事前訓練**: CFRP欠陥検出データセットで訓練
2. **ファインチューニング**: NiTiデータ（シミュレーションまたは限定的な実データ）で訓練
3. **利点**: 限られたNiTiデータでも高精度を達成可能

### 戦略2: シミュレーションデータの生成

1. **FEAシミュレーション**: NiTiの材料特性を使用
2. **相変態モデル**: Brinsonモデルなどを実装
3. **欠陥パターン**: 様々な欠陥をシミュレーション
4. **利点**: 大量のラベル付きデータを生成可能

### 戦略3: ハイブリッドアプローチ（最推奨）⭐⭐⭐⭐⭐

1. **シミュレーションデータ**: 大量の訓練データを生成
2. **転移学習**: CFRPデータで事前訓練
3. **実データ**: 限定的なNiTi実データでファインチューニング
4. **利点**: 実用性と汎用性の両立

---

## 🎯 推奨データセット（優先順位）

### 最優先（すぐに使える）

1. **CFRP Thermal Inspection Dataset** (Mendeley)
   - 欠陥セグメンテーション用
   - 転移学習の事前訓練に最適

2. **CFRP Surface & Internal Damage Dataset** (Mendeley)
   - 内部欠陥データ
   - 3D欠陥位置特定に有用

3. **Image-based Data on Strain Fields with Porosity** (Mendeley)
   - 欠陥とひずみ場の関係
   - 微細組織認識に有用

### 次優先（参考・補助用）

4. **NASA SMMD**
   - NiTiの基本特性の確認

5. **Mendeley NiTi Alloys Dataset**
   - NiTiの組成と特性の関係

6. **NIST CoCr AM XCT Data**
   - 3D内部構造の参考

### 長期（研究発展用）

7. **Materials Project**
   - 材料特性の参照
   - 欠陥形成エネルギーの予測

8. **NIST Interatomic Potentials**
   - 原子スケールシミュレーション

---

## ⚠️ 注意事項

### データセットの制限

1. **NiTi専用の欠陥検出データセットは少ない**
   - 直接的なデータセットは限定的
   - 転移学習やシミュレーションデータの活用が必要

2. **データの質と量**
   - 実データは限られる
   - ラベル付けが困難
   - シミュレーションデータとの差異

3. **スケールの違い**
   - 欠陥は複数のスケールにまたがる（原子、ナノ、マイクロ）
   - 適切な解像度の選択が必要

### 推奨アプローチ

1. **シミュレーションデータを主に使用**
   - FEAで大量のデータを生成
   - 様々な欠陥パターンをシミュレーション

2. **転移学習を活用**
   - CFRPデータで事前訓練
   - NiTiデータでファインチューニング

3. **実データで検証**
   - 限定的な実データで最終検証
   - シミュレーションと実データの差異を評価

---

## 🔗 データセットへのアクセス方法

### 一般的な手順

1. **登録**: 多くのデータベースは無料登録が必要
2. **ダウンロード**: 直接ダウンロードまたはAPI経由
3. **引用**: 使用時は適切に引用
4. **ライセンス確認**: 使用目的に応じたライセンスの確認

### 主要なデータリポジトリ

- **Mendeley Data**: https://data.mendeley.com
- **Zenodo**: https://zenodo.org
- **4TU.ResearchData**: https://data.4tu.nl
- **NIST Materials Data Repository**: https://materialsdata.nist.gov
- **NASA Data Portal**: https://data.nasa.gov
- **Materials Project**: https://materialsproject.org

---

## 📊 データセット比較表

| データセット | NiTi専用 | 欠陥検出 | データ量 | 転移学習 | 推奨度 |
|------------|---------|---------|---------|---------|--------|
| **CFRP Thermal Inspection** | ❌ | ✅ | 大 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **CFRP Surface & Internal** | ❌ | ✅ | 中 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Strain Fields with Porosity** | ❌ | ✅ | 大 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **NASA SMMD** | ✅ | ❌ | 中 | ⭐⭐ | ⭐⭐⭐ |
| **Mendeley NiTi** | ✅ | ❌ | 小 | ⭐⭐ | ⭐⭐⭐ |
| **NIST CoCr XCT** | ❌ | ✅ | 大 | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Materials Project** | ⚠️ | ⚠️ | 超大 | ⭐⭐ | ⭐⭐⭐ |

---

## 🚀 次のステップ

1. **CFRPデータセットのダウンロード**
   - Thermal Inspection Dataset
   - Surface & Internal Damage Dataset

2. **転移学習モデルの構築**
   - CFRPデータで事前訓練
   - NiTiデータでファインチューニング

3. **シミュレーションデータの生成**
   - FEAでNiTiデータを生成
   - 様々な欠陥パターンをシミュレーション

4. **実データの収集**（可能であれば）
   - XCT、SEM、EBSDデータ
   - 実際のNiTi試料

---

**最終更新**: 2026年1月20日
