## Mapping of Existing Documents to Final English Report

This note maps sections of the planned English full report to existing documentation in `data_collection/`.

### 0. Global references
- **Project-level summaries**: `FINAL_REPORT.md`, `COMPLETE_SUMMARY.md`
- **Dataset construction and statistics**: `DATASET_DOCUMENTATION.md`, `DATASET_DOCUMENTATION.tex`
- **Modeling and optimization results**: `結果レポート_最終版.md`, `FINAL_RESULTS.md`, `MODEL_TRAINING_SUMMARY.md`, `VISUALIZATION_SUMMARY.md`, `FINAL_OPTIMIZATION_REPORT.md`
- **Data source details and quality**: `DATASET_SOURCES_ANALYSIS.md`, `DATASET_SIZE_ANALYSIS.md`, `データセット出典レポート.md`, `最終データセット出典レポート.md`, `データセット検証レポート.md`

---

### Title & Abstract
- **Main content source**:
  - High-level summary of goals and achievements from:
    - `FINAL_REPORT.md` → project overview, key numbers (322 samples, 29 features, Random Forest baseline)
    - `結果レポート_最終版.md` → optimized results (SVR R² ≈ 0.59, Gradient Boosting R² ≈ 0.67 for experimental data)
  - Dataset scale and mixed experimental/computed nature from:
    - `DATASET_DOCUMENTATION.md` → 5,339 final samples, source breakdown

### 1. Introduction
- **Motivation and biomedical context**:
  - `FINAL_REPORT.md` / `結果レポート_最終版.md` → description of elastic modulus mismatch and goal of 30–90 GPa range.
- **Existing work and databases**:
  - `DATASET_DOCUMENTATION.md` → description and citations for DOE/OSTI, Gorsse, Materials Project, MPEA nano-indentation, DISMA, etc.
- **Project objectives and scope**:
  - `FINAL_REPORT.md` / `COMPLETE_SUMMARY.md` → overall project goals and completion status.

### 2. Data and Dataset Construction
- **2.1 Data Sources**:
  - `DATASET_DOCUMENTATION.md` / `.tex` → detailed per-source description and references.
  - `データセット出典レポート.md`, `最終データセット出典レポート.md` → Japanese summaries of data origins.
- **2.2 Data Integration Pipeline**:
  - `DATA_COLLECTION_FINAL_REPORT.md`, `DATA_COLLECTION_SUCCESS_REPORT.md` → description of integration scripts and workflow.
  - `PREPROCESSING_TRAINING_REPORT.md`, `EXECUTION_SUMMARY.md` → practical notes on processing and training.
- **2.3 Final Dataset Characteristics**:
  - `DATASET_DOCUMENTATION.md` → 5,339-sample integrated dataset statistics.
  - `FINAL_REPORT.md`, `結果レポート_最終版.md` → 322-sample experimental subset statistics (27–466 GPa, mean ≈ 160–165 GPa, 30 samples in 30–90 GPa range).

### 3. Features and Machine Learning Methods
- **3.1 Feature Engineering**:
  - `FINAL_REPORT.md` → list of 29 features, including atomic radius/electronegativity descriptors, VEC, entropy, density, and 17 composition features.
  - `FEATURES.md` in `gnn_transformer_models/` (for cross-references if needed).
- **3.2 Models**:
  - `MODEL_TRAINING_SUMMARY.md`, `結果レポート_最終版.md` → list of models (Linear, Ridge, Lasso, Polynomial, KNN, Random Forest, SVR, MLFFNN, Gradient Boosting, Stacking).
- **3.3 Training/Evaluation Protocol**:
  - `結果レポート_最終版.md` / `MODEL_TRAINING_SUMMARY.md` → train/test split (80/20), cross-validation, metrics (R², RMSE, MAE) and optimization strategies.

### 4. Results
- **4.1 Overall Model Performance**:
  - `結果レポート_最終版.md` → tables with baseline and optimized performance (SVR R² ≈ 0.59 on full 322-sample dataset; Gradient Boosting and stacking for experimental-only subset).
  - `FINAL_RESULTS.md`, `FINAL_OPTIMIZATION_REPORT.md` → additional result summaries where needed.
- **4.2 Feature Importance and Interpretation**:
  - `FINAL_REPORT.md` → Random Forest feature importance, especially Ti content, electronegativity, VEC, density.
  - `VISUALIZATION_SUMMARY.md` → qualitative commentary on feature-importance plots.
- **4.3 Error and Residual Analysis**:
  - `VISUALIZATION_SUMMARY.md` → residual plots, predicted vs. actual plots.
  - Figures in `figures/` → referenced but not embedded.
- **4.4 Case Studies**:
  - `結果レポート_最終版.md` / `VISUALIZATION_SUMMARY.md` → candidate alloys in 30–90 GPa range for biomedical applications.

### 5. Discussion
- **Experimental vs computed data and bias**:
  - `DATASET_DOCUMENTATION.md` → breakdown of experimental vs computed, limitations.
  - `結果レポート_最終版.md` → performance differences between full dataset and experimental-only subset.
- **Data and model limitations**:
  - `DATASET_DOCUMENTATION.md` → comments on temperature, overestimation of DFT moduli, auxiliary datasets.
  - `FINAL_REPORT.md`, `MODEL_TRAINING_SUMMARY.md` → notes on overfitting and model weaknesses.

### 6. Conclusion and Future Work
- **Main achievements**:
  - `FINAL_REPORT.md`, `COMPLETE_SUMMARY.md`, `結果レポート_最終版.md` → lists of completed tasks and headline performance.
- **Future extensions**:
  - `結果レポート_最終版.md`, `FINAL_REPORT.md` → short- and long-term improvement ideas (more data, feature expansion, deep learning, active learning).

### 7. References
- **Primary reference list**:
  - `DATASET_DOCUMENTATION.md` / `.tex` → canonical references for all datasets and Materials Project.
  - `結果レポート_最終版.md` → additional citations if needed.

---

### Japanese Summary Report (PROJECT_SUMMARY_JA.md)
- **1. プロジェクト概要**:
  - `FINAL_REPORT.md`, `COMPLETE_SUMMARY.md` → overall goals and completion summary.
- **2. データと手法**:
  - `DATASET_DOCUMENTATION.md` (for numbers) + Japanese context from `データセット出典レポート.md`, `データセット検証レポート.md`.
- **3. 主な結果**:
  - `結果レポート_最終版.md` → key R² / RMSE values and model ranking in Japanese.
- **4. 考察と今後の課題**:
  - Improvement and future work sections from `結果レポート_最終版.md` / `FINAL_REPORT.md`.

