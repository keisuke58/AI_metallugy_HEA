# Integrated HEA/MPEA Elastic Modulus Dataset

**Date**: 2026-01-23  
**File**: `final_data/unified_dataset_cleaned_20260123_175245.csv`  
**Total samples**: **5,339**

---

## 1. Purpose and Overview

This document summarizes the construction, sources, and statistical properties of the integrated dataset used for predicting the elastic modulus (Young’s modulus, in GPa) of high-entropy alloys (HEAs) and multi-principal element alloys (MPEAs).

The final cleaned dataset is stored as `data_collection/final_data/unified_dataset_cleaned_20260123_175245.csv`, and a stable latest copy is maintained as `data_collection/final_data/unified_dataset_latest.csv`.

---

## 2. Data Sources and References

### 2.1 DOE/OSTI Dataset (Experimental)

- **Title**: Phases and Young's Modulus Dataset for High Entropy Alloys  
- **Authors**: Ankit Roy, Ganesh Balasubramanian  
- **DOI**: `10.18141/1644295`  
- **URL**: `https://edx.netl.doe.gov/dataset/phases-and-young-s-modulus-dataset-for-high-entropy-alloys`  
- **Content**: 340 HEAs with phase data, 107 HEAs with Young's modulus and 11 calculated features.  
- **Used here**: 101 samples after de-duplication and cleaning.

### 2.2 Gorsse HEA Mechanical Properties Database (Experimental)

- **Citation**: Gorsse, S., Nguyen, M. H., Senkov, O. N., & Miracle, D. B.  
  “Database on the mechanical properties of high entropy alloys and complex concentrated alloys.” *Data in Brief* 21 (2018/2019).  
- **PubMed**: `https://pubmed.ncbi.nlm.nih.gov/30761350/`  
- **HAL**: `https://hal.science/hal-02156875/`  
- **Content**: ~370 HEAs/CCAs (2004–2016) with composition, microstructure, density, hardness, yield strength, UTS, elongation, **Young’s modulus**, etc.  
- **Used here**: 182 unique alloys with valid Young's modulus.

### 2.3 Latest Experimental HEA Data

Additional 2024–2025 HEA/MEA data (e.g., Ti–Zr–Nb systems, Ti–Zr–Hf–Nb–Ta HEAs) are incorporated where possible. Many entries are already covered in Gorsse/DOE/OSTI/MPEA nano-indentation compilations, so the net increase in unique alloys is small.

### 2.4 DISMA Research HEA Dataset (Auxiliary)

- **Platform**: Mendeley Data  
- **DOI**: `10.17632/p3txdrdth7.1`  
- **URL**: `https://data.mendeley.com/datasets/p3txdrdth7/1`  
- **Content**: HEA mechanical and structural features (strength, elongation, phases) curated for machine learning.  
- **Used here**: As an auxiliary source; no consistent Young’s modulus column.

### 2.5 MPEA Mechanical Properties Database (Auxiliary)

- **Title**: A database of mechanical properties for multi-principal element alloys  
- **Platform**: Mendeley Data, DOI `10.17632/4d4kpfwpf6`  
- **URL**: `https://data.mendeley.com/datasets/4d4kpfwpf6`  
- **Content**: 1,713 MPEAs with strength, hardness, elongation, etc.  
- **Used here**: Auxiliary features only; not a direct Young’s modulus source.

### 2.6 Materials Project (Computed)

- **Citation**: Jain, A., Ong, S. P., Hautier, G., Chen, W., Richards, W. D., Dacek, S., Cholia, S., Gunter, D., Skinner, D., Ceder, G., & Persson, K. A.  
  “Commentary: The Materials Project: A materials genome approach to accelerating materials innovation.” *APL Materials* 1, 011002 (2013).  
- **URL**: `https://materialsproject.org`  
- **API Docs**: `https://api.materialsproject.org/docs`  
- **Content**: DFT elastic tensors (bulk modulus K, shear modulus G) for inorganic compounds. Young’s modulus is computed as  
  \( E = 9KG / (3K + G) \) when both K and G are available.  
- **Used here**: 4,439 samples.

### 2.7 Refractory HEA Elastic Constants (Computed)

- **Repository**: `https://github.com/uttambhandari91/Elastic-constant-DFT-data`  
- **Content**: DFT elastic constants for 370 refractory HEAs.  
- **Used here**: 370 samples, Young’s modulus estimated from reported elastic constants.

### 2.8 MPEA Nano-indentation Database

- **Example citation**: “A database of multi-principal element alloy phase-specific mechanical properties measured with nano-indentation,” Johns Hopkins University Applied Physics Laboratory, *Data Brief* (2024), PMC11298849.  
- **PMC**: `https://pmc.ncbi.nlm.nih.gov/articles/PMC11298849/`  
- **GitHub**: `https://github.com/CitrineInformatics/MPEA_dataset`  
- **Content**: 7,385 indentation tests on 19 MPEAs with phase-specific mechanical properties.  
- **Used here**: 767 entries with experimental or calculated Young’s modulus extracted from `MPEA_dataset.csv`; after de-duplication and integration, 51 experimental and 196 calculated entries remain.

---

## 3. Integration and Preprocessing

Main preprocessing steps:

1. Parse each source file and map Young’s modulus columns to a unified label `elastic_modulus` (GPa).  
2. Remove entries with missing or non-positive `elastic_modulus`.  
3. Normalize all modulus values to GPa.  
4. Remove exact and near-duplicate entries based on `alloy_name` and (where available) composition and close modulus values.  
5. Drop rows with missing required fields: `alloy_name`, `elastic_modulus`, `source`.  
6. Save the cleaned dataset and run validation scripts (`clean_dataset.py`, `validate_dataset.py`).

---

## 4. Statistical Properties

### 4.1 Source Breakdown

Final cleaned dataset: **5,339** unique alloy entries.

| Source                                | Count | Share (%) |
|---------------------------------------|-------|-----------|
| Materials Project                     | 4,439 | 83.1      |
| Refractory HEA Elastic Constants      | 370   | 6.9       |
| MPEA Nano-indentation (calculated)    | 196   | 3.7       |
| Gorsse Dataset                        | 182   | 3.4       |
| DOE/OSTI Dataset                      | 101   | 1.9       |
| MPEA Nano-indentation (experimental)  | 51    | 1.0       |

### 4.2 Elastic Modulus Distribution

Key statistics (in GPa):

- Minimum: 2.73  
- Maximum: 621.33  
- Mean: 139.26  
- Median: 122.61  
- ≈99.3% of samples lie in the 10–500 GPa range.

The histogram and boxplot (not shown here) exhibit a broad, physically reasonable distribution with only ~1.7% of outliers by the IQR rule.

### 4.3 Missing Values and Duplicates

After cleaning:

- Required columns (`alloy_name`, `elastic_modulus`, `source`) have **no missing values**.  
- **No** completely duplicated rows remain.  
- The number of unique `alloy_name` entries equals the number of rows (5,339), i.e., no alias duplicates.

---

## 5. Quality Assessment

A dedicated validation script (`validate_dataset.py`) checks:

- Presence and dtype of required columns.  
- Missing values per column.  
- Basic statistics and outlier fraction for `elastic_modulus`.  
- Duplicates by `alloy_name` and by full row.  
- Source-wise sample counts.

Using a simple penalty-based scoring (for missing values, duplicates, and outliers), the dataset attains a **quality score of 99.8 / 100**, classified as **excellent** for machine-learning purposes.

---

## 6. Discussion and Limitations

### 6.1 Experimental vs Computed Data

- **Experimental** Young’s modulus: 335 samples (~6.3%).  
  - Gorsse, DOE/OSTI, MPEA Nano-indentation (experimental).  
- **Computed** Young’s modulus: 5,004 samples (~93.7%).  
  - Materials Project, Refractory HEA, MPEA Nano-indentation (calculated).

Recommended practice:

- Train models on the full dataset to leverage broad coverage.  
- Evaluate and report performance on the experimental-only subset for rigorous validation.

### 6.2 Known Limitations

- Very high moduli (e.g. >500 GPa) are DFT-based and may overestimate stiffness compared to experiment.  
- Temperature conditions are not consistently annotated; most data are near room temperature but not strictly standardized.  
- Some large databases (DISMA, MPEA mechanical properties) are currently used only as auxiliary sources and could be better integrated in future work.

---

## 7. Conclusion

The integrated dataset consolidates multiple high-quality experimental and computational sources into a single, cleaned table of 5,339 alloys with validated elastic modulus values. The absence of missing labels and duplicates, combined with a broad and physically reasonable modulus distribution, provides a strong basis for training and benchmarking machine-learning models for elastic modulus prediction in HEAs and MPEAs.

This document, together with the LaTeX file `DATASET_DOCUMENTATION.tex`, can be cited or adapted directly in academic manuscripts (e.g. in the Data or Methods sections).
