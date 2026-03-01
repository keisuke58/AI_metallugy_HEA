# Final Artifacts Snapshot - AI_metallurgy Project

**Date**: 2026-01-23  
**Project**: Machine Learning-based Prediction of Elastic Modulus in HEA/MPEA Alloys

---

## Final Dataset

- **File**: `final_data/unified_dataset_latest.csv`
- **Version**: 2026-01-23
- **Total samples**: 5,339 unique alloys
- **Description**: Integrated and cleaned dataset containing elastic modulus values from multiple experimental and computational sources
- **Key statistics**:
  - Experimental data: 335 samples (≈6.3%)
  - Computed data: 5,004 samples (≈93.7%)
  - Elastic modulus range: 2.73 - 621.33 GPa
  - Mean: 139.26 GPa, Median: 122.61 GPa

---

## Best-Performing Models

### 1. Optimized SVR Model (Best on full 322-sample dataset)

- **Model file**: `models/model_SVR_optimized.pkl`
- **Scaler file**: `models/scaler_SVR_optimized.pkl`
- **Performance**:
  - Test R² ≈ 0.59
  - Test RMSE ≈ 36.2 GPa
  - Test MAE ≈ (see results JSON)
- **Dataset**: Full 322-sample experimental subset
- **Usage**: Suitable for predictions on the full experimental dataset

### 2. Optimized Gradient Boosting Model (Best on experimental-only subset)

- **Model file**: `models/ultimate_final_20260119_200158/best_model.pkl`
- **Additional files**:
  - `models/ultimate_final_20260119_200158/imputer.pkl` (for handling missing values)
- **Performance**:
  - Test R² ≈ 0.67
  - Test RMSE ≈ 49.7 GPa
  - Test MAE ≈ 37.1 GPa
- **Dataset**: Experimental-only subset (≈109 samples)
- **Usage**: Recommended for rigorous validation and interpretation on purely experimental data

---

## Key Figures for Publication

The following figures from `figures/` are recommended for inclusion in manuscripts:

1. **`data_distribution.png`** - Distribution of elastic modulus values in the integrated dataset
2. **`model_performance_comparison.png`** - Comparison of model performance metrics across algorithms
3. **`feature_importance.png`** - Feature importance analysis from Random Forest
4. **`predicted_vs_actual.png`** - Predicted vs. actual elastic modulus scatter plot
5. **`residuals_analysis.png`** - Residual analysis for model validation
6. **`elastic_modulus_comparison.png`** - Comparison of bone vs. implant vs. target modulus ranges (optional, for introduction)

---

## Reproducibility

To reproduce the results:

1. Load the dataset: `final_data/unified_dataset_latest.csv`
2. Run feature engineering scripts in `scripts/` to generate 29 features
3. Load the appropriate model files (SVR or Gradient Boosting)
4. Use the corresponding scaler/imputer files for preprocessing
5. Evaluate on test sets using the same train/test split (80/20) as reported

All preprocessing, feature engineering, and model training scripts are available in the `scripts/` directory.

---

## Notes

- The SVR model is optimized for the full 322-sample dataset and shows good general performance
- The Gradient Boosting model achieves higher accuracy on experimental-only data and is recommended for rigorous validation
- Both models use 29 features including composition descriptors, atomic descriptors, and physical properties
- Feature importance consistently highlights Ti content, electronegativity, VEC, and density as key drivers

---

**Last updated**: 2026-01-23
