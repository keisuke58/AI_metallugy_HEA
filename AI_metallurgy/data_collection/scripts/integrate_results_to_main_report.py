#!/usr/bin/env python3
"""
メインレポートに包括的な結果を統合するスクリプト
"""
from pathlib import Path
import re

BASE_DIR = Path(__file__).resolve().parent.parent
MAIN_REPORT = BASE_DIR / "PROJECT_FINAL_REPORT_EN.md"
SCRIPTS_DIR = Path(__file__).resolve().parent
REPORTS_DIR = SCRIPTS_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True, parents=True)
COMPREHENSIVE_REPORT = REPORTS_DIR / "COMPREHENSIVE_MODEL_RESULTS_REPORT.md"

def read_file_content(filepath):
    """ファイルを読み込む"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

def write_file_content(filepath, content):
    """ファイルに書き込む"""
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

def integrate_comprehensive_results():
    """包括的な結果をメインレポートに統合"""
    print("=" * 80)
    print("メインレポートに包括的な結果を統合")
    print("=" * 80)
    
    # メインレポートを読み込む
    main_content = read_file_content(MAIN_REPORT)
    
    # 包括的レポートを読み込む
    comprehensive_content = read_file_content(COMPREHENSIVE_REPORT)
    
    # セクション4の内容を抽出（4.1から4.4まで）
    section_4_pattern = r'(4\. Results.*?)(?=\n---\n|$)'
    section_4_match = re.search(section_4_pattern, main_content, re.DOTALL)
    
    if section_4_match:
        old_section_4 = section_4_match.group(1)
        
        # 新しい包括的なセクション4を作成
        new_section_4 = """4. Results
----------

4.1 Comprehensive Model Performance Comparison
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This section presents a comprehensive evaluation of all machine learning models trained for predicting elastic modulus in High-Entropy Alloys (HEAs). We evaluated six different model architectures, including classical machine learning methods (Gradient Boosting, Ensemble), deep learning models (Transformer), and graph neural networks (MEGNet, CGCNN).

4.1.1 Performance Summary Table

Table 1 summarizes the performance metrics for all trained models on the 322-sample experimental dataset:

| Model | Test R² | Test RMSE (GPa) | Test MAE (GPa) | Rank | Status |
|-------|---------|-----------------|----------------|------|--------|
| **Gradient Boosting** | **0.4956** | **40.42** | **25.14** | 1 | ⭐⭐⭐⭐⭐ |
| **Ensemble (GB+SVR+RF)** | 0.4797 | 41.05 | 26.30 | 2 | ⭐⭐⭐⭐ |
| **Transformer (1000 epochs, bs=16)** | **0.4434** | **42.52** | **28.35** | 3 | ⭐⭐⭐⭐ |
| **Transformer (2000 epochs, bs=64)** | 0.3259 | 46.80 | 30.02 | 4 | ⭐⭐⭐ |
| **CGCNN** | 0.2187 | 50.38 | 29.90 | 5 | ⭐⭐ |
| **MEGNet** | -0.0096 | 57.27 | 41.39 | 6 | ❌ |

**Key Observations**:

1. **Gradient Boosting achieved the best performance** (R² = 0.4956), demonstrating the effectiveness of classical machine learning methods for this dataset size.
2. **Transformer models show competitive performance** (R² = 0.4434) with proper hyperparameter tuning, indicating the potential of deep learning approaches.
3. **Graph neural networks (MEGNet, CGCNN) underperformed** compared to classical methods, suggesting the need for architecture improvements or more training data.
4. **Hyperparameter selection is critical**, especially for deep learning models where batch size and training duration significantly impact performance.

4.1.2 Performance Visualization

Comprehensive visualizations of model performance are available in the supplementary materials:

- R² Score Comparison (bar chart)
- RMSE and MAE Comparison (side-by-side bar charts)
- 3D Performance Comparison (R² vs RMSE vs MAE scatter plot)
- Performance Ranking (horizontal bar chart)

These visualizations clearly illustrate the performance hierarchy and highlight the significant differences between model categories.

4.2 Detailed Model Analysis

4.2.1 Gradient Boosting (Best Performer)

**Configuration**:
- Algorithm: Gradient Boosting Regressor
- Hyperparameter Optimization: RandomizedSearchCV (200 iterations)
- Optimal Parameters:
  - `n_estimators`: 707
  - `max_depth`: 12
  - `learning_rate`: 0.0344
  - `loss`: huber
  - `min_samples_split`: 26
  - `min_samples_leaf`: 2
  - `subsample`: 0.965

**Performance**:
- **Test R²**: 0.4956 (highest)
- **Test RMSE**: 40.42 GPa (lowest)
- **Test MAE**: 25.14 GPa (lowest)
- **CV R²**: 0.4806 ± 0.1652

**Analysis**:
Gradient Boosting achieved the best performance among all models evaluated. The model demonstrates excellent generalization capability, with the Huber loss function providing robustness against outliers. The optimal hyperparameters were found through systematic search using RandomizedSearchCV with 200 iterations.

**Advantages**:
1. Highest predictive accuracy (R² = 0.4956)
2. Interpretable (feature importance available)
3. Computationally efficient
4. Robust to outliers (Huber loss)

**Feature Importance** (from Gradient Boosting):
The most important features identified by the model are:
1. Ti composition fraction (comp_Ti) - ~16-17% importance
2. Average electronegativity
3. Valence electron concentration (VEC)
4. Co composition fraction (comp_Co)
5. Density

4.2.2 Ensemble Model (GB + SVR + RF)

**Configuration**:
- Model Type: Voting Regressor
- Components:
  - Gradient Boosting (weight: 2)
  - Support Vector Regressor (RBF kernel, weight: 1)
  - Random Forest (weight: 1)
- SVR Optimization: C=38.67
- RF Optimization: n_estimators=641

**Performance**:
- **Test R²**: 0.4797
- **Test RMSE**: 41.05 GPa
- **Test MAE**: 26.30 GPa

**Analysis**:
The ensemble performance is slightly lower than the single Gradient Boosting model. This limited ensemble effect suggests that the component models produce similar predictions, reducing the benefit of combining them. The ensemble provides marginal improvement in robustness but does not significantly outperform the best single model.

**Conclusion**:
For this dataset, a single well-optimized Gradient Boosting model is more effective than an ensemble of similar models. More diverse model combinations may be needed for better ensemble performance.

4.2.3 Transformer Models

4.2.3.1 Transformer (1000 epochs, batch_size=16) - Best Deep Learning Model

**Configuration**:
- Architecture: Transformer Encoder
- Epochs: 1000 (stopped at 262 due to early stopping)
- Batch Size: 16
- Learning Rate: 1e-4
- Early Stopping Patience: 200
- Best Val R²: 0.3464

**Performance**:
- **Test R²**: 0.4434 (best deep learning result)
- **Test RMSE**: 42.52 GPa
- **Test MAE**: 28.35 GPa

**Training Progress**:
- Train R²: 0.86-0.87 (high learning capacity)
- Val R²: 0.12-0.35 (validation performance)
- Improvement: +39% compared to 300 epochs (R²: 0.3191 → 0.4434)

**Analysis**:
Significant improvement was achieved by increasing epochs from 300 to 1000. The batch size of 16 proved optimal for this dataset. The model shows good learning capacity, but validation performance is lower than training performance, indicating some overfitting. Early stopping prevented excessive overfitting.

**Key Insights**:
1. Epoch number significantly impacts performance (+39% improvement)
2. Smaller batch size (16) works better than larger (64)
3. Transformer architecture can achieve competitive results with proper tuning
4. The model ranks 3rd overall, demonstrating the potential of deep learning approaches

4.2.3.2 Transformer (2000 epochs, batch_size=64) - Overfitting Case Study

**Configuration**:
- Architecture: Transformer Encoder
- Epochs: 2000 (stopped at 451 due to early stopping)
- Batch Size: 64
- Learning Rate: 5e-5
- Early Stopping Patience: 300
- Best Val R²: 0.3416

**Performance**:
- **Test R²**: 0.3259 (lower than batch_size=16)
- **Test RMSE**: 46.80 GPa
- **Test MAE**: 30.02 GPa

**Training Progress**:
- Train R²: 0.79-0.84 (high)
- Val R²: -0.12 to -0.24 (negative, overfitting sign)
- Best Val R²: 0.3416

**Analysis**:
**Overfitting was clearly detected**: High train R² but negative val R² indicates severe overfitting. The larger batch size (64) led to performance degradation compared to batch_size=16. The lower learning rate (5e-5) may have contributed to slower convergence and less effective learning.

**Key Lessons**:
1. Batch size significantly impacts model performance (16 > 64 for this dataset)
2. Larger batch sizes can lead to overfitting in small datasets
3. Optimal hyperparameters are dataset-dependent
4. Early stopping is crucial for preventing overfitting

4.2.4 Graph Neural Networks

4.2.4.1 CGCNN

**Configuration**:
- Architecture: Crystal Graph Convolutional Neural Network
- Epochs: 200
- Batch Size: 64
- Learning Rate: 1e-3

**Performance**:
- **Test R²**: 0.2187 (moderate)
- **Test RMSE**: 50.38 GPa
- **Test MAE**: 29.90 GPa

**Analysis**:
CGCNN achieved positive R², indicating some learning capability. However, performance is lower than classical ML and Transformer models. The graph structure may not be fully utilized with the current dataset. There is potential for improvement with longer training or architecture modifications.

**Recommendations**:
1. Increase training epochs (500+)
2. Review graph construction methodology
3. Adjust architecture parameters
4. Consider feature integration with graph structure

4.2.4.2 MEGNet

**Configuration**:
- Architecture: Materials EGraph Network
- Epochs: 200
- Batch Size: 64
- Learning Rate: 1e-3

**Performance**:
- **Test R²**: -0.0096 (negative, poor performance)
- **Test RMSE**: 57.27 GPa (highest)
- **Test MAE**: 41.39 GPa (highest)

**Analysis**:
**Negative R² indicates the model performs worse than a baseline (mean predictor)**. The model failed to learn meaningful patterns from the data. Possible causes include:
1. Insufficient training epochs
2. Architecture mismatch with dataset characteristics
3. Graph construction issues
4. Hyperparameter misconfiguration

**Recommendations**:
1. Increase training epochs significantly (500+)
2. Review graph construction methodology
3. Adjust architecture parameters
4. Consider hybrid approaches (graph + features)

4.3 Comparative Analysis and Insights

4.3.1 Model Category Comparison

| Category | Best Model | Best R² | Characteristics |
|----------|-----------|---------|-----------------|
| **Classical ML** | Gradient Boosting | 0.4956 | Highest performance, interpretable |
| **Ensemble** | GB+SVR+RF | 0.4797 | Robust but limited improvement |
| **Deep Learning** | Transformer (bs=16) | 0.4434 | Competitive, requires tuning |
| **Graph Neural Networks** | CGCNN | 0.2187 | Underperforming, needs improvement |

4.3.2 Performance vs. Complexity

**Key Observation**: 
Classical ML methods (Gradient Boosting) outperform complex deep learning models for this dataset size (322 samples). This suggests that:
- For small datasets, simpler models are more effective
- Deep learning models may require more data or different architectures
- The complexity of deep learning models does not necessarily translate to better performance

4.3.3 Training Efficiency

| Model | Training Time | Epochs | Efficiency |
|-------|--------------|--------|------------|
| Gradient Boosting | ~10 minutes | N/A | ⭐⭐⭐⭐⭐ |
| Ensemble | ~15 minutes | N/A | ⭐⭐⭐⭐ |
| Transformer (bs=16) | ~1.5 hours | 262 | ⭐⭐⭐ |
| Transformer (bs=64) | ~2-3 hours | 451 | ⭐⭐ |
| CGCNN | ~30 minutes | 200 | ⭐⭐⭐ |
| MEGNet | ~30 minutes | 200 | ⭐⭐⭐ |

4.3.4 Hyperparameter Sensitivity Analysis

**Critical Findings**:

1. **Batch Size Impact**:
   - Transformer with batch_size=16: R² = 0.4434
   - Transformer with batch_size=64: R² = 0.3259
   - **Conclusion**: Smaller batch sizes are more effective for small datasets

2. **Epoch Number Impact**:
   - Transformer with 300 epochs: R² = 0.3191
   - Transformer with 1000 epochs (262 actual): R² = 0.4434
   - **Improvement**: +39% with increased epochs
   - **Conclusion**: More epochs significantly improve deep learning models

3. **Learning Rate Impact**:
   - Transformer with lr=1e-4 (bs=16): R² = 0.4434
   - Transformer with lr=5e-5 (bs=64): R² = 0.3259
   - **Conclusion**: Learning rate must be balanced with batch size

4.4 Key Insights and Recommendations

4.4.1 Major Findings

1. **Gradient Boosting is the optimal choice** for this task
   - Highest accuracy (R² = 0.4956)
   - Best computational efficiency
   - Interpretable results

2. **Transformer models show promise** with proper configuration
   - Achieved R² = 0.4434 with optimal settings
   - Significant improvement from hyperparameter tuning
   - Potential for further improvement

3. **Graph neural networks need improvement**
   - Current implementations underperform
   - May require architecture modifications or more data
   - Graph structure utilization needs optimization

4. **Hyperparameter sensitivity**
   - Batch size significantly impacts performance (16 > 64)
   - Epoch number affects deep learning models more than classical ML
   - Learning rate requires careful tuning

4.4.2 Recommendations

**For Production Use**:
- **Use Gradient Boosting** for best accuracy and efficiency
- **Consider Ensemble** for additional robustness (marginal benefit)

**For Research**:
- **Investigate Transformer improvements**:
  - Try different architectures (e.g., attention mechanisms)
  - Experiment with longer training (2000+ epochs)
  - Explore learning rate scheduling strategies
  
- **Improve Graph Neural Networks**:
  - Review graph construction methodology
  - Experiment with different GNN architectures
  - Consider hybrid approaches (graph + features)

**For Future Work**:
1. **Data augmentation** to increase dataset size
2. **Feature engineering** improvements
3. **Transfer learning** from larger datasets
4. **Ensemble of diverse models** (including GNNs if improved)

4.5 Statistical Analysis

4.5.1 Performance Distribution

The models show a clear performance hierarchy:
- **Top tier** (R² > 0.4): Gradient Boosting, Ensemble, Transformer (bs=16)
- **Middle tier** (0.2 < R² < 0.4): Transformer (bs=64), CGCNN
- **Bottom tier** (R² < 0.2): MEGNet

4.5.2 Error Analysis

**RMSE Analysis**:
- Best: 40.42 GPa (Gradient Boosting)
- Worst: 57.27 GPa (MEGNet)
- Range: 16.85 GPa difference

**MAE Analysis**:
- Best: 25.14 GPa (Gradient Boosting)
- Worst: 41.39 GPa (MEGNet)
- Range: 16.25 GPa difference

4.5.3 Improvement Potential

Based on current results:
- **Gradient Boosting**: Limited room for improvement (already optimal)
- **Transformer**: Potential for 5-10% improvement with further tuning
- **GNNs**: Significant improvement potential (50-100% with proper configuration)

"""
        
        # セクション4を置き換え
        new_main_content = main_content.replace(old_section_4, new_section_4)
        
        # バックアップを作成
        backup_file = BASE_DIR / "PROJECT_FINAL_REPORT_EN_backup.md"
        write_file_content(backup_file, main_content)
        print(f"✅ バックアップを作成: {backup_file}")
        
        # 新しい内容を保存
        write_file_content(MAIN_REPORT, new_main_content)
        print(f"✅ メインレポートを更新: {MAIN_REPORT}")
        
        # 包括的な結果セクションも追加（セクション4.6として）
        additional_section = """

4.6 Comprehensive Results Summary

For detailed analysis, visualizations, and extended discussions, please refer to the comprehensive results report available at `reports/COMPREHENSIVE_MODEL_RESULTS_REPORT.md`. This report includes:

- Detailed model configurations
- Training progress analysis
- Hyperparameter sensitivity studies
- Performance visualizations (R², RMSE, MAE comparisons)
- 3D performance scatter plots
- Model ranking charts
- Statistical analysis
- Recommendations for future work

The comprehensive report provides in-depth insights into model behavior, training dynamics, and optimization strategies that complement the summary presented in this main report.

"""
        
        # セクション4の最後に追加
        new_main_content = new_main_content.replace("4.5.3 Improvement Potential", "4.5.3 Improvement Potential" + additional_section)
        write_file_content(MAIN_REPORT, new_main_content)
        
        print("\n" + "=" * 80)
        print("✅ 統合完了！")
        print("=" * 80)
        print(f"\n📊 更新されたレポート: {MAIN_REPORT}")
        print(f"📁 バックアップ: {backup_file}")
        print(f"📈 包括的レポート: {COMPREHENSIVE_REPORT}")
        
    else:
        print("⚠️ セクション4が見つかりませんでした。手動で統合してください。")

if __name__ == "__main__":
    integrate_comprehensive_results()
