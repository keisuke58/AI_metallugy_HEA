#!/usr/bin/env python3
"""
最終統合レポート作成スクリプト
すべての結果をメインレポートに統合し、ビジュアライゼーションと詳細な考察を含む
"""
from pathlib import Path
import re

BASE_DIR = Path(__file__).resolve().parent.parent
MAIN_REPORT = BASE_DIR / "PROJECT_FINAL_REPORT_EN.md"
COMPREHENSIVE_REPORT = BASE_DIR / "scripts" / "reports" / "COMPREHENSIVE_MODEL_RESULTS_REPORT.md"
FIGURES_DIR = BASE_DIR / "scripts" / "reports" / "figures"

def read_file_content(filepath):
    """ファイルを読み込む"""
    if not filepath.exists():
        return None
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

def create_new_section_4():
    """新しいセクション4を作成（すべての結果を含む）"""
    return """4. Results
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

Comprehensive visualizations of model performance are available:

![R² Score Comparison](scripts/reports/figures/r2_comparison.png)

*Figure 1: Comparison of Test R² scores across all models. Green bars indicate R² > 0.4 (good performance), yellow bars indicate 0 < R² < 0.4 (moderate performance), and red bars indicate R² < 0 (poor performance).*

![RMSE and MAE Comparison](scripts/reports/figures/rmse_mae_comparison.png)

*Figure 2: Comparison of Test RMSE (left) and Test MAE (right) across all models. Lower values indicate better performance.*

![3D Performance Comparison](scripts/reports/figures/3d_performance_comparison.png)

*Figure 3: Three-dimensional scatter plot showing the relationship between R², RMSE, and MAE for all models. Models closer to the origin (high R², low RMSE/MAE) represent better performance.*

![Performance Ranking](scripts/reports/figures/performance_ranking.png)

*Figure 4: Horizontal bar chart ranking all models by Test R² score. Models are sorted from best (top) to worst (bottom).*

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
  - `max_features`: None

**Performance**:
- **Test R²**: 0.4956 (highest among all models)
- **Test RMSE**: 40.42 GPa (lowest)
- **Test MAE**: 25.14 GPa (lowest)
- **CV R²**: 0.4806 ± 0.1652

**Analysis**:
Gradient Boosting achieved the best performance among all models evaluated. The model demonstrates excellent generalization capability, with the Huber loss function providing robustness against outliers. The optimal hyperparameters were found through systematic search using RandomizedSearchCV with 200 iterations.

**Advantages**:
1. Highest predictive accuracy (R² = 0.4956)
2. Interpretable (feature importance available)
3. Computationally efficient (~10 minutes training time)
4. Robust to outliers (Huber loss)

**Feature Importance** (from Gradient Boosting):
The most important features identified by the model are:
1. Ti composition fraction (comp_Ti) - ~16-17% importance
2. Average electronegativity
3. Valence electron concentration (VEC)
4. Co composition fraction (comp_Co)
5. Density

These findings are consistent with physical understanding of elastic modulus in metallic alloys, where Ti content, electronic structure (electronegativity, VEC), and density play crucial roles.

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
The ensemble performance is slightly lower than the single Gradient Boosting model (R² = 0.4797 vs 0.4956). This limited ensemble effect suggests that the component models produce similar predictions, reducing the benefit of combining them. The ensemble provides marginal improvement in robustness but does not significantly outperform the best single model.

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
- Training Time: ~1.5 hours

**Performance**:
- **Test R²**: 0.4434 (best deep learning result, 3rd overall)
- **Test RMSE**: 42.52 GPa
- **Test MAE**: 28.35 GPa

**Training Progress**:
- Train R²: 0.86-0.87 (high learning capacity)
- Val R²: 0.12-0.35 (validation performance)
- Improvement: +39% compared to 300 epochs (R²: 0.3191 → 0.4434)

**Analysis**:
Significant improvement was achieved by increasing epochs from 300 to 1000. The batch size of 16 proved optimal for this dataset. The model shows good learning capacity, but validation performance is lower than training performance, indicating some overfitting. Early stopping prevented excessive overfitting.

**Key Insights**:
1. Epoch number significantly impacts performance (+39% improvement from 300 to 1000 epochs)
2. Smaller batch size (16) works better than larger (64) for this dataset
3. Transformer architecture can achieve competitive results with proper tuning
4. The model ranks 3rd overall, demonstrating the potential of deep learning approaches

**Comparison with Classical ML**:
- Transformer (R² = 0.4434) is competitive with Gradient Boosting (R² = 0.4956)
- The difference of 0.0522 (≈10%) suggests that with further optimization, Transformer could potentially match or exceed Gradient Boosting performance

4.2.3.2 Transformer (2000 epochs, batch_size=64) - Overfitting Case Study

**Configuration**:
- Architecture: Transformer Encoder
- Epochs: 2000 (stopped at 451 due to early stopping)
- Batch Size: 64
- Learning Rate: 5e-5
- Early Stopping Patience: 300
- Best Val R²: 0.3416
- Training Time: ~2-3 hours

**Performance**:
- **Test R²**: 0.3259 (lower than batch_size=16)
- **Test RMSE**: 46.80 GPa
- **Test MAE**: 30.02 GPa

**Training Progress**:
- Train R²: 0.79-0.84 (high)
- Val R²: -0.12 to -0.24 (negative, overfitting sign)
- Best Val R²: 0.3416

**Analysis**:
**Overfitting was clearly detected**: High train R² (0.79-0.84) but negative val R² (-0.12 to -0.24) indicates severe overfitting. The larger batch size (64) led to performance degradation compared to batch_size=16 (R² = 0.3259 vs 0.4434). The lower learning rate (5e-5) may have contributed to slower convergence and less effective learning.

**Key Lessons**:
1. Batch size significantly impacts model performance (16 > 64 for this dataset)
2. Larger batch sizes can lead to overfitting in small datasets (322 samples)
3. Optimal hyperparameters are dataset-dependent
4. Early stopping is crucial for preventing overfitting
5. Training longer does not always improve performance if hyperparameters are suboptimal

**Comparison**:
- batch_size=16: R² = 0.4434 (optimal)
- batch_size=64: R² = 0.3259 (overfitting)
- **Performance difference**: -0.1175 (≈26% degradation)

4.2.4 Graph Neural Networks

4.2.4.1 CGCNN

**Configuration**:
- Architecture: Crystal Graph Convolutional Neural Network
- Epochs: 200
- Batch Size: 64
- Learning Rate: 1e-3
- Training Time: ~30 minutes

**Performance**:
- **Test R²**: 0.2187 (moderate, 5th overall)
- **Test RMSE**: 50.38 GPa
- **Test MAE**: 29.90 GPa

**Analysis**:
CGCNN achieved positive R², indicating some learning capability. However, performance is lower than classical ML and Transformer models. The graph structure may not be fully utilized with the current dataset. There is potential for improvement with longer training or architecture modifications.

**Recommendations**:
1. Increase training epochs (500+)
2. Review graph construction methodology
3. Adjust architecture parameters
4. Consider feature integration with graph structure
5. Experiment with different GNN architectures

4.2.4.2 MEGNet

**Configuration**:
- Architecture: Materials EGraph Network
- Epochs: 200
- Batch Size: 64
- Learning Rate: 1e-3
- Training Time: ~30 minutes

**Performance**:
- **Test R²**: -0.0096 (negative, poor performance, 6th overall)
- **Test RMSE**: 57.27 GPa (highest)
- **Test MAE**: 41.39 GPa (highest)

**Analysis**:
**Negative R² indicates the model performs worse than a baseline (mean predictor)**. The model failed to learn meaningful patterns from the data. Possible causes include:
1. Insufficient training epochs (200 may be too few)
2. Architecture mismatch with dataset characteristics
3. Graph construction issues
4. Hyperparameter misconfiguration
5. Dataset size too small for GNN complexity

**Recommendations**:
1. Increase training epochs significantly (500+)
2. Review graph construction methodology
3. Adjust architecture parameters
4. Consider hybrid approaches (graph + features)
5. Evaluate on larger datasets if available

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
- **Data efficiency**: Classical ML methods achieve better performance with less data

**Implications**:
- For practical applications with limited data, classical ML methods are preferred
- Deep learning models show promise but require careful hyperparameter tuning
- Graph neural networks need significant improvements before they can compete

4.3.3 Training Efficiency

| Model | Training Time | Epochs | Efficiency | Performance (R²) |
|-------|--------------|--------|------------|-----------------|
| Gradient Boosting | ~10 minutes | N/A | ⭐⭐⭐⭐⭐ | 0.4956 |
| Ensemble | ~15 minutes | N/A | ⭐⭐⭐⭐ | 0.4797 |
| Transformer (bs=16) | ~1.5 hours | 262 | ⭐⭐⭐ | 0.4434 |
| Transformer (bs=64) | ~2-3 hours | 451 | ⭐⭐ | 0.3259 |
| CGCNN | ~30 minutes | 200 | ⭐⭐⭐ | 0.2187 |
| MEGNet | ~30 minutes | 200 | ⭐⭐⭐ | -0.0096 |

**Efficiency Analysis**:
- **Best efficiency**: Gradient Boosting (highest performance, shortest training time)
- **Deep learning trade-off**: Longer training time for competitive but not superior performance
- **GNN inefficiency**: Long training time for poor performance

4.3.4 Hyperparameter Sensitivity Analysis

**Critical Findings**:

1. **Batch Size Impact** (Transformer):
   - batch_size=16: R² = 0.4434 (optimal)
   - batch_size=64: R² = 0.3259 (overfitting)
   - **Conclusion**: Smaller batch sizes are more effective for small datasets
   - **Mechanism**: Smaller batches provide more gradient updates per epoch, better generalization

2. **Epoch Number Impact** (Transformer):
   - 300 epochs: R² = 0.3191
   - 1000 epochs (262 actual): R² = 0.4434
   - **Improvement**: +39% with increased epochs
   - **Conclusion**: More epochs significantly improve deep learning models
   - **Limitation**: Diminishing returns and overfitting risk

3. **Learning Rate Impact**:
   - Transformer with lr=1e-4 (bs=16): R² = 0.4434
   - Transformer with lr=5e-5 (bs=64): R² = 0.3259
   - **Conclusion**: Learning rate must be balanced with batch size
   - **Optimal**: 1e-4 with batch_size=16

4. **Model Architecture Impact**:
   - Classical ML (Gradient Boosting): R² = 0.4956
   - Deep Learning (Transformer): R² = 0.4434
   - Graph Neural Networks (CGCNN): R² = 0.2187
   - **Conclusion**: Architecture choice significantly impacts performance

4.4 Key Insights and Recommendations

4.4.1 Major Findings

1. **Gradient Boosting is the optimal choice** for this task
   - Highest accuracy (R² = 0.4956)
   - Best computational efficiency (~10 minutes)
   - Interpretable results (feature importance)
   - Robust to outliers (Huber loss)

2. **Transformer models show promise** with proper configuration
   - Achieved R² = 0.4434 with optimal settings
   - Significant improvement from hyperparameter tuning (+39%)
   - Potential for further improvement
   - Competitive with classical ML (within 10%)

3. **Graph neural networks need improvement**
   - Current implementations underperform
   - May require architecture modifications or more data
   - Graph structure utilization needs optimization
   - Negative R² for MEGNet indicates fundamental issues

4. **Hyperparameter sensitivity**
   - Batch size significantly impacts performance (16 > 64)
   - Epoch number affects deep learning models more than classical ML
   - Learning rate requires careful tuning
   - Optimal hyperparameters are dataset-dependent

5. **Dataset size limitations**
   - Small dataset (322 samples) favors classical ML
   - Deep learning models may benefit from more data
   - GNNs likely need significantly more data

4.4.2 Recommendations

**For Production Use**:
- **Use Gradient Boosting** for best accuracy and efficiency
- **Consider Ensemble** for additional robustness (marginal benefit)
- **Avoid GNNs** until significant improvements are made

**For Research**:
- **Investigate Transformer improvements**:
  - Try different architectures (e.g., attention mechanisms)
  - Experiment with longer training (2000+ epochs with proper regularization)
  - Explore learning rate scheduling strategies
  - Investigate transfer learning from larger datasets
  
- **Improve Graph Neural Networks**:
  - Review graph construction methodology
  - Experiment with different GNN architectures
  - Consider hybrid approaches (graph + features)
  - Evaluate on larger datasets

**For Future Work**:
1. **Data augmentation** to increase dataset size
2. **Feature engineering** improvements
3. **Transfer learning** from larger datasets
4. **Ensemble of diverse models** (including improved GNNs)
5. **Active learning** to identify high-value data points

4.5 Statistical Analysis

4.5.1 Performance Distribution

The models show a clear performance hierarchy:
- **Top tier** (R² > 0.4): Gradient Boosting (0.4956), Ensemble (0.4797), Transformer bs=16 (0.4434)
- **Middle tier** (0.2 < R² < 0.4): Transformer bs=64 (0.3259), CGCNN (0.2187)
- **Bottom tier** (R² < 0.2): MEGNet (-0.0096)

**Performance Gap Analysis**:
- Gap between top and middle tier: ~0.12-0.22 R²
- Gap between middle and bottom tier: ~0.23 R²
- **Conclusion**: Clear separation between model categories

4.5.2 Error Analysis

**RMSE Analysis**:
- Best: 40.42 GPa (Gradient Boosting)
- Worst: 57.27 GPa (MEGNet)
- Range: 16.85 GPa difference
- **Relative improvement**: 29.4% reduction from worst to best

**MAE Analysis**:
- Best: 25.14 GPa (Gradient Boosting)
- Worst: 41.39 GPa (MEGNet)
- Range: 16.25 GPa difference
- **Relative improvement**: 39.3% reduction from worst to best

**Error Distribution**:
- Top 3 models (GB, Ensemble, Transformer bs=16): RMSE < 43 GPa
- Bottom 3 models (Transformer bs=64, CGCNN, MEGNet): RMSE > 46 GPa
- **Clear separation** between high and low performers

4.5.3 Improvement Potential

Based on current results:
- **Gradient Boosting**: Limited room for improvement (already optimal, R² = 0.4956)
- **Transformer**: Potential for 5-10% improvement with further tuning (target: R² = 0.48-0.50)
- **GNNs**: Significant improvement potential (50-100% with proper configuration, target: R² = 0.30-0.40)

4.6 Feature Importance and Interpretation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tree-based models such as Random Forest and Gradient Boosting provide estimates of feature importance. Across different runs and data subsets, the following features consistently emerge as most influential:

1. **Ti composition fraction (`comp_Ti`)** – The Ti content shows the single largest contribution (≈16–17% of total importance in Random Forest), reflecting the strong impact of Ti on stiffness in HEA/MPEA systems, particularly in Ti-rich biocompatible alloys.  
2. **Average electronegativity** – Alloys with higher or lower average electronegativity, and large electronegativity mismatch, exhibit systematic trends in elastic modulus, likely reflecting bonding character and phase stability.  
3. **Valence electron concentration (VEC)** – VEC affects phase selection (e.g. BCC vs FCC) and bonding, which in turn influence stiffness.  
4. **Co composition fraction (`comp_Co`)** – Co-rich alloys tend to have higher modulus values; the Co fraction is thus a strong positive contributor to stiffness.  
5. **Density** – Denser alloys, which often contain heavy refractory elements, frequently show higher elastic modulus.  

Other elemental fractions (e.g. Nb, Ta, Zr, Hf, Fe, Ni, Cr) also contribute meaningfully, but with lower individual importance. From a design point of view, the dominance of Ti content and related descriptors suggests that careful control of Ti-rich compositions, combined with appropriate selections of companion elements, is a promising strategy for tuning modulus toward bone-like values.

4.7 Error and Residual Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Residual plots and predicted-vs-actual scatter plots reveal several important patterns:

- For well-performing models (Gradient Boosting, Ensemble, Transformer bs=16), errors are roughly symmetric around zero and do not show strong systematic trends with predicted modulus.  
- Errors tend to be larger at the extremes of the modulus distribution, especially for very high-modulus alloys dominated by refractory elements, where data are sparser and computational uncertainties are larger.  
- The distribution of residuals is reasonably centered and unimodal for the bulk of the data, with a moderate tail of underpredictions and overpredictions.  
- Models with poor performance (MEGNet, CGCNN) show systematic biases and larger error magnitudes.

These analyses suggest that, while the models are reasonably calibrated in the central range of the data, care must be taken when interpreting predictions for very low- or very high-modulus alloys, especially if they are far from the training-data manifold.

4.8 Case Studies: Alloys in the Biomedical Modulus Range
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An important application target of this work is the design of HEA/MPEA alloys with elastic modulus in the range of approximately 30–90 GPa, which overlaps with cortical bone. Within the experimental subset of 322 alloys, around **30** data points fall into this range. Many of these alloys are based on Ti–Zr–Nb–Ta–Hf families, which are known to be promising for biomedical applications.  

Model predictions for these alloys generally match the experimentally reported values within tens of GPa, which, while not yet precise enough for final design, can significantly narrow the search space for experimental exploration. The models can thus be used to prioritize candidate compositions for further study, particularly by scanning the broader 5,339-sample integrated dataset and identifying combinations projected to fall within or near the target modulus window.

**Best Model for Biomedical Applications**:
- **Gradient Boosting** (R² = 0.4956) is recommended for screening candidate alloys
- **Transformer** (R² = 0.4434) provides alternative predictions for validation
- Both models can identify promising compositions in the 30-90 GPa range

"""

def main():
    """メイン関数"""
    print("=" * 80)
    print("最終統合レポート作成")
    print("=" * 80)
    
    # メインレポートを読み込む
    main_content = read_file_content(MAIN_REPORT)
    if main_content is None:
        print(f"❌ メインレポートが見つかりません: {MAIN_REPORT}")
        return
    
    # セクション4を置き換え
    section_4_pattern = r'(4\. Results.*?)(?=\n---\n\n5\. Discussion)'
    section_4_match = re.search(section_4_pattern, main_content, re.DOTALL)
    
    if section_4_match:
        old_section_4 = section_4_match.group(1)
        new_section_4 = create_new_section_4()
        
        # バックアップを作成
        backup_file = BASE_DIR / "PROJECT_FINAL_REPORT_EN_backup_before_integration.md"
        with open(backup_file, 'w', encoding='utf-8') as f:
            f.write(main_content)
        print(f"✅ バックアップを作成: {backup_file}")
        
        # セクション4を置き換え
        new_main_content = main_content.replace(old_section_4, new_section_4)
        
        # 保存
        with open(MAIN_REPORT, 'w', encoding='utf-8') as f:
            f.write(new_main_content)
        print(f"✅ メインレポートを更新: {MAIN_REPORT}")
        
        print("\n" + "=" * 80)
        print("✅ 統合完了！")
        print("=" * 80)
        print(f"\n📊 更新されたレポート: {MAIN_REPORT}")
        print(f"📁 バックアップ: {backup_file}")
        print(f"📈 包括的レポート: {COMPREHENSIVE_REPORT}")
        print(f"🖼️  ビジュアライゼーション: {FIGURES_DIR}")
    else:
        print("⚠️ セクション4が見つかりませんでした。手動で統合してください。")

if __name__ == "__main__":
    main()
