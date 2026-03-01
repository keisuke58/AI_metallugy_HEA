Machine Learning-based Prediction of Elastic Modulus in High-Entropy and Multi-Principal Element Alloys
======================================================================================================

*AI_metallurgy Project – Final Report (English)*

Date: 2026-01-23  
Project: AI-based design of biomedical HEA/MPEA alloys  
Author: Keisuke Nishioka (10081049)

---

Abstract
--------

The elastic modulus (Young’s modulus) mismatch between metallic implants and human bone remains a critical challenge in biomedical materials design. High-entropy alloys (HEAs) and multi-principal element alloys (MPEAs) offer great compositional flexibility for tuning mechanical properties, but systematic exploration of the vast composition space is experimentally expensive. In this work, we compile and integrate elastic-modulus data from multiple experimental and computational sources, and develop machine-learning models to predict the elastic modulus of HEA/MPEA systems. The integrated dataset contains 5,339 alloys (≈6.3% experimental, ≈93.7% computed) after cleaning and de-duplication, with elastic modulus values spanning 2.7–621 GPa. We construct 29 physically motivated features including composition descriptors (e.g. Ti, Zr, Nb fractions), atomic descriptors (atomic radius, electronegativity, valence electron concentration), and density-related quantities.  

Using this feature set, we benchmark linear and non-linear regression models such as Linear/Ridge/Lasso Regression, Polynomial Regression, K-Nearest Neighbors, Random Forest, Support Vector Regression (SVR), gradient boosting methods, and a feedforward neural network. Models are trained on the full dataset and evaluated on both the complete set and an experimental-only subset. An optimized SVR model achieves a test \(R^2 \approx 0.59\) on the combined 322-sample experimental dataset used in the earlier phase of the project, while a gradient boosting model attains \(R^2 \approx 0.67\) on an experimental-only subset, demonstrating practically useful predictive capability. Feature-importance analyses consistently highlight Ti content, average electronegativity, valence electron concentration, and density as key drivers of the elastic modulus. We discuss the implications for the design of low-modulus HEAs for biomedical implants, limitations arising from mixed experimental and computational data, and directions for extending the models and dataset.  

---

1. Introduction
---------------

Metallic implants, such as hip and knee replacements, must satisfy multiple mechanical and biological requirements. A particularly important issue is the elastic modulus mismatch between conventional metallic biomaterials (e.g. Ti–6Al–4V, Co–Cr alloys, stainless steels) and cortical bone. While human bone typically exhibits Young’s modulus values in the 10–30 GPa range, commonly used metallic implants have moduli in the 100–200 GPa range or higher. This mismatch can lead to stress shielding, bone resorption, and long-term implant loosening.  

High-entropy alloys (HEAs) and multi-principal element alloys (MPEAs) have emerged as promising candidates for next-generation structural and functional materials. Their compositional flexibility enables tailoring of properties such as strength, ductility, corrosion resistance, and elastic modulus. In the biomedical context, HEAs based on biocompatible elements (e.g. Ti, Zr, Nb, Ta, Hf) offer the possibility of achieving reduced stiffness while maintaining adequate strength and corrosion resistance. However, the design space spanned by multi-component compositions and processing conditions is enormous, making exhaustive experimental exploration infeasible.  

Several databases and studies have documented mechanical properties of HEAs and related alloys. Gorsse et al. compiled a comprehensive database of mechanical properties for HEAs and complex concentrated alloys, including Young’s modulus for a large set of alloys. The DOE/OSTI database provides experimental Young’s modulus values and related descriptors for high-entropy alloys. In parallel, large-scale computational databases such as the Materials Project publish density functional theory (DFT) elastic tensors for inorganic compounds, from which bulk, shear, and Young’s moduli can be derived. Additional datasets, such as MPEA mechanical-property compilations and nano-indentation databases, further expand the available information.  

The AI_metallurgy project leverages these resources to build an integrated, cleaned dataset of HEA/MPEA elastic modulus values and to train machine-learning models that can predict the modulus from composition and derived descriptors. Earlier phases of the project focused on an experimentally dominated subset of 322 alloys, for which we developed and optimized multiple regression models. In the present report, we consolidate the full data pipeline and modeling results into a single narrative suitable for use in scientific manuscripts or extended technical documentation. Our objectives are:

1. To construct a unified dataset of experimental and computed Young’s modulus values for HEAs and MPEAs, with documented provenance and quality checks.  
2. To engineer a compact but physically meaningful feature set that captures key compositional and atomic descriptors.  
3. To benchmark a range of machine-learning models and perform hyperparameter optimization on both the integrated and experimental-only datasets.  
4. To analyze feature importance and discuss implications for the design of low-modulus HEA/MPEA alloys for biomedical applications.  

---

2. Data and Dataset Construction
--------------------------------

2.1 Data Sources
~~~~~~~~~~~~~~~~

The final integrated dataset is constructed from a combination of experimental and computational sources. The main data sources are:

- **DOE/OSTI high-entropy alloy dataset (experimental)**  
  A curated dataset of high-entropy alloys with phase information and, for a subset, experimentally measured Young’s modulus and several derived features. After de-duplication and cleaning, 101 alloys with valid elastic modulus values are retained.

- **Gorsse HEA mechanical-properties database (experimental)**  
  A large literature-based database of HEAs and complex concentrated alloys with composition, microstructure, density, hardness, yield strength, ultimate tensile strength, elongation, and, for many entries, Young’s modulus. From this source, 182 unique alloys with valid Young’s modulus values are included.

- **Latest experimental HEA data (experimental)**  
  Additional data from recent (2024–2025) publications on Ti–Zr–Nb, Ti–Zr–Hf–Nb–Ta, and related biocompatible HEA/MPEA systems are incorporated where possible. Due to overlaps with earlier databases and nano-indentation compilations, the net increase in unique alloys is modest.

- **MPEA nano-indentation dataset (experimental and computed)**  
  A dataset of phase-specific mechanical properties measured by nano-indentation for multi-principal element alloys. We extract 767 entries with experimental or derived Young’s modulus values. After integration, 51 experimental entries and 196 calculated entries remain as distinct samples.

- **Refractory HEA elastic-constants dataset (computed)**  
  A set of DFT-computed elastic constants for 370 refractory HEAs. Young’s modulus is computed from reported bulk and shear moduli or elastic tensors.

- **Materials Project (computed)**  
  The Materials Project database provides DFT-derived elastic tensors for a large number of inorganic compounds. For suitable compositions related to HEA/MPEA chemistries, we compute Young’s modulus via  
  \( E = 9 K G / (3 K + G) \),  
  where \(K\) and \(G\) are bulk and shear moduli, respectively. This source contributes 4,439 samples.

- **Auxiliary databases (mechanical-property compilations)**  
  Additional databases such as DISMA and MPEA mechanical-property compilations are used to cross-check or augment metadata (e.g. strength, hardness, phase information), but do not systematically provide Young’s modulus values and therefore are not primary sources for the target property.

2.2 Integration and Cleaning Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each dataset is first parsed into a standardized tabular format with at least the following fields: alloy identifier (name or composition string), elastic modulus (GPa), data source, and, where available, composition and auxiliary descriptors. The main integration steps are:

1. **Label harmonization** – Columns corresponding to Young’s modulus in different source formats are mapped to a unified label `elastic_modulus` and converted to units of GPa.  
2. **Filtering invalid entries** – Entries with missing, non-positive, or obviously unphysical modulus values are removed.  
3. **Duplicate detection and removal** – Exact and near-duplicate rows are removed based on alloy name, composition, and close modulus values. For repeated alloys across sources, priority rules are applied (e.g. experimental over computed data where possible).  
4. **Mandatory fields** – Rows lacking `alloy_name` (or equivalent identifier), `elastic_modulus`, or `source` fields are dropped.  
5. **Consolidation and validation** – The cleaned data from all sources are concatenated and passed through validation scripts that check schema consistency, missing values, duplicates, and basic statistics.

The resulting integrated dataset is stored as `data_collection/final_data/unified_dataset_cleaned_20260123_175245.csv`, with a stable copy at `final_data/unified_dataset_latest.csv`. After integration and cleaning, the dataset contains **5,339** unique alloy entries.

2.3 Final Dataset Characteristics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The 5,339-sample integrated dataset exhibits the following key characteristics:

- **Source breakdown**:  
  - Materials Project: 4,439 samples (83.1%)  
  - Refractory HEA elastic-constants dataset: 370 samples (6.9%)  
  - MPEA nano-indentation (calculated): 196 samples (3.7%)  
  - Gorsse database: 182 samples (3.4%)  
  - DOE/OSTI: 101 samples (1.9%)  
  - MPEA nano-indentation (experimental): 51 samples (1.0%)  

- **Experimental vs computed**:  
  - Experimental Young’s modulus: 335 samples (≈6.3%)  
  - Computed Young’s modulus: 5,004 samples (≈93.7%)  

- **Elastic modulus distribution (GPa)**:  
  - Minimum: 2.73  
  - Maximum: 621.33  
  - Mean: 139.26  
  - Median: 122.61  
  - ≈99.3% of samples lie between 10 and 500 GPa.  

The earlier phase of the project used a predominantly experimental subset of **322** alloys (from DOE/OSTI, Gorsse, and a few recent experiments) to develop and interpret models. In that subset, the elastic modulus ranges approximately from 27 to 466 GPa, with an average around 160–165 GPa, and about **30** alloys fall in the 30–90 GPa range relevant for bone-like stiffness. This subset plays a key role in evaluating model performance on purely experimental data.

---

3. Features and Machine Learning Methods
----------------------------------------

3.1 Feature Engineering
~~~~~~~~~~~~~~~~~~~~~~~

To map alloy compositions to elastic modulus, we construct a set of **29 features** that combine composition information with atomic and thermodynamic descriptors. The feature design is guided by prior knowledge of factors that influence stiffness in metallic alloys. The main categories are:

- **Atomic-radius descriptors**  
  - Average atomic radius  
  - Standard deviation of atomic radius  
  - Maximum and minimum atomic radius  
  - Difference between maximum and minimum radius (Δr)  

- **Electronegativity descriptors**  
  - Average electronegativity  
  - Standard deviation of electronegativity  
  - Difference between maximum and minimum electronegativity (Δχ)  

- **Electronic and thermodynamic descriptors**  
  - Valence electron concentration (VEC)  
  - Configurational mixing entropy  
  - Number of principal elements in the alloy  

- **Composition fractions for key elements**  
  - Fractions of 17 elements commonly appearing in HEA/MPEA systems:
    Ti, Zr, Hf, Nb, Ta, V, Cr, Mo, W, Fe, Co, Ni, Cu, Al, Mn, Si, Sn  

- **Additional physical descriptors**  
  - Density (where available)  
  - Selected derived quantities from specific datasets (e.g. existing descriptors in the DOE/OSTI dataset).  

These features are computed from nominal alloy compositions and tabulated atomic properties. For experimental alloys where compositions are not strictly equiatomic, the actual reported atomic or weight percentages are used whenever available. All features are standardized or scaled as appropriate for the machine-learning models.

3.2 Models
~~~~~~~~~~

We benchmark a variety of regression algorithms to capture potential linear and non-linear relationships between the feature set and elastic modulus:

- **Linear models**  
  - Ordinary least-squares Linear Regression  
  - Ridge Regression (L2 regularization)  
  - Lasso Regression (L1 regularization)

- **Polynomial Regression**  
  - Linear regression on polynomial feature expansions to capture higher-order interactions between descriptors.

- **Instance-based method**  
  - K-Nearest Neighbors (KNN) regression with distance-based weighting.

- **Tree-based ensemble models**  
  - Random Forest Regression  
  - Gradient Boosting Regression  
  - Stacking ensemble models combining multiple base learners

- **Kernel-based model**  
  - Support Vector Regression (SVR) with radial basis function (RBF) kernels and tuned hyperparameters.

- **Feedforward neural network (MLFFNN)**  
  - A dense neural network with several hidden layers, ReLU activations, and Adam optimization (implemented with TensorFlow/Keras).

The initial benchmarking phase focuses on models trained on the 322-sample experimental subset. Subsequently, optimized models (especially SVR and Gradient Boosting) are trained and evaluated on both the full dataset and an experimental-only subset to assess robustness and bias.

3.3 Training and Evaluation Protocol
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The general training and evaluation protocol is as follows:

- **Data splitting**  
  - For model development on the 322-sample subset, the data are randomly split into training and test sets with an 80/20 ratio.  
  - For experiments focusing on experimental-only data, the subset of experimental entries is selected first, and then split similarly.  

- **Cross-validation and hyperparameter tuning**  
  - For baseline models, default or moderately tuned hyperparameters are used.  
  - For optimized models, grid search or random search with 5-fold cross-validation is applied to tune hyperparameters such as regularization strength, tree depth, number of estimators, kernel parameters, and learning rates.  

- **Evaluation metrics**  
  - Coefficient of determination \(R^2\)  
  - Root mean squared error (RMSE, in GPa)  
  - Mean absolute error (MAE, in GPa)  
  - Where relevant, training vs test \(R^2\) is monitored to assess overfitting.  

All experiments are implemented in Python using scikit-learn for classical machine-learning models, TensorFlow/Keras for the neural network, and standard data-science libraries (pandas, NumPy, matplotlib, seaborn) for data processing and visualization.

---

4. Results
----------

4.1 Overall Model Performance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

On the initial 322-sample dataset, we first benchmarked eight models with baseline hyperparameters. The Random Forest model achieved the best baseline performance with:

- Test \(R^2 \approx 0.57\)  
- Test RMSE \(\approx 37.5\) GPa  
- Test MAE \(\approx 25.9\) GPa  

Linear, Ridge, and Lasso regression models attained more modest performance with \(R^2 \approx 0.35–0.37\), while KNN showed poor generalization and Polynomial Regression suffered from severe overfitting (negative \(R^2\) on the test set). The MLFFNN model achieved intermediate performance with \(R^2 \approx 0.49\).  

Subsequent hyperparameter optimization led to substantial improvements, especially for SVR and Gradient Boosting:

- **Optimized SVR on the full 322-sample dataset**  
  - Test \(R^2 \approx 0.59\)  
  - Test RMSE \(\approx 36.2\) GPa  
  - Improvement of roughly 67% in \(R^2\) compared to the non-optimized SVR baseline.  

- **Optimized Gradient Boosting on experimental-only data**  
  - When focusing on an experimental-only subset (≈109 samples), an optimized Gradient Boosting model achieves  
    - Test \(R^2 \approx 0.67\)  
    - Test RMSE \(\approx 49.7\) GPa  
    - Test MAE \(\approx 37.1\) GPa  

Stacking ensembles combining multiple base learners achieve comparable performance to the best single models. Overall, the range \(R^2 \approx 0.59–0.67\) indicates moderate to good predictive power given the inherent noise and heterogeneity in the underlying data.

4.2 Feature Importance and Interpretation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tree-based models such as Random Forest and Gradient Boosting provide estimates of feature importance. Across different runs and data subsets, the following features consistently emerge as most influential:

1. **Ti composition fraction (`comp_Ti`)** – The Ti content shows the single largest contribution (≈16–17% of total importance in Random Forest), reflecting the strong impact of Ti on stiffness in HEA/MPEA systems, particularly in Ti-rich biocompatible alloys.  
2. **Average electronegativity** – Alloys with higher or lower average electronegativity, and large electronegativity mismatch, exhibit systematic trends in elastic modulus, likely reflecting bonding character and phase stability.  
3. **Valence electron concentration (VEC)** – VEC affects phase selection (e.g. BCC vs FCC) and bonding, which in turn influence stiffness.  
4. **Co composition fraction (`comp_Co`)** – Co-rich alloys tend to have higher modulus values; the Co fraction is thus a strong positive contributor to stiffness.  
5. **Density** – Denser alloys, which often contain heavy refractory elements, frequently show higher elastic modulus.  

Other elemental fractions (e.g. Nb, Ta, Zr, Hf, Fe, Ni, Cr) also contribute meaningfully, but with lower individual importance. From a design point of view, the dominance of Ti content and related descriptors suggests that careful control of Ti-rich compositions, combined with appropriate selections of companion elements, is a promising strategy for tuning modulus toward bone-like values.

4.3 Error and Residual Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Residual plots and predicted-vs-actual scatter plots reveal several important patterns:

- For well-performing models (Random Forest, optimized SVR, Gradient Boosting), errors are roughly symmetric around zero and do not show strong systematic trends with predicted modulus.  
- Errors tend to be larger at the extremes of the modulus distribution, especially for very high-modulus alloys dominated by refractory elements, where data are sparser and computational uncertainties are larger.  
- The distribution of residuals is reasonably centered and unimodal for the bulk of the data, with a moderate tail of underpredictions and overpredictions.  

These analyses suggest that, while the models are reasonably calibrated in the central range of the data, care must be taken when interpreting predictions for very low- or very high-modulus alloys, especially if they are far from the training-data manifold.

4.4 Case Studies: Alloys in the Biomedical Modulus Range
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An important application target of this work is the design of HEA/MPEA alloys with elastic modulus in the range of approximately 30–90 GPa, which overlaps with cortical bone. Within the experimental subset of 322 alloys, around **30** data points fall into this range. Many of these alloys are based on Ti–Zr–Nb–Ta–Hf families, which are known to be promising for biomedical applications.  

Model predictions for these alloys generally match the experimentally reported values within tens of GPa, which, while not yet precise enough for final design, can significantly narrow the search space for experimental exploration. The models can thus be used to prioritize candidate compositions for further study, particularly by scanning the broader 5,339-sample integrated dataset and identifying combinations projected to fall within or near the target modulus window.

---

5. Discussion
-------------

5.1 Experimental vs Computed Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The integrated dataset combines a relatively small fraction of experimental data (≈6.3%) with a large fraction of computed values from DFT-based sources such as the Materials Project and refractory HEA elastic-constants datasets. This mixture has both advantages and drawbacks:

- **Advantages**  
  - Greatly increased coverage of composition space, including hypothetical or metastable alloys not yet synthesized.  
  - Ability to explore trends in modulus across broad families of compositions.  

- **Drawbacks**  
  - DFT-calculated moduli often overestimate stiffness compared to experiment, particularly for materials with strong bonding.  
  - Temperature conditions and microstructural states differ between computation (0 K ideal crystals) and experiment (room temperature, various processing histories).  
  - Noise and bias in experimental measurements are not directly reflected in computed data.  

To manage these issues, we adopt the following strategy:

1. Train models on the full integrated dataset to leverage broad coverage.  
2. Evaluate performance and interpretability on experimental-only subsets to obtain more realistic estimates of predictive power.  
3. Use experimental-only evaluations (e.g. Gradient Boosting with \(R^2 \approx 0.67\)) as a primary indicator of model quality for practical applications.  

5.2 Limitations and Uncertainties
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Several limitations should be kept in mind when interpreting the results:

- **Data quantity and balance** – The number of experimental data points (≈300–350) remains modest relative to the complexity of the composition space. Certain compositional regions are under-represented, leading to higher uncertainty in those regimes.  
- **Lack of processing information** – Many entries lack detailed information on processing history (e.g. heat treatment, cold working, aging), which can significantly affect modulus. Incorporating such metadata would likely improve model performance.  
- **Temperature effects** – Most data correspond to near-room-temperature measurements or 0 K DFT calculations, but precise temperature information is not consistently recorded.  
- **Model complexity vs interpretability** – While non-linear models such as SVR and Gradient Boosting achieve higher accuracy, they can be less interpretable than simpler linear models. Feature-importance measures help but do not fully resolve this issue.  

Despite these limitations, the current dataset and models already provide a useful starting point for data-driven design of HEA/MPEA alloys with targeted elastic modulus.

5.3 Implications for Materials Design
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The feature-importance analysis suggests several qualitative design guidelines:

- **Ti-rich compositions** – Increasing Ti content tends to reduce the elastic modulus in many HEA/MPEA systems, which is beneficial for biomedical applications.  
- **Control of electronegativity and VEC** – Tuning average electronegativity and VEC can shift phase stability and bonding character, influencing stiffness.  
- **Balancing heavy refractory elements** – While elements such as W, Mo, and Ta contribute to high strength and stiffness, they also tend to increase modulus; their fractions must be balanced against the desire for lower modulus.  
- **Density as a proxy** – Lower-density alloys, often with higher Ti and lower refractory content, tend to have lower modulus, though this is not a strict rule.  

These guidelines, when combined with model-based screening, can help identify promising alloy compositions for further experimental validation.

---

6. Conclusion and Future Work
-----------------------------

6.1 Summary of Achievements
~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this project, we:

1. **Constructed a unified dataset** of **5,339** HEA/MPEA alloys with elastic modulus values, integrating multiple experimental and computational sources and performing systematic cleaning and validation.  
2. **Designed a 29-feature representation** capturing key compositional, atomic, and physical descriptors relevant to stiffness.  
3. **Implemented and benchmarked multiple machine-learning models**, including linear models, tree-based ensembles, SVR, KNN, polynomial regression, and a feedforward neural network.  
4. **Optimized selected models**, achieving test \(R^2 \approx 0.59\) for an SVR model on the 322-sample dataset and \(R^2 \approx 0.67\) for a Gradient Boosting model on experimental-only data.  
5. **Analyzed feature importance and residual patterns**, identifying Ti content, average electronegativity, VEC, and density as key drivers of elastic modulus.  

These results demonstrate that data-driven models can provide practically useful predictions of elastic modulus for HEA/MPEA alloys, particularly when combined with domain knowledge and careful consideration of data limitations.

6.2 Future Directions
~~~~~~~~~~~~~~~~~~~~~

Future work can proceed along several directions:

- **Data expansion and refinement**  
  - Incorporate additional experimental data, particularly for Ti-based and biomedical HEAs, to strengthen model training and validation.  
  - Integrate more detailed processing and microstructural metadata where available.  

- **Advanced modeling approaches**  
  - Explore graph neural networks (GNNs), Fourier neural operators (FNOs), and other advanced architectures developed in related parts of the AI_metallurgy project for elastic-modulus prediction.  
  - Develop multi-task models that jointly predict modulus, strength, hardness, and other properties.  

- **Uncertainty quantification and active learning**  
  - Quantify predictive uncertainty using probabilistic models or ensemble methods.  
  - Implement active-learning strategies to propose new experiments in regions of high uncertainty or potential performance gains.  

- **Inverse design and optimization**  
  - Use the trained models in combination with optimization algorithms to propose new alloy compositions targeting specific modulus ranges (e.g. 30–90 GPa) under additional constraints on strength or composition.  

---

7. References
-------------

The primary data sources and tools used in this work include:

1. Gorsse, S., Nguyen, M. H., Senkov, O. N., & Miracle, D. B. “Database on the mechanical properties of high entropy alloys and complex concentrated alloys.” *Data in Brief* 21 (2018/2019).  
2. Roy, A., & Balasubramanian, G. “Phases and Young’s Modulus Dataset for High Entropy Alloys.” DOE/OSTI, DOI: 10.18141/1644295.  
3. Jain, A., Ong, S. P., Hautier, G., Chen, W., Richards, W. D., Dacek, S., et al. “The Materials Project: A materials genome approach to accelerating materials innovation.” *APL Materials* 1, 011002 (2013).  
4. MPEA nano-indentation database, Johns Hopkins University Applied Physics Laboratory, *Data in Brief* (2024), and associated GitHub repository.  
5. Refractory HEA elastic-constants dataset, GitHub repository for DFT elastic-constant data.  
6. Additional HEA/MPEA mechanical-property compilations and project-specific scripts documented in the AI_metallurgy `data_collection` folder.  

