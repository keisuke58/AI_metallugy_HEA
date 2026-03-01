# Model Training Results

This document presents the comprehensive results of training Graph Neural Network (GNN) and Transformer models for predicting elastic modulus in High-Entropy Alloys (HEAs).

## Performance Summary

| Model | Dataset Size | Test R² | Test RMSE (GPa) | Test MAE (GPa) |
|-------|--------------|---------|-----------------|----------------|
| Transformer* | Standard (322) | 0.6249 | 34.91 | 26.17 |
| GNN | Standard (322) | 0.2586 | 49.08 | 29.53 |
| Transformer | Large (5340) | 0.0496 | 88.80 | 66.99 |
| Transformer | Standard (322) | -0.2384 | 63.43 | 41.68 |

*Best result from previous training session

## Best Performing Model

The best performing model is the **Transformer** model with a Test R² score of **0.6249**, achieving a Test RMSE of 34.91 GPa and Test MAE of 26.17 GPa.
