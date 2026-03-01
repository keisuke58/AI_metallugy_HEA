# AI Applications in Metallurgy: High Entropy Alloy (HEA) Elastic Modulus Prediction

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Research-green.svg)](LICENSE)

A comprehensive machine learning project for predicting elastic modulus of High Entropy Alloys (HEAs) using various deep learning architectures including Neural Operators, Graph Neural Networks, and Transformers.

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Data Collection](#data-collection)
- [Models](#models)
- [Results](#results)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project aims to predict the elastic modulus of High Entropy Alloys (HEAs) using state-of-the-art machine learning models. The project includes:

- **Data Collection**: Comprehensive dataset collection from multiple sources (Gorsse, DOE/OSTI, Materials Project, and recent literature)
- **Data Preprocessing**: Feature engineering and data integration
- **Model Training**: Multiple deep learning architectures for property prediction
- **Model Evaluation**: Comprehensive performance analysis and comparison

### Key Objectives

- Predict elastic modulus of HEAs from composition
- Compare different deep learning architectures
- Provide reproducible research pipeline
- Document data sources and methodologies

## 📁 Project Structure

```
AI_metallurgy/
├── data_collection/          # Data collection and preprocessing
│   ├── raw_data/            # Raw datasets from various sources
│   ├── processed_data/      # Preprocessed and integrated data
│   ├── final_data/          # Final unified datasets
│   ├── scripts/             # Data collection and processing scripts
│   ├── models/              # Trained ML models
│   ├── results/             # Model evaluation results
│   └── figures/             # Visualization figures
│
├── fno_models/              # Neural Operator models
│   ├── models/              # Model implementations (FNO, DeepONet, MEGNet, etc.)
│   ├── data_loaders/        # Data loaders for different models
│   ├── checkpoints/         # Trained model checkpoints
│   ├── results/             # Training results
│   └── train.py             # Training script
│
├── gnn_transformer_models/  # GNN and Transformer models
│   ├── models/              # Model implementations
│   ├── results/             # Training results
│   └── train.py             # Training script
│
├── AI_Applications_in_Metallurgy/  # Course materials and papers
└── AI_Metallurgy_Temp/      # Temporary files and notes
```

## ✨ Features

### Data Collection
- **Multiple Data Sources**: Gorsse dataset, DOE/OSTI dataset, Materials Project API, recent literature
- **Data Integration**: Automated data merging and cleaning
- **Feature Engineering**: Material descriptors (mixing entropy, enthalpy, VEC, etc.)

### Model Architectures

#### Neural Operators (`fno_models/`)
- **FNO (Fourier Neural Operator)**: Frequency domain processing
- **DeepONet**: Branch-trunk network architecture
- **MEGNet**: Materials-specific graph networks
- **CGCNN**: Crystal graph convolutional networks
- **Neural ODE**: Continuous dynamics modeling
- **PINNs**: Physics-informed neural networks

#### Graph Neural Networks & Transformers (`gnn_transformer_models/`)
- **GNN**: Graph-based representation learning
- **Transformer**: Sequence-based attention mechanisms

### Model Evaluation
- Comprehensive metrics (R², RMSE, MAE)
- Visualization tools
- Model comparison analysis

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended, CPU also supported)
- Git

### Setup

1. **Clone the repository** (if not already done):
```bash
git clone https://github.com/keisuke58/AI_metallugy_HEA.git
cd AI_metallurgy
```

2. **Create a virtual environment** (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:

For Neural Operator models:
```bash
cd fno_models
pip install -r requirements.txt
pip install torch-geometric  # For MEGNet and CGCNN
pip install torchdiffeq      # For Neural ODE
```

For GNN/Transformer models:
```bash
cd gnn_transformer_models
pip install -r requirements.txt
pip install torch-geometric
```

## 🎯 Quick Start

### 1. Data Preparation

Ensure you have the processed data file:
```bash
# Data should be located at:
data_collection/processed_data/data_with_features.csv
```

If you need to collect and process data:
```bash
cd data_collection/scripts
# Follow the data collection guide in data_collection/README.md
```

### 2. Train Models

#### Neural Operator Models
```bash
cd fno_models
python train.py --model all --data_path ../data_collection/processed_data/data_with_features.csv
```

Train a specific model:
```bash
python train.py --model fno --data_path ../data_collection/processed_data/data_with_features.csv
```

#### GNN/Transformer Models
```bash
cd gnn_transformer_models
python train.py
```

### 3. Run Inference

```bash
cd fno_models
python inference.py --model all --data_path ../data_collection/processed_data/data_with_features.csv
```

## 📊 Data Collection

The project includes comprehensive data collection from multiple sources:

- **Gorsse Dataset**: ~370 alloys with elastic modulus data
- **DOE/OSTI Dataset**: 107 alloys with Young's modulus, 340 alloys with phase data
- **Materials Project**: DFT-calculated elastic tensors
- **Recent Literature**: 2024-2025 research data

For detailed data collection instructions, see [`data_collection/README.md`](data_collection/README.md).

## 🤖 Models

### Neural Operator Models

Located in `fno_models/`, these models implement various neural operator architectures:

- **FNO**: Fourier Neural Operator for frequency domain processing
- **DeepONet**: Deep Operator Network with branch-trunk architecture
- **MEGNet**: Materials-specific graph networks
- **CGCNN**: Crystal graph convolutional networks
- **Neural ODE**: Continuous dynamics modeling
- **PINNs**: Physics-informed neural networks

See [`fno_models/README.md`](fno_models/README.md) for detailed documentation.

### GNN & Transformer Models

Located in `gnn_transformer_models/`, these models implement:

- **GNN**: Graph Neural Network with edge gating and attention
- **Transformer**: Sequence-based transformer for composition encoding

See [`gnn_transformer_models/README.md`](gnn_transformer_models/README.md) for detailed documentation.

## 📈 Results

Model evaluation results are stored in:
- `fno_models/results/`: Neural operator model results
- `gnn_transformer_models/results/`: GNN/Transformer model results
- `data_collection/results/`: Traditional ML model results

Visualization figures are available in:
- `data_collection/figures/`: Comprehensive analysis figures

## 📚 Documentation

- [`data_collection/README.md`](data_collection/README.md): Data collection guide
- [`fno_models/README.md`](fno_models/README.md): Neural operator models documentation
- [`gnn_transformer_models/README.md`](gnn_transformer_models/README.md): GNN/Transformer models documentation
- [`FNO_IMPLEMENTATION_PLAN.md`](FNO_IMPLEMENTATION_PLAN.md): FNO implementation plan
- [`data_collection/PROJECT_FINAL_REPORT_EN.pdf`](data_collection/PROJECT_FINAL_REPORT_EN.pdf): Final project report

## 🔧 Configuration

### Data Paths

Default data paths are configured in each training script. Modify as needed:
- Neural Operators: `fno_models/train.py`
- GNN/Transformer: `gnn_transformer_models/train.py`

### Hyperparameters

Hyperparameters can be adjusted in the training scripts or via command-line arguments.

## 🐛 Troubleshooting

### CUDA Out of Memory
Reduce batch size in training scripts:
```python
batch_size = 16  # or smaller
```

### Missing Dependencies
```bash
pip install torch-geometric torchdiffeq
```

### Data File Not Found
Ensure data files are in the correct paths:
- Processed data: `data_collection/processed_data/data_with_features.csv`
- Raw data: `data_collection/raw_data/`

## 📝 Citation

If you use this project in your research, please cite:

```bibtex
@software{ai_metallurgy_hea,
  title = {AI Applications in Metallurgy: HEA Elastic Modulus Prediction},
  author = {Nishioka, Keisuke},
  year = {2026},
  url = {https://github.com/keisuke58/AI_metallugy_HEA}
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is for research and educational purposes.

## 👤 Author

**Keisuke Nishioka**

- GitHub: [@keisuke58](https://github.com/keisuke58)
- Repository: [AI_metallugy_HEA](https://github.com/keisuke58/AI_metallugy_HEA)

## 🙏 Acknowledgments

- Gorsse et al. for the HEA mechanical properties database
- DOE/OSTI for the HEA dataset
- Materials Project for the computational materials database
- All researchers whose work inspired this project

## 📅 Last Updated

January 2026

---

**Note**: This project is part of the "AI Applications in Metallurgy" course. For course materials, see `AI_Applications_in_Metallurgy/`.
