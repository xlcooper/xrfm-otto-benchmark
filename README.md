# xRFM Reproduction & Comparison

This project reproduces the feature importance method from the paper **xRFM: Gradient Covariance Feature Importance** and compares it against traditional algorithms (**XGBoost**, **Random Forest**) on tabular classification data.

## Project Overview

xRFM is a novel feature importance method based on gradient covariance. This project:

1. **Reads and reproduces** the core xRFM method (AGOP gradient-covariance feature importance)
2. **Runs the xRFM library** (PyTorch backend) on tabular classification data
3. **Compares with classical algorithms**: XGBoost and Random Forest on the same dataset
4. **Hyperparameter tuning**: GridSearchCV for XGBoost and Random Forest
5. **Scaling analysis**: Accuracy and training time across different sample sizes
6. **Feature importance comparison**: AGOP, Permutation Importance, PCA, and Mutual Information

## Current Dataset

**Otto Group** — A classic multi-class classification dataset from Kaggle

- **Original size**: 61,878 samples
- **Used in this project**: 20,000 samples (stratified random subsample, see `src/preprocess_otto.py`)
- **Features**: 93 anonymous numerical features (no missing values)
- **Classes**: 9 product categories (class-imbalanced)
- **Characteristics**: High-dimensional feature space, anonymous features, uneven class distribution

> **Note**: The original 61K dataset is preserved at `data/train_full.csv`. Preprocessing script: `python src/preprocess_otto.py`

## Feature Work

- [ ] **Extend to more tabular datasets**: UCI ML Repository, other Kaggle classification datasets (e.g., Covertype, Santander Customer Transaction)
- [ ] **Validate xRFM performance across different data scales**
- [ ] **Compare with more baselines**: LightGBM, CatBoost, TabNet, etc.
- [ ] **Feature selection experiments**: Use xRFM AGOP for feature selection and observe impact on model performance

## Project Structure

```
xrfm-otto-benchmark/
├── data/
│   └── train.csv
├── src/                    # Core modules
│   ├── preprocessing.py    # Data loading, label encoding, train/val/test split
│   ├── eda.py              # Exploratory data analysis
│   ├── evaluation.py       # Accuracy, F1, AUC-ROC, confusion matrix
│   ├── feature_analysis.py # Permutation Importance
│   └── reporting.py        # Save results as JSON and Markdown
├── configs/
│   ├── config.py                       # Model configs and hyperparameter grids
│   ├── best_xgboost_multiclass.json    # XGBoost optimal params
│   └── best_random_forest_clf.json     # Random Forest optimal params
├── feature_analysis/       # Standalone feature importance scripts
│   ├── mi_analysis.py      # Mutual Information analysis
│   ├── pca_analysis.py     # PCA scree plot and biplot
│   └── perm_importance.py  # Permutation Importance boxplot
├── results/                # Auto-generated experiment results
├── tune.py                 # GridSearchCV hyperparameter tuning
├── run_main.py             # Evaluate with optimal parameters
├── run_xrfm.py             # xRFM model evaluation
├── scaling_test.py         # Scaling experiments at varying sample sizes
└── analyze_features.py     # Four-way feature importance comparison
```

## Environment Setup

```bash
conda create -n otto python=3.11
conda activate otto
pip install -r requirements.txt
```

## Usage

### 1. Exploratory Data Analysis

```bash
python src/eda.py
```

### 2. Hyperparameter Tuning

```bash
python tune.py xgboost_multiclass
python tune.py random_forest_clf
```

### 3. Model Evaluation

```bash
python run_main.py xgboost_multiclass
python run_main.py random_forest_clf
python run_xrfm.py
```

### 4. Scaling Experiments

```bash
python scaling_test.py xgboost_multiclass
python scaling_test.py random_forest_clf
python scaling_test.py xrfm
python scaling_test.py plot
```

### 5. Feature Importance Analysis

```bash
python analyze_features.py            # Run all four methods and plot
python analyze_features.py agop       # xRFM AGOP
python analyze_features.py perm       # Permutation Importance (XGBoost)
python analyze_features.py pca        # PCA component loadings
python analyze_features.py mi         # Mutual Information
python analyze_features.py plot       # Plot from existing results
```

## Results

实验结果分为三个部分：主实验、Scaling 实验、特征分析。

### 1. Main Experiment (Otto Group, 20K samples)

| Model | Accuracy | AUC-ROC | Training Time | Key Hyperparameters |
|---|---|---|---|---|
| **XGBoost** | **0.8025** | **0.9715** | 10.95s | `lr=0.1, max_depth=9, n_estimators=500` |
| **xRFM** | 0.7965 | 0.9587 | 100.59s | Calls xRFM library (PyTorch backend), requires one-hot encoding |
| Random Forest | 0.7883 | 0.9655 | 1.39s | `max_depth=None, min_samples_leaf=1, n_estimators=100` |

> **Conclusion**: XGBoost achieves the best accuracy and AUC-ROC; xRFM is competitive on accuracy but training time is significantly higher (~10× XGBoost); Random Forest is the fastest but slightly less accurate.

**可视化**: `results/main_experiment_comparison.png`

### 2. Scaling Experiment

> Based on the **Otto Group** dataset (9-class classification, 93 features, 20K samples). We subsample at 1K / 2K / 5K / 10K / 20K to observe scaling trends.

| Sample Size | XGBoost Accuracy | RF Accuracy | xRFM Accuracy |
|---|---|---|---|
| 1,000 | 0.665 | 0.670 | **0.705** |
| 2,000 | 0.705 | 0.718 | **0.713** |
| 5,000 | 0.751 | 0.736 | 0.740 |
| 10,000 | **0.788** | 0.769 | 0.772 |
| 20,000 | **0.799** | 0.773 | 0.786 |

> **Observation**: XGBoost dominates at large data sizes; xRFM performs better on small samples but training time increases sharply (182s at 20k samples).

**结果文件**: `results/scaling/`

### 3. Feature Importance Analysis

> Based on the **Otto Group** dataset (9-class classification, 93 features, 61K samples). We subsample at 1K / 2K / 5K / 10K / 20K to observe scaling trends.

| Sample Size | XGBoost Accuracy | RF Accuracy | xRFM Accuracy |
|---|---|---|---|
| 1,000 | 0.665 | 0.670 | **0.705** |
| 2,000 | 0.705 | 0.718 | **0.713** |
| 5,000 | 0.751 | 0.736 | 0.740 |
| 10,000 | **0.788** | 0.769 | 0.772 |
| 20,000 | **0.799** | 0.773 | 0.786 |

> **Observation**: XGBoost dominates at large data sizes; xRFM performs better on small samples but training time increases sharply (182s at 20k samples).

## Feature Importance Methods

- **AGOP** — xRFM gradient-covariance diagonal (xRFM model-specific)
- **Permutation Importance** — Shuffle feature values and observe performance drop; model-agnostic
- **PCA Loadings** — Principal Component Analysis, unsupervised linear projection
- **Mutual Information** — Statistical dependency between feature and target

## Main Dependencies

- `scikit-learn==1.8.0`
- `xgboost==3.2.0`
- `xrfm==0.4.3`
- `torch==2.11.0`
- `pandas==3.0.2`, `numpy==2.4.4`
- `matplotlib==3.10.8`
