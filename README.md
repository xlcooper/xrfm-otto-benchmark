# xRFM Benchmark: Classification & Regression

Reproduction and comparison of **xRFM** (Gradient Covariance Feature Importance) against classical algorithms (**XGBoost**, **Random Forest**) on tabular data, covering both **classification** and **regression** tasks.

## Project Overview

1. **Classification benchmark** — Otto Group (20K samples, 9 classes)
2. **Regression benchmark** — Bike Sharing (731 samples, multi-target)
3. **Hyperparameter tuning** — GridSearchCV for XGBoost and Random Forest
4. **Scaling analysis** — Accuracy and training time across sample sizes (Otto only)
5. **Feature importance comparison** — AGOP, Permutation Importance, PCA, Mutual Information

## Datasets

### Otto Group (Classification)
- **Original size**: 61,878 samples
- **Used**: 20,000 (stratified subsample)
- **Features**: 93 anonymous numerical features
- **Classes**: 9 product categories

### Bike Sharing (Regression)
- **Source**: UCI Bike Sharing (day.csv)
- **Samples**: 731
- **Features**: 32 (after One-Hot encoding + time engineering)
- **Target**: `[casual, registered]` (multi-target, evaluated as `cnt = casual + registered`)
- **Note**: hour.csv (17,379) cannot be used because xRFM regression triggers **SIGSEGV on Apple Silicon** with >2K samples (see Limitations)

## Project Structure

```
xrfm-otto-benchmark/
├── data/
│   ├── train.csv              # Otto (20K)
│   ├── train_full.csv         # Otto (61K original)
│   ├── bike_X.csv             # Bike features
│   ├── bike_y.csv             # Bike targets [casual, registered]
│   └── bike_y_cnt.csv         # Bike cnt reference
├── src/
│   ├── preprocessing.py       # Generic load & split
│   ├── otto_sampler.py        # Otto 61K → 20K sampling
│   ├── bike_engineer.py       # Bike feature engineering (One-Hot + time)
│   ├── evaluation.py          # Classification & regression metrics
│   └── reporting.py           # Plotting ( Otto / Bike comparisons )
├── configs/
│   ├── config.py              # Model configs & param grids
│   ├── best_xgboost_multiclass.json
│   ├── best_random_forest_clf.json
│   ├── best_xgboost_regressor.json
│   └── best_random_forest_regressor.json
├── results/
│   ├── otto_main/             # Otto classification results
│   ├── bike_main/             # Bike regression results
│   ├── scaling/               # Otto scaling experiment
│   └── feature_analysis/      # Feature importance results
├── tune_otto.py               # Otto hyperparameter tuning
├── tune_bike.py               # Bike hyperparameter tuning
├── run_otto_main.py           # Otto XGBoost / RF evaluation
├── run_otto_xrfm.py           # Otto xRFM evaluation
├── run_bike_main.py           # Bike XGBoost / RF evaluation
├── run_bike_xrfm.py           # Bike xRFM evaluation
├── scaling_otto.py            # Otto scaling experiment
├── analyze_otto_features.py   # Otto feature analysis (perm / pca / mi)
├── analyze_bike_features.py   # Bike feature analysis (agop / perm / pca / mi)
├── plot_bike_heatmap.py       # Bike combined feature importance heatmap
└── plot_bike_agop_leaf.py     # Bike leaf-level AGOP (single leaf, see Limitations)
```

## Environment

```bash
conda activate ML   # ~/miniconda3/envs/ML
# Key packages: xrfm==0.4.3, xgboost, scikit-learn, torch, pandas, matplotlib
```

## Usage

### Otto (Classification)

```bash
# 1. Tune
python tune_otto.py xgboost_multiclass
python tune_otto.py random_forest_clf

# 2. Evaluate
python run_otto_main.py xgboost_multiclass
python run_otto_main.py random_forest_clf
python run_otto_xrfm.py

# 3. Scaling
python scaling_otto.py xgboost_multiclass
python scaling_otto.py random_forest_clf
python scaling_otto.py xrfm
python scaling_otto.py plot

# 4. Feature analysis
python analyze_otto_features.py              # all methods + plot
python analyze_otto_features.py perm         # single method
python analyze_otto_features.py plot         # plot only
```

### Bike (Regression)

```bash
# 1. Prepare data (run once)
python src/bike_engineer.py

# 2. Tune
python tune_bike.py xgboost_regressor
python tune_bike.py random_forest_regressor

# 3. Evaluate
python run_bike_main.py xgboost_regressor
python run_bike_main.py random_forest_regressor
python run_bike_xrfm.py

# 4. Feature analysis
python analyze_bike_features.py              # all 4 methods + plot
python analyze_bike_features.py agop         # single method

# 5. Combined heatmap
python plot_bike_heatmap.py
```

## Results

### Otto Group — Classification

| Model | Accuracy | AUC-ROC | Train Time |
|---|---|---|---|
| **XGBoost** | **0.8025** | **0.9715** | 10.95s |
| xRFM | 0.7965 | 0.9587 | 100.59s |
| Random Forest | 0.7883 | 0.9655 | 1.39s |

### Bike Sharing — Regression

| Model | R² | MSE | MAE |
|---|---|---|---|
| **XGBoost** | **0.8922** | 430,993 | 472 |
| xRFM | 0.8897 | 441,047 | 510 |
| Random Forest | 0.8703 | 518,894 | 524 |

> XGBoost improvement after tuning is significant (+0.030 R²) compared to untuned baseline.

### Scaling (Otto)

| Samples | XGBoost | RF | xRFM |
|---|---|---|---|
| 1K | 0.665 | 0.670 | **0.705** |
| 2K | 0.705 | 0.718 | **0.713** |
| 5K | **0.751** | 0.736 | 0.740 |
| 10K | **0.788** | 0.769 | 0.772 |
| 20K | **0.799** | 0.773 | 0.786 |

## Feature Importance

### Otto — Top 3 Consensus
| Method | #1 | #2 | #3 |
|---|---|---|---|
| Permutation (XGBoost) | feat_34 | feat_11 | feat_60 |
| PCA | feat_24 | feat_90 | feat_67 |
| Mutual Information | feat_11 | feat_14 | feat_25 |

### Bike — Top 3 Consensus
| Method | #1 | #2 | #3 |
|---|---|---|---|
| AGOP (xRFM) | temp | atemp | hum |
| Permutation (XGBoost) | yr_1 | temp | hum |
| PCA | month | season_4 | season_2 |
| Mutual Information | atemp | temp | month |

**Combined heatmap**: `results/feature_analysis/bike_feature_heatmap.png`

## Limitations & Further Work

1. **xRFM regression crashes on large datasets (>2K samples) on Apple Silicon**
   - `hour.csv` (17,379 samples) triggers **SIGSEGV** during xRFM regression fit
   - Root cause: likely MPS backend memory issue in xRFM 0.4.3 regression branch
   - **Workaround**: Use `day.csv` (731 samples) with `n_tree_iters=1, n_threads=1`
   - **Future**: Test on Linux CUDA or wait for upstream fix

2. **Leaf-level AGOP analysis not meaningful on Bike**
   - xRFM regression on 731 samples produces **only 1 leaf** (tree cannot split)
   - `plot_bike_agop_leaf.py` exists but output is trivial
   - **Future**: Run on larger regression dataset (e.g., hour.csv on cloud GPU, or a different benchmark) to get multi-leaf AGOP comparison

3. **Hyperparameter search space**
   - Current grids are coarse (3×3×3 = 27 combos)
   - **Future**: Bayesian optimization (Optuna) for fine-grained search

4. **More datasets**
   - Only 2 datasets (1 classification, 1 small regression)
   - **Future**: Covertype, California Housing, Santander, etc.

## Dependencies

- `scikit-learn>=1.8.0`
- `xgboost>=2.1.0`
- `xrfm==0.4.3`
- `torch>=2.0.0`
- `pandas`, `numpy`, `matplotlib`
