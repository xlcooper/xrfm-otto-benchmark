# Otto Group 产品分类 — ML 模型对比

基于 [Otto Group Kaggle 数据集](https://www.kaggle.com/c/otto-group-product-classification-challenge) 的多分类任务，对比 **XGBoost**、**Random Forest** 和 **xRFM** 三种模型，重点关注超参数调优、规模扩展行为和特征可解释性。

## 数据集

- 61,878 条样本，93 个匿名数值特征，9 个产品类别
- 数据位于 `data/train.csv`

## 项目结构

```
otto-group-classification-comparison/
├── data/
│   └── train.csv
├── src/                    # 核心模块
│   ├── preprocessing.py    # 数据加载、标签编码、训练/验证/测试集划分
│   ├── eda.py              # 探索性数据分析
│   ├── evaluation.py       # Accuracy、F1、AUC-ROC、混淆矩阵
│   ├── feature_analysis.py # Permutation Importance
│   └── reporting.py        # 结果保存为 JSON 和 Markdown
├── configs/
│   ├── config.py                       # 模型配置和超参数网格
│   ├── best_xgboost_multiclass.json    # XGBoost 最优参数
│   └── best_random_forest_clf.json     # Random Forest 最优参数
├── feature_analysis/       # 独立特征重要性分析脚本
│   ├── mi_analysis.py      # 互信息分析
│   ├── pca_analysis.py     # PCA 碎石图和双标图
│   └── perm_importance.py  # Permutation Importance 箱线图
├── results/                # 自动生成的实验结果
├── tune.py                 # GridSearchCV 超参数调优
├── run_main.py             # 使用最优参数评估模型
├── run_xrfm.py             # xRFM 模型评估
├── scaling_test.py         # 不同样本量下的规模扩展实验
└── analyze_features.py     # 四种特征重要性方法对比
```

## 环境配置

```bash
conda create -n otto python=3.11
conda activate otto
pip install -r requirements.txt
```

## 使用方法

### 1. 探索性数据分析

```bash
python src/eda.py
```

输出数据维度、数据类型、缺失值情况和类别分布。

### 2. 超参数调优

```bash
python tune.py xgboost_multiclass
python tune.py random_forest_clf
```

执行 5 折 GridSearchCV，将最优参数保存至 `configs/best_{model}.json`。

### 3. 模型评估

```bash
python run_main.py xgboost_multiclass
python run_main.py random_forest_clf
python run_xrfm.py
```

使用调优后的参数在全量训练集上训练，在测试集上评估，结果保存至 `results/`。

### 4. 规模扩展实验

```bash
python scaling_test.py xgboost_multiclass
python scaling_test.py random_forest_clf
python scaling_test.py xrfm
python scaling_test.py plot          # 汇总结果并可视化
```

在 1k、2k、5k、10k、20k 样本量下测试准确率和训练时间。

### 5. 特征重要性分析

```bash
python analyze_features.py            # 运行全部四种方法并绘图
python analyze_features.py agop       # xRFM AGOP
python analyze_features.py perm       # Permutation Importance（XGBoost）
python analyze_features.py pca        # PCA 主成分载荷
python analyze_features.py mi         # 互信息
python analyze_features.py plot       # 基于已有结果直接绘图
```

生成 2×2 对比图，保存至 `results/feature_analysis/`。

## 模型对比

| 模型 | Accuracy | AUC-ROC | 训练时间 | 关键超参数 |
|---|---|---|---|---|
| **XGBoost** | **0.7875** | **0.9633** | 9.61s | `lr=0.1, max_depth=9, n_estimators=500` |
| **xRFM** | 0.7720 | 0.9551 | 18.27s | 基于 PyTorch，目标变量需 one-hot 编码 |
| Random Forest | 0.7685 | 0.9614 | 0.82s | `max_depth=None, min_samples_leaf=1, n_estimators=100` |

> **结论**：XGBoost 在准确率和 AUC-ROC 上表现最优；Random Forest 训练速度最快但准确率略低；xRFM 作为基于梯度协方差的新方法，表现介于两者之间。

## 规模扩展行为

| 样本量 | XGBoost Accuracy | RF Accuracy | xRFM Accuracy |
|---|---|---|---|
| 1,000 | 0.665 | 0.670 | **0.705** |
| 2,000 | 0.705 | 0.718 | **0.713** |
| 5,000 | 0.751 | 0.736 | 0.740 |
| 10,000 | **0.788** | 0.769 | 0.772 |
| 20,000 | **0.799** | 0.773 | 0.786 |

> **观察**：XGBoost 在大数据量下优势明显；xRFM 在小样本时表现较好，但随数据量增加训练时间急剧上升（20k 样本需 182s）。

## 特征重要性方法

- **AGOP** — xRFM 梯度协方差对角线（模型专属）
- **Permutation Importance** — 模型无关，基于 XGBoost 评估
- **PCA 载荷** — 无监督线性投影
- **互信息** — 特征与目标之间的统计依赖度

## 主要依赖

- `scikit-learn==1.8.0`
- `xgboost==3.2.0`
- `xrfm==0.4.3`
- `torch==2.11.0`
- `pandas==3.0.2`、`numpy==2.4.4`
- `matplotlib==3.10.8`
