# Otto Group Product Classification（Project 1）

9417 课程项目 — 分类任务。训练 DecisionTree（baseline）和 XGBoost，对比模型能力差异。
后续 Project 3 会在此基础上加入 xRFM 和 AGOP 特征分析。

## 数据集

Otto Group Product Classification Challenge（Kaggle）
- 61,878 条样本，93 个数值特征，9 个产品类别
- 数据位于 `data/train.csv`

## 项目结构

```
├── train.py                 # 训练入口，用法：python train.py decision_tree / xgboost
├── config.py                # 模型配置 & GridSearchCV 参数网格
├── src/
│   ├── preprocessing.py     # 数据加载、LabelEncoder、60/20/20 分层划分
│   ├── evaluation.py        # Accuracy、F1、AUC-ROC、混淆矩阵
│   ├── feature_analysis.py  # Permutation Importance（top-15 柱状图）
│   ├── reporting.py         # 结果保存为 JSON + Markdown
│   └── eda.py               # 探索性数据分析（shape、缺失值、类别分布）
├── results/                 # 自动生成的实验报告和特征重要性图
└── data/                    # 原始数据（不纳入 Git）
```

## 实验结果

| | Decision Tree (baseline) | XGBoost |
|---|---|---|
| Val Accuracy | 71.8% | 81.6% |
| Val F1 (macro) | 0.644 | 0.777 |
| Val AUC-ROC | 0.882 | 0.975 |
| Train Accuracy | 83.5% | 99.97% |
| Train-Val Gap | 11.7% | 18.4% |

- DecisionTree：max_depth=None, min_samples_leaf=4, min_samples_split=20
- XGBoost：learning_rate=0.3, max_depth=9, n_estimators=500

## 已完成

- [x] DecisionTree baseline 训练
- [x] XGBoost 对比实验
- [x] 60/20/20 分层划分
- [x] GridSearchCV 超参数搜索
- [x] 评估指标：Accuracy / F1 / AUC-ROC
- [x] Permutation Importance 特征分析（两个模型各一张图）

## 待做（Project 3 回来补）

- [ ] xRFM 模型训练
- [ ] AGOP 特征重要性提取
- [ ] AGOP vs Permutation Importance vs PCA vs MI 四方对比
