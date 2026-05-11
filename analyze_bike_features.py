"""
analyze_bike_features.py — Bike Sharing 回归特征分析

四种方法：
  1. AGOP diagonal     — xRFM 回归特有，梯度协方差矩阵对角线
  2. Permutation Importance — XGBoost Regressor，基于打乱特征后 R² 下降
  3. PCA loadings      — 无监督，主成分方向上的特征贡献
  4. Mutual Information — mutual_info_regression，特征与 cnt 的互信息

用法：
  python analyze_bike_features.py            # 跑全部四种方法
  python analyze_bike_features.py agop       # 只跑 AGOP
  python analyze_bike_features.py perm       # 只跑 Permutation Importance
  python analyze_bike_features.py pca        # 只跑 PCA
  python analyze_bike_features.py mi         # 只跑 Mutual Information
  python analyze_bike_features.py plot       # 合并已有结果画图
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

from src.preprocessing import load_bike_data, split_data, split_data_ontrain

RANDOM_STATE = 42
TOP_K = 10


# =====================================================================
#  四种方法
# =====================================================================

def run_agop(X_train, y_train, X_val, y_val, feature_names):
    """xRFM 回归的 AGOP 对角线作为特征重要性"""
    from xrfm import xRFM

    model = xRFM(
        device='cpu',
        tuning_metric='mse',
        n_tree_iters=1,
        n_threads=1,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train, X_val, y_val)

    # 收集所有叶节点的 AGOP 矩阵，取对角线的平均
    agops = model.collect_best_agops()
    diag_sum = np.zeros(X_train.shape[1])
    for agop in agops:
        diag_sum += np.diag(agop)
    importance = diag_sum / len(agops)

    return importance


def run_perm(X_train, y_train, X_test, y_test, feature_names):
    """XGBoost Regressor 上的 Permutation Importance"""
    from configs.config import MODEL_CONFIG

    with open("configs/best_xgboost_regressor.json") as f:
        best_params = json.load(f)

    estimator = MODEL_CONFIG["xgboost_regressor"]["estimator"]
    estimator.set_params(**best_params)
    estimator.fit(X_train, y_train)

    result = permutation_importance(
        estimator, X_test, y_test,
        scoring='r2',
        n_repeats=10,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return result.importances_mean


def run_pca(X_train, feature_names):
    """PCA 第一主成分的 loadings（绝对值）作为特征重要性"""
    pca = PCA(n_components=1, random_state=RANDOM_STATE)
    pca.fit(X_train)

    importance = np.abs(pca.components_[0])
    return importance


def run_mi(X_train, y_train, feature_names):
    """Mutual Information：每个特征与 cnt 的互信息"""
    importance = mutual_info_regression(
        X_train, y_train,
        random_state=RANDOM_STATE,
        n_neighbors=5,
    )
    return importance


# =====================================================================
#  主逻辑
# =====================================================================

def run_method(method_name):
    """跑单个方法，保存结果到 JSON"""
    X, y, y_cnt, feature_names = load_bike_data()

    # 划分 60/20/20（和主实验一致）
    X_train, X_temp, y_train, y_temp, y_cnt_train, y_cnt_temp = train_test_split(
        X, y, y_cnt,
        train_size=0.6,
        random_state=RANDOM_STATE,
    )
    X_test, _, y_test, _, y_cnt_test, _ = train_test_split(
        X_temp, y_temp, y_cnt_temp,
        test_size=0.5,
        random_state=RANDOM_STATE,
    )

    print(f"\n{'='*50}")
    print(f"Running: {method_name}")
    print(f"{'='*50}")

    if method_name == "agop":
        # xRFM 需要 val
        X_tr, X_val, y_tr, y_val = split_data_ontrain(X_train, y_train)
        importance = run_agop(X_tr, y_tr, X_val, y_val, feature_names)

    elif method_name == "perm":
        # PI 用单目标 cnt（避免多目标 scoring 复杂性）
        importance = run_perm(
            X_train, y_cnt_train,
            X_test, y_cnt_test,
            feature_names,
        )

    elif method_name == "pca":
        importance = run_pca(X_train, feature_names)

    elif method_name == "mi":
        # MI 也用单目标 cnt
        importance = run_mi(X_train, y_cnt_train, feature_names)

    else:
        raise ValueError(f"Unknown method: {method_name}")

    # top-k 排序
    top_idx = importance.argsort()[-TOP_K:][::-1]
    top_features = [feature_names[i] for i in top_idx]
    top_values = importance[top_idx].tolist()

    print(f"\nTop {TOP_K} features:")
    for rank, (name, val) in enumerate(zip(top_features, top_values), 1):
        print(f"  {rank:2d}. {name:>20s}  {val:.6f}")

    # 保存
    result = {
        "method": method_name,
        "feature_names": feature_names,
        "importance": importance.tolist(),
        "top_features": top_features,
        "top_values": top_values,
    }
    Path("results/feature_analysis").mkdir(parents=True, exist_ok=True)
    with open(f"results/feature_analysis/bike_feature_{method_name}.json", "w") as f:
        json.dump(result, f, indent=4)

    print(f"Saved to results/feature_analysis/bike_feature_{method_name}.json")


def plot():
    """读取各方法的结果，画四方对比图"""
    methods = ["agop", "perm", "pca", "mi"]
    labels = ["AGOP (xRFM)", "Permutation (XGBoost)", "PCA Loadings", "Mutual Information"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, method, label in zip(axes, methods, labels):
        path = f"results/feature_analysis/bike_feature_{method}.json"
        if not Path(path).exists():
            print(f"Warning: {path} not found, skipping")
            ax.set_title(f"{label} (no data)")
            continue

        with open(path) as f:
            data = json.load(f)

        top_features = data["top_features"][:TOP_K]
        top_values = data["top_values"][:TOP_K]

        y_pos = range(len(top_features) - 1, -1, -1)
        ax.barh(list(y_pos), top_values)
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(top_features)
        ax.set_xlabel("Importance")
        ax.set_title(label)

    plt.suptitle("Bike Sharing — Feature Importance Comparison", fontsize=14, fontweight='bold')
    plt.tight_layout()
    Path("results/feature_analysis").mkdir(parents=True, exist_ok=True)
    plt.savefig("results/feature_analysis/bike_feature_comparison.png", dpi=150)
    plt.close()
    print("Plot saved to results/feature_analysis/bike_feature_comparison.png")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        for m in ["agop", "perm", "pca", "mi"]:
            run_method(m)
        plot()
    elif sys.argv[1] == "plot":
        plot()
    elif sys.argv[1] in ("agop", "perm", "pca", "mi"):
        run_method(sys.argv[1])
    else:
        print(f"Unknown command: {sys.argv[1]}")
        print("Usage: python analyze_bike_features.py [agop|perm|pca|mi|plot]")
