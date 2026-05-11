"""
analyze_otto_features.py — Otto Group 特征分析（三种方法）

  1. Permutation Importance — XGBoost 多分类，基于打乱特征后 accuracy 下降
  2. PCA loadings            — 无监督，第一主成分方向上的特征贡献
  3. Mutual Information      — 统计方法，特征与标签的互信息

用法：
  python analyze_otto_features.py            # 跑全部三种方法 + 画图
  python analyze_otto_features.py perm       # 只跑 Permutation Importance
  python analyze_otto_features.py pca        # 只跑 PCA
  python analyze_otto_features.py mi         # 只跑 Mutual Information
  python analyze_otto_features.py plot       # 只画图（基于已有 JSON）
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance

from src.preprocessing import load_otto_data, split_data

RANDOM_STATE = 42
TOP_K = 10


# =====================================================================
#  三种方法
# =====================================================================

def run_perm(X_train, y_train, X_test, y_test, feature_names):
    """XGBoost 多分类上的 Permutation Importance"""
    from configs.config import MODEL_CONFIG

    with open("configs/best_xgboost_multiclass.json") as f:
        best_params = json.load(f)["best_params"]

    estimator = MODEL_CONFIG["xgboost_multiclass"]["estimator"]
    estimator.set_params(**best_params)
    estimator.fit(X_train, y_train)

    result = permutation_importance(
        estimator, X_test, y_test,
        scoring='accuracy',
        n_repeats=10,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return result.importances_mean


def run_pca(X_train, feature_names):
    """PCA 第一主成分的 loadings（绝对值）"""
    pca = PCA(n_components=1, random_state=RANDOM_STATE)
    pca.fit(X_train)
    return np.abs(pca.components_[0])


def run_mi(X_train, y_train, feature_names):
    """Mutual Information：每个特征与标签的互信息"""
    return mutual_info_classif(
        X_train, y_train,
        random_state=RANDOM_STATE,
        n_neighbors=5,
    )


# =====================================================================
#  主逻辑
# =====================================================================

def run_method(method_name):
    """跑单个方法，保存结果到 JSON"""
    X, y, _ = load_otto_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    feature_names = list(X.columns)

    print(f"\n{'='*50}")
    print(f"Running: {method_name}")
    print(f"{'='*50}")

    if method_name == "perm":
        importance = run_perm(X_train, y_train, X_test, y_test, feature_names)
    elif method_name == "pca":
        importance = run_pca(X_train, feature_names)
    elif method_name == "mi":
        importance = run_mi(X_train, y_train, feature_names)
    else:
        raise ValueError(f"Unknown method: {method_name}")

    top_idx = importance.argsort()[-TOP_K:][::-1]
    top_features = [feature_names[i] for i in top_idx]
    top_values = importance[top_idx].tolist()

    print(f"\nTop {TOP_K} features:")
    for rank, (name, val) in enumerate(zip(top_features, top_values), 1):
        print(f"  {rank:2d}. {name:>10s}  {val:.6f}")

    result = {
        "method": method_name,
        "feature_names": feature_names,
        "importance": importance.tolist(),
        "top_features": top_features,
        "top_values": top_values,
    }

    save_dir = Path("results/feature_analysis")
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / f"otto_feature_{method_name}.json", "w") as f:
        json.dump(result, f, indent=4)

    print(f"Saved to results/feature_analysis/otto_feature_{method_name}.json")


def plot():
    """读取各方法结果，画三方对比图（1x3 横排）"""
    methods = ["perm", "pca", "mi"]
    labels = ["Permutation (XGBoost)", "PCA Loadings", "Mutual Information"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, method, label in zip(axes, methods, labels):
        path = f"results/feature_analysis/otto_feature_{method}.json"
        if not Path(path).exists():
            print(f"Warning: {path} not found, skipping")
            ax.set_title(f"{label} (no data)")
            continue

        with open(path) as f:
            data = json.load(f)

        top_features = data["top_features"][:TOP_K]
        top_values = data["top_values"][:TOP_K]

        y_pos = range(len(top_features) - 1, -1, -1)
        ax.barh(list(y_pos), top_values, color='steelblue')
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(top_features)
        ax.set_xlabel("Importance")
        ax.set_title(label)
        ax.grid(axis='x', alpha=0.3)

    plt.suptitle("Otto Group — Feature Importance Comparison", fontsize=14, fontweight='bold')
    plt.tight_layout()

    save_dir = Path("results/feature_analysis")
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_dir / "otto_feature_comparison.png", dpi=150)
    plt.close()
    print("Plot saved to results/feature_analysis/otto_feature_comparison.png")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        for m in ["perm", "pca", "mi"]:
            run_method(m)
        plot()
    elif sys.argv[1] == "plot":
        plot()
    elif sys.argv[1] in ("perm", "pca", "mi"):
        run_method(sys.argv[1])
    else:
        print(f"Unknown command: {sys.argv[1]}")
        print("Usage: python analyze_otto_features.py [perm|pca|mi|plot]")
