"""
analyze_features.py — 可解释性四方对比实验

四种特征重要性方法：
  1. AGOP diagonal     — xRFM 特有，梯度协方差矩阵的对角线
  2. Permutation Importance — 模型无关，基于打乱特征后 accuracy 下降
  3. PCA loadings      — 无监督，主成分方向上的特征贡献
  4. Mutual Information — 统计方法，特征与标签的互信息

用法：
  python analyze_features.py            # 跑全部四种方法
  python analyze_features.py agop       # 只跑 AGOP
  python analyze_features.py perm       # 只跑 Permutation Importance
  python analyze_features.py pca        # 只跑 PCA
  python analyze_features.py mi         # 只跑 Mutual Information
  python analyze_features.py plot       # 合并已有结果画图
"""

import sys, json
import numpy as np
import matplotlib.pyplot as plt

from src.preprocessing import load_otto_data, split_data, split_data_ontrain
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance
from pathlib import Path


RANDOM_STATE = 42
TOP_K = 10


# =====================================================================
#  四种方法
# =====================================================================

def run_agop(X_train, y_train, X_val, y_val, feature_names):
    """xRFM 的 AGOP 对角线作为特征重要性"""
    from xrfm import xRFM

    enc = OneHotEncoder(sparse_output=False)
    y_train_oh = enc.fit_transform(y_train.reshape(-1, 1)).astype(np.float32)
    y_val_oh   = enc.transform(y_val.reshape(-1, 1)).astype(np.float32)

    model = xRFM(
        device='cpu',
        tuning_metric='accuracy',
        n_tree_iters=3,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train_oh, X_val, y_val_oh)

    # 收集所有叶节点的 AGOP 矩阵，取对角线的平均
    agops = model.collect_best_agops()
    diag_sum = np.zeros(X_train.shape[1])
    for agop in agops:
        diag_sum += np.diag(agop)
    importance = diag_sum / len(agops)

    return importance


def run_perm(X_train, y_train, X_test, y_test, feature_names):
    """XGBoost 上的 Permutation Importance"""
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
    """PCA 第一主成分的 loadings（绝对值）作为特征重要性"""
    pca = PCA(n_components=1, random_state=RANDOM_STATE)
    pca.fit(X_train)

    # components_[0] 是第一主成分方向，长度 = 特征数
    importance = np.abs(pca.components_[0])
    return importance


def run_mi(X_train, y_train, feature_names):
    """Mutual Information：每个特征与标签的互信息"""
    importance = mutual_info_classif(
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
    X, y, le = load_otto_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    feature_names = list(X.columns)

    print(f"\n{'='*50}")
    print(f"Running: {method_name}")
    print(f"{'='*50}")

    if method_name == "agop":
        # xRFM 需要 val + numpy + one-hot
        X_tr, X_val, y_tr, y_val = split_data_ontrain(X_train, y_train)
        importance = run_agop(
            X_tr.values, y_tr.values,
            X_val.values, y_val.values,
            feature_names,
        )

    elif method_name == "perm":
        importance = run_perm(
            X_train, y_train,
            X_test, y_test,
            feature_names,
        )

    elif method_name == "pca":
        importance = run_pca(X_train, feature_names)

    elif method_name == "mi":
        importance = run_mi(
            X_train, y_train,
            feature_names,
        )

    else:
        raise ValueError(f"Unknown method: {method_name}")

    # top-k 排序
    top_idx = importance.argsort()[-TOP_K:][::-1]
    top_features = [feature_names[i] for i in top_idx]
    top_values = importance[top_idx].tolist()

    print(f"\nTop {TOP_K} features:")
    for rank, (name, val) in enumerate(zip(top_features, top_values), 1):
        print(f"  {rank:2d}. {name:>10s}  {val:.6f}")

    # 保存
    result = {
        "method": method_name,
        "feature_names": feature_names,
        "importance": importance.tolist(),
        "top_features": top_features,
        "top_values": top_values,
    }
    Path("results/feature_analysis").mkdir(parents=True, exist_ok=True)
    with open(f"results/feature_analysis/feature_{method_name}.json", "w") as f:
        json.dump(result, f, indent=4)

    print(f"Saved to results/feature_analysis/feature_{method_name}.json")


def plot():
    """读取各方法的结果，画四方对比图"""
    methods = ["agop", "perm", "pca", "mi"]
    labels  = ["AGOP (xRFM)", "Permutation (XGBoost)", "PCA Loadings", "Mutual Information"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, method, label in zip(axes, methods, labels):
        path = f"results/feature_analysis/feature_{method}.json"
        if not Path(path).exists():
            # 兼容旧路径
            old_path = f"results/feature_{method}.json"
            if Path(old_path).exists():
                path = old_path
            else:
                print(f"Warning: {path} not found, skipping")
                ax.set_title(f"{label} (no data)")
                continue

        with open(path) as f:
            data = json.load(f)

        top_features = data["top_features"][:TOP_K]
        top_values   = data["top_values"][:TOP_K]

        # 水平柱状图，最重要的在上方
        y_pos = range(len(top_features) - 1, -1, -1)
        ax.barh(list(y_pos), top_values)
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(top_features)
        ax.set_xlabel("Importance")
        ax.set_title(label)

    plt.tight_layout()
    Path("results/feature_analysis").mkdir(parents=True, exist_ok=True)
    plt.savefig("results/feature_analysis/feature_comparison.png", dpi=150)
    plt.close()
    print("Plot saved to results/feature_analysis/feature_comparison.png")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # 跑全部
        for m in ["agop", "perm", "pca", "mi"]:
            run_method(m)
        plot()
    elif sys.argv[1] == "plot":
        plot()
    elif sys.argv[1] in ("agop", "perm", "pca", "mi"):
        run_method(sys.argv[1])
    else:
        print(f"Unknown command: {sys.argv[1]}")
        print("Usage: python analyze_features.py [agop|perm|pca|mi|plot]")
