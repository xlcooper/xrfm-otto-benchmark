"""
plot_bike_heatmap.py — Bike 特征分析 Combined Heatmap

读取四种方法的结果，画一个 heatmap：
  行 = 特征（各方法 Top 特征的并集）
  列 = AGOP / Permutation / PCA / MI
  颜色 = 每个方法内部归一化到 [0, 1] 的重要性

用法：python plot_bike_heatmap.py
"""

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

METHODS = ["agop", "perm", "pca", "mi"]
LABELS = ["AGOP (xRFM)", "Permutation\n(XGBoost)", "PCA\nLoadings", "Mutual\nInformation"]
TOP_K = 15  # 每种方法取前 K 个特征


def load_results():
    """加载四种方法的结果，返回 {method: {feature: importance}}"""
    results = {}
    for method in METHODS:
        path = f"results/feature_analysis/bike_feature_{method}.json"
        with open(path) as f:
            data = json.load(f)
        results[method] = dict(zip(data["feature_names"], data["importance"]))
    return results


def build_matrix(results):
    """构建 heatmap 矩阵：行=特征，列=方法"""
    # 收集每种方法 Top-K 特征
    top_features = set()
    for method in METHODS:
        importances = list(results[method].items())
        importances.sort(key=lambda x: x[1], reverse=True)
        for name, _ in importances[:TOP_K]:
            top_features.add(name)

    features = sorted(top_features)  # 按字母排序，稳定
    n_features = len(features)
    n_methods = len(METHODS)

    # 构建矩阵并归一化（每种方法内部 min-max）
    matrix = np.zeros((n_features, n_methods))
    for j, method in enumerate(METHODS):
        vals = np.array([results[method].get(f, 0) for f in features])
        vmin, vmax = vals.min(), vals.max()
        if vmax > vmin:
            vals = (vals - vmin) / (vmax - vmin)
        matrix[:, j] = vals

    return features, matrix


def plot(features, matrix):
    """画 heatmap"""
    fig, ax = plt.subplots(figsize=(8, max(6, len(features) * 0.35)))

    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)

    # 坐标轴
    ax.set_xticks(range(len(METHODS)))
    ax.set_xticklabels(LABELS, fontsize=10)
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features, fontsize=9)

    # 在每个格子里写数值
    for i in range(len(features)):
        for j in range(len(METHODS)):
            text = ax.text(
                j, i, f"{matrix[i, j]:.2f}",
                ha="center", va="center",
                color="black" if matrix[i, j] < 0.6 else "white",
                fontsize=8,
            )

    ax.set_title(
        "Bike Sharing — Feature Importance Heatmap\n"
        "(normalized per method)",
        fontsize=12, fontweight="bold", pad=15,
    )

    # colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.6)
    cbar.set_label("Normalized Importance", rotation=270, labelpad=20)

    plt.tight_layout()

    save_path = "results/feature_analysis/bike_feature_heatmap.png"
    Path("results/feature_analysis").mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Heatmap saved: {save_path}")


def main():
    results = load_results()
    features, matrix = build_matrix(results)
    plot(features, matrix)
    print(f"Features in heatmap: {len(features)}")


if __name__ == "__main__":
    main()
