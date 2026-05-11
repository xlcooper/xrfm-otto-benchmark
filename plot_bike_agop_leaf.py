"""
plot_bike_agop_leaf.py — Bike xRFM Leaf-Level AGOP Heatmap

每个叶节点一个 AGOP 矩阵，取对角线得到该叶节点的特征重要性分布。
Heatmap：行=叶节点，列=特征，颜色=对角线值（已做 min-max 归一化）。

用法：python plot_bike_agop_leaf.py
"""

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from src.preprocessing import load_bike_data, split_data_ontrain

RANDOM_STATE = 42


def run_agop_leaf(X_train, y_train, X_val, y_val, feature_names):
    """返回每个叶节点的 AGOP 对角线矩阵 (n_leaves, n_features)"""
    from xrfm import xRFM

    model = xRFM(
        device='cpu',
        tuning_metric='mse',
        n_tree_iters=1,
        n_threads=1,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train, X_val, y_val)

    agops = model.collect_best_agops()
    diags = np.array([np.diag(a) for a in agops])  # (n_leaves, n_features)

    return diags


def plot_leaf_heatmap(diags, feature_names):
    """画 leaf-level AGOP heatmap"""
    n_leaves, n_features = diags.shape

    # 每行（每个叶节点）内部 min-max 归一化
    diags_norm = np.zeros_like(diags)
    for i in range(n_leaves):
        row = diags[i]
        vmin, vmax = row.min(), row.max()
        if vmax > vmin:
            diags_norm[i] = (row - vmin) / (vmax - vmin)
        else:
            diags_norm[i] = row  # 全 0 或常数行

    # 只展示有实际变化的特征（去掉全 0 列）
    active_cols = diags_norm.max(axis=0) > 0.01
    diags_plot = diags_norm[:, active_cols]
    features_plot = [f for f, ok in zip(feature_names, active_cols) if ok]

    fig, ax = plt.subplots(figsize=(10, max(4, n_leaves * 0.6)))
    im = ax.imshow(diags_plot, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(features_plot)))
    ax.set_xticklabels(features_plot, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n_leaves))
    ax.set_yticklabels([f"Leaf {i+1}" for i in range(n_leaves)], fontsize=9)

    # 格子数值
    for i in range(n_leaves):
        for j in range(len(features_plot)):
            ax.text(
                j, i, f"{diags_plot[i, j]:.2f}",
                ha="center", va="center",
                color="black" if diags_plot[i, j] < 0.5 else "white",
                fontsize=6,
            )

    ax.set_title(
        f"Bike xRFM — Leaf-Level AGOP Diagonal\n"
        f"({n_leaves} leaves, {len(features_plot)} active features)",
        fontsize=12, fontweight="bold", pad=15,
    )

    cbar = plt.colorbar(im, ax=ax, shrink=0.6)
    cbar.set_label("Normalized AGOP (per leaf)", rotation=270, labelpad=20)

    plt.tight_layout()

    save_path = "results/feature_analysis/bike_agop_leaf_heatmap.png"
    Path("results/feature_analysis").mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Leaf heatmap saved: {save_path}")


def save_leaf_table(diags, feature_names):
    """保存 leaf-level AGOP 为 JSON（原始值 + 归一化值）"""
    n_leaves = diags.shape[0]

    # 归一化
    diags_norm = np.zeros_like(diags)
    for i in range(n_leaves):
        row = diags[i]
        vmin, vmax = row.min(), row.max()
        if vmax > vmin:
            diags_norm[i] = (row - vmin) / (vmax - vmin)

    result = {
        "n_leaves": n_leaves,
        "feature_names": feature_names,
        "leaf_raw": {f"leaf_{i+1}": diags[i].tolist() for i in range(n_leaves)},
        "leaf_normalized": {f"leaf_{i+1}": diags_norm[i].tolist() for i in range(n_leaves)},
    }

    save_path = "results/feature_analysis/bike_agop_leaf_table.json"
    with open(save_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Leaf table saved: {save_path}")


def main():
    X, y, _, feature_names = load_bike_data()

    # 划分 60/20/20（和主实验一致）
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, train_size=0.6, random_state=RANDOM_STATE,
    )
    X_test, _, y_test, _ = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE,
    )

    # xRFM 需要 val
    X_tr, X_val, y_tr, y_val = split_data_ontrain(X_train, y_train)

    print(f"📊 Fitting xRFM... Train={X_tr.shape}, Val={X_val.shape}")
    diags = run_agop_leaf(X_tr, y_tr, X_val, y_val, feature_names)
    print(f"   Collected {diags.shape[0]} leaves, {diags.shape[1]} features")

    plot_leaf_heatmap(diags, feature_names)
    save_leaf_table(diags, feature_names)


if __name__ == "__main__":
    main()
