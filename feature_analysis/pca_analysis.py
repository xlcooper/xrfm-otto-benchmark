"""
feature_analysis/pca_analysis.py — PCA 可视化

经典输出：
  1. 碎石图（Scree Plot）— 各主成分解释方差 + 累计曲线，判断有效成分数
  2. 双标图（Biplot）   — 样本在 PC1/PC2 平面散点（按类着色）+ 特征 loading 向量

用法：
  python m feature_analysis/pca_analysis.py
"""

import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing import load_otto_data, split_data
from sklearn.decomposition import PCA


RANDOM_STATE = 42
N_COMPONENTS = 20
TOP_K_LOADINGS = 10     # biplot 中展示的特征箭头数
SCATTER_N = 2000        # biplot 散点子采样，避免过密
SAVE_DIR = Path("results/feature_analysis")


def run():
    print("\n" + "="*50)
    print("Running: PCA Analysis")
    print("="*50)

    X, y, le = load_otto_data()
    X_train, _, y_train, _ = split_data(X, y)
    feature_names = list(X.columns)

    pca = PCA(n_components=N_COMPONENTS, random_state=RANDOM_STATE)
    pca.fit(X_train)

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    with open(SAVE_DIR / "pca_analysis.json", "w") as f:
        json.dump({
            "method": "pca",
            "n_components": N_COMPONENTS,
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "components": pca.components_.tolist(),
            "feature_names": feature_names,
        }, f, indent=4)

    evr = pca.explained_variance_ratio_
    cumulative = np.cumsum(evr)

    # --- 1. 碎石图 ---
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.bar(range(1, N_COMPONENTS + 1), evr * 100, alpha=0.7, color='steelblue', label='Individual')
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Explained Variance (%)")
    ax1.set_title("PCA Scree Plot")
    ax1.set_xticks(range(1, N_COMPONENTS + 1))

    ax2 = ax1.twinx()
    ax2.plot(range(1, N_COMPONENTS + 1), cumulative * 100, 'r.-', label='Cumulative')
    ax2.set_ylabel("Cumulative Explained Variance (%)")
    # 标注 80% 参考线
    ax2.axhline(80, color='gray', linestyle='--', linewidth=0.8, alpha=0.7, label='80% threshold')
    ax2.set_ylim(0, 105)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')

    plt.tight_layout()
    plt.savefig(SAVE_DIR / "pca_scree.png", dpi=150)
    plt.close()

    # --- 2. 双标图 ---
    X_proj = pca.transform(X_train.values)

    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(len(X_proj), size=min(SCATTER_N, len(X_proj)), replace=False)
    X_sub, y_sub = X_proj[idx], y_train.values[idx]

    # 取在 PC1/PC2 平面上 loading 向量最长的特征
    loadings = pca.components_[:2].T        # (n_features, 2)
    loading_norms = np.linalg.norm(loadings, axis=1)
    top_feat_idx = loading_norms.argsort()[-TOP_K_LOADINGS:][::-1]

    scale = np.max(np.abs(X_sub[:, :2])) / np.max(loading_norms) * 0.8

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        X_sub[:, 0], X_sub[:, 1],
        c=y_sub, cmap='tab10', alpha=0.3, s=8,
    )
    plt.colorbar(scatter, ax=ax, label='Class')

    for i in top_feat_idx:
        ax.arrow(
            0, 0,
            loadings[i, 0] * scale, loadings[i, 1] * scale,
            head_width=0.15, head_length=0.1,
            fc='red', ec='red', alpha=0.8,
        )
        ax.text(
            loadings[i, 0] * scale * 1.12,
            loadings[i, 1] * scale * 1.12,
            feature_names[i],
            fontsize=8, color='darkred', ha='center',
        )

    ax.axhline(0, color='k', linewidth=0.5, alpha=0.4)
    ax.axvline(0, color='k', linewidth=0.5, alpha=0.4)
    ax.set_xlabel(f"PC1 ({evr[0]*100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({evr[1]*100:.1f}% variance)")
    ax.set_title(f"PCA Biplot — Top {TOP_K_LOADINGS} feature loadings")
    plt.tight_layout()
    plt.savefig(SAVE_DIR / "pca_biplot.png", dpi=150)
    plt.close()

    print(f"\nPC1 explains {evr[0]*100:.1f}%, PC2 {evr[1]*100:.1f}%")
    n80 = int(np.searchsorted(cumulative, 0.80)) + 1
    print(f"{n80} components needed to reach 80% variance")
    print(f"\nSaved to {SAVE_DIR}/pca_{{scree,biplot}}.png + pca_analysis.json")


if __name__ == "__main__":
    run()
