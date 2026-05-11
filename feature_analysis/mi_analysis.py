"""
feature_analysis/mi_analysis.py — Mutual Information 可视化

经典输出：水平柱状图，颜色渐变按分数排序，直观展示特征与标签的统计依赖强度

用法：
  python feature_analysis/mi_analysis.py
"""

import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing import load_otto_data, split_data
from sklearn.feature_selection import mutual_info_classif


RANDOM_STATE = 42
TOP_K = 20
SAVE_DIR = Path("results/feature_analysis")


def run():
    print("\n" + "="*50)
    print("Running: Mutual Information")
    print("="*50)

    X, y, le = load_data()
    X_train, _, y_train, _ = split_data(X, y)
    feature_names = list(X.columns)

    importance = mutual_info_classif(
        X_train, y_train,
        random_state=RANDOM_STATE,
        n_neighbors=5,
    )

    # 升序取 top-k，barh 自底向顶，最重要在顶
    sorted_idx = importance.argsort()[-TOP_K:]
    top_names = [feature_names[i] for i in sorted_idx]
    top_values = importance[sorted_idx]

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    with open(SAVE_DIR / "mi_analysis.json", "w") as f:
        json.dump({
            "method": "mi",
            "feature_names": feature_names,
            "importance": importance.tolist(),
            "top_features": top_names[::-1],
            "top_values": top_values[::-1].tolist(),
        }, f, indent=4)

    colors = cm.Blues(np.linspace(0.35, 0.9, TOP_K))
    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(range(TOP_K), top_values, color=colors)

    # 在每个柱右侧标注数值
    for bar, val in zip(bars, top_values):
        ax.text(
            bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va='center', fontsize=8,
        )

    ax.set_yticks(range(TOP_K))
    ax.set_yticklabels(top_names)
    ax.set_xlabel("Mutual Information Score")
    ax.set_title(f"Mutual Information — Top {TOP_K} Features")
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(SAVE_DIR / "mi_analysis.png", dpi=150)
    plt.close()

    print(f"\nTop {TOP_K} features by MI:")
    for name, val in zip(top_names[::-1], top_values[::-1]):
        print(f"  {name:>10s}  {val:.6f}")
    print(f"\nSaved to {SAVE_DIR}/mi_analysis.{{json,png}}")


if __name__ == "__main__":
    run()
