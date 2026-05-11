"""
feature_analysis/perm_importance.py — Permutation Importance 可视化

经典输出：箱线图（每个特征在 n_repeats 次打乱中的 accuracy 下降分布）
分布宽说明重要性不稳定，分布窄且靠右说明稳健重要。

用法：
  python feature_analysis/perm_importance.py
"""

import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing import load_otto_data, split_data
from sklearn.inspection import permutation_importance


RANDOM_STATE = 42
TOP_K = 15
N_REPEATS = 30
SAVE_DIR = Path("results/feature_analysis")


def run():
    from configs.config import MODEL_CONFIG

    print("\n" + "="*50)
    print("Running: Permutation Importance")
    print("="*50)

    X, y, le = load_otto_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    feature_names = list(X.columns)

    with open("configs/best_xgboost_multiclass.json") as f:
        best_params = json.load(f)["best_params"]

    estimator = MODEL_CONFIG["xgboost_multiclass"]["estimator"]
    estimator.set_params(**best_params)
    estimator.fit(X_train, y_train)

    result = permutation_importance(
        estimator, X_test, y_test,
        scoring='accuracy',
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    # 按均值升序取 top-k（升序方便 barh 从底到顶排列，最重要在顶）
    sorted_idx = result.importances_mean.argsort()[-TOP_K:]
    top_names = [feature_names[i] for i in sorted_idx]
    top_matrix = result.importances[sorted_idx]     # (TOP_K, N_REPEATS)

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    with open(SAVE_DIR / "perm_importance.json", "w") as f:
        json.dump({
            "method": "perm_importance",
            "feature_names": feature_names,
            "importances_mean": result.importances_mean.tolist(),
            "top_features": top_names[::-1],
            "top_matrix": top_matrix.tolist(),
        }, f, indent=4)

    # 箱线图：vert=False，sorted_idx 升序保证最重要的特征在顶部
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.boxplot(
        top_matrix.T,
        vert=False,
        labels=top_names,
        patch_artist=True,
        boxprops=dict(facecolor='steelblue', alpha=0.6),
        medianprops=dict(color='red', linewidth=1.5),
        flierprops=dict(marker='o', markersize=3, alpha=0.4),
    )
    ax.axvline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.set_xlabel("Accuracy Decrease (mean ± distribution across repeats)")
    ax.set_title(f"Permutation Importance (XGBoost) — Top {TOP_K} Features\n"
                 f"n_repeats={N_REPEATS}")
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(SAVE_DIR / "perm_importance.png", dpi=150)
    plt.close()

    print(f"\nTop {TOP_K} features (mean accuracy decrease):")
    for name, val in zip(top_names[::-1], result.importances_mean[sorted_idx[::-1]]):
        print(f"  {name:>10s}  {val:.6f}")
    print(f"\nSaved to {SAVE_DIR}/perm_importance.{{json,png}}")


if __name__ == "__main__":
    run()
