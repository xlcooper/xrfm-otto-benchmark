from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
from pathlib import Path


def run_permutation_importance(model, X_val, y_val,
                                feature_names, model_name,
                                n_repeats=10, random_state=42, save_dir="results"):
    """
    对每个特征：打乱该列的值 → 重新预测 → 看 accuracy 掉多少。
    掉得越多，说明模型越依赖这个特征。
    """

    # 核心调用：对 93 个特征逐一打乱，每个特征重复 n_repeats 次取平均
    # scoring='accuracy' → 用 accuracy 下降幅度作为重要性度量
    # 总共会跑 93 × n_repeats = 930 次 predict
    result = permutation_importance(
        model, X_val, y_val,
        random_state=random_state,
        n_repeats=n_repeats,
        scoring='accuracy',
        n_jobs=-1,
    )

    # result.importances_mean: 长度 93 的数组，每个值 = 该特征被打乱后 accuracy 的平均下降量
    importances = result.importances_mean

    # 取重要性最高的 15 个特征（argsort 升序排列，取最后 15 个再翻转为降序）
    top_idx = importances.argsort()[-15:][::-1]
    top_names = [feature_names[i] for i in top_idx]
    top_values = importances[top_idx]

    # 画水平柱状图，最重要的特征在最上方
    plt.figure(figsize=(10, 6))
    plt.barh(range(14, -1, -1), top_values)
    plt.yticks(range(14, -1, -1), top_names)
    plt.xlabel("Mean Accuracy Decrease")
    plt.title(f"Permutation Importance - {model_name}")
    plt.tight_layout()

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    plt.savefig(Path(save_dir) / f"{model_name}_permutation_importance.png")
    plt.close()  # 释放图形对象，避免内存泄漏

    return top_names, top_values