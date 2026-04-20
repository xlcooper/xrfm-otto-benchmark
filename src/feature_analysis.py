from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
from pathlib import Path

def run_permutation_importance(model, X_val, y_val,
                                n_repeats=10, random_state=42, feature_names, model_name, save_dir="results"):
    result = permutation_importance(
        model, X_val, y_val,
        random_state=random_state,
        n_repeats=n_repeats,
        scoring='accuracy',
        n_jobs=-1,
    )


    # 取 top 15
    importances = result.importances_mean
    top_idx = importances.argsort()[-15:][::-1]
    top_names = [feature_names[i] for i in top_idx]
    top_values = importances[top_idx]

    # 画图
    plt.figure(figsize=(10, 6))
    plt.barh(range(14, -1, -1), top_values)
    plt.yticks(range(14, -1, -1), top_names)
    plt.xlabel("Mean Accuracy Decrease")
    plt.title(f"Permutation Importance - {model_name}")
    plt.tight_layout()

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    plt.savefig(Path(save_dir) / f"{model_name}_permutation_importance.png")
    plt.close()

    return top_names, top_values