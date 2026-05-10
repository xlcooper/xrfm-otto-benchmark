import json
from pathlib import Path
import matplotlib.pyplot as plt


def plot_main_comparison(results_dir="results/main", save_path="results/main/main_experiment_comparison.png"):
    """读取三个模型的主实验结果，画对比图"""
    results = {}
    for model in ['xgboost_multiclass', 'random_forest_clf', 'xrfm']:
        try:
            with open(f'{results_dir}/{model}.json') as f:
                results[model] = json.load(f)
        except FileNotFoundError:
            return False  # 结果还没齐，跳过

    models = ['XGBoost', 'Random Forest', 'xRFM']
    keys = ['xgboost_multiclass', 'random_forest_clf', 'xrfm']

    accuracy = [results[k]['accuracy'] for k in keys]
    auc_roc = [results[k]['auc_roc'] for k in keys]
    train_time = [results[k]['train_time'] for k in keys]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    colors = ['#2E86AB', '#A23B72', '#F18F01']

    axes[0].bar(models, accuracy, color=colors)
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Test Accuracy')
    axes[0].set_ylim(min(accuracy) * 0.98, max(accuracy) * 1.01)
    for i, v in enumerate(accuracy):
        axes[0].text(i, v + 0.001, f'{v:.4f}', ha='center', va='bottom')

    axes[1].bar(models, auc_roc, color=colors)
    axes[1].set_ylabel('AUC-ROC')
    axes[1].set_title('AUC-ROC Score')
    axes[1].set_ylim(min(auc_roc) * 0.995, max(auc_roc) * 1.003)
    for i, v in enumerate(auc_roc):
        axes[1].text(i, v + 0.001, f'{v:.4f}', ha='center', va='bottom')

    axes[2].bar(models, train_time, color=colors)
    axes[2].set_ylabel('Time (seconds)')
    axes[2].set_title('Training Time')
    for i, v in enumerate(train_time):
        axes[2].text(i, v + max(train_time)*0.02, f'{v:.2f}s', ha='center', va='bottom')

    plt.suptitle('Otto Group Dataset — Main Experiment Results', fontsize=14, fontweight='bold')
    plt.tight_layout()
    Path(results_dir).mkdir(exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved: {save_path}")
    plt.close()
    return True


def save_result_json(result, model_name, timestamp, save_dir="results"):
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    file_path = Path(save_dir) / f"{model_name}_{timestamp}.json"

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    return file_path


def save_result_md(result, model_name, timestamp, save_dir="results"):
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    file_path = Path(save_dir) / f"{model_name}_{timestamp}.md"
    confusion_matrix_text = "\n".join(
        str(row) for row in result["confusion_matrix"]
    )

    md_text = f"""# Experiment Result

## Model
{result["model"]}

## Best Params
{result["best_params"]}

## Best CV Score
{result["best_cv_score"]:.6f}

## Validation F1 Score
{result["val_f1_score"]:.6f}

## Validation Accuracy
{result["val_accuracy"]:.6f}

## Trainset Accuracy
{result["train_accuracy"]:.6f}

## Validation AUC-ROC
{result["val_auc_roc"]:.6f}

## Confusion Matrix
```text
{confusion_matrix_text}
```

## Classification Report
```text
{result["report"]}
```
"""

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(md_text)

    return file_path
