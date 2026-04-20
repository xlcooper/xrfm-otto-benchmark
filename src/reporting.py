import json
from pathlib import Path


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
