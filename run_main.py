import sys, json, time

from src.preprocessing import load_data, split_data
from src.evaluation import get_accuracy, get_auc_roc
from src.reporting import save_result_json, plot_main_comparison
from configs.config import MODEL_CONFIG

from pathlib import Path

def main(model_name):

    X, y, _ = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    with open(f"configs/best_{model_name}.json") as f:
        best = json.load(f)
    model_config = MODEL_CONFIG[model_name]
    estimator = model_config["estimator"]
    estimator.set_params(**best["best_params"])

    # train by best_params
    start_time = time.perf_counter()
    estimator.fit(X_train, y_train)
    train_time = time.perf_counter() - start_time

    # predict
    start_time = time.perf_counter()
    y_pred = estimator.predict(X_test)
    pred_time = (time.perf_counter() - start_time) / len(X_test)

    # evaluate
    y_prob = estimator.predict_proba(X_test)
    accuracy = get_accuracy(y_test, y_pred)
    auc_roc = get_auc_roc(y_test, y_prob)
    
    # save result
    result = {
    "model": model_name,
    "best_params": best["best_params"],
    "accuracy": float(accuracy),
    "auc_roc": float(auc_roc),
    "train_time": float(train_time),
    "infer_time_per_sample": float(pred_time),
    }

    Path("results/otto_main").mkdir(parents=True, exist_ok=True)
    with open(f"results/otto_main/{model_name}.json", "w") as f:
        json.dump(result, f, indent=4)

    # 尝试画对比图（如果三个模型结果都齐了）
    plot_main_comparison()

if __name__ == "__main__":
    model_name = sys.argv[1]
    main(model_name)