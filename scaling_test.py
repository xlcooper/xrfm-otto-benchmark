"""
scaling_test.py — 子采样实验

在 Otto 数据集上，取不同大小的子集，测试模型的 scaling 表现。
分开跑各模型，跑完后合并画图。

用法：
  python scaling_test.py xgboost     # 只跑 XGBoost
  python scaling_test.py xrfm        # 只跑 xRFM
  python scaling_test.py plot         # 合并已有结果画图
"""

import sys, json, time
import numpy as np
import matplotlib.pyplot as plt

from src.preprocessing import load_data, split_data
from src.evaluation import get_accuracy
from sklearn.preprocessing import OneHotEncoder
from pathlib import Path


N_VALUES = [500, 1000, 2000, 5000, 10000, 20000]
RANDOM_STATE = 42


def run_xgboost(X_train, y_train, X_test, y_test, best_params):
    """用已知的 best_params 训练 XGBoost，返回 accuracy 和 training time"""
    from xgboost import XGBClassifier

    model = XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
    )
    model.set_params(**best_params)

    start = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - start

    y_pred = model.predict(X_test)
    accuracy = get_accuracy(y_test, y_pred)

    return accuracy, train_time


def run_xrfm(X_train, y_train, X_test, y_test):
    """训练 xRFM，返回 accuracy 和 training time"""
    from xrfm import xRFM
    from sklearn.model_selection import train_test_split

    # xRFM 需要从 train 中切出 val
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.25,
        random_state=RANDOM_STATE, stratify=y_train
    )

    # one-hot 编码（xRFM 分类任务要求）
    enc = OneHotEncoder(sparse_output=False)
    y_tr_oh  = enc.fit_transform(y_tr.reshape(-1, 1)).astype(np.float32)
    y_val_oh = enc.transform(y_val.reshape(-1, 1)).astype(np.float32)

    model = xRFM(
        device='cpu',
        tuning_metric='accuracy',
        n_tree_iters=3,
        random_state=RANDOM_STATE,
    )

    start = time.perf_counter()
    model.fit(X_tr, y_tr_oh, X_val, y_val_oh)
    train_time = time.perf_counter() - start

    y_pred = model.predict(X_test)
    accuracy = get_accuracy(y_test, y_pred)

    return accuracy, train_time


def run_model(model_name):
    """对指定模型跑所有 n 值，保存结果到 JSON"""
    X, y, le = load_data()

    # XGBoost 需要读 best_params
    xgb_best_params = None
    if model_name == "xgboost":
        with open("configs/best_xgboost.json") as f:
            xgb_best_params = json.load(f)["best_params"]

    results = {"n_values": N_VALUES, "accuracy": [], "train_time": []}

    for n in N_VALUES:
        print(f"\n{'='*50}")
        print(f"{model_name} | n = {n}")
        print(f"{'='*50}")

        X_train, X_test, y_train, y_test = split_data(X, y, n_samples=n)

        # 转 numpy
        X_train_np, X_test_np = X_train.values, X_test.values
        y_train_np, y_test_np = y_train.values, y_test.values

        if model_name == "xgboost":
            acc, t = run_xgboost(X_train_np, y_train_np, X_test_np, y_test_np, xgb_best_params)
        elif model_name == "xrfm":
            acc, t = run_xrfm(X_train_np, y_train_np, X_test_np, y_test_np)

        results["accuracy"].append(float(acc))
        results["train_time"].append(float(t))
        print(f"  Accuracy: {acc:.4f}, Time: {t:.2f}s")

    # 保存单个模型的结果
    Path("results").mkdir(exist_ok=True)
    with open(f"results/scaling_{model_name}.json", "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nDone! Saved to results/scaling_{model_name}.json")


def plot():
    """读取各模型的 scaling 结果，画对比图"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for name in ["xgboost", "xrfm"]:
        path = f"results/scaling_{name}.json"
        if not Path(path).exists():
            print(f"Warning: {path} not found, skipping")
            continue
        with open(path) as f:
            data = json.load(f)
        marker = 'o-' if name == "xgboost" else 's-'
        axes[0].plot(data["n_values"], data["accuracy"], marker, label=name)
        axes[1].plot(data["n_values"], data["train_time"], marker, label=name)

    axes[0].set_xlabel("Number of samples (n)")
    axes[0].set_ylabel("Test Accuracy")
    axes[0].set_title("Test Accuracy vs Sample Size")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Number of samples (n)")
    axes[1].set_ylabel("Training Time (s)")
    axes[1].set_title("Training Time vs Sample Size")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/scaling_test.png", dpi=150)
    plt.close()
    print("Plot saved to results/scaling_test.png")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "plot":
        plot()
    elif cmd in ("xgboost", "xrfm"):
        run_model(cmd)
    else:
        print(f"Unknown command: {cmd}")
        print("Usage: python scaling_test.py [xgboost|xrfm|plot]")
