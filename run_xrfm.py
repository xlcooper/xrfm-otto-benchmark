"""
run_xrfm.py — xRFM 主结果评估脚本

xRFM 与 sklearn 模型的区别：
  1. fit() 需要传验证集 (X_val, y_val)
  2. 分类任务的 y 需要 one-hot 编码
  3. 输入必须是 numpy array，不接受 DataFrame
  4. 内部自动调参，不需要 tune.py

用法：python run_xrfm.py
"""

import json, time
import numpy as np

from src.preprocessing import load_data, split_data, split_data_ontrain
from src.evaluation import get_accuracy, get_auc_roc
from src.reporting import plot_main_comparison
from sklearn.preprocessing import OneHotEncoder
from xrfm import xRFM
from pathlib import Path


def main():

    # ---- 1. 加载数据 & 划分 ----
    X, y, le = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    # xRFM 需要单独的验证集做内部调参
    X_train, X_val, y_train, y_val = split_data_ontrain(X_train, y_train)

    # ---- 2. 转 numpy（xRFM 底层用 PyTorch，不接受 DataFrame）----
    X_train = X_train.values
    X_val   = X_val.values
    X_test  = X_test.values

    # 保留原始标签用于评估（一维整数）
    y_test_labels = y_test.values

    # ---- 3. y one-hot 编码（xRFM 分类任务要求二维 float）----
    enc = OneHotEncoder(sparse_output=False)
    y_train_oh = enc.fit_transform(y_train.values.reshape(-1, 1)).astype(np.float32)
    y_val_oh   = enc.transform(y_val.values.reshape(-1, 1)).astype(np.float32)

    # ---- 4. 初始化模型 ----
    estimator = xRFM(
        device='cpu',
        tuning_metric='accuracy',
        n_tree_iters=3,
        random_state=42,
    )

    # ---- 5. 训练 + 计时 ----
    start_time = time.perf_counter()
    estimator.fit(X_train, y_train_oh, X_val, y_val_oh)
    train_time = time.perf_counter() - start_time

    # ---- 6. 推断 + 计时 ----
    start_time = time.perf_counter()
    y_pred = estimator.predict(X_test)
    pred_time = (time.perf_counter() - start_time) / len(X_test)

    # ---- 7. 评估 ----
    y_prob = estimator.predict_proba(X_test)
    accuracy = get_accuracy(y_test_labels, y_pred)
    auc_roc  = get_auc_roc(y_test_labels, y_prob)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC-ROC:  {auc_roc:.4f}")
    print(f"Train time: {train_time:.2f}s")
    print(f"Infer time/sample: {pred_time:.6f}s")

    # ---- 8. 保存结果 ----
    result = {
        "model": "xrfm",
        "accuracy": float(accuracy),
        "auc_roc": float(auc_roc),
        "train_time": float(train_time),
        "infer_time_per_sample": float(pred_time),
    }

    Path("results/otto_main").mkdir(parents=True, exist_ok=True)
    with open("results/otto_main/xrfm.json", "w") as f:
        json.dump(result, f, indent=4)

    # 尝试画对比图（如果三个模型结果都齐了）
    plot_main_comparison()


if __name__ == "__main__":
    main()
