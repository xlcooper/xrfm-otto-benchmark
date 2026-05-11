"""
run_bike_xrfm.py — Bike Sharing 回归实验（xRFM）

xRFM 回归特殊点：
  1. fit() 需要传验证集 (X_val, y_val)
  2. y 必须是二维（已满足：[casual, registered]）
  3. 输入必须是 numpy array
  4. device='cpu', tuning_metric='mse'

用法：python run_bike_xrfm.py
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from xrfm import xRFM
from src.evaluation import get_r2_score, get_mse, get_mae
from src.reporting import plot_bike_main_comparison

RANDOM_STATE = 42
DATA_DIR = Path("./data")
RESULT_DIR = Path("results/bike_main")


def load_bike_data():
    """加载已预处理的 Bike 数据"""
    X = pd.read_csv(DATA_DIR / "bike_X.csv")
    y = pd.read_csv(DATA_DIR / "bike_y.csv")
    return X, y


def main():
    # ---- 1. 加载数据 ----
    X, y = load_bike_data()
    print(f"📊 X={X.shape}, y={y.shape}")

    # ---- 2. 划分 60/20/20 ----
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, train_size=0.6, random_state=RANDOM_STATE,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE,
    )
    print(f"   Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # ---- 3. 转 numpy ----
    X_train = X_train.values.astype(np.float32)
    X_val   = X_val.values.astype(np.float32)
    X_test  = X_test.values.astype(np.float32)
    y_train = y_train.values.astype(np.float32)
    y_val   = y_val.values.astype(np.float32)

    # ---- 4. 初始化 xRFM ----
    estimator = xRFM(
        device='cpu',
        tuning_metric='mse',
        n_tree_iters=1,
        n_threads=1,
        random_state=RANDOM_STATE,
    )

    # ---- 5. 训练 ----
    start = time.perf_counter()
    estimator.fit(X_train, y_train, X_val, y_val)
    train_time = time.perf_counter() - start

    # ---- 6. 预测 ----
    start = time.perf_counter()
    y_pred = estimator.predict(X_test)
    pred_time = (time.perf_counter() - start) / len(X_test)

    # ---- 7. 评估（cnt = casual + registered） ----
    y_pred_cnt = y_pred[:, 0] + y_pred[:, 1]
    y_test_cnt = y_test["casual"].values + y_test["registered"].values

    r2  = get_r2_score(y_test_cnt, y_pred_cnt)
    mse = get_mse(y_test_cnt, y_pred_cnt)
    mae = get_mae(y_test_cnt, y_pred_cnt)

    print(f"\n🏆 xRFM (regression)")
    print(f"   R²:    {r2:.4f}")
    print(f"   MSE:   {mse:,.2f}")
    print(f"   MAE:   {mae:,.2f}")
    print(f"   Train: {train_time:.2f}s")
    print(f"   Infer: {pred_time:.6f}s/sample")

    # ---- 8. 保存结果 ----
    result = {
        "model": "xrfm",
        "r2": float(r2),
        "mse": float(mse),
        "mae": float(mae),
        "train_time": float(train_time),
        "infer_time_per_sample": float(pred_time),
    }

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULT_DIR / "xrfm.json", "w") as f:
        json.dump(result, f, indent=4)

    print(f"\n✅ Saved to {RESULT_DIR}/xrfm.json")

    # 尝试画对比图（如果三个模型结果都齐了）
    plot_bike_main_comparison()


if __name__ == "__main__":
    main()
