"""
run_bike_main.py — Bike Sharing 回归主实验（XGBoost / Random Forest）

用法：
    python run_bike_main.py xgboost_regressor
    python run_bike_main.py random_forest_regressor

评估目标：cnt = casual + registered（双目标预测后求和）
"""

import json
import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from configs.config import MODEL_CONFIG
from src.evaluation import get_r2_score, get_mse, get_mae
from src.reporting import save_result_json, plot_bike_main_comparison

RANDOM_STATE = 42
DATA_DIR = Path("./data")
RESULT_DIR = Path("results/bike_main")


def load_bike_data():
    """加载已预处理的 Bike 数据"""
    X = pd.read_csv(DATA_DIR / "bike_X.csv")
    y = pd.read_csv(DATA_DIR / "bike_y.csv")  # [casual, registered]
    return X, y


def main(model_name):
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

    # ---- 3. 加载最优参数 ----
    with open(f"configs/best_{model_name}.json") as f:
        best = json.load(f)

    estimator = MODEL_CONFIG[model_name]["estimator"]
    estimator.set_params(**best)

    # ---- 4. 训练 ----
    start = time.perf_counter()
    estimator.fit(X_train, y_train)
    train_time = time.perf_counter() - start

    # ---- 5. 预测（双目标 → 求和得 cnt） ----
    start = time.perf_counter()
    y_pred = estimator.predict(X_test)  # shape: (n, 2)
    pred_time = (time.perf_counter() - start) / len(X_test)

    # casual + registered = cnt
    y_pred_cnt = y_pred[:, 0] + y_pred[:, 1]
    y_test_cnt = y_test["casual"].values + y_test["registered"].values

    # ---- 6. 评估 ----
    r2 = get_r2_score(y_test_cnt, y_pred_cnt)
    mse = get_mse(y_test_cnt, y_pred_cnt)
    mae = get_mae(y_test_cnt, y_pred_cnt)

    print(f"\n🏆 {model_name}")
    print(f"   R²:    {r2:.4f}")
    print(f"   MSE:   {mse:,.2f}")
    print(f"   MAE:   {mae:,.2f}")
    print(f"   Train: {train_time:.2f}s")
    print(f"   Infer: {pred_time:.6f}s/sample")

    # ---- 7. 保存结果 ----
    result = {
        "model": model_name,
        "best_params": best,
        "r2": float(r2),
        "mse": float(mse),
        "mae": float(mae),
        "train_time": float(train_time),
        "infer_time_per_sample": float(pred_time),
    }

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULT_DIR / f"{model_name}.json", "w") as f:
        json.dump(result, f, indent=4)

    print(f"\n✅ Saved to {RESULT_DIR}/{model_name}.json")

    # 尝试画对比图（如果三个模型结果都齐了）
    plot_bike_main_comparison()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python run_bike_main.py <model_name>")
        sys.exit(1)
    main(sys.argv[1])
