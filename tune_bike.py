"""
tune_reg.py — 回归模型调参（Bike Sharing）

用法：
    python tune_reg.py xgboost_regressor
    python tune_reg.py random_forest_regressor
"""

import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split

from configs.config import MODEL_CONFIG

RANDOM_STATE = 42
DATA_DIR = Path("./data")
CONFIG_DIR = Path("./configs")


def load_data():
    """加载 Bike 数据（已特征工程）"""
    X = pd.read_csv(DATA_DIR / "bike_X.csv")
    y = pd.read_csv(DATA_DIR / "bike_y.csv")  # [casual, registered]

    # 评估用 cnt = casual + registered，但调参也用双目标
    return X, y


def run_tune(model_name, X_train, y_train):
    """GridSearchCV 搜索最优参数"""
    model_config = MODEL_CONFIG[model_name]

    grid_search = GridSearchCV(
        estimator=model_config["estimator"],
        param_grid=model_config["param_grid"],
        cv=5,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
    )
    grid_search.fit(X_train, y_train)

    print(f"\n🏆 Best params ({model_name}):")
    print(json.dumps(grid_search.best_params_, indent=2))
    print(f"Best score (neg_MSE): {grid_search.best_score_:.6f}")

    return grid_search.best_params_


def save_params(model_name, best_params):
    """保存最优参数到 configs/ 目录"""
    CONFIG_DIR.mkdir(exist_ok=True)
    output_path = CONFIG_DIR / f"best_{model_name}.json"

    with open(output_path, "w") as f:
        json.dump(best_params, f, indent=2)

    print(f"\n✅ Saved to {output_path}")


def main():
    if len(sys.argv) < 2:
        print("用法: python tune_reg.py <model_name>")
        print("  model_name: xgboost_regressor | random_forest_regressor")
        sys.exit(1)

    model_name = sys.argv[1]

    if model_name not in MODEL_CONFIG:
        print(f"❌ 未知模型: {model_name}")
        sys.exit(1)

    print(f"📊 Loading Bike data...")
    X, y = load_data()
    print(f"   X={X.shape}, y={y.shape}")

    # 调参只用训练集（60%），不用验证/测试集
    X_train, _, y_train, _ = train_test_split(
        X, y,
        train_size=0.6,
        random_state=RANDOM_STATE,
    )
    print(f"   Train: {X_train.shape}")

    print(f"\n🔍 Tuning {model_name}...")
    best_params = run_tune(model_name, X_train, y_train)
    save_params(model_name, best_params)


if __name__ == "__main__":
    main()
