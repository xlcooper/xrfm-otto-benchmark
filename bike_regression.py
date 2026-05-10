"""
bike_regression.py — Bike Sharing 回归主实验

对比三个模型：
  1. xRFM (多目标回归 [casual, registered], tuning_metric='mse')
  2. XGBoost Regressor
  3. Random Forest Regressor

数据集: data/bike_X.csv, data/bike_y.csv (由 src/preprocess_bike.py 生成)
目标变量: [casual, registered]（双目标回归，xrfm 0.4.3 要求 >= 2D）
最终评估用 cnt = casual + registered

结果保存: results/bike/main/
"""

import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from xrfm import xRFM
import torch

RANDOM_STATE = 42
RESULTS_DIR = "results/bike/main"


def load_bike_data():
    """加载预处理后的 Bike Sharing 数据"""
    X = pd.read_csv("data/bike_X.csv").values.astype(np.float32)
    y = pd.read_csv("data/bike_y.csv").values.astype(np.float32)
    y_cnt = pd.read_csv("data/bike_y_cnt.csv").values.astype(np.float32).ravel()

    with open("data/bike_feature_names.txt") as f:
        feature_names = [line.strip() for line in f]

    return X, y, y_cnt, feature_names


def run_xrfm(X_train, y_train, X_val, y_val, X_test, y_cnt_test):
    """运行 xRFM 多目标回归"""
    print("\n" + "="*50)
    print("Running: xRFM (Multi-Target Regression)")
    print("="*50)

    device = 'cpu'
    model = xRFM(
        device=device,
        tuning_metric='mse',
        n_tree_iters=3,
        random_state=RANDOM_STATE,
    )

    start = time.time()
    model.fit(
        torch.tensor(X_train), torch.tensor(y_train),
        torch.tensor(X_val), torch.tensor(y_val)
    )
    train_time = time.time() - start

    y_pred = model.predict(torch.tensor(X_test)).numpy()
    y_pred_cnt = y_pred[:, 0] + y_pred[:, 1]

    metrics = {
        'rmse': float(np.sqrt(mean_squared_error(y_cnt_test, y_pred_cnt))),
        'mae': float(mean_absolute_error(y_cnt_test, y_pred_cnt)),
        'r2': float(r2_score(y_cnt_test, y_pred_cnt)),
        'train_time': train_time,
    }

    print(f"  RMSE: {metrics['rmse']:.2f}")
    print(f"  MAE:  {metrics['mae']:.2f}")
    print(f"  R²:   {metrics['r2']:.4f}")
    print(f"  Time: {train_time:.2f}s")

    return metrics, model


def run_xgboost(X_train, y_train, X_test, y_test):
    """运行 XGBoost 回归"""
    print("\n" + "="*50)
    print("Running: XGBoost Regressor")
    print("="*50)

    model = XGBRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.1,
        random_state=RANDOM_STATE, n_jobs=-1,
    )

    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start

    y_pred = model.predict(X_test)

    metrics = {
        'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
        'mae': float(mean_absolute_error(y_test, y_pred)),
        'r2': float(r2_score(y_test, y_pred)),
        'train_time': train_time,
    }

    print(f"  RMSE: {metrics['rmse']:.2f}")
    print(f"  MAE:  {metrics['mae']:.2f}")
    print(f"  R²:   {metrics['r2']:.4f}")
    print(f"  Time: {train_time:.2f}s")

    return metrics, model


def run_random_forest(X_train, y_train, X_test, y_test):
    """运行 Random Forest 回归"""
    print("\n" + "="*50)
    print("Running: Random Forest Regressor")
    print("="*50)

    model = RandomForestRegressor(
        n_estimators=500, max_depth=None,
        random_state=RANDOM_STATE, n_jobs=-1,
    )

    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start

    y_pred = model.predict(X_test)

    metrics = {
        'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
        'mae': float(mean_absolute_error(y_test, y_pred)),
        'r2': float(r2_score(y_test, y_pred)),
        'train_time': train_time,
    }

    print(f"  RMSE: {metrics['rmse']:.2f}")
    print(f"  MAE:  {metrics['mae']:.2f}")
    print(f"  R²:   {metrics['r2']:.4f}")
    print(f"  Time: {train_time:.2f}s")

    return metrics, model


def plot_bike_comparison(results, save_path):
    """画 Bike Sharing 回归对比图（跳过 None 值）"""
    # 过滤掉 None 值（如 xRFM 未跑通时）
    valid = {k: v for k, v in results.items() if v.get('rmse') is not None}
    if len(valid) < 2:
        print("Not enough valid results to plot, skipping")
        return

    model_labels = {'xgboost': 'XGBoost', 'random_forest': 'Random Forest', 'xrfm': 'xRFM'}
    models = [model_labels[k] for k in valid.keys()]
    keys = list(valid.keys())
    colors = ['#2E86AB', '#A23B72', '#F18F01'][:len(keys)]

    rmse = [valid[k]['rmse'] for k in keys]
    mae = [valid[k]['mae'] for k in keys]
    r2 = [valid[k]['r2'] for k in keys]
    train_time = [valid[k]['train_time'] for k in keys]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].bar(models, rmse, color=colors)
    axes[0].set_ylabel('RMSE')
    axes[0].set_title('RMSE (lower is better)')

    axes[1].bar(models, mae, color=colors)
    axes[1].set_ylabel('MAE')
    axes[1].set_title('MAE (lower is better)')

    axes[2].bar(models, r2, color=colors)
    axes[2].set_ylabel('R²')
    axes[2].set_title('R² (higher is better)')
    axes[2].set_ylim(min(r2) * 0.98, 1.0)

    axes[3].bar(models, train_time, color=colors)
    axes[3].set_ylabel('Time (seconds)')
    axes[3].set_title('Training Time')

    plt.suptitle('Bike Sharing Dataset — Regression Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved: {save_path}")
    plt.close()


def main():
    print("Bike Sharing Regression Experiment")
    print("="*50)

    X, y, y_cnt, feature_names = load_bike_data()
    print(f"Data: X={X.shape}, y={y.shape}, features={len(feature_names)}")

    # 划分：60% train / 20% val / 20% test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    _, _, y_cnt_trainval, y_cnt_test = train_test_split(
        X, y_cnt, test_size=0.2, random_state=RANDOM_STATE
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.25, random_state=RANDOM_STATE
    )
    _, _, y_cnt_train, y_cnt_val = train_test_split(
        X_trainval, y_cnt_trainval, test_size=0.25, random_state=RANDOM_STATE
    )

    print(f"Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")

    # 标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # 跑模型
    # Note: xRFM 0.4.3 在 Apple Silicon 上大数据量 SIGSEGV，暂时跳过
    xrfm_metrics = {
        'rmse': None, 'mae': None, 'r2': None,
        'train_time': None,
        'note': 'xRFM 0.4.3 SIGSEGV on Apple Silicon with >2K samples, skipped'
    }
    
    xgb_metrics, _ = run_xgboost(X_train, y_cnt_train, X_test, y_cnt_test)
    rf_metrics, _ = run_random_forest(X_train, y_cnt_train, X_test, y_cnt_test)

    # 汇总
    results = {
        'dataset': 'Bike Sharing (hour.csv)',
        'task': 'regression (multi-target: [casual, registered])',
        'n_samples': len(y_cnt),
        'n_features': len(feature_names),
        'feature_names': feature_names,
        'models': {
            'xrfm': xrfm_metrics,
            'xgboost': xgb_metrics,
            'random_forest': rf_metrics,
        }
    }

    # 保存
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    for name, m in results['models'].items():
        with open(f"{RESULTS_DIR}/{name}.json", "w") as f:
            json.dump(m, f, indent=4)

    # 画图
    plot_bike_comparison(results['models'], f"{RESULTS_DIR}/bike_comparison.png")

    # 打印汇总表
    print("\n" + "="*50)
    print("Summary")
    print("="*50)
    print(f"{'Model':<15} {'RMSE':<10} {'MAE':<10} {'R²':<10} {'Time(s)':<10}")
    print("-"*50)
    for name, m in results['models'].items():
        if m.get('rmse') is not None:
            print(f"{name:<15} {m['rmse']:<10.2f} {m['mae']:<10.2f} {m['r2']:<10.4f} {m['train_time']:<10.2f}")
        else:
            print(f"{name:<15} {'--':<10} {'--':<10} {'--':<10} {'--':<10}  ({m.get('note', 'skipped')})")

    print(f"\nResults saved to: {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
