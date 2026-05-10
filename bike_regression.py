"""
bike_regression.py — Bike Sharing 回归主实验

对比三个模型：
  1. xRFM (多目标回归 [casual, registered], tuning_metric='mse')
  2. XGBoost Regressor
  3. Random Forest Regressor

数据集: data/bike_sharing.csv (hour.csv)
目标变量: casual + registered (双目标回归，xrfm 0.4.3 要求 >= 2D targets)
最终评估用 cnt = casual + registered
"""

import sys
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from xrfm import xRFM
import torch

RANDOM_STATE = 42


def load_bike_data():
    """加载 Bike Sharing 数据，做基础预处理"""
    df = pd.read_csv("data/bike_sharing.csv")

    # 特征工程：提取时间特征
    df['hour'] = pd.to_datetime(df['dteday']).dt.hour
    df['month'] = pd.to_datetime(df['dteday']).dt.month
    df['dayofweek'] = pd.to_datetime(df['dteday']).dt.dayofweek

    # 双目标回归: [casual, registered] (xrfm 0.4.3 要求 shape[1] >= 2)
    y = df[['casual', 'registered']].values.astype(np.float32)
    
    # cnt 用于最终评估
    y_cnt = df['cnt'].values.astype(np.float32)

    # 特征列（去掉 instant, dteday, casual, registered, cnt）
    drop_cols = ['instant', 'dteday', 'casual', 'registered', 'cnt']
    X = df.drop(columns=drop_cols)

    # 类别特征编码
    categorical_cols = ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday',
                        'workingday', 'weathersit']
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    feature_names = list(X.columns)
    X = X.values.astype(np.float32)

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
    
    # 双目标预测之和 ≈ cnt
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
        n_estimators=500,
        max_depth=6,
        learning_rate=0.1,
        random_state=RANDOM_STATE,
        n_jobs=-1,
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
        n_estimators=500,
        max_depth=None,
        random_state=RANDOM_STATE,
        n_jobs=-1,
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


def main():
    print("Bike Sharing Regression Experiment")
    print("="*50)

    X, y, y_cnt, feature_names = load_bike_data()
    print(f"Data shape: X={X.shape}, y={y.shape}")
    print(f"Features: {len(feature_names)}")

    # 划分：60% train / 20% val / 20% test（与 Otto Group 项目一致）
    # step1: 80% train+val / 20% test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    _, _, y_cnt_trainval, y_cnt_test = train_test_split(
        X, y_cnt, test_size=0.2, random_state=RANDOM_STATE
    )
    # step2: 从 train+val 切出 val (25% of 80% = 20%)
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

    # 跑三个模型
    xrfm_metrics, xrfm_model = run_xrfm(X_train, y_train, X_val, y_val, X_test, y_cnt_test)
    xgb_metrics, xgb_model = run_xgboost(X_train, y_cnt_train, X_test, y_cnt_test)
    rf_metrics, rf_model = run_random_forest(X_train, y_cnt_train, X_test, y_cnt_test)

    # 汇总
    results = {
        'dataset': 'Bike Sharing (hour.csv)',
        'task': 'regression (multi-target: [casual, registered])',
        'n_samples': len(y_cnt),
        'n_features': len(feature_names),
        'feature_names': feature_names,
        'models': {
            'xRFM': xrfm_metrics,
            'XGBoost': xgb_metrics,
            'RandomForest': rf_metrics,
        }
    }

    # 保存
    import os
    os.makedirs("results/bike", exist_ok=True)
    with open("results/bike/bike_regression.json", "w") as f:
        json.dump(results, f, indent=4)

    # 打印汇总表
    print("\n" + "="*50)
    print("Summary")
    print("="*50)
    print(f"{'Model':<15} {'RMSE':<10} {'MAE':<10} {'R²':<10} {'Time(s)':<10}")
    print("-"*50)
    for name, m in results['models'].items():
        print(f"{name:<15} {m['rmse']:<10.2f} {m['mae']:<10.2f} {m['r2']:<10.4f} {m['train_time']:<10.2f}")

    print("\nResults saved to: results/bike/bike_regression.json")


if __name__ == "__main__":
    main()
