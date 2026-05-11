"""
bike_preprocessing.py — Bike Sharing 数据预处理

原始数据: data/bike_sharing.csv (day.csv, 731 samples)
处理后:   特征工程 + One-Hot 编码后的特征矩阵 X 和目标矩阵 y

说明:
- 使用 UCI Bike Sharing Dataset 的 day.csv（按日粒度）
- 原始 731 条样本，不做子采样
- 目标变量: [casual, registered]（双目标回归，xrfm 0.4.3 需要 >= 2D）
- 评估变量: cnt = casual + registered
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

RANDOM_STATE = 42


def engineer_features(input_path="./data/bike_sharing.csv", output_dir="./data"):
    """特征工程：提取时间特征 + One-Hot，保存处理后的数据（手动运行一次）"""
    df = pd.read_csv(input_path)

    print(f"Original dataset: {len(df):,} samples")

    # 时间特征
    df['hour'] = pd.to_datetime(df['dteday']).dt.hour
    df['month'] = pd.to_datetime(df['dteday']).dt.month
    df['dayofweek'] = pd.to_datetime(df['dteday']).dt.dayofweek

    # 目标变量
    y = df[['casual', 'registered']]
    y_cnt = df[['cnt']]

    # 特征列
    drop_cols = ['instant', 'dteday', 'casual', 'registered', 'cnt']
    X = df.drop(columns=drop_cols)

    # One-Hot 编码类别特征（day.csv 没有 hr 列）
    categorical_cols = [c for c in ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday',
                        'workingday', 'weathersit'] if c in X.columns]
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    feature_names = list(X.columns)

    print(f"After preprocessing: X={X.shape}, y={y.shape}, features={len(feature_names)}")

    # 保存
    Path(output_dir).mkdir(exist_ok=True)
    X.to_csv(f"{output_dir}/bike_X.csv", index=False)
    y.to_csv(f"{output_dir}/bike_y.csv", index=False)
    y_cnt.to_csv(f"{output_dir}/bike_y_cnt.csv", index=False)

    with open(f"{output_dir}/bike_feature_names.txt", "w") as f:
        f.write("\n".join(feature_names))

    print(f"Saved to {output_dir}/bike_X.csv, bike_y.csv, bike_y_cnt.csv")
    return X, y, y_cnt, feature_names


def load_data():
    """加载预处理后的 Bike Sharing 数据"""
    X = pd.read_csv("data/bike_X.csv").values.astype('float32')
    y = pd.read_csv("data/bike_y.csv").values.astype('float32')
    y_cnt = pd.read_csv("data/bike_y_cnt.csv").values.astype('float32').ravel()

    with open("data/bike_feature_names.txt") as f:
        feature_names = [line.strip() for line in f]

    return X, y, y_cnt, feature_names


def split_data(X, y, test_size=0.2, random_state=42):
    """划分 train/val/test（60/20/20）"""
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.25, random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    engineer_features()
