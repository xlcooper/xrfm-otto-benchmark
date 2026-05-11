"""
prepare_bike.py — Bike Sharing 数据准备

功能：读取原始 day.csv，做特征工程（提取时间特征 + One-Hot 编码类别变量），
      保存为 model-ready 的数值矩阵，供训练脚本直接加载。

手动运行一次即可：python src/prepare_bike.py
"""

import pandas as pd
from pathlib import Path


def engineer_features(input_path="./data/bike_sharing.csv", output_dir="./data"):
    """特征工程"""
    df = pd.read_csv(input_path)
    print(f"Original: {len(df):,} samples")

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

    # One-Hot 编码
    categorical_cols = []
    for c in ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday',
              'workingday', 'weathersit']:
        if c in X.columns:
            categorical_cols.append(c)
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    feature_names = list(X.columns)
    print(f"After: X={X.shape}, features={len(feature_names)}")

    # 保存
    Path(output_dir).mkdir(exist_ok=True)
    X.to_csv(f"{output_dir}/bike_X.csv", index=False)
    y.to_csv(f"{output_dir}/bike_y.csv", index=False)
    y_cnt.to_csv(f"{output_dir}/bike_y_cnt.csv", index=False)

    with open(f"{output_dir}/bike_feature_names.txt", "w") as f:
        f.write("\n".join(feature_names))

    print(f"Saved to {output_dir}/")
    return X, y, y_cnt, feature_names


if __name__ == "__main__":
    engineer_features()
