"""
prepare_otto.py — Otto Group 数据准备

从原始 61K 数据中采样 20K，保存为 train.csv
手动运行一次即可：python src/prepare_otto.py
"""

import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42


def sample_data(input_path="./data/train_full.csv", output_path="./data/train.csv", n_keep=20000):
    """从原始数据中采样"""
    df = pd.read_csv(input_path)
    print(f"Original: {len(df):,} samples")

    if n_keep and n_keep < len(df):
        y = df['target']
        df, _ = train_test_split(df, train_size=n_keep, random_state=RANDOM_STATE, stratify=y)
        df = df.sort_index().reset_index(drop=True)

    df.to_csv(output_path, index=False)
    print(f"Saved: {len(df):,} samples -> {output_path}")
    return df


if __name__ == "__main__":
    sample_data()
