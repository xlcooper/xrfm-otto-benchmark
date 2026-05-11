"""
preprocessing.py — Otto Group 数据预处理

包含：
  1. sample_data() — 从原始 61K 采样到 20K（手动运行一次）
  2. load_data() — 加载已采样数据 + LabelEncoder
  3. split_data() — 划分 train/test
  4. split_data_ontrain() — 从 train 切出 val
"""

from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42


def sample_data(input_path="./data/train_full.csv", output_path="./data/train.csv", n_keep=20000):
    """从原始 61K 数据中采样 20K，保存到 train.csv（手动运行一次）"""
    df = pd.read_csv(input_path)
    original_size = len(df)
    print(f"Original dataset: {original_size:,} samples")

    if n_keep and n_keep < original_size:
        y = df['target']
        df, _ = train_test_split(df, train_size=n_keep, random_state=RANDOM_STATE, stratify=y)
        df = df.sort_index().reset_index(drop=True)

    df.to_csv(output_path, index=False)
    print(f"Preprocessed dataset: {len(df):,} samples (saved to {output_path})")
    return df


def load_data(data_path="./data/train.csv"):
    """加载已采样的数据，做 LabelEncoder"""
    df = pd.read_csv(data_path)
    if 'id' in df.columns:
        df = df.drop(columns=['id'])

    le = LabelEncoder()
    df['target'] = le.fit_transform(df['target'])

    X = df.drop(columns='target')
    y = df['target']

    return X, y, le


def split_data(X, y, test_size=0.2, random_state=42):
    """划分 train/test"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


def split_data_ontrain(X_train, y_train, test_size=0.25, random_state=42):
    """从 train 中切出 val"""
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=test_size, random_state=random_state, stratify=y_train
    )
    return X_train, X_val, y_train, y_val