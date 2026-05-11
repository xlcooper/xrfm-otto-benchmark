"""
preprocessing.py — 通用训练接口

所有数据集共用的加载和划分函数。
数据准备（采样/特征工程）请先用 prepare_xxx.py 处理。
"""

from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42


# ====== Otto Group ======

def load_otto_data(data_path="./data/train.csv"):
    """加载 Otto 数据 + LabelEncoder"""
    df = pd.read_csv(data_path)
    if 'id' in df.columns:
        df = df.drop(columns=['id'])

    le = LabelEncoder()
    df['target'] = le.fit_transform(df['target'])

    X = df.drop(columns='target')
    y = df['target']
    return X, y, le


# ====== Bike Sharing ======

def load_bike_data():
    """加载 Bike Sharing 数据"""
    X = pd.read_csv("data/bike_X.csv").values.astype('float32')
    y = pd.read_csv("data/bike_y.csv").values.astype('float32')
    y_cnt = pd.read_csv("data/bike_y_cnt.csv").values.astype('float32').ravel()

    with open("data/bike_feature_names.txt") as f:
        feature_names = [line.strip() for line in f]

    return X, y, y_cnt, feature_names


# ====== 通用划分（所有数据集共用）======

def split_data(X, y, test_size=0.2, random_state=42):
    """划分 train/test"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


def split_data_ontrain(X_train, y_train, test_size=0.25, random_state=42):
    """从 train 中切出 val"""
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=test_size, random_state=random_state
    )
    return X_train, X_val, y_train, y_val
