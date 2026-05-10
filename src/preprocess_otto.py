"""
preprocess_otto.py — Otto Group 数据预处理

原始数据: 61,878 条样本 (data/train.csv)
处理后:   20,000 条随机样本 (覆盖原文件)

说明:
- 原始数据来自 Kaggle Otto Group 竞赛，共 61,878 条样本
- 本项目保留 20,000 条作为实验数据集（已足够支撑模型对比）
- 若需全量数据，请从原始竞赛页面重新下载
"""

import pandas as pd
from sklearn.model_selection import train_test_split


def preprocess_otto(input_path="./data/train.csv", output_path="./data/train.csv", n_keep=20000, random_state=42):
    """读取原始数据，随机保留 n_keep 条，覆盖保存"""
    df = pd.read_csv(input_path)
    
    original_size = len(df)
    print(f"Original dataset: {original_size:,} samples")
    
    if n_keep and n_keep < original_size:
        # 随机保留 n_keep 条（stratify 保持类别比例）
        y = df['target']
        df, _ = train_test_split(
            df, train_size=n_keep,
            random_state=random_state, stratify=y
        )
        df = df.sort_index().reset_index(drop=True)  # 按原始顺序排列
    
    df.to_csv(output_path, index=False)
    print(f"Preprocessed dataset: {len(df):,} samples (saved to {output_path})")
    
    return df


if __name__ == "__main__":
    preprocess_otto()
