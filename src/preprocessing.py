from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(data_path="./data/train.csv"):

    df = pd.read_csv(data_path)
    df = df.drop(columns=['id'])

    #  encoder targets
    le = LabelEncoder()
    df['target'] = le.fit_transform(df['target'])

    X = df.drop(columns='target') 
    y = df['target']

    return X, y, le


def split_data(
        X,
        y,
        test_size=0.2,
        random_state=42):
    
    # keep a 8:2 split across train/val/test
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y     # 保持分后数据类别均衡
        )

    return X_train, X_test, y_train, y_test

