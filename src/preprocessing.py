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
        val_size=0.2,
        test_size=0.2,
        random_state=42):
    
    # keep a 6:2:2 split across train/val/test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
        )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=val_size/(1-test_size),
        random_state=random_state,
        stratify=y_temp
        )

    return X_train, y_train, X_val, y_val, X_test, y_test

