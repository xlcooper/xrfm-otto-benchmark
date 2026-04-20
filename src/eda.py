import pandas as pd

def main(data_path='./data/train.csv'):
    df = pd.read_csv(data_path, encoding='latin-1')
    print("=== 数据形状 ===")
    print(df.shape)

    print("=== 特征类型 ===")
    print(df.dtypes)

    print("\n=== 前5行 ===")
    print(df.head())

    print("\n=== 缺失值 ===")
    print(df.isnull().sum())

    print("\n=== 类别分布 ===")
    print(df['target'].value_counts())
    print(df["target"].value_counts(normalize=True))

if __name__ == "__main__":
    main()