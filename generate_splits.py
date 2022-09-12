import os
import pandas as pd
from sklearn.model_selection import train_test_split

split_dir = '/home/guillaume/Documents/uda/project-augmented-embeddings/3-downstream/splits/sicapv2'
labels_path = '/home/guillaume/Documents/uda/project-augmented-embeddings/3-downstream/datasets_csv/labels.csv'

def get_dataset_from_csv(path, x_col, y_col):
    df = pd.read_csv(path)
    # print(df.head(5))

    X = df[x_col].tolist()
    y = df[y_col].tolist()

    # print(len(X))
    return (X, y)

def generate_splits(X, y):
    splits = {}

    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.20, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, stratify=y_train_val)

    splits = {
        "train": X_train,
        "val": X_val,
        "test": X_test,
    }
    print(len(splits["train"]))
    print(len(splits["val"]))
    print(len(splits["test"]))

    return splits


def get_split_df(splits):
    train_df = pd.DataFrame(splits["train"], columns=["train"])
    val_df = pd.DataFrame(splits["val"], columns=["val"])
    test_df = pd.DataFrame(splits["test"], columns=["test"])
    
    df = pd.concat([train_df, val_df, test_df], axis=1).fillna("")
    print(df.head)

    return df


if __name__ == "__main__":
    X, y = get_dataset_from_csv(labels_path, "image_id", "isup_grade")
    NUM_SPLITS = 5
    print(X[:5])
    print(y[:5])

    for i in range(NUM_SPLITS):
        splits = generate_splits(X, y)
        df = get_split_df(splits)
        df.to_csv(os.path.join(split_dir, f"splits_{i}.csv"))
