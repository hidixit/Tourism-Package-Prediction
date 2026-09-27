"""
Data Preparation Script
-------------------------
Loads the dataset from the repository's data folder, removes unnecessary
columns, cleans up known data-entry inconsistencies, and splits the data
into training and testing sets. The splits are saved locally as CSV files
so the GitHub Actions workflow can hand them to the next job as a
workflow artifact.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = "tourism_project/data/tourism.csv"
TARGET_COL = "ProdTaken"

# Columns that carry no predictive signal (identifiers / row numbers).
DROP_COLS = ["Unnamed: 0", "CustomerID"]


def load_and_clean(data_path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(data_path)

    # Drop unnecessary identifier columns if present.
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    # Fix a known data-entry typo in Gender ("Fe Male" -> "Female").
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})

    # Drop exact duplicate rows, if any.
    df = df.drop_duplicates()

    return df


def split_and_save(df: pd.DataFrame):
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train.to_csv("Xtrain.csv", index=False)
    X_test.to_csv("Xtest.csv", index=False)
    y_train.to_csv("ytrain.csv", index=False)
    y_test.to_csv("ytest.csv", index=False)

    print("Data preparation complete.")
    print(f"Cleaned dataset shape : {df.shape}")
    print(f"X_train shape         : {X_train.shape}")
    print(f"X_test shape          : {X_test.shape}")
    print(f"y_train distribution  :\n{y_train.value_counts()}")
    print(f"y_test distribution   :\n{y_test.value_counts()}")
    print("\nSaved: Xtrain.csv, Xtest.csv, ytrain.csv, ytest.csv")


if __name__ == "__main__":
    df = load_and_clean()
    split_and_save(df)
