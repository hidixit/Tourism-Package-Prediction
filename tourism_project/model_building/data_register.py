"""
Data Registration Script
-------------------------
Reads the tourism dataset from the repository's data folder, validates that
all expected columns are present, and prints a short summary of the data.
This "registers" the dataset as the single source of truth for the rest of
the pipeline -- no external dataset store is required since the CSV lives
inside the GitHub repository itself.
"""

import os
import sys
import pandas as pd

DATA_PATH = "tourism_project/data/tourism.csv"

EXPECTED_COLUMNS = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier",
    "DurationOfPitch", "Occupation", "Gender", "NumberOfPersonVisiting",
    "NumberOfFollowups", "ProductPitched", "PreferredPropertyStar",
    "MaritalStatus", "NumberOfTrips", "Passport", "PitchSatisfactionScore",
    "OwnCar", "NumberOfChildrenVisiting", "Designation", "MonthlyIncome",
]


def register_dataset(data_path: str = DATA_PATH) -> pd.DataFrame:
    if not os.path.exists(data_path):
        print(f"ERROR: dataset not found at {data_path}")
        sys.exit(1)

    df = pd.read_csv(data_path)

    # Drop the stray index column some exports include, if present.
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_cols:
        print(f"ERROR: dataset is missing expected columns: {missing_cols}")
        sys.exit(1)

    print("Dataset registered successfully.")
    print(f"Source path        : {data_path}")
    print(f"Shape (rows, cols) : {df.shape}")
    print(f"Columns            : {list(df.columns)}")
    print("\nMissing values per column:")
    print(df.isnull().sum())
    print("\nTarget distribution (ProdTaken):")
    print(df["ProdTaken"].value_counts())
    print("\nSample rows:")
    print(df.head())

    return df


if __name__ == "__main__":
    register_dataset()
