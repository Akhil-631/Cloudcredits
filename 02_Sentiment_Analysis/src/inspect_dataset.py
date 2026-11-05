import pandas as pd
from pathlib import Path

# Project directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dataset path
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "IMDB Dataset.csv"

# Load dataset
df = pd.read_csv(DATASET_PATH)

# Display dataset information
print("\n========== DATASET OVERVIEW ==========")

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nFirst Five Records:")
print(df.head())

print("\nDataset Information:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())

print("\nSentiment Distribution:")
print(df["sentiment"].value_counts())

print("\nSentiment Percentages:")
print(df["sentiment"].value_counts(normalize=True) * 100)

print("\nDuplicate Records:")
print(df.duplicated().sum())