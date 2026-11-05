import pandas as pd
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cleaned_reviews.csv"
)

df = pd.read_csv(DATA_PATH)

print("\n========== DATASET VALIDATION ==========")

# Dataset dimensions
print("\nDataset Shape:", df.shape)

# Column names
print("\nColumns:")
print(df.columns.tolist())

# Missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Duplicate records
print("\nDuplicate Records:", df.duplicated().sum())

# Empty cleaned reviews
empty_reviews = (
    df["cleaned_review"].str.strip() == ""
).sum()

print("\nEmpty Cleaned Reviews:", empty_reviews)

# Remaining HTML tags
html_pattern = r"<[^>]+>"

remaining_html = df["cleaned_review"].str.contains(
    html_pattern,
    regex=True,
    na=False
).sum()

print("\nReviews Containing HTML Tags:", remaining_html)

# Sentiment distribution
print("\nSentiment Distribution:")
print(df["sentiment"].value_counts())

print("\nSentiment Percentages:")
print(
    df["sentiment"].value_counts(normalize=True) * 100
)

# Review length after cleaning
df["cleaned_length"] = df["cleaned_review"].str.len()

print("\nCleaned Review Length Statistics:")
print(df["cleaned_length"].describe())

# Sample comparison
print("\n========== SAMPLE COMPARISON ==========")

for i in range(3):
    print(f"\nReview {i + 1}")
    print("Original:", df.loc[i, "review"][:200])
    print("Cleaned :", df.loc[i, "cleaned_review"][:200])

print("\n========== VALIDATION COMPLETE ==========")