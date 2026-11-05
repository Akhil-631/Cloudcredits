import pandas as pd
from pathlib import Path

# Project directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dataset path
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "IMDB Dataset.csv"

# Load dataset
df = pd.read_csv(DATASET_PATH)

print("\n========== DATA QUALITY ANALYSIS ==========")

# 1. Exact duplicate rows
print("\n1. Exact Duplicate Rows:")
print(df.duplicated().sum())

# 2. Duplicate reviews, regardless of sentiment
print("\n2. Duplicate Review Texts:")
print(df["review"].duplicated().sum())

# 3. Reviews associated with multiple sentiment labels
review_label_counts = (
    df.groupby("review")["sentiment"]
    .nunique()
)

conflicting_reviews = review_label_counts[
    review_label_counts > 1
]

print("\n3. Reviews With Conflicting Sentiment Labels:")
print(len(conflicting_reviews))

# 4. Display conflicting examples
if len(conflicting_reviews) > 0:
    print("\nConflicting Review Examples:")

    examples = df[
        df["review"].isin(conflicting_reviews.index)
    ]

    print(examples.head(10))

# 5. Review length analysis
df["review_length"] = df["review"].str.len()

print("\n4. Review Length Statistics:")
print(df["review_length"].describe())

# 6. Shortest and longest reviews
print("\n5. Shortest Review:")
print(df.loc[df["review_length"].idxmin(), "review"])

print("\n6. Longest Review Length:")
print(df["review_length"].max())

# 7. HTML tag inspection
html_count = df["review"].str.contains(
    r"<[^>]+>",
    regex=True
).sum()

print("\n7. Reviews Containing HTML Tags:")
print(html_count)

print("\n========== ANALYSIS COMPLETED ==========")