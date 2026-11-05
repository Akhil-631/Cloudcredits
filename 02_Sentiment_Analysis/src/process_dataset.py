import sys
import pandas as pd
from pathlib import Path

# Define project directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Add project directory to Python path
sys.path.append(str(PROJECT_ROOT))

# Import preprocessing function
from src.preprocess import clean_text

# Define dataset paths
RAW_DATA_PATH = (
    PROJECT_ROOT / "data" / "raw" / "IMDB Dataset.csv"
)

PROCESSED_DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "cleaned_reviews.csv"
)

# Load original dataset
df = pd.read_csv(RAW_DATA_PATH)

print("\n========== DATA PREPROCESSING ==========")

print("\nOriginal Dataset Shape:", df.shape)

# Remove duplicate records
df = df.drop_duplicates().copy()

print("Shape After Duplicate Removal:", df.shape)

# Apply text preprocessing
print("\nCleaning movie reviews...")

df["cleaned_review"] = df["review"].apply(clean_text)

# Remove empty reviews
df = df[df["cleaned_review"].str.strip() != ""].copy()

# Reset index
df = df.reset_index(drop=True)

# Save processed dataset
df.to_csv(PROCESSED_DATA_PATH, index=False)

# Final validation
print("\n========== FINAL DATASET SUMMARY ==========")

print("Final Dataset Shape:", df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nSentiment Distribution:")
print(df["sentiment"].value_counts())

print("\nDuplicate Records:", df.duplicated().sum())

print("\nSample Original and Cleaned Reviews:")

for index in range(3):
    print(f"\nReview {index + 1}")
    print("Original:", df.loc[index, "review"][:200])
    print("Cleaned :", df.loc[index, "cleaned_review"][:200])

print("\nProcessed dataset saved successfully:")
print(PROCESSED_DATA_PATH)