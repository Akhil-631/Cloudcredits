import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_reviews.csv"

TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train_reviews.csv"
TEST_PATH = PROJECT_ROOT / "data" / "processed" / "test_reviews.csv"

# Load processed dataset
df = pd.read_csv(DATA_PATH)

print("\n========== TRAIN TEST SPLITTING ==========")

print("\nOriginal Dataset Shape:", df.shape)

# Select features and target
X = df["cleaned_review"]
y = df["sentiment"]

# Perform stratified 80:20 split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Create training and testing DataFrames
train_df = pd.DataFrame({
    "cleaned_review": X_train,
    "sentiment": y_train
})

test_df = pd.DataFrame({
    "cleaned_review": X_test,
    "sentiment": y_test
})

# Save datasets
train_df.to_csv(TRAIN_PATH, index=False)
test_df.to_csv(TEST_PATH, index=False)

# Display results
print("\nTraining Dataset Shape:", train_df.shape)
print("Testing Dataset Shape:", test_df.shape)

print("\nTraining Sentiment Distribution:")
print(train_df["sentiment"].value_counts())

print("\nTesting Sentiment Distribution:")
print(test_df["sentiment"].value_counts())

print("\nTraining Percentage:", len(train_df) / len(df) * 100)
print("Testing Percentage:", len(test_df) / len(df) * 100)

print("\nTraining and testing datasets saved successfully.")

print("\n========== SPLITTING COMPLETE ==========")