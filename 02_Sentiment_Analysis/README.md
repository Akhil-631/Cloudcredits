# 🎬 Movie Review Sentiment Analysis

A Machine Learning and Natural Language Processing project that classifies IMDb movie reviews as **Positive** or **Negative** using TF-IDF feature extraction and supervised machine learning algorithms.

Developed as part of the **Cloudcredits Technologies Machine Learning & Artificial Intelligence Internship**.

---

## 1. Project Overview

The objective of this project is to build a sentiment classification system capable of identifying whether a movie review expresses a positive or negative opinion.

The project uses the IMDb Movie Reviews dataset containing 50,000 labelled reviews.

Two machine learning algorithms were implemented and compared:

- Multinomial Naive Bayes
- Logistic Regression

Logistic Regression achieved the best overall performance, with **90.65% test accuracy** and a **90.82% F1-score**.

The final model is integrated into an interactive Streamlit application supporting individual and batch sentiment analysis.

## 2. Objectives

- Explore and understand the IMDb movie reviews dataset.
- Perform text cleaning and preprocessing.
- Extract numerical features using TF-IDF.
- Train and compare multiple machine learning classifiers.
- Evaluate model performance using classification metrics.
- Develop an interactive sentiment analysis application.
- Support individual and batch review predictions.

## 3. Dataset Information

**Dataset:** IMDb Movie Reviews Dataset  
**Original records:** 50,000  
**Source:** [IMDb Movie Reviews Dataset on Zenodo](https://zenodo.org/records/7928582)

The dataset contains movie reviews labelled as Positive or Negative. The original dataset has 25,000 reviews in each class.

### Dataset preparation

The dataset is not included in this GitHub repository. To reproduce the data-processing and training workflow:

1. Download the dataset from the [Zenodo record](https://zenodo.org/records/7928582).
2. Extract or copy `IMDB Dataset.csv` into:

   ```text
   data/raw/IMDB Dataset.csv
   ```

3. Run the relevant scripts in the `src/` directory to clean, validate, split and process the dataset.

The `data/raw/` and `data/processed/` directories are excluded from Git. Processed CSV files are generated locally during the workflow and are not required to run the already-trained Streamlit application.

### Dataset summary

| Attribute | Details |
|---|---:|
| Original records | 50,000 |
| Features | Review, Sentiment |
| Target classes | Positive, Negative |
| Original class distribution | 25,000 each |
| Duplicate records removed | 418 |
| Final records after cleaning | 49,582 |
| Missing values | 0 |

### Final sentiment distribution

| Sentiment | Records | Percentage |
|---|---:|---:|
| Positive | 24,884 | 50.19% |
| Negative | 24,698 | 49.81% |

The cleaned dataset was divided into training and testing subsets using an 80:20 stratified split.

| Dataset | Records |
|---|---:|
| Training | 39,665 |
| Testing | 9,917 |

**Random state:** `42`

## 4. Exploratory Data Analysis

Exploratory Data Analysis was performed using Pandas, Matplotlib and Seaborn.

The analysis included:

- Dataset dimensions and data types.
- Missing and duplicate value analysis.
- Sentiment distribution.
- Review length distribution.
- Review length comparison by sentiment.
- HTML tag analysis.
- Punctuation and capitalization analysis.
- Most frequent words.
- Positive and negative vocabulary comparison.
- TF-IDF feature exploration.

### Key observations

1. The original dataset contains an equal number of positive and negative reviews.
2. HTML tags were present in approximately 58.4% of the original reviews.
3. There were 418 exact duplicate records.
4. Positive and negative reviews had relatively similar length distributions.
5. Words associated with positive and negative reviews showed different patterns.
6. TF-IDF was selected for numerical text representation.

EDA visualizations are available in the `images/` directory.

## 5. Text Preprocessing

A custom preprocessing function was implemented in `src/preprocess.py`.

The preprocessing pipeline performs:

1. Handling missing or invalid inputs.
2. HTML entity decoding.
3. HTML tag removal.
4. Lowercase conversion.
5. Contraction expansion.
6. Apostrophe normalization.
7. Punctuation and special-character removal.
8. Whitespace normalization.

### Important preprocessing decisions

- Negation words such as `not` are preserved.
- Contractions such as `can't` and `don't` are expanded.
- Hyphenated words are separated.
- HTML artifacts are removed.
- Original review text is retained alongside cleaned text.

The preprocessing function was tested using 19 test cases, along with additional diagnostic tests.

## 6. Feature Extraction – TF-IDF

Term Frequency–Inverse Document Frequency (TF-IDF) was used to convert textual reviews into numerical feature vectors.

Configuration:

| Parameter | Value |
|---|---:|
| `max_features` | 50,000 |
| `ngram_range` | (1, 2) |
| `min_df` | 2 |
| `max_df` | 0.95 |
| `sublinear_tf` | True |

Both unigrams and bigrams were considered.

TF-IDF was integrated into Scikit-learn Pipelines to ensure that vocabulary and IDF weights were learned exclusively from training data during model evaluation.

## 7. Machine Learning Algorithms

Two supervised classification algorithms were implemented.

### A. Multinomial Naive Bayes

Multinomial Naive Bayes is a probabilistic classification algorithm commonly used for text classification. It estimates class probabilities using feature distributions and Bayes' theorem.

It was implemented as the baseline algorithm.

### B. Logistic Regression

Logistic Regression is a supervised classification algorithm that learns feature weights to distinguish between positive and negative reviews.

It was selected as the final model after outperforming Multinomial Naive Bayes in cross-validation and test-set evaluation.

## 8. Model Evaluation

Both models were evaluated using five-fold stratified cross-validation on the training dataset.

### Cross-validation results

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Multinomial Naive Bayes | 88.29% | 88.29% | 88.29% | 88.29% |
| Logistic Regression | 90.58% | 90.60% | 90.58% | 90.58% |

### Final test-set results

The precision, recall and F1-score values below are for the **Positive** class.

| Metric | Multinomial Naive Bayes | Logistic Regression |
|---|---:|---:|
| Accuracy | 88.32% | **90.65%** |
| Positive-class precision | 88.09% | **89.58%** |
| Positive-class recall | 88.73% | **92.08%** |
| Positive-class F1-score | 88.41% | **90.82%** |

### Confusion Matrix – Logistic Regression

| Actual / Predicted | Negative | Positive |
|---|---:|---:|
| Negative | 4,407 | 533 |
| Positive | 394 | 4,583 |

### Final Model Selection

Logistic Regression was selected as the final model because it achieved better accuracy, positive-class recall and positive-class F1-score than Multinomial Naive Bayes.

**Final model performance:**

- Accuracy: 90.65%
- Positive-class precision: 89.58%
- Positive-class recall: 92.08%
- Positive-class F1-score: 90.82%

The final trained pipeline is saved as:

```text
models/final_sentiment_model.pkl
```

### Evaluation note

The test-set results were also considered during the final comparison and selection of the model. Therefore, the test set was not completely untouched during model selection. The reported test metrics should be interpreted with this limitation in mind. A stricter future evaluation would select the model using training data and cross-validation, lock the model, and then evaluate it once on a separate untouched test set.

## 9. Streamlit Application

An interactive web application was developed using Streamlit.

### Features

**Single Review Analysis**

- Enter an individual movie review.
- Predict positive or negative sentiment.
- Display model probability.
- Load sample reviews for demonstration.

**Batch Review Analysis**

- Submit multiple reviews.
- Generate individual predictions.
- Display positive and negative review counts.
- Visualize sentiment distribution.
- View individual model probabilities.
- Download prediction results as CSV.

**About Model**

- Dataset information.
- Training and testing statistics.
- Model architecture.
- Evaluation metrics.
- Technologies used.

### Application Architecture

```text
Movie Review
     ↓
Text Preprocessing
     ↓
TF-IDF Vectorization
     ↓
Logistic Regression
     ↓
Positive / Negative Prediction
```

## 10. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Scikit-learn | Machine learning and evaluation |
| NLTK | NLP toolkit |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Streamlit | Interactive web application |
| Joblib | Model serialization |
| Jupyter Notebook | Exploratory Data Analysis |

## 11. Project Structure

The following shows the project files tracked in the repository. Dataset CSV files are excluded and must be downloaded or generated locally as described in the Dataset Information section.

```text
02_Sentiment_Analysis/
│
├── .gitignore
├── app.py
├── README.md
├── requirements.txt
│
├── data/
│   ├── raw/                 # Download the original dataset here
│   └── processed/           # Generated locally during preprocessing
│
├── images/
│   ├── 01_sentiment_distribution.png
│   ├── 02_review_length_distribution.png
│   ├── 03_review_length_by_sentiment.png
│   ├── 04_top_20_frequent_words.png
│   ├── 05_sentiment_vocabulary_comparison.png
│   ├── 06_single_review_analysis.png
│   ├── 07_batch_review_analysis.png
│   └── 08_model_information.png
│
├── models/
│   ├── naive_bayes_model.pkl
│   ├── logistic_regression_model.pkl
│   └── final_sentiment_model.pkl
│
├── notebooks/
│   └── 01_Exploratory_Data_Analysis.ipynb
│
├── reports/
│   └── Sentiment_Analysis_Project_Report_Akhil_Marada.pdf
│
└── src/
    ├── inspect_dataset.py
    ├── analyze_data_quality.py
    ├── preprocess.py
    ├── test_preprocessing.py
    ├── process_dataset.py
    ├── validate_dataset.py
    ├── split_dataset.py
    ├── compare_models.py
    ├── train_model.py
    ├── evaluate_models.py
    └── save_final_model.py
```

## 12. Installation and Execution

### Prerequisites

- Python 3.10 or later.
- pip package manager.

### Step 1: Clone the repository

```bash
git clone <your-repository-url>
```

Navigate to the project directory:

```bash
cd Cloudcredits/02_Sentiment_Analysis
```

### Step 2: Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Launch the application

The trained model is included in the repository, so the Streamlit application can be launched without downloading the dataset.

```bash
streamlit run app.py
```

The application will open in your default browser.

**Note:** The application expects the trained model at `models/final_sentiment_model.pkl`.

### Reproducing the training workflow

To reproduce data preparation and model training, first download the IMDb dataset and place `IMDB Dataset.csv` in `data/raw/`. Then run the relevant scripts in `src/` in the project workflow order:

1. Inspect and analyze the dataset.
2. Preprocess and validate the cleaned data.
3. Split the data into training and testing sets.
4. Compare and train the models.
5. Evaluate the models and save the final model.

The exact scripts for these steps are available in the `src/` directory.

## 13. Limitations

- The classifier is trained on IMDb movie reviews and may not generalize equally well to every type of text.
- Mixed or sarcastic reviews can be difficult to classify.
- The model performs binary sentiment classification only.
- TF-IDF represents text using statistical features and does not fully understand contextual meaning.
- Prediction probabilities are model outputs and should not be interpreted as guaranteed probabilities of correctness.
- Some missing spaces and formatting inconsistencies exist in the original dataset.
- The test set was considered during model selection; an untouched holdout evaluation remains a useful future improvement.

## 14. Future Improvements

Potential extensions include:

- Experimenting with LSTM-based sentiment classification.
- Exploring transformer-based language models.
- Hyperparameter optimization.
- Probability calibration.
- Improving handling of sarcasm and mixed sentiment.
- Evaluating the selected model on a separate untouched test set.
- Deploying the application to a cloud platform.

## 15. Conclusion

This project demonstrates an end-to-end machine learning workflow for movie review sentiment classification.

It covers dataset exploration, text preprocessing, TF-IDF feature extraction, supervised model training, cross-validation, model evaluation and interactive application development.

Two classification algorithms were compared, with Logistic Regression achieving the best reported test performance of **90.65% accuracy and 90.82% positive-class F1-score**.

The final Streamlit application supports both individual and batch movie review sentiment analysis.

---

**Project:** Movie Review Sentiment Analysis  
**Organization:** Cloudcredits Technologies  
**Project Category:** Machine Learning and Natural Language Processing
