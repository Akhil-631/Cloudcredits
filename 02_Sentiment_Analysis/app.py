import streamlit as st
import pandas as pd
import joblib

from pathlib import Path
from src.preprocess import clean_text


# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_PATH = PROJECT_ROOT / "models" / "final_sentiment_model.pkl"

MAX_REVIEW_LENGTH = 10000
MAX_BATCH_REVIEWS = 100


# ==========================================
# 2. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Movie Review Sentiment Analyzer",
    page_icon="🎬",
    layout="wide"
)


# ==========================================
# 3. CUSTOM CSS
# ==========================================

st.markdown("""
<style>

.main-title {
    font-size: 2.5rem;
    font-weight: 750;
    margin-bottom: 5px;
}

.subtitle {
    color: #A0A0A0;
    font-size: 1.05rem;
}

.result-positive {
    padding: 18px;
    border-radius: 10px;
    background-color: rgba(46, 204, 113, 0.15);
    border-left: 5px solid #2ecc71;
    color: #2ecc71;
    font-size: 1.2rem;
    font-weight: 600;
}

.result-negative {
    padding: 18px;
    border-radius: 10px;
    background-color: rgba(231, 76, 60, 0.15);
    border-left: 5px solid #e74c3c;
    color: #ff6b6b;
    font-size: 1.2rem;
    font-weight: 600;
}

div[data-testid="stMetric"] {
    background-color: rgba(128, 128, 128, 0.08);
    padding: 15px;
    border-radius: 10px;
}

.footer {
    text-align: center;
    color: #888888;
    font-size: 0.85rem;
    padding: 20px;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# 4. LOAD TRAINED MODEL
# ==========================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    model = load_model()

except Exception as error:
    st.error(f"Unable to load the trained model: {error}")
    st.stop()


# ==========================================
# 5. PREDICTION FUNCTION
# ==========================================

def predict_sentiment(reviews):

    # Apply the same preprocessing used during training
    cleaned_reviews = [
        clean_text(review) for review in reviews
    ]

    # Generate predictions
    predictions = model.predict(cleaned_reviews)

    # Generate prediction probabilities
    probabilities = model.predict_proba(cleaned_reviews)

    results = []

    for review, prediction, probability in zip(
        reviews, predictions, probabilities
    ):

        confidence = probability[
            list(model.classes_).index(prediction)
        ] * 100

        results.append({
            "Review": review,
            "Predicted Sentiment": prediction.capitalize(),
            "Model Probability (%)": round(confidence, 2)
        })

    return pd.DataFrame(results)


# ==========================================
# 6. APPLICATION HEADER
# ==========================================

st.markdown(
    '<div class="main-title">🎬 Movie Review Sentiment Analyzer</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'An NLP-powered application for classifying movie reviews '
    'as positive or negative.'
    '</div>',
    unsafe_allow_html=True
)

st.write("")

# Application overview
col1, col2, col3 = st.columns(3)

col1.metric("Final Algorithm", "Logistic Regression")
col2.metric("Test Accuracy", "90.65%")
col3.metric("Test F1-score", "90.82%")

st.divider()


# ==========================================
# 7. APPLICATION TABS
# ==========================================

single_tab, batch_tab, about_tab = st.tabs([
    "📝 Single Review",
    "📊 Batch Analysis",
    "ℹ️ About Model"
])


# ==========================================
# 8. SINGLE REVIEW ANALYSIS
# ==========================================

with single_tab:

    st.subheader("Analyze a Movie Review")

    sample_reviews = {
        "Positive Example": (
            "This movie was absolutely fantastic. "
            "The acting was brilliant and the storyline "
            "was engaging. I enjoyed every minute of it."
        ),

        "Negative Example": (
            "This was one of the worst movies I have ever watched. "
            "The acting was terrible and the storyline was boring."
        ),

        "Mixed Example": (
            "The acting was excellent, but the storyline "
            "was predictable and the ending was disappointing."
        )
    }

    selected_example = st.selectbox(
        "Try an example review:",
        ["Choose an example"] + list(sample_reviews.keys())
    )

    if st.button("Load Example Review"):

        if selected_example != "Choose an example":

            st.session_state["single_review"] = (
                sample_reviews[selected_example]
            )

    review = st.text_area(
        "Enter your movie review:",
        height=180,
        max_chars=MAX_REVIEW_LENGTH,
        key="single_review",
        placeholder="Write or paste your movie review here..."
    )

    if st.button(
        "Analyze Sentiment",
        type="primary",
        key="single_analysis"
    ):

        if not review.strip():

            st.warning("Please enter a movie review.")

        else:

            result = predict_sentiment([review])

            sentiment = result.iloc[0]["Predicted Sentiment"]

            probability = result.iloc[0]["Model Probability (%)"]

            st.divider()

            st.subheader("Prediction Result")

            if sentiment == "Positive":

                st.markdown(
                    '<div class="result-positive">'
                    '😊 Positive Review'
                    '</div>',
                    unsafe_allow_html=True
                )

            else:

                st.markdown(
                    '<div class="result-negative">'
                    '😞 Negative Review'
                    '</div>',
                    unsafe_allow_html=True
                )

            st.write("")

            st.metric(
                "Model Probability",
                f"{probability:.2f}%"
            )

            st.progress(int(probability))


# ==========================================
# 9. BATCH SENTIMENT ANALYSIS
# ==========================================

with batch_tab:

    st.subheader("Analyze Multiple Movie Reviews")

    st.write(
        "Enter multiple movie reviews. "
        "Place each review on a separate line."
    )

    batch_input = st.text_area(
        "Enter your reviews:",
        height=250,
        placeholder=(
            "This movie was amazing!\n"
            "The acting was terrible.\n"
            "I really enjoyed this film."
        ),
        key="batch_input"
    )

    if st.button(
        "Analyze All Reviews",
        type="primary",
        key="batch_analysis"
    ):

        reviews = [
            line.strip()
            for line in batch_input.splitlines()
            if line.strip()
        ]

        if not reviews:

            st.warning("Please enter at least one review.")

        elif len(reviews) > MAX_BATCH_REVIEWS:

            st.warning(
                f"You can analyse a maximum of "
                f"{MAX_BATCH_REVIEWS} reviews at a time."
            )

        elif any(len(review) > MAX_REVIEW_LENGTH for review in reviews):

            st.warning(
                f"Each review must contain at most "
                f"{MAX_REVIEW_LENGTH:,} characters."
            )

        else:

            results_df = predict_sentiment(reviews)

            positive_count = (
                results_df["Predicted Sentiment"] == "Positive"
            ).sum()

            negative_count = (
                results_df["Predicted Sentiment"] == "Negative"
            ).sum()

            st.divider()

            st.subheader("Batch Analysis Summary")

            col1, col2, col3 = st.columns(3)

            col1.metric("Total Reviews", len(results_df))

            col2.metric("Positive Reviews", positive_count)

            col3.metric("Negative Reviews", negative_count)

            st.subheader("Sentiment Distribution")

            chart_data = pd.DataFrame({
                "Sentiment": ["Positive", "Negative"],
                "Count": [positive_count, negative_count]
            })

            st.bar_chart(
                chart_data.set_index("Sentiment")
            )

            st.subheader("Individual Predictions")

            st.dataframe(
                results_df,
                use_container_width=True,
                hide_index=True
            )

            csv_data = results_df.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                "📥 Download Results as CSV",
                data=csv_data,
                file_name="sentiment_analysis_results.csv",
                mime="text/csv"
            )


# ==========================================
# 10. MODEL INFORMATION
# ==========================================

with about_tab:

    st.subheader("About the Machine Learning Model")

    st.markdown("""
    This application uses Natural Language Processing (NLP)
    and supervised machine learning to classify movie reviews.
    """)

    st.markdown("### Dataset Information")

    dataset_info = {
        "Dataset": "IMDb Movie Reviews",
        "Original Reviews": "50,000",
        "Reviews After Duplicate Removal": "49,582",
        "Positive Reviews": "24,884",
        "Negative Reviews": "24,698",
        "Training Records": "39,665",
        "Testing Records": "9,917"
    }

    for key, value in dataset_info.items():

        st.write(f"**{key}:** {value}")

    st.divider()

    st.markdown("### Model Architecture")

    st.code(
        """
Movie Review
     |
Text Preprocessing
     |
TF-IDF Vectorization
     |
Logistic Regression
     |
Positive / Negative Prediction
        """,
        language="text"
    )

    st.markdown("### Final Model Performance")

    col1, col2, col3 = st.columns(3)

    col1.metric("Accuracy", "90.65%")
    col2.metric("Precision", "89.58%")
    col3.metric("Recall", "92.08%")

    st.metric("F1-score", "90.82%")

    st.markdown("### Technologies Used")

    st.write(
        "Python · Pandas · Scikit-learn · "
        "NLTK · TF-IDF · Logistic Regression · Streamlit"
    )

    st.info(
        "Multinomial Naive Bayes was also evaluated as a baseline "
        "and achieved 88.32% test accuracy."
    )


# ==========================================
# 11. FOOTER
# ==========================================

st.divider()

st.markdown(
    '<div class="footer">'
    'Cloudcredits Internship Project | '
    'IMDb Movie Reviews Dataset | '
    'TF-IDF + Logistic Regression'
    '</div>',
    unsafe_allow_html=True
)