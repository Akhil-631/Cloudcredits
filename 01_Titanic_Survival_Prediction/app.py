import streamlit as st
import pandas as pd

from src.predict import predict_survival


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# --------------------------------------------------
# LOAD PREDICTION MODULE
# --------------------------------------------------

@st.cache_resource
def initialize_prediction_module():
    from src.predict import load_model
    return load_model()


model = initialize_prediction_module()


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("🚢 Titanic Survival Prediction")

st.markdown(
    """
    Predict the probability of survival for a Titanic passenger
    using a trained Machine Learning model.
    """
)

st.divider()


# --------------------------------------------------
# SAMPLE PASSENGER PRESETS
# --------------------------------------------------

st.subheader("👤 Passenger Information")

preset = st.selectbox(
    "Choose a sample passenger (optional)",
    [
        "Custom Passenger",
        "First-Class Female",
        "Third-Class Male",
        "Second-Class Child"
    ]
)

presets = {
    "First-Class Female": {
        "pclass": 1,
        "sex": "female",
        "age": 30.0,
        "sibsp": 0,
        "parch": 0,
        "fare": 80.0,
        "embarked": "C"
    },

    "Third-Class Male": {
        "pclass": 3,
        "sex": "male",
        "age": 25.0,
        "sibsp": 0,
        "parch": 0,
        "fare": 8.0,
        "embarked": "S"
    },

    "Second-Class Child": {
        "pclass": 2,
        "sex": "female",
        "age": 8.0,
        "sibsp": 1,
        "parch": 1,
        "fare": 26.0,
        "embarked": "S"
    }
}

selected = presets.get(preset, {})


# --------------------------------------------------
# PASSENGER INPUT FORM
# --------------------------------------------------

with st.form("passenger_form"):

    st.markdown("#### Basic Information")

    col1, col2 = st.columns(2)

    with col1:

        pclass = st.selectbox(
            "Passenger Class",
            [1, 2, 3],
            index=[1, 2, 3].index(selected.get("pclass", 1))
        )

        sex = st.selectbox(
            "Gender",
            ["male", "female"],
            index=["male", "female"].index(
                selected.get("sex", "male")
            )
        )

        age = st.number_input(
            "Age",
            min_value=0.0,
            max_value=100.0,
            value=float(selected.get("age", 25.0)),
            step=1.0
        )

        sibsp = st.number_input(
            "Siblings / Spouses Aboard",
            min_value=0,
            max_value=10,
            value=int(selected.get("sibsp", 0)),
            step=1
        )

    with col2:

        parch = st.number_input(
            "Parents / Children Aboard",
            min_value=0,
            max_value=10,
            value=int(selected.get("parch", 0)),
            step=1
        )

        fare = st.number_input(
            "Ticket Fare",
            min_value=0.0,
            max_value=600.0,
            value=float(selected.get("fare", 32.0)),
            step=1.0
        )

        embarked = st.selectbox(
            "Port of Embarkation",
            ["S", "C", "Q"],
            index=["S", "C", "Q"].index(
                selected.get("embarked", "S")
            ),
            format_func=lambda x: {
                "S": "Southampton",
                "C": "Cherbourg",
                "Q": "Queenstown"
            }[x]
        )

    st.divider()

    submitted = st.form_submit_button(
        "🚀 Predict Survival",
        use_container_width=True,
        type="primary"
    )


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if submitted:

    passenger = {
        "pclass": pclass,
        "sex": sex,
        "age": age,
        "sibsp": sibsp,
        "parch": parch,
        "fare": fare,
        "embarked": embarked
    }

    result = predict_survival(passenger, model=model)

    prediction = result["prediction"]

    survival_probability = result["survival_probability"]

    death_probability = result["non_survival_probability"]

    st.divider()

    st.subheader("📊 Prediction Results")

    result_col1, result_col2 = st.columns(2)

    with result_col1:

        if prediction == 1:
            st.success("### ✅ Predicted: Survived")
        else:
            st.error("### ❌ Predicted: Did Not Survive")

    with result_col2:

        st.metric(
            "Survival Probability",
            f"{survival_probability * 100:.2f}%"
        )

    st.progress(float(survival_probability))

    prob_col1, prob_col2 = st.columns(2)

    with prob_col1:
        st.metric(
            "Probability of Survival",
            f"{survival_probability * 100:.2f}%"
        )

    with prob_col2:
        st.metric(
            "Probability of Not Surviving",
            f"{death_probability * 100:.2f}%"
        )

    # Passenger summary

    with st.expander("View Passenger Details"):

        summary = pd.DataFrame({
            "Feature": [
                "Passenger Class",
                "Gender",
                "Age",
                "Siblings / Spouses",
                "Parents / Children",
                "Ticket Fare",
                "Port of Embarkation"
            ],

            "Value": [
                pclass,
                sex.title(),
                age,
                sibsp,
                parch,
                f"{fare:.2f}",
                embarked
            ]
        })

        st.table(summary)


# --------------------------------------------------
# PROJECT INFORMATION
# --------------------------------------------------

st.divider()

with st.expander("About This Project"):

    st.markdown(
        """
        **Project:** Titanic Survival Prediction

        **Objective:** Predict passenger survival using
        historical Titanic passenger information.

        **Machine Learning Algorithm:** Logistic Regression

        **Libraries:** Python, Pandas, Scikit-learn,
        Joblib and Streamlit.

        **Model:** Trained and saved as a serialized
        Scikit-learn pipeline.
        """
    )

st.caption(
    "Developed as part of the Cloudcredits Technologies "
    "Machine Learning & Artificial Intelligence Internship."
)