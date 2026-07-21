
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Rental Price Prediction",
    page_icon="🏠",
    layout="centered"
)

st.title("🏠 Rental Price Prediction")
st.write(
    "Enter the property details below to estimate the rental price."
)


# --------------------------------------------------
# Load the trained model
# --------------------------------------------------

MODEL_PATH = Path(__file__).parent / "rental_price_prediction_model_v1_0.joblib"


@st.cache_resource
def load_model():
    """
    Load the serialized model once and keep it cached.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file was not found: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


try:
    model = load_model()
except Exception as error:
    st.error(f"Unable to load the model: {error}")
    st.stop()


# --------------------------------------------------
# User input form
# --------------------------------------------------

with st.form("prediction_form"):

    accommodates = st.number_input(
        "Number of guests accommodated",
        min_value=1,
        max_value=16,
        value=2,
        step=1
    )

    bathrooms = st.number_input(
        "Number of bathrooms",
        min_value=0.0,
        max_value=8.0,
        value=1.0,
        step=0.5
    )

    bedrooms = st.number_input(
        "Number of bedrooms",
        min_value=0,
        max_value=10,
        value=1,
        step=1
    )

    beds = st.number_input(
        "Number of beds",
        min_value=0,
        max_value=18,
        value=1,
        step=1
    )

    review_scores_rating = st.number_input(
        "Review score",
        min_value=20.0,
        max_value=100.0,
        value=95.0,
        step=1.0
    )

    room_type = st.selectbox(
        "Room type",
        options=[
            "Entire home/apt",
            "Private room",
            "Shared room"
        ]
    )

    cancellation_policy = st.selectbox(
        "Cancellation policy",
        options=[
            "flexible",
            "moderate",
            "strict"
        ]
    )

    cleaning_fee = st.selectbox(
        "Cleaning fee",
        options=[True, False]
    )

    instant_bookable = st.selectbox(
        "Instant bookable",
        options=["t", "f"]
    )

    submitted = st.form_submit_button(
        "Predict Rental Price"
    )


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if submitted:

    input_data = pd.DataFrame({
        "accommodates": [accommodates],
        "bathrooms": [bathrooms],
        "review_scores_rating": [review_scores_rating],
        "bedrooms": [bedrooms],
        "beds": [beds],
        "room_type": [room_type],
        "cancellation_policy": [cancellation_policy],
        "cleaning_fee": [cleaning_fee],
        "instant_bookable": [instant_bookable]
    })

    try:
        prediction = model.predict(input_data)

        predicted_value = float(prediction[0])

        # Use this only if the model was trained to predict log_price
        predicted_price = np.exp(predicted_value)

        st.success(
            f"Estimated rental price: ${predicted_price:,.2f}"
        )

        with st.expander("View submitted data"):
            st.dataframe(input_data, use_container_width=True)

        with st.expander("View raw model output"):
            st.write(f"Predicted log price: {predicted_value:.4f}")

    except Exception as error:
        st.error(f"Prediction failed: {error}")
