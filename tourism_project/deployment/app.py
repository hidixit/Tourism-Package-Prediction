"""
Streamlit App: Wellness Tourism Package Purchase Predictor
--------------------------------------------------------------
Loads the model trained and committed by the MLOps pipeline, collects
customer details from the user, and predicts whether the customer is
likely to purchase the new Wellness Tourism Package.
"""

import os
import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_model.joblib")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


st.set_page_config(page_title="Wellness Tourism Package Predictor", page_icon="🧘")

st.title("🧘 Wellness Tourism Package - Purchase Predictor")
st.write(
    "This app predicts whether a customer is likely to purchase the "
    "newly introduced **Wellness Tourism Package**, so the sales team "
    "can prioritize outreach before contacting them."
)

st.header("Customer Details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    type_of_contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
    city_tier = st.selectbox("City Tier", [1, 2, 3])
    occupation = st.selectbox(
        "Occupation", ["Salaried", "Free Lancer", "Small Business", "Large Business"]
    )
    gender = st.selectbox("Gender", ["Male", "Female"])
    number_of_person_visiting = st.number_input(
        "Number of Persons Visiting", min_value=1, max_value=10, value=2
    )
    number_of_followups = st.number_input(
        "Number of Follow-ups", min_value=0, max_value=10, value=3
    )
    product_pitched = st.selectbox(
        "Product Pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"]
    )
    preferred_property_star = st.selectbox("Preferred Property Star", [3.0, 4.0, 5.0])
    duration_of_pitch = st.number_input(
        "Duration of Pitch (minutes)", min_value=1, max_value=60, value=10
    )

with col2:
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
    number_of_trips = st.number_input(
        "Average Number of Trips per Year", min_value=0, max_value=20, value=3
    )
    passport = st.selectbox("Holds Passport?", ["Yes", "No"])
    pitch_satisfaction_score = st.slider("Pitch Satisfaction Score", 1, 5, 3)
    own_car = st.selectbox("Owns a Car?", ["Yes", "No"])
    number_of_children_visiting = st.number_input(
        "Number of Children Visiting (below age 5)", min_value=0, max_value=5, value=0
    )
    designation = st.selectbox(
        "Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"]
    )
    monthly_income = st.number_input(
        "Monthly Income", min_value=1000, max_value=200000, value=20000, step=500
    )

input_df = pd.DataFrame(
    [
        {
            "Age": age,
            "TypeofContact": type_of_contact,
            "CityTier": city_tier,
            "DurationOfPitch": duration_of_pitch,
            "Occupation": occupation,
            "Gender": gender,
            "NumberOfPersonVisiting": number_of_person_visiting,
            "NumberOfFollowups": number_of_followups,
            "ProductPitched": product_pitched,
            "PreferredPropertyStar": preferred_property_star,
            "MaritalStatus": marital_status,
            "NumberOfTrips": number_of_trips,
            "Passport": 1 if passport == "Yes" else 0,
            "PitchSatisfactionScore": pitch_satisfaction_score,
            "OwnCar": 1 if own_car == "Yes" else 0,
            "NumberOfChildrenVisiting": number_of_children_visiting,
            "Designation": designation,
            "MonthlyIncome": monthly_income,
        }
    ]
)

st.header("Input Summary")
st.dataframe(input_df)

if st.button("Predict Purchase Likelihood"):
    model = load_model()
    prediction = model.predict(input_df)[0]
    try:
        probability = model.predict_proba(input_df)[0][1]
    except Exception:
        probability = None

    if prediction == 1:
        st.success("✅ This customer is **likely to purchase** the Wellness Tourism Package.")
    else:
        st.warning("❌ This customer is **unlikely to purchase** the Wellness Tourism Package.")

    if probability is not None:
        st.write(f"Predicted purchase probability: **{probability:.2%}**")
