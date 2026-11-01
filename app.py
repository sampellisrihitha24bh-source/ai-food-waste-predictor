import streamlit as st
import pandas as pd
import joblib

# PAGE CONFIGURATION
st.set_page_config(
    page_title="AI Food Waste Predictor",
    page_icon="🌱",
    layout="wide"
)

# LOAD TRAINED MODELS
@st.cache_resource
def load_models():
    rf = joblib.load("random_forest_model.pkl")
    lr = joblib.load("linear_regression_model.pkl")
    return rf, lr

rf_model, lr_model = load_models()

# PROFESSIONAL DESIGN
st.markdown("""
<style>
.stApp {
    background-color: #f5f8f5;
}
.main-title {
    font-size: 38px;
    font-weight: 800;
    color: #174d35;
}
.subtitle {
    color: #607568;
    font-size: 16px;
}
div.stButton > button {
    background-color: #176b45;
    color: white;
    border-radius: 10px;
    border: none;
    height: 48px;
    font-weight: bold;
    width: 100%;
}
div.stButton > button:hover {
    background-color: #104d32;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# HEADER
st.markdown(
    '<div class="main-title">🌱 AI Food Waste Predictor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Smart food waste prediction using Machine Learning</div>',
    unsafe_allow_html=True
)

st.divider()

# SIDEBAR
st.sidebar.title("🌿 Navigation")
page = st.sidebar.radio(
    "Choose a page",
    ["Food Waste Prediction", "About the Project"]
)

if page == "Food Waste Prediction":

    st.subheader("📋 Enter Food Information")

    with st.form("prediction_form"):

        col1, col2 = st.columns(2)

        with col1:
            day = st.selectbox(
                "Day of the Week",
                ["Monday", "Tuesday", "Wednesday", "Thursday",
                 "Friday", "Saturday", "Sunday"]
            )

            meal = st.selectbox(
                "Meal Type",
                ["Breakfast", "Lunch", "Dinner"]
            )

            food = st.selectbox(
                "Food Type",
                ["Biryani", "Chapati", "Dal", "Idli",
                 "Pulao", "Rice", "Vegetable Curry"]
            )

            customers = st.number_input(
                "Expected Customers",
                min_value=1,
                max_value=10000,
                value=150
            )

            portion = st.number_input(
                "Portion Size (grams)",
                min_value=1,
                max_value=2000,
                value=250
            )

        with col2:
            prepared = st.number_input(
                "Planned Food Prepared (kg)",
                min_value=0.1,
                max_value=10000.0,
                value=40.0,
                step=1.0
            )

            previous_waste = st.number_input(
                "Previous Waste of Same Food (kg)",
                min_value=0.0,
                max_value=10000.0,
                value=5.0,
                step=0.5
            )

            storage_issue = st.selectbox(
                "Known Storage Issue?",
                ["No", "Yes"]
            )

            storage_duration = st.number_input(
                "Storage Duration (hours)",
                min_value=0.0,
                max_value=1000.0,
                value=2.0,
                step=0.5
            )

            food_cost = st.number_input(
                "Estimated Food Cost (₹ per kg)",
                min_value=1,
                max_value=100000,
                value=80
            )

        submitted = st.form_submit_button(
            "🌱 Predict Food Waste"
        )

    if submitted:

        input_data = pd.DataFrame([{
            "day_of_week": day,
            "meal_type": meal,
            "food_type": food,
            "expected_customers": customers,
            "portion_size_g": portion,
            "planned_food_prepared_kg": prepared,
            "previous_waste_same_food_kg": previous_waste,
            "storage_issue_known_before_service":
                1 if storage_issue == "Yes" else 0,
            "storage_duration_hours": storage_duration
        }])

        # MODEL PREDICTIONS
        rf_prediction = float(rf_model.predict(input_data)[0])
        lr_prediction = float(lr_model.predict(input_data)[0])

        # Do not display physically impossible negative waste
        rf_prediction = max(0, rf_prediction)
        lr_prediction = max(0, lr_prediction)

        st.divider()
        st.subheader("📊 Prediction Results")

        result1, result2 = st.columns(2)

        with result1:
            st.metric(
                "Random Forest Prediction",
                f"{rf_prediction:.2f} kg"
            )

        with result2:
            st.metric(
                "Linear Regression Prediction",
                f"{lr_prediction:.2f} kg"
            )

               
         # Use Linear Regression as the preliminary best model
        final_prediction = max(0, lr_prediction)
        estimated_cost = final_prediction * food_cost

        st.markdown("### Final Estimated Food Waste")
        st.success(
            f"Final Predicted Waste: {final_prediction:.2f} kg"
        )

        st.metric(
            "Estimated Food Cost Loss",
            f"₹{estimated_cost:.2f}"
        )

        # COMPARISON CHART
        st.subheader("Model Performance Comparison")

        chart_data = pd.DataFrame({
            "Model": ["Random Forest", "Linear Regression"],
            "Predicted Waste (kg)": [
                rf_prediction,
                lr_prediction
            ]
        })

        st.bar_chart(
            chart_data.set_index("Model")
        )

        st.info(
            "Prediction estimates are based on the trained "
            "machine-learning models. Actual food waste may vary."
        )

elif page == "About the Project":

    st.subheader("About Our Project")

    st.write("""
    The AI Food Waste Predictor is a machine-learning application
    designed to estimate food waste using food preparation,
    customer demand and storage-related information.

    The objective is to help food service organisations make
    better food preparation decisions and reduce unnecessary waste.
    """)

    st.subheader("Machine Learning Algorithms")

    st.write("""
    **1. Random Forest Regression**

    Uses multiple decision trees to predict food waste.

    **2. Linear Regression**

    Estimates food waste using relationships between input variables.
    """)

    st.subheader("Model Evaluation")

    results = pd.DataFrame({
        "Evaluation Metric": ["MAE", "RMSE", "R² Score"],
        "Random Forest": [2.027, 2.948, 0.487],
        "Linear Regression": [1.998, 2.769, 0.547]
    })

    st.dataframe(results, hide_index=True)

    st.success(
        "Linear Regression performed better on our current test set."
    )

    st.subheader("Dataset Information")
    st.write("Total Records: 300")
    st.write("Training Records: 240")
    st.write("Testing Records: 60")
    st.write("Prediction Target: Total Food Waste (kg)")

st.divider()

st.caption("AI Food Waste Predictor | NCP-3 Academic Project")