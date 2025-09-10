import streamlit as st
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
import uuid

# -----------------------------
# Load pipeline
# -----------------------------
model = joblib.load("artifacts/advanced_churn_pipeline.joblib")

st.set_page_config(page_title="ChurnShield", layout="wide")
st.title("📊 ChurnShield Dashboard")
st.write("Predict customer churn and analyze feature importance")

# -----------------------------
# Collect user input
# -----------------------------
st.sidebar.header("Customer Info")

customer = {}
customer["customerID"] = str(uuid.uuid4())[:8]  # auto-generate short ID

customer["gender"] = st.sidebar.selectbox("Gender", ["Male", "Female"])
customer["SeniorCitizen"] = st.sidebar.selectbox("Senior Citizen", [0, 1])
customer["Partner"] = st.sidebar.selectbox("Partner", ["Yes", "No"])
customer["Dependents"] = st.sidebar.selectbox("Dependents", ["Yes", "No"])

customer["tenure"] = st.sidebar.number_input("Tenure (months)", min_value=0, max_value=72, step=1)

customer["PhoneService"] = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
customer["MultipleLines"] = st.sidebar.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])

customer["InternetService"] = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
customer["OnlineSecurity"] = st.sidebar.selectbox("Online Security", ["Yes", "No", "No internet service"])
customer["OnlineBackup"] = st.sidebar.selectbox("Online Backup", ["Yes", "No", "No internet service"])
customer["DeviceProtection"] = st.sidebar.selectbox("Device Protection", ["Yes", "No", "No internet service"])
customer["TechSupport"] = st.sidebar.selectbox("Tech Support", ["Yes", "No", "No internet service"])
customer["StreamingTV"] = st.sidebar.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
customer["StreamingMovies"] = st.sidebar.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

customer["Contract"] = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
customer["PaperlessBilling"] = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
customer["PaymentMethod"] = st.sidebar.selectbox("Payment Method", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])

customer["MonthlyCharges"] = st.sidebar.number_input("Monthly Charges", min_value=0.0, step=1.0)
customer["TotalCharges"] = st.sidebar.number_input("Total Charges", min_value=0.0, step=1.0)

# Derived features
if customer["tenure"] <= 12:
    customer["tenure_group"] = "0-12"
elif customer["tenure"] <= 24:
    customer["tenure_group"] = "13-24"
elif customer["tenure"] <= 48:
    customer["tenure_group"] = "25-48"
elif customer["tenure"] <= 60:
    customer["tenure_group"] = "49-60"
else:
    customer["tenure_group"] = "61+"

customer["monthly_per_tenure"] = (
    customer["MonthlyCharges"] / customer["tenure"] if customer["tenure"] > 0 else 0
)

# Convert to DataFrame
input_df = pd.DataFrame([customer])

# -----------------------------
# Prediction
# -----------------------------
pred_prob = model.predict_proba(input_df)[0][1]
pred_class = model.predict(input_df)[0]

st.subheader("Prediction")
st.write(f"**Churn Probability:** {pred_prob:.2f}")
st.write(f"**Churn Prediction:** {'Yes' if pred_class==1 else 'No'}")

# -----------------------------
# SHAP Feature Importance
# -----------------------------
st.subheader("Feature Importance")

preprocessor = model.named_steps["preprocessor"]
transformed_input = preprocessor.transform(input_df)

if hasattr(preprocessor, "get_feature_names_out"):
    transformed_features = preprocessor.get_feature_names_out()
else:
    transformed_features = input_df.columns

explainer = shap.TreeExplainer(model.named_steps["classifier"])
shap_values = explainer.shap_values(transformed_input)

plt.figure(figsize=(8,6))
shap.summary_plot(shap_values, transformed_input, feature_names=transformed_features, show=False)
st.pyplot(plt.gcf())
plt.close()
