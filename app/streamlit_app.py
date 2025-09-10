import streamlit as st
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt

# -----------------------------
# Load pipeline
# -----------------------------
model = joblib.load("artifacts/advanced_churn_pipeline.joblib")

st.set_page_config(page_title="ChurnShield", layout="wide")
st.title("📊 ChurnShield Dashboard")
st.write("Predict customer churn and analyze feature importance")

# -----------------------------
# Helper functions
# -----------------------------
def encode_input(df):
    """Encode user input to match training preprocessing."""
    # Yes/No mapping
    yes_no_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection"]
    for col in yes_no_cols:
        if col in df.columns:
            df[col] = df[col].map({"Yes": 1, "No": 0})
    
    # tenure_group mapping (optional if you trained without it)
    # Contract mapping
    contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
    if "Contract" in df.columns:
        df["Contract"] = df["Contract"].map(contract_map)
    
    # PaymentMethod mapping
    payment_map = {
        "Electronic check": 0,
        "Mailed check": 1,
        "Bank transfer (automatic)": 2,
        "Credit card (automatic)": 3
    }
    if "PaymentMethod" in df.columns:
        df["PaymentMethod"] = df["PaymentMethod"].map(payment_map)
    
    # InternetService mapping
    internet_map = {"DSL": 0, "Fiber optic": 1, "No": 2}
    if "InternetService" in df.columns:
        df["InternetService"] = df["InternetService"].map(internet_map)
    
    # Gender mapping
    gender_map = {"Male": 0, "Female": 1}
    if "gender" in df.columns:
        df["gender"] = df["gender"].map(gender_map)
    
    return df

# -----------------------------
# Collect user input
# -----------------------------
st.sidebar.header("Customer Info")

customer = {}
customer["gender"] = st.sidebar.selectbox("Gender", ["Male", "Female"])
customer["SeniorCitizen"] = st.sidebar.selectbox("Senior Citizen", [0, 1])
customer["tenure"] = st.sidebar.number_input("Tenure (months)", min_value=0, max_value=72, step=1)
customer["MonthlyCharges"] = st.sidebar.number_input("Monthly Charges", min_value=0.0, step=1.0)
customer["TotalCharges"] = st.sidebar.number_input("Total Charges", min_value=0.0, step=1.0)

customer["Contract"] = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
customer["PaymentMethod"] = st.sidebar.selectbox("Payment Method", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])
customer["InternetService"] = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
customer["OnlineSecurity"] = st.sidebar.selectbox("Online Security", ["Yes", "No"])
customer["OnlineBackup"] = st.sidebar.selectbox("Online Backup", ["Yes", "No"])
customer["DeviceProtection"] = st.sidebar.selectbox("Device Protection", ["Yes", "No"])

# Convert to DataFrame
input_df = pd.DataFrame([customer])

# Encode input to numeric
input_df = encode_input(input_df)

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
