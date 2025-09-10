import streamlit as st
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt

# -----------------------------
# Load pipeline
# -----------------------------
model = joblib.load("artifacts/model_xgb_pipeline.joblib")

st.set_page_config(page_title="ChurnShield", layout="wide")
st.title("📊 ChurnShield Dashboard")
st.write("Predict customer churn and analyze feature importance")

# -----------------------------
# Load raw data for feature selection
# -----------------------------
df = pd.read_csv("data/raw/telecom_customer_churn.csv")
features = df.drop(columns=["Churn"]).columns.tolist()

# -----------------------------
# Collect user input dynamically
# -----------------------------
st.sidebar.header("Customer Info")
input_data = {}
for col in features:
    if df[col].dtype == "object":
        input_data[col] = st.sidebar.selectbox(col, df[col].unique())
    elif df[col].dtype in ["int64", "float64"]:
        min_val = int(df[col].min())
        max_val = int(df[col].max())
        input_data[col] = st.sidebar.slider(col, min_val, max_val, int(df[col].median()))

# Convert input to DataFrame
input_df = pd.DataFrame([input_data])

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

# Get preprocessor
preprocessor = model.named_steps["preprocessor"]

# Transform input for SHAP
transformed_input = preprocessor.transform(input_df)

# Get actual feature names after preprocessing
if hasattr(preprocessor, "get_feature_names_out"):
    transformed_features = preprocessor.get_feature_names_out()
else:
    transformed_features = features  # fallback

# Explain predictions
explainer = shap.TreeExplainer(model.named_steps["classifier"])
shap_values = explainer.shap_values(transformed_input)

# Headless plotting
plt.figure(figsize=(8,6))
shap.summary_plot(shap_values, transformed_input, feature_names=transformed_features, show=False)
st.pyplot(plt.gcf())
plt.close()
