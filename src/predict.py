import joblib
import pandas as pd

def load_model(path="artifacts/advanced_churn_pipeline.joblib"):
    return joblib.load(path)

def predict(df, pipeline):
    return pipeline.predict(df), pipeline.predict_proba(df)[:,1]
