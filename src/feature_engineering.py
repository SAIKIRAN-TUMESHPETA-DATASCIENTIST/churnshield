import pandas as pd
from sklearn.preprocessing import LabelEncoder

def create_features(df):
    df = df.copy()
    if "tenure" in df.columns:
        df["tenure_group"] = pd.cut(df["tenure"], bins=[0,12,24,48,60,72], labels=False)
    if "MonthlyCharges" in df.columns and "tenure" in df.columns:
        df["monthly_per_tenure"] = df["MonthlyCharges"] / (df["tenure"]+1)
    df = df.ffill().bfill()
    return df

def encode_features(df, encoders=None):
    df = df.copy()
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    encoders = encoders or {}

    for col in cat_cols:
        if col not in encoders:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        else:
            le = encoders[col]
            df[col] = df[col].map(lambda s: le.transform([s])[0] if s in le.classes_ else -1)
    return df, encoders
