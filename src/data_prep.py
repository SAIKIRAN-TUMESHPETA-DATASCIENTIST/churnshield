import pandas as pd
from sklearn.model_selection import train_test_split

def load_raw(path="data/raw/telecom_customer_churn.csv"):
    df = pd.read_csv(path)
    return df

def basic_clean(df):
    df = df.drop_duplicates()
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.ffill().bfill()
    return df

def split(df, target="Churn", test_size=0.2, random_state=42):
    X = df.drop(columns=[target])
    y = df[target].apply(lambda v: 1 if str(v).strip().lower() in ("yes","y","true","1") else 0)
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

