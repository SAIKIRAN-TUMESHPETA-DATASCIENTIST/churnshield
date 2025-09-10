from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import joblib, os

def build_pipeline(numeric_cols, categorical_cols, model_type="xgb"):
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ])

    if model_type == "xgb":
        clf = XGBClassifier(random_state=42, eval_metric="logloss", n_jobs=-1)
    else:
        raise ValueError("Currently only XGBClassifier supported")

    pipeline = ImbPipeline([
        ("preprocessor", preprocessor),
        ("smote", SMOTE(random_state=42)),
        ("classifier", clf)
    ])
    return pipeline

def save_model(pipeline, name="model_pipeline.joblib", artifact_dir="artifacts"):
    os.makedirs(artifact_dir, exist_ok=True)
    joblib.dump(pipeline, os.path.join(artifact_dir, name))
