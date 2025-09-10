import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from src.data_prep import load_raw, basic_clean
from src.feature_engineering import create_features, encode_features
from src.model_utils import build_pipeline, save_model

df = load_raw()
df = basic_clean(df)
df = create_features(df)
df, encoders = encode_features(df)

X = df.drop(columns=["Churn"])
y = df["Churn"]

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

numeric_cols = X_train.select_dtypes(include=["int64","float64"]).columns.tolist()
categorical_cols = X_train.select_dtypes(include=["int32","int64"]).columns.tolist()

def objective(trial):
    n_estimators = trial.suggest_int("n_estimators", 100, 500)
    max_depth = trial.suggest_int("max_depth", 3, 10)
    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3)
    
    pipeline = build_pipeline(numeric_cols, categorical_cols)
    pipeline.set_params(classifier__n_estimators=n_estimators,
                        classifier__max_depth=max_depth,
                        classifier__learning_rate=learning_rate)
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_valid)
    return accuracy_score(y_valid, preds)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)

print("Best trial:", study.best_trial.params)

best_pipeline = build_pipeline(numeric_cols, categorical_cols)
best_pipeline.set_params(
    classifier__n_estimators=study.best_trial.params["n_estimators"],
    classifier__max_depth=study.best_trial.params["max_depth"],
    classifier__learning_rate=study.best_trial.params["learning_rate"]
)
best_pipeline.fit(X_train, y_train)

preds = best_pipeline.predict(X_valid)
print("Final Accuracy:", accuracy_score(y_valid, preds))
print(classification_report(y_valid, preds))

save_model(best_pipeline, name="advanced_churn_pipeline.joblib")
