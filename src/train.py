import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from src.data_prep import load_raw, basic_clean
from src.feature_engineering import create_features, encode_features
from src.model_utils import build_pipeline, save_model

# Step 1: Load and clean data
df = load_raw()
df = basic_clean(df)
df = create_features(df)

# Step 2: Keep only important features
important_features = [
    'tenure', 'MonthlyCharges', 'TotalCharges',
    'Contract', 'PaymentMethod', 'InternetService',
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'gender', 'SeniorCitizen', 'Churn'
]
df = df[important_features]

# Step 3: Encode categorical features
df, encoders = encode_features(df)

X = df.drop(columns=["Churn"])
y = df["Churn"]

# Step 4: Train/Validation split
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Step 5: Identify numeric & categorical features
numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
categorical_cols = ['Contract', 'PaymentMethod', 'InternetService',
                    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                    'gender', 'SeniorCitizen']

# Step 6: Optuna hyperparameter optimization
def objective(trial):
    n_estimators = trial.suggest_int("n_estimators", 100, 500)
    max_depth = trial.suggest_int("max_depth", 3, 10)
    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3)
    
    pipeline = build_pipeline(numeric_cols, categorical_cols)
    pipeline.set_params(
        classifier__n_estimators=n_estimators,
        classifier__max_depth=max_depth,
        classifier__learning_rate=learning_rate
    )
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_valid)
    return accuracy_score(y_valid, preds)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)

print("Best trial:", study.best_trial.params)

# Step 7: Train final pipeline with best parameters
best_pipeline = build_pipeline(numeric_cols, categorical_cols)
best_pipeline.set_params(
    classifier__n_estimators=study.best_trial.params["n_estimators"],
    classifier__max_depth=study.best_trial.params["max_depth"],
    classifier__learning_rate=study.best_trial.params["learning_rate"]
)
best_pipeline.fit(X_train, y_train)

# Step 8: Evaluate
preds = best_pipeline.predict(X_valid)
print("Final Accuracy:", accuracy_score(y_valid, preds))
print(classification_report(y_valid, preds))
print("Final feature columns:", X_train.columns.tolist())

# Step 9: Save model
save_model(best_pipeline, name="advanced_churn_pipeline.joblib")
