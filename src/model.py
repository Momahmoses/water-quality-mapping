import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib


FEATURES = [
    "source_type_code", "depth_m", "ph", "turbidity_ntu", "nitrate_mgl",
    "coliform_cfu100ml", "arsenic_ugl", "fluoride_mgl", "conductivity_us",
    "iron_mgl", "distance_to_latrine_m", "last_tested_days"
]
TARGET = "is_contaminated"


def train(df, config):
    valid = df[df["functional"] == 1].copy()
    X = valid[FEATURES]
    y = valid[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["model"]["test_size"],
        random_state=config["model"]["random_state"],
        stratify=y
    )
    model = RandomForestClassifier(
        n_estimators=config["model"]["n_estimators"],
        random_state=config["model"]["random_state"],
        class_weight="balanced", n_jobs=-1
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred),
    }
    return model, metrics


def predict_contamination(model, df):
    df = df.copy()
    probs = model.predict_proba(df[FEATURES])[:, 1]
    df["contamination_probability"] = probs
    df["predicted_contaminated"] = model.predict(df[FEATURES])
    return df


def feature_importance(model):
    return pd.DataFrame({
        "feature": FEATURES,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)


def prioritize_interventions(df):
    contaminated = df[df["predicted_contaminated"] == 1].copy()
    contaminated["intervention_score"] = (
        contaminated["contamination_probability"]
        * np.log1p(contaminated["population_served"])
    )
    return contaminated.sort_values("intervention_score", ascending=False)


def save_model(model, path):
    joblib.dump(model, path)
