"""
Water quality safety classification model.
Predicts WHO-compliance (safe/unsafe) from physicochemical measurements.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

FEATURES = [
    "depth_m", "dist_to_latrine_m", "ph", "turbidity_ntu",
    "nitrate_mg_l", "total_dissolved_solids_mg_l", "coliform_per_100ml",
    "fluoride_mg_l", "iron_mg_l", "arsenic_ug_l", "lead_ug_l",
    "is_rural", "source_type_enc", "treatment_enc",
]
TARGET = "is_safe"
MODEL_PATH = Path("assets/water_quality_model.pkl")
META_PATH = Path("assets/water_quality_meta.json")


def _encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    le_src = LabelEncoder()
    le_trt = LabelEncoder()
    df["source_type_enc"] = le_src.fit_transform(df["source_type"])
    df["treatment_enc"] = le_trt.fit_transform(df["treatment_status"])
    return df, le_src, le_trt


def train(df: pd.DataFrame) -> tuple[GradientBoostingClassifier, dict]:
    df, le_src, le_trt = _encode_categoricals(df)
    X = df[FEATURES].values
    y = df[TARGET].values

    clf = GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.08, random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    cv_aucs = []
    for train_idx, val_idx in cv.split(X, y):
        clf.fit(X[train_idx], y[train_idx])
        prob = clf.predict_proba(X[val_idx])[:, 1]
        cv_aucs.append(roc_auc_score(y[val_idx], prob))

    clf.fit(X, y)
    preds = clf.predict(X)
    report = classification_report(y, preds, output_dict=True)

    meta = {
        "cv_roc_auc_mean": float(np.mean(cv_aucs)),
        "cv_roc_auc_std": float(np.std(cv_aucs)),
        "train_accuracy": report["accuracy"],
        "feature_importances": dict(zip(FEATURES, clf.feature_importances_.tolist())),
        "source_type_classes": le_src.classes_.tolist(),
        "treatment_classes": le_trt.classes_.tolist(),
    }
    return clf, meta


def save_model(clf: GradientBoostingClassifier, meta: dict) -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    META_PATH.write_text(json.dumps(meta, indent=2))


def load_model() -> tuple[GradientBoostingClassifier, dict]:
    clf = joblib.load(MODEL_PATH)
    meta = json.loads(META_PATH.read_text())
    return clf, meta


def predict_source(
    clf: GradientBoostingClassifier,
    meta: dict,
    source_type: str,
    treatment: str,
    depth_m: float,
    dist_to_latrine_m: float,
    is_rural: int,
    ph: float,
    turbidity_ntu: float,
    nitrate_mg_l: float,
    tds_mg_l: float,
    coliform: int,
    fluoride: float,
    iron: float,
    arsenic: float,
    lead: float,
) -> dict:
    src_enc = meta["source_type_classes"].index(source_type) if source_type in meta["source_type_classes"] else 0
    trt_enc = meta["treatment_classes"].index(treatment) if treatment in meta["treatment_classes"] else 0
    row = np.array([[
        depth_m, dist_to_latrine_m, ph, turbidity_ntu,
        nitrate_mg_l, tds_mg_l, coliform, fluoride, iron, arsenic, lead,
        is_rural, src_enc, trt_enc,
    ]])
    prob = clf.predict_proba(row)[0][1]
    label = "Safe" if prob >= 0.5 else "Unsafe"
    return {"probability_safe": round(float(prob), 4), "label": label}
