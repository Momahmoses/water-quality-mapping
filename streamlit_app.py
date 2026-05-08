"""
Water Quality & Safe Water Access Mapping
==========================================
WHO-compliance classification of rural water sources across Northern Nigeria.
Identifies contaminated boreholes, wells, and river abstractions using
physicochemical parameters and ML-based safety scoring.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))
from data_generator import generate_water_quality_dataset
from model import (
    META_PATH, MODEL_PATH, load_model, predict_source, save_model, train,
)

st.set_page_config(
    page_title="Water Quality Mapping | Nigeria",
    page_icon="💧",
    layout="wide",
)

# ── data & model ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data() -> pd.DataFrame:
    return generate_water_quality_dataset()


@st.cache_resource
def get_model():
    if MODEL_PATH.exists() and META_PATH.exists():
        return load_model()
    df = load_data()
    clf, meta = train(df)
    save_model(clf, meta)
    return clf, meta


df = load_data()
clf, meta = get_model()

# ── sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("💧 Water Quality Mapping")
page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Quality Map", "Source Inspector", "Model Performance"],
)
state_filter = st.sidebar.multiselect("Filter by State", sorted(df["state"].unique()), default=[])
type_filter = st.sidebar.multiselect("Source Type", sorted(df["source_type"].unique()), default=[])

filtered = df.copy()
if state_filter:
    filtered = filtered[filtered["state"].isin(state_filter)]
if type_filter:
    filtered = filtered[filtered["source_type"].isin(type_filter)]

# ── overview ──────────────────────────────────────────────────────────────────
if page == "Overview":
    st.title("Water Quality & Safe Water Access — Northern Nigeria")
    st.markdown(
        "WHO-compliance analysis of **{:,} water sources** across 12 states. "
        "Identifies contaminated sources and underserved communities.".format(len(df))
    )

    col1, col2, col3, col4 = st.columns(4)
    safe_pct = filtered["is_safe"].mean() * 100
    col1.metric("Safe Sources", f"{safe_pct:.1f}%", f"{int(filtered['is_safe'].sum()):,} of {len(filtered):,}")
    col2.metric("Avg WHO Violations", f"{filtered['who_violations'].mean():.2f}", "per source")
    col3.metric("Coliform-Positive", f"{(filtered['coliform_per_100ml'] > 0).mean()*100:.1f}%")
    col4.metric("Population at Risk", f"{filtered[filtered['is_safe']==0]['population_served'].sum():,.0f}")

    col_a, col_b = st.columns(2)
    with col_a:
        state_safe = filtered.groupby("state")["is_safe"].mean().reset_index()
        state_safe.columns = ["state", "safe_rate"]
        fig = px.bar(
            state_safe.sort_values("safe_rate"),
            x="safe_rate", y="state", orientation="h",
            color="safe_rate", color_continuous_scale="RdYlGn",
            labels={"safe_rate": "Safe Rate", "state": "State"},
            title="Safe Water Rate by State",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        type_counts = filtered.groupby(["source_type", "is_safe"]).size().reset_index(name="count")
        type_counts["safety"] = type_counts["is_safe"].map({1: "Safe", 0: "Unsafe"})
        fig2 = px.bar(
            type_counts, x="source_type", y="count", color="safety",
            color_discrete_map={"Safe": "#2ecc71", "Unsafe": "#e74c3c"},
            barmode="stack", title="Safety by Source Type",
            labels={"source_type": "Source Type", "count": "Count"},
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("WHO Violation Distribution")
    fig3 = px.histogram(
        filtered, x="who_violations", color="source_type",
        barmode="overlay", nbins=10,
        title="Number of WHO Parameter Violations per Source",
        labels={"who_violations": "Violations", "count": "Sources"},
    )
    st.plotly_chart(fig3, use_container_width=True)

# ── quality map ───────────────────────────────────────────────────────────────
elif page == "Quality Map":
    st.title("Water Quality Geographic Map")
    color_by = st.selectbox(
        "Color by",
        ["is_safe", "who_violations", "quality_index", "turbidity_ntu", "coliform_per_100ml"],
    )
    sample = filtered.sample(min(1500, len(filtered)), random_state=42)
    fig = px.scatter_mapbox(
        sample, lat="latitude", lon="longitude",
        color=color_by,
        color_continuous_scale="RdYlGn" if color_by in ("is_safe", "quality_index") else "OrRd",
        size="population_served",
        size_max=18,
        hover_data=["state", "source_type", "treatment_status", "who_violations"],
        zoom=5, height=580,
        mapbox_style="carto-positron",
        title=f"Water Sources colored by: {color_by}",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Correlation: Distance to Latrine vs Coliform Count")
    fig2 = px.scatter(
        filtered.sample(800, random_state=1),
        x="dist_to_latrine_m", y="coliform_per_100ml",
        color="is_safe", color_discrete_map={1: "#2ecc71", 0: "#e74c3c"},
        opacity=0.7, trendline="ols",
        labels={"dist_to_latrine_m": "Distance to Latrine (m)", "coliform_per_100ml": "Coliform /100ml"},
    )
    st.plotly_chart(fig2, use_container_width=True)

# ── source inspector ──────────────────────────────────────────────────────────
elif page == "Source Inspector":
    st.title("Individual Source Safety Predictor")
    st.markdown("Enter measurements for a specific water source to assess WHO compliance.")

    c1, c2, c3 = st.columns(3)
    with c1:
        src_type = st.selectbox("Source Type", meta["source_type_classes"])
        treatment = st.selectbox("Treatment Status", meta["treatment_classes"])
        is_rural = st.radio("Location", ["Rural", "Urban"], index=0)
    with c2:
        depth_m = st.slider("Depth (m)", 1.0, 100.0, 30.0)
        dist_latrine = st.slider("Distance to Latrine (m)", 1.0, 100.0, 25.0)
        ph = st.slider("pH", 5.0, 10.0, 7.0, step=0.1)
        turbidity = st.slider("Turbidity (NTU)", 0.0, 50.0, 3.0, step=0.5)
    with c3:
        nitrate = st.slider("Nitrate (mg/L)", 0.0, 100.0, 10.0)
        tds = st.slider("TDS (mg/L)", 10.0, 2000.0, 300.0, step=10.0)
        coliform = st.slider("Coliform (/100ml)", 0, 200, 0)
        fluoride = st.slider("Fluoride (mg/L)", 0.0, 5.0, 0.5, step=0.1)

    c4, c5 = st.columns(2)
    with c4:
        iron = st.slider("Iron (mg/L)", 0.0, 3.0, 0.2, step=0.05)
        arsenic = st.slider("Arsenic (µg/L)", 0.0, 50.0, 3.0, step=0.5)
    with c5:
        lead = st.slider("Lead (µg/L)", 0.0, 50.0, 2.0, step=0.5)

    if st.button("Assess Source", type="primary"):
        result = predict_source(
            clf, meta, src_type, treatment, depth_m, dist_latrine,
            1 if is_rural == "Rural" else 0,
            ph, turbidity, nitrate, tds, coliform, fluoride, iron, arsenic, lead,
        )
        colour = "green" if result["label"] == "Safe" else "red"
        st.markdown(
            f"### :{colour}[{result['label']}] — {result['probability_safe']*100:.1f}% probability safe"
        )
        violations = sum([
            ph < 6.5 or ph > 8.5, turbidity > 4, nitrate > 50,
            tds > 1000, coliform > 0, fluoride > 1.5, iron > 0.3,
            arsenic > 10, lead > 10,
        ])
        st.info(f"WHO parameter violations detected: **{violations}/9**")

# ── model performance ─────────────────────────────────────────────────────────
elif page == "Model Performance":
    st.title("Model Performance — Water Safety Classifier")
    col1, col2 = st.columns(2)
    col1.metric("CV ROC-AUC", f"{meta['cv_roc_auc_mean']:.4f}", f"±{meta['cv_roc_auc_std']:.4f}")
    col2.metric("Training Accuracy", f"{meta['train_accuracy']:.4f}")

    fi = pd.Series(meta["feature_importances"]).sort_values(ascending=True)
    fig = px.bar(
        fi.reset_index(), x=0, y="index", orientation="h",
        title="Feature Importances",
        labels={0: "Importance", "index": "Feature"},
        color=0, color_continuous_scale="Blues",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Parameter Thresholds (WHO Guidelines)")
    thresholds = {
        "pH": "6.5–8.5", "Turbidity": "< 4 NTU", "Nitrate": "< 50 mg/L",
        "TDS": "< 1000 mg/L", "Coliform": "0 /100ml", "Fluoride": "< 1.5 mg/L",
        "Iron": "< 0.3 mg/L", "Arsenic": "< 10 µg/L", "Lead": "< 10 µg/L",
    }
    st.table(pd.DataFrame(list(thresholds.items()), columns=["Parameter", "WHO Limit"]))
