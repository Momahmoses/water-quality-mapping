import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import yaml


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


SOURCE_TYPES = ["borehole", "hand_pump", "river", "well", "rainwater_harvest", "piped"]
SOURCE_WEIGHTS = [0.30, 0.25, 0.15, 0.15, 0.08, 0.07]


def generate_water_sources(n=2000, seed=42):
    np.random.seed(seed)
    data = {
        "source_id": [f"WS_{i:05d}" for i in range(n)],
        "source_type": np.random.choice(SOURCE_TYPES, n, p=SOURCE_WEIGHTS),
        "latitude": np.random.uniform(4.0, 14.5, n),
        "longitude": np.random.uniform(3.0, 15.0, n),
        "state": np.random.choice(
            ["Kano", "Borno", "Sokoto", "Zamfara", "Kebbi", "Adamawa",
             "Bauchi", "Gombe", "Yobe", "Niger", "Kwara", "Jigawa"], n
        ),
        "depth_m": np.random.exponential(scale=30, size=n),
        "ph": np.random.normal(7.0, 0.8, n).clip(4.5, 9.5),
        "turbidity_ntu": np.random.exponential(scale=5, size=n),
        "nitrate_mgl": np.random.exponential(scale=15, size=n),
        "coliform_cfu100ml": np.random.poisson(lam=5, size=n),
        "arsenic_ugl": np.random.exponential(scale=5, size=n),
        "fluoride_mgl": np.random.exponential(scale=0.8, size=n),
        "conductivity_us": np.random.gamma(shape=3, scale=200, size=n),
        "iron_mgl": np.random.exponential(scale=0.5, size=n),
        "distance_to_latrine_m": np.random.exponential(scale=30, size=n),
        "population_served": np.random.lognormal(mean=5.5, sigma=1.2, size=n).astype(int),
        "functional": np.random.choice([0, 1], n, p=[0.2, 0.8]),
        "last_tested_days": np.random.exponential(scale=200, size=n),
        "year_installed": np.random.choice(range(1990, 2024), n),
    }
    df = pd.DataFrame(data)
    df["source_type_code"] = pd.Categorical(df["source_type"]).codes
    return df


def classify_quality(df, config):
    g = config["who_guidelines"]
    df = df.copy()
    violations = (
        (df["ph"] < g["ph_min"]) | (df["ph"] > g["ph_max"]) |
        (df["turbidity_ntu"] > g["turbidity_max_ntu"]) |
        (df["nitrate_mgl"] > g["nitrate_max_mgl"]) |
        (df["coliform_cfu100ml"] > g["coliform_max_cfu100ml"]) |
        (df["arsenic_ugl"] > g["arsenic_max_ugl"]) |
        (df["fluoride_mgl"] > g["fluoride_max_mgl"]) |
        (df["conductivity_us"] > g["conductivity_max_us"])
    )
    violation_count = sum([
        (df["ph"] < g["ph_min"]) | (df["ph"] > g["ph_max"]),
        df["turbidity_ntu"] > g["turbidity_max_ntu"],
        df["nitrate_mgl"] > g["nitrate_max_mgl"],
        df["coliform_cfu100ml"] > g["coliform_max_cfu100ml"],
        df["arsenic_ugl"] > g["arsenic_max_ugl"],
        df["fluoride_mgl"] > g["fluoride_max_mgl"],
    ])
    df["violation_count"] = violation_count
    df["quality_class"] = pd.cut(
        df["violation_count"],
        bins=[-1, 0, 1, 2, np.inf],
        labels=["safe", "moderate_risk", "high_risk", "unsafe"]
    )
    df["is_contaminated"] = violations
    return df


def to_geodataframe(df):
    return gpd.GeoDataFrame(
        df,
        geometry=[Point(lon, lat) for lon, lat in zip(df["longitude"], df["latitude"])],
        crs="EPSG:4326"
    )
