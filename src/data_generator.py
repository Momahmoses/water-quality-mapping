"""
Synthetic dataset generator for water quality and borehole access mapping.
Each record is a water source (borehole, well, river point, spring) with quality measurements.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

RANDOM_SEED = 42
N_SOURCES = 4000

STATES = [
    ("Katsina", 12.98, 7.60), ("Jigawa", 12.23, 9.56), ("Sokoto", 13.06, 5.24),
    ("Yobe", 12.29, 11.44), ("Borno", 11.83, 13.15), ("Bauchi", 10.31, 9.84),
    ("Gombe", 10.29, 11.17), ("Adamawa", 9.33, 12.40), ("Niger", 9.93, 6.56),
    ("Zamfara", 12.17, 6.66), ("Kebbi", 12.45, 4.20), ("Kwara", 8.50, 4.55),
]

SOURCE_TYPES = ["Borehole", "Hand-Dug Well", "River Abstraction", "Spring", "Rainwater Harvest", "Piped Water"]
TREATMENT_STATUS = ["None", "Chlorinated", "UV Treated", "Filtered", "Boiled"]


def generate_water_quality_dataset(n_sources: int = N_SOURCES, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    records = []

    for src_id in range(n_sources):
        state, lat, lon = STATES[rng.integers(0, len(STATES))]
        source_type = SOURCE_TYPES[rng.integers(0, len(SOURCE_TYPES))]
        treatment = TREATMENT_STATUS[rng.integers(0, len(TREATMENT_STATUS))]
        is_rural = int(rng.random() < 0.70)
        dist_to_latrine_m = max(1.0, float(rng.exponential(25)))
        depth_m = max(1.0, float(rng.normal(30, 15))) if source_type == "Borehole" else float(rng.uniform(2, 12))

        ph = float(np.clip(rng.normal(7.0, 0.8), 5.5, 9.5))
        turbidity_ntu = max(0.0, float(rng.exponential(5)))
        nitrate_mg_l = max(0.0, float(rng.exponential(15)))
        total_dissolved_solids_mg_l = max(10.0, float(rng.lognormal(5.5, 0.8)))
        coliform_per_100ml = max(0, int(rng.poisson(20 if dist_to_latrine_m < 10 else 5)))
        fluoride_mg_l = max(0.0, float(rng.exponential(0.8)))
        iron_mg_l = max(0.0, float(rng.exponential(0.4)))
        arsenic_ug_l = max(0.0, float(rng.exponential(5)))
        lead_ug_l = max(0.0, float(rng.exponential(3)))

        # WHO threshold violations
        who_violations = sum([
            ph < 6.5 or ph > 8.5,
            turbidity_ntu > 4,
            nitrate_mg_l > 50,
            total_dissolved_solids_mg_l > 1000,
            coliform_per_100ml > 0,
            fluoride_mg_l > 1.5,
            iron_mg_l > 0.3,
            arsenic_ug_l > 10,
            lead_ug_l > 10,
        ])

        quality_index = max(0.0, 1.0 - who_violations / 9)
        is_safe = int(who_violations <= 1 and coliform_per_100ml == 0)
        pop_served = max(1, int(rng.lognormal(4.5, 1.2)))

        records.append({
            "source_id": src_id, "state": state, "source_type": source_type,
            "latitude": round(lat + rng.normal(0, 0.4), 6),
            "longitude": round(lon + rng.normal(0, 0.4), 6),
            "is_rural": is_rural, "depth_m": round(depth_m, 1),
            "treatment_status": treatment,
            "dist_to_latrine_m": round(dist_to_latrine_m, 1),
            "ph": round(ph, 2),
            "turbidity_ntu": round(turbidity_ntu, 2),
            "nitrate_mg_l": round(nitrate_mg_l, 2),
            "total_dissolved_solids_mg_l": round(total_dissolved_solids_mg_l, 1),
            "coliform_per_100ml": coliform_per_100ml,
            "fluoride_mg_l": round(fluoride_mg_l, 3),
            "iron_mg_l": round(iron_mg_l, 3),
            "arsenic_ug_l": round(arsenic_ug_l, 2),
            "lead_ug_l": round(lead_ug_l, 2),
            "who_violations": who_violations,
            "quality_index": round(quality_index, 4),
            "population_served": pop_served,
            "is_safe": is_safe,
        })

    return pd.DataFrame(records)


def save_dataset(output_dir: str | Path = "data/raw") -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = generate_water_quality_dataset()
    path = output_dir / "water_quality_data.csv"
    df.to_csv(path, index=False)
    return path
