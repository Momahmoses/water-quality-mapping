import folium
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from folium.plugins import HeatMap, MarkerCluster


QUALITY_COLORS = {"safe": "#1a9850", "moderate_risk": "#fee08b",
                  "high_risk": "#fc8d59", "unsafe": "#d73027"}
SOURCE_ICONS = {
    "borehole": "tint", "hand_pump": "tint", "river": "water",
    "well": "circle", "rainwater_harvest": "cloud", "piped": "home"
}


def plot_water_map(gdf, output_path):
    center = [gdf["latitude"].mean(), gdf["longitude"].mean()]
    m = folium.Map(location=center, zoom_start=6, tiles="CartoDB positron")

    heat_data = [
        [row["latitude"], row["longitude"], row["contamination_probability"]]
        for _, row in gdf[gdf["predicted_contaminated"] == 1].iterrows()
    ]
    if heat_data:
        HeatMap(heat_data, radius=14, blur=10, name="Contamination Risk").add_to(m)

    cluster = MarkerCluster(name="Water Sources").add_to(m)
    for _, row in gdf.iterrows():
        color = QUALITY_COLORS.get(str(row.get("quality_class", "safe")), "#ccc")
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=max(3, min(8, row["population_served"] / 1000)),
            color=color, fill=True, fill_opacity=0.7,
            popup=folium.Popup(
                f"<b>{row['source_type'].replace('_', ' ').title()}</b><br>"
                f"Quality: {row.get('quality_class', 'N/A')}<br>"
                f"pH: {row['ph']:.2f}<br>"
                f"Turbidity: {row['turbidity_ntu']:.2f} NTU<br>"
                f"Coliform: {row['coliform_cfu100ml']:.0f} CFU/100ml<br>"
                f"Population Served: {row['population_served']:,}<br>"
                f"State: {row['state']}",
                max_width=250
            )
        ).add_to(cluster)

    folium.LayerControl().add_to(m)
    m.save(output_path)
    print(f"Water quality map saved: {output_path}")


def plot_access_chart(df, output_path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    quality_counts = df["quality_class"].value_counts()
    colors = [QUALITY_COLORS.get(str(k), "#ccc") for k in quality_counts.index]
    axes[0, 0].pie(quality_counts.values, labels=quality_counts.index, colors=colors, autopct="%1.1f%%")
    axes[0, 0].set_title("Water Quality Classification")

    state_contamination = df.groupby("state")["predicted_contaminated"].mean().sort_values(ascending=False)
    state_contamination.plot(kind="barh", color="#d73027", ax=axes[0, 1])
    axes[0, 1].set_title("Contamination Rate by State")
    axes[0, 1].set_xlabel("Contamination Rate")

    source_quality = df.groupby("source_type")["contamination_probability"].mean().sort_values()
    source_quality.plot(kind="barh", color="#2166ac", ax=axes[1, 0])
    axes[1, 0].set_title("Avg Contamination Risk by Source Type")

    sns.histplot(df["ph"], bins=40, kde=True, color="#1a9850", ax=axes[1, 1])
    axes[1, 1].axvline(6.5, color="red", linestyle="--", label="WHO Min (6.5)")
    axes[1, 1].axvline(8.5, color="red", linestyle="--", label="WHO Max (8.5)")
    axes[1, 1].set_title("pH Distribution")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
