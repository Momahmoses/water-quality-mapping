import os
from src.data_loader import load_config, generate_water_sources, classify_quality, to_geodataframe
from src.model import train, predict_contamination, prioritize_interventions, save_model
from src.visualize import plot_water_map, plot_access_chart


def main():
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    config = load_config("config.yaml")
    print(f"[1/5] Config loaded — WASH Program: {', '.join(config['targets'])}")

    df = generate_water_sources(n=2000)
    df = classify_quality(df, config)
    contaminated_count = df["is_contaminated"].sum()
    print(f"[2/5] {len(df)} water sources analyzed — {contaminated_count} ({contaminated_count/len(df):.1%}) contaminated")

    model, metrics = train(df, config)
    print(f"[3/5] Model trained — Accuracy: {metrics['accuracy']:.4f}")
    print(metrics["classification_report"])
    save_model(model, "models/water_quality_model.pkl")

    result_df = predict_contamination(model, df)
    result_df.to_csv(config["output"]["quality_report"], index=False)

    interventions = prioritize_interventions(result_df)
    interventions.to_csv(config["output"]["contamination_report"], index=False)
    print(f"[4/5] {len(interventions)} sources prioritized for intervention")
    print(f"      Top 5 by population impact:")
    print(interventions[["source_id", "state", "source_type", "population_served", "quality_class"]].head(5).to_string(index=False))

    gdf = to_geodataframe(result_df)
    plot_water_map(gdf, config["output"]["water_map"])
    plot_access_chart(result_df, config["output"]["access_chart"])
    print("[5/5] All outputs saved to /outputs/")
    print("\nDone.")


if __name__ == "__main__":
    main()
