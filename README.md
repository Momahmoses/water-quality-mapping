
# Water Quality & Safe Water Access Mapping
WHO-compliance classification of rural water sources (boreholes, wells, river abstractions, springs) across 12 Northern Nigerian states. Identifies contaminated sources using physicochemical measurements and recommends intervention priorities.

---

## Features

- Gradient Boosting safety classifier on 14 water quality parameters
- WHO threshold violation counter (9 parameters)
- Scatter mapbox visualization colored by safety, turbidity, or coliform
- Individual source inspector with real-time safety prediction
- Population-weighted risk exposure estimates

## Project Structure

```
water-quality-mapping/
├── src/
│   ├── data_generator.py   # Synthetic borehole/well dataset (4,000 sources)
│   └── model.py            # GBT classifier + WHO threshold logic
├── streamlit_app.py        # 4-page Streamlit dashboard
├── requirements.txt
└── README.md
```

## Quick Start

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Dataset

Synthetic dataset of 4,000 water sources across Katsina, Jigawa, Sokoto, Yobe, Borno, Bauchi, Gombe, Adamawa, Niger, Zamfara, Kebbi, and Kwara states. Features include depth, pH, turbidity, nitrate, TDS, coliform count, fluoride, iron, arsenic, lead, and treatment status.

## Tech Stack

| Layer | Library |
|---|---|
| Dashboard | Streamlit |
| ML Model | scikit-learn GradientBoostingClassifier |
| Visualisation | Plotly Express + Mapbox |
| Data | NumPy / Pandas synthetic generator |

---

*Dataset is synthetic and generated for demonstration purposes.*
