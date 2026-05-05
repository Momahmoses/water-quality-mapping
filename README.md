# Water Quality & Borehole Access Mapping

Maps water sources (boreholes, hand pumps, rivers, wells) + population data + WHO quality test results to identify contaminated or waterless communities across rural Nigeria and Northern Africa — supporting WASH programs.

## Features
- WHO guideline compliance checking (pH, turbidity, nitrate, coliform, arsenic, fluoride)
- Random Forest contamination prediction model
- Population-weighted intervention priority scoring
- Interactive contamination heatmap with per-source details
- State-level and source-type contamination analysis

## Project Structure
```
water-quality-mapping/
├── src/
│   ├── data_loader.py    # Water source data, WHO quality classification
│   ├── model.py          # RF contamination predictor, intervention prioritization
│   └── visualize.py      # Water map, access charts
├── data/raw/             # Borehole/spring test data, population rasters
├── models/               # Saved classifier
├── outputs/              # Maps, reports
├── config.yaml
├── main.py
└── requirements.txt
```

## Setup
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

## Data Sources
| Layer | Source |
|-------|--------|
| Water source locations | RWSS / State Water Agencies |
| Quality test results | NAFDAC / WHO WASH data |
| Population | WorldPop Nigeria 100m |
| Community boundaries | GADM Nigeria Level-3 |

## Output
- `outputs/water_quality_map.html` — interactive contamination map
- `outputs/water_quality_report.csv` — per-source quality scores
- `outputs/contamination_zones.csv` — ranked intervention priorities
- `outputs/water_access_chart.png` — quality and access breakdown

## Author
MOMAH MOSES .C.
