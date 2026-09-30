# CPCB Next-Day AQI Classifier

Predicts tomorrow's Air Quality Index category for Kanpur from today's pollutant readings, using a Random Forest trained on real historical data from India's Central Pollution Control Board (CPCB).

![Feature importance](output/feature_importance.png)

## Why this isn't a toy model

- **Chronological train/test split** — the model is trained on earlier dates and tested on later ones, never on a random shuffle. Time-series data leaks information if you split it randomly; this avoids that.
- **Persistence baseline included** — every run also reports the accuracy of the naive "tomorrow will look like today" guess, so the model's accuracy number means something instead of being reported in isolation.
- **Cyclical season encoding** — day-of-year is encoded as `sin`/`cos` pairs rather than a raw 1–365 number, so December 31st and January 1st are correctly treated as adjacent instead of maximally different.
- **Class-balanced training** — AQI categories are naturally imbalanced (Kanpur skews toward "Poor"/"Very Poor" more than "Good"), so the classifier is weighted to avoid just always predicting the majority class.

## Data

This project trains on **real CPCB data only** — it does not ship a bundled dataset and will not present generated rows as real measurements. See [`DATA.md`](DATA.md) for exactly what to download and which column names are supported.

**Want to see the pipeline run before downloading real data?** Use the included generator to create a clearly-labeled synthetic CSV:

```bash
python3 generate_sample_data.py
python3 classifier.py sample_kanpur_synthetic.csv
```

This synthetic data is for demoing the pipeline only — it uses a made-up AQI formula and random noise, not real measurements or CPCB's actual calculation. Every run of `generate_sample_data.py` says so explicitly. Don't draw real conclusions from it; use it only to confirm the code runs end-to-end.


**Have a Vonter Parquet file?** Place it in the repo root and run:

    python prepare_real_data.py

This filters for Kanpur, aggregates 15-minute readings into daily averages, computes AQI from PM2.5 using the CPCB breakpoint table, and writes `kanpur_real.csv` — which you can then pass directly to `classifier.py`.

## Setup and run

```bash
python3 -m pip install -r requirements.txt

# with real CPCB data downloaded per DATA.md:
python3 classifier.py kanpur_real.csv

# or with generated demo data:
python3 generate_sample_data.py
python3 classifier.py sample_kanpur_synthetic.csv
```

**Windows (PowerShell):**

```powershell
cd C:\Users\91902\Documents\GitHub\Weather-predictor
python classifier.py kanpur_real.csv
```

Both produce identical output — the only difference is `python` vs `python3` and path separators.

## Output

Each run prints:
- Row counts, feature list, model accuracy, and the persistence baseline accuracy
- A full classification report and confusion matrix
- A prediction for the next day, based on the most recent row in your data

...and saves two plots to `output/`:
- `feature_importance.png` — which inputs the model actually relies on
- `confusion_matrix.png` — where the model's predictions go right and wrong, category by category

## Real-world testing

The pipeline has been validated end-to-end against a real Parquet export of CPCB station data (105,120 Kanpur observations at 15-minute resolution, aggregated to 365 daily rows). The classifier trained on this real data achieved **67.1% test accuracy vs. 61.6% for the persistence baseline**.

The column-matching logic (`DATA.md`) is written from CPCB's documented export format. If you hit a column-matching error on a different CPCB export, check the exact header names against `DATA.md` or adjust `COLUMN_ALIASES` in `classifier.py` — the matching is intentionally centralized in one place to make this easy.


## Limitations

- Predictions are next-day only, for the city configured (`--city`, default Kanpur).
- Accuracy depends heavily on how much historical data you provide — CPCB recommends at least several months for meaningful results.
- This is an educational project, not an official forecast — always treat the "This is an educational estimate, not an official CPCB forecast" line in the output as literal.

## How AQI is computed

The model's target variable is an AQI category. Real CPCB AQI is the **maximum sub-index across six pollutants** (PM2.5, PM10, NO2, SO2, CO, O3), using a piecewise linear breakpoint table per pollutant.

For this project, AQI is driven by **PM2.5 alone**. This is a deliberate simplification: on high-pollution days in Kanpur, PM2.5 almost always drives the maximum sub-index, so the approximation is close on the days that matter most. The breakpoints match CPCB's published PM2.5 sub-index table:

| PM2.5 (µg/m³) | AQI category   |
|---------------|----------------|
| 0–30          | Good           |
| 31–60         | Satisfactory   |
| 61–90         | Moderate       |
| 91–120        | Poor           |
| 121–250       | Very Poor      |
| 251+          | Severe         |

Days when PM10 or NO2 spikes without a corresponding PM2.5 spike will be miscategorised by this simplification. A full multi-pollutant implementation would take the maximum sub-index across all six pollutants.
