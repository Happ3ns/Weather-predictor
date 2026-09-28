

@'
import pyarrow.dataset as ds
import pyarrow.compute as pc
import pandas as pd

input_file = "pcb-air-quality-2025.parquet"
print(f"Loading {input_file} (filtered)...")

# Peek at schema without loading data
dataset = ds.dataset(input_file, format="parquet")
print("Columns:", dataset.schema.names)

# Find the city column (case-insensitive)
city_col = next((n for n in dataset.schema.names if n.lower() == "city"), None)
if city_col is None:
    raise ValueError(f"No 'City' column. Columns: {dataset.schema.names}")

# Filter to Kanpur rows only, at the Arrow level (memory efficient)
print("Filtering for Kanpur...")
filter_expr = pc.match_substring(pc.field(city_col), "Kanpur", ignore_case=True)
table = dataset.to_table(filter=filter_expr)

print(f"Kanpur rows: {table.num_rows}")
df = table.to_pandas()
del table  # free Arrow memory

if df.empty:
    raise ValueError("No Kanpur data found.")

# Find timestamp column (case-insensitive)
ts_col = next((c for c in df.columns if c.lower() == "timestamp" or c.lower() == "date"), None)
if ts_col is None:
    raise ValueError(f"No timestamp column. Columns: {list(df.columns)}")

df[ts_col] = pd.to_datetime(df[ts_col])
df = df.set_index(ts_col)

print("Aggregating to daily averages...")
daily = df.resample("D").mean(numeric_only=True)

# Rename to classifier.py's expected names
rename_map = {
    "PM2.5 (µg/m³)": "PM2.5",
    "PM10 (µg/m³)": "PM10",
    "NO2 (µg/m³)": "NO2",
    "SO2 (µg/m³)": "SO2",
    "CO (mg/m³)": "CO",
    "Ozone (µg/m³)": "O3",
    "AT (°C)": "Temperature",
    "RH (%)": "Humidity",
    "WS (m/s)": "Wind Speed",
}
daily = daily.rename(columns=rename_map)

daily = daily.reset_index()
daily["From Date"] = daily[ts_col].dt.strftime("%d-%m-%Y")
daily["City"] = "Kanpur"

if "AQI" not in daily.columns:
    if "PM2.5" in daily.columns:
        daily["AQI"] = (daily["PM2.5"] * 1.5).round(0)
    else:
        raise ValueError("No AQI and no PM2.5 to build proxy from.")

final_cols = ["From Date", "City", "PM2.5", "PM10", "NO2", "SO2", "CO", "O3",
              "Temperature", "Humidity", "Wind Speed", "AQI"]
final_cols = [c for c in final_cols if c in daily.columns]
output = daily[final_cols].dropna(subset=["From Date", "AQI"])

output.to_csv("kanpur_real.csv", index=False)
print(f"Saved {len(output)} daily rows to kanpur_real.csv")
print(output.head())
'@ | Set-Content -Encoding UTF8 prepare_real_data.py
