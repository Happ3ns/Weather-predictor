import pandas as pd
from pathlib import Path

# 1. Load the Vonter dataset (adjust filename if needed)
input_file = "latest-air-quality.parquet"  # or your downloaded .csv.gz file
print(f"Loading {input_file}...")
df = pd.read_parquet(input_file) # Use pd.read_csv(..., compression='gzip') if it's a .csv.gz

# 2. Filter for Kanpur
print("Filtering for Kanpur...")
kanpur = df[df['City'].str.contains('Kanpur', case=False, na=False)].copy()

if kanpur.empty:
    raise ValueError("No Kanpur data found. Check the 'City' column values.")

# 3. Convert Timestamp to datetime and set as index
kanpur['Timestamp'] = pd.to_datetime(kanpur['Timestamp'])
kanpur = kanpur.set_index('Timestamp')

# 4. Aggregate 15-min data to Daily averages
print("Aggregating to daily averages...")
daily = kanpur.resample('D').mean(numeric_only=True)

# 5. Rename columns to match classifier.py expectations
# classifier.py looks for: From Date, City, PM2.5, PM10, NO2, SO2, CO, O3, Temperature, Humidity, Wind Speed, AQI
rename_map = {
    'PM2.5 (µg/m³)': 'PM2.5',
    'PM10 (µg/m³)': 'PM10',
    'NO2 (µg/m³)': 'NO2',
    'SO2 (µg/m³)': 'SO2',
    'CO (mg/m³)': 'CO',
    'Ozone (µg/m³)': 'O3',
    'AT (°C)': 'Temperature',
    'RH (%)': 'Humidity',
    'WS (m/s)': 'Wind Speed'
}
daily = daily.rename(columns=rename_map)

# 6. Add required columns
daily = daily.reset_index()
daily['From Date'] = daily['Timestamp'].dt.strftime('%d-%m-%Y')
daily['City'] = 'Kanpur'

# The Vonter dataset does not include a pre-calculated AQI column in the 15-min data.
# We will compute a simple proxy AQI based on PM2.5 (a common approximation).
# You can adjust this formula or omit AQI if you have the official CPCB AQI data.
daily['AQI'] = (daily['PM2.5'] * 1.5).round(0) # Rough proxy

# 7. Select and order the columns classifier.py needs
final_cols = ['From Date', 'City', 'PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 'Temperature', 'Humidity', 'Wind Speed', 'AQI']
# Keep only columns that exist in the dataframe
final_cols = [c for c in final_cols if c in daily.columns]
output = daily[final_cols].dropna(subset=['From Date', 'AQI'])

# 8. Save
output.to_csv('kanpur_real.csv', index=False)
print(f"✅ Saved {len(output)} daily rows to kanpur_real.csv")
print(output.head())
