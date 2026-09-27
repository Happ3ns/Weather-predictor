# CPCB AQI classifier data

The classifier trains from a CSV downloaded from the official [CPCB Central Control Room](https://airquality.cpcb.gov.in/ccr/). Open its AQI Data Repository and download historical observations for Kanpur. No dataset is bundled here, so the script will not present sample or generated rows as CPCB measurements.

## Expected CSV

Use a wide-format CSV with one observation per row. Required fields:

- A date/time field, such as `From Date`, `Date`, `Timestamp`, or `DateTime`
- A numeric AQI field, such as `AQI` or `Air Quality Index`
- At least one pollutant field: `PM2.5`, `PM10`, `NO2`, `SO2`, `CO`, or `O3`

The city field is optional. When present, it should be named `City`, `City Name`, or `Location`; the script filters it to Kanpur. If the export contains no city field, it assumes the file is already filtered to Kanpur. Weather fields are optional: temperature, relative humidity, and wind speed are used if available. The script averages observations into daily city-level values.

The model predicts the next calendar day's category from the current day's pollutant readings, current AQI, and season. It uses a chronological 80/20 split and also reports a persistence baseline (today's category used as tomorrow's prediction). CPCB's six AQI categories are used: Good, Satisfactory, Moderate, Poor, Very Poor, and Severe.

## Run

Create a virtual environment, install dependencies, then pass the downloaded file:

```bash
python3 -m pip install -r requirements.txt
python3 classifier.py path/to/kanpur_cpcb.csv
```

On Windows (PowerShell):

```powershell
python -m pip install -r requirements.txt
python classifier.py path\to\kanpur_cpcb.csv
```

The latest-day forecast is an educational model estimate, not an official CPCB forecast. Model performance depends on the amount and quality of the downloaded historical data.

## Don't have a real export yet?

Run `python3 generate_sample_data.py` to create a synthetic (clearly labeled, not real) CSV in the same format, so you can confirm the pipeline runs before downloading real historical data.
