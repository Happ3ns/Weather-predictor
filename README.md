# Kanpur AQI Predictor

I wanted to know if tomorrow's air quality in Kanpur could be predicted 
from today's pollution readings. Turns out: kind of, but not very well.

## What it does

Trains a Random Forest classifier on real CPCB data from Kanpur stations. 
Predicts tomorrow's AQI category (Good, Satisfactory, Moderate, Poor, Very Poor, 
Severe).

## Architecture

```mermaid
flowchart TD
    subgraph SOURCES["Data Sources"]
        A1[Vonter India CPCB Archive<br/>761 MB Parquet file]
        A2[generate_sample_data.py<br/>Synthetic demo data]
    end

    subgraph PREP["Data Preparation — prepare_real_data.py"]
        B1[Filter for Kanpur stations]
        B2[Aggregate 15-min → daily averages]
        B3[Compute AQI from PM2.5<br/>using CPCB breakpoints]
        B4[(kanpur_real.csv<br/>365 daily rows)]
    end

    subgraph FEATURES["Feature Engineering"]
        C1[Pollutant features<br/>PM2.5, PM10, NO2, SO2, CO, O3]
        C2[Weather features<br/>Temperature, Humidity, Wind Speed]
        C3[Cyclical season encoding<br/>sin/cos of day-of-year]
        C4[Today's AQI]
    end

    subgraph MODEL["Model — classifier.py"]
        D1[Chronological train/test split<br/>80/20]
        D2[SimpleImputer<br/>median strategy]
        D3[RandomForestClassifier<br/>300 estimators<br/>class-balanced]
        D4[Persistence baseline<br/>tomorrow = today]
    end

    subgraph OUTPUT["Outputs"]
        E1[Accuracy report]
        E2[Classification report]
        E3[feature_importance.png]
        E4[confusion_matrix.png]
        E5[Next-day AQI prediction]
    end

    A1 --> B1
    A2 --> B4
    B1 --> B2 --> B3 --> B4
    B4 --> C1 & C2 & C3 & C4
    C1 & C2 & C3 & C4 --> D1
    D1 --> D2 --> D3
    D1 --> D4
    D3 --> E1 & E2 & E3 & E4 & E5
    D4 --> E1
```
## Results

- Model accuracy: 67.1%
- Naive baseline ("tomorrow = today"): 61.6%

So it beats the dumb guess by about 5 points. Not amazing. I'll explain why below.

## Why the accuracy is low

Two reasons I figured out while building this:

1. **The AQI formula I used at first was wrong.** I was doing `PM2.5 × 1.5`, which 
   underestimates high-pollution days. The real CPCB formula uses non-linear breakpoints. 
   After I fixed this, the labels changed a lot.

2. **The model only sees today's weather.** Tomorrow's wind speed and direction 
   matter a lot for AQI, but I don't have those as inputs. Adding them would probably help.

## What I'd do differently

- Add weather forecast features
- Try an LSTM or ARIMA instead of a classifier
- Predict the AQI number, not the category

## How to run it

1. Install dependencies: `pip install -r requirements.txt`
2. Download the Vonter Parquet file (see DATA.md)
3. Run `python prepare_real_data.py` to create `kanpur_real.csv`
4. Run `python classifier.py kanpur_real.csv`

Or just run the synthetic demo:
`python generate_sample_data.py`
`python classifier.py sample_kanpur_synthetic.csv`
