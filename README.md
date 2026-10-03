# Kanpur AQI Predictor

I wanted to know if tomorrow's air quality in Kanpur could be predicted 
from today's pollution readings. Turns out: kind of, but not very well.

## What it does

Trains a Random Forest classifier on real CPCB data from Kanpur stations. 
Predicts tomorrow's AQI category (Good, Satisfactory, Moderate, Poor, Very Poor, 
Severe).

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
