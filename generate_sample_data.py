"""Generate a SYNTHETIC CPCB-style CSV so the classifier pipeline can be run
and inspected without first downloading real historical data.

This does NOT produce real air quality measurements. Values are randomly
generated with a seasonal bias (worse air quality in winter months, which is
a real, well-documented pattern for North Indian cities, used here only to
make the synthetic numbers plausible-looking) -- they are not derived from
any actual CPCB reading. Do not use this data, or any model trained on it,
to draw real conclusions about Kanpur's actual air quality.

Usage:
    python generate_sample_data.py [output_path] [--days N] [--seed S]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd


def synthetic_daily_frame(days: int, seed: int | None = None) -> pd.DataFrame:
    # If no seed is provided, derive one from the current time so every run
    # produces fresh values. Pass --seed N for reproducible output.
    if seed is None:
        seed = int(time.time_ns() % (2**32))

    rng = np.random.default_rng(seed)

    # End the series on today's date so the "latest available day" printed by
    # classifier.py is current. The first row is (days - 1) days before today.
    end_date = pd.Timestamp.today().normalize()
    dates = pd.date_range(end=end_date, periods=days, freq="D")

    day_of_year = dates.dayofyear.to_numpy()
    # Winter (low day-of-year / high day-of-year, i.e. Dec-Feb) biased worse;
    # monsoon months (roughly day 150-270) biased better. Purely illustrative.
    seasonal = np.cos(2 * np.pi * (day_of_year - 15) / 365.25)
    winter_bias = np.clip(seasonal, 0, 1)  # 1 near winter, 0 near monsoon

    base_pm25 = 60 + 140 * winter_bias + rng.normal(0, 15, days)
    base_pm10 = base_pm25 * rng.uniform(1.4, 1.9, days)
    no2 = 20 + 30 * winter_bias + rng.normal(0, 8, days)
    so2 = 8 + 12 * winter_bias + rng.normal(0, 4, days)
    co = 0.8 + 1.2 * winter_bias + rng.normal(0, 0.3, days)
    o3 = 25 + 20 * (1 - winter_bias) + rng.normal(0, 6, days)
    temperature = 25 - 12 * winter_bias + rng.normal(0, 2, days)
    humidity = 45 + 25 * (1 - winter_bias) + rng.normal(0, 8, days)
    wind_speed = 8 + rng.normal(0, 2.5, days)

    pm25 = np.clip(base_pm25, 5, None)
    pm10 = np.clip(base_pm10, 10, None)

    # A simple, deliberately approximate AQI proxy driven mainly by PM2.5/PM10,
    # nudged by the gas pollutants, plus noise -- not CPCB's real formula.
    aqi = (
        0.6 * pm25
        + 0.25 * pm10 / 1.6
        + 0.5 * no2
        + 0.4 * so2
        + 10 * co
        + rng.normal(0, 10, days)
    )
    aqi = np.clip(aqi, 10, 480)

    frame = pd.DataFrame(
        {
            "From Date": dates.strftime("%d-%m-%Y"),
            "City": "Kanpur",
            "PM2.5": pm25.round(1),
            "PM10": pm10.round(1),
            "NO2": np.clip(no2, 2, None).round(1),
            "SO2": np.clip(so2, 1, None).round(1),
            "CO": np.clip(co, 0.1, None).round(2),
            "O3": np.clip(o3, 5, None).round(1),
            "Temperature": temperature.round(1),
            "Humidity": np.clip(humidity, 10, 100).round(1),
            "Wind Speed": np.clip(wind_speed, 0, None).round(1),
            "AQI": aqi.round(0),
        }
    )
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        default=Path("sample_kanpur_synthetic.csv"),
        help="Where to write the synthetic CSV (default: %(default)s)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=400,
        help="Number of consecutive synthetic days to generate (default: %(default)s)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible output (default: random each run)",
    )
    args = parser.parse_args()

    # Make sure the output folder exists (matters if you pass a path with a folder).
    args.output.parent.mkdir(parents=True, exist_ok=True)

    frame = synthetic_daily_frame(args.days, seed=args.seed)
    frame.to_csv(args.output, index=False)
    first = frame["From Date"].iloc[0]
    last = frame["From Date"].iloc[-1]
    print(f"Wrote {len(frame)} rows of SYNTHETIC data to {args.output}")
    print(f"Date range: {first} -> {last} (last row is today)")
    print("Reminder: this is fabricated demo data, not real CPCB measurements.")
    print(f"Try it: python classifier.py {args.output}")


if __name__ == "__main__":
    main()
