#!/usr/bin/env python3
"""Interactive single-day Kanpur AQI predictor and advisory CLI.

Computes official CPCB piecewise sub-indices across pollutants and forecasts
tomorrow's AQI category using meteorological drift and seasonal priors.

Usage:
    python predict_today.py --pm25 145 --pm10 210 --temp 24 --humidity 65 --wind 1.8
    python predict_today.py --interactive
    python predict_today.py
"""

from __future__ import annotations

import argparse
import datetime as dt
import math
import sys

# Official CPCB AQI Categories & Breakpoints
CATEGORIES = [
    ("Good", 0, 50, "\033[92m"),           # Green
    ("Satisfactory", 51, 100, "\033[92m"),   # Light Green
    ("Moderate", 101, 200, "\033[93m"),     # Yellow
    ("Poor", 201, 300, "\033[91m"),         # Orange/Red
    ("Very Poor", 301, 400, "\033[95m"),    # Magenta
    ("Severe", 401, 500, "\033[91;1m"),     # Bold Red
]
RESET_COLOR = "\033[0m"


def get_category_info(aqi: float) -> tuple[str, str]:
    for name, low, high, color in CATEGORIES:
        if low <= aqi <= high:
            return name, color
    if aqi > 500:
        return "Severe", "\033[91;1m"
    return "Good", "\033[92m"


# Piecewise linear sub-index calculator (Official CPCB methodology)
def calc_sub_index(val: float | None, breakpoints: list[tuple[float, float, float, float]]) -> float | None:
    if val is None or math.isnan(val) or val < 0:
        return None
    for bp_low, bp_high, idx_low, idx_high in breakpoints:
        if bp_low <= val <= bp_high:
            return round(idx_low + (idx_high - idx_low) / (bp_high - bp_low) * (val - bp_low))
    # Above highest defined breakpoint
    last = breakpoints[-1]
    return min(500.0, round(last[2] + (last[3] - last[2]) / (last[1] - last[0]) * (val - last[0])))


def calc_pm25_subindex(pm: float | None) -> float | None:
    bps = [
        (0, 30, 0, 50),
        (30, 60, 51, 100),
        (60, 90, 101, 200),
        (90, 120, 201, 300),
        (120, 250, 301, 400),
        (250, 500, 401, 500),
    ]
    return calc_sub_index(pm, bps)


def calc_pm10_subindex(pm: float | None) -> float | None:
    bps = [
        (0, 50, 0, 50),
        (50, 100, 51, 100),
        (100, 250, 101, 200),
        (250, 350, 201, 300),
        (350, 430, 301, 400),
        (430, 600, 401, 500),
    ]
    return calc_sub_index(pm, bps)


def calc_no2_subindex(val: float | None) -> float | None:
    bps = [
        (0, 40, 0, 50),
        (40, 80, 51, 100),
        (80, 180, 101, 200),
        (180, 280, 201, 300),
        (280, 400, 301, 400),
        (400, 1000, 401, 500),
    ]
    return calc_sub_index(val, bps)


def calc_so2_subindex(val: float | None) -> float | None:
    bps = [
        (0, 40, 0, 50),
        (40, 80, 51, 100),
        (80, 380, 101, 200),
        (380, 800, 201, 300),
        (800, 1600, 301, 400),
        (1600, 2500, 401, 500),
    ]
    return calc_sub_index(val, bps)


def compute_composite_aqi(readings: dict[str, float | None]) -> tuple[float, str]:
    sub_indices: dict[str, float] = {}
    if readings.get("pm25") is not None:
        idx = calc_pm25_subindex(readings["pm25"])
        if idx is not None:
            sub_indices["PM2.5"] = idx
    if readings.get("pm10") is not None:
        idx = calc_pm10_subindex(readings["pm10"])
        if idx is not None:
            sub_indices["PM10"] = idx
    if readings.get("no2") is not None:
        idx = calc_no2_subindex(readings["no2"])
        if idx is not None:
            sub_indices["NO2"] = idx
    if readings.get("so2") is not None:
        idx = calc_so2_subindex(readings["so2"])
        if idx is not None:
            sub_indices["SO2"] = idx

    if not sub_indices:
        return 100.0, "Default (No Pollutant Data)"

    max_pollutant = max(sub_indices, key=sub_indices.get)  # type: ignore
    return sub_indices[max_pollutant], max_pollutant


def predict_next_day(
    today_aqi: float,
    temp: float,
    humidity: float,
    wind: float,
    day_of_year: int,
) -> tuple[str, float, dict[str, float]]:
    """Predict tomorrow's category accounting for meteorological dispersal and seasonality."""
    # Meteorological dispersion factor
    # Higher wind and temperature increase boundary layer mixing, lowering accumulation
    # High humidity (>75%) promotes aerosol swelling and stagnation in Indo-Gangetic Plain
    dispersion = 1.0
    if wind > 3.0:
        dispersion -= 0.12 * min(2.0, (wind - 3.0))
    elif wind < 1.0:
        dispersion += 0.08  # Stagnation trap

    if humidity > 70:
        dispersion += 0.05 * ((humidity - 70) / 30)

    # Seasonal baseline shift (Kanpur winter peaks around day 300 - 365, low in monsoon day 180-240)
    seasonal_bias = math.sin((day_of_year - 80) / 365.0 * 2 * math.pi)
    # Winter brings high baseline (+), Monsoon brings washdown (-)
    season_factor = 1.0 - (0.15 * seasonal_bias)

    predicted_val = today_aqi * dispersion * season_factor
    pred_category, _ = get_category_info(predicted_val)

    # Compute probability distribution over categories
    probs: dict[str, float] = {}
    total_w = 0.0
    for name, low, high, _ in CATEGORIES:
        mid = (low + high) / 2.0
        dist = abs(predicted_val - mid)
        w = math.exp(-0.5 * (dist / 45.0) ** 2)
        probs[name] = w
        total_w += w

    for k in probs:
        probs[k] = round(probs[k] / total_w, 3)

    confidence = probs[pred_category]
    return pred_category, confidence, probs


def get_health_advisory(category: str) -> str:
    advisories = {
        "Good": "Air quality is satisfactory. Ideal for all outdoor recreational activities.",
        "Satisfactory": "Minor breathing discomfort to sensitive people. Normal outdoor activity permitted.",
        "Moderate": "Breathing discomfort to people with lungs, asthma, and heart diseases. Wear mask if sensitive.",
        "Poor": "Breathing discomfort to most people on prolonged exposure. Avoid strenuous outdoor exercise.",
        "Very Poor": "Respiratory illness on prolonged exposure. Use HEPA purifiers indoors and N95 masks outdoors.",
        "Severe": "Affects healthy people and seriously impacts those with existing diseases. Stay indoors.",
    }
    return advisories.get(category, "Check local pollution advisories.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict next-day Kanpur AQI from observed meteorological data.")
    parser.add_argument("--pm25", type=float, default=None, help="Today's PM2.5 24-hr average (µg/m³)")
    parser.add_argument("--pm10", type=float, default=None, help="Today's PM10 24-hr average (µg/m³)")
    parser.add_argument("--temp", type=float, default=26.0, help="Ambient temperature (°C, default 26)")
    parser.add_argument("--humidity", type=float, default=60.0, help="Relative humidity (%%, default 60)")
    parser.add_argument("--wind", type=float, default=1.8, help="Wind speed (m/s, default 1.8)")
    parser.add_argument("--no2", type=float, default=None, help="NO2 level (µg/m³)")
    parser.add_argument("--so2", type=float, default=None, help="SO2 level (µg/m³)")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive question prompt")

    args = parser.parse_args()

    pm25 = args.pm25
    pm10 = args.pm10
    temp = args.temp
    humidity = args.humidity
    wind = args.wind

    if args.interactive or (pm25 is None and len(sys.argv) == 1):
        print("\n=== Kanpur Next-Day AQI Predictor (Interactive Mode) ===")
        try:
            val = input("Enter today's PM2.5 (µg/m³) [default: 135]: ").strip()
            pm25 = float(val) if val else 135.0
            val = input("Enter today's PM10 (µg/m³) [default: 210]: ").strip()
            pm10 = float(val) if val else 210.0
            val = input("Enter temperature (°C) [default: 25.0]: ").strip()
            temp = float(val) if val else 25.0
            val = input("Enter relative humidity (%) [default: 65.0]: ").strip()
            humidity = float(val) if val else 65.0
            val = input("Enter wind speed (m/s) [default: 1.5]: ").strip()
            wind = float(val) if val else 1.5
        except (ValueError, EOFError, KeyboardInterrupt):
            print("\nUsing Kanpur typical seasonal values...")
            pm25 = 135.0
            pm10 = 210.0

    if pm25 is None:
        pm25 = 135.0
    if pm10 is None:
        pm10 = pm25 * 1.55  # typical ratio in Kanpur urban airshed

    today_date = dt.date.today()
    day_of_year = today_date.timetuple().tm_yday

    readings = {
        "pm25": pm25,
        "pm10": pm10,
        "no2": args.no2,
        "so2": args.so2,
    }

    current_aqi, driver = compute_composite_aqi(readings)
    cur_cat, cur_col = get_category_info(current_aqi)

    pred_cat, conf, probs = predict_next_day(current_aqi, temp, humidity, wind, day_of_year)
    _, pred_col = get_category_info(
        {"Good": 25, "Satisfactory": 75, "Moderate": 150, "Poor": 250, "Very Poor": 350, "Severe": 450}[pred_cat]
    )

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    print("\n" + "=" * 62)
    print("      KANPUR AIR QUALITY INDEX - OFFICIAL CPCB REPORT")
    print(f"      Date: {today_date.strftime('%B %d, %Y')} (Day {day_of_year} of 365)")
    print("=" * 62)
    print(f" Observed Inputs:")
    print(f"   * PM2.5:     {pm25:.1f} ug/m3   (Sub-index: {calc_pm25_subindex(pm25)})")
    print(f"   * PM10:      {pm10:.1f} ug/m3   (Sub-index: {calc_pm10_subindex(pm10)})")
    print(f"   * Weather:   {temp:.1f}C  |  {humidity:.1f}% RH  |  {wind:.1f} m/s wind")
    print("-" * 62)
    print(f" Today's Computed AQI:  {current_aqi:.0f} [{cur_cat.upper()}] (Driven by {driver})")
    print(f" Tomorrow's Forecast:   {pred_cat.upper()} (Confidence: {conf*100:.1f}%)")
    print(f" Persistence Baseline:  {cur_cat.upper()} (tomorrow = today assumption)")
    print("-" * 62)
    print(" Category Probability Distribution:")
    for cat_name, p in probs.items():
        bar = "#" * int(p * 25)
        print(f"   {cat_name:<14} {p*100:5.1f}% | {bar}")
    print("-" * 62)
    print(f" Health Advisory:\n   {get_health_advisory(pred_cat)}")
    print("=" * 62 + "\n")


if __name__ == "__main__":
    main()
