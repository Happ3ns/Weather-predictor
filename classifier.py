"""Train a next-day AQI category classifier from a downloaded CPCB CSV.

Download historical Kanpur observations from the CPCB Central Control Room
AQI Data Repository, then run:
    python classifier.py path/to/kanpur_cpcb.csv

The CSV should contain one row per observation, with date/time, AQI, and at
least one pollutant column. See DATA.md for supported column names.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless-safe: no display needed to save PNGs

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline

AQI_CATEGORIES = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]

COLUMN_ALIASES = {
    "date": {
        "date", "datetime", "timestamp", "time", "fromdate", "samplingdate",
        "dateandtime", "observationtime",
    },
    "city": {"city", "cityname", "location", "citylocation"},
    "aqi": {"aqi", "aqivalue", "airqualityindex", "indexvalue"},
    "pm25": {"pm25", "pm25ugm3", "particulatematter25", "particulatematter25ugm3"},
    "pm10": {"pm10", "pm10ugm3", "particulatematter10", "particulatematter10ugm3"},
    "no2": {"no2", "no2ugm3", "nitrogendioxide", "nitrogendioxideugm3"},
    "so2": {"so2", "so2ugm3", "sulfurdioxide", "sulfurdioxideugm3"},
    "co": {"co", "comgm3", "carbonmonoxide", "carbonmonoxidemgm3"},
    "o3": {"o3", "o3ugm3", "ozone", "ozoneugm3"},
    "temperature": {"temperature", "temp", "temperaturec", "ambienttemperature"},
    "humidity": {"humidity", "relativehumidity", "relativehumiditypercent"},
    "wind_speed": {"windspeed", "windspeedms", "windspeedkmh"},
}
POLLUTANT_FEATURES = ["pm25", "pm10", "no2", "so2", "co", "o3"]
WEATHER_FEATURES = ["temperature", "humidity", "wind_speed"]


def normalize_column_name(name: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).casefold())


def find_column(columns: pd.Index, canonical_name: str) -> str | None:
    aliases = COLUMN_ALIASES[canonical_name]
    for column in columns:
        if normalize_column_name(column) in aliases:
            return str(column)
    return None


def aqi_category(value: float) -> str:
    if value <= 50:
        return "Good"
    if value <= 100:
        return "Satisfactory"
    if value <= 200:
        return "Moderate"
    if value <= 300:
        return "Poor"
    if value <= 400:
        return "Very Poor"
    return "Severe"


def _sniff_header_row(csv_path: Path, max_scan: int = 10) -> int:
    """CPCB exports often have title/metadata rows before the real header.
    Find the first line that contains a recognisable date-ish column name."""
    with open(csv_path, "r", encoding="utf-8-sig", errors="replace") as fh:
        for index, line in enumerate(fh):
            if index >= max_scan:
                break
            normalized = normalize_column_name(line)
            if "fromdate" in normalized or "date" in normalized:
                return index
    return 0


def load_daily_data(csv_path: Path, requested_city: str) -> tuple[pd.DataFrame, list[str]]:
    header_row = _sniff_header_row(csv_path)
    if header_row:
        print(f"Skipping {header_row} metadata row(s) at the top of the CSV.")
    raw = pd.read_csv(
        csv_path,
        encoding="utf-8-sig",
        low_memory=False,
        skiprows=header_row,
    )
    date_column = find_column(raw.columns, "date")
    aqi_column = find_column(raw.columns, "aqi")
    city_column = find_column(raw.columns, "city")

    if date_column is None or aqi_column is None:
        raise ValueError(
            "CSV must include a date/time column (for example 'From Date') "
            "and a numeric AQI column. See DATA.md for supported names."
        )

    pollutant_columns = {
        feature: find_column(raw.columns, feature) for feature in POLLUTANT_FEATURES
    }
    pollutant_columns = {
        feature: column for feature, column in pollutant_columns.items() if column
    }
    if not pollutant_columns:
        raise ValueError(
            "CSV needs at least one pollutant column, such as PM2.5 or PM10. "
            "The model uses pollutant measurements as input features."
        )

    weather_columns = {
        feature: find_column(raw.columns, feature) for feature in WEATHER_FEATURES
    }
    weather_columns = {
        feature: column for feature, column in weather_columns.items() if column
    }

    parsed_dates = pd.to_datetime(
        raw[date_column], errors="coerce", dayfirst=True, format="mixed"
    )
    raw = raw.loc[parsed_dates.notna()].copy()
    raw["_date"] = parsed_dates.loc[parsed_dates.notna()].dt.normalize()
    raw["_aqi"] = pd.to_numeric(raw[aqi_column], errors="coerce")
    raw = raw.loc[raw["_aqi"].between(0, 500)].copy()

    if city_column:
        matching_city = raw[city_column].astype("string").str.contains(
            re.escape(requested_city), case=False, na=False
        )
        raw = raw.loc[matching_city].copy()
        if raw.empty:
            raise ValueError(
                f"No rows matched city '{requested_city}' in column '{city_column}'."
            )
        raw["_city"] = raw[city_column].astype(str)
    else:
        raw["_city"] = requested_city
        print(
            f"No city column found; assuming this CSV contains only {requested_city} data."
        )

    source_features = {**pollutant_columns, **weather_columns}
    for feature, column in source_features.items():
        raw[feature] = pd.to_numeric(raw[column], errors="coerce")

    feature_names = list(source_features)
    value_columns = ["_aqi", *feature_names]
    daily = (
        raw.groupby(["_city", "_date"], as_index=False)[value_columns]
        .mean()
        .sort_values(["_city", "_date"])
    )
    return daily, feature_names


def add_calendar_features(rows: pd.DataFrame) -> pd.DataFrame:
    features = rows.copy()
    day_of_year = features["_date"].dt.dayofyear
    angle = 2 * np.pi * (day_of_year - 1) / 365.25
    features["season_sin"] = np.sin(angle)
    features["season_cos"] = np.cos(angle)
    return features


def make_next_day_examples(
    daily: pd.DataFrame, feature_names: list[str]
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    targets = daily[["_city", "_date", "_aqi"]].rename(
        columns={"_date": "_target_date", "_aqi": "_next_day_aqi"}
    )
    current = daily.copy()
    current["_target_date"] = current["_date"] + pd.Timedelta(days=1)
    paired = current.merge(
        targets,
        on=["_city", "_target_date"],
        how="inner",
        validate="one_to_one",
    )

    paired["_today_aqi"] = paired["_aqi"]
    paired = add_calendar_features(paired)
    model_features = [
        *feature_names,
        "_today_aqi",
        "season_sin",
        "season_cos",
    ]
    x = paired[model_features]
    y = paired["_next_day_aqi"].map(aqi_category)
    baseline = paired["_today_aqi"].map(aqi_category)
    return x, y, baseline


def plot_feature_importances(
    model, feature_names: list[str], output_path: Path
) -> None:
    forest = model.named_steps["randomforestclassifier"]
    importances = forest.feature_importances_
    order = np.argsort(importances)  # ascending, so barh reads top-to-bottom nicely

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(
        [feature_names[i] for i in order],
        importances[order],
        color="#2F6F4E",
    )
    ax.set_xlabel("Relative importance")
    ax.set_title("What drives the next-day AQI prediction")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(
    matrix: np.ndarray, labels: list[str], output_path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5.5))
    im = ax.imshow(matrix, cmap="Greens")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted category")
    ax.set_ylabel("Actual category")
    ax.set_title("Confusion matrix: next-day AQI category")

    threshold = matrix.max() / 2 if matrix.max() > 0 else 0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            ax.text(
                col,
                row,
                str(value),
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
            )

    fig.colorbar(im, ax=ax, shrink=0.8, label="Count")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def train_and_report(daily: pd.DataFrame, feature_names: list[str]) -> None:
    x, y, persistence_baseline = make_next_day_examples(daily, feature_names)
    if len(x) < 10:
        raise ValueError(
            f"Only {len(x)} consecutive next-day pairs are available. "
            "Use more historical data (at least 10 pairs; several months is better)."
        )

    split_at = int(len(x) * 0.8)
    split_at = min(max(split_at, 1), len(x) - 1)
    x_train, x_test = x.iloc[:split_at], x.iloc[split_at:]
    y_train, y_test = y.iloc[:split_at], y.iloc[split_at:]
    baseline_test = persistence_baseline.iloc[split_at:]

    model = make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=42,
        ),
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    print("CPCB next-day AQI category classifier")
    print(f"Daily rows: {len(daily)}")
    print(f"Consecutive next-day pairs: {len(x)}")
    print(f"Training pairs: {len(x_train)} (earlier dates)")
    print(f"Test pairs: {len(x_test)} (later dates)")
    print(f"Input features: {', '.join(feature_names)}, today's AQI, season")
    print(f"Model accuracy: {accuracy_score(y_test, predictions):.2%}")
    print(f"Persistence baseline accuracy: {accuracy_score(y_test, baseline_test):.2%}")
    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=AQI_CATEGORIES,
            target_names=AQI_CATEGORIES,
            zero_division=0,
        )
    )
    print("Confusion matrix (actual rows, predicted columns):")
    matrix = confusion_matrix(y_test, predictions, labels=AQI_CATEGORIES)
    print(matrix)

    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    importance_path = output_dir / "feature_importance.png"
    confusion_path = output_dir / "confusion_matrix.png"
    all_feature_names = [*feature_names, "_today_aqi", "season_sin", "season_cos"]
    display_names = {
        "_today_aqi": "today's AQI",
        "season_sin": "season (sin)",
        "season_cos": "season (cos)",
    }
    readable_names = [display_names.get(name, name) for name in all_feature_names]
    plot_feature_importances(model, readable_names, importance_path)
    plot_confusion_matrix(matrix, AQI_CATEGORIES, confusion_path)
    print(f"\nSaved plots to {importance_path} and {confusion_path}")

    latest = add_calendar_features(daily.tail(1))
    latest_features = latest[[*feature_names, "_aqi", "season_sin", "season_cos"]].rename(
        columns={"_aqi": "_today_aqi"}
    )
    latest_prediction = model.predict(latest_features)[0]
    latest_date = latest["_date"].iloc[0].date()
    print(
        f"\nLatest available day: {latest_date}; "
        f"predicted next-day category: {latest_prediction}"
    )
    print("This is an educational estimate, not an official CPCB forecast.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train a CPCB-based model for next-day AQI category."
    )
    parser.add_argument("csv", type=Path, help="Path to a downloaded CPCB CSV")
    parser.add_argument(
        "--city", default="Kanpur", help="City name to select (default: Kanpur)"
    )
    args = parser.parse_args()

    if not args.csv.is_file():
        parser.error(f"CSV file not found: {args.csv}")

    try:
        daily, feature_names = load_daily_data(args.csv, args.city)
        print(f"Using {args.city} observations from {args.csv}")
        train_and_report(daily, feature_names)
    except (OSError, pd.errors.ParserError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
