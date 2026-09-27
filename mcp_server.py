from pathlib import Path
from mcp.server.fastmcp import FastMCP
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline

# Import core functions from your existing classifier script
from classifier import (
    load_daily_data,
    make_next_day_examples,
    add_calendar_features,
)

# Initialize the FastMCP server
mcp = FastMCP("Weather Predictor MCP")

@mcp.tool()
def predict_next_day_aqi(csv_path: str, city: str = "Kanpur") -> str:
    """
    Trains a Random Forest classifier on historical CPCB data[cite: 1] 
    and predicts the next-day AQI category for a given city[cite: 4].
    """
    path = Path(csv_path)
    if not path.is_file():
        return f"Error: CSV file not found at {csv_path}"
    
    try:
        daily, feature_names = load_daily_data(path, city)
        x, y, persistence_baseline = make_next_day_examples(daily, feature_names)
        
        if len(x) < 10:
            return f"Error: Only {len(x)} consecutive pairs available. Provide more historical data."

        split_at = int(len(x) * 0.8)
        split_at = min(max(split_at, 1), len(x) - 1)
        x_train, x_test = x.iloc[:split_at], x.iloc[split_at:]
        y_train, y_test = y.iloc[:split_at], y.iloc[split_at:]

        # Train model
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
        acc = accuracy_score(y_test, predictions)

        # Predict next day based on latest row
        latest = add_calendar_features(daily.tail(1))
        latest_features = latest[[*feature_names, "_aqi", "season_sin", "season_cos"]].rename(
            columns={"_aqi": "_today_aqi"}
        )
        latest_prediction = model.predict(latest_features)[0]
        latest_date = latest["_date"].iloc[0].date()

        return (
            f"City: {city}\n"
            f"Latest Date in Data: {latest_date}\n"
            f"Model Test Accuracy: {acc:.2%}\n"
            f"Predicted Next-Day AQI Category: {latest_prediction}\n"
            f"Note: Educational estimate, not an official CPCB forecast[cite: 4]."
        )
    except Exception as error:
        return f"Error executing model pipeline: {str(error)}"

if __name__ == "__main__":
    mcp.run()
