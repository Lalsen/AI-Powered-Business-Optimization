import os
import pandas as pd
import xgboost as xgb
from datetime import datetime
from sklearn.preprocessing import LabelEncoder


# Define file paths
DATA_DIR = "sales_data"
LATEST_FILE = os.path.join(DATA_DIR, "latest_sales.csv")
MODEL_FILE = "sales_forecast.model"

# Ensure the data directory exists
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Delete outdated CSV files (keep only latest data)
def clean_old_data():
    for file in os.listdir(DATA_DIR):
        if file.endswith(".csv") and file != "latest_sales.csv":
            os.remove(os.path.join(DATA_DIR, file))

# Load the latest dataset
def load_latest_data():
    if os.path.exists(LATEST_FILE):
        return pd.read_csv(LATEST_FILE)
    else:
        raise FileNotFoundError("Latest sales data file not found!")

# Train the XGBoost model


def train_model(data):
    X = data.drop(columns=["pid", "product_name"])  # Features (monthly sales)
    y = X.iloc[:, -1]  # Use the last month as target for prediction

    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)
    model.fit(X, y)
    model.save_model(MODEL_FILE)
    print("Model retrained and saved!")


# Predict next month's sales
def predict_next_month(data):
    model = xgb.XGBRegressor()
    model.load_model(MODEL_FILE)
    
    X_future = data.drop(columns=["pid", "product_name"])  # Future input
    predictions = model.predict(X_future)
    
    # Save predictions with product ID and name
    next_month = datetime.now().strftime("%B_%Y")
    pred_file = os.path.join(DATA_DIR, f"sales_forecast_{next_month}.csv")
    
    result = data[["pid", "product_name"]].copy()
    result["predicted_sales"] = predictions
    result.to_csv(pred_file, index=False)

    print(f"Predictions saved for {next_month}: {pred_file}")

# Main process
def main():
    clean_old_data()
    data = load_latest_data()
    train_model(data)
    predict_next_month(data)

if __name__ == "__main__":
    main()
