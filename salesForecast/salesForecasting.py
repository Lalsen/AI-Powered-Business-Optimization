# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.metrics import mean_absolute_error

# Step 1: Load Dataset
df = pd.read_csv("cleaned_sales_data.csv")  # Replace with actual file path

# Step 2: Data Preprocessing
df['Date'] = pd.to_datetime(df['Date'])  # Convert Date column to datetime format
df.dropna(inplace=True)  # Remove missing values
df = df.sort_values(by='Date')  # Ensure data is sorted chronologically

# Step 3: Aggregate sales by month
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month

monthly_sales = df.groupby(['Year', 'Month'])['Total'].sum().reset_index()
monthly_sales['Date'] = pd.to_datetime(monthly_sales[['Year', 'Month']].assign(day=1))
monthly_sales.set_index('Date', inplace=True)
monthly_sales.drop(columns=['Year', 'Month'], inplace=True)

# Step 4: Train-Test Split (Last 20% as test set)
train_size = int(len(monthly_sales) * 0.8)
train, test = monthly_sales.iloc[:train_size], monthly_sales.iloc[train_size:]

# Step 5: Visualize Sales Trend
plt.figure(figsize=(12,5))
plt.plot(train, label="Train Data")
plt.plot(test, label="Test Data", linestyle="dashed")
plt.xlabel("Date")
plt.ylabel("Total Sales")
plt.title("Monthly Sales Trend (Train-Test Split)")
plt.legend()
plt.grid()
plt.show()

# Step 6: Train ARIMA Model on Train Data
model_arima = ARIMA(train['Total'], order=(5,1,0))
model_arima_fit = model_arima.fit()

# Predict on Test Data
y_pred_arima = model_arima_fit.forecast(steps=len(test))
mae_arima = mean_absolute_error(test['Total'], y_pred_arima)

print("📊 ARIMA Model - Mean Absolute Error (MAE):", round(mae_arima, 2))

# Step 7: Train Facebook Prophet Model
df_prophet = train.reset_index()
df_prophet.columns = ['ds', 'y']

model_prophet = Prophet()
model_prophet.fit(df_prophet)

# Predict on Test Data
future = model_prophet.make_future_dataframe(periods=len(test), freq='M')
forecast_prophet = model_prophet.predict(future)

# Extract test predictions and compute MAE
test_pred_prophet = forecast_prophet[['ds', 'yhat']].tail(len(test))['yhat'].values
mae_prophet = mean_absolute_error(test['Total'], test_pred_prophet)

print("📊 Prophet Model - Mean Absolute Error (MAE):", round(mae_prophet, 2))

# Step 8: Predict Next Month's Sales
next_month_arima = model_arima_fit.forecast(steps=1)[0]
next_month_prophet = forecast_prophet[['ds', 'yhat']].iloc[-1]['yhat']

print("\n🔹 ARIMA Predicted Sales for Next Month:", round(next_month_arima, 2))
print("🔹 Prophet Predicted Sales for Next Month:", round(next_month_prophet, 2))

# Step 9: Plot Prophet Forecast
model_prophet.plot(forecast_prophet)
plt.title("Sales Forecast using Prophet")
plt.show()
