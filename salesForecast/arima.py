import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error

# Load Dataset
file_path = "cleaned_sales_data.csv"
df = pd.read_csv(file_path)

# Convert Date column to datetime format
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y %H:%M')

# Aggregate sales by month
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
monthly_sales = df.groupby(['Year', 'Month'])['Total'].sum().reset_index()
monthly_sales['Date'] = pd.to_datetime(monthly_sales[['Year', 'Month']].assign(day=1))
monthly_sales.set_index('Date', inplace=True)
monthly_sales.drop(columns=['Year', 'Month'], inplace=True)

# Train-Test Split
train_size = int(len(monthly_sales) * 0.8)
train, test = monthly_sales.iloc[:train_size], monthly_sales.iloc[train_size:]

# Train ARIMA Model
model_arima = ARIMA(train['Total'], order=(5,1,0))
model_arima_fit = model_arima.fit()

# Forecast future sales
y_pred_arima = model_arima_fit.forecast(steps=len(test))
mae_arima = mean_absolute_error(test['Total'], y_pred_arima)
print("ARIMA Model - Mean Absolute Error (MAE):", round(mae_arima, 2))

# Plot ARIMA Forecast
plt.figure(figsize=(12,5))
plt.plot(train, label="Train Data")
plt.plot(test, label="Test Data", linestyle="dashed")
plt.plot(test.index, y_pred_arima, label="ARIMA Forecast", linestyle="dotted", color='red')
plt.xlabel("Date")
plt.ylabel("Total Sales")
plt.title("Sales Forecast using ARIMA")
plt.legend()
plt.grid()
plt.show()

# Predict next month's sales
next_month_arima = model_arima_fit.forecast(steps=1)[0]
print("\nPredicted Sales for Next Month:", round(next_month_arima, 2))
