#sales forecast

# forecast.py

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error

def run_arima_forecast():
    df = pd.read_csv("cleaned_sales_data.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df.dropna(inplace=True)
    df = df.sort_values(by='Date')

    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month

    monthly_sales = df.groupby(['Year', 'Month'])['Total'].sum().reset_index()
    monthly_sales['Date'] = pd.to_datetime(monthly_sales[['Year', 'Month']].assign(day=1))
    monthly_sales.set_index('Date', inplace=True)
    monthly_sales.drop(columns=['Year', 'Month'], inplace=True)

    train_size = int(len(monthly_sales) * 0.8)
    train, test = monthly_sales.iloc[:train_size], monthly_sales.iloc[train_size:]

    model_arima = ARIMA(train['Total'], order=(5,1,0))
    model_arima_fit = model_arima.fit()

    y_pred_arima = model_arima_fit.forecast(steps=len(test))
    next_month_arima = model_arima_fit.forecast(steps=1)[0]

    # Plot and save figure
    plt.figure(figsize=(12, 5))
    plt.plot(monthly_sales.index, monthly_sales['Total'], label="Actual Sales", color="blue")
    plt.plot(train.index, train['Total'], label="Train Data", color="green")
    plt.plot(test.index, y_pred_arima, label="ARIMA Forecast", color="red", linestyle="dashed")
    plt.xlabel("Date")
    plt.ylabel("Total Sales")
    plt.title("Sales Forecast using ARIMA")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plot_path = "static/forecast_plot.png"
    plt.savefig(plot_path)
    plt.close()

    return round(next_month_arima, 2), plot_path
