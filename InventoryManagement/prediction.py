## Load Dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet

# Load dataset
file_path = "cleaned_sales_data.csv"  # Update with actual file path
df = pd.read_csv(file_path)

# Convert Date column to datetime format
df['Date'] = pd.to_datetime(df['Date'])

## Aggregate Sales by Product Category and Month
df['YearMonth'] = df['Date'].dt.to_period('M')
monthly_sales = df.groupby(['YearMonth', 'ProductCategory'])['Quantity'].sum().reset_index()
monthly_sales['YearMonth'] = monthly_sales['YearMonth'].astype(str)
monthly_sales['Date'] = pd.to_datetime(monthly_sales['YearMonth'])

## Forecast Next Month's Stock Per Category
forecast_results = []
lead_time = 5  # Example lead time in days

for category in monthly_sales['ProductCategory'].unique():
    category_data = monthly_sales[monthly_sales['ProductCategory'] == category]
    category_data = category_data[['Date', 'Quantity']].set_index('Date')
    
    # Train ARIMA Model
    model = ARIMA(category_data, order=(5,1,0))
    model_fit = model.fit()
    
    # Forecast next month's sales
    forecast = model_fit.forecast(steps=1)
    next_month_sales = forecast.iloc[0]
    
    # Calculate Reorder Level and Restock Quantity
    daily_sales = next_month_sales / 30  # Approximate daily sales
    reorder_level = daily_sales * lead_time
    restock_quantity = 2 * reorder_level
    
    forecast_results.append({
        'ProductCategory': category,
        'Predicted Sales': round(next_month_sales, 2),
        'Reorder Level': round(reorder_level, 2),
        'Restock Quantity': round(restock_quantity, 2)
    })

## Save Forecast Results
forecast_df = pd.DataFrame(forecast_results)
forecast_df.to_csv("predicted_stock_levels.csv", index=False)

print("Forecasting completed! Check predicted_stock_levels.csv for results.")
