# Inventory Management System


import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
# Prophet is imported but not used in this version — remove if not needed

def run_forecast():
    try:
        # Load dataset
        file_path = "cleaned_sales_data.csv"
        df = pd.read_csv(file_path)

        # Convert Date column to datetime
        df['Date'] = pd.to_datetime(df['Date'])

        # Aggregate monthly sales per product category
        df['YearMonth'] = df['Date'].dt.to_period('M')
        monthly_sales = df.groupby(['YearMonth', 'ProductCategory'])['Quantity'].sum().reset_index()
        monthly_sales['YearMonth'] = monthly_sales['YearMonth'].astype(str)
        monthly_sales['Date'] = pd.to_datetime(monthly_sales['YearMonth'])

        # Forecasting setup
        forecast_results = []
        lead_time = 5  # days

        for category in monthly_sales['ProductCategory'].unique():
            category_data = monthly_sales[monthly_sales['ProductCategory'] == category]
            category_data = category_data[['Date', 'Quantity']].set_index('Date')

            try:
                model = ARIMA(category_data, order=(5, 1, 0))
                model_fit = model.fit()

                forecast = model_fit.forecast(steps=1)
                next_month_sales = forecast.iloc[0]

                daily_sales = next_month_sales / 30
                reorder_level = daily_sales * lead_time
                restock_quantity = 2 * reorder_level

                forecast_results.append({
                    'ProductCategory': category,
                    'Predicted Sales': round(next_month_sales, 2),
                    'Reorder Level': round(reorder_level, 2),
                    'Restock Quantity': round(restock_quantity, 2)
                })

            except Exception as e:
                forecast_results.append({
                    'ProductCategory': category,
                    'Error': f"Forecasting failed: {str(e)}"
                })

        forecast_df = pd.DataFrame(forecast_results)
        forecast_df.to_csv("predicted_stock_levels.csv", index=False)
        return forecast_df

    except Exception as main_error:
        print(f"Forecasting failed: {main_error}")
        return pd.DataFrame([{'Error': str(main_error)}])
