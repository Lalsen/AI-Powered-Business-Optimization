# Inventory Management System

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import itertools
import warnings
warnings.filterwarnings("ignore")

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

        # Parameters for inventory management
        lead_time = 5  # days
        service_level = 0.95  # 95% service level
        z_score = 1.645  # Z-score for 95% service level
        
        forecast_results = []

        for category in monthly_sales['ProductCategory'].unique():
            category_data = monthly_sales[monthly_sales['ProductCategory'] == category]
            category_data = category_data[['Date', 'Quantity']].set_index('Date')

            if len(category_data) < 12:  # Need at least 12 data points
                forecast_results.append({
                    'ProductCategory': category,
                    'Error': "Not enough data for forecasting"
                })
                continue

            # Split data for validation
            train_size = int(len(category_data) * 0.8)
            train = category_data.iloc[:train_size]
            test = category_data.iloc[train_size:]
            
            # Try both ARIMA and Exponential Smoothing models
            best_model = None
            best_mape = float('inf')
            best_forecast = None
            best_model_name = None
            
            # 1. Try ARIMA with auto-parameter selection
            try:
                # Find best ARIMA parameters
                best_aic = float('inf')
                best_order = None
                
                for p, d, q in itertools.product(range(0, 3), range(0, 2), range(0, 3)):
                    try:
                        model = ARIMA(train, order=(p, d, q))
                        model_fit = model.fit()
                        if model_fit.aic < best_aic:
                            best_aic = model_fit.aic
                            best_order = (p, d, q)
                    except:
                        continue
                
                if best_order:
                    model = ARIMA(train, order=best_order)
                    model_fit = model.fit()
                    forecast = model_fit.forecast(steps=len(test))
                    
                    # Calculate MAPE
                    mape = np.mean(np.abs((test['Quantity'] - forecast) / test['Quantity'])) * 100
                    
                    if mape < best_mape:
                        best_mape = mape
                        best_model = model_fit
                        best_forecast = forecast
                        best_model_name = f"ARIMA{best_order}"
            except Exception as e:
                print(f"ARIMA failed for {category}: {str(e)}")
            
            # 2. Try Exponential Smoothing
            try:
                model = ExponentialSmoothing(train, 
                                            seasonal_periods=12,
                                            trend='add', 
                                            seasonal='add', 
                                            use_boxcox=True)
                model_fit = model.fit()
                forecast = model_fit.forecast(steps=len(test))
                
                # Calculate MAPE
                mape = np.mean(np.abs((test['Quantity'] - forecast) / test['Quantity'])) * 100
                
                if mape < best_mape:
                    best_mape = mape
                    best_model = model_fit
                    best_forecast = forecast
                    best_model_name = "Exponential Smoothing"
            except Exception as e:
                print(f"Exponential Smoothing failed for {category}: {str(e)}")
            
            # If no model worked, use simple moving average
            if best_model is None:
                avg_sales = train['Quantity'].mean()
                next_month_sales = avg_sales
                model_name = "Moving Average"
                mape = "N/A"
                # Calculate standard deviation for safety stock
                std_dev = train['Quantity'].std()
            else:
                # Retrain on full data for prediction
                if best_model_name.startswith("ARIMA"):
                    order = eval(best_model_name.replace("ARIMA", ""))
                    full_model = ARIMA(category_data, order=order)
                    full_model_fit = full_model.fit()
                else:
                    full_model = ExponentialSmoothing(category_data, 
                                                     seasonal_periods=12,
                                                     trend='add', 
                                                     seasonal='add', 
                                                     use_boxcox=True)
                    full_model_fit = full_model.fit()
                
                # Forecast next month
                next_month_sales = full_model_fit.forecast(steps=1)[0]
                model_name = best_model_name
                
                # Calculate standard deviation of forecast errors for safety stock
                if len(test) > 0:
                    errors = test['Quantity'] - best_forecast
                    std_dev = errors.std()
                else:
                    std_dev = category_data['Quantity'].std() * 0.5  # Conservative estimate
            
            # Calculate inventory metrics
                daily_sales = next_month_sales / 30
            
            # Safety stock calculation using service level
            safety_stock = z_score * std_dev * np.sqrt(lead_time)
            
            # Reorder level includes safety stock
            reorder_level = (daily_sales * lead_time) + safety_stock
            
            # Economic Order Quantity calculation (simplified)
            holding_cost_ratio = 0.25  # 25% of item cost per year
            ordering_cost = 20  # Fixed cost per order
            annual_demand = daily_sales * 365
            
            try:
                eoq = np.sqrt((2 * annual_demand * ordering_cost) / (holding_cost_ratio * daily_sales))
                restock_quantity = max(eoq, 2 * reorder_level)  # Use EOQ or min 2x reorder level
            except:
                restock_quantity = 2 * reorder_level  # Fallback if EOQ calculation fails

                forecast_results.append({
                    'ProductCategory': category,
                'Model': model_name,
                'MAPE': round(best_mape, 2) if best_mape != float('inf') else "N/A",
                'Predicted Monthly Sales': round(next_month_sales, 2),
                'Daily Sales': round(daily_sales, 2),
                'Safety Stock': round(safety_stock, 2),
                    'Reorder Level': round(reorder_level, 2),
                    'Restock Quantity': round(restock_quantity, 2)
                })

        forecast_df = pd.DataFrame(forecast_results)
        forecast_df.to_csv("predicted_stock_levels.csv", index=False)
        return forecast_df

    except Exception as main_error:
        print(f"Forecasting failed: {main_error}")
        return pd.DataFrame([{'Error': str(main_error)}])
