#sales forecast

# forecast.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
import itertools
import warnings
warnings.filterwarnings("ignore")

def run_arima_forecast():
    # Load data
    df = pd.read_csv("cleaned_sales_data.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Aggregate to monthly
    monthly_sales = df.groupby(pd.Grouper(key='Date', freq='M'))['Total'].sum().reset_index()
    monthly_sales.set_index('Date', inplace=True)

    # Split data for training and testing
    train_size = int(len(monthly_sales) * 0.8)
    train, test = monthly_sales.iloc[:train_size], monthly_sales.iloc[train_size:]

    # Simple ARIMA model with fixed parameters
    model = ARIMA(train, order=(5, 1, 0))
    model_fit = model.fit()
    
    # Forecast test period
    forecast = model_fit.forecast(steps=len(test))
    
    # Calculate MSE
    mse = mean_squared_error(test, forecast)
    
    # Forecast next 3 months
    full_model = ARIMA(monthly_sales, order=(5, 1, 0))
    full_model_fit = full_model.fit()
    future_forecast = full_model_fit.forecast(steps=3)
    
    # Create future dates for plotting
    last_date = monthly_sales.index[-1]
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=3, freq='MS')
    
    # Plot results
    plt.figure(figsize=(12, 6))
    
    # Format dates on x-axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    
    # Plot actual values
    plt.plot(monthly_sales.index, monthly_sales, label="Actual Sales", color="blue", linewidth=2)
    
    # Plot test forecast
    plt.plot(test.index, forecast, label="Test Forecast", color="red", linestyle="dashed", linewidth=2)
    
    # Plot future forecast
    plt.plot(future_dates, future_forecast, label="Future Forecast", color="green", linewidth=2, marker='o')
    
    # Add points to show exact forecast values
    for i, date in enumerate(future_dates):
        plt.annotate(f'{future_forecast[i]:.0f}', 
                    (date, future_forecast[i]), 
                    textcoords="offset points",
                    xytext=(0,10), 
                    ha='center')
    
    plt.xlabel("Date")
    plt.ylabel("Total Sales")
    plt.title(f"Sales Forecast - MSE: {mse:.2f}")
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.legend(loc='best')
    plt.tight_layout()
    
    plot_path = "static/forecast_plot.png"
    plt.savefig(plot_path)
    plt.close()

    # Return only the essential data
    forecast_result = {
        "next_month_forecast": round(future_forecast[0], 2),
        "plot_path": plot_path,
        "mse": round(mse, 2)
    }
    
    return forecast_result

def run_arima_sarima(series, train, test, p_auto, d, q_auto, has_seasonality):
    try:
        # Auto-select ARIMA parameters with cross-validation
        best_rmse = float('inf')
        best_order = None
        best_seasonal_order = None
        
        # Consider regular and seasonal ARIMA models
        p_values = [1, 2, p_auto] if p_auto > 2 else [1, 2]
        d_values = [d]
        q_values = [1, 2, q_auto] if q_auto > 2 else [1, 2]
        
        # Use cross-validation
        tscv = TimeSeriesSplit(n_splits=3)
        
        for p, d, q in itertools.product(p_values, d_values, q_values):
            # Try both regular ARIMA and seasonal ARIMA if seasonality detected
            models_to_try = [(p, d, q, None)]
            if has_seasonality:
                models_to_try.append((p, d, q, (1, 0, 1, 12)))
                models_to_try.append((p, d, q, (0, 1, 1, 12)))
                
            for order_params in models_to_try:
                p, d, q, seasonal_order = order_params
                
                try:
                    cv_errors = []
                    
                    # Cross-validation
                    for train_idx, test_idx in tscv.split(train):
                        cv_train = train.iloc[train_idx]
                        cv_test = train.iloc[test_idx]
                        
                        # Skip if too little data
                        if len(cv_train) <= max(p, q) + d + 4:
                            continue
                            
                        # Fit model
                        if seasonal_order:
                            model = SARIMAX(cv_train, order=(p, d, q), 
                                         seasonal_order=seasonal_order,
                                         enforce_stationarity=False,
                                         enforce_invertibility=False)
                        else:
                            model = ARIMA(cv_train, order=(p, d, q))
                            
                        model_fit = model.fit(disp=False)
                        forecast = model_fit.forecast(steps=len(cv_test))
                        
                        # Calculate error
                        mse = mean_squared_error(cv_test, forecast)
                        cv_errors.append(mse)
                    
                    # Average CV error
                    if cv_errors:
                        avg_rmse = np.sqrt(np.mean(cv_errors))
                        
                        if avg_rmse < best_rmse:
                            best_rmse = avg_rmse
                            best_order = (p, d, q)
                            best_seasonal_order = seasonal_order
                            
                except Exception as e:
                    continue
        
        # Train final model
        if best_seasonal_order:
            model = SARIMAX(train, order=best_order, 
                           seasonal_order=best_seasonal_order,
                           enforce_stationarity=False,
                           enforce_invertibility=False)
            model_name = f"SARIMA{best_order}x{best_seasonal_order}"
        else:
            model = ARIMA(train, order=best_order)
            model_name = f"ARIMA{best_order}"
            
        model_fit = model.fit(disp=False)
        
        # Forecast test period
        forecast_test = model_fit.forecast(steps=len(test))
        
        # Evaluate
        mae = mean_absolute_error(test, forecast_test)
        
        # Train model on full data for final forecast
        if best_seasonal_order:
            full_model = SARIMAX(series, order=best_order, 
                               seasonal_order=best_seasonal_order,
                               enforce_stationarity=False,
                               enforce_invertibility=False)
        else:
            full_model = ARIMA(series, order=best_order)
            
        full_model_fit = full_model.fit(disp=False)
        
        # Forecast next 3 months
        future_forecast = full_model_fit.forecast(steps=3)
        
        # Combine test and future forecasts
        full_forecast = np.concatenate([forecast_test, future_forecast])
        
        return {
            'forecast': full_forecast,
            'future_forecast': future_forecast,
            'mae': mae,
            'params': {
                'order': best_order,
                'seasonal_order': best_seasonal_order,
                'model': model_name
            }
        }
    except Exception as e:
        print(f"ARIMA/SARIMA failed: {str(e)}")
        return None

def run_exponential_smoothing(series, train, test):
    try:
        best_mape = float('inf')
        best_model = None
        best_params = {}
        
        # Test different exponential smoothing models
        for trend in ['add', 'mul', None]:
            for seasonal in ['add', 'mul', None]:
                # Skip if seasonal without trend
                if seasonal and not trend:
                    continue
                
                try:
                    # Set seasonal periods
                    if seasonal:
                        periods = 12
                    else:
                        periods = None
                        
                    model = ExponentialSmoothing(
                        train,
                        trend=trend,
                        seasonal=seasonal,
                        seasonal_periods=periods,
                        damped=True if trend else False
                    )
                    model_fit = model.fit(optimized=True)
                    
                    # Forecast and evaluate
                    forecast_test = model_fit.forecast(len(test))
                    mape = np.mean(np.abs((test - forecast_test) / test)) * 100
                    
                    if mape < best_mape:
                        best_mape = mape
                        best_model = model_fit
                        best_params = {
                            'trend': trend,
                            'seasonal': seasonal,
                            'damped': True if trend else False,
                            'seasonal_periods': periods
                        }
                except:
                    continue
        
        if best_model is None:
            return None
        
        # Train best model on full data
        full_model = ExponentialSmoothing(
            series,
            trend=best_params['trend'],
            seasonal=best_params['seasonal'],
            seasonal_periods=best_params['seasonal_periods'],
            damped=best_params['damped']
        )
        full_model_fit = full_model.fit(optimized=True)
        
        # Get test forecast
        forecast_test = best_model.forecast(len(test))
        
        # Calculate metrics
        mae = mean_absolute_error(test, forecast_test)
        
        # Get future forecast
        future_forecast = full_model_fit.forecast(3)
        
        # Combine test and future forecasts
        full_forecast = np.concatenate([forecast_test, future_forecast])
        
        # Create param description
        trend_str = best_params['trend'] if best_params['trend'] else 'None'
        seasonal_str = best_params['seasonal'] if best_params['seasonal'] else 'None'
        model_name = f"ETS(trend={trend_str}, seasonal={seasonal_str}, damped={best_params['damped']})"
        
        return {
            'forecast': full_forecast,
            'future_forecast': future_forecast,
            'mae': mae,
            'params': {
                'model': model_name,
                'details': best_params
            }
        }
    except Exception as e:
        print(f"Exponential Smoothing failed: {str(e)}")
        return None

def run_ml_models(df, train, test, features, target):
    try:
        # Prepare data
        X_train = train[features]
        y_train = train[target]
        X_test = test[features]
        y_test = test[target]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Prepare for future forecast
        last_obs = df.iloc[-1:][features]
        
        # Create future data points (3 months ahead)
        future_data = []
        for i in range(1, 4):
            # Copy last observation and modify date-related features
            new_obs = last_obs.copy()
            
            # Update month
            current_month = last_obs['month'].iloc[0]
            new_month = ((current_month + i - 1) % 12) + 1
            new_obs['month'] = new_month
            
            # Update monthly dummies
            for m in range(1, 13):
                new_obs[f'month_{m}'] = 1 if m == new_month else 0
                
            # Increment trend
            new_obs['trend'] = last_obs['trend'].iloc[0] + i
            
            # For lag features, use predictions or latest actual values
            for lag in range(1, 4):
                if i >= lag:
                    # This will be updated with predictions later
                    new_obs[f'lag_{lag}'] = 0
                else:
                    # Use last known values
                    new_obs[f'lag_{lag}'] = df.iloc[-lag][target]
            
            future_data.append(new_obs)
        
        future_df = pd.concat(future_data).reset_index(drop=True)
        X_future = scaler.transform(future_df)
        
        # Models to try
        models = [
            ("RandomForest", RandomForestRegressor(n_estimators=100, random_state=42)),
            ("ElasticNet", ElasticNet(alpha=0.5, l1_ratio=0.5, random_state=42)),
        ]
        
        best_model = None
        best_mape = float('inf')
        best_name = ""
        
        # Train and evaluate models
        for name, model in models:
            try:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                
                mae = mean_absolute_error(y_test, y_pred)
                
                if mae < best_mape:
                    best_mape = mae
                    best_model = model
                    best_name = name
            except:
                continue
        
        if best_model is None:
            return None
            
        # Make future predictions
        # Update lag values for multi-step forecasting
        future_preds = []
        for i in range(3):
            # Update lag features with predictions
            if i > 0:
                for lag in range(1, min(i+1, 4)):
                    idx = future_df.columns.get_loc(f'lag_{lag}')
                    X_future[i, idx] = future_preds[i-lag]
            
            # Make prediction
            pred = best_model.predict([X_future[i]])[0]
            future_preds.append(pred)
            
            # Update rolling mean/std if present
            if 'rolling_mean_3' in future_df.columns:
                if i == 0:
                    rolling_vals = list(df[target].iloc[-2:]) + [pred]
                elif i == 1:
                    rolling_vals = [df[target].iloc[-1], future_preds[0], pred]
                else:
                    rolling_vals = future_preds[:i+1][-3:]
                
                if len(rolling_vals) == 3:
                    # Update rolling mean and std for next prediction
                    rm_idx = future_df.columns.get_loc('rolling_mean_3')
                    rs_idx = future_df.columns.get_loc('rolling_std_3')
                    
                    if i < 2:  # Only update for the next prediction
                        X_future[i+1, rm_idx] = np.mean(rolling_vals)
                        X_future[i+1, rs_idx] = np.std(rolling_vals)
        
        # Final predictions
        y_pred = best_model.predict(X_test_scaled)
        future_forecast = np.array(future_preds)
        
        # Concatenate test and future forecasts
        full_forecast = np.concatenate([y_pred, future_forecast])
        
        return {
            'forecast': full_forecast,
            'future_forecast': future_forecast,
            'mae': mean_absolute_error(y_test, y_pred),
            'params': {
                'model': best_name
            }
        }
    except Exception as e:
        print(f"ML models failed: {str(e)}")
        return None