from flask import Flask, render_template, jsonify
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import os

app = Flask(__name__)

# Route: Home
@app.route('/')
def home():
    return render_template('home.html')

# Route: Inventory Forecast (Multi-category)
@app.route('/inventory')
def inventory():
    return render_template('index.html')

# Route: Inventory Forecast Logic (Multi-category)
@app.route('/predict')
def predict():
    try:
        df = pd.read_csv("cleaned_sales_data.csv")
        df['Date'] = pd.to_datetime(df['Date'])
        df['YearMonth'] = df['Date'].dt.to_period('M')
        monthly_sales = df.groupby(['YearMonth', 'ProductCategory'])['Quantity'].sum().reset_index()
        monthly_sales['YearMonth'] = monthly_sales['YearMonth'].astype(str)
        monthly_sales['Date'] = pd.to_datetime(monthly_sales['YearMonth'])

        forecast_results = []
        lead_time = 5

        for category in monthly_sales['ProductCategory'].unique():
            category_data = monthly_sales[monthly_sales['ProductCategory'] == category]
            category_data = category_data[['Date', 'Quantity']].set_index('Date')
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

        forecast_df = pd.DataFrame(forecast_results)
        forecast_df.to_csv("predicted_stock_levels.csv", index=False)

        return jsonify(forecast_results)

    except Exception as e:
        print("Forecasting failed:", str(e))
        return jsonify({"error": str(e)})

# Route: ARIMA Forecast Page
@app.route('/forecast')
def forecast():
    return render_template('forecast.html')

# Route: ARIMA Forecast Logic with Graph
@app.route('/run_forecast')
def run_forecast():
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

    model_arima = ARIMA(train['Total'], order=(5, 1, 0))
    model_arima_fit = model_arima.fit()
    y_pred_arima = model_arima_fit.forecast(steps=len(test))
    next_month_arima = model_arima_fit.forecast(steps=1)[0]

    # Plot and save graph
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

    plot_path = os.path.join('static', 'forecast_plot.png')
    plt.savefig(plot_path)
    plt.close()

    return render_template('forecast.html', prediction=str(round(next_month_arima, 2)), plot_url=plot_path)


# Run the app
if __name__ == '__main__':
    app.run(debug=True)
