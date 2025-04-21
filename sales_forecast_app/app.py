from flask import Flask, render_template, jsonify, request, redirect, flash
from dotenv import load_dotenv
import mysql.connector
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import os
from segmentation import run_rfm_segmentation

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY')

# -------------------- Login page --------------------
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        # Dummy check (replace with your DB logic)
        if username == 'username' and password == 'password':
            return redirect('/home')  # Redirect to home after login
        else:
            return "<h3 style='text-align:center; color:red;'>Invalid credentials</h3>"
    return render_template('login.html')

# -------------------- Home Route --------------------
@app.route('/home')
def home():
    return render_template('home.html')

# -------------------- Default Route (Redirect to Login) --------------------
@app.route('/')
def index():
    return redirect('/login')  # Default route now redirects to login page

# -------------------- Inventory Forecast Page --------------------
@app.route('/inventory')
def inventory():
    return render_template('index.html')

# -------------------- Inventory Forecast Logic --------------------
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

# -------------------- ARIMA Forecast Page --------------------
@app.route('/forecast')
def forecast():
    return render_template('forecast.html')

# -------------------- ARIMA Forecast Logic with Graph --------------------
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

# -------------------- MySQL DB Configuration --------------------
db = mysql.connector.connect(
    host=os.getenv("DB_HOST"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    database=os.getenv("DB_NAME")
)

cursor = db.cursor()

# -------------------- Customer Form Page --------------------
@app.route('/customer_form')
def customer_form():
    return render_template('customer_form.html')

# -------------------- Submit Sale Logic --------------------
@app.route('/submit_sale', methods=['POST'])
def submit_sale():
    if request.method == 'POST':
        customer_type = request.form['customer_type']
        product_category = request.form['product_category']
        unit_price = float(request.form['unit_price'])
        quantity = int(request.form['quantity'])
        total = float(request.form['total'])
        full_name = request.form['full_name']

        insert_query = """
            INSERT INTO CustomerSales (customer_type, product_category, unit_price, quantity, total, full_name)
            VALUES (%s, %s, %s, %s, %s, %s)
        """
        values = (customer_type, product_category, unit_price, quantity, total, full_name)

        try:
            cursor.execute(insert_query, values)
            db.commit()
            flash("Sale successfully recorded!", "success")
        except Exception as e:
            db.rollback()
            flash(f"Error: {e}", "danger")

        return redirect('/customer_form')

# -------------------- Customer Segmentation Page --------------------
@app.route('/segmentationPage', methods=['GET', 'POST'])
def segmentation_page():
    show_result = False
    table_data = []
    
    if request.method == 'POST':
        rfm = run_rfm_segmentation()
        table_data = rfm.to_dict(orient='records')
        show_result = True

    return render_template('segmentationPage.html', show_result=show_result, table_data=table_data)

# -------------------- Run the App --------------------
if __name__ == '__main__':
    app.run(debug=True)