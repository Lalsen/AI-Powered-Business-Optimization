import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Load data (Assume we have sales data in a CSV)
df = pd.read_csv("sales_data.csv")

# Convert sale_date to datetime and extract month/year
df['sale_date'] = pd.to_datetime(df['sale_date'])
df['month'] = df['sale_date'].dt.month
df['year'] = df['sale_date'].dt.year

# Selecting features (independent variables)
X = df[['month', 'year', 'product_id', 'quantity_sold']]
y = df['quantity_sold'].shift(-1)  # Predicting next month's sales

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict future sales
predictions = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, predictions)
print(f"Mean Absolute Error: {mae}")
