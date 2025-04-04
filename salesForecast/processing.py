import pandas as pd

# Load the dataset
file_path = "cleaned_sales_data.csv"  # Replace with your actual file path
df = pd.read_csv(file_path)

# Convert Date column to datetime format
df['Date'] = pd.to_datetime(df['Date'])
df.dropna(inplace=True)  # Remove missing values
df = df.sort_values(by='Date')  # Ensure chronological order

# Aggregate sales by month
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
monthly_sales = df.groupby(['Year', 'Month'])['Total'].sum().reset_index()
monthly_sales['Date'] = pd.to_datetime(monthly_sales[['Year', 'Month']].assign(day=1))
monthly_sales.set_index('Date', inplace=True)
monthly_sales.drop(columns=['Year', 'Month'], inplace=True)

# Save processed data as CSV
processed_file_path = "processed_sales_data.csv"
monthly_sales.to_csv(processed_file_path)
print(f"Processed data saved as {processed_file_path}")
