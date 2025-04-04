import pandas as pd

# Load the dataset

df = pd.read_csv("supermarket_sales.csv")

# Select the required columns
columns_to_extract = [
    'Invoice ID', 'Customer type', 'Gender', 'Product line', 'Unit price', 
    'Quantity', 'Total', 'Date', 'Time', 'Payment', 'Rating'
]
df_selected = df[columns_to_extract]

# Save to a new CSV file

df_selected.to_csv("extracted_supermarket_sales.csv", index=False)

print(f"Extracted data saved to extracted_supermarket_sales.csv")
