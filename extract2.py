import mysql.connector
import pandas as pd

# Database connection
conn = mysql.connector.connect(
    host='localhost',      # Change this to your MySQL host
    user='root',      # Change this to your MySQL username
    password='Lalsen@123',  # Change this to your MySQL password
    database='business_db'   # Change this to your database name
)
cursor = conn.cursor()

# Query to extract required columns
query = """
SELECT 
    s.sale_date, s.product_id, s.quantity_sold, s.total_price, 
    i.product_name, i.category, i.stock_quantity, i.unit_price, 
    c.customer_id, c.name, c.created_at 
FROM Sales s
JOIN Inventory i ON s.product_id = i.product_id
JOIN Customers c ON s.customer_id = c.customer_id
"""

# Fetch data
df = pd.read_sql(query, conn)

# Save to CSV
df.to_csv('sales_data.csv', index=False)
print("Data extracted and saved to sales_data.csv")

# Close connection
cursor.close()
conn.close()
