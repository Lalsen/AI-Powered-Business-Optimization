import mysql.connector
import pandas as pd
import os

# Database connection
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "Lalsen@123",
    "database": "business_db"
}

DATA_DIR = "sales_data"
LATEST_FILE = os.path.join(DATA_DIR, "latest_sales.csv")

# Ensure the data directory exists
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

def fetch_sales_data():
    """Extracts total monthly sales per product from the database and updates the sales file."""
    try:
        # Connect to the database
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()

        # SQL Query to get total monthly sales per product for the last 6 months
        query = """
            SELECT 
                s.product_id AS pid, 
                i.product_name, 
                SUM(s.quantity_sold) AS total_sales,
                DATE_FORMAT(s.sale_date, '%Y-%m') AS sale_month
            FROM Sales s
            JOIN Inventory i ON s.product_id = i.product_id
            WHERE s.sale_date >= DATE_SUB(CURDATE(), INTERVAL 6 MONTH)  -- Fetch last 6 months of data
            GROUP BY s.product_id, sale_month
            ORDER BY s.product_id, sale_month;
        """

        # Execute the query
        cursor.execute(query)
        rows = cursor.fetchall()

        # Convert to DataFrame
        df = pd.DataFrame(rows, columns=["pid", "product_name", "total_sales", "sale_month"])
        
        # Pivot the data so each row represents a product with monthly sales as columns
        df_pivot = df.pivot(index=["pid", "product_name"], columns="sale_month", values="total_sales").fillna(0).reset_index()

        # Delete the previous outdated sales file if it exists
        if os.path.exists(LATEST_FILE):
            os.remove(LATEST_FILE)
            print(f"Deleted outdated file: {LATEST_FILE}")

        # Save the new file
        df_pivot.to_csv(LATEST_FILE, index=False)
        print(f"Data successfully saved to {LATEST_FILE}")

    except mysql.connector.Error as err:
        print(f"Error: {err}")
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    fetch_sales_data()
