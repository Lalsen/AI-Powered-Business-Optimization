import pandas as pd
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

def run_rfm_segmentation(input_path='cleaned_sales_data.csv',
                         output_csv='customer_segments.csv',
                         plot_path='static/rfm_plot.png'):
    # Load the dataset
    df = pd.read_csv(input_path)
    df['Date'] = pd.to_datetime(df['Date'])
    current_date = df['Date'].max()

    # Calculate RFM
    rfm = df.groupby('FullName').agg({
        'Date': lambda x: (current_date - x.max()).days,
        'FullName': 'count',
        'Total': 'sum'
    })
    rfm.columns = ['Recency', 'Frequency', 'Monetary']

    # Normalize
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm)

    # K-Means clustering
    kmeans = KMeans(n_clusters=4, random_state=42)
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

    # Save CSV
    rfm.reset_index(inplace=True)
    rfm.to_csv(output_csv, index=False)

    # Plot
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=rfm, x='Recency', y='Monetary', hue='Cluster', palette='viridis')
    plt.title('Customer Segments by K-Means Clustering')
    plt.xlabel('Recency (days)')
    plt.ylabel('Monetary Value')
    plt.savefig(plot_path)
    plt.close()

    return rfm  # Return dataframe to show in frontend
