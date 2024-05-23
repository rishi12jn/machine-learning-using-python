import pandas as pd
from sklearn_extra.cluster import KMedoids
import matplotlib.pyplot as plt

# Load data from CSV file
data = pd.read_csv('Height Weight Data.csv')  # Replace 'your_data.csv' with your actual file name

# Extract relevant features
X = data[['Height(Inches)', 'Weight(Pounds)']]

# Number of clusters
k = 3

# Initialize K-medoids
kmedoids = KMedoids(n_clusters=k, random_state=0)

# Fit the model
kmedoids.fit(X)

# Get cluster labels and medoids
labels = kmedoids.labels_
medoids = kmedoids.cluster_centers_
x
# Add cluster labels to the dataframe
data['Cluster'] = labels

# Plot clusters
plt.scatter(data['Height(Inches)'], data['Weight(Pounds)'], c=data['Cluster'], cmap='viridis')
plt.scatter(medoids[:, 0], medoids[:, 1], s=300, c='red', label='Medoids')
plt.xlabel('Height(Inches)')
plt.ylabel('Weight(Pounds)')
plt.title('K-Medoids Clustering')
plt.legend()
plt.show()
