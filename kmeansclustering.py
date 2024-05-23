import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

data = pd.read_csv('Height Weight Data.csv')

print(data.head())

X = data[['Height(Inches)', 'Weight(Pounds)']] 

k =3
kmeans = KMeans(n_clusters=k)
kmeans.fit(X)
labels = kmeans.labels_
data['Cluster'] = labels 

plt.scatter(data['Height(Inches)'], data['Weight(Pounds)'], c=data['Cluster'], cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', label='Centroids')
plt.xlabel('Height(Inches)')
plt.ylabel('Weight(Pounds)')
plt.title('K-Means Clustering')

plt.legend()
plt.show()
