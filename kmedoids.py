import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances_argmin_min

data = pd.read_csv('Height Weight Data.csv')

X = data[['Height(Inches)', 'Weight(Pounds)']]

k = 3
def kmedoids(X, k, max_iter=100):
    init_medoids = np.random.choice(len(X), k, replace=False)
    medoids = X[init_medoids]

    for _ in range(max_iter):
        labels, _ = pairwise_distances_argmin_min(X, medoids, metric='euclidean')

        new_medoids = np.array([X[labels == i].mean(axis=0) for i in range(k)])

        if np.all(medoids == new_medoids):
            break
        medoids = new_medoids

    return medoids, labels

medoids, labels = kmedoids(X.values, k)

data['Cluster'] = labels

plt.figure(figsize=(8, 6))
colors = ['yellow', 'red', 'blue']
for i in range(k):
    cluster_data = data[data['Cluster'] == i]
    plt.scatter(cluster_data['Height(Inches)'], cluster_data['Weight(Pounds)'], c=colors[i], label=f'Cluster {i}')
plt.scatter(medoids[:, 0], medoids[:, 1], s=100, color='black', marker='x', label='Medoids')
plt.xlabel('Height (Inches)')
plt.ylabel('Weight (Pounds)')
plt.title('K-medoids Clustering of Height and Weight Data')
plt.legend()
plt.show()