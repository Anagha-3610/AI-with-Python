from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_wine
from sklearn.cluster import KMeans

#Iris data without labels
X=load_iris().data
#print(X)

#Train KMeans Clustering model
kmeans=KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)
#print(kmeans.labels_)

#Plot the Clusters
plt.scatter(X[:,0],X[:,1],c=kmeans.labels_, cmap='viridis')
plt.title("Iris Clustering (unsupervised)")
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.show()

# Load the wine dataset
X = load_wine().data


# Apply KMeans clustering
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(X)
print(kmeans.labels_)
# Plot the clusters using the first two features
plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='viridis')
plt.title("Wine Clusters (Unsupervised)")
plt.xlabel("Alcohol")
plt.ylabel("Malic Acid")
plt.show()