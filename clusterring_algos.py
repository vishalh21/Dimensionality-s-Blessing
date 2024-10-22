from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
import numpy as np
from sklearn.metrics import pairwise_distances_argmin_min

# K-Means Clustering
def kmeans_clustering(features, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters)
    labels = kmeans.fit_predict(features)
    return labels

# GMM Clustering (Gaussian Mixture Model)
def gmm_clustering(features, n_clusters):
    gmm = GaussianMixture(n_components=n_clusters)
    labels = gmm.fit_predict(features)
    return labels

# Spectral Clustering
def spectral_clustering(features, n_clusters):
    spectral = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors')
    labels = spectral.fit_predict(features)
    return labels

# proclus_clustering
def proclus_clustering(features, n_clusters, n_iterations=10):
    labels = np.zeros(features.shape[0], dtype=int)

    for _ in range(n_iterations):
        kmeans = KMeans(n_clusters=n_clusters)
        labels = kmeans.fit_predict(features)
        selected_features = select_features(features, labels, n_clusters)

        new_labels = np.zeros_like(labels)
        for cluster, features_indices in selected_features.items():
            if len(features_indices) > 0:
                cluster_features = features[:, features_indices]
                kmeans = KMeans(n_clusters=1) 
                sub_labels = kmeans.fit_predict(cluster_features)
                new_labels[labels == cluster] = sub_labels
        
        labels = new_labels
    return labels

def select_features(features, labels, n_clusters):
    """Select relevant features for each cluster."""
    selected_features = {}
    for cluster in range(n_clusters):
        cluster_points = features[labels == cluster]
        
        feature_variances = np.var(cluster_points, axis=0)
        
        important_features = np.where(feature_variances > np.mean(feature_variances))[0]
        selected_features[cluster] = important_features
        
    return selected_features