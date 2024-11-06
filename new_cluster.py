import torch
import numpy as np
import scipy.spatial
from torch import nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from PIL import Image
import os

def new_dimension_clustering(data_dir, batch_size=32, thres=0.07, min_clus=5, max_dist=2.0, normalize=True, device='cpu'):
    
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = models.alexnet(pretrained=True).features
            self.pool = nn.AdaptiveAvgPool2d((1, 1))  

        def forward(self, x):
            x = self.features(x)
            x = self.pool(x)
            x = x.view(x.size(0), -1)  
            return x

    class ImageDataset(torch.utils.data.Dataset):
        def __init__(self, image_dir, transform=None):
            self.image_dir = image_dir
            self.image_paths = sorted(os.listdir(image_dir))
            self.transform = transform

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            img_path = os.path.join(self.image_dir, self.image_paths[idx])
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = ImageDataset(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model = TestModel().to(device)
    model.eval()

    features = []
    with torch.no_grad():
        for inputs in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs).cpu().numpy()  
            features.append(outputs)

    features = np.vstack(features)  # Stack all batches together

    # Step 1: Obtain initial clusters using Euclidean distances
    initial_labels = initial_clustering(features, min_clus)
    print("Initial clusters obtained using Euclidean distances")

    # Step 2: FLD-inspired refinement on features
    refined_features = fld_refinement(features, initial_labels)
    print("FLD-inspired feature refinement applied")

    return cluster(refined_features, thres=thres, min_clus=min_clus, max_dist=max_dist, normalize=normalize)

# Perform initial clustering with Euclidean distances
def initial_clustering(features, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    labels = kmeans.fit_predict(features)
    return labels

def fld_refinement(features, labels):
    refined_features = features.copy()
    unique_labels = np.unique(labels)

    for label in unique_labels:
        cluster_points = features[labels == label]
        centroid = np.mean(cluster_points, axis=0)
        variance = np.var(cluster_points, axis=0)

        for i in range(len(cluster_points)):
            refined_features[labels == label][i] += 0.05 * (centroid - refined_features[labels == label][i])  # Intra-cluster tightening

    overall_centroid = np.mean(features, axis=0)
    for label in unique_labels:
        cluster_points = refined_features[labels == label]
        cluster_centroid = np.mean(cluster_points, axis=0)
        
        refined_features[labels == label] += 0.05 * (cluster_centroid - overall_centroid)  # Inter-cluster separation

    return refined_features

# Modified clustering function with FLD-inspired refined features
def cluster(features, thres=0.07, min_clus=5, max_dist=2.0, normalize=True):
    features = np.array(features)

    if normalize:
        feat_norms = np.linalg.norm(features, axis=1, keepdims=True)
        feat_norms[feat_norms == 0] = 1
        features /= feat_norms
        print("Normalized features")

    # Computing pairwise distances with the refined features
    pair_dist = scipy.spatial.distance.pdist(features, 'sqeuclidean')
    pair_dist = scipy.spatial.distance.squareform(pair_dist)
    print("Computed pair distances with FLD-inspired adjustment")

    # Initializing affinity matrix for second-order clustering
    affinity_matrix = compute_affinity_matrix(pair_dist)
    print("Computed affinity matrix")

    # Clustering using the second-order distance
    clusters, sample_clusters = distribution_clustering(affinity_matrix, thres, min_clus, max_dist)
    print("Computed distribution clustering")

    return clusters, sample_clusters

def compute_affinity_matrix(pair_dist):
    diffs = pair_dist[:, np.newaxis, :] - pair_dist[np.newaxis, :, :]
    affinity_matrix = np.mean(diffs ** 2, axis=2)
    return affinity_matrix

def distribution_clustering(affinity_matrix, thres, min_clus, max_dist):

    n = affinity_matrix.shape[0]
    sample_clusters = np.zeros(n, dtype=int) 
    cur_cluster = 1 

    for i in range(n):
        if sample_clusters[i] == 0:  # Unclustered point
            candidate_cluster = [i]
            for j in range(n):
                if i != j and sample_clusters[j] == 0:  # Check unclustered points
                    if affinity_matrix[i, j] < thres:  # Use threshold to form a cluster
                        candidate_cluster.append(j)

            # Only form a cluster if it meets the minimum cluster size requirement
            if len(candidate_cluster) >= min_clus:
                for idx in candidate_cluster:
                    sample_clusters[idx] = cur_cluster
                cur_cluster += 1

    # Handling unclustered points by marking them as outliers or putting them in single-point clusters
    for i in range(n):
        if sample_clusters[i] == 0:  # If the point is still unclustered
            sample_clusters[i] = cur_cluster  # Assign it a new cluster (outlier)
            cur_cluster += 1

    # Determining the cluster centers by averaging the affinity matrix rows corresponding to each cluster
    unique_clusters = np.unique(sample_clusters)
    clusters = [np.mean(affinity_matrix[sample_clusters == c], axis=0) for c in unique_clusters if c > 0]

    return clusters, sample_clusters.tolist()
