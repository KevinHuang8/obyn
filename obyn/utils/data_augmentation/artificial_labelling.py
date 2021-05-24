import numpy as np
from sklearn import cluster
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm
from .. import constants as C
from .. import process_lidar

def optimal_kmeans(points, labels, kmax):
    '''
    Perform k-means clustering on a point cloud. Chooses the number of 
    clusters k by randomly trying k=2...kmax and taking the k with the max
    silhouette score.

    points - shape (N, 3)
    labels - shape(N, ): only cluster points with nonzero label
    '''
    sil_scores = []
    prev_s = -999
    nonzero_idx = np.where(labels != 0)[0]
    points = points[nonzero_idx][:, :3]
    for k in range(2, kmax + 1):
        kmeans = KMeans(n_clusters=k).fit(points)
        labels = kmeans.labels_
        s = silhouette_score(points, labels, metric='euclidean')
        sil_scores.append(s)
        # Check for max greedily (otherwise it takes too long). I.e., once the
        # silhouette decreases, we stop
        if s < prev_s:
            opt_k = k
            break
        prev_s = s
    try:
        return KMeans(n_clusters=opt_k).fit(points), nonzero_idx
    except NameError:
        return KMeans(n_clusters=kmax).fit(points), nonzero_idx

def create_artificial_labels(lidar_pts, old_labels, kmax=50):
    '''
    Artificially create labels for lidar points through k means clustering.

    lidar_pts: Nx3 array of points
    old_labels: junk labels for lidar_pts. Only info this needs to contain 
    is whether a point is a background point (zero) or a tree point (nonzero).
    '''
    N = lidar_pts.shape[0]
    labels = []

    print('Creating artificial labels...')
    for i in tqdm(range(N)):
        label = old_labels[i]
        points = lidar_pts[i]

        new_labels = np.zeros(label.shape[0])
        
        kmeans, nonzero_idx = optimal_kmeans(points, label, kmax=kmax)
        cluster_labels = kmeans.labels_
        cluster_labels += 1
        new_labels[nonzero_idx] = cluster_labels

        labels.append(new_labels)

    return np.array(labels)
