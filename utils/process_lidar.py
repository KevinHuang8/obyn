from utils import read_data
import numpy as np
from . import constants

def remove_excess(points, amount):
    '''
    points - nx5 point cloud array in format of [x, y, z, intensity, id]
    
    Returns a copy of points that removes 'amount' number of points which have 
    an id of 0.
    '''
    zero_indices = np.where(points[:, 4] == 0)[0]
    if amount > len(zero_indices):
        raise ValueError(f'Not enough 0 id points to remove. {amount} vs {len(zero_indices)}')
    remove_indices = np.random.choice(zero_indices, size=(amount, ), replace=False)
    return np.delete(points, remove_indices, axis=0)

def pad_zero(points, amount, intensity=0):
    '''
    Adds extra points to the the point cloud 'points' with a id of 0.
    
    These extra points are uniformly distributed at z = 0 and span the
    extent of the x and y dimensions, and have an intensity of 'intensity'.
    '''    
    min_x = points[:, 0].min()
    max_x = points[:, 0].max()
    min_y = points[:, 1].min()
    max_y = points[:, 1].max()
            
    while amount >= 1:
        n = int(np.sqrt(amount))
        
        x = np.linspace(min_x, max_x, n)
        y = np.linspace(min_y, max_y, n)
        
        xx, yy = np.meshgrid(x, y)
    
        zeros = np.zeros((n**2,))
        intensities = np.zeros((n**2,)) + intensity

        extra_points = np.c_[xx.flatten().T, yy.flatten().T, zeros, intensities, zeros]
        
        points = np.r_[points, extra_points]
        
        amount = amount - n**2
        
    return points

def downsample(points, target):
    '''
    Downsample point cloud 'points' until the number of points
    is equal to target.
    
    Downsample method is performed by removing points at random,
    but prioritize points below the median z value (as the canopy contains
    the most information.)
    '''
    z_med = np.median(points[:, 2])
    
    n = points.shape[0]
    m = len(points[points[:, 2] < z_med])
    lower_ind = np.where(points[:, 2] < z_med)[0]
    upper_ind = np.where(points[:, 2] >= z_med)[0]
    
    to_remove = n - target
    
    # if can remove less than half of points from bottom,
    # take only from bottom
    if to_remove < m / 2:
        keep = np.random.choice(lower_ind, size=(m - to_remove, ), replace=False)
        new_points = points[keep, :]
        return new_points
    
    # Otherwise, remove 2/3 of points from bottom, 1/3 from top if possible
    to_remove_b = int(to_remove * (2/3))
    
    if to_remove_b > m:
        keep = np.random.choice(np.arange(n), size=(n - to_remove, ), replace=False)
        return points[keep, :]
    
    to_remove_t = to_remove - to_remove_b
    
    keepb = np.random.choice(lower_ind, size=(m - to_remove_b, ), replace=False)
    keept = np.random.choice(upper_ind, size=((n - m) - to_remove_t, ), replace=False)
    return np.r_[points[keepb], points[keept]]

def sync_size(lidar_data, n=1024):
    '''
    Given a list of lidar point clouds 'lidar_data', return a numpy array
    of lidar point clouds such that all the clouds have the same number of 
    points.

    n - number of points to use
    '''
    lens = []
    for img in lidar_data:
        n_zero = len(img[img[:, 4] == 0])
        nonzero = len(img) - n_zero
        lens.append(nonzero)
        
    lens = np.array(lens)
    ordered_ind = np.argsort(lens)[::-1]

    new_points = []
    for i in ordered_ind:
        nonzero_pts = lidar_data[i][lidar_data[i][:, 4] != 0]
        zero_pts = lidar_data[i][lidar_data[i][:, 4] == 0]
        
        if len(nonzero_pts) > n:
            nonzero_pts = downsample(nonzero_pts, n)
            old_pts = np.r_[nonzero_pts, zero_pts]
        else:
            old_pts = lidar_data[i]
        
        amount = old_pts.shape[0] - n
        if amount > 0:
            points = remove_excess(old_pts, amount)
        elif amount < 0:
            points = pad_zero(old_pts, -amount)
        else:
            points = old_pts
        new_points.append(points)

    return np.array(new_points)

def quad_points(lidar_data, threshold=50):
    '''
    Divide each point cloud into quadrants.

    Discard quadrants that have less non-zero (tree) points than 
    'threshold'.
    '''

    smaller_points = []
    for points in lidar_data:
        min_x = points[:, 0].min()
        max_x = points[:, 0].max()
        avg_x = (min_x + max_x) / 2
        min_y = points[:, 1].min()
        max_y = points[:, 1].max()
        avg_y = (min_y + max_y) / 2
            
        s = points[(points[:, 0] < avg_x) & (points[:, 0] < avg_y)]
        if len(s[s[:, 4] != 0]) >= threshold:
            smaller_points.append(s)
        s = points[(points[:, 0] < avg_x) & (points[:, 0] >= avg_y)]
        if len(s[s[:, 4] != 0]) >= threshold:
            smaller_points.append(s)
        s = points[(points[:, 0] >= avg_x) & (points[:, 0] < avg_y)]
        if len(s[s[:, 4] != 0]) >= threshold:
            smaller_points.append(s)
        s = points[(points[:, 0] >= avg_x) & (points[:, 0] >= avg_y)]
        if len(s[s[:, 4] != 0]) >= threshold:
            smaller_points.append(s)
    return smaller_points

def create_group_matrix(data, n=1024):
    '''
    Creates the nxn ground truth matrix G, where G_ij = 1 if points i and j
    belong to the same tree, and G_ij = 0 otherwise. 
    '''
    labels = []
    for points in data:
        mat = np.zeros((n, n))
        ids = np.unique(points[:, 4])
        for unique_id in ids:
            id_indices = np.where(points[:, 4] == unique_id)[0]
            i = np.array(np.meshgrid(id_indices, id_indices)).reshape(2, -1)
            mat[i[0], i[1]] = 1
        labels.append(mat)
    labels = np.array(labels)
    return labels

def process_lidar(lidar_data, split=True, n=None, threshold=None):
    '''
    Fully calls the pipeline for loading the lidar data.

    n - number of points per sample. Processes each point cloud so that each
    cloud has n points.
    threshold - any lidar images with less than this number of non background
    points are discarded
    split - whether to split the image into quadrants to decrease the # of
    points

    returns x, y - x is the point cloud, y is the gt labels
        x - shape (num samples, n, 3)
        y - shape (num samples, n, n)
    '''
    if n is None:
        n = constants.NUM_POINTS

    if threshold is None:
        threshold = constants.NONZERO_POINT_THRESHOLD

    # Whether to split lidar into quadrants (do it for neon)
    if split:
        lidar_data = quad_points(lidar_data, threshold)
    synced = sync_size(lidar_data, n)
    labels = create_group_matrix(synced, n)
    return synced[:, :, :3], labels