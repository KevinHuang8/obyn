import numpy as np
from tqdm import tqdm
from .data_augmentation import rotation
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

def downsample_voxel_grid(points, voxel_size):
    '''
    Apply a voxel grid filter to a point cloud for downsampling.
    
    Taken from https://towardsdatascience.com/how-to-automate-lidar-point-cloud-processing-with-python-a027454a536c
    '''
    # See link for explanation for code - pretty crazy
    non_empty_voxel_keys, inverse, nb_pts_per_voxel= np.unique(((points[:, :3] - np.min(points[:, :3], axis=0)) // voxel_size).astype(int), axis=0, return_inverse=True, return_counts=True)
    idx_pts_vox_sorted=np.argsort(inverse)
    voxel_grid={}
    grid_candidate_center=[]
    last_seen=0

    for idx,vox in enumerate(non_empty_voxel_keys):
        voxel_grid[tuple(vox)]=points[idx_pts_vox_sorted[last_seen:last_seen+nb_pts_per_voxel[idx]]]
        representative = voxel_grid[tuple(vox)][np.linalg.norm(voxel_grid[tuple(vox)][:, :3] -np.mean(voxel_grid[tuple(vox)][:, :3],axis=0),axis=1).argmin()]
        grid_candidate_center.append(representative)
        last_seen+=nb_pts_per_voxel[idx]

    return np.array(grid_candidate_center)

def downsample_decimation(points, target):
    '''
    Downsample point cloud 'points' until the number of points
    is equal to target, through random decimation.
    '''
    n = points.shape[0]
    to_remove = n - target
    
    keep = np.random.choice(np.arange(n), size=(n - to_remove), replace=False)
    return points[keep, :]

def downsample_voxel_match_size(points, n, grid_start=0.5, step_size=0.25):
    '''
    Downsample points with a voxel grid filter. However, the grid size is 
    chosen through a brute force search. 

    We start at grid_size = grid_start, incrementing by step_size each iteration
    until the downsampled points have less than n points.

    Returns the downsampled points.
    '''
    N_points = len(points)
    grid_size = grid_start
    while N_points > n:
        down_pts = downsample_voxel_grid(points, grid_size)
        N_points = len(down_pts)
        grid_size += step_size
        
    return down_pts

def sync_size(lidar_data, n=1024, optimal_voxel=True):
    '''
    Given a list of lidar point clouds 'lidar_data', return a numpy array
    of lidar point clouds such that all the clouds have the same number of
    points.

    n - number of points to use
    '''
    new_points = []
    for points in tqdm(lidar_data):
        nonzero_pts = points[points[:, 4] != 0]
        zero_pts = points[points[:, 4] == 0]
        
        # If we have more nonzero (i.e., points that actually belong to trees)
        # than the target size n, then we must remove some of them.
        # The goal is to first get the number of nonzero points under the target
        # because then we can manipulate just the zero (i.e. background) points
        # to get to n. The background points are much less important, and so
        # we want to manipulate those if possible.
        if len(nonzero_pts) > n:
            # First, use a voxel grid filter to remove some of the nonzero points
            # use downsample_voxel_match_size for more data retained, but is
            # slower.
            if optimal_voxel:
                down_pts = downsample_voxel_match_size(nonzero_pts, n)
            else:
                down_pts = downsample_voxel_grid(points, 1)
            
            # If we still have excess points, we need to remove exactly enough
            # to get to n points. (Note, this should never happen when using
            # downsample_voxel_match_size)
            if len(down_pts) > n:
                # We just remove nonzero points at random to get to the target.
                nonzero_pts = downsample_decimation(down_pts, n)
                old_pts = np.r_[nonzero_pts, zero_pts]
            # Otherwise, we are done
            else:
                old_pts = np.r_[down_pts, zero_pts]
        else:
            old_pts = points
        
        # We now have less nonzero points than n. Thus, we can just pad/remove
        # zero points until we get to the target size n.
        amount = old_pts.shape[0] - n
        if amount > 0:
            npoints = remove_excess(old_pts, amount)
        elif amount < 0:
            npoints = pad_zero(old_pts, -amount)
        else:
            npoints = old_pts
        new_points.append(npoints)

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

        s = points[(points[:, 0] < avg_x) & (points[:, 1] < avg_y)]
        if len(s[s[:, 4] != 0]) >= threshold:
            smaller_points.append(standardize_points(s))
        s = points[(points[:, 0] < avg_x) & (points[:, 1] >= avg_y)]
        if len(s[s[:, 4] != 0]) >= threshold:
            smaller_points.append(standardize_points(s))
        s = points[(points[:, 0] >= avg_x) & (points[:, 1] < avg_y)]
        if len(s[s[:, 4] != 0]) >= threshold:
            smaller_points.append(standardize_points(s))
        s = points[(points[:, 0] >= avg_x) & (points[:, 1] >= avg_y)]
        if len(s[s[:, 4] != 0]) >= threshold:
            smaller_points.append(standardize_points(s))
    return smaller_points

def standardize_points(points):
    min_points = np.min(points, axis=0)
    points = points - min_points
    return points

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

def process_lidar(lidar_data, split=True, n=None, threshold=None,
    optimal_voxel=True, augment=False):
    '''
    Fully calls the pipeline for loading the lidar data.

    n - number of points per sample. Processes each point cloud so that each
    cloud has n points.
    threshold - any lidar images with less than this number of non background
    points are discarded
    split - whether to split the image into quadrants to decrease the # of
    points
    augment - whether to augment the data with rotations. If true, return
    the length of augmented data as well

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
    print('Syncing point cloud sizes...')
    synced = sync_size(lidar_data, n, optimal_voxel=optimal_voxel)

    if augment:
        print('Augmenting data...')
        augmented_data = rotation.augment_rotation(lidar_data)
        print('Syncing augmented data sizes...')
        augmented_synced = sync_size(augmented_data, n, optimal_voxel=optimal_voxel)
        augmented_size = augmented_synced.shape[0]
        synced = np.r_[synced, augmented_synced]

    #labels = create_group_matrix(synced, n)
    labels = synced[:,:,4]
    if augment:
        return synced[:, :, :3], labels, augmented_size
    return synced[:, :, :3], labels
