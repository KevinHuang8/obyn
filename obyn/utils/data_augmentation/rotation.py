import numpy as np
from ..process_lidar import standardize_points

def random_square(xmin, ymin, xmax, ymax, size):
    diag_len = size*np.sqrt(2) / 2
    
    xstart = xmin + diag_len
    xend = xmax - diag_len
    ystart = ymin + diag_len
    yend = ymax - diag_len
    
    x = np.random.uniform(xstart, xend)
    y = np.random.uniform(ystart, yend)
    center = (x, y)
    
    theta = np.random.uniform(0, 2*np.pi)
    
    dx1 = diag_len * np.cos(theta)
    dy1 = diag_len * np.sin(theta)
    dx2 = diag_len * np.cos(theta - np.pi/2)
    dy2 = diag_len * np.sin(theta - np.pi/2)
    
    return [np.array([x - dx1, y - dy1]), np.array([x + dx2, y + dy2]), np.array([x + dx1, y + dy1]),  np.array([x - dx2, y - dy2])], theta - np.pi / 4, center

def point_in_square(square, point):
    A, B, C, D = square
    
    AB = B - A
    AM = point - A
    BC = C - B
    BM = point - B
    
    return (0 <= np.dot(AB, AM) <= np.dot(AB, AB)) and (0 <= np.dot(BC, BM) <= np.dot(BC, BC))

def filter_points(square, points):
    filtered_points = []
    for pt in points:
        if point_in_square(square, np.r_[pt[0], pt[1]]):
            filtered_points.append(pt)
    return np.array(filtered_points)

def rotate_points(points, theta, center):
    x, y = center
    n = points.shape[0]
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta), -x*np.cos(theta) - y*np.sin(theta) + x],
        [np.sin(theta), np.cos(theta), -x*np.sin(theta) - y*np.cos(theta) + y],
        [0, 0, 1]
    ])
    
    X = np.c_[points[:, :2], np.ones(n)]
    rotated = X @ rotation_matrix.T
    
    points[:, :2] = rotated[:, :2]
    return points

def augment_rotation(lidar_data, size=20, num=3, threshold=50):
    augmented_points = []
    
    for points in tqdm(lidar_data):
        min_x = points[:, 0].min()
        max_x = points[:, 0].max()
        min_y = points[:, 1].min()
        max_y = points[:, 1].max()

        for i in range(num):
            square, theta, center = random_square(min_x, min_y, max_x, max_y, size)
            s = filter_points(square, points)
            if len(s[s[:, 4] != 0]) >= threshold:
                rotated = rotate_points(s, -theta, center)
                rotated[:, 0] = np.abs(rotated[:, 0] - square[0][0])
                rotated[:, 1] = np.abs(rotated[:, 1] - square[0][1])
                augmented_points.append(standardize_points(rotated))
                
    return augmented_points