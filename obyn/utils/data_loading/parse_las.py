import os
from pathlib import Path

import laspy
import numpy as np

from ..constants import *

def las_to_matrix(lasfile, xy_max=40):
    '''
    lasfile - a pathlib Path to a .las lidar file, relative from DATA_DIR

    Returns a matrix representation of the lidar data - a numpy array
    of shape (N, 5), where N is the total number of points in the point cloud.
    Columns 1, 2, and 3 represent the x, y, and z coordinates, 
    column 4 represents the intensity, and column 5 represents the point's
    classification (labeled by tree # or 0 for ground)
    '''

    if not lasfile.name.endswith('.las'):
        raise ValueError(f'Not a .las file: {lasfile}')

    las = laspy.file.File(DATA_DIR / lasfile, mode='r')
    
    # Standardize X,Y coordinates to 0-xy_max
    min_coords = las.header.min
    X = np.clip(las.x - min_coords[0], 0.0, xy_max)
    Y = np.clip(las.y - min_coords[1], 0.0, xy_max)
    Z = las.z - min_coords[2]

    try:
        return np.c_[X, Y, Z, las.intensity, las.label]
    except:
        return np.c_[X, Y, Z, las.intensity, las.user_data]

def load_las_directory(dirpath, xy_max=40):
    '''
    dirpath - a pathlib Path to a directory of .las files, relative to
    DATA_DIR

    Loads all of the .las lidar files in the directory to a list of numpy
    arrays, where each element in the list contains the data for one point
    cloud.

    Returns a dict where the keys are the filenames, and the values are the
    numpy matrices with the data. 
    '''
    directory = DATA_DIR / dirpath

    data_dict = {}
    for filename in os.listdir(directory):
        #print('start', filename)

        if not filename.endswith('.las'):
            continue

        #print(filename)

        data = las_to_matrix(dirpath / filename, xy_max=xy_max)
        
        filename = Path(filename).stem

        data_dict[filename] = data 

    return data_dict