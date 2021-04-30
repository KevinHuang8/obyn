import os
from pathlib import Path

import laspy
import numpy as np

from ..constants import *

def las_to_matrix(lasfile):
    '''
    lasfile - a pathlib Path to a .las lidar file, relative from DATA_DIR

    Returns a matrix representation of the lidar data - a numpy array
    of shape (N, 4), where N is the total number of points in the point cloud.
    columns, 1, 2, and 3 represent the x, y, and z coordinates, while
    the last column contains the intensity of the point.
    '''

    if not lasfile.name.endswith('.las'):
        raise ValueError(f'Not a .las file: {lasfile}')

    las = laspy.file.File(DATA_DIR / lasfile, mode='r')

    return np.c_[las.X, las.Y, las.Z, las.intensity]

def load_las_directory(dirpath):
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
        print('start', filename)

        if not filename.endswith('.las'):
            continue

        print(filename)

        data = las_to_matrix(dirpath / filename)
        
        filename = Path(filename).stem

        data_dict[filename] = data 

    return data_dict