import rasterio
import os
import laspy
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

DATA_DIR = Path('data')

def tif_to_matrix(tiffile):
    '''
    tiffile - patlib.Path to a '.tif' geotiff, relative from DATA_DIR

    Loads the raster data of tiffile to a numpy matrix with shape
    (height, width, #channels)
    '''

    if not tiffile.name.endswith('.tif'):
        raise ValueError(f'Not a .tif file: {tiffile}')

    with rasterio.open(DATA_DIR / tiffile, 'r') as geotiff:
        img_array = geotiff.read()
        img_array = np.moveaxis(img_array, 0, -1)
        img_array = img_array.astype(np.uint8)
        return img_array

def load_tif_directory(dirpath):
    '''
    dirpath - a pathlib Path to a directory of geotif files, relative to 
    DATA_DIR

    Loads all of the .tif geotiff files in the directory to a numpy matrix
    with shape (#images, height, width, #channels)

    Note: all .tif files in the directory must have the same exact shape
    '''
    directory = DATA_DIR / dirpath

    n = 0
    for filename in os.listdir(directory):
        if not filename.endswith('.tif'):
            continue
        n += 1

    if n == 0:
        raise ValueError(f'No .tif files in directory {directory}')
    
    for filename in os.listdir(directory):
        if not filename.endswith('.tif'):
            continue
        img = tif_to_matrix(dirpath / filename)
        img_shape = img.shape
        break
        
    data = np.zeros((n, *img_shape), dtype=np.uint8)
    i = 0
    for filename in os.listdir(directory):
        if not filename.endswith('.tif'):
            continue

        img = tif_to_matrix(dirpath / filename)
        if img.shape != img_shape:
            raise ValueError(
                f'Mismatching image shapes: {img_shape} != {img.shape}')

        data[i, :, :, :] = img
        i += 1

    return data

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
    '''
    data = []
    directory = DATA_DIR / dirpath

    for filename in os.listdir(directory):
        if not filename.endswith('.las'):
            continue
        
        data.append(las_to_matrix(dirpath / filename))

    return data

if __name__ == '__main__':
    data = load_tif_directory(Path('train') / 'RemoteSensing' / 'RGB')
    plt.imshow(data[0, :])
    plt.show()
