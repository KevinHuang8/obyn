import os
from pathlib import Path

import numpy as np
import rasterio
from rasterio.errors import RasterioIOError

from ..constants import *

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

def load_tif_directory(dirpath, remove=None):
    '''
    dirpath - a pathlib Path to a directory of geotif files, relative to
    DATA_DIR

    Loads all of the .tif geotiff files in the directory to a list of
    numpy matrices of shape (height, width, #channels).

    Returns a dict where the keys are the filenames, and the values are the
    numpy matrices with the image data. 

    'remove' is a string that should be removed from filenames. For example,
    hyperpsectral filenames have a trailing '_hyperspectral', which should be
    removed to match filenames from other types of data. 
    '''
    directory = DATA_DIR / dirpath
        
    data_dict = {}
    for filename in os.listdir(directory):
        if not filename.endswith('.tif'):
            continue
        
        try:
            img = tif_to_matrix(dirpath / filename)
        except RasterioIOError:
            continue

        filename = Path(filename).stem
        if remove:
            filename = filename.replace(remove, '')

        data_dict[filename] = img

    return data_dict
