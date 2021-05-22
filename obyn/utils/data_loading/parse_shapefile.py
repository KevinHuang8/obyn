import os
import matplotlib.pyplot as plt
import geopandas as gpd
import rasterio
import rasterio.features
import rasterio.warp
from pathlib import Path
import numpy as np
from shapely.geometry import mapping
from PIL import Image
from ..constants import *

class Picture:
    
    def __init__(self, tif_filename):
        with rasterio.open(tif_filename) as dataset:
            self.left = dataset.bounds.left
            self.right = dataset.bounds.right
            self.bottom = dataset.bounds.bottom
            self.top = dataset.bounds.top
            self.dataset = dataset
            self.height = dataset.height
            self.width = dataset.width
    
    def polygon_in_picture(self, polygon):
        # Get the coordinates of the polygon
        coords = mapping(polygon)["coordinates"][0][1:]
        
        # Check if each coordinate of the polygon lies in our picture bounds
        in_picture = True
        for c in coords:
            x = c[0]
            y = c[1]
            if (x < self.left) or (x > self.right) or (y < self.bottom) or (y > self.top):
                in_picture = False
                break
        
        return in_picture
    
    def convert_polygon_to_pixels(self, polygon):
        # Get the coordinates of the polygon
        coords = mapping(polygon)["coordinates"][0][1:]
        
        # Return (xmin, ymin, xmax, ymax) of the polygon
        x_vals = []
        y_vals = []
        for c in coords:
            x = c[0]
            y = c[1]
            row, col = self.dataset.index(x,y)
            x_vals.append(max(0, min(col, self.width-1)))
            y_vals.append(max(0, min(row, self.height-1)))

        return([min(x_vals), min(y_vals), max(x_vals), max(y_vals)])

def parse_shapefile():
    shp_mlbs = gpd.read_file(str(IDTREES_DIR_RAW / 'ITC' / 'train_MLBS.shp'))
    shp_osbs = gpd.read_file(str(IDTREES_DIR_RAW / 'ITC' / 'train_OSBS.shp'))
    all_polygons = np.concatenate([shp_mlbs['geometry'], shp_osbs['geometry']])

    polygon_to_filename = {}
    filename_to_polygon = {}

    rgb_directory = IDTREES_DIR_RAW / 'RemoteSensing' / 'RGB'
    for filename in os.listdir(rgb_directory):
        filename_key, extension = os.path.splitext(filename)
        if extension != '.tif':
            continue
        
        filename_to_polygon[filename_key] = []
        picture = Picture(rgb_directory / filename)
        
        for polygon in all_polygons:
            if not mapping(polygon)["coordinates"][0][1:] in polygon_to_filename:
                polygon_to_filename[mapping(polygon)["coordinates"][0][1:]] = []
                
            if picture.polygon_in_picture(polygon):
                polygon_to_filename[mapping(polygon)["coordinates"][0][1:]].append(filename_key)
                filename_to_polygon[filename_key].append(picture.convert_polygon_to_pixels(polygon))

    return filename_to_polygon