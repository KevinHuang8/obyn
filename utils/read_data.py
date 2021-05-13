import os
import pickle
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from .data_loading.parse_xml import parse_xml_dir
from .data_loading.parse_las import load_las_directory
from .data_loading.parse_tif import load_tif_directory
from .data_loading.parse_shapefile import parse_shapefile
from .process_lidar import *
from .constants import *

class Data:
    '''
    Condenses all of the types of data into one class.

    Attributes
        - rgb: list of rgb data
        - lidar: list of lidar data
        - hyperspectral: list of hyperspectral data
        - chm: list of canopy height data
        - labels: list of ground truth bounding boxes

    Note that all lists are in the correct order. I.e., the first element
    of self.rgb corresponds to the first element of self.labels

    Missing data has a value of None.
    '''
    # Where the data is serialized. We store the data in a serialized file
    # so repeated loads are fast.
    DATA_LOCATION_NEON = DATA_DIR / 'data_neon'
    DATA_LOCATION_IDTREES = DATA_DIR / 'data_idtrees'

    def __init__(self, category='all', force_reload=False):
        '''
        'category' specifies which dataset to load.
            - 'data_neon' loads the Neon Trees Evaluation dataset
            - 'data_idtrees' loads the IDTrees dataset
            - 'all' loads both (but note that images will be of different sizes)

        'force_reload' forces a generation of the serialized dataset from 
        scratch. Can take a while.
        '''
        self.force_reload = force_reload

        self.rgb = []
        self.lidar = []
        self.hyperspectral = []
        self.chm = []
        self.labels = []

        if category == 'all':
            self._load_all()
        elif category == 'data_neon':
            self._load_neon()
        elif category == 'data_idtrees':
            self._load_idtrees()
        else:
            raise ValueError(f'Incorrect data category: {category}')

    def _load_all(self):
        self._load_neon()
        neon_rgb = self.rgb
        neon_lidar = self.lidar
        neon_hyperspectral = self.hyperspectral
        neon_chm = self.chm
        neon_labels = self.labels

        self._load_idtrees()

        self.rgb.extend(neon_rgb)
        self.lidar.extend(neon_lidar)
        self.hyperspectral.extend(neon_hyperspectral)
        self.chm.extend(neon_chm)
        self.labels.extend(neon_labels)

    def _load_neon(self):
        if not self.force_reload and Data.DATA_LOCATION_NEON.is_file():
            # load from file
            with open(Data.DATA_LOCATION_NEON, 'rb') as f:
                data_obj = pickle.load(f)
            self.rgb = data_obj.rgb
            self.lidar = data_obj.lidar
            self.hyperspectral = data_obj.hyperspectral
            self.chm = data_obj.chm
            self.labels = data_obj.labels
            return

        # Create file from raw data
        label_dict = parse_xml_dir(NEON_DIR_RAW / 'annotations')
        image_dir = NEON_DIR_RAW / 'evaluation'
        rgb_dict = load_tif_directory(image_dir / 'RGB')
        hyperspectral_dict = load_tif_directory(image_dir / 'Hyperspectral',
            remove='_hyperspectral')
        chm_dict = load_tif_directory(image_dir / 'CHM', remove='_CHM')
        lidar_dict = load_las_directory(image_dir / 'LiDAR')

        # We iterate through all filenames like this so that the order
        # in each category (rgb, lidar, labels, etc.) match up
        # I.e., the first element in self.rgb corresponds with the 
        # first element in self.labels
        for filename in label_dict:
            # Some annotated files cannot be found in the dataset. These
            # probably are uncropped versions of other images already in 
            # the data (duplicates)
            # Thus, we only take the images that exist in RGB
            # RGB is the most general category, i.e. all lidar/hyperspectral/chm
            # images have an RGB counterpart (almost), but not vice versa
            if filename in rgb_dict:
                self.rgb.append(rgb_dict[filename])
                self.labels.append(label_dict[filename])

                # If any types of images don't exist, we put None
                for attrib, d in zip([self.hyperspectral, self.lidar, self.chm],
                    [hyperspectral_dict, lidar_dict, chm_dict]):
                    try:
                        attrib.append(d[filename])
                    except KeyError:
                        attrib.append(None)

        # Save to file
        with open(Data.DATA_LOCATION_NEON, 'wb+') as f:
            pickle.dump(self, f)

    def _load_idtrees(self):
        if not self.force_reload and Data.DATA_LOCATION_IDTREES.is_file():
            # load from file
            with open(Data.DATA_LOCATION_IDTREES, 'rb') as f:
                data_obj = pickle.load(f)
            self.rgb = data_obj.rgb
            self.lidar = data_obj.lidar
            self.hyperspectral = data_obj.hyperspectral
            self.chm = data_obj.chm
            self.labels = data_obj.labels
            return

        label_dict = parse_shapefile()
        image_dir = IDTREES_DIR_RAW / 'RemoteSensing'
        rgb_dict = load_tif_directory(image_dir / 'RGB')
        hyperspectral_dict = load_tif_directory(image_dir / 'HSI')
        chm_dict = load_tif_directory(image_dir / 'CHM')
        lidar_dict = load_las_directory(image_dir / 'LAS')

        for filename in label_dict:
            if filename in rgb_dict:
                self.rgb.append(rgb_dict[filename])
                self.labels.append(label_dict[filename])

                for attrib, d in zip([self.hyperspectral, self.lidar, self.chm],
                    [hyperspectral_dict, lidar_dict, chm_dict]):
                    try:
                        attrib.append(d[filename])
                    except KeyError:
                        attrib.append(None)

        # Save to file
        with open(Data.DATA_LOCATION_IDTREES, 'wb+') as f:
            pickle.dump(self, f)


    def remove_missing(self, name):
        '''
        Upon loading, Data might have None values for missing data. For example,
        not all locations have lidar images, or hyperspectral images.

        This method removes all None values from a single type of image (the
        type is given by 'name') and the corresponding bounding box labels, and
        returns the modified image data and labels.

        For example, remove_missing('hyperspectral') returns a list
        of hyperspectral images with None values removed, along with the list of
        labels corresponding to the None values also removed.

        Note that this doesn't change the other categories of images, so
        calling remove_missing('hyperspectral') doesn't remove None's from
        lidar images. Thus, the labels returned from remove_missing may not
        match the image data from types other than 'name'.

        Mainly used if only one type of image is needed. For example, if you
        wanted to train only on lidar data (and want to ignore the other
        types).
        '''
        images = getattr(self, name)

        where_missing = {i for i, val in enumerate(images) if val is None}
        new_images = []
        new_labels = []

        for i in range(len(images)):
            if i in where_missing:
                continue
            new_images.append(images[i])
            new_labels.append(self.labels[i])

        return new_images, new_labels

class LidarData(Data):
    DATA_LOCATION_NEON_LIDAR = DATA_DIR / 'data_neon_lidar'
    DATA_LOCATION_IDTREES_LIDAR = DATA_DIR / 'data_idtrees_lidar'

    def _load_neon(self):
        if not self.force_reload and LidarData.DATA_LOCATION_NEON_LIDAR.is_file():
            # load from file
            with open(LidarData.DATA_LOCATION_NEON_LIDAR, 'rb') as f:
                data_obj = pickle.load(f)
            self.lidar = data_obj.lidar
            self.x = data_obj.x
            self.y = data_obj.y
            return

        # Create file from raw data
        image_dir = NEON_DIR_RAW / 'evaluation'
        label_dict = parse_xml_dir(NEON_DIR_RAW / 'annotations')
        lidar_dict = load_las_directory(image_dir / 'LiDAR')

        for filename in label_dict:
            if filename in lidar_dict:
                self.lidar.append(lidar_dict[filename])

        self.x, self.y = process_lidar(self.lidar)

        # Save to file
        with open(LidarData.DATA_LOCATION_NEON_LIDAR, 'wb+') as f:
            pickle.dump(self, f)


if __name__ == '__main__':
    data = load_tif_directory(Path('train') / 'RemoteSensing' / 'RGB')
    plt.imshow(data[0])
    plt.show()
