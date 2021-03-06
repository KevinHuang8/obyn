import os
import pickle
import numpy as np
import laspy
from pathlib import Path
import matplotlib.pyplot as plt
from .data_loading.parse_xml import parse_xml_dir
from .data_loading.parse_las import load_las_directory
from .data_loading.parse_tif import load_tif_directory
from .data_loading.parse_shapefile import parse_shapefile
from .data_augmentation.artificial_labelling import create_artificial_labels
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

    def __init__(self, category='data_neon', force_reload=False):
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
    def __init__(self, name, category='data_neon', force_reload=False,
            group_threshold=GROUP_THRESHOLD):
        '''
        'name' - the name of the model associated with this data

        'category' specifies which dataset to load.
            - 'data_neon' loads the Neon Trees Evaluation dataset
            - 'data_idtrees' loads the IDTrees dataset
            - 'all' loads both (but note that images will be of different sizes)

        'force_reload' forces a generation of the serialized dataset from 
        scratch. Can take a while.

        'group_threshold' - trees with less than this # of points are removed.
        '''
        if category != 'data_neon':
            raise ValueError('Only "data_neon" has correctly labeled data.')
        self.name = name
        self.savefile = DATA_GENERATED_DIR / name
        self.group_threshold = group_threshold
        super().__init__(category=category, force_reload=force_reload)

    def _load_neon(self):
        if not self.force_reload and self.savefile.is_file():
            # load from file
            with open(self.savefile, 'rb') as f:
                data_obj = pickle.load(f)
            self.lidar = data_obj.lidar
            self.lidar_filenames = data_obj.lidar_filenames
            self.group_threshold = data_obj.group_threshold
            self.x = data_obj.x
            self.y = data_obj.y
            self.num_points = data_obj.num_pts
            return

        print('Reloading data...')
        # Create file from raw data
        image_dir = NEON_DIR_RAW / 'evaluation'
        lidar_dict = load_las_directory(image_dir / 'LiDAR')

        self.lidar_filenames = []

        for filename in lidar_dict:
            las = laspy.file.File(image_dir / 'LiDAR' / (filename + '.las'), 
                mode='r')
            # las images without 'label' field do not have labels
            try:
                las.label
            except:
                continue

            self.lidar.append(lidar_dict[filename])
            self.lidar_filenames.append(filename)

        self.x, self.y, self.num_pts = process_lidar(self.lidar, group_threshold=self.group_threshold)

        # Save to file
        with open(self.savefile, 'wb+') as f:
            print('Caching data...')
            pickle.dump(self, f)

    def _load_idtrees(self):
        if not self.force_reload and self.savefile.is_file():
            # load from file
            with open(self.savefile, 'rb') as f:
                data_obj = pickle.load(f)
            self.lidar = data_obj.lidar
            self.x = data_obj.x
            self.y = data_obj.y
            return

        # Create file from raw data
        image_dir = IDTREES_DIR_RAW / 'RemoteSensing'
        label_dict = parse_shapefile()
        lidar_dict = load_las_directory(image_dir / 'LAS', xy_max=20)

        for filename in label_dict:
            if filename in lidar_dict:
                self.lidar.append(lidar_dict[filename])

        self.x, self.y = process_lidar(self.lidar, split=False)

        # Save to file
        with open(self.savefile, 'wb+') as f:
            pickle.dump(self, f)

    def _load_all(self):
        self._load_neon()
        x = self.x
        y = self.y
        neon_lidar = self.lidar

        self._load_idtrees()
        self.x = np.r_[x, self.x]
        self.y = np.r_[y, self.y]
        self.lidar.extend(neon_lidar)

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y

    def train_test_split(self, validation_size):
        idx = np.arange(self.x.shape[0])
        np.random.shuffle(idx)
        cutoff_idx = int(idx.shape[0] * validation_size)
        train_idx = idx[cutoff_idx:]
        valid_idx = idx[:cutoff_idx]

        self.x_train = self.x[train_idx]
        self.x_valid = self.x[valid_idx]
        self.y_train = self.y[train_idx]
        self.y_valid = self.y[valid_idx]

        self.train_idx = train_idx
        self.valid_idx = valid_idx

    def save_indices(self, name):
        np.save(IDX_DIR / (name + '_train.npy'), self.valid_idx)
        np.save(IDX_DIR / (name + '_test.npy'), self.train_idx)

    def load_indices(self, name):
        valid = np.load(IDX_DIR / (name + '_train.npy'))
        train = np.load(IDX_DIR / (name + '_test.npy'))
        return train, valid

class LidarDataAugmented(LidarData):
    def _load_neon(self):
        if not self.force_reload and self.savefile.is_file():
            # load from file
            with open(self.savefile, 'rb') as f:
                data_obj = pickle.load(f)
            self.lidar = data_obj.lidar
            self.lidar_filenames = data_obj.lidar_filenames
            self.group_threshold = data_obj.group_threshold
            self.x = data_obj.x
            self.y = data_obj.y

            self.num_points = data_obj.num_pts
            self.augment_size = data_obj.augment_size
            return

        print('Reloading data...')
        # Create file from raw data
        image_dir = NEON_DIR_RAW / 'evaluation'
        lidar_dict = load_las_directory(image_dir / 'LiDAR')

        self.lidar_filenames = []

        for filename in lidar_dict:
            las = laspy.file.File(image_dir / 'LiDAR' / (filename + '.las'), 
                mode='r')
            # las images without 'label' field do not have labels
            try:
                las.label
            except:
                continue

            self.lidar.append(lidar_dict[filename])
            self.lidar_filenames.append(filename)

        self.x, self.y, self.augment_size, self.num_pts = process_lidar(self.lidar, 
            augment=True, group_threshold=self.group_threshold)

        # Save to file
        with open(self.savefile, 'wb+') as f:
            print('Caching data...')
            pickle.dump(self, f)

    def train_test_split(self, validation_size):
        real_size = self.x.shape[0] - self.augment_size
        idx = np.arange(real_size)
        np.random.shuffle(idx)
        cutoff_idx = int(idx.shape[0] * validation_size)
        train_idx = idx[cutoff_idx:]
        valid_idx = idx[:cutoff_idx]

        augment_idx = np.arange(real_size, self.x.shape[0])

        train_idx = np.r_[train_idx, augment_idx]
        np.random.shuffle(train_idx)

        self.x_train = self.x[train_idx]
        self.x_valid = self.x[valid_idx]
        self.y_train = self.y[train_idx]
        self.y_valid = self.y[valid_idx]

        self.train_idx = train_idx
        self.valid_idx = valid_idx 
    

class LidarDataArtificial(LidarData):
    def __init__(self, name, category='data_neon', force_reload=False, skip=1):
        '''
        skip - take only every 'skip' samples from the nonlabled data
        (as there are ~6800 samples, which can take a long time to process)
        '''
        self.skip = skip
        super().__init__(name, category=category, force_reload=force_reload)

    def _load_neon(self):
        if not self.force_reload and self.savefile.is_file():
            # load from file
            with open(self.savefile, 'rb') as f:
                data_obj = pickle.load(f)
            self.lidar = data_obj.lidar
            self.lidar_filenames = data_obj.lidar_filenames
            self.x = data_obj.x
            self.y = data_obj.y

            self.x_artificial = data_obj.x_artificial
            self.y_artificial = data_obj.y_artificial
            return

        print('Reloading data...')
        # Create file from raw data
        image_dir = NEON_DIR_RAW / 'evaluation'
        lidar_dict = load_las_directory(image_dir / 'LiDAR')

        non_labeled_clouds = []
        self.lidar_filenames = []

        for filename in lidar_dict:
            las = laspy.file.File(image_dir / 'LiDAR' / (filename + '.las'), 
                mode='r')
            # las images without 'label' field do not have labels
            try:
                las.label
            except:
                non_labeled_clouds.append(lidar_dict[filename])
                continue

            self.lidar.append(lidar_dict[filename])
            self.lidar_filenames.append(filename)

        self.x, self.y = process_lidar(self.lidar)
        self.x_artificial, self.y_artificial = process_lidar(
            non_labeled_clouds[::(self.skip)], optimal_voxel=False)
        self.y_artificial = create_artificial_labels(
            self.x_artificial, self.y_artificial)

        # Save to file
        with open(self.savefile, 'wb+') as f:
            print('Caching data...')
            pickle.dump(self, f)

    def get_combined(self):
        '''
        Get the ground truth data along with the artificially labeled data
        in combined form.
        '''
        x = np.r_[self.x, self.x_artificial]
        y = np.r_[self.y, self.y_artificial]
        return x, y

    def get_x(self):
        return self.get_combined()[0]

    def get_y(self):
        return self.get_combined()[1]

    def train_test_split(self, validation_size):
        idx = np.arange(self.x.shape[0])
        np.random.shuffle(idx)
        cutoff_idx = int(idx.shape[0] * validation_size)
        train_idx = idx[cutoff_idx:]
        valid_idx = idx[:cutoff_idx]

        artificial_idx = np.arange(self.x.shape[0], 
            self.x.shape[0] + self.x_artificial.shape[0])

        train_idx = np.r_[train_idx, artificial_idx]
        np.random.shuffle(train_idx)

        x, y = self.get_combined()

        self.x_train = x[train_idx]
        self.x_valid = x[valid_idx]
        self.y_train = y[train_idx]
        self.y_valid = y[valid_idx]

        self.train_idx = train_idx
        self.valid_idx = valid_idx


