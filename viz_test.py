from utils.read_data import Data
import numpy as np
import utils.visualization.show3d_balls as viz
import random

d = Data(category='data_neon')
assert len(d.lidar) == len(d.labels)

point = d.lidar[108]
labels = point[:,4]
unique_labels = np.unique(labels)
num_labels = len(unique_labels)
print("{} Trees in this image".format(num_labels-1))

color_map = {k: np.random.rand(3) for k in unique_labels}
c_gt = np.array([color_map[k] for k in labels])

viz.showpoints(point[:,:3], c_gt=c_gt, ballradius=3)