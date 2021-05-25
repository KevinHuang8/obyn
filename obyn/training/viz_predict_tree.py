import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import argparse
import tensorflow as tf
import numpy as np
import os
import sys
from ..models import model
from ..utils import read_data as read_data
import matplotlib.pyplot as plt
from ..utils import constants as C
from ..utils.test_utils import BlockMerging, GroupMerging, obtain_rank, Get_Ths
from tqdm import tqdm
from ..utils.visualization import show3d_balls as viz

tf.logging.set_verbosity(tf.logging.ERROR)

gpu_number = 0 # GPU number to use
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_number)

# Declare constants
POINT_NUM = C.NUM_POINTS # Number of points per point cloud
BATCH_SIZE = 1
NUM_GROUPS = C.NUM_GROUPS # Maximum number of instances (trees) in a point cloud
NUM_CATEGORY = 2 # Number of different classes (tree, ground)

def printout(flog, data):
    print(data)
    flog.write(data + '\n')

def predict(X, y, model_path):
    with tf.Graph().as_default():
        is_training = False

        with tf.device('/gpu:' + str(gpu_number)):
            is_training_ph = tf.placeholder(tf.bool, shape=())

            pointclouds_ph, ptsseglabel_ph, ptsgroup_label_ph, _, _, _ = \
                model.placeholder_inputs(BATCH_SIZE, POINT_NUM, NUM_GROUPS, NUM_CATEGORY)

            net_output = model.get_model(pointclouds_ph, is_training_ph, group_cate_num=NUM_CATEGORY)

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True

        with tf.Session(config=config) as sess:

            # Restore variables from disk.
            saver.restore(sess, model_path)
            print ("Model loaded in file: %s" % model_path)

            # Load all data into memory
            all_data = X # Lidar points NxPOINT_NUMx3
            all_group = y # Group/instance labels NxPOINT_NUM, will be one-hot encoded later
            all_seg = np.where(all_group > 0, 1, 0) # Segmentation results NxPOINT_NUM: 0 for ground, 1 for tree
            num_data = all_data.shape[0]

            for i in range(num_data):
                point_cloud = np.expand_dims(all_data[i], 0)
                instance_labels = np.expand_dims(all_group[i], 0)
                seg_labels = np.expand_dims(all_seg[i], 0)

                feed_dict = {
                    pointclouds_ph: point_cloud,
                    is_training_ph: is_training,
                }

                pts_corr_val0, pred_confidence_val0, ptsclassification_val0 = \
                    sess.run([net_output['simmat'],
                              net_output['conf'],
                              net_output['semseg']],
                              feed_dict=feed_dict)

                # Set Volume Statistics
                gap = 0.1
                volume_num = int(50 / gap)
                volume = -1* np.ones([volume_num,volume_num,volume_num]).astype(np.int32)
                volume_seg = -1* np.ones([volume_num,volume_num,volume_num, NUM_CATEGORY]).astype(np.int32)

                pts_corr_val = np.squeeze(pts_corr_val0)
                pred_confidence_val = np.squeeze(pred_confidence_val0)
                ptsclassification_val = np.argmax(np.squeeze(ptsclassification_val0),axis=1)

                #print(pts_corr_val.shape, pred_confidence_val.shape, ptsclassification_val.shape)

                # Make Prediction
                gt_group = obtain_rank(all_group[i])
                ths = np.zeros(NUM_CATEGORY)
                ths_ = np.zeros(NUM_CATEGORY)
                cnt = np.zeros(NUM_CATEGORY)
                ths, ths_, cnt = Get_Ths(pts_corr_val, all_seg[i], gt_group, ths, ths_, cnt)
                bin_label = [ths[i]/cnt[i] if cnt[i] != 0 else 0.2 for i in range(len(cnt))]

                groupids_block, refineseg, group_seg = GroupMerging(pts_corr_val, pred_confidence_val, ptsclassification_val, bin_label)
                #groupids = BlockMerging(volume, volume_seg, np.squeeze(point_cloud), groupids_block.astype(np.int32), group_seg, gap)
                groupids = obtain_rank(groupids_block)


                unique_pred_labels = np.unique(groupids)
                unique_true_labels = np.unique(gt_group)
                num_pred_labels = len(unique_pred_labels)
                num_true_labels = len(unique_true_labels)
                print("Predicted {} Trees in this image".format(num_pred_labels-1))
                print("Acutally {} Trees in this image".format(num_true_labels-1))

                color_map_p = {k: np.random.rand(3) for k in range(num_pred_labels)}
                color_map_gt = {k: np.random.rand(3) for k in range(num_true_labels)}
                c_p = np.array([color_map_p[k] for k in groupids])
                c_gt = np.array([color_map_gt[k] for k in gt_group])

                viz.showpoints(np.squeeze(point_cloud), c_gt=c_gt, c_pred=c_p, ballradius=3)
