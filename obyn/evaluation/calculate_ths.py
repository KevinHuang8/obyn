import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import tensorflow as tf
import numpy as np
import os
from ..models import model
from ..utils import read_data as read_data
from ..utils import constants as C
from ..utils.test_utils import GroupMerging, obtain_rank, Get_Ths
from tqdm import tqdm

tf.logging.set_verbosity(tf.logging.ERROR)

gpu_number = 0 # GPU number to use
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_number)

# Declare constants
POINT_NUM = C.NUM_POINTS # Number of points per point cloud
BATCH_SIZE = 1
NUM_GROUPS = C.NUM_GROUPS # Maximum number of instances (trees) in a point cloud
NUM_CATEGORY = 2 # Number of different classes (tree, ground)

def calculate_ths(X, y, name):
    '''
    Calculates Ths by running the model saved at 'model_path' on X and y.
    X and y should be training, not validation, data.

    Saves Ths to a file with name 'name'.
    '''
    model_path = str(C.CHECKPOINT_DIR / (name + '.ckpt'))

    with tf.Graph().as_default():
        is_training = False

        with tf.device('/gpu:' + str(gpu_number)):
            is_training_ph = tf.placeholder(tf.bool, shape=())

            pointclouds_ph, ptsseglabel_ph, ptsgroup_label_ph, _, _, _ = \
                model.placeholder_inputs(BATCH_SIZE, POINT_NUM, NUM_GROUPS,
                    NUM_CATEGORY)

            net_output = model.get_model(pointclouds_ph, is_training_ph,
                group_cate_num=NUM_CATEGORY, bn_decay=C.BN_DECAY)

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True

        with tf.Session(config=config) as sess:
            ths = np.zeros(NUM_CATEGORY)
            ths_ = np.zeros(NUM_CATEGORY)
            cnt = np.zeros(NUM_CATEGORY)

            # Restore variables from disk.
            saver.restore(sess, model_path)
            print ("Model loaded in file: %s" % model_path)

            # Load all data into memory
            all_data = X # Lidar points NxPOINT_NUMx3
             # Group/instance labels NxPOINT_NUM, will be one-hot encoded later
            all_group = y
            # Segmentation results NxPOINT_NUM: 0 for ground, 1 for tree
            num_data = all_data.shape[0]
            all_seg = np.where(all_group > 0, 1, 0)

            print('Calculating Ths...')
            for i in tqdm(range(num_data)):
                point_cloud = np.expand_dims(all_data[i], 0)

                feed_dict = {
                    pointclouds_ph: point_cloud,
                    is_training_ph: is_training,
                }

                pts_corr_val0, _, _ = \
                    sess.run([net_output['simmat'],
                              net_output['conf'],
                              net_output['semseg']],
                              feed_dict=feed_dict)

                pts_corr_val = np.squeeze(pts_corr_val0)

                gt_group = obtain_rank(all_group[i])

                ths, ths_, cnt = Get_Ths(pts_corr_val, all_seg[i], gt_group,
                    ths, ths_, cnt)

            ths = [ths[i]/cnt[i] if cnt[i] != 0 else 0.2 for i in range(len(cnt))]

            np.save(C.THS_DIR / (name + '.npy'), ths)

    return ths
