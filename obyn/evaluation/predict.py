import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import tensorflow as tf
import numpy as np
import os
from ..models import model
from ..utils import read_data as read_data
import matplotlib.pyplot as plt
from ..utils import constants as C
from ..utils.test_utils import BlockMerging, GroupMerging, obtain_rank, Get_Ths
from ..evaluation.calculate_ths import calculate_ths
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

def model_output(X, model_path):
    '''
    Get model output on input X. 

    Model checkpoint is saved at 'model_path'
    '''
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

            # Restore variables from disk.
            saver.restore(sess, model_path)
            print ("Model loaded in file: %s" % model_path)

            # Load all data into memory
            all_data = X # Lidar points NxPOINT_NUMx3
            # Segmentation results NxPOINT_NUM: 0 for ground, 1 for tree
            num_data = all_data.shape[0]

            model_outputs = []
            print('Getting model outputs...')
            for i in tqdm(range(num_data)):
                point_cloud = np.expand_dims(all_data[i], 0)

                feed_dict = {
                    pointclouds_ph: point_cloud,
                    is_training_ph: is_training,
                }

                pts_corr_val0, pred_confidence_val0, ptsclassification_val0 = \
                    sess.run([net_output['simmat'],
                              net_output['conf'],
                              net_output['semseg']],
                              feed_dict=feed_dict)

                pts_corr_val = np.squeeze(pts_corr_val0)
                pred_confidence_val = np.squeeze(pred_confidence_val0)
                ptsclassification_val = np.argmax(
                    np.squeeze(ptsclassification_val0),axis=1)

                model_outputs.append({
                    'simmat': pts_corr_val,
                    'conf': pred_confidence_val,
                    'semseg': ptsclassification_val
                    })

    return model_outputs 


def predict(name, model_outputs=None, confidence_threshold=C.DEFAULT_CONFIDENCE_THRESHOLD, 
    X=None, y=None, reload_ths=False, use_outputs=True):
    '''
    Return model predictions.

    'name' - name of the model (same as the name of the Ths file)

    Can either pass in 'model_outputs' and 'use_outputs=True', in which case
    the model_outputs are used to make predictions, 
    OR
    pass in 'X' and 'use_outputs=False' to predictions on input 
    point cloud X using model 'name'.

    If Ths is not calculated yet, pass in gt labels 'y' to recalculate.

    'reload_ths' - whether to force reload ths. Note, 'y' needs to be provided
    to calculate Ths.
    '''
    model_path = str(C.CHECKPOINT_DIR / (name + '.ckpt'))

    if use_outputs:
        if model_outputs is None:
            raise ValueError('Need to provide model outputs.')
    else:
        if X is None:
            raise ValueError('Need to provide X to compute outputs.')
        model_outputs = model_output(X, model_path)

    if not reload_ths and (C.THS_DIR / (name + '.npy')).is_file():
        ths = np.load(C.THS_DIR / (name + '.npy'))
    else:
        if y is None:
            raise ValueError('Need to pass in labels to calculate ths.')
        ths = calculate_ths(X, y, model_path, name)

    predictions = []
    for output in tqdm(model_outputs):
        pts_corr_val = output['simmat']
        pred_confidence_val = output['conf']
        ptsclassification_val = output['semseg']

        # Make Prediction
        groupids_block, _, _ = GroupMerging(pts_corr_val, 
            pred_confidence_val, ptsclassification_val, ths, 
            confidence_threshold)
        groupids = obtain_rank(groupids_block)

        predictions.append(groupids)

    return predictions