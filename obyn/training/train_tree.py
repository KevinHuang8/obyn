import argparse
import tensorflow as tf
import numpy as np
import os
import sys
from ..models import model
from ..utils import read_data as read_data
import matplotlib.pyplot as plt
from ..utils import constants as C
from tqdm import tqdm

gpu_number = 0 # GPU number to use
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_number)

# Declare constants
POINT_NUM = 1024 # Number of points per point cloud
BATCH_SIZE = 32
NUM_GROUPS = C.NUM_GROUPS # Maximum number of instances (trees) in a point cloud
NUM_CATEGORY = 2 # Number of different classes (tree, ground)
TRAINING_EPOCHES = 20

print('#### Batch Size: {0}'.format(BATCH_SIZE))
print('#### Point Number: {0}'.format(POINT_NUM))
print('### Number of training epochs: {0}'.format(TRAINING_EPOCHES))

DECAY_STEP = 800000.
DECAY_RATE = 0.5

LEARNING_RATE_CLIP = 1e-6
BASE_LEARNING_RATE = 1e-4
MOMENTUM = 0.9

def printout(flog, data):
    print(data)
    flog.write(data + '\n')

def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:' + str(gpu_number)):
            batch = tf.Variable(0, trainable=False, name='batch')
            learning_rate = tf.train.exponential_decay(
                BASE_LEARNING_RATE,  # base learning rate
                batch * BATCH_SIZE,  # global_var indicating the number of steps
                DECAY_STEP,  # step size
                DECAY_RATE,  # decay rate
                staircase=True  # Stair-case or continuous decreasing
            )
            learning_rate = tf.maximum(learning_rate, LEARNING_RATE_CLIP)

            lr_op = tf.summary.scalar('learning_rate', learning_rate)

            # Placeholders to initialize the model
            # pointclouds_ph: BATCH_SIZExPOINT_NUMx3 tensor of lidar points
            # ptsseglabel_ph: NxPOINTxNUM_CATEGORY tensor of segmentation ground truth (one-hot encoded)
            # ptsgroup_label_ph: NxPOINT_NUMxNUM_GROUPS tensor of instance segmentation ground truth (one-hot encoded)
            # pts_seglabel_mask_ph: NxPOINT_NUM tensor of segmentation mask
            # pts_group_mask_ph: NxPOINT_NUM tensor of instance segmentation mask
            pointclouds_ph, ptsseglabel_ph, ptsgroup_label_ph, pts_seglabel_mask_ph, pts_group_mask_ph, alpha_ph = \
                model.placeholder_inputs(BATCH_SIZE, POINT_NUM, NUM_GROUPS, NUM_CATEGORY)

            is_training_ph = tf.placeholder(tf.bool, shape=())

            # labels dict represents ground truth
            labels = {'ptsgroup': ptsgroup_label_ph,
                      'semseg': ptsseglabel_ph,
                      'semseg_mask': pts_seglabel_mask_ph,
                      'group_mask': pts_group_mask_ph}

            # Get the initialized model
            net_output = model.get_model(pointclouds_ph, is_training_ph, group_cate_num=NUM_CATEGORY)
            loss, grouperr, same, same_cnt, diff, diff_cnt, pos, pos_cnt = model.get_loss(net_output, labels, alpha_ph)

            total_training_loss_ph = tf.placeholder(tf.float32, shape=())
            group_err_loss_ph = tf.placeholder(tf.float32, shape=())
            total_train_loss_sum_op = tf.summary.scalar('total_training_loss', total_training_loss_ph)
            group_err_op = tf.summary.scalar('group_err_loss', group_err_loss_ph)

        train_variables = tf.trainable_variables()

        trainer = tf.train.AdamOptimizer(learning_rate)
        train_op = trainer.minimize(loss, var_list=train_variables, global_step=batch)

        loader = tf.train.Saver([v for v in tf.global_variables()#])
                                 if
                                   ('conf_logits' not in v.name) and
                                    ('Fsim' not in v.name) and
                                    ('Fsconf' not in v.name) and
                                    ('batch' not in v.name)
                                ])

        saver = tf.train.Saver([v for v in tf.global_variables()])

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        sess = tf.Session(config=config)

        init = tf.global_variables_initializer()
        sess.run(init)

        train_writer = tf.summary.FileWriter('./train', sess.graph)
        flog = open('log.txt', 'w')

        # Load all data into memory
        data = read_data.LidarData(category='all')
        all_data = data.x # Lidar points NxPOINT_NUMx3
        all_group = data.y # Group/instance labels NxPOINT_NUM, will be one-hot encoded later
        all_seg = np.where(all_group > 0, 1, 0) # Segmentation results NxPOINT_NUM: 0 for ground, 1 for tree

        # Train/Validation Split
        idx = np.arange(all_data.shape[0])
        validation_percentage = 0.0
        np.random.shuffle(idx)
        cutoff_idx = int(len(idx) * validation_percentage)
        train_idx = idx[cutoff_idx:]
        valid_idx = idx[:cutoff_idx]

        train_data = all_data[train_idx]
        train_group = all_group[train_idx]
        train_seg = all_seg[train_idx]

        valid_data = all_data[valid_idx]
        valid_group = all_group[valid_idx]
        valid_seg = all_seg[valid_idx]

        num_data_train = len(train_data)
        num_data_valid = len(valid_data)
        num_batch_train = num_data_train // BATCH_SIZE
        num_batch_valid = num_data_valid // BATCH_SIZE

        def train_one_epoch(epoch_num):

            ### NOTE: is_training = False: We do not update bn parameters during training due to the small batch size. This requires pre-training PointNet with large batchsize (say 32).
            is_training = True

            order = np.arange(num_data_train)
            np.random.shuffle(order)

            total_loss = 0.0
            total_grouperr = 0.0
            total_same = 0.0
            total_diff = 0.0
            total_pos = 0.0
            same_cnt0 = 0

            for j in tqdm(range(num_batch_train)):
                begidx = j * BATCH_SIZE
                endidx = (j + 1) * BATCH_SIZE

                # Convert the ground-truth labels to one-hot encode
                pts_label_one_hot, pts_label_mask = model.convert_seg_to_one_hot(train_seg[order[begidx: endidx]])
                pts_group_label, pts_group_mask = model.convert_groupandcate_to_one_hot(train_group[order[begidx: endidx]])

                feed_dict = {
                    pointclouds_ph: train_data[order[begidx: endidx], ...],
                    ptsseglabel_ph: pts_label_one_hot,
                    ptsgroup_label_ph: pts_group_label,
                    pts_seglabel_mask_ph: pts_label_mask,
                    pts_group_mask_ph: pts_group_mask,
                    is_training_ph: is_training,
                    alpha_ph: min(10., (float(epoch_num-1) / 5.) * 2. + 2.),
                }

                _, loss_val, simmat_val, grouperr_val, same_val, same_cnt_val, diff_val, diff_cnt_val, pos_val, pos_cnt_val = sess.run([train_op, loss, net_output['simmat'], grouperr, same, same_cnt, diff, diff_cnt, pos, pos_cnt], feed_dict=feed_dict)
                total_loss += loss_val
                total_grouperr += grouperr_val
                total_diff += (diff_val / diff_cnt_val)
                if same_cnt_val > 0:
                    total_same += same_val / same_cnt_val
                    same_cnt0 += 1
                total_pos += pos_val / pos_cnt_val



            # Return train loss values per epoch
            loss_train = total_loss/num_batch_train
            grouperr_train = total_grouperr/num_batch_train
            same_train = total_same/same_cnt0
            diff_train = total_diff/num_batch_train
            pos_train = total_pos/num_batch_train

            '''
            # Evaluate validation error
            total_loss = 0.0
            total_grouperr = 0.0
            total_same = 0.0
            total_diff = 0.0
            total_pos = 0.0
            same_cnt0 = 0

            for j in tqdm(range(num_batch_valid)):
                begidx = j * BATCH_SIZE
                endidx = (j + 1) * BATCH_SIZE

                # Convert the ground-truth labels to one-hot encode
                pts_label_one_hot, pts_label_mask = model.convert_seg_to_one_hot(valid_seg[begidx: endidx])
                pts_group_label, pts_group_mask = model.convert_groupandcate_to_one_hot(valid_group[begidx: endidx])

                feed_dict = {
                    pointclouds_ph: valid_data[begidx: endidx, ...],
                    ptsseglabel_ph: pts_label_one_hot,
                    ptsgroup_label_ph: pts_group_label,
                    pts_seglabel_mask_ph: pts_label_mask,
                    pts_group_mask_ph: pts_group_mask,
                    is_training_ph: is_training,
                    alpha_ph: min(10., (float(epoch_num-1) / 5.) * 2. + 2.),
                }
                _, loss_val, simmat_val, grouperr_val, same_val, same_cnt_val, diff_val, diff_cnt_val, pos_val, pos_cnt_val = sess.run([train_op, loss, net_output['simmat'], grouperr, same, same_cnt, diff, diff_cnt, pos, pos_cnt], feed_dict=feed_dict)
                total_loss += loss_val
                total_grouperr += grouperr_val
                total_diff += (diff_val / diff_cnt_val)
                if same_cnt_val > 0:
                    total_same += same_val / same_cnt_val
                    same_cnt0 += 1
                total_pos += pos_val / pos_cnt_val

            # Return valid loss values per epoch
            loss_valid = total_loss/num_batch_valid
            grouperr_valid = total_grouperr/num_batch_valid
            same_valid = total_same/same_cnt0
            diff_valid = total_diff/num_batch_valid
            pos_valid = total_pos/num_batch_valid
            '''

            return loss_train, 0


        train_loss = []
        valid_loss = []
        for epoch in range(1, TRAINING_EPOCHES+1):
            printout(flog, '\n>>> Training epoch %d/%d ...' % (epoch, TRAINING_EPOCHES))

            epoch_train_loss, epoch_valid_loss = train_one_epoch(epoch)
            train_loss.append(epoch_train_loss)
            valid_loss.append(epoch_valid_loss)
            print("Training Loss: {}, Validation Loss: {}".format(epoch_train_loss, epoch_valid_loss))
            flog.flush()

            '''
            cp_filename = saver.save(sess,
                                     os.path.join(MODEL_STORAGE_PATH, 'epoch_' + str(epoch + 1) + '.ckpt'))
            printout(flog, 'Successfully store the checkpoint model into ' + cp_filename)
            '''

        # Plot training loss
        plt.plot(train_loss)
        plt.plot(valid_loss)
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()

        flog.close()


if __name__ == '__main__':
    train()
