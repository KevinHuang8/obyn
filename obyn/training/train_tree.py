import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import tensorflow as tf
import numpy as np
import os
from ..models import model
from ..utils import read_data as read_data
import matplotlib.pyplot as plt
from ..utils import constants as C
from ..utils.test_utils import BlockMerging, GroupMerging, obtain_rank
from tqdm import tqdm

tf.logging.set_verbosity(tf.logging.ERROR)

gpu_number = 0 # GPU number to use
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_number)

# Declare constants
POINT_NUM = C.NUM_POINTS # Number of points per point cloud
BATCH_SIZE = C.BATCH_SIZE
NUM_GROUPS = C.NUM_GROUPS # Maximum number of instances (trees) in a point cloud
NUM_CATEGORY = 2 # Number of different classes (tree, ground)
TRAINING_EPOCHES = C.TRAINING_EPOCHES

print('#### Batch Size: {0}'.format(BATCH_SIZE))
print('#### Point Number: {0}'.format(POINT_NUM))
print('### Number of training epochs: {0}'.format(TRAINING_EPOCHES))

DECAY_STEP = C.DECAY_STEP
DECAY_RATE = C.DECAY_RATE

LEARNING_RATE_CLIP = C.LEARNING_RATE_CLIP
BASE_LEARNING_RATE = C.BASE_LEARNING_RATE
MOMENTUM = C.MOMENTUM

def printout(flog, data):
    print(data)
    flog.write(data + '\n')

def train(data, name):
    '''
    Train the model.

    data - Data object
    name - a string for the model name. The model checkpoint will be saved
    with this name.
    '''
    print('Starting training...')
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
            # ptsseglabel_ph: NxPOINTxNUM_CATEGORY tensor of segmentation ground 
            # truth (one-hot encoded)
            # ptsgroup_label_ph: NxPOINT_NUMxNUM_GROUPS tensor of instance 
            # segmentation ground truth (one-hot encoded)
            # pts_seglabel_mask_ph: NxPOINT_NUM tensor of segmentation mask
            # pts_group_mask_ph: NxPOINT_NUM tensor of instance segmentation mask
            pointclouds_ph, ptsseglabel_ph, ptsgroup_label_ph, pts_seglabel_mask_ph, \
                pts_group_mask_ph, alpha_ph = \
                model.placeholder_inputs(BATCH_SIZE, POINT_NUM, NUM_GROUPS, NUM_CATEGORY)
            group_mat_label = tf.matmul(ptsgroup_label_ph, tf.transpose(ptsgroup_label_ph, perm=[0, 2, 1]))

            is_training_ph = tf.placeholder(tf.bool, shape=())

            # labels dict represents ground truth
            labels = {'ptsgroup': ptsgroup_label_ph,
                      'semseg': ptsseglabel_ph,
                      'semseg_mask': pts_seglabel_mask_ph,
                      'group_mask': pts_group_mask_ph}

            # Get the initialized model
            net_output = model.get_model(pointclouds_ph, is_training_ph, 
                group_cate_num=NUM_CATEGORY, bn_decay=C.BN_DECAY)
            loss, grouperr, same, same_cnt, diff, diff_cnt, pos, pos_cnt = \
                model.get_loss(net_output, labels, alpha_ph)

            total_training_loss_ph = tf.placeholder(tf.float32, shape=())
            group_err_loss_ph = tf.placeholder(tf.float32, shape=())
            total_train_loss_sum_op = tf.summary.scalar('total_training_loss', 
                total_training_loss_ph)
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
        # Lidar points NxPOINT_NUMx3
        all_data = data.get_x()
        # Group/instance labels NxPOINT_NUM, will be one-hot encoded later
        all_group = data.get_y()
        # Segmentation results NxPOINT_NUM: 0 for ground, 1 for tree
        all_seg = np.where(all_group > 0, 1, 0)

        data.train_test_split(C.VALIDATION_SIZE)

        train_data = data.x_train
        train_group = data.y_train
        train_seg = all_seg[data.train_idx]

        valid_data = data.x_valid
        valid_group = data.y_valid
        valid_seg = all_seg[data.valid_idx]

        num_data_train = len(train_data)
        num_data_valid = len(valid_data)
        num_batch_train = num_data_train // BATCH_SIZE
        num_batch_valid = num_data_valid // BATCH_SIZE

        def train_one_epoch(epoch_num):

            ### NOTE: is_training = False: We do not update bn parameters during 
            # training due to the small batch size. This requires pre-training 
            # PointNet with large batchsize (say 32).
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
                pts_label_one_hot, pts_label_mask = model.convert_seg_to_one_hot(
                    train_seg[order[begidx: endidx]])
                pts_group_label, pts_group_mask = model.convert_groupandcate_to_one_hot(
                    train_group[order[begidx: endidx]])

                feed_dict = {
                    pointclouds_ph: train_data[order[begidx: endidx], ...],
                    ptsseglabel_ph: pts_label_one_hot,
                    ptsgroup_label_ph: pts_group_label,
                    pts_seglabel_mask_ph: pts_label_mask,
                    pts_group_mask_ph: pts_group_mask,
                    is_training_ph: is_training,
                    alpha_ph: C.MIN_ALPHA,
                }

                _, loss_val, simmat_val, grouperr_val, same_val, same_cnt_val, \
                    diff_val, diff_cnt_val, pos_val, pos_cnt_val = sess.run(
                        [train_op, loss, net_output['simmat'], grouperr, same, \
                            same_cnt, diff, diff_cnt, pos, pos_cnt], feed_dict=feed_dict)
                total_loss += loss_val / BATCH_SIZE
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

            return loss_train

        def validate():
            is_training = False
            num_data = valid_data.shape[0]
            total_loss = 0

            for j in range(num_batch_valid):
                begidx = j * BATCH_SIZE
                endidx = (j + 1) * BATCH_SIZE

                pts = valid_data[begidx:endidx]
                seg = valid_seg[begidx:endidx]
                group = valid_group[begidx:endidx]

                pts_label_one_hot, pts_label_mask = model.convert_seg_to_one_hot(seg)
                pts_group_label, pts_group_mask = model.convert_groupandcate_to_one_hot(group)

                feed_dict = {
                    pointclouds_ph: pts,
                    is_training_ph: is_training,
                    ptsseglabel_ph: pts_label_one_hot,
                    ptsgroup_label_ph: pts_group_label,
                    pts_seglabel_mask_ph: pts_label_mask,
                    pts_group_mask_ph: pts_group_mask,
                    alpha_ph: C.MIN_ALPHA
                }

                loss_val = sess.run(loss, feed_dict=feed_dict)
                total_loss += ((loss_val / BATCH_SIZE)/num_batch_valid)

            return total_loss

        train_loss = []
        valid_loss = []
        for epoch in range(1, TRAINING_EPOCHES+1):
            printout(flog, '\n>>> Training epoch %d/%d ...' % (epoch, TRAINING_EPOCHES))
            epoch_train_loss = train_one_epoch(epoch)

            if epoch == TRAINING_EPOCHES:
                cp_filename = saver.save(sess, 
                    str(C.CHECKPOINT_DIR / (name + '.ckpt')))
                printout(flog, 'Successfully store the checkpoint model into ' + cp_filename)

            epoch_valid_loss = validate()
            train_loss.append(epoch_train_loss)
            valid_loss.append(epoch_valid_loss)
            print("Training Loss: {}, Validation Loss: {}".format(epoch_train_loss, 
                epoch_valid_loss))
            flog.flush()

        # Plot training/valid loss
        plt.figure()
        plt.plot(train_loss)
        plt.plot(valid_loss)
        plt.title('Model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig(C.FIGURES_DIR / f'loss_{name}.png')

        flog.close()

