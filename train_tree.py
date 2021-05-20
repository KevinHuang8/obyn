import argparse
import tensorflow as tf
import numpy as np
import os
import sys
from models import model
import utils.read_data as read_data

gpu_number = 1 # GPU number to use
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_number)

# Declare constants
POINT_NUM = 1024 # Number of points per point cloud
BATCH_SIZE = 16
NUM_GROUPS = 200 # Maximum number of instances (trees) in a point cloud
NUM_CATEGORY = 2 # Number of different classes (tree, ground)
TRAINING_EPOCHES = 2

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

            is_training_ph = tf.constant(True, dtype=tf.bool)

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
        data = read_data.LidarData(category='data_neon')
        all_data = data.x # Lidar points NxPOINT_NUMx3
        all_group = data.y # Group/instance labels NxPOINT_NUM, will be one-hot encoded later
        all_seg = np.where(all_group > 0, 1, 0) # Segmentation results NxPOINT_NUM: 0 for ground, 1 for tree

        num_data = all_data.shape[0]
        num_batch = num_data // BATCH_SIZE

        def train_one_epoch(epoch_num):

            ### NOTE: is_training = False: We do not update bn parameters during training due to the small batch size. This requires pre-training PointNet with large batchsize (say 32).
            is_training = True

            order = np.arange(num_data)
            np.random.shuffle(order)

            total_loss = 0.0
            total_grouperr = 0.0
            total_same = 0.0
            total_diff = 0.0
            total_pos = 0.0
            same_cnt0 = 0

            for j in range(num_batch):
                begidx = j * BATCH_SIZE
                endidx = (j + 1) * BATCH_SIZE

                # Convert the ground-truth labels to one-hot encode
                pts_label_one_hot, pts_label_mask = model.convert_seg_to_one_hot(all_seg[order[begidx: endidx]])
                pts_group_label, pts_group_mask = model.convert_groupandcate_to_one_hot(all_group[order[begidx: endidx]])

                feed_dict = {
                    pointclouds_ph: all_data[order[begidx: endidx], ...],
                    ptsseglabel_ph: pts_label_one_hot,
                    ptsgroup_label_ph: pts_group_label,
                    pts_seglabel_mask_ph: pts_label_mask,
                    pts_group_mask_ph: pts_group_mask,
                    is_training_ph: is_training,
                    alpha_ph: min(10., (float(epoch_num) / 5.) * 2. + 2.),
                }

                _, loss_val, simmat_val, grouperr_val, same_val, same_cnt_val, diff_val, diff_cnt_val, pos_val, pos_cnt_val = sess.run([train_op, loss, net_output['simmat'], grouperr, same, same_cnt, diff, diff_cnt, pos, pos_cnt], feed_dict=feed_dict)
                total_loss += loss_val
                total_grouperr += grouperr_val
                total_diff += (diff_val / diff_cnt_val)
                if same_cnt_val > 0:
                    total_same += same_val / same_cnt_val
                    same_cnt0 += 1
                total_pos += pos_val / pos_cnt_val


                if j % 10 == 9:
                    printout(flog, 'Batch: %d, loss: %f, grouperr: %f, same: %f, diff: %f, pos: %f' % (j, total_loss/10, total_grouperr/10, total_same/same_cnt0, total_diff/10, total_pos/10))

                    lr_sum, batch_sum, train_loss_sum, group_err_sum = sess.run( \
                        [lr_op, batch, total_train_loss_sum_op, group_err_op], \
                        feed_dict={total_training_loss_ph: total_loss / 10.,
                                   group_err_loss_ph: total_grouperr / 10., })

                    train_writer.add_summary(train_loss_sum, batch_sum)
                    train_writer.add_summary(lr_sum, batch_sum)
                    train_writer.add_summary(group_err_sum, batch_sum)

                    total_grouperr = 0.0
                    total_loss = 0.0
                    total_diff = 0.0
                    total_same = 0.0
                    total_pos = 0.0
                    same_cnt0 = 0


        for epoch in range(TRAINING_EPOCHES):
            printout(flog, '\n>>> Training epoch %d/%d ...' % (epoch, TRAINING_EPOCHES))

            train_one_epoch(epoch)
            flog.flush()

            '''
            cp_filename = saver.save(sess,
                                     os.path.join(MODEL_STORAGE_PATH, 'epoch_' + str(epoch + 1) + '.ckpt'))
            printout(flog, 'Successfully store the checkpoint model into ' + cp_filename)
            '''


        flog.close()


if __name__ == '__main__':
    train()
