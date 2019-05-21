import tensorflow as tf
import numpy as np
import C3D_model
import time
import data_processing
import os
import os.path
from os.path import join
import logging

logging.basicConfig(level=logging.WARNING,
                    format='%(message)s',
                    filename='Log/wwj/C3D_wwj.log',
                    filemode='w')
TRAIN_CHECK_POINT = 'model/C3D/'
#train_weight = 'model/C3D.ckpt-35'
TRAIN_LIST_PATH = 'train.txt'
VALIDATION_LIST_PATH = 'test.txt'
#TEST_LIST_PATH = 'video_depth_test.txt'
BATCH_SIZE = 10
NUM_CLASSES = 49
CROP_SZIE = 112
CHANNEL_NUM = 3
CLIP_LENGTH = 32
EPOCH_NUM = 1000
INITIAL_LEARNING_RATE = 0.0003
LR_DECAY_FACTOR = 0.5
EPOCHS_PER_LR_DECAY = 2
MOVING_AV_DECAY = 0.9999
#Get shuffle index
train_video_indices = data_processing.get_video_indices(TRAIN_LIST_PATH)
validation_video_indices = data_processing.get_video_indices(VALIDATION_LIST_PATH)

with tf.Graph().as_default():
    batch_clips = tf.placeholder(tf.float32, [BATCH_SIZE, CLIP_LENGTH, CROP_SZIE, CROP_SZIE, CHANNEL_NUM], name='X')
    batch_labels = tf.placeholder(tf.int32, [BATCH_SIZE, NUM_CLASSES], name='Y')
    keep_prob = tf.placeholder(tf.float32)
    logits = C3D_model.C3D(batch_clips, NUM_CLASSES, keep_prob)

    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=batch_labels))
        tf.summary.scalar('entropy_loss', loss)

    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(batch_labels, 1)), np.float32))
        tf.summary.scalar('accuracy', accuracy)
    #global_step = tf.Variable(0, name='global_step', trainable=False) 
    #decay_step = EPOCHS_PER_LR_DECAY * len(train_video_indices) // BATCH_SIZE
    learning_rate = 0.0003 #tf.train.exponential_decay(INITIAL_LEARNING_RATE, global_step, decay_step, LR_DECAY_FACTOR, staircase=True)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)#, global_step=global_step)
    saver = tf.train.Saver()
    summary_op = tf.summary.merge_all()
    gpu = '/gpu:' + str(0)
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    with tf.device(gpu):
      with tf.Session(config=config) as sess:
        #train_summary_writer = tf.summary.FileWriter(TRAIN_LOG_DIR, sess.graph)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        #saver.restore(sess, train_weight)  # 加载模型
        step = 0
        for epoch in range(EPOCH_NUM):
            accuracy_epoch = 0
            loss_epoch = 0
            batch_index = 0
            for i in range(len(train_video_indices) // BATCH_SIZE):
                step += 1
                batch_data, batch_index = data_processing.get_batches(TRAIN_LIST_PATH, NUM_CLASSES, CLIP_LENGTH, batch_index,
                                                         train_video_indices, BATCH_SIZE)
                _, loss_out, accuracy_out, summary = sess.run([optimizer, loss, accuracy, summary_op],
                                                              feed_dict={batch_clips:batch_data['clips'],
                                                              batch_labels:batch_data['labels'],
                                                                        keep_prob: 0.5})
                loss_epoch += loss_out
                accuracy_epoch += accuracy_out

                if i % 10 == 0:
                    print('Epoch %d, Batch %d: Loss is %.5f; Accuracy is %.5f'%(epoch+1, i, loss_out, accuracy_out))
                    logging.warning('Epoch %d, Batch %d: Loss is %.5f; Accuracy is %.5f'%(epoch+1, i, loss_out, accuracy_out))
                    #train_summary_writer.add_summary(summary, step)

            print('Epoch %d: Average loss is: %.5f; Average accuracy is: %.5f' % (epoch + 1, loss_epoch / (len(train_video_indices) // BATCH_SIZE), accuracy_epoch / (len(train_video_indices) // BATCH_SIZE)))
            logging.warning('Epoch %d: Average loss is: %.5f; Average accuracy is: %.5f' % (epoch + 1, loss_epoch / (len(train_video_indices) // BATCH_SIZE), accuracy_epoch / (len(train_video_indices) // BATCH_SIZE)))

            accuracy_epoch = 0
            loss_epoch = 0
            batch_index = 0
            for i in range(len(validation_video_indices) // BATCH_SIZE):
                batch_data, batch_index = data_processing.get_batches(VALIDATION_LIST_PATH, NUM_CLASSES, CLIP_LENGTH, batch_index,
                                                                      validation_video_indices, BATCH_SIZE)
                loss_out, accuracy_out = sess.run([loss, accuracy],
                                                  feed_dict={batch_clips:batch_data['clips'],
                                                             batch_labels:batch_data['labels'],
                                                            keep_prob: 1.0})
                loss_epoch += loss_out
                accuracy_epoch += accuracy_out

            print('Validation loss is %.5f; Accuracy is %.5f'%(loss_epoch / (len(validation_video_indices) // BATCH_SIZE),
                                                               accuracy_epoch /(len(validation_video_indices) // BATCH_SIZE)))
            logging.warning('Validation loss is %.5f; Accuracy is %.5f'%(loss_epoch / (len(validation_video_indices) // BATCH_SIZE),
                                                               accuracy_epoch /(len(validation_video_indices) // BATCH_SIZE)))
            saver.save(sess, TRAIN_CHECK_POINT + 'C3D.ckpt', global_step=epoch)





