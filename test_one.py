from res3d import td_res3d
import cv2
import random
from scipy.misc import imread, imresize
import tensorflow as tf
import os
keras = tf.contrib.keras
import numpy as np
import C3D_model
TRAIN_CHECK_POINT = './C3D.ckpt-48'
NUM_CLASSES = 49
p1 = 0.5
p2 = 0.5


def prepare_con_depth_data(video_path, CLIP_LENGTH=32, crop_size=112, channel_num=3):
    assert os.path.exists(video_path)
    processed_images = np.empty((CLIP_LENGTH, crop_size, crop_size, channel_num), dtype=np.float32)
    crop_random = random.random()
    vid = cv2.VideoCapture(video_path)
    idx = 0
    while (True):
        ret, frame = vid.read()  # 捕获一帧图像
        if ret:
            image = frame
            image_h, image_w, image_c = np.shape(image)
            square_sz = min(image_h, image_w)
            crop_h = int((image_h - square_sz) / 2)
            crop_w = int((image_w - square_sz) / 2)
            image_crop = image[crop_h:crop_h + square_sz, crop_w:crop_w + square_sz, ::]
            processed_images[idx] = imresize(image_crop, (crop_size, crop_size))
            idx = idx + 1
        else:
            break
    vid.release()
    # for idx in range(0, CLIP_LENGTH):
    #     image = vid.get_data(idx)
    #     image_h, image_w, image_c = np.shape(image)
    #     square_sz = min(image_h, image_w)
    #     crop_h = int((image_h - square_sz) / 2)
    #     crop_w = int((image_w - square_sz) / 2)
    #     image_crop = image[crop_h:crop_h + square_sz, crop_w:crop_w + square_sz, ::]
    #     processed_images[idx] = imresize(image_crop, (crop_size, crop_size))
    return processed_images


def test(datapath):
    data = prepare_con_depth_data(datapath)
    data1 = np.reshape(data, [1, 32, 112, 112, 3])
    weight_decay = 0.00005
    num_classes = 49
    keras.backend.clear_session()
    inputs = keras.layers.Input(batch_shape=(1, 32, 112, 112, 3))
    outputs = td_res3d(inputs, weight_decay, num_classes)
    model = keras.models.Model(inputs=inputs, outputs=outputs)
    model.load_weights('./wwj_0.001_weights.0.9296.h5')
    pred1 = model.predict(data1)
    tf.reset_default_graph()
     data2 = np.reshape(data, (1,) + data.shape)
     batch_clips = tf.placeholder(tf.float32, [1, 32, 112, 112, 3], name='X')
     keep_prob = tf.placeholder(tf.float32)
     logits = C3D_model.C3D(batch_clips, NUM_CLASSES, keep_prob)
     restorer = tf.train.Saver()
     config = tf.ConfigProto()
     config.gpu_options.allow_growth = True
     with tf.Session(config=config) as sess:
         restorer.restore(sess, TRAIN_CHECK_POINT)
         pred2 = sess.run(logits, feed_dict={batch_clips: data2,
                                                      keep_prob: 1.0})
     tf.reset_default_graph()
    pred = pred1*p1 + pred2*p2
    return pred


if __name__ == '__main__':
    data = './32video/4/0101.avi'
    y = test(data)
    print(y)
#GUIdemo的依赖项
