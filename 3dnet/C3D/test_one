import tensorflow as tf
import numpy as np
import C3D_model
import data_processing
import cv2
import os
import imageio
import random
from scipy.misc import imread, imresize

TRAIN_CHECK_POINT = 'model/wwj/C3D.ckpt-44'


def prepare_con_depth_data(video_path, CLIP_LENGTH=32, crop_size=112, channel_num=3):

  assert os.path.exists(video_path)

  processed_images = np.empty((CLIP_LENGTH, crop_size, crop_size, channel_num), dtype=np.float32)
  crop_random = random.random()
  vid = imageio.get_reader(video_path, "ffmpeg")
  for idx in range(0, CLIP_LENGTH):
    image = vid.get_data(idx)
    image_h, image_w, image_c = np.shape(image)
    square_sz = min(image_h, image_w)
    crop_h = int((image_h - square_sz) / 2)
    crop_w = int((image_w - square_sz) / 2)
    image_crop = image[crop_h:crop_h + square_sz, crop_w:crop_w + square_sz, ::]
    processed_images[idx] = imresize(image_crop, (crop_size, crop_size))
  return processed_images

def test(data, NUM_CLASSES):
    data = np.reshape(data, (1,) + data.shape)
    batch_clips = tf.placeholder(tf.float32, [1, 32, 112, 112, 3], name='X')
    keep_prob = tf.placeholder(tf.float32)
    logits = C3D_model.C3D(batch_clips, NUM_CLASSES, keep_prob)

    restorer = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        restorer.restore(sess, TRAIN_CHECK_POINT)
        pred = sess.run(logits, feed_dict={batch_clips: data,
                                                     keep_prob: 1.0})
    return pred



if __name__ == '__main__':
    data = '/media/liuyu/_data2/gesture/avi/all32video/1/0101.avi'
    images = prepare_con_depth_data(data)
    y = test(images, 49)
    print(y)
