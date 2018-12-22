import numpy as np
import tensorflow as tf
import os
import cv2
from keras.utils import np_utils
def pre_data(filename):
    assert os.path.exists(filename)
    f = open(filename, 'r')
    f_lines = f.readlines()
    f.close()
    image_data = []
    for idx, line in enumerate(f_lines):
        image_data.append(line)
    return image_data

def minibatches(inputs=None, targets=None, batch_size=None, shuffle=False):
  assert len(inputs) == len(targets)
  if shuffle:
    indices = np.arange(len(inputs))
    np.random.shuffle(indices)
  for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
    if shuffle:
      excerpt = indices[start_idx:start_idx + batch_size]
    else:
      excerpt = slice(start_idx, start_idx + batch_size)
    yield inputs[excerpt], targets[excerpt]

def conTrainImageBoundaryGenerator(training_datalist, batch_size, images_size, mun_classes):
    image_data = pre_data(training_datalist)
    X_idx = np.asarray(np.arange(0, len(image_data)), dtype=np.int32)
    while 1:
        for X_indices, _ in minibatches(X_idx, X_idx, batch_size, shuffle=True):
            y_label = []
            X_data = np.empty((batch_size, images_size, images_size, 3), dtype=np.float32)
            for i in range(0, batch_size):
                idx = X_indices[i]
                image_path = image_data[idx].split(' ')[0]
                image = cv2.resize(cv2.imread(image_path), (images_size, images_size))
                X_data[i] = image
                label = int(image_data[idx].split(' ')[1])
                y_label.append(label)
            y_label = np_utils.to_categorical(y_label, mun_classes)
            yield (X_data, y_label)

def conTestImageBoundaryGenerator(testing_datalist, batch_size, images_size, mun_classes):
  X_test = pre_data(testing_datalist)
  X_teidx = np.asarray(np.arange(0, len(X_test)), dtype=np.int32)
  while 1:
    for X_indices, _ in minibatches(X_teidx, X_teidx,
                                    batch_size, shuffle=False):
      y_label = []
      X_data = np.empty((batch_size, images_size, images_size, 3), dtype=np.float32)
      for i in range(0, batch_size):
         idx = X_indices[i]
         image_path = X_test[idx].split(' ')[0]
         image = cv2.resize(cv2.imread(image_path), (images_size, images_size))
         X_data[i] = image
         label = int(X_test[idx].split(' ')[1])
         y_label.append(label)
      y_label = np_utils.to_categorical(y_label, mun_classes)
      yield (X_data, y_label)
%.py
