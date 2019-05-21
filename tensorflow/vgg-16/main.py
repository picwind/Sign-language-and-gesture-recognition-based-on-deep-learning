import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import tensorflow as tf
from util import pre_data, conTrainImageBoundaryGenerator, conTestImageBoundaryGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.callbacks import LearningRateScheduler
keras = tf.contrib.keras
l2 = keras.regularizers.l2
K = tf.contrib.keras.backend
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
import cv2, numpy as np
from keras.optimizers import SGD, RMSprop, Adam
from datetime import datetime
#定义网络参数

nb_epoch = 1000
init_epoch = 0
keep_prob = 0.5
batch_size = 4
images_size = 224
num_channels = 3
num_classes = 49
OPTIMIZER = SGD()
train_filename = "train.txt"
test_filename = "test.txt"

train_data = pre_data(train_filename)
train_steps = len(train_data) / batch_size
test_data = pre_data(test_filename)
test_steps = len(test_data)/batch_size
model_prefix = '.'
dataset_name = 'congr'
weights_file = '%s/model/%s_weights.{epoch:02d}-{val_loss:.2f}.h5' % (model_prefix, dataset_name)

def VGG_16(weights_path=None): #根据keras官方文档建立VGG_16模型
     model = Sequential()
     model.add(ZeroPadding2D((1,1),input_shape=(224,224,3)))
     model.add(Conv2D(64, kernel_size=3, padding='same', activation='relu'))
     model.add(ZeroPadding2D((1,1)))
     model.add(Conv2D(64,kernel_size=3, padding='same', activation='relu'))
     model.add(MaxPooling2D((2,2), strides=(2,2)))
     model.add(ZeroPadding2D((1,1)))
     model.add(Conv2D(128, kernel_size=3, padding='same', activation='relu'))
     model.add(ZeroPadding2D((1,1)))
     model.add(Conv2D(128, kernel_size=3, padding='same', activation='relu'))
     model.add(MaxPooling2D((2,2), strides=(2,2)))
     model.add(ZeroPadding2D((1,1)))
     model.add(Conv2D(256, kernel_size=3, padding='same', activation='relu'))
     model.add(ZeroPadding2D((1,1)))
     model.add(Conv2D(256, kernel_size=3, padding='same', activation='relu'))
     model.add(ZeroPadding2D((1,1)))
     model.add(Conv2D(256, kernel_size=3, padding='same', activation='relu'))
     model.add(MaxPooling2D((2,2), strides=(2,2)))
     model.add(ZeroPadding2D((1,1)))
     model.add(Conv2D(512, kernel_size=3, padding='same', activation='relu'))
     model.add(ZeroPadding2D((1,1)))
     model.add(Conv2D(512, kernel_size=3, padding='same', activation='relu'))
     model.add(ZeroPadding2D((1,1)))
     model.add(Conv2D(512, kernel_size=3, padding='same', activation='relu'))
     model.add(MaxPooling2D((2,2), strides=(2,2)))
     model.add(ZeroPadding2D((1,1)))
     model.add(Conv2D(512, kernel_size=3, padding='same', activation='relu'))
     model.add(ZeroPadding2D((1,1)))
     model.add(Conv2D(512, kernel_size=3, padding='same', activation='relu'))
     model.add(ZeroPadding2D((1,1)))
     model.add(Conv2D(512, kernel_size=3, padding='same', activation='relu'))
     model.add(MaxPooling2D((2,2), strides=(2,2)))
     model.add(Flatten())
     model.add(Dense(4096, activation='relu'))
     model.add(Dropout(0.5))
     model.add(Dense(4096, activation='relu'))
     model.add(Dropout(0.5))
     model.add(Dense(num_classes, activation='softmax'))
     if weights_path:
         model.load_weights(weights_path)
     return model


def lr_polynomial_decay(global_step):
    learning_rate = 0.0001
    end_learning_rate = 0.000001
    decay_steps = train_steps * nb_epoch
    power = 0.9
    p = float(global_step) / float(decay_steps)
    lr = (learning_rate - end_learning_rate) * np.power(1 - p, power) + end_learning_rate
    if global_step > 0:
        curtime = '%s' % datetime.now()
        info = ' - lr: %.6f @ %s %d' % (lr, curtime.split('.')[0], global_step)
        print(info, )
    else:
        print('learning_rate: %.6f - end_learning_rate: %.6f - decay_steps: %d' % (
        learning_rate, end_learning_rate, decay_steps))
    return lr

model = VGG_16()
model.compile(loss='categorical_crossentropy', optimizer=OPTIMIZER,
              metrics=['accuracy'])
lr_reducer = LearningRateScheduler(lr_polynomial_decay, train_steps)
model_checkpoint = ModelCheckpoint(weights_file, monitor="val_acc",
                                   save_best_only=False, save_weights_only=True, mode='auto')
callbacks = [lr_reducer, model_checkpoint]
model.fit_generator(conTrainImageBoundaryGenerator(train_filename, batch_size, images_size, num_classes),
                    steps_per_epoch=train_steps,
                    epochs=nb_epoch,
                    verbose=1,
                    callbacks=callbacks,
                    validation_data=conTestImageBoundaryGenerator(test_filename, batch_size, images_size, num_classes),
                    validation_steps=test_steps,
                    initial_epoch=init_epoch
                    )
%.py
