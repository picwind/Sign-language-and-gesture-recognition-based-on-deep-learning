import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import io
import sys
sys.path.append("./networks")
import numpy as np
import tensorflow as tf
keras = tf.contrib.keras
l2 = keras.regularizers.l2
K = tf.contrib.keras.backend
import inputs as data
from res3d import td_res3d
from callbacks import LearningRateScheduler
from datagen import conTrainImageBoundaryGenerator_my_single, conTestImageBoundaryGenerator_my_single
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


Depth = 1
nb_epoch = 1000
init_epoch = 0
depth = 32
batch_size = 10
num_classes = 49
weight_decay = 0.00005
dataset_name = 'wwj_0.001'
training_datalist = 'train.txt'
testing_datalist = 'test.txt'
model_prefix = '.'
weights_file = '%s/trained_models/wwj/%s_weights.{val_acc:.4f}.h5' % (model_prefix, dataset_name)


train_data = data.load_con_video_list(training_datalist)
train_steps = len(train_data) / batch_size
test_data = data.load_con_video_list(testing_datalist)
test_steps = len(test_data)/batch_size
print('nb_epoch: %d - maxdepth: %d - batch_size: %d - weight_decay: %.6f' % (nb_epoch, depth, batch_size, weight_decay))


def lr_polynomial_decay(global_step):
    learning_rate = 0.001
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

inputs = keras.layers.Input(batch_shape=(None, depth, 112, 112, 3))
outputs = td_res3d(inputs, weight_decay, num_classes)

model = keras.models.Model(inputs=inputs, outputs=outputs)
pretrained_model = '%s/trained_models/tdres3d_depth_0.001.h5' %model_prefix

model.load_weights(pretrained_model, by_name=True)


optimizer = keras.optimizers.SGD(lr=0.0001, decay=0, momentum=0.9, nesterov=False)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

lr_reducer = LearningRateScheduler(lr_polynomial_decay, train_steps)
model_checkpoint = ModelCheckpoint(weights_file, monitor="val_acc",
                                   save_best_only=False, save_weights_only=True, mode='auto')
callbacks = [lr_reducer, model_checkpoint]
model.fit_generator(conTrainImageBoundaryGenerator_my_single(training_datalist, batch_size, depth, Depth, num_classes),
                    steps_per_epoch=train_steps,
                    epochs=nb_epoch,
                    verbose=1,
                    callbacks=callbacks,
                    validation_data=conTestImageBoundaryGenerator_my_single(testing_datalist, batch_size, depth, Depth, num_classes),
                    validation_steps=test_steps,
                    initial_epoch=init_epoch
                    )
