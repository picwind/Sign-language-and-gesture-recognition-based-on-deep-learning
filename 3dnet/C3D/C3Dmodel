import tensorflow as tf
import tensorflow.contrib.slim as slim

def C3D(input, num_classes, keep_pro=0.5):
    with tf.variable_scope('C3D'):
        with slim.arg_scope([slim.conv3d],
                            padding='SAME',
                            weights_regularizer=slim.l2_regularizer(0.0005),
                            activation_fn=tf.nn.relu,
                            kernel_size=[3, 3, 3],
                            stride=[1, 1, 1]
                            ):
            net = slim.conv3d(input, 64, scope='conv1')  #(?,16,112,112,64)
            net = slim.max_pool3d(net, kernel_size=[1, 3, 3], stride=[1, 2, 2], padding='SAME', scope='max_pool1')   #(?,16,56,56,64)
            net = slim.conv3d(net, 128, scope='conv2')  #(?,16,56,56,128)
            net = slim.max_pool3d(net, kernel_size=[3, 3, 3], stride=[2, 2, 2], padding='SAME', scope='max_pool2')   #(?,8,28,28,128)

            net = slim.repeat(net, 2, slim.conv3d, 256, scope='conv3')   #(?,8,28,28,256)
            net = slim.max_pool3d(net, kernel_size=[3, 3, 3], stride=[2, 2, 2], padding='SAME', scope='max_pool3')   #(?,4,12,12,256)
            net = slim.repeat(net, 2, slim.conv3d, 512, scope='conv4')    #(?,4,14,14,512)
            net = slim.max_pool3d(net, kernel_size=[3, 3, 3], stride=[2, 2, 2], padding='SAME', scope='max_pool4')   #(?,2,7,7,512)
            net = slim.repeat(net, 2, slim.conv3d, 512, scope='conv5')  #(?,2,7,7,512)
            net = slim.max_pool3d(net, kernel_size=[3, 3, 3], stride=[2, 2, 2], padding='SAME', scope='max_pool5')   #(?,1,4,4,512)

            net = tf.reshape(net, [-1, 512 * 4 * 4 * 2])
            net = slim.fully_connected(net, 4096, weights_regularizer=slim.l2_regularizer(0.0005), scope='fc6')
            net = slim.dropout(net, keep_pro, scope='dropout1')
            net = slim.fully_connected(net, 4096, weights_regularizer=slim.l2_regularizer(0.0005), scope='fc7')
            net = slim.dropout(net, keep_pro, scope='dropout2')
            out = slim.fully_connected(net, num_classes, weights_regularizer=slim.l2_regularizer(0.0005), \
                                       activation_fn=None, scope='out')
            out1 = slim.flatten(out)
            out1 = slim.softmax(out1)
            return out1


