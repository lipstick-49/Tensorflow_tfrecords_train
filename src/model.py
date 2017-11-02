import tensorflow as tf
import numpy as np

def _new(images, classes):

    # conv1_1
    with tf.name_scope('conv1_1') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 3, 16], dtype=tf.float32,stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[16], dtype=tf.float32),trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv1_1 = tf.nn.relu(out, name=scope)

    # pool1
    pool1 = tf.nn.max_pool(conv1_1,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME',name='pool1')

    # conv2_1
    with tf.name_scope('conv2_1') as scope:
        kernel = tf.Variable(tf.truncated_normal([3,3,16,16], dtype=tf.float32,stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[16], dtype=tf.float32),trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv2_1 = tf.nn.relu(out, name=scope)

    # pool2
    pool2 = tf.nn.max_pool(conv2_1,ksize=[1,3,3,1], strides=[1,1,1,1],padding='SAME',name='pool2')

    # fc1
    with tf.name_scope('fc1') as scope:
        shape = int(np.prod(pool2.get_shape()[1:]))
        fc1w = tf.Variable(tf.truncated_normal([shape, 128],dtype=tf.float32,stddev=1e-1), name='weights')
        fc1b = tf.Variable(tf.constant(1.0, shape=[128], dtype=tf.float32),trainable=True, name='biases')
        pool5_flat = tf.reshape(pool2, [-1, shape])
        fc1l = tf.nn.bias_add(tf.matmul(pool5_flat, fc1w), fc1b)
        fc1 = tf.nn.relu(fc1l)

    # fc2
    with tf.name_scope('fc2') as scope:
        fc2w = tf.Variable(tf.truncated_normal([128, 128],dtype=tf.float32,stddev=1e-1), name='weights')
        fc2b = tf.Variable(tf.constant(1.0, shape=[128], dtype=tf.float32),trainable=True, name='biases')
        fc2l = tf.nn.bias_add(tf.matmul(fc1, fc2w), fc2b)
        fc2 = tf.nn.relu(fc2l)

    # fc3
    with tf.name_scope('fc3') as scope:
        fc3w = tf.Variable(tf.truncated_normal([128, classes],dtype=tf.float32,stddev=1e-1), name='weights')
        fc3b = tf.Variable(tf.constant(1.0, shape=[classes], dtype=tf.float32),trainable=True, name='biases')
        fc3l = tf.nn.bias_add(tf.matmul(fc2, fc3w), fc3b)
        prediction = tf.nn.softmax(fc3l)

    return fc3l
