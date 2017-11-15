import tensorflow as tf
import numpy as np
import os
import Net
import read_tfrecords
import vgg16

logs_train_dir = 'log_train'
logs_val_dir = 'log_val'

train_batch, train_label_batch = read_tfrecords.read_and_decode('tfrecords/tarin.tfrecords', 21, 16)
val_batch, val_label_batch = read_tfrecords.read_and_decode('tfrecords/val.tfrecords', 21, 16)

X = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3])
Y = tf.placeholder(dtype=tf.int64, shape=[None, 21])

logits = Net._new(X, 21)
# logits = vgg16.vgg16(X, 21)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y,logits=logits))
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
correct_prediction = tf.equal(tf.argmax(Y,1),tf.argmax(logits,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

summary_op = tf.summary.merge_all()
saver = tf.train.Saver()

with tf.Session() as sess:
    train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
    val_writer = tf.summary.FileWriter(logs_val_dir, sess.graph)
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    for step in np.arange(10000):
        if coord.should_stop():
            break
        train_images, train_labels = sess.run([train_batch, train_label_batch])
        val_images, val_labels = sess.run([val_batch, val_label_batch])
        _ , train_loss, train_acc = sess.run([train_op, loss, accuracy], feed_dict={X: train_images, Y : train_labels})
        val_loss, val_acc = sess.run([loss, accuracy], feed_dict={X: val_images, Y: val_labels})
        if step % 50 == 0:
            print('Step %d, loss %f, acc %.2f%% --- * val_loss %f, val_acc %.2f%%' % (step, train_loss ,train_acc * 100.0, val_loss ,val_acc * 100.0))
            summary_str = sess.run(summary_op)
            train_writer.add_summary(summary_str, step)
            val_writer.add_summary(summary_str, step)

        checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)
        
    coord.request_stop()
    coord.join(threads)
