import tensorflow as tf

def read_and_decode(filename, n_classes, batch_size):
    filename_queue = tf.train.string_input_producer([filename])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw' : tf.FixedLenFeature([], tf.string),
                                       })

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [224, 224, 3])
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    label = tf.cast(features['label'], tf.int64)

    image_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                    batch_size=batch_size,
                                                    capacity=2000,
                                                    min_after_dequeue=1000)

    label_batch = tf.one_hot(label_batch, n_classes)
    label_batch = tf.cast(label_batch, dtype=tf.int64)
    label_batch = tf.reshape(label_batch, [batch_size, n_classes])

    return image_batch, label_batch
