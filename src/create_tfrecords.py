import os
import tensorflow as tf
from PIL import Image

file_dir = ''

def create_tfrecords():
    writer = tf.python_io.TFRecordWriter("tfrecords/tarin.tfrecords")
    list = os.listdir(file_dir)
    for index, name in enumerate(list):
        class_path = file_dir + name + '/'
        for img_name in os.listdir(class_path):
            img_path = class_path + img_name
            img = Image.open(img_path)
            if img.mode != "RGB":
                continue
            img = img.resize((224, 224))
            img_raw = img.tobytes()
            example = tf.train.Example(features=tf.train.Features(feature={
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            }))
            writer.write(example.SerializeToString())
    writer.close()

create_tfrecords()