import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib.image as mpimg
#import urllib3
import tensorflow_datasets as tfds

#  urllib3.disable_warnings()
keras=tf.keras
tfds.disable_progress_bar()

(raw_train, raw_validation, raw_test), metadata= tfds.load(
    'cats_vs_dogs',
    split=[tfds.Split.TRAIN.subsplit(tfds.percent[:80]),
    tfds.Split.TRAIN.subsplit(tfds.percent[80:90]),
    tfds.Split.TRAIN.subsplit(tfds.percent[90:])],
    with_info=True,
    as_supervised=True
)

get_label_name = metadata.features['label'].int2str


IMG_SIZE = 160
def format_example(image, label):
  image = tf.cast(image, tf.float32)
  image = (image/127.5)-1
  image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
  return image, label

train = raw_train.map(format_example)
validation = raw_validation.map(format_example)
test=raw_test.map(format_example)


model = keras.models.load_model('dogs_vs_cats.h5')

