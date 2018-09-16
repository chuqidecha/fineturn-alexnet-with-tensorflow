#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-9-12 下午9:48
# @Author  : yinwb
# @File    : build_tf.py

import tensorflow as tf
from setting import *


def tf_record_parser(serilized_example):
    features = tf.parse_single_example(
        serilized_example,
        features={
            'label': tf.FixedLenFeature([], tf.int64),
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'channel': tf.FixedLenFeature([], tf.int64),
            'img_raw': tf.FixedLenFeature([], tf.string),
        })

    image = tf.decode_raw(features['img_raw'], tf.uint8)
    label = tf.cast(features['label'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    height = tf.cast(features['height'], tf.int32)
    channel = tf.cast(features['channel'], tf.int32)
    image_shape = tf.stack([height, width, channel])
    image = tf.reshape(image, image_shape)

    image_clip = tf.image.resize_image_with_crop_or_pad(image, IMAGE_SIZE, IMAGE_SIZE)

    label_onehot = tf.one_hot(label, NUM_CLASS)

    return image_clip, label_onehot


def next_batch(batch_size):
    pass


if __name__ == "__main__":

    dataset = tf.data.TFRecordDataset("../data/test.tfrecord").map(tf_record_parser, num_parallel_calls=4)
    dataset = dataset.repeat(1).shuffle(100).batch(100)
    init = tf.global_variables_initializer()

    iter = dataset.make_initializable_iterator()
    image_batch, label_batch = iter.get_next()

    print()

    images = tf.image.random_flip_left_right(image_batch)

    with tf.Session() as sess:
        for i in range(100):
            sess.run(iter.initializer)
            while True:
                try:
                    iamges_ = sess.run(images)
                    print(iamges_.shape)
                except tf.errors.OutOfRangeError:
                    print("i={0}".format(i))
                    break