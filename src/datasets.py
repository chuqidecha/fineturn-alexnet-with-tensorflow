#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-9-12 下午9:48
# @Author  : yinwb
# @File    : build_tf.py

import tensorflow as tf


def tf_record_num_examples(tf_file):
    count = 0
    for _ in tf.python_io.tf_record_iterator(tf_file):
        count += 1
    return count


def tf_record_parser(image_height, image_width, num_class, image_mean):
    def helper(serilized_example):
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
        image = tf.cast(image, tf.float32)
        if image_mean is not None:
            image = tf.subtract(image, tf.constant(image_mean, shape=[3], name='image_mean', dtype=tf.float32))

        image_clip = tf.image.resize_image_with_crop_or_pad(image, image_height, image_width)

        label_onehot = tf.one_hot(label, num_class)

        return image_clip, label_onehot

    return helper
