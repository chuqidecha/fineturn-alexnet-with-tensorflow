#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-9-12 下午9:48
# @Author  : yinwb
# @File    : build_tf.py

import argparse
import os

import numpy as np
import tensorflow as tf
import cv2


def _int64_feature(value):
    '''
    convert int to tf.train.Feature
    :param value:
    :return:
    '''

    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    '''
    convert byte array to tf.train.Feature
    :param value:
    :return:
    '''

    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert(fname, root_dir, output):
    file_labels = np.loadtxt(fname, dtype='str')

    writer = tf.python_io.TFRecordWriter(output)
    for file, label in file_labels:
        img = cv2.imread(os.path.join(root_dir, file))
        height, width, channel = img.shape
        img_raw = img.tostring()

        feature = {
            'label': _int64_feature(int(label)),
            'height': _int64_feature(height),
            'width': _int64_feature(width),
            'channel': _int64_feature(channel),
            'img_raw': _bytes_feature(img_raw)
        }

        example = tf.train.Example(features=tf.train.Features(feature=feature))

        writer.write(example.SerializeToString())

    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="a text file contains image list ", type=str)
    parser.add_argument("root_dir", help="root path", type=str)
    parser.add_argument("output", help="output path ", type=str)

    args = parser.parse_args()
    print(args)
    convert(args.file, args.root_dir, args.output)
