#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-9-14 下午10:35
# @Author  : yinwb
# @File    : setting.py

# layers need to train
TRAIN_LAYERS = ("fc6", "fc7", "fc8")

# number of output class
NUM_CLASS = 21

IMAGE_SIZE = 227
IMAGE_CHANNEL = 3

BATCH_SIZE = 32
EPOCH = 100

LEARNING_RATE_BASE = 0.00001
LEARNING_RATE_DECAY = 0.99

REGULARIZATION_RATE = 0.01
MOVING_AVERAGE_DECAY = 0.99

# train dataset
TRAIN_TF_RECORD = '../data/train.tfrecord'

# validation dataset
VALIDATION_TF_RECORD = '../data/validation.tfrecord'

# testing dataset
TEST_TF_RECORD = '../data/test.tfrecord'

# pretrained model on ImageNet which convert from caffemodel to npy by caffe-tensorflow
PRE_TRAIN_MODLE ='../data/bvlc_alexnet.npy'

# the mean file of train dataset
IMAGE_MEAN_FILE='../data/alexnet.mean.227.npy'

# where to save model
SAVE_MODEL_PATH_NAME ='../data/model/'

# summary dir
SUMMARY_PATH='../data/visual_metrics'
