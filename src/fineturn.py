#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-9-14 下午10:39
# @Author  : yinwb
# @File    : fineturn.py

import tensorflow as tf

import numpy as np

from inference import inference
from inference import loss_with_l2
from inference import load_weights_biases
from setting import *




def train():
    x = tf.placeholder(tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNEL], name="x-input")
    y = tf.placeholder(tf.float32, [None, NUM_CLASS], name="y-input")

    # 计算L2正则化损失函数
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)

    # 计算在当前参数下网络前向传播的结果。这里不使用滑动平均
    y_halt = inference(x, regularizer=regularizer)

    # 带l2正则的交叉熵损失
    loss = loss_with_l2(y_halt, y)

    # 定义存储训练轮数的变量，并将其指定为不可训练的。在使用TensorFlow训练神经网络时，一般将代表轮数的变量指定为不可训练
    global_step = tf.Variable(0, trainable=False)

    # 用滑动平均衰减率和训练轮数变量初始化滑动平均类。给定训练轮数可以加快训练早期变量更新速度
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)

    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,  # 基础学习率,学习率在此基础上随着迭代次数而减小
        global_step,  # 当前迭代的次数
        np.ceil(TRAIN_NUM_EXAMPLES / BATCH_SIZE),  # 一轮迭代所需要的迭代次数
        LEARNIING_RATE_DECAY  # 学习率的衰减率
    )
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # 在训练神经网络模型时,每过一遍数据既需要通过反向传播算法来跟新神经网络中的参数,又要更新每一个参数的滑动平均值.
    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name="train_op")

    # 初始化持久化类
    saver = tf.train.Saver()


    with tf.Session() as sess:
        load_weights_biases(sess)



