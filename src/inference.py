# -*- coding: utf-8 -*-
# @Time    : 18-9-14 10:39 pm
# @Author  : yinwb
# @File    : inference.py

import numpy as np
import tensorflow as tf


def load_weights_biases(sess, pre_train_model, train_layers):
    '''
    从预训练模型中加载参数，跳过需要训练的层
    :param sess:
    :param pre_train_model:
    :param train_layers:
    :return:
    '''
    weights_dict = np.load(pre_train_model, encoding='bytes').item()
    for op_name in weights_dict:
        with tf.variable_scope(op_name, reuse=True):
            if op_name not in train_layers:
                for item in weights_dict[op_name]:
                    if len(item.shape) == 1:
                        biases = tf.get_variable("biases")
                        sess.run(biases.assign(item))
                    else:
                        weights = tf.get_variable("weights")
                        sess.run(weights.assign(item))


def trainable_variable_summaries():
    '''
    日志
    :param var:
    :param name:
    :return:
    '''
    with tf.name_scope("params"):
        for variable in tf.trainable_variables():
            name = variable.name.split(':')[0]
            tf.summary.histogram(name, variable)
            mean = tf.reduce_mean(variable)
            tf.summary.scalar("mean/" + name, mean)
            stddev = tf.sqrt(tf.reduce_mean(tf.square(variable - mean)))
            tf.summary.scalar("stddev/" + name, stddev)


def _conv_with_groups(name_scope, xs, ws, groups, strides, padding):
    '''
    模拟多个GPU
    :param name_scope: 命名空间
    :param xs: 输入Tensor
    :param ws: 权值Tensor
    :param groups: GPU数目
    :param strides: 步长
    :param padding: 边缘填充方式
    :return:
    '''
    with tf.name_scope(name_scope):
        ws_groups = tf.split(value=ws, num_or_size_splits=groups, axis=3)
        xs_groups = tf.split(value=xs, num_or_size_splits=groups, axis=3)
        conv_groups = [tf.nn.conv2d(x, w, strides, padding=padding) for w, x in zip(ws_groups, xs_groups)]
        conv = tf.concat(values=conv_groups, axis=3)
    return conv


def inference(input_tensor, output_dim, keep_prob, regularizer=None):
    '''
    AlexNet模型实现
    :param input_tensor: 输入[None,227,227,3]
    :param output_dim: 分类数
    :param keep_prob: dropout概率
    :param regularizer: 正则化项
    :return:
    '''
    with tf.variable_scope("conv1"):
        weights = tf.get_variable('weights', [11, 11, 3, 96], initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable('biases', [96], initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.bias_add(tf.nn.conv2d(input_tensor, weights, [1, 4, 4, 1], padding="VALID"), biases)

    with tf.name_scope("relu1"):
        relu1 = tf.nn.relu(conv1)

    with tf.name_scope("lrn1"):
        lrn1 = tf.nn.lrn(relu1, depth_radius=2, bias=1, alpha=0.00002, beta=0.75)
    with tf.name_scope("pool1"):
        pool1 = tf.nn.max_pool(lrn1, [1, 3, 3, 1], [1, 2, 2, 1], padding="VALID")

    with tf.variable_scope("conv2"):
        weights = tf.get_variable('weights', [5, 5, 48, 256], initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable('biases', [256], initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.bias_add(_conv_with_groups("conv2-groups", pool1, weights, 2, [1, 1, 1, 1], padding="SAME"),
                               biases)

    with tf.name_scope("relu2"):
        relu2 = tf.nn.relu(conv2)
    with tf.name_scope("lrn2"):
        lrn2 = tf.nn.lrn(relu2, depth_radius=2, bias=1, alpha=0.00002, beta=0.75)

    with tf.name_scope("pool2"):
        pool2 = tf.nn.max_pool(lrn2, [1, 3, 3, 1], [1, 2, 2, 1], padding="VALID")

    with tf.variable_scope("conv3"):
        weights = tf.get_variable('weights', [3, 3, 256, 384], initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable('biases', [384], initializer=tf.constant_initializer(0.0))
        conv3 = tf.nn.bias_add(tf.nn.conv2d(pool2, weights, [1, 1, 1, 1], padding="SAME"), biases)

    with tf.name_scope("relu3"):
        relu3 = tf.nn.relu(conv3)

    with tf.variable_scope("conv4"):
        weights = tf.get_variable('weights', [3, 3, 192, 384], initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable('biases', [384], initializer=tf.constant_initializer(0.0))
        conv4 = tf.nn.bias_add(_conv_with_groups("conv4-groups", relu3, weights, 2, [1, 1, 1, 1], padding="SAME"),
                               biases)

    with tf.name_scope("relu4"):
        relu4 = tf.nn.relu(conv4)

    with tf.variable_scope("conv5"):
        weights = tf.get_variable('weights', [3, 3, 192, 256], initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable('biases', [256], initializer=tf.constant_initializer(0.0))
        conv5 = tf.nn.bias_add(_conv_with_groups("conv5-groups", relu4, weights, 2, [1, 1, 1, 1], padding="SAME"),
                               biases)

    with tf.name_scope("relu5"):
        relu5 = tf.nn.relu(conv5)

    with tf.name_scope("pool5"):
        pool5 = tf.nn.max_pool(relu5, [1, 3, 3, 1], [1, 2, 2, 1], padding="VALID")

    with tf.variable_scope("fc6"):
        weights = tf.get_variable('weights', [9216, 4096], initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable('biases', [4096], initializer=tf.constant_initializer(0.0))
        flattened = tf.reshape(pool5, [-1, 6 * 6 * 256])
        fc6 = tf.nn.xw_plus_b(flattened, weights, biases)
        if regularizer is not None:
            tf.add_to_collection("losses", regularizer(weights))

    with tf.name_scope("relu6"):
        relu6 = tf.nn.relu(fc6)

    with tf.name_scope("dropout6"):
        relu6 = tf.nn.dropout(relu6, keep_prob)

    with tf.variable_scope("fc7"):
        weights = tf.get_variable('weights', [4096, 4096], initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable('biases', [4096], initializer=tf.constant_initializer(0.0))
        fc7 = tf.nn.xw_plus_b(relu6, weights, biases)
        if regularizer is not None:
            tf.add_to_collection("losses", regularizer(weights))

    with tf.name_scope("relu7"):
        relu7 = tf.nn.relu(fc7)

    with tf.name_scope("dropout7"):
        relu7 = tf.nn.dropout(relu7, keep_prob)

    with tf.variable_scope("fc8"):
        weights = tf.get_variable('weights', [4096, output_dim],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        if regularizer is not None:
            tf.add_to_collection("losses", regularizer(weights))
        fc8 = tf.nn.xw_plus_b(relu7, weights, biases)

    return fc8
