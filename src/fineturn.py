#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-9-14 下午10:39
# @Author  : yinwb
# @File    : fineturn.py
import logging

import numpy as np
import tensorflow as tf

from datasets import tf_record_num_examples
from datasets import tf_record_parser
from inference import inference
from inference import loss_with_l2
from setting import *

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT)


def load_weights_biases(sess, pre_train_model):
    weights_dict = np.load(pre_train_model, encoding='bytes').item()

    for op_name in weights_dict:
        with tf.variable_scope(op_name, reuse=True):
            if op_name not in TRAIN_LAYERS:
                for item in weights_dict[op_name]:
                    if len(item.shape) == 1:

                        biases = tf.get_variable("biases", trainable=False)
                        sess.run(biases.assign(item))
                    else:
                        weights = tf.get_variable("weights", trainable=False)
                        sess.run(weights.assign(item))


def train_val(train_tf_file, validation_tf_file):
    train_num_examples = tf_record_num_examples(train_tf_file)
    val_num_examples = tf_record_num_examples(validation_tf_file)

    with tf.name_scope('pre-processing'):
        paser = tf_record_parser(IMAGE_SIZE, IMAGE_SIZE, NUM_CLASS, IMAGE_MEAN)
        train_dataset = tf.data.TFRecordDataset(train_tf_file).map(paser)
        train_dataset = train_dataset.repeat(None).shuffle(1024).batch(BATCH_SIZE)
        validation_dataset = tf.data.TFRecordDataset(validation_tf_file).map(paser)
        validation_dataset = validation_dataset.repeat(1).shuffle(1024).batch(BATCH_SIZE)
        iter = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
        image_batch, label_batch = iter.get_next()

        training_init_op = iter.make_initializer(train_dataset)
        validation_init_op = iter.make_initializer(validation_dataset)

    # 计算L2正则化损失函数
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)

    # 计算在当前参数下网络前向传播的结果
    y_halt = inference(image_batch, regularizer=regularizer)

    # 带l2正则的交叉熵损失
    loss = loss_with_l2(y_halt, label_batch)

    # 训练次数
    global_step = tf.Variable(0, trainable=False)

    train_batches_per_epoch = train_num_examples // BATCH_SIZE

    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,  # 基础学习率,学习率在此基础上随着迭代次数而减小
        global_step,  # 当前迭代的次数
        train_batches_per_epoch,  # 一轮迭代所需要的迭代次数
        LEARNIING_RATE_DECAY  # 学习率的衰减率
    )
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    # 初始化持久化类
    saver = tf.train.Saver(max_to_keep=EPOCH)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_halt, 1), tf.argmax(label_batch, 1))
        correct_count = tf.reduce_sum(tf.cast(correct_prediction, tf.int32))

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        # load_weights_biases(sess, PRE_TRAIN_MODLE)
        for epoch in range(EPOCH):
            sess.run(training_init_op)
            for batch in range(train_batches_per_epoch):
                loss_, _ = sess.run([loss, train_step])
                logging.info(
                    'after {0} step(s),training loss on batch is {1:.5f}'.format(
                        epoch * train_batches_per_epoch + batch,
                        loss_))
            saver.save(sess, SAVE_MODEL_PATH_NAME, global_step=global_step)
            sess.run(validation_init_op)
            loss_ = 0
            validation_correct_count = 0
            while True:
                try:
                    batch_loss, batch_correct_count = sess.run([loss, correct_count])
                    loss_ += batch_loss
                    validation_correct_count += batch_correct_count
                    logging.info(
                        'after {0} epoch(s),loss on validation is {1:.5f},and accuracy is {2:.5f}'.format(
                            epoch, loss_, correct_count / val_num_examples))
                except tf.errors.OutOfRangeError:
                    break


if __name__ == '__main__':
    train_val(TRAIN_TF_RECORD, VALIDATION_TF_RECORD)
