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
                        biases = tf.get_variable("biases")
                        sess.run(biases.assign(item))
                    else:
                        weights = tf.get_variable("weights")
                        sess.run(weights.assign(item))


def train_val(train_tf_file, validation_tf_file):
    train_num_examples = tf_record_num_examples(train_tf_file)
    val_num_examples = tf_record_num_examples(validation_tf_file)

    with tf.name_scope('pre-processing'):
        image_mean = np.load(IMAGE_MEAN_FILE)
        train_paser = tf_record_parser(IMAGE_SIZE, IMAGE_SIZE, NUM_CLASS, image_mean)
        train_dataset = tf.data.TFRecordDataset(train_tf_file).map(train_paser)
        train_dataset = train_dataset.repeat(None).shuffle(1024).batch(BATCH_SIZE)
        validation_parse = tf_record_parser(IMAGE_SIZE, IMAGE_SIZE, NUM_CLASS, image_mean, train=False)
        validation_dataset = tf.data.TFRecordDataset(validation_tf_file).map(validation_parse)
        validation_dataset = validation_dataset.repeat(1).shuffle(1024).batch(2100)
        iter = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
        image_batch, label_batch = iter.get_next()

        training_init_op = iter.make_initializer(train_dataset)
        validation_init_op = iter.make_initializer(validation_dataset)

    # L2 regularization
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)

    keep_prob = tf.placeholder(dtype=tf.float32)

    # forward propagation
    y_halt = inference(image_batch, NUM_CLASS, keep_prob, regularizer=regularizer)

    # learning rate
    global_step = tf.Variable(0, trainable=False)
    train_batches_per_epoch = train_num_examples // BATCH_SIZE
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        train_batches_per_epoch,
        LEARNING_RATE_DECAY
    )

    # cross entropy loss with L2 regularization
    with tf.name_scope("loss"):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_halt, labels=label_batch))
        tf.add_to_collection("losses", cross_entropy)
        loss = tf.add_n(tf.get_collection("losses"))

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_halt, 1), tf.argmax(label_batch, 1))
        score = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.name_scope('train'):
        var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in TRAIN_LAYERS]

        # Get gradients of all trainable variables
        gradients = tf.gradients(loss, var_list)
        gradients = list(zip(gradients, var_list))

        # Create optimizer and apply gradient descent to the trainable variables
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train_op = optimizer.apply_gradients(grads_and_vars=gradients, global_step=global_step)
        # train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    saver = tf.train.Saver(max_to_keep=EPOCH)

    merged = tf.summary.merge_all()

    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter(SUMMARY_PATH, sess.graph)
        init = tf.global_variables_initializer()
        sess.run(init)
        # load_weights_biases(sess, PRE_TRAIN_MODLE)
        for epoch in range(EPOCH):
            sess.run(training_init_op)
            for batch in range(train_batches_per_epoch):
                summary, loss_, _ = sess.run([merged, loss, train_op], feed_dict={keep_prob: 0.5})
                logging.info(
                    'after {0} step(s),training loss on batch is {1:.5f}'.format(
                        epoch * train_batches_per_epoch + batch,
                        loss_))
                summary_writer.add_summary(summary, epoch * train_batches_per_epoch + batch)
            saver.save(sess, SAVE_MODEL_PATH_NAME, global_step=global_step)
            sess.run(validation_init_op)

            while True:
                try:
                    loss_, score_ = sess.run([loss, score], feed_dict={keep_prob: 1.0})
                except tf.errors.OutOfRangeError:
                    logging.info(
                        'after {0} epoch(s),loss on validation is {1:.5f},and accuracy is {2:.5f}'.format(epoch, loss_,
                                                                                                          score_))
                    break


if __name__ == '__main__':
    train_val(TRAIN_TF_RECORD, VALIDATION_TF_RECORD)
