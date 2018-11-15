# -*- coding: utf8 -*-
# @Time     : 11/14/18 9:11 PM
# @Author   : yinwb
# @File     : validate.py

import logging
import tensorflow as tf
import numpy as np

import setting
from datasets import ImageTFRecordDataset
from datasets import tf_record_parser
from inference import inference

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT)


def evaldate():
    # with tf.Graph().as_default() as g:
    with tf.name_scope('input'):
        parser = tf_record_parser(setting.IMAGE_SIZE, setting.IMAGE_SIZE, setting.NUM_CLASS,
                                  image_mean=np.load(setting.IMAGE_MEAN_FILE), train=False)
        dataset = ImageTFRecordDataset(setting.VALIDATION_TF_RECORD, parser, setting.BATCH_SIZE)
        image_batch, label_batch = dataset.get_next()
        initializer = dataset.initializer()

    # L2 regularization
    regularizer = tf.contrib.layers.l2_regularizer(setting.REGULARIZATION_RATE)
    keep_prob = tf.placeholder(dtype=tf.float32)
    y_halt = inference(image_batch, setting.NUM_CLASS, keep_prob, regularizer)

    with tf.name_scope('loss'):
        cross_entropy = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_halt, labels=label_batch))
        tf.add_to_collection("losses", cross_entropy)
        loss = tf.add_n(tf.get_collection("losses"))

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_halt, 1), tf.argmax(label_batch, 1))
        accuracy = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))

    saver = tf.train.Saver()
    num_examples = dataset.num_examples()
    print(num_examples)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        cpkt = tf.train.get_checkpoint_state('../data/model')
        if cpkt is not None and len(cpkt.all_model_checkpoint_paths) > 0:
            for model_checkpoint_path in cpkt.all_model_checkpoint_paths:
                # 获取迭代的轮数
                glob_step = model_checkpoint_path.split('/')[-1].split('-')[-1]
                # 加载模型参数
                saver.restore(sess, model_checkpoint_path)

                total_loss = 0
                total_accuracy = 0
                sess.run(initializer)
                while True:
                    try:
                        loss_, accuracy_ = sess.run([loss, accuracy], feed_dict={keep_prob: 1.0})
                        total_loss += loss_
                        total_accuracy += accuracy_
                    except tf.errors.OutOfRangeError:
                        break
                logging.info('%4s step(s), loss is %5.3f, average is %1.3f' % (
                    glob_step, total_loss / num_examples, total_accuracy / num_examples))


if __name__ == "__main__":
    evaldate()
