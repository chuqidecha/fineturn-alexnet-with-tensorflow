# -*- coding: utf8 -*-
# @Time     : 11/14/18 9:11 PM
# @Author   : yinwb
# @File     : validate.py

import logging
import tensorflow as tf
import numpy as np

from datasets import ImageTFRecordDataset
from datasets import tf_record_parser
from inference import inference
import setting

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT)


def validate(tf_file,
             image_size,
             num_class,
             model_path,
             batch_size=32,
             regularization_rate=0.01,
             image_mean=None):
    with tf.name_scope('input'):
        parser = tf_record_parser(image_size, image_size, num_class,
                                  image_mean=np.load(image_mean), train=False)
        dataset = ImageTFRecordDataset(tf_file, parser, batch_size)
        image_batch, label_batch = dataset.get_next()
        initializer = dataset.initializer()

    # L2 regularization
    regularizer = tf.contrib.layers.l2_regularizer(regularization_rate)
    keep_prob = tf.placeholder(dtype=tf.float32)
    y_halt = inference(image_batch, num_class, keep_prob, regularizer)

    with tf.name_scope('loss'):
        cross_entropy = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_halt, labels=label_batch))
        tf.add_to_collection("losses", cross_entropy)
        loss = tf.add_n(tf.get_collection("losses"))

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_halt, 1), tf.argmax(label_batch, 1))
        accuracy = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))

    saver = tf.train.Saver()
    num_examples = dataset.num_examples()

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        cpkt = tf.train.get_checkpoint_state(model_path)
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
    validate(setting.VALIDATION_TF_RECORD,
             setting.IMAGE_SIZE,
             setting.NUM_CLASS,
             setting.SAVE_MODEL_PATH,
             setting.BATCH_SIZE,
             setting.REGULARIZATION_RATE,
             setting.IMAGE_MEAN_FILE)
