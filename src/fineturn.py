# -*- coding: utf-8 -*-
# @Time    : 18-9-14 10:39 pm
# @Author  : yinwb
# @File    : fineturn.py
import os
import logging
import math

import numpy as np
import tensorflow as tf

from datasets import ImageTFRecordDataset
from datasets import tf_record_parser
from inference import inference, load_weights_biases, trainable_variable_summaries

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT)


def fineturn(tf_file,
             pretrain_model,
             image_size,
             num_class,
             train_layers,
             save_model_path,
             summary_path,
             batch_size=32,
             epoch=20,
             regularization_rate=0.01,
             learning_rate_base=1e-4,
             learning_rate_decay=0.99,
             image_mean=None):
    '''
    fineturn and validation
    '''

    with tf.name_scope('input'):
        parser = tf_record_parser(image_size, image_size, num_class, image_mean)
        dataset = ImageTFRecordDataset(tf_file, parser, batch_size)
        image_batch, label_batch = dataset.get_next()
        initializer = dataset.initializer()

    # L2 regularization
    regularizer = tf.contrib.layers.l2_regularizer(regularization_rate)

    keep_prob = tf.placeholder(dtype=tf.float32)

    # forward propagation
    y_halt = inference(image_batch, num_class, keep_prob, regularizer=regularizer)

    # learning rate
    global_step = tf.Variable(0, trainable=False)
    train_batches_per_epoch = math.ceil(dataset.num_examples() / batch_size)
    learning_rate = tf.train.exponential_decay(
        learning_rate_base,
        global_step,
        train_batches_per_epoch,
        learning_rate_decay
    )

    with tf.name_scope('loss'):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_halt, labels=label_batch))
        tf.add_to_collection("losses", cross_entropy)
        loss = tf.add_n(tf.get_collection("losses"))

        tf.summary.scalar("cross-entropy", cross_entropy)
        tf.summary.scalar("loss", loss)


    with tf.name_scope('train'):
        trainable_variables = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step,
                                                                               var_list=trainable_variables)

    saver = tf.train.Saver(max_to_keep=epoch)

    trainable_variable_summaries()
    summary_all = tf.summary.merge_all()

    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter(summary_path, sess.graph)
        init = tf.global_variables_initializer()
        sess.run(init)
        load_weights_biases(sess, pretrain_model, train_layers)
        for epoch_ in range(epoch):
            sess.run(initializer)
            try:
                while True:
                    summary, loss_, global_step_, _ = sess.run([summary_all, loss, global_step, train_step],
                                                               feed_dict={keep_prob: 0.5})
                    logging.info('%5d step(s), loss on current training batch is %5.3f' % (global_step_, loss_))
                    summary_writer.add_summary(summary, global_step_)
            except tf.errors.OutOfRangeError:
                saver.save(sess, save_model_path, global_step=global_step)

if __name__ == '__main__':
    import setting

    image_mean = np.load(setting.IMAGE_MEAN_FILE)

    fineturn(setting.TRAIN_TF_RECORD,
             setting.PRE_TRAIN_MODLE,
             setting.IMAGE_SIZE,
             setting.NUM_CLASS,
             setting.TRAIN_LAYERS,
             os.path.join(setting.SAVE_MODEL_PATH,setting.SAVE_MODEL_NAME),
             setting.SUMMARY_PATH,
             setting.BATCH_SIZE,
             setting.EPOCH,
             setting.REGULARIZATION_RATE,
             setting.LEARNING_RATE_BASE,
             setting.LEARNING_RATE_DECAY,
             image_mean)
