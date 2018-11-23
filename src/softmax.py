# -*- coding: utf8 -*-
# @Time     : 11/21/18 10:20 PM
# @Author   : yinwb
# @File     : softmax.py

import logging
import tensorflow as tf
import numpy as np
import pandas as pd

from datasets import ImageTFRecordDataset
from datasets import tf_record_parser
from inference import inference
import setting

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT)

classes = ["agricultural", "airplane", "baseballdiamond", "beach", "buildings", "chaparral", "denseresidential",
           "forest", "freeway", "golfcourse", "harbor", "intersection", "mediumresidential", "mobilehomepark",
           "overpass", "parkinglot", "river", "runway", "sparseresidential", "storagetanks", "tenniscourt"]


def softmax(tf_file,
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

    with tf.name_scope('softmax'):
        probilities = tf.nn.softmax(y_halt)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        saver.restore(sess, model_path)
        sess.run(initializer)
        result = pd.DataFrame()
        while True:
            try:
                probilities_, label_batch_ = sess.run([probilities, label_batch],
                                                      feed_dict={keep_prob: 1.0})
                data_frames = [pd.DataFrame(x) for x in
                               [probilities_, np.argmax(label_batch_, axis=1), np.argmax(probilities_, axis=1)]]
                tmp = pd.concat(data_frames, axis=1)
                result = pd.concat([result, tmp], axis=0)
            except tf.errors.OutOfRangeError:
                break
        result.columns = [*classes, "label", "predicted"]
        result.to_csv("../data/result.csv")


if __name__ == "__main__":
    softmax(setting.TEST_TF_RECORD,
            setting.IMAGE_SIZE,
            setting.NUM_CLASS,
            setting.SAVE_MODEL_PATH + "/alexnet-1007",
            setting.BATCH_SIZE,
            setting.REGULARIZATION_RATE,
            setting.IMAGE_MEAN_FILE)
