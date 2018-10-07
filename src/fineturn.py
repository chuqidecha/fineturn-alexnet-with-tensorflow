# -*- coding: utf-8 -*-
# @Time    : 18-9-14 10:39 pm
# @Author  : yinwb
# @File    : fineturn.py
import logging

import numpy as np
import tensorflow as tf

from datasets import tf_record_num_examples
from datasets import tf_record_parser
from inference import inference

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT)


def load_weights_biases(sess, pre_train_model, train_layers):
    '''
    load pretrain weights and biases
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


def fineturn(train_tf_file,
             validation_tf_file,
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
    train_num_examples = tf_record_num_examples(train_tf_file)
    validation_num_examples = tf_record_num_examples(validation_tf_file)

    with tf.name_scope('input'):
        train_paser = tf_record_parser(image_size, image_size, num_class, image_mean)
        train_dataset = tf.data.TFRecordDataset(train_tf_file).map(train_paser)
        train_dataset = train_dataset.repeat(None).shuffle(256).batch(batch_size)
        validation_parse = tf_record_parser(image_size, image_size, num_class, image_mean, train=False)
        validation_dataset = tf.data.TFRecordDataset(validation_tf_file).map(validation_parse)
        validation_dataset = validation_dataset.repeat(1).shuffle(256).batch(batch_size)
        iter = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
        image_batch, label_batch = iter.get_next()

        train_init_op = iter.make_initializer(train_dataset)
        validation_init_op = iter.make_initializer(validation_dataset)

    # L2 regularization
    regularizer = tf.contrib.layers.l2_regularizer(regularization_rate)

    keep_prob = tf.placeholder(dtype=tf.float32)
    summary_type = tf.placeholder(dtype=tf.string)

    # forward propagation
    y_halt = inference(image_batch, num_class, keep_prob, regularizer=regularizer)

    # learning rate
    global_step = tf.Variable(0, trainable=False)
    train_batches_per_epoch = train_num_examples // batch_size
    learning_rate = tf.train.exponential_decay(
        learning_rate_base,
        global_step,
        train_batches_per_epoch,
        learning_rate_decay
    )

    # cross entropy loss with L2 regularization
    with tf.name_scope("loss"):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_halt, labels=label_batch))
        tf.add_to_collection("losses", cross_entropy)
        loss = tf.add_n(tf.get_collection("losses"))

    with tf.name_scope('accuracy'):
        prediction = tf.equal(tf.argmax(y_halt, 1), tf.argmax(label_batch, 1))
        score = tf.reduce_sum(tf.cast(prediction, tf.int32))

    with tf.name_scope('train'):
        var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]

        # Get gradients of all trainable variables
        gradients = tf.gradients(loss, var_list)
        gradients = list(zip(gradients, var_list))

        # Create optimizer and apply gradient descent to the trainable variables
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train_op = optimizer.apply_gradients(grads_and_vars=gradients, global_step=global_step)

    saver = tf.train.Saver(max_to_keep=epoch)

    merged = tf.summary.merge_all()

    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter(summary_path, sess.graph)
        init = tf.global_variables_initializer()
        sess.run(init)
        load_weights_biases(sess, pretrain_model, train_layers)
        for epoch_ in range(epoch):
            sess.run(train_init_op)
            for batch in range(train_batches_per_epoch):
                summary, loss_, _ = sess.run([merged, loss, train_op], feed_dict={keep_prob: 0.5})
                logging.info(
                    'after {0} step(s), loss on current training batch is {1:.5f}'.format(
                        epoch_ * train_batches_per_epoch + batch, loss_))
                summary_writer.add_summary(summary, epoch * train_batches_per_epoch + batch)
            saver.save(sess, save_model_path, global_step=global_step)

            total_score = 0
            total_loss = 0
            count = 0
            sess.run(validation_init_op)
            while True:
                loss_, score_ = sess.run([loss, score], feed_dict={keep_prob: 1.0})
                count += batch_size
                total_score += score_
                if count > validation_num_examples:
                    total_loss += loss_ * (validation_num_examples + batch_size - count)
                    logging.info('after {0} epoch(s) ,'
                                 'loss on validation is {1:.5f},'
                                 'accuracy is {2:.5f}'.format(epoch_,
                                                              total_loss / validation_num_examples,
                                                              total_score / validation_num_examples))
                    break
                else:
                    total_loss += loss_ * batch_size


if __name__ == '__main__':
    import setting

    image_mean = np.load(setting.IMAGE_MEAN_FILE)

    fineturn(setting.TRAIN_TF_RECORD,
             setting.VALIDATION_TF_RECORD,
             setting.PRE_TRAIN_MODLE,
             setting.IMAGE_SIZE,
             setting.NUM_CLASS,
             setting.TRAIN_LAYERS,
             setting.SAVE_MODEL_PATH_NAME,
             setting.SUMMARY_PATH,
             setting.BATCH_SIZE,
             setting.EPOCH,
             setting.REGULARIZATION_RATE,
             setting.LEARNING_RATE_BASE,
             setting.LEARNING_RATE_DECAY,
             image_mean)
