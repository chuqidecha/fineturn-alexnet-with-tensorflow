import tensorflow as tf

from setting import *


def inference(input_tensor, train=True, regularizer=None):
    with tf.variable_scope("conv1"):
        weights = tf.get_variable('weights', [11, 11, 3, 96], initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable('biases', [96], initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(input_tensor, weights, [1, 4, 4, 1], padding="VALID")

    relu1 = tf.nn.relu(tf.nn.bias_add(conv1, biases))
    lrn1 = tf.nn.lrn(relu1, depth_radius=5, bias=2, alpha=0.0001, beta=0.75)
    pool1 = tf.nn.max_pool(lrn1, [1, 3, 3, 1], [1, 2, 2, 1], padding="VALID")

    with tf.variable_scope("conv2"):
        weights = tf.get_variable('weights', [5, 5, 96, 256], initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable('biases', [256], initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(pool1, weights, [1, 1, 1, 1], padding="SAME")

    relu2 = tf.nn.relu(tf.nn.bias_add(conv2, biases))
    lrn2 = tf.nn.lrn(relu2, depth_radius=5, bias=2, alpha=0.0001, beta=0.75)
    pool2 = tf.nn.max_pool(lrn2, [1, 3, 3, 1], [1, 2, 2, 1], padding="VALID")

    with tf.variable_scope("conv3"):
        weights = tf.get_variable('weights', [3, 3, 256, 384], initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable('biases', [384], initializer=tf.constant_initializer(0.0))

    conv3 = tf.nn.conv2d(pool2, weights, [1, 1, 1, 1], padding="SAME")
    relu3 = tf.nn.relu(tf.nn.bias_add(conv3, biases))

    with tf.variable_scope("conv4"):
        weights = tf.get_variable('weights', [3, 3, 384, 384], initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable('biases', [384], initializer=tf.constant_initializer(0.0))

    conv4 = tf.nn.conv2d(relu3, weights, [1, 1, 1, 1], padding="SAME")
    relu4 = tf.nn.relu(tf.nn.bias_add(conv4, biases))

    with tf.variable_scope("conv5"):
        weights = tf.get_variable('weights', [3, 3, 384, 256], initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable('biases', [256], initializer=tf.constant_initializer(0.0))
        conv5 = tf.nn.conv2d(relu4, weights, [1, 1, 1, 1], padding="SAME")

    relu5 = tf.nn.relu(tf.nn.bias_add(conv5, biases))
    pool5 = tf.nn.max_pool(relu5, [1, 3, 3, 1], [1, 2, 2, 1], padding="VALID")

    with tf.variable_scope("fc6"):
        weights = tf.get_variable('weights', [9216, 4096], initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable('biases', [4096], initializer=tf.constant_initializer(0.0))
        flattened = tf.reshape(pool5, [-1, 6 * 6 * 256])
        fc6 = tf.nn.xw_plus_b(flattened, weights, biases)
        if regularizer is not None:
            tf.add_to_collection("losses", regularizer(weights))

    relu6 = tf.nn.relu(fc6)
    if train:
        relu6 = tf.nn.dropout(relu6, 0.5)

    with tf.variable_scope("fc7"):
        weights = tf.get_variable('weights', [4096, 4096], initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable('biases', [4096], initializer=tf.constant_initializer(0.0))
        fc7 = tf.nn.xw_plus_b(relu6, weights, biases)
        if regularizer is not None:
            tf.add_to_collection("losses", regularizer(weights))

    relu7 = tf.nn.relu(fc7)
    if train:
        relu7 = tf.nn.dropout(relu7, 0.5)

    with tf.variable_scope("fc8"):
        weights = tf.get_variable('weights', [4096, NUM_CLASS], initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable('biases', [NUM_CLASS], initializer=tf.constant_initializer(0.0))
        if regularizer is not None:
            tf.add_to_collection("losses", regularizer(weights))
        fc8 = tf.nn.xw_plus_b(relu7, weights, biases)

    return fc8

def loss_with_l2(logits, labels):
    with tf.name_scope("cross_entropy"):
        # 定义损失函数为交叉熵.
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels)

        # 计算当前batch中所有样本的交叉熵均值
        cross_entropy_mean = tf.reduce_mean(cross_entropy)

        # 总损失等于交叉熵损失和正则化损失之和
        loss = cross_entropy_mean + tf.add_n(tf.get_collection("losses"))

    return loss