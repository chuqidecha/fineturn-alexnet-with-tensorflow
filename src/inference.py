import tensorflow as tf


def variable_summaries(var, name):
    with tf.variable_scope("log-summaries"):
        tf.summary.histogram(name, var)

        mean = tf.reduce_mean(var)

        tf.summary.scalar("mean/" + name, mean)

        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))

        tf.summary.scalar("stddev/" + name, stddev)


def conv_with_groups(name_scope, xs, ws, groups, strides, padding):
    with tf.name_scope(name_scope):
        ws_groups = tf.split(value=ws, num_or_size_splits=groups, axis=3)
        xs_groups = tf.split(value=xs, num_or_size_splits=groups, axis=3)
        conv_groups = [tf.nn.conv2d(x, w, [1, 1, 1, 1], padding=padding) for w, x in zip(ws_groups, xs_groups)]
        conv = tf.concat(values=conv_groups, axis=3)
    return conv


def inference(input_tensor, output_dim, keep_prob, regularizer=None):
    '''
    AlexNet
    '''
    with tf.variable_scope("conv1"):
        weights = tf.get_variable('weights', [11, 11, 3, 96], initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable('biases', [96], initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.bias_add(tf.nn.conv2d(input_tensor, weights, [1, 4, 4, 1], padding="VALID"), biases)
        variable_summaries(weights, "conv1/weights")
        variable_summaries(biases, "conv1/biases")

    with tf.name_scope("relu1"):
        relu1 = tf.nn.relu(conv1)

    with tf.name_scope("lrn1"):
        lrn1 = tf.nn.lrn(relu1, depth_radius=2, bias=1, alpha=0.00002, beta=0.75)
    with tf.name_scope("pool1"):
        pool1 = tf.nn.max_pool(lrn1, [1, 3, 3, 1], [1, 2, 2, 1], padding="VALID")

    with tf.variable_scope("conv2"):
        weights = tf.get_variable('weights', [5, 5, 48, 256], initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable('biases', [256], initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.bias_add(conv_with_groups("conv2-groups", pool1, weights, 2, [1, 1, 1, 1], padding="SAME"),
                               biases)
        variable_summaries(weights, "conv2/weights")
        variable_summaries(biases, "conv2/biases")

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
        variable_summaries(weights, "conv3/weights")
        variable_summaries(biases, "conv3/biases")

    with tf.name_scope("relu3"):
        relu3 = tf.nn.relu(conv3)

    with tf.variable_scope("conv4"):
        weights = tf.get_variable('weights', [3, 3, 192, 384], initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable('biases', [384], initializer=tf.constant_initializer(0.0))
        conv4 = tf.nn.bias_add(conv_with_groups("conv4-groups", relu3, weights, 2, [1, 1, 1, 1], padding="SAME"),
                               biases)
        variable_summaries(weights, "conv4/weights")
        variable_summaries(biases, "conv4/biases")

    with tf.name_scope("relu4"):
        relu4 = tf.nn.relu(conv4)

    with tf.variable_scope("conv5"):
        weights = tf.get_variable('weights', [3, 3, 192, 256], initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable('biases', [256], initializer=tf.constant_initializer(0.0))
        conv5 = tf.nn.bias_add(conv_with_groups("conv5-groups", relu4, weights, 2, [1, 1, 1, 1], padding="SAME"),
                               biases)
        variable_summaries(weights, "conv5/weights")
        variable_summaries(biases, "conv5/biases")

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
        variable_summaries(weights, "fc6/weights")
        variable_summaries(biases, "fc6/biases")

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
        variable_summaries(weights, "fc7/weights")
        variable_summaries(biases, "fc7/biases")

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
        variable_summaries(weights, "fc8/weights")
        variable_summaries(biases, "fc8/biases")
    return fc8
