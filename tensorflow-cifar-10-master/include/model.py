import tensorflow as tf


def model():
    _IMAGE_SIZE = 32
    _IMAGE_CHANNELS = 3
    _NUM_CLASSES = 10
    tf.reset_default_graph()
    con1_filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 3, 64], mean=0, stddev=0.08))
    con2_filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 128], mean=0, stddev=0.08))
    con3_filter = tf.Variable(tf.truncated_normal(shape=[5, 5, 128, 256], mean=0, stddev=0.08))
    con4_filter = tf.Variable(tf.truncated_normal(shape=[5, 5, 256, 512], mean=0, stddev=0.08))

    with tf.name_scope('main_params'):
        x = tf.placeholder(tf.float32, shape=[None, _IMAGE_SIZE * _IMAGE_SIZE * _IMAGE_CHANNELS], name='Input')
        y = tf.placeholder(tf.float32, shape=[None, _NUM_CLASSES], name='Output')
        keep_prob = tf.placeholder(tf.float32, shape=[], name='keep_prob')
        x_image = tf.reshape(x, [-1, _IMAGE_SIZE, _IMAGE_SIZE, _IMAGE_CHANNELS], name='images')

        global_step = tf.Variable(initial_value=0, trainable=False, name='global_step')
        learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')

    with tf.variable_scope('conv1') as scope:
        # 1, 2
        conv1 = tf.nn.conv2d(x_image, con1_filter, strides=[1, 1, 1, 1], padding='SAME', name='Conv1')
        conv1 = tf.nn.relu(conv1, name='Relu')
        conv1_pool = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='Max_pooling')
        conv1_bn = tf.layers.batch_normalization(conv1_pool, name='Normalization')
    with tf.variable_scope('conv2') as scope:
        # 3, 4
        conv2 = tf.nn.conv2d(conv1_bn, con2_filter, strides=[1, 1, 1, 1], padding='SAME', name='Conv2')
        conv2 = tf.nn.relu(conv2, name='Relu')
        conv2_pool = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='Max_pooling')
        conv2_bn = tf.layers.batch_normalization(conv2_pool, name='Normalization')
    with tf.variable_scope('conv3') as scope:
        # 5, 6
        conv3 = tf.nn.conv2d(conv2_bn, con3_filter, strides=[1, 1, 1, 1], padding='SAME', name='Conv3')
        conv3 = tf.nn.relu(conv3, name='Relu')
        conv3_pool = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='Max_pooling')
        conv3_bn = tf.layers.batch_normalization(conv3_pool, name='Normalization')
    with tf.variable_scope('conv4') as scope:
        # 7, 8
        conv4 = tf.nn.conv2d(conv3_bn, con4_filter, strides=[1, 1, 1, 1], padding='SAME', name='Conv4')
        conv4 = tf.nn.relu(conv4, name='Relu')
        conv4_pool = tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='Max_pooling')
        conv4_bn = tf.layers.batch_normalization(conv4_pool, name='Normalization')

    # 9
    flat = tf.contrib.layers.flatten(conv4_bn)
    with tf.variable_scope('full1') as scope:
        # 10
        full1 = tf.contrib.layers.fully_connected(inputs=flat, num_outputs=128, activation_fn=tf.nn.relu)
        full1 = tf.nn.dropout(full1, keep_prob)
        full1 = tf.layers.batch_normalization(full1)
    with tf.variable_scope('full2') as scope:
        # 11
        full2 = tf.contrib.layers.fully_connected(inputs=full1, num_outputs=256, activation_fn=tf.nn.relu)
        full2 = tf.nn.dropout(full2, keep_prob)
        full2 = tf.layers.batch_normalization(full2)
    with tf.variable_scope('full3') as scope:
        # 12
        full3 = tf.contrib.layers.fully_connected(inputs=full2, num_outputs=512, activation_fn=tf.nn.relu)
        full3 = tf.nn.dropout(full3, keep_prob)
        full3 = tf.layers.batch_normalization(full3)
    with tf.variable_scope('full4') as scope:
        # 13
        full4 = tf.contrib.layers.fully_connected(inputs=full3, num_outputs=1024, activation_fn=tf.nn.relu)
        full4 = tf.nn.dropout(full4, keep_prob)
        full4 = tf.layers.batch_normalization(full4)
    with tf.variable_scope('full5_softmax') as scope:
        softmax = tf.layers.dense(inputs=full4, units=_NUM_CLASSES, activation=None, name=scope.name)
        y_pred_cls = tf.argmax(softmax, axis=1)
    return x, y, keep_prob, softmax, y_pred_cls, global_step, learning_rate


def lr(epoch):
    learning_rate = 1e-3

    if epoch> 10:
        learning_rate = 1e-4
    return learning_rate
