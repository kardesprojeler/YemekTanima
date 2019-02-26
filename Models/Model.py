import numpy as np
import tensorflow as tf
from Datas import Data as data


class Model:

    def __init__(self):
        self.dt = data.Data()
    dt = None
    image_height = 64
    image_width = 64
    image_deep = 3
    num_class = 0
    training_images, training_labels = None, None
    x = None
    y_true = None
    accuracy = None
    sess = None
    optimizer = None
    loss = None
    batch_size = 10

    def make_model(self):
        training_images, training_labels = self.dt.readimages(self.image_height, self.image_width)
        self.num_class = self.dt.getsinifcount()
        self.x = tf.placeholder(tf.float32, [None, self.image_height, self.image_width, self.image_deep])
        self.y_true = tf.placeholder(tf.float32, [None, self.num_class])

        conv1 = self.conv_layer(self.x, self.image_deep, 32, use_pooling=True)
        conv2 = self.conv_layer(conv1, 32, 64, use_pooling=True)
        conv3 = self.conv_layer(conv2, 64, 64, use_pooling=True)

        flattened = tf.reshape(conv3, [-1, 8 * 8 * 64])
        fc1 = self.fc_layer(flattened, 8 * 8 * 64, 512, use_relu=True)
        fc2 = self.fc_layer(fc1, 512, 256, use_relu=True)

        logits = self.fc_layer(fc2, 256, self.num_class, use_relu=False)
        y = tf.nn.softmax(logits)

        xent = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.y_true)
        self.loss = tf.reduce_mean(xent)

        correct_predict = tf.equal(tf.argmax(y, 1), tf.argmax(self.y_true, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))

        self.optimizer = tf.train.AdamOptimizer(5e-4).minimize(self.loss)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def conv_layer(self, input, input_size, output_size, use_pooling=True):
        w = tf.Variable(tf.truncated_normal([3, 3, input_size, output_size], stddev=0.1))
        b = tf.Variable(tf.constant(0.1, shape=[output_size]))

        conv = tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding='SAME')
        y = tf.nn.relu(conv)

        if use_pooling:
            y = tf.nn.max_pool(y, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        return y
        pass

    def fc_layer(self, input, input_size, output_size, use_relu=True):
        w = tf.Variable(tf.truncated_normal([input_size, output_size], stddev=0.1))
        b = tf.Variable(tf.constant(0.1, shape=[output_size]))

        logit = tf.matmul(input, w) + b

        if use_relu:
            return tf.nn.relu(logit)
        else:
            return logit
        pass

    def train_step(self, iteration):
        for i in range(iteration):
            x_batch, y_batch = self.dt.random_batch(self.batch_size, len(self.dt.training_images))
            feed_dict_train = {self.x: x_batch, self.y_true: y_batch}
            self.sess.run(self.optimizer, feed_dict=feed_dict_train)
            if i % 100 == 0:
                train_acc = self.sess.run(self.accuracy, feed_dict=feed_dict_train)
                print('Iteration:', i, 'Training accuracy:', train_acc)

    def test_accuracy(self):
        feed_dict_test = {self.x: self.dt.training_images, self.y_true: self.dt.training_labels}
        acc = self.sess.run(self.accuracy, feed_dict=feed_dict_test)
        print('Testing accuracy:', acc)
