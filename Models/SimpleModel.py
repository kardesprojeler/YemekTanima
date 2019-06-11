from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import tensorflow as tf
from Datas.Data import *
import os
import wx
from tensorflow.python import keras


class SimpleModel(keras.Model):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.l2 = keras.regularizers.l2
        self.pool_initial = False
        self.init_filters = (3, 3)
        self.stride = (1, 1)
        self.grow_rate = 12
        self.image_height = 64
        self.image_width = 64
        self.image_deep = 3
        self.num_class = 0
        self.training_images, training_labels = None, None
        self.x = None
        self.y_true = None
        self.accuracy = None
        self.sess = None
        self.optimizer = None
        self.loss = None
        self.batch_size = 10
        self.phase = None
        self.save_path = ''
        self.global_step = tf.Variable(0, trainable=False)
        self.saver = None
        self.make_model()


    def call(self, inputs):
        inputs = np.array(inputs).reshape((1, 200, 200, 3))
        x = self.conv1(tf.cast(inputs, tf.float32))
        x = keras.layers.BatchNormalization(axis=-1)(x)
        x = self.conv2(x)
        x = keras.layers.BatchNormalization(axis=-1)(x)
        x = self.conv3(x)
        x = keras.layers.BatchNormalization(axis=-1)(x)

        x = self.flattened(x)

        x = self.fc1(x)
        x = self.fc2(x)
        return self.fc_out(x)

    def is_model_prepared(self):
            return self.sess is not None

    def global_variable_initializer(self):
        checkpoint_path = 'checkpoints/'
        if not os.path.exists(checkpoint_path):
            os.mkdir(checkpoint_path)
        self.save_path = os.path.join(checkpoint_path, 'yemek_tanima')
        self.saver = tf.train.Saver()

        try:
            print("Checkpoint Yükleniyor")
            latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir=checkpoint_path)
            self.saver.restore(self.sess, save_path=latest_checkpoint)
        except:
            print("Checkpoint bulunamadı")
            self.sess.run()

    def batch_normalization(self, input):
        return keras.layers.BatchNormalization(input)

    def make_model(self):
        self.num_class = getsinifcount()

        self.conv1 = self.conv_layer(24)
        self.conv2 = self.conv_layer(48)
        self.conv3 = self.conv_layer(24)

        self.flattened = keras.layers.Flatten()

        self.fc1 = self.dense_layer(64, activation='relu')
        self.fc2 = self.dense_layer(32, activation='relu')

        self.fc_out = self.dense_layer(self.num_class, activation='softmax')

    def conv_layer(self, num_filters):
        return keras.layers.Conv2D(num_filters,
                            self.init_filters,
                            strides=self.stride,
                            padding="same",
                            use_bias=False,
                            data_format='channels_last',
                            kernel_initializer="he_normal",
                            kernel_regularizer=self.l2(1e-4))
        pass

    def dense_layer(self, size=24, activation='relu'):
        return keras.layers.Dense(units=size, activation=activation)
        pass

    def train_step(self, iteration):
        self.dt.read_train_images(self.image_height, self.image_width)
        for i in range(iteration):
            x_batch, y_batch = self.dt.random_batch(self.batch_size, len(self.dt.training_images), is_training=True,
                                                    append_preprocess=False)
            feed_dict_train = {self.x: x_batch, self.y_true: y_batch, self.phase: True}
            [_, train_acc, g_step] = self.sess.run([self.optimizer, self.loss, self.global_step], feed_dict=feed_dict_train)
            if i % 100 == 0:
                train_acc = self.sess.run(self.accuracy, feed_dict=feed_dict_train)
                print('Iteration:', i, 'Training accuracy:', train_acc)

            if g_step % 300 == 0:
                self.saver.save(self.sess, self.save_path)
                print("Checkpoint kaydedildi")

    def test_accuracy(self):
        self.dt.read_test_images(self.image_height, self.image_width)
        x_batch, y_batch = self.dt.random_batch(self.batch_size, len(self.dt.test_images), is_training=False,
                                                append_preprocess=False)
        feed_dict_test = {self.x: x_batch, self.y_true: y_batch, self.phase: False}
        acc = self.sess.run(self.accuracy, feed_dict=feed_dict_test)
        print('Testing accuracy:', acc)

    def test_accuracy_for_one_image(self):
        x_batch = self.dt.read_image(self.image_width, self.image_height)

        graph = tf.get_default_graph()

        y_pred = graph.get_tensor_by_name("y_pred:0")

        y_test_images = np.zeros((1, 10))
        feed_dict_test = {self.x: x_batch, self.y_true: y_test_images, self.phase: False}

        result = self.sess.run(y_pred, feed_dict=feed_dict_test)

        probability = np.max(result)
        sinif_one_hot = np.argmax(result, 1)

        sinif_name = self.dt.get_sinif_list()[sinif_one_hot[0]].sinifname

        wx.MessageBox((probability * 100).__str__() + " " + sinif_name, 'Bilgilendirme', wx.OK | wx.ICON_INFORMATION)

    def test_accuracy_for_tray(self):
        tray_images = self.dt.get_fragment_tray_images(self.image_height, self.image_width)

        y_pred = tf.get_default_graph().get_tensor_by_name("y_pred:0")
        toplam_fiyat = 0
        y_test_images = np.zeros((1, 10))
        for image in tray_images:

            feed_dict_test = {self.x: image, self.y_true: y_test_images, self.phase: False}

            result = self.sess.run(y_pred, feed_dict=feed_dict_test)

            probability = np.max(result)
            sinif_one_hot = np.argmax(result, 1)

            sinif_name = self.dt.get_sinif_list()[sinif_one_hot[0]].sinifname
            toplam_fiyat += self.dt.get_sinif_list()[sinif_one_hot[0]].fiyat

            wx.MessageBox((probability * 100).__str__() + " " + sinif_name, 'Bilgilendirme', wx.OK | wx.ICON_INFORMATION)

        wx.MessageBox('Toplam Fiyat: ' + toplam_fiyat.__str__(), 'Bilgilendirme', wx.OK | wx.ICON_INFORMATION)

