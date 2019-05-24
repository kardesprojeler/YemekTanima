import numpy as np
import tensorflow as tf
from Datas import Data as data
import os
import wx

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
    phase = tf.placeholder(tf.bool)
    save_path = ''
    global_step = tf.Variable(0, trainable=False)
    saver = None

    def is_model_prepared(self):
            return self.sess is not None

<<<<<<< HEAD


#model = keras.Sequential([
 #   keras.layers.Flatten(input_shape=(28, 28)),
  #  keras.layers.Dense(128, activation=tf.nn.relu),
   # keras.layers.Dense(10, activation=tf.nn.softmax)
#])
#model.compile(optimizer='adam',
              #loss='sparse_categorical_crossentropy',
              #metrics=['accuracy'])

=======
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
            self.sess.run(tf.global_variables_initializer())

    def batch_normalization(self, input, phase, scope):
        return tf.cond(phase,
                       lambda: tf.contrib.layers.batch_norm(input, decay=0.99, is_training=True,
                                                            updates_collections=None, center=True, scope=scope),
                       lambda: tf.contrib.layers.batch_norm(input, decay=0.99, is_training=False,
                                                            updates_collections=None, center=True, scope=scope, reuse=True))

    def make_model(self):
        self.num_class = self.dt.getsinifcount()
        self.x = tf.placeholder(tf.float32, [None, self.image_height, self.image_width, self.image_deep])
        self.y_true = tf.placeholder(tf.float32, [None, self.num_class], name='y_true')

        conv1 = self.conv_layer(self.x, self.image_deep, 32, scope='conv1', use_pooling=True)
        conv2 = self.conv_layer(conv1, 32, 64, scope='conv2', use_pooling=True)
        conv3 = self.conv_layer(conv2, 64, 64, scope='conv3', use_pooling=True)

        flattened = tf.reshape(conv3, [-1, 8 * 8 * 64])
        fc1 = self.fc_layer(flattened, 8 * 8 * 64, 512, scope='fc1', use_relu=True, batch_normalization=True)
        fc2 = self.fc_layer(fc1, 512, 256, scope='fc2', use_relu=True, batch_normalization=True)

        logits = self.fc_layer(fc2, 256, self.num_class, scope='fc_out', use_relu=False, batch_normalization=False)
        y = tf.nn.softmax(logits, name="y_pred")

        xent = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=self.y_true)
        self.loss = tf.reduce_mean(xent)

        correct_predict = tf.equal(tf.argmax(y, 1), tf.argmax(self.y_true, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))

        self.optimizer = tf.train.AdamOptimizer(5e-4).minimize(self.loss, self.global_step)

        self.sess = tf.Session()
        self.global_variable_initializer()

    def conv_layer(self, input, input_size, output_size, scope, use_pooling=True):
        w = tf.Variable(tf.truncated_normal([3, 3, input_size, output_size], stddev=0.1))
        b = tf.Variable(tf.constant(0.1, shape=[output_size]))

        conv = tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding='SAME')
        conv_bn = self.batch_normalization(conv, self.phase, scope)
        y = tf.nn.relu(conv_bn)

        if use_pooling:
            y = tf.nn.max_pool(y, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        return y
        pass

    def fc_layer(self, input, input_size, output_size, scope, use_relu=True, batch_normalization=False):
        w = tf.Variable(tf.truncated_normal([input_size, output_size], stddev=0.1))
        b = tf.Variable(tf.constant(0.1, shape=[output_size]))

        logit = tf.matmul(input, w) + b

        if batch_normalization:
            logit = self.batch_normalization(logit, self.phase, scope)

        if use_relu:
            return tf.nn.relu(logit)
        else:
            return logit
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
>>>>>>> 3a007f1799aae5a4e2cbeca31a6c08a20ea39dd9
