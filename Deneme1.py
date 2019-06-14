from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from tensorflow.python.keras.layers import Dense, Flatten, Conv2D
from tensorflow.python.keras import Model

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Add a channels dimension
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(1000).batch(10)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(10)


class MyModel(Model):
  def __init__(self):
    super(MyModel, self).__init__()
    self.conv1 = Conv2D(16, 1, activation='relu')
    self.flatten = Flatten()
    self.d1 = Dense(16, activation='relu')
    self.d2 = Dense(10, activation='softmax')

  def call(self, x):
    x = self.conv1(x)
    x = self.flatten(x)
    x = self.d1(x)
    return self.d2(x)


class Train(object):
    def __init__(self, epochs, enable_function, batch_size, per_replica_batch_size):
        self.epochs = epochs
        self.enable_function = enable_function
        self.batch_size = batch_size
        self.per_replica_batch_size = per_replica_batch_size
        self.learning_rate = 0.1
        self.model = MyModel()
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            reduction=tf.keras.losses.Reduction.SUM)
        self.optimizer = tf.keras.optimizers.Adam()
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    def loss_function(self, real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = self.loss_object(real, pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_sum(loss_) * 1. / self.batch_size

    def train_stepx(self, inputs):
        images, labels = inputs
        with tf.GradientTape() as tape:
            predictions = self.model(images)
            loss = self.loss_function(labels, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_loss(loss)
        self.train_accuracy(labels, predictions)

    def test_step(self, inputs):
        images, labels = inputs
        predictions = self.model(images)
        t_loss = self.loss_function(labels, predictions)

        self.test_loss(t_loss)
        self.test_accuracy(labels, predictions)

    def training_loop(self, train_dataset, test_dataset):
        if self.enable_function:
            self.train_step = tf.function(self.train_stepx)
            self.test_step = tf.function(self.test_step)
        for epoch in range(self.epochs):
            self.train_loss.reset_states()
            self.test_loss.reset_states()
            self.train_accuracy.reset_states()
            self.test_accuracy.reset_states()

            for images, labels in train_dataset:
                self.train_stepx((images, labels))
            for test_images, test_labels in test_dataset:
                self.test_step((test_images, test_labels))

            template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
            print(template.format(epoch + 1,
                                  self.train_loss.result(),
                                  self.train_accuracy.result() * 100,
                                  self.test_loss.result(),
                                  self.test_accuracy.result() * 100))


if __name__ == '__main__':
    epochs = 1
    enable_function = True
    batch_size = 10
    per_replica_batch_size = 10
    train_obj = Train(epochs, enable_function, batch_size, per_replica_batch_size)
    train_obj.training_loop(train_ds, test_ds)
    tf.saved_model.save(train_obj.model, 'model')