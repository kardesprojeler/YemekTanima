from absl import app
from Models.DenseNet import DenseNet
from Models.SimpleModel import SimpleModel
from Datas.Data import *
from tensorflow.python import keras
import numpy as np

class Train(object):
  """Train class.
  Args:
    epochs: Number of epochs
    enable_function: If True, wraps the train_step and test_step in tf.function
    model: Densenet model.
  """

  def __init__(self, epochs, enable_function, model):
      self.epochs = epochs
      self.enable_function = enable_function
      self.autotune = tf.data.experimental.AUTOTUNE
      self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
          reduction=tf.keras.losses.Reduction.SUM)

      self.optimizer = tf.keras.optimizers.Adam()
      self.train_loss_metric = keras.metrics.Mean(name='train_loss')
      self.train_acc_metric = keras.metrics.SparseCategoricalAccuracy(
          name='train_accuracy')
      self.test_loss_metric = keras.metrics.Mean(name='test_loss')
      self.test_acc_metric = keras.metrics.SparseCategoricalAccuracy(
          name='test_accuracy')
      self.model = model
      self.model.load_weights(GeneralFlags.checkpoint_dir.value)
      pass

  def load_initial_weights(self):
      if os.path.exists(GeneralFlags.checkpoint_dir.value) and self.model is not None:
          self.model.load_weights(GeneralFlags.checkpoint_dir.value)
          pass

  def decay(self, epoch):
    if epoch < 150:
      return 0.1
    if epoch >= 150 and epoch < 225:
      return 0.01
    if epoch >= 225:
      return 0.001

  def keras_fit(self, train_dataset, test_dataset):
    self.model.compile(
        optimizer=self.optimizer, loss=self.loss_object, metrics=['accuracy'])
    history = self.model.fit(
        train_dataset, epochs=self.epochs, validation_data=test_dataset,
        verbose=2, callbacks=[keras.callbacks.LearningRateScheduler(
            self.decay)])
    return (history.history['loss'][-1],
            history.history['accuracy'][-1],
            history.history['val_loss'][-1],
            history.history['val_accuracy'][-1])

  def loss_function(self, real, pred):
      mask = tf.math.logical_not(tf.math.equal(real, 0))
      loss_ = tf.nn.softmax_cross_entropy_with_logits(real, pred)
      mask = tf.cast(mask, dtype=loss_.dtype)
      loss_ *= mask
      return tf.reduce_sum(loss_) * 1. / 1

  def train_step(self, train_ds, model_name='SimpleModel'):
    """One train step.
    Args:
      image: Batch of images.
      label: corresponding label for the batch of images.
    """

    image_batch, label_batch = train_ds

    with tf.GradientTape() as tape:
        loss = None
        predictions = None

        if model_name == 'SimpleModel':
            predictions = self.model(image_batch)
            loss = self.loss_function(label_batch, predictions)
            pass
        elif model_name == 'DenseNet':
            predictions = self.model(image_batch, training=True)
            loss = self.loss_function(label_batch, predictions)
            loss += sum(self.model.losses)
            pass
    gradients = tape.gradient(loss, self.model.trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

    self.train_loss_metric(loss)
    self.train_acc_metric(label_batch, predictions)

  def test_step(self, data):
    """One test step.
    Args:
      image: Batch of images.
      label: corresponding label for the batch of images.
    """
    image, label = data
    predictions = self.model(image, training=False)
    loss = self.loss_object(label, predictions)

    self.test_loss_metric(loss)
    self.test_acc_metric(label, predictions)

  def custom_loop(self, train_dataset, test_dataset):
    """Custom training and testing loop.
    Args:
      train_dataset: Training dataset
      test_dataset: Testing dataset
    Returns:
      train_loss, train_accuracy, test_loss, test_accuracy
    """
    template = ('Epoch: {}, Train Loss: {}, Train Accuracy: {}, '
                'Test Loss: {}, Test Accuracy: {}')

    for epoch in range(self.epochs):
        self.optimizer.learning_rate = self.decay(epoch)
        """
        if False:
            self.train_step = tf.function(self.train_step)
            self.test_step = tf.function(self.test_step)
            pass
        """
        i = 0
        for epoch in range(self.epochs):
            for data in train_dataset:
                i = i + 1
                self.train_step(data)
                if i % 10 == 0:
                    print(template.format(epoch, self.train_loss_metric.result(),
                                          self.train_acc_metric.result(),
                                          self.test_loss_metric.result(),
                                          self.test_acc_metric.result()))
                    pass

                if (i + 1) % 300 == 0:
                    self.model.save_weights(GeneralFlags.checkpoint_dir.value)
                    print("Checkpoint kaydedildi")
                    pass
                pass

            for data in test_dataset:
                self.test_step(data)
                pass

        if epoch != self.epochs - 1:
            self.train_loss_metric.reset_states()
            self.train_acc_metric.reset_states()
            self.test_loss_metric.reset_states()
            self.test_acc_metric.reset_states()

        return (self.train_loss_metric.result().numpy(),
                self.train_acc_metric.result().numpy(),
                self.test_loss_metric.result().numpy(),
                self.test_acc_metric.result().numpy())


def run_main(model_name, argv):
  main(model_name, GeneralFlags.epoch.value, GeneralFlags.enable_function.value, GeneralFlags.train_mode.value)


def main(model_name, epochs, enable_function, train_mode):
    model = None
    batch_size = 1
    if model_name == 'SimpleModel':
        model = SimpleModel(SimpleModelFlags.pool_initial.value, SimpleModelFlags.init_filter.value,
                            SimpleModelFlags.stride.value, SimpleModelFlags.growth_rate.value,
                            SimpleModelFlags.image_height.value, SimpleModelFlags.image_width.value,
                            SimpleModelFlags.image_deep.value, SimpleModelFlags.batch_size.value,
                            SimpleModelFlags.save_path.value)
        batch_size = SimpleModelFlags.batch_size.value
        pass
    elif model_name == 'DenseNet':
        model = DenseNet(DenseNetFlags.mode.value, DenseNetFlags.growth_rate.value, DenseNetFlags.output_classes.value,
                         DenseNetFlags.depth_of_model.value, DenseNetFlags.num_of_blocks.value,
                         DenseNetFlags.num_layers_in_each_block.value, DenseNetFlags.data_format.value,
                         DenseNetFlags.bottleneck.value, DenseNetFlags.compression.value,
                         DenseNetFlags.weight_decay.value, DenseNetFlags.dropout_rate.value,
                         DenseNetFlags.pool_initial.value, DenseNetFlags.include_top.value)
        batch_size = DenseNetFlags.batch_size.value
        pass

    train_obj = Train(epochs, enable_function, model)

    train_dataset = read_train_images(200, 200, batch_size)
    test_dataset = read_test_images(200, 200, batch_size)

    print('Training...')
    if train_mode == 'custom_loop':
        return train_obj.custom_loop(train_dataset, test_dataset)
    elif train_mode == 'keras_fit':
        return train_obj.keras_fit(train_dataset, test_dataset)


def train_model():
    app.run(run_main)