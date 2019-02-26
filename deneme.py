import tensorflow as tf
from Datas import Data as data

dt = data.Data()
image_height = 64
image_width = 64
image_deep = 3
training_images, training_labels = dt.readimages(image_height, image_width)
num_class = dt.getsinifcount()
x = tf.placeholder(tf.float32, [None, image_height, image_width, image_deep])
y_true = tf.placeholder(tf.float32, [None, num_class])


def conv_layer(input, input_size, output_size, use_pooling=True):
    w = tf.Variable(tf.truncated_normal([3, 3, input_size, output_size], stddev=0.1))
    b = tf.Variable(tf.constant(0.1, shape=[output_size]))

    conv = tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding='SAME')
    y = tf.nn.relu(conv)

    if use_pooling:
        y = tf.nn.max_pool(y, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    return y
    pass


def fc_layer(input, input_size, output_size, use_relu=True):
    w = tf.Variable(tf.truncated_normal([input_size, output_size], stddev=0.1))
    b = tf.Variable(tf.constant(0.1, shape=[output_size]))

    logit = tf.matmul(input, w) + b

    if use_relu:
        return tf.nn.relu(logit)
    else:
        return logit
    pass


conv1 = conv_layer(x, image_deep, 32, use_pooling=True)
conv2 = conv_layer(conv1, 32, 64, use_pooling=True)
conv3 = conv_layer(conv2, 64, 64, use_pooling=True)

flattened = tf.reshape(conv3, [-1, 8 * 8 * 64])
fc1 = fc_layer(flattened, 8 * 8 * 64, 512, use_relu=True)
fc2 = fc_layer(fc1, 512, 256, use_relu=True)

logits = fc_layer(fc2, 256, num_class, use_relu=False)
y = tf.nn.softmax(logits)

xent = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_true)
loss = tf.reduce_mean(xent)

correct_predict = tf.equal(tf.argmax(y, 1), tf.argmax(y_true, 1))
accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))

optimizer = tf.train.AdamOptimizer(5e-4).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

def train_step(iteration):
    for i in range(iteration):
        x_batch, y_batch = dt.random_batch(10, num_class)
        feed_dict_train = {x: x_batch, y_true: y_batch}
        sess.run(optimizer, feed_dict=feed_dict_train)
        if i % 100 == 0:
            train_acc = sess.run(accuracy, feed_dict=feed_dict_train)
            print('Iteration:', i, 'Training accuracy:', train_acc, )

def test_accuracy(self):
    feed_dict_test = {self.x: self.dt.training_images, self.y_true: self.dt.training_labels}
    acc = self.sess.run(self.accuracy, feed_dict=feed_dict_test)
    print('Testing accuracy:', acc)


train_step(1000)