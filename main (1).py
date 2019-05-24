import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time
from datetime import timedelta
from Datas import Data as data

x = tf.placeholder(tf.float32, [None, 64, 64, 3])
y_true = tf.placeholder(tf.float32, [None, 10])

dt = data.Data()
dt.readimages(64, 64)


def conv_layer(input, size_in, size_out, use_pooling=True):
    w = tf.Variable(tf.truncated_normal([3, 3, size_in, size_out], stddev=0.1))
    b = tf.Variable(tf.constant(0.1, shape=[size_out]))
    conv = tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding='SAME')
    y = tf.nn.relu(conv + b)

    if use_pooling:
        y = tf.nn.max_pool(y, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    return y


def fc_layer(input, size_in, size_out, relu=True):
    w = tf.Variable(tf.truncated_normal([size_in, size_out], stddev=0.1))
    b = tf.Variable(tf.constant(0.1, shape=[size_out]))
    logits = tf.matmul(input, w) + b

    if relu:
        return tf.nn.relu(logits)
    else:
        return logits


conv1 = conv_layer(x, 3, 32, use_pooling=True)
conv2 = conv_layer(conv1, 32, 64, use_pooling=True)
conv3 = conv_layer(conv2, 64, 64, use_pooling=True)
flattened = tf.reshape(conv3, [-1, 8 * 8 * 64])
fc1 = fc_layer(flattened, 8 * 8 * 64, 512, relu=True)
fc2 = fc_layer(fc1, 512, 256, relu=True)
logits = fc_layer(fc2, 256, 10, relu=False)
y = tf.nn.softmax(logits)

y_pred_cls = tf.argmax(y, 1)

xent = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_true)
loss = tf.reduce_mean(xent)

correct_prediction = tf.equal(y_pred_cls, tf.argmax(y_true, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

optimizer = tf.train.AdamOptimizer(5e-4).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

batch_size = 33

loss_graph = []

def training_step(iterations):
    start_time = time.time()
    for i in range(iterations):
        x_batch, y_batch = dt.random_batch(batch_size, 32)
        feed_dict_train = {x: x_batch, y_true: y_batch}
        [_, train_loss] = sess.run([optimizer, loss], feed_dict=feed_dict_train)
        loss_graph.append(train_loss)

        if i % 100 == 0:
            acc = sess.run(accuracy, feed_dict=feed_dict_train)
            print('Iteration:', i, 'Training accuracy:', acc, 'Training loss:', train_loss)

    end_time = time.time()
    time_dif = end_time - start_time
    print("Time usage:", timedelta(seconds=int(round(time_dif))))


batch_size_test = 256


def test_accuracy():
    num_images = len(test_img)
    cls_pred = np.zeros(shape=num_images, dtype=np.int)

    i = 0
    while i < num_images:
        j = min(i + batch_size_test, num_images)
        feed_dict = {x: test_img[i:j, :], y_true: test_labels[i:j, :]}
        cls_pred[i:j] = sess.run(y_pred_cls, feed_dict=feed_dict)
        i = j

    correct = (test_cls == cls_pred)
    print('Testing accuracy:', correct.mean())


training_step(1000)

plt.plot(loss_graph, 'k-')
plt.title('Loss grafiği')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.show()
