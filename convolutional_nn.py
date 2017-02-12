from __future__ import absolute_import
from __future__ import division

import csv
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import cross_validation
from sklearn.utils import resample

# Constants
IMAGE_SIZE = 28 * 28
NUM_CLASS = 10
BATCH_SIZE = 100
ITERATIONS = 8000

# Helper functions
def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1, dtype=tf.float32)
	return tf.Variable(initial)
def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape, dtype=tf.float32)
	return tf.Variable(initial)
def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# Read csv files
train_data = pd.read_csv('data/train.csv').as_matrix()
test_data = pd.read_csv('data/test.csv').as_matrix().astype(np.float32)

# Sample a training set while holding out 5% of the data for evaluating our trained classifier
X = train_data[:, 1:].astype(np.float32)
y = np.zeros((X.shape[0], NUM_CLASS))
X = (X / 255.0) - 1.0
test_data = (test_data / 255.0) - 1.0
y[np.arange(X.shape[0]), train_data[:, 0]] = 1
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.05, random_state=0)

# Construct the computational graph
X_ = tf.placeholder(tf.float32, shape=(None, IMAGE_SIZE))
y_ = tf.placeholder(tf.float32, shape=(None, NUM_CLASS))
X_image = tf.reshape(X_, [-1,28,28,1])

W_conv1 = weight_variable([5, 5, 1, 20])
b_conv1 = bias_variable([20])
h_conv1 = tf.nn.relu(conv2d(X_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
h_pool1_flat = tf.reshape(h_pool1, [-1, 14*14*20])

W_fc1 = weight_variable([14*14*20, 10])
b_fc1 = bias_variable([10])
y_conv = tf.matmul(h_pool1_flat, W_fc1) + b_fc1
tf.Print(y_conv, [y_conv])

loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv)

predictions = tf.argmax(y_conv, 1)
correct_labels = tf.argmax(y_, 1)
correct_prediction = tf.equal(predictions, correct_labels)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Define the optimizer
opt = tf.train.AdamOptimizer(1e-4)
opt_op = opt.minimize(loss)

# Start tf session
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# Train
for i in range(ITERATIONS):
    batch_xs, batch_ys = resample(X_train, y_train, replace=False, n_samples=BATCH_SIZE)
    if (i + 1) % 100 == 0:
        tmp_accuracy = sess.run(accuracy, feed_dict={X_: X_test, y_: y_test})
        print 'Before the iteration ' + str(i+1) + ', validation accuracy = ' + str(tmp_accuracy)
    sess.run(opt_op, feed_dict={X_: batch_xs, y_: batch_ys})

# Evaluate the trained model
print 'Training accuracy: ' + str(sess.run(accuracy, feed_dict={X_: X_train, y_: y_train}))
print 'Validation accuracy: ' + str(sess.run(accuracy, feed_dict={X_: X_test, y_: y_test}))

# Create the submission file
predictions = sess.run(predictions, feed_dict={X_: test_data}).tolist()
with open('submissions/convolutional_nn.csv', 'w+') as csvfile:
    fieldnames = ['ImageId', 'Label']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for i in range(len(predictions)):
        writer.writerow({'ImageId': str(i+1), 'Label': predictions[i]})