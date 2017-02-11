from __future__ import absolute_import
from __future__ import division

import csv
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import cross_validation
from sklearn.utils import resample

# Constants
LEARNING_RATE = 0.0005
NUM_CLASS = 10
BATCH_SIZE = 100
ITERATIONS = 6000

# Read csv files
train_data = pd.read_csv('data/train.csv').as_matrix()
test_data = pd.read_csv('data/test.csv').as_matrix()

# Sample a training set while holding out 5% of the data for evaluating our trained classifier
X = train_data[:, 1:]
y = np.zeros((X.shape[0], NUM_CLASS))
X = (X / 255.0) - 1.0
test_data = (test_data / 255.0) - 1.0
y[np.arange(X.shape[0]), train_data[:, 0]] = 1
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.05, random_state=0)


# Construct the computational graph
X_ = tf.placeholder(tf.float32, shape=(None, 784))
y_ = tf.placeholder(tf.float32, shape=(None, 10))
W  = tf.Variable(tf.zeros([784, 10], tf.float32))
b  = tf.Variable(tf.zeros([10], tf.float32))
y = tf.matmul(X_, W) + b
predictions = tf.argmax(y, 1)
loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)

# Define the optimizer
opt = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE)
opt_op = opt.minimize(loss)

# Start tf session
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# Train
for _ in range(ITERATIONS):
    batch_xs, batch_ys = resample(X_train, y_train, replace=False, n_samples=BATCH_SIZE)
    sess.run(opt_op, feed_dict={X_: batch_xs, y_: batch_ys})

# Evaluate the trained model
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print 'Training accuracy: ' + str(sess.run(accuracy, feed_dict={X_: X_train, y_: y_train}))
print 'Validation accuracy: ' + str(sess.run(accuracy, feed_dict={X_: X_test, y_: y_test}))

# Create the submission file
predictions = sess.run(predictions, feed_dict={X_: test_data}).tolist()
with open('submissions/softmax_regression.csv', 'w+') as csvfile:
    fieldnames = ['ImageId', 'Label']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for i in range(len(predictions)):
        writer.writerow({'ImageId': str(i+1), 'Label': predictions[i]})