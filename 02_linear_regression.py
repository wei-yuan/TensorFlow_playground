#---------------------------------------
# reference: aymericdamien/TensorFlow-Examples
# Practicing using TensorFlow implemetation linear regression algorithm
#---------------------------------------
from __future__ import print_function
import tensorflow as tf
import numpy
import matplotlib.pyplot as plt

rng = numpy.random

# Define parameters
learning_rate = 0.01
training_epochs = 1000
display_step = 50

# Training data
train_X = numpy.asarray([3.8,6.4,1.1,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
                         7.666,10.791,5.313,7.885,5.654,9.27,3.1])
train_Y = numpy.asarray([2.7,3.76,2.09,3.2,1.798,1.666,3.366,2.596,2.53,1.221,
                         2.827,3.465,2.65,2.904,2.42,2.94,5.3])
n_samples = train_X.shape[0]

# Data structure
X = tf.placeholder("float")
Y = tf.placeholder("float")

# Define model parameters
W = tf.Variable(rng.randn(), name= "weight")
B = tf.Variable(rng.randn(), name= "bias")

# Create linear model of linear regression
pred = tf.add(tf.multiply(X,W),B)
# Mean square error
cost = tf.reduce_sum(tf.pow(pred-Y,2)) / n_samples
# Use gradient descent to find the best configuration of W and B
# minimize function minimize the loss of function
# it knows how to modify W and B because Variable object
# are trainable = True by default
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Variables initialization
init = tf.global_variables_initializer()

# Execution the computation of graph
with tf.Session() as sess:
    sess.run(init)

    # Fit all training data
    for epoch in range(training_epochs):
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X: x, Y: y})

        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            cost_for_display = sess.run(cost, feed_dict={X: train_X, Y:train_Y})
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(cost_for_display), \
                "W=", sess.run(W), "B=", sess.run(B))

    print("Optimization Finished!")
    training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
    print("Training cost=", training_cost, "W=", sess.run(W), "B=", sess.run(B), '\n')

    # Graphic display
    plt.plot(train_X, train_Y, 'ro', label='Original data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(B), label='Fitted line')
    plt.legend()
    plt.show()
    # Draw Sumarry of graph
    writer = tf.summary.FileWriter("/home/alex504/TensorFlow_playground/02/nearest_neighbor/02/linear_regression/")
    writer.add_graph(sess.graph)    