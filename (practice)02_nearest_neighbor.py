#-------------------------------------------------------------- 
# 2017/8/11
# re-implementation nearest neighbor algorithm
#-------------------------------------------------------------- 

# Import library & dataset
# Library
# from __future__ import print_function
import tensorflow as tf
import numpy as np
# MNIST dataset
from tensorflow.examples.tutorials.mnist import input_data
# Download and read data using one_hot encoding reshape data structure
mnist = input_data.read_data_sets("/tmp/data", one_hot = True)

# image data, label of images = MNIST(randomly(?) pick chosen number of images)
Xtr, Ytr = mnist.train.next_batch(5000)
Xte, Yte = mnist.test.next_batch(200)

# Define data structure of TensorFlow a.k.a Tensor
xtr = tf.placeholder("float", [None,784])
xte = tf.placeholder("float", [784])

# Graph Definition 
shortest_distance = tf.reduce_sum(tf.abs(tf.add(xtr, tf.negative(xte))), reduction_indices= 1)
pred = tf.arg_min(shortest_distance, 0)

accuracy = 0

# Define the variable of graph
init = tf.global_variables_initializer()


# Computing
with tf.Session() as sess:
    # Graph initialization
    sess.run(init)
    # Computation of each test case        
    for i in range(len(Xte)):
        # Index computation
        # Feed in test data
        nn_index = sess.run(pred, feed_dict = {xtr: Xtr, xte: Xte[i,:]})
        print("Test:", i+1, "Prediction number:", np.argmax(Ytr[nn_index]), "Real number:", np.argmax(Yte[i]))         
        if np.argmax(Ytr[nn_index]) == np.argmax(Yte[i]):
            accuracy += 1./len(Xte)
        print("Accuracy: ", accuracy, "\n")