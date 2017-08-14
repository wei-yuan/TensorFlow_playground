#---------------------------------------
# reference: aymericdamien/TensorFlow-Examples
# A nearest neighbor learning algorithm example using TensorFlow library.
# This example uses the MNIST database of handwritten digits
# nearest neighbor learning algorithm: greedy algorithm, not always optimal 
#---------------------------------------

from __future__ import print_function
import tensorflow as tf
import numpy as np


# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data", one_hot = True)

# Limit mnist input data number
Xtr, Ytr = mnist.train.next_batch(5000) #5000 for 'tr'aining (nearest neighbor candidates)
Xte, Yte = mnist.test.next_batch(200) #200 for 'te'sting

# TensorFlow graph input
xtr = tf.placeholder("float", [None,784]) #784 = 28^2
xte = tf.placeholder("float", [784])

# Nearest neighbor calculation using L1 distance
# Calculate L1 distance
distance = tf.reduce_sum(tf.abs(tf.add(xtr, tf.negative(xte))), reduction_indices = 1)
# Prediction
pred = tf.arg_min(distance, 0) # For vector, use zero dimension 

accuracy = 0

# Initialization the variables
init = tf.global_variables_initializer()

# Launch the graph and compute
with tf.Session() as sess:
    sess.run(init)
    # Loop over test data
    for i in range(len(Xte)):
        # Get nearest neighbor
        # xtr: Xtr <- feed all training samples
        # xte: Xte[i,:] <- predict only one hand-written digit image
        # Get the index number of the closest number in dataset
        nn_index = sess.run(pred, feed_dict= {xtr: Xtr, xte: Xte[i,:]}) 
        print("images_data_Xtr:", Xtr) 
        print("labels_data_Ytr:", Ytr, "\n")
        # Get nearest neighbor class label and compare to its true label
        print("Test", i+1, "Prediction:", np.argmax(Ytr[nn_index]), \
                "True class", np.argmax(Yte[i]))
        print("nn_index:", nn_index, "Ytr[nn_index]:", Ytr[nn_index], "Yte[i]", Yte[i])
        # Calculate accuracy
        if np.argmax(Ytr[nn_index]) == np.argmax(Yte[i]):
            accuracy += 1./len(Xte)
        print("Accuracy:", accuracy,'\n')
    print("Done!")
    # Write a summary of computational graph
    writer = tf.summary.FileWriter("/home/alex504/TensorFlow_playground/02/nearest_neighbor/")
    writer.add_graph(sess.graph)