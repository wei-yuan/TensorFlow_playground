#---------------------------------------------------------
# 2017/08/
#
#--------------------------------------------------------- 

from __future__ import print_function
import tensorflow as tf
import numpy as np

# Basic operations with variable as graph input
# The value was returned by the constructor represents the output 
# of the variable operations (defined as input when running session)
# tf Graph input
a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)

# define operations
add = tf.add(a,b)
mul = tf.multiply(a,b)

with tf.Session() as sess: 
    #run operations with variable input
    print("Addtion with variables: %i" % sess.run(add, feed_dict={a:5, b:10}) )
    print("Multiplication with variables: %i" % sess.run(mul, feed_dict={a:5, b:10}) )
    writer = tf.summary.FileWriter("/home/alex504/TensorFlow_playground/01/1")
    writer.add_graph(sess.graph)