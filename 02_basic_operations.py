"""
refrence aymericdamien/TensorFlow-Examples

Goal: Basic operations using TensorFlow library

Date: 2017/08/08
"""
from __future__ import print_function
import tensorflow as tf
import numpy as np

# Basic constant operations
a = tf.constant(5)
b = tf.constant(10)

# Start session
with tf.Session() as sess:
    print("a = 5,b = 10")
    print("tf.constant a add tf.constant b : %i" % sess.run(a+b))
    print("tf.constant a multiply tf.constant b : %i" % sess.run(a*b))

# Basic operations with variable as graph input
# The value was returned by the constructor represents the output 
# of the variable operations (defined as input when running session)
# tf Graph input
a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)

# define operations
add = tf.add(a,b)
mul = tf.multiply(a,b)

# Launch the default graph
# with...as... indentation - coding style
with tf.Session() as sess: 
    #run operations with variable input
    print("Addtion with variables: %i" % sess.run(add, feed_dict={a:5, b:10}) )
    print("Multiplication with variables: %i" % sess.run(mul, feed_dict={a:5, b:10}) )

# Matrix Multiplication practice
# Create a matrix of rank 2
matrix1 = tf.constant([[3.,3.]])
# Create a matrix of 2 x 1 with constant
matrix2 = tf.constant([[2.],[2.]]) 

matrix3 = tf.reshape(np.arange(10),[2,5])

with tf.Session() as sess:
    print("matrix1: ")
    print(sess.run(matrix1))
    print("matrix2: ")
    print(sess.run(matrix2))
    print("matrix3: ")
    print(sess.run(matrix3))    
    print("rank: ",sess.run(tf.rank(matrix3))," size: ",sess.run(tf.size(matrix3)))

# Create a matmul operation that take matrix1 and matirx2 as input
# product represents to the result of matmul
product = tf.matmul(matrix1,matrix2)

# Output result of operation
with tf.Session() as sess:
    result = sess.run(product)
    print(result)

