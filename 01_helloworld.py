"""
refrence aymericdamien/TensorFlow-Examples

Goal: Hello World using TensorFlow library

Date: 2017/08/08
"""
from __future__ import print_function
import tensorflow as tf

#Create a constant operator
hello = tf.constant('Hello, World!')

#Start session
sess = tf.Session()

#Run operator
print(sess.run(hello))
