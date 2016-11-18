import tensorflow as tf
import numpy as np

def weightVar(dim):
	return tf.Variable(tf.random_normal(dim,stddev=0.1))

def biasVar(dim,init_value=0.1):
	return tf.Variable(tf.constant(init_value,shape=dim))

def conv2dLayer(input,weights,stride,name='',padding='VALID'):
	return tf.nn.conv2d(input,weights,strides=stride,padding=padding,name=name)

def conv2dDepthwise(input,weights,stride,name='',padding='VALID'):
	return tf.nn.depthwise_conv2d(input,weights,strides=stride,padding=padding,name=name)

def maxPool(input,ksize=[1,2,2,1],stride=[1,2,2,1],padding='VALID'):
	return tf.nn.max_pool(input,ksize,stride,padding)