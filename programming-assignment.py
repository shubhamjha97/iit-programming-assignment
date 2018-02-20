import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # ignore tensorflow warnings

def softmax_implementation(mat, mask, s):
	'''
	Custom implementation of the softmax function.

	Inputs: mat : the matrix which contains the input vector stacked s times vertically
			mask : triangular mask
			s : length of the test list
	Outputs: A matrix with the row-wise softmax calculated on the nonzero values
	'''
	num = tf.multiply(tf.exp(mat), mask) # apply mask to matrix and calculate the numerator
	den = tf.reshape(tf.reduce_sum(num, reduction_indices=[1]), shape=[s,1]) # calculate denominator by summing all the numerators in a particular row
	return tf.divide(num, den)

def f(l):
	'''
	Function implements the required graph.

	Inputs: l: The input list
	Outputs: Tensorflow graph operation for computing required function
	'''
	s=len(l) # find length of test  list
	ip=tf.placeholder(tf.float32, shape=[1, s]) # create input placeholder
	mat=tf.tile(ip, [s, 1]) # stack the input array vertically s times
	temp_ones=tf.ones([s,s], tf.float32)

	''' The following line of code creates a mask consisting of 1s in the upper
	 triangle and 0s in the lower triangle, and then reverses it to create a 
	 mask that looks something like this (s=4):
	 [[ 1.  1.  1.  1.]
	  [ 1.  1.  1.  0.]
	  [ 1.  1.  0.  0.]
	  [ 1.  0.  0.  0.]] '''

	mask=tf.reverse(tf.matrix_band_part(temp_ones,0,s), axis=[1])

	# finally, calculate the softmax of the masked matrix using the softmax_implementation function
	final=softmax_implementation(mat, mask, s)

	# Run the above graph with test list as the input
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		res=sess.run(final, feed_dict={ip:np.array(l).reshape(1,s)})
	return res

if __name__=='__main__':
	test_list=[1,2,3,4] # test list
	print(f(test_list)) # print result