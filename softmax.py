import tensorflow as tf
import numpy as np
from time import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def timeit(func):
	def wrapper(*args, **kwargs):
		start_time=time();
		result=func(*args, **kwargs)
		total_time=(time()-start_time)
		print('time', total_time)
		return result
	return wrapper
def softmax_implementation(mat, mask, s):
	num = tf.multiply(tf.exp(mat), mask)
	den = tf.reshape(tf.reduce_sum(num, reduction_indices=[1]), shape=[s,1])
	return tf.divide(num, den)

@timeit
def f(l):
	s=len(l)
	ip=tf.placeholder(tf.float32, shape=[1, s])
	mat=tf.tile(ip, [s, 1])
	temp_ones=tf.ones([s,s], tf.float32)
	mask=tf.reverse(tf.matrix_band_part(temp_ones,0,s), axis=[1])
	final=softmax_implementation(mat, mask, s)
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		ans=sess.run(final, feed_dict={ip:np.array(l).reshape(1,s)})
	return ans

if __name__=='__main__':
	test_vec=np.array([1, 2, 3, 4])
	print(f(test_vec))