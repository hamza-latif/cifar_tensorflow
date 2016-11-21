import tensorflow as tf
import numpy as np
from data import DataHandler
from network_defs import *

def conv_net(x):
	w_conv1a = weightVar([5, 5, 3, 64])
	b_conv1a = biasVar([3*64])#tf.Variable(tf.constant(0.1, shape=[16]))
	conv_1a = conv2dDepthwise(x, w_conv1a, [1, 1, 1, 1], name='conv1a')

	h_conv1a = tf.nn.relu(conv_1a + b_conv1a)

	h_conv1a = tf.nn.local_response_normalization(maxPool(h_conv1a))
	# h_conv1a = tf.nn.relu(tf.nn.depthwise_conv2d(x,W_conv1,strides=[1,1,1,1],padding='VALID',name='conv1') + b_conv1)
	w_conv1b = weightVar([5, 5, 3, 64])
	b_conv1b = biasVar([64])
	conv_1b = conv2dLayer(x, w_conv1b, [1, 1, 1, 1], name='conv1b')

	h_conv1b = tf.nn.relu(conv_1b + b_conv1b)
	h_conv1b = tf.nn.local_response_normalization(maxPool(h_conv1b))

	w_fc1 = weightVar([14 * 14 * 64 * 4, 1024])
	b_fc1 = biasVar([1024])

	h_conv1a_flat = tf.reshape(h_conv1a, [-1, 14 * 14 * 64 * 3])
	h_conv1b_flat = tf.reshape(h_conv1b, [-1, 14 * 14 * 64])

	h_conv1_flat = tf.concat(1, [h_conv1a_flat, h_conv1b_flat])
	h_fc1 = tf.nn.relu(tf.matmul(h_conv1_flat, w_fc1) + b_fc1)

	w_fc2 = weightVar([1024, 1024])
	b_fc2 = biasVar([1024])

	h_fc2 = tf.nn.relu(tf.matmul(h_fc1, w_fc2) + b_fc2)

	w_fc3 = weightVar([1024, 10])
	b_fc3 = biasVar([10])

	output = tf.matmul(h_fc2, w_fc3) + b_fc3
	return output

def full_net(x):

	input = tf.reshape(x, [-1,32*32*3])

	w_fc1 = tf.Variable(tf.random_normal([32 * 32 * 3, 1024], stddev=0.1))
	b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]))

	h_fc1 = tf.nn.relu(tf.matmul(input, w_fc1) + b_fc1)

	w_fc2 = tf.Variable(tf.random_normal([1024, 1024], stddev=0.1))
	b_fc2 = tf.Variable(tf.constant(0.1, shape=[1024]))

	h_fc2 = tf.nn.relu(tf.matmul(h_fc1, w_fc2) + b_fc2)

	w_fc3 = tf.Variable(tf.random_normal([1024, 10], stddev=0.1))
	b_fc3 = tf.Variable(tf.constant(0.1, shape=[10]))

	output = tf.matmul(h_fc2, w_fc3) + b_fc3
	return output


def train_nn(c_or_f, data_handler):

	batch_size = data_handler.mini_batch_size
	x = tf.placeholder('float', [None, 32, 32, 3])
	y = tf.placeholder('float', [None, 10])

	test_data, test_labels = data_handler.get_test_data()
	test_data = test_data.reshape([-1,32,32,3])

	ntrain = data_handler.train_size
	ntest = data_handler.meta['num_cases_per_batch']

	# ntrain = len(train_data)
	# ntest = len(test_data)

	if c_or_f == 0:
		prediction = conv_net(x)
	else:
		prediction = full_net(x)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, y))
	train_step = tf.train.AdamOptimizer().minimize(cost)
	correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())
		hm_epochs = 5000
		for epoch in range(hm_epochs):
			epoch_loss = 0
			for i in range(int(ntrain / batch_size)):
				# randindx = np.random.randint(ntrain, size=batch_size)
				# batch_data = train_data[randindx, :]
				# batch_labels = train_labels[randindx, :]
				batch_data, batch_labels = data_handler.get_next_mini_batch()
				print('batch_data_size', len(batch_data))
				batch_data = batch_data.reshape([-1,32,32,3])
				_, c = sess.run([train_step, cost], feed_dict={x: batch_data, y: batch_labels})
				epoch_loss += c
				#print('Epoch', epoch + 1, ' : Minibatch', i+1, ' out of ', ntrain/batch_size, ' Loss: ', c)
			#print('Epoch', epoch + 1, 'completed out of', hm_epochs, 'loss:', epoch_loss)
			if epoch % 100 == 0:
				print('Accuracy:', accuracy.eval({x: test_data, y: test_labels}))
		print('Accuracy:', accuracy.eval({x: test_data, y: test_labels}))

if __name__ == '__main__':
	import sys

	bl = sys.argv[1]
	nb = int(sys.argv[2])
	mbs = int(sys.argv[3])

	handler = DataHandler(bl,nb,mbs)

	train_nn(0,handler)
