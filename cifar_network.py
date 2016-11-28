import tensorflow as tf
import tflearn
import numpy as np
from data import DataHandler
from network_defs import *
import time
import os

# def conv_net(x):
# 	w_conv1a = weightVar([5, 5, 3, 64])
# 	b_conv1a = biasVar([3*64])#tf.Variable(tf.constant(0.1, shape=[16]))
# 	conv_1a = conv2dDepthwise(x, w_conv1a, [1, 1, 1, 1], name='conv1a')
#
# 	h_conv1a = tf.nn.relu(conv_1a + b_conv1a)
#
# 	h_conv1a = tf.nn.local_response_normalization(maxPool(h_conv1a))
# 	# h_conv1a = tf.nn.relu(tf.nn.depthwise_conv2d(x,W_conv1,strides=[1,1,1,1],padding='VALID',name='conv1') + b_conv1)
# 	w_conv1b = weightVar([5, 5, 3, 64])
# 	b_conv1b = biasVar([64])
# 	conv_1b = conv2dLayer(x, w_conv1b, [1, 1, 1, 1], name='conv1b')
#
# 	h_conv1b = tf.nn.relu(conv_1b + b_conv1b)
# 	h_conv1b = tf.nn.local_response_normalization(maxPool(h_conv1b))
#
# 	w_fc1 = weightVar([14 * 14 * 64 * 4, 1024])
# 	b_fc1 = biasVar([1024])
#
# 	h_conv1a_flat = tf.reshape(h_conv1a, [-1, 14 * 14 * 64 * 3])
# 	h_conv1b_flat = tf.reshape(h_conv1b, [-1, 14 * 14 * 64])
#
# 	h_conv1_flat = tf.concat(1, [h_conv1a_flat, h_conv1b_flat])
# 	h_fc1 = tf.nn.relu(tf.matmul(h_conv1_flat, w_fc1) + b_fc1)
#
# 	w_fc2 = weightVar([1024, 1024])
# 	b_fc2 = biasVar([1024])
#
# 	h_fc2 = tf.nn.relu(tf.matmul(h_fc1, w_fc2) + b_fc2)
#
# 	w_fc3 = weightVar([1024, 10])
# 	b_fc3 = biasVar([10])
#
# 	output = tf.matmul(h_fc2, w_fc3) + b_fc3
# 	return output
#
# def full_net(x):
#
# 	input = tf.reshape(x, [-1,32*32*3])
#
# 	w_fc1 = tf.Variable(tf.random_normal([32 * 32 * 3, 1024], stddev=0.1))
# 	b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]))
#
# 	h_fc1 = tf.nn.relu(tf.matmul(input, w_fc1) + b_fc1)
#
# 	w_fc2 = tf.Variable(tf.random_normal([1024, 1024], stddev=0.1))
# 	b_fc2 = tf.Variable(tf.constant(0.1, shape=[1024]))
#
# 	h_fc2 = tf.nn.relu(tf.matmul(h_fc1, w_fc2) + b_fc2)
#
# 	w_fc3 = tf.Variable(tf.random_normal([1024, 10], stddev=0.1))
# 	b_fc3 = tf.Variable(tf.constant(0.1, shape=[10]))
#
# 	output = tf.matmul(h_fc2, w_fc3) + b_fc3
# 	return output
#
# def conv_net2(x):
# 	w_conv1 = weightVar([5,5,3,64])
# 	b_conv1 = biasVar([64])
#
# 	conv1 = conv2dLayer(x,w_conv1,[1,1,1,1],name='conv1')
#
# 	h_conv1 = tf.nn.relu(conv1+b_conv1)
# 	pool1 = maxPool(h_conv1)
# 	h_conv1_norm = tf.nn.local_response_normalization(pool1)
#
# 	w_conv2 = weightVar([3,3,64,32])
# 	b_conv2 = biasVar([32])
#
# 	conv2 = conv2dLayer(h_conv1_norm,w_conv2,[1,1,1,1],name='conv2',padding='SAME')
#
# 	h_conv2 = tf.nn.relu(conv2 + b_conv2)
#
# 	pool2 = maxPool(h_conv2)
# 	h_conv2_norm = tf.nn.local_response_normalization(pool2)
#
# 	h_conv2_flat = tf.reshape(h_conv2_norm,[-1,7*7*32])
#
# 	w_fc1 = weightVar([7*7*32,1024])
# 	b_fc1 = biasVar([1024])
#
# 	h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat,w_fc1) + b_fc1)
#
# 	w_fc2 = weightVar([1024, 1024])
# 	b_fc2 = biasVar([1024])
#
# 	h_fc2 = tf.nn.relu(tf.matmul(h_fc1,w_fc2) + b_fc2)
#
# 	w_fc3 = weightVar([1024, 10])
# 	b_fc3 = biasVar([10])
#
# 	h_fc3 = tf.matmul(h_fc2,w_fc3) + b_fc3
#
# 	return h_fc3
#
# def tflearn_convnet(x):
# 	conv1 = tflearn.conv_2d(x,64,5,activation='relu',name='conv1')
#
# 	pool1 = tflearn.max_pool_2d(conv1,3,2,name='pool1')
#
# 	norm1 = tflearn.local_response_normalization(pool1,name='norm1')
#
# 	conv2 = tflearn.conv_2d(norm1,64,5,activation='relu',name='conv2')
#
# 	norm2 = tflearn.local_response_normalization(conv2, name='norm2')
#
# 	pool2 = tflearn.max_pool_2d(norm2,3,2,name='pool2')
#
# 	flat = tflearn.flatten(pool2)
#
# 	fc1 = tflearn.fully_connected(flat,384,activation='relu',name='fc1')
#
# 	fc2 = tflearn.fully_connected(fc1,192,activation='relu',name='fc2')
#
# 	fc3 = tflearn.fully_connected(fc2,10,name='fc3')
#
# 	return fc3
#
# def net3(x):
#
# 	conv1 = tflearn.conv_2d(x, 64, 5, activation='relu',regularizer='L2', name='conv1')
#
# 	pool1 = tflearn.max_pool_2d(conv1, 2, name='pool1')
#
# 	norm1 = tflearn.local_response_normalization(pool1, name='norm1')
#
# 	conv2 = tflearn.conv_2d(norm1, 32, 3, activation='relu',regularizer='L2', name='conv2')
#
# 	pool2 = tflearn.max_pool_2d(conv2, 2, name='pool2')
#
# 	norm2 = tflearn.local_response_normalization(pool2, name='norm2')
#
# 	flat = tflearn.flatten(norm2)
#
# 	do1 = tflearn.dropout(flat,.5)
#
# 	fc1 = tflearn.fully_connected(do1, 1024, activation='relu', name='fc1')
#
# 	do2 = tflearn.dropout(fc1,.5)
#
# 	fc2 = tflearn.fully_connected(do2, 1024, activation='relu', name='fc2')
#
# 	do3 = tflearn.dropout(fc2,.5)
#
# 	fc3 = tflearn.fully_connected(do3, 10, name='fc3')
#
# 	return fc3

def example_net(x):
	network = tflearn.conv_2d(x, 32, 3, activation='relu')
	network = tflearn.max_pool_2d(network, 2)
	network = tflearn.conv_2d(network, 64, 3, activation='relu')
	network = tflearn.conv_2d(network, 64, 3, activation='relu')
	network = tflearn.max_pool_2d(network, 2)
	network = tflearn.fully_connected(network, 512, activation='relu')
	network = tflearn.dropout(network, 0.5)
	network = tflearn.fully_connected(network, 10, activation='softmax')

	return network

# def train_nn(c_or_f, data_handler):
#
# 	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
#
# 	batch_size = data_handler.mini_batch_size
#
# 	img_prep = tflearn.ImagePreprocessing()
# 	img_prep.add_featurewise_zero_center()
# 	img_prep.add_featurewise_stdnorm()
#
# 	img_aug = tflearn.ImageAugmentation()
# 	img_aug.add_random_flip_leftright()
# 	img_aug.add_random_rotation(max_angle=25)
#
# 	x = tflearn.input_data(shape=[None,32,32,3],dtype='float', data_preprocessing=img_prep, data_augmentation=img_aug)
# 	#x = tf.placeholder('float', [None, 32, 32, 3])
# 	y = tf.placeholder('float', [None, 10])
#
# 	#test_data, test_labels = data_handler.get_test_data()
# 	#test_data = test_data.reshape([-1,32,32,3])
#
# 	ntrain = data_handler.train_size
# 	ntest = data_handler.meta['num_cases_per_batch']
#
# 	# ntrain = len(train_data)
# 	# ntest = len(test_data)
#
# 	#if c_or_f == 0:
# 	#prediction = net3(x)
# 	prediction = example_net(x)
# 	#else:
# 	#	prediction = full_net(x)
# 	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, y))
# 	train_step = tf.train.AdamOptimizer(learning_rate=0.001,epsilon=0.01).minimize(cost)
# 	#train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cost)
# 	#train_step = tf.train.MomentumOptimizer(.001,.0001).minimize(cost)
# 	correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
# 	accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
#
# 	#saver = tf.train.Saver()
#
# 	#if not os.path.isdir("/tmp/cifar"):
# 	#	os.mkdir("/tmp/cifar")
#
# 	with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
# 		sess.run(tf.initialize_all_variables())
# 		hm_epochs = 5000
# 		for epoch in range(hm_epochs):
# 			epoch_loss = 0
# 			start_time = time.time()
# 			for i in range(int(ntrain / batch_size)):
# 				# randindx = np.random.randint(ntrain, size=batch_size)
# 				# batch_data = train_data[randindx, :]
# 				# batch_labels = train_labels[randindx, :]
# 				batch_data, batch_labels = data_handler.get_next_mini_batch()
# 				#print('batch_data_size', len(batch_data))
# 				batch_data = np.dstack((batch_data[:, :1024], batch_data[:, 1024:2048], batch_data[:, 2048:]))
# 				batch_data = batch_data.reshape([-1,32,32,3])/(255.0/2) - 1.0
# 				_, c = sess.run([train_step, cost], feed_dict={x: batch_data, y: batch_labels})
# 				epoch_loss += c
# 				#print('Epoch', epoch + 1, ' : Minibatch', i+1, ' out of ', ntrain/batch_size, ' Loss: ', c)
# 			duration = time.time() - start_time
# 			print('Epoch', epoch + 1, 'completed out of', hm_epochs, 'loss:', epoch_loss, 'time:', duration)
# 			if epoch % 10 == 0:
# 				trc = 0
# 				for i in range(int(ntrain / batch_size)):
# 					batch_data, batch_labels = data_handler.get_next_mini_batch()
# 					batch_data = np.dstack((batch_data[:,:1024],batch_data[:,1024:2048],batch_data[:,2048:]))
# 					batch_data = batch_data.reshape([-1, 32, 32, 3])/(255.0/2) - 1.0
# 					c = accuracy.eval({x: batch_data, y: batch_labels})
# 					trc += c
# 				tc = 0
# 				for i in range(ntest/data_handler.mini_batch_size):
# 					test_data, test_labels = data_handler.get_next_mini_test_batch()
# 					test_data = np.dstack((test_data[:, :1024], test_data[:, 1024:2048], test_data[:, 2048:]))
# 					test_data = test_data.reshape([-1, 32, 32, 3])/(255.0/2) - 1.0
# 					tc += accuracy.eval({x: test_data, y: test_labels})
# 				print('Accuracy test:', tc/(ntest/data_handler.mini_batch_size), 'Accuracy train:', trc/(ntrain/data_handler.mini_batch_size))
#
# 	#		if epoch % 50 == 0:
# 	#			save_path = saver.save(sess,"/tmp/cifar/model_epoch" + str(epoch) + ".ckpt")
# 		tc = 0
# 		for i in range(ntest / data_handler.mini_batch_size):
# 			test_data, test_labels = data_handler.get_next_mini_test_batch()
# 			test_data = test_data.reshape([-1, 32, 32, 3])/(255.0/2) - 1.0
# 			tc += accuracy.eval({x: test_data, y: test_labels})
# 		print('Accuracy:', tc / (ntest / data_handler.mini_batch_size))

def trythisnet(x):
	network = tflearn.conv_2d(x,64,5,activation='relu')
	network = tflearn.max_pool_2d(network,3,2)
	#network = tflearn.local_response_normalization(network,4,alpha=0.001/9.0)
	network = tflearn.conv_2d(network,64,5,activation='relu')
	#network = tflearn.local_response_normalization(network,4,alpha=0.001/9.0)
	network = tflearn.max_pool_2d(network,3,2)
	network = tflearn.fully_connected(network,384,activation='relu',weight_decay=0.004)
	network = tflearn.fully_connected(network,192,activation='relu',weight_decay=0.004)
	network = tflearn.fully_connected(network,10,activation='linear',weight_decay=0.0)

	return network

def train_nn_tflearn(data_handler):

	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)

	batch_size = data_handler.mini_batch_size

	img_prep = tflearn.ImagePreprocessing()
	img_prep.add_featurewise_zero_center()
	img_prep.add_featurewise_stdnorm()

	img_aug = tflearn.ImageAugmentation()
	img_aug.add_random_flip_leftright()
	img_aug.add_random_rotation(max_angle=25)

	x = tflearn.input_data(shape=[None, 32, 32, 3], dtype='float', data_preprocessing=img_prep,
						   data_augmentation=img_aug)
	# x = tf.placeholder('float', [None, 32, 32, 3])
	#y = tf.placeholder('float', [None, 10])

	# test_data, test_labels = data_handler.get_test_data()
	# test_data = test_data.reshape([-1,32,32,3])

	ntrain = data_handler.train_size
	ntest = data_handler.meta['num_cases_per_batch']

	# from tflearn.datasets import cifar10
	# (X, Y), (X_test, Y_test) = cifar10.load_data(dirname="/home/hamza/meh/bk_fedora24/Documents/tflearn_example/cifar-10-batches-py")
	# X, Y = tflearn.data_utils.shuffle(X, Y)
	# Y = tflearn.data_utils.to_categorical(Y, 10)
	# Y_test = tflearn.data_utils.to_categorical(Y_test, 10)

	X, Y = data_handler.get_all_train_data()

	X, Y = tflearn.data_utils.shuffle(X, Y)

	X = np.dstack((X[:, :1024], X[:, 1024:2048], X[:, 2048:]))

	X = X/255.0

	X = X.reshape([-1,32,32,3])

	Y = tflearn.data_utils.to_categorical(Y,10)

	X_test, Y_test = data_handler.get_test_data()

	X_test = np.dstack((X_test[:, :1024], X_test[:, 1024:2048], X_test[:, 2048:]))

	X_test = X_test/255.0

	X_test = X_test.reshape([-1,32,32,3])

	#network = tflearn.regression(net3(x),optimizer='adam',loss='categorical_crossentropy',learning_rate=0.001)
	network = tflearn.regression(trythisnet(x),optimizer='sgd',loss='categorical_crossentropy',learning_rate=0.001)

	print np.shape(X)
	print np.shape(Y)
	print network

	model = tflearn.DNN(network,tensorboard_verbose=0)
	model.fit(X, Y, n_epoch=50, shuffle=True, validation_set=(X_test, Y_test),
			  show_metric=True, batch_size=data_handler.mini_batch_size, run_id='cifar10_cnn')

if __name__ == '__main__':
	import sys

	bl = sys.argv[1]
	nb = int(sys.argv[2])
	mbs = int(sys.argv[3])

	handler = DataHandler(bl,nb,mbs)
	#train_nn(0,handler)
	train_nn_tflearn(handler)
