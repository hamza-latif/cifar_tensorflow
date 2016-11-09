import tensorflow as tf
import numpy as np
import cPickle as pickle


# filename_queue = tf.train.string_input_producer(["sat.trn", "sat.tst"])

# reader = tf.TextLineReader()
# key, value = reader.read(filename_queue)

# Default values, in case of empty columns. Also specifies the type of the
# decoded result.
# record_defaults = [[1] for i in range(37)]
# col = tf.decode_csv(
#   value,record_defaults, field_delim = ' ')
# labelcol = col[36]
# features = tf.pack(col[:36])

# with tf.Session() as sess:
# Start populating the filename queue.
# coord = tf.train.Coordinator()
# threads = tf.train.start_queue_runners(coord=coord)

# for i in range(5000):
# Retrieve a single instance:
# example, label = sess.run([features, labelcol])
# print example,'\n', label

# coord.request_stop()
# coord.join(threads)

def get_meta(folder,filename):

	meta_file = open(folder + '/' + filename,'r')

	_meta_ = pickle.load(meta_file)

	meta_file.close()

	return _meta_

def get_batch(folder,batch_num,one_hot=False,num_labels=0):
	data_file = open(folder+"/data_batch_" + str(batch_num),'r')

	batch = pickle.load(data_file)

	data_file.close()

	batch_data = batch['data']
	batch_labels = np.array(batch['labels'])

	if one_hot:
		oh = np.zeros(len(batch_labels),num_labels)

		oh[np.arange(len(batch_labels)),batch_labels] = 1

		batch_labels = oh

	return batch_data, batch_labels

def weightVar(dim):
	return tf.Variable(tf.random_normal(dim,stddev=0.1))

def biasVar(dim,init_value=0.1):
	return tf.Variable(tf.constant(init_value,shape=dim))

def conv2dLayer(input,weights,stride,name='',padding='VALID'):
	return tf.nn.conv2d(input,weights,strides=stride,padding=padding,name=name)

def conv2dDepthwise(input,weights,stride,name='',padding='VALID'):
	return tf.nn.depthwise_conv2d(input,weights,strides=stride,padding=padding,name=name)


def conv_net(x):
	w_conv1a = weightVar([2, 2, 4, 4])
	b_conv1a = biasVar([16])#tf.Variable(tf.constant(0.1, shape=[16]))
	conv_1a = conv2dDepthwise(x, w_conv1a, [1, 1, 1, 1], name='conv1')

	h_conv1a = tf.nn.relu(conv_1a + b_conv1a)
	# h_conv1a = tf.nn.relu(tf.nn.depthwise_conv2d(x,W_conv1,strides=[1,1,1,1],padding='VALID',name='conv1') + b_conv1)
	w_conv1b = weightVar([1, 1, 4, 8])
	b_conv1b = biasVar([8])
	conv_1b = conv2dLayer(x, w_conv1b, [1, 1, 1, 1], name='conv2')

	h_conv1b = tf.nn.relu(conv_1b + b_conv1b)

	w_fc1 = weightVar([2 * 2 * 4 * 4 + 3 * 3 * 8, 128])
	b_fc1 = biasVar([128])

	h_conv1a_flat = tf.reshape(h_conv1a, [-1, 2 * 2 * 4 * 4])
	h_conv1b_flat = tf.reshape(h_conv1b, [-1, 3 * 3 * 8])

	h_conv1_flat = tf.concat(1, [h_conv1a_flat, h_conv1b_flat])
	h_fc1 = tf.nn.relu(tf.matmul(h_conv1_flat, w_fc1) + b_fc1)

	w_fc2 = weightVar([128, 128])
	b_fc2 = biasVar([128])

	h_fc2 = tf.nn.relu(tf.matmul(h_fc1, w_fc2) + b_fc2)

	w_fc3 = weightVar([128, 6])
	b_fc3 = biasVar([6])

	output = tf.matmul(h_fc2, w_fc3) + b_fc3
	return output

def full_net(x):

	input = tf.reshape(x, [-1,3*3*4])

	w_fc1 = tf.Variable(tf.random_normal([3 * 3 * 4, 512], stddev=0.1))
	b_fc1 = tf.Variable(tf.constant(0.1, shape=[512]))

	h_fc1 = tf.nn.relu(tf.matmul(input, w_fc1) + b_fc1)

	w_fc2 = tf.Variable(tf.random_normal([512, 512], stddev=0.1))
	b_fc2 = tf.Variable(tf.constant(0.1, shape=[512]))

	h_fc2 = tf.nn.relu(tf.matmul(h_fc1, w_fc2) + b_fc2)

	w_fc3 = tf.Variable(tf.random_normal([512, 6], stddev=0.1))
	b_fc3 = tf.Variable(tf.constant(0.1, shape=[6]))

	output = tf.matmul(h_fc2, w_fc3) + b_fc3
	return output


def train_nn(c_or_f):
	batch_size = 500
	x = tf.placeholder('float', [None, 3, 3, 4])
	y = tf.placeholder('float', [None, 6])

	train_data, train_labels = process_data('sat.trn')
	test_data, test_labels = process_data('sat.tst')

	ntrain = train_data.shape[0]
	ntest = test_data.shape[0]

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
			for _ in range(int(ntrain / batch_size)):
				randindx = np.random.randint(ntrain, size=batch_size)
				batch_data = train_data[randindx, :]
				batch_labels = train_labels[randindx, :]
				_, c = sess.run([train_step, cost], feed_dict={x: batch_data, y: batch_labels})
				epoch_loss += c
			print('Epoch', epoch + 1, 'completed out of', hm_epochs, 'loss:', epoch_loss)
			if epoch % 100 == 0:
				print('Accuracy:', accuracy.eval({x: test_data, y: test_labels}))
		print('Accuracy:', accuracy.eval({x: test_data, y: test_labels}))


train_nn(1)
