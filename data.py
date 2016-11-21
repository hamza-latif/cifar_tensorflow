import numpy as np
import cPickle as pickle

class DataHandler:

	def __init__(self,batch_location,number_batches,mini_batch_size,one_hot=True):
		self.batch_location = batch_location
		self.number_batches = number_batches
		self.mini_batch_size = mini_batch_size
		self.current_batch = number_batches
		self.current_mini_batch = 0

		with open(batch_location + '/batches.meta') as f:
			self.meta = pickle.load(f)

		self.num_mini_batches = self.meta['num_cases_per_batch'] / mini_batch_size

		self.num_labels = len(self.meta['label_names'])

	#Return meta data
	def get_meta(self):
		return self.meta

	#Return batch data from batch file
	# def get_batch(self,folder,batch_num,one_hot=False,num_labels=0):
	# 	data_file = open(folder+"/data_batch_" + str(batch_num),'r')
	#
	# 	batch = pickle.load(data_file)
	#
	# 	data_file.close()
	#
	# 	batch_data = batch['data']
	# 	batch_labels = np.array(batch['labels'])
	#
	# 	if one_hot:
	# 		oh = np.zeros(len(batch_labels),num_labels)
	#
	# 		oh[np.arange(len(batch_labels)),batch_labels] = 1
	#
	# 		batch_labels = oh
	#
	# 	return batch_data, batch_labels

	def next_batch(self):
		self.current_batch = self.current_batch + 1

		if self.current_batch > self.number_batches:
			self.current_batch = 1

		with open(self.batch_location + '/data_batch_' + str(self.current_batch)) as f:
			self.current_batch_data = pickle.load(f)

		self.batch_data = self.current_batch_data['data']
		self.batch_labels = np.array(batch['labels'])

		if one_hot:
			oh = np.zeros(len(self.batch_labels),self.num_labels)

			oh[np.arange(len(self.batch_labels)),self.batch_labels] = 1

			self.batch_labels = oh

	def shuffle_batch(self):
		ind = np.arange(self.meta['num_cases_per_batch'])

		np.random.shuffle(ind)

		self.batch_data = self.batch_data[ind]
		self.batch_labels = self.batch_labels[ind]

	def get_next_mini_batch(self):
		if self.current_mini_batch == 0:
			self.next_batch()
			self.shuffle_batch()

		start = self.mini_batch_size*self.current_batch_data
		end = start + self.mini_batch_size
		mini_batch_data = self.batch_data[start:end]
		mini_batch_labels = self.batch_labels[start:end]

		self.current_mini_batch = self.current_mini_batch + 1

		return mini_batch_data, mini_batch_labels

