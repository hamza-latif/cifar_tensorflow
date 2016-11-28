import numpy as np
import cPickle as pickle

class DataHandler:

	def __init__(self,batch_location,number_batches,mini_batch_size,one_hot=True):
		self.batch_location = batch_location
		self.number_batches = number_batches
		self.mini_batch_size = mini_batch_size
		self.current_batch = number_batches
		self.current_mini_batch = 0
		self.one_hot = one_hot

		with open(batch_location + '/batches.meta') as f:
			self.meta = pickle.load(f)

		self.train_size = self.meta['num_cases_per_batch']*self.number_batches

		self.num_mini_batches = self.meta['num_cases_per_batch'] / mini_batch_size

		self.num_labels = len(self.meta['label_names'])

		self.init_test_data()

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
		self.batch_labels = np.array(self.current_batch_data['labels'])

		if self.one_hot:
			oh = np.zeros((len(self.batch_labels),self.num_labels))

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

		start = self.mini_batch_size*self.current_mini_batch
		end = start + self.mini_batch_size
		mini_batch_data = self.batch_data[start:end]
		mini_batch_labels = self.batch_labels[start:end]

		self.current_mini_batch = (self.current_mini_batch + 1) % self.num_mini_batches

		return mini_batch_data, mini_batch_labels

	def init_test_data(self):
		with open(self.batch_location + '/test_batch') as f:
			self.test_batch_data = pickle.load(f)

		self.test_data = self.test_batch_data['data']
		self.test_labels = np.array(self.test_batch_data['labels'])
		self.test_batch = 0

		if self.one_hot:
			oh = np.zeros((len(self.test_labels),self.num_labels))

			oh[np.arange(len(self.test_labels)),self.test_labels] = 1

			self.test_labels = oh

	def get_test_data(self):
		return self.test_data, self.test_labels

	def get_next_mini_test_batch(self):
		start = self.mini_batch_size*self.test_batch
		end = start + self.mini_batch_size
		mini_batch_data = self.test_data[start:end]
		mini_batch_labels = self.test_labels[start:end]

		self.test_batch = (self.test_batch + 1) % self.num_mini_batches

		return mini_batch_data, mini_batch_labels

	def get_all_train_data(self):
		train_x = np.zeros([0,3072])
		train_y = []

		for i in range(self.number_batches):
			with open(self.batch_location + '/data_batch_' + str(i+1)) as df:
				data = pickle.load(df)
				train_x = np.concatenate((train_x,data['data']))
				train_y = train_y + data['labels']

		return train_x, train_y

def test(batch_location,number_batches,mini_batch_size,one_hot=True):
	tester = DataHandler(batch_location,number_batches,mini_batch_size,one_hot)

	d,l = tester.get_next_mini_batch()

	print d
	print l

if __name__ == '__main__':
	import sys
	bl = sys.argv[1]
	nb = int(sys.argv[2])
	mbs = int(sys.argv[3])
	test(bl,nb,mbs)