import numpy as np
import cPickle as pickle

class DataHandler:

	def __init__(self,batch_location,number_batches,mini_batch_size):
		self.batch_location = batch_location
		self.number_batches = number_batches
		self.mini_batch_size = mini_batch_size
		self.current_batch = 1
		self.current_mini_batch = 0

		with open(batch_location + '/batches.meta') as f:
			self.meta = pickle.load(f)

		self.num_mini_batches = meta['num_cases_per_batch'] / mini_batch_size

		with open(batch_location + '/data_batch_' + str(self.current_batch)) as f:
			self.current_batch_data = pickle.load(f)

	#Return meta data
	def get_meta(self):
		return self.meta

	#Return batch data from batch file
	def get_batch(self,folder,batch_num,one_hot=False,num_labels=0):
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

	def get_next_mini_batch(self):
		pass
