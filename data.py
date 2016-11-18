import numpy as np
import cPickle as pickle

#Return unpickled meta file
def get_meta(folder,filename):

	meta_file = open(folder + '/' + filename,'r')

	_meta_ = pickle.load(meta_file)

	meta_file.close()

	return _meta_

#Return batch data from batch file
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

