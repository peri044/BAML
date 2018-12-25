import numpy as np
import h5py
import os
import pdb

def load_data():
    root_path="/shared/kgcoe-research/mil/peri/birds_data"
    train_data = h5py.File(os.path.join(root_path, 'train_data.h5'), 'r')
    test_data = h5py.File(os.path.join(root_path, 'test_data.h5'), 'r')
    
    return train_data, test_data
	
def compute_accuracy(predictions, groundtruth):
	"""
	Computes classification accuracy
	"""
	# pdb.set_trace()
	num_samples = predictions.shape[0]
	correct = 0
	for i in range(num_samples):
		if predictions[i]==int(groundtruth[i]):
			correct+=1
	
	return float(correct)/num_samples