import numpy as np
import argparse
import os
import h5py
from collections import defaultdict
from skimage import io
from skimage.transform import resize
import pdb

def read_groundtruth_data(root_path):
    # Read all image file names
    input_file = open(os.path.join(root_path, 'images.txt'), 'r').readlines()
    input_filenames = [line.strip().split(' ')[1] for line in input_file]
    
    # Load class labels
    image_class_label_file = open(os.path.join(root_path, 'image_class_labels.txt'), 'r').readlines()
    class_labels = [int(line.strip().split(' ')[1]) for line in image_class_label_file]
    
    # Zip the above
    data = zip(input_filenames, class_labels)
    
    return data
    
def write_positive_pairs(images, name):
    """
    Write Positive pairs in text file
    """
    file = open(name, 'w')
    keys = images.keys()
    class_count=0
    try:
        for key in images.keys():
            print "Number of classes written: {}".format(class_count)
            assert len(images[key]) > 0, "Number of images per class should be greater than zero"
            for i in range(len(images[key])):
                for j in range(i+1, len(images[key])):
					file.write(str(images[key][i]) + ' ' + str(images[key][j]) + ' ' + str(key) + '\n')
            class_count+=1
    except:
		print "Unexpected error in forming positive pairs"
            
    file.close()
    print "Done writing positive pairs"
    
    
def write_data(args):

	data = read_groundtruth_data(args.root_path)

	# Form train and test sets
	train_data=defaultdict(list)
	test_data = defaultdict(list)

	# First 100 class images are used for training and rest are used for testing
	index = 5864
		
	# Gather all training images with labels
	print "Processing training set of first 100 classes"
	for i in range(index):
		if i%500==0: print "Processed: {} images".format(i)
		label = data[i][1]
		file_abs_path = os.path.join(args.root_path, 'images', data[i][0])
		resized_image = io.imread(file_abs_path)
		# Ignore gray scale images
		if len(resized_image.shape) !=3: continue
		train_data[label].append(file_abs_path)
		
	# Gather all testing images with labels
	print "---------------------------------------------"
	print "Processing test set of remaining 100 classes"
	for i in range(index, len(data)):
		if i%500==0: print "Processed: {} images".format(i)
		label = data[i][1]
		file_abs_path = os.path.join(args.root_path, 'images', data[i][0])
		resized_image = io.imread(file_abs_path)
		# Ignore gray scale images
		if len(resized_image.shape) !=3: continue
		test_data[label].append(file_abs_path)

	print "Total training images: {}".format(index)
	print "Total test images: {}".format(len(data) - index)

	write_positive_pairs(train_data, 'train_pos_pairs.txt')
	write_positive_pairs(test_data, 'test_pos_pairs.txt')

	print "Done writing train and test datasets"

    
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, default="/shared/kgcoe-research/mil/peri/birds_data/CUB_200_2011", help="Path to birds dataset")
    args = parser.parse_args()
    write_data(args)