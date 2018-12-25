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
    
    # Read train and test splits
    data_split_label_file = open(os.path.join(root_path, 'train_test_split.txt'), 'r').readlines()
    data_split_labels = [int(line.strip().split(' ')[1]) for line in data_split_label_file]
    
    # Load class labels
    image_class_label_file = open(os.path.join(root_path, 'image_class_labels.txt'), 'r').readlines()
    class_labels = [int(line.strip().split(' ')[1]) for line in image_class_label_file]
    
    # Zip the above
    data = zip(input_filenames, data_split_labels, class_labels)
    
    return data

def write_data(args):

    data = read_groundtruth_data(args.root_path)
    
    # Form train and test sets
    train_data=defaultdict(list)
    test_data = defaultdict(list)
    train_h5 = h5py.File('train_data.h5', 'w')
    test_h5 = h5py.File('test_data.h5', 'w')
    train_count = 0
    test_count = 0
    for i in range(len(data)):
        if i%500==0: print "Processed: {} images".format(i)
        label = data[i][2]
        train_flag = data[i][1]
        file_abs_path = os.path.join(args.root_path, 'images', data[i][0])
        if train_flag==1:
            resized_image = resize(io.imread(file_abs_path), [224, 224], preserve_range=True)
            if len(resized_image.shape) !=3: continue
            train_data[label].append(resized_image)
            train_count+=1
            
        elif train_flag==0:
            resized_image = resize(io.imread(file_abs_path), [224, 224], preserve_range=True)
            if len(resized_image.shape) !=3: continue
            test_data[label].append(resized_image)
            test_count+=1
        else:
            raise ValueError("Invalid train or test label")
    
    print "Total training images: {}".format(train_count)
    print "Total test images: {}".format(test_count)

    print "Creating train and test splits"
    for k, v in train_data.items():
        train_h5.create_dataset(str(k), data=v)

    for k, v in test_data.items():
        test_h5.create_dataset(str(k), data=v)
    print "Done writing train and test datasets"
    
    return train_data, test_data
    
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, default="/shared/kgcoe-research/mil/peri/birds_data/CUB_200_2011", help="Path to birds dataset")
    args = parser.parse_args()
    write_data(args)