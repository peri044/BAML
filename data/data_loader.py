import tensorflow as tf
import numpy as np
import random
import os
import pdb

imagenet_means_rgb = [123.68, 116.78, 103.94]

class DataLoader(object):
    """
    Data loading, Preprocessing, Reading and writing TF record functionality
    """
    def __init__(self, data_path='train_pos_pairs.txt', record_path='/shared/kgcoe-research/mil/peri/birds_data/birds_ob_test_mask.tfrecord', batch_size=16, num_epochs=10, mode='train'):
        self.data_path = data_path
        self.record_path = record_path
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.mode = mode
        
    def _make_single_example(self, data_and_labels):
        image = tf.gfile.FastGFile(os.path.join('/shared/kgcoe-research/mil/peri/birds_data/CUB_200_2011/images', data_and_labels[0]), "rb").read()
        mask_file_name=data_and_labels[0].split('/')[1]+'_seg'+'.jpg'
        mask_root_path='/shared/kgcoe-research/mil/peri/birds_data/CUB_200_2011/seg_masks'
        mask = tf.gfile.FastGFile(os.path.join(mask_root_path, mask_file_name), "rb").read()
        features_dict = {"image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
                         "mask": tf.train.Feature(bytes_list=tf.train.BytesList(value=[mask])),
                         "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[int(data_and_labels[1])]))}
                                        
        example = tf.train.Example(features=tf.train.Features(feature=features_dict))	
        
        return example

    def _make_dataset(self, label_path, start_index=0, end_index=5864, num=None):
        """
        Make online batching dataset. Each TF record should have an image and a label.
        Args: 
            label_path: Path to labels ()
        """
        data = open(self.data_path, 'r').readlines()
        filenames = [file.strip().split(' ')[1] for file in data]
        label_file = open(label_path, 'r').readlines()
        labels = [label.strip().split(' ')[1] for label in label_file]
        data_and_labels = zip(filenames, labels)
        tfrecord_writer = tf.python_io.TFRecordWriter(self.record_path)

        if num is not None: end_index = num
        for i in range(start_index, end_index):
            if i%500==0: print "Processing: {} images".format(i)
            example = self._make_single_example(data_and_labels[i])
            tfrecord_writer.write(example.SerializeToString())
            
        print "Done generating TF records"
        
    def _vgg_preprocess(self, image):
        """
        VGG preprocessing. Mean subtraction of image with imagenet means
        """
        channels = tf.split(axis=2, num_or_size_splits=3, value=tf.cast(image, tf.float32))
        for i in range(3):
            channels[i] = tf.subtract(channels[i], imagenet_means_rgb[i])
            
        return tf.concat(values=channels, axis=2, name='preprocessed_image')
        
    def _inception_preprocess(self, image):
        
        """
        Pre-processing for inception
        """
        return (2.0/255)*image -1
        
    def _parse_single_example(self, example_proto):
        example = tf.parse_single_example(example_proto, 
                                        features={
                                        "image": tf.FixedLenFeature([], tf.string),
                                        "label": tf.FixedLenFeature([], tf.int64)  
                                        })
        image = tf.image.decode_jpeg(example["image"], channels=3)
        image_rs = tf.image.resize_images(image, size=[224, 224])
        
        # Convert the range to [-1, 1]
        preprocess_image = self._inception_preprocess(image_rs)
        label = tf.cast(example["label"], tf.int64)
        
        return preprocess_image, label
        
    def _parse_triplet_example(self, example_proto):
        example = tf.parse_single_example(example_proto, 
                                        features={
                                        "top_image": tf.FixedLenFeature([], tf.string),
                                        "bottom_image": tf.FixedLenFeature([], tf.string),
                                        "label": tf.FixedLenFeature([], tf.int64),
                                        "pos_flag": tf.FixedLenFeature([], tf.int64)    
                                        })
        top_image = tf.image.decode_jpeg(example["top_image"], channels=3)
        bottom_image = tf.image.decode_jpeg(example["bottom_image"], channels=3)
        top_image_rs = tf.image.resize_images(top_image, size=[224, 224])
        bottom_image_rs = tf.image.resize_images(bottom_image, size=[224, 224])
        
        # Convert the range to [-1, 1]
        top_preprocess_image = self._inception_preprocess(top_image_rs)
        bottom_preprocess_image = self._inception_preprocess(bottom_image_rs)
        label = tf.cast(example["label"], tf.int64)
        pos_flag = tf.cast(example["pos_flag"], tf.int64)
        
        return top_preprocess_image, bottom_preprocess_image, label, pos_flag
        
    def _parse_mask_example(self, example_proto):
        example = tf.parse_single_example(example_proto, 
                                        features={
                                        "image": tf.FixedLenFeature([], tf.string),
                                        "mask": tf.FixedLenFeature([], tf.string),
                                        "label": tf.FixedLenFeature([], tf.int64)  
                                        })
        top_image = tf.image.decode_jpeg(example["image"], channels=3)
        mask_image = tf.cast(tf.image.decode_jpeg(example["mask"], channels=1), tf.float32)
        top_image_rs = tf.image.resize_images(top_image, size=[224, 224])
        
        # background subtraction before preprocessing
        mask_not = tf.tile(tf.cast(tf.logical_not(tf.cast(mask_image, tf.bool)), tf.float32), [1,1,3])
        background_image = tf.multiply(top_image_rs, mask_not)
        
        # Object image (background is black)
        object_image = tf.multiply(top_image_rs, mask_image)
        
        # Convert the range to [-1, 1]
        top_preprocess_image = self._inception_preprocess(top_image_rs)
        background_image = self._inception_preprocess(background_image)
        object_image = self._inception_preprocess(object_image)
        label = tf.cast(example["label"], tf.int64)
        
        return top_preprocess_image, mask_image, background_image, object_image, label
        
    def _read_triplet_data(self):
        filenames = self.record_path
        dataset = tf.data.TFRecordDataset(filenames)
        dataset = dataset.shuffle(buffer_size=10000)
        dataset = dataset.map(self._parse_triplet_example)
        dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(self.batch_size))
        dataset = dataset.repeat(self.num_epochs)

        iterator = dataset.make_one_shot_iterator()
            
        top_image, bottom_image, label, pos_flag = iterator.get_next()

        return top_image, bottom_image, label, pos_flag
        
    def _read_data(self):
        filenames = self.record_path
        dataset = tf.data.TFRecordDataset(filenames)
        dataset = dataset.shuffle(buffer_size=10000)
        dataset = dataset.map(self._parse_single_example)
        dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(self.batch_size))
        dataset = dataset.repeat(self.num_epochs)

        iterator = dataset.make_one_shot_iterator()
            
        image, label = iterator.get_next()

        return image, label
        
    def _read_mask_data(self):
        filenames = self.record_path
        dataset = tf.data.TFRecordDataset(filenames)
        if self.mode=='train':
            dataset = dataset.shuffle(buffer_size=10000)
        dataset = dataset.map(self._parse_mask_example)
        dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(self.batch_size))
        dataset = dataset.repeat(self.num_epochs)

        iterator = dataset.make_one_shot_iterator()
            
        image, mask, background_image, object_image, label = iterator.get_next()

        return image, mask, background_image, object_image, label
			
		