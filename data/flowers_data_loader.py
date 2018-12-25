"""
Data writer, loader for  flowers dataset
"""
import tensorflow as tf
import numpy as np
import os
import argparse

class FlowersDataLoader(object):
	"""
	Data loader and writer object for flowers dataset
	"""
	def __init__(self, path):
		self.data_path=path

	def _make_single_example(self, record):
		"""
		Make a single example in a TF record
		"""
		image_path=record.split('__')[0]
		image = tf.gfile.FastGFile(image_path, "rb").read()
		caption=record.split('__')[1]
		label=record.split('__')[2]
		features_dict = {"image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
						 "caption": tf.train.Feature(bytes_list=tf.train.BytesList(value=[caption])),
						 "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[int(label)]))}
										
		example = tf.train.Example(features=tf.train.Features(feature=features_dict))	

		return example
		
	def _make_dataset(self, record_path, num=None):
		"""
		Write the whole dataset to a TF record.
		"""
		data=open(self.data_path, 'r').readlines()
		tfrecord_writer = tf.python_io.TFRecordWriter(record_path)
		count=0
		if num is not None:
			data = data[:num]
			
		for record in data:
			if count%100==0 and count!=0: print "Generated: {}".format(count)
			example = self._make_single_example(record.strip())
			tfrecord_writer.write(example.SerializeToString())
			count+=1
			
		print "Done generating TF records"


def main(args):

	dataset = FlowersDataLoader(args.data_path)
	dataset._make_dataset(args.record_path, num=40000)

if __name__=="__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--record_path', default='/shared/kgcoe-research/mil/peri/flowers_data/flowers_train.tfrecord', help='TF record path')
	parser.add_argument('--data_path', default='/shared/kgcoe-research/mil/peri/flowers_data/record_data/flowers_all_data.txt', help='TF record path')
	args=parser.parse_args()
	main(args)
				