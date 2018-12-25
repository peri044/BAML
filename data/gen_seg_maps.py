import tensorflow as tf
import pdb
from PIL import Image
import numpy as np
from skimage import io
from skimage.transform import resize


class DeepLabModel(object):
  """Class to load deeplab model and run inference."""

  INPUT_TENSOR_NAME = 'ImageTensor:0'
  OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
  INPUT_SIZE = 513
  FROZEN_GRAPH_NAME = 'frozen_inference_graph'

  def __init__(self, tarball_path):
	"""Creates and loads pretrained deeplab model."""
	self.graph = tf.Graph()
	with open(tarball_path, 'r') as file:
		graph_def = tf.GraphDef.FromString(file.read())

	with self.graph.as_default():
		tf.import_graph_def(graph_def, name='')

	self.sess = tf.Session(graph=self.graph)

  def run(self, image):
	"""Runs inference on a single image.
	Args:
	  image: A PIL.Image object, raw input image.

	Returns:
	  resized_image: RGB image resized from original input image.
	  seg_map: Segmentation map of `resized_image`.
	"""
	width, height = image.size
	resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
	target_size = (int(resize_ratio * width), int(resize_ratio * height))
	resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
	batch_seg_map = self.sess.run(
		self.OUTPUT_TENSOR_NAME,
		feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
	seg_map = batch_seg_map[0]
	return resized_image, seg_map

def main():
	image_names=open('/shared/kgcoe-research/mil/peri/birds_data/CUB_200_2011/images.txt').readlines()
	image_files = [file.strip().split(' ')[1] for file in image_names]
	MODEL = DeepLabModel('/home/dp1248/cvs/DAML/data/pretrained/deeplabv3_pascal_train_aug/frozen_inference_graph.pb')
	save_path="/shared/kgcoe-research/mil/peri/birds_data/CUB_200_2011/seg_masks/"
	root_path='/shared/kgcoe-research/mil/peri/birds_data/CUB_200_2011/images/'
	# Segment the image 
	for i, image in enumerate(image_files):
		if i%250==0: print 'Processed: {}'.format(i)
		seg_name=image.split('/')[1]+'_seg'
		original_im=Image.open(root_path+image)
		resized_im, seg_map = MODEL.run(original_im)
		seg_map_rs = resize(seg_map, [224, 224], preserve_range=True)
		io.imsave(save_path+seg_name+'.jpg', 255*seg_map_rs.astype(np.uint8))
	
	

if __name__=="__main__":
	main()
