import tensorflow as tf
import numpy as np
from data_loader import DataLoader
import argparse

def main(args):
	
	if args.n==0:
		dataset = DataLoader(args.path)
		dataset._make_dataset(label_path='/shared/kgcoe-research/mil/peri/birds_data/CUB_200_2011/image_class_labels.txt', start_index=5864, end_index=11788)
	else:
		dataset = DataLoader(args.path)
		dataset._make_dataset(num=args.n)
		

if __name__=="__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--path", type=str, required=True)
	parser.add_argument("--n", type=int, default=0, help="Number of examples in TFrecord")
	
	args = parser.parse_args()
	main(args)