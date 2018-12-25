import tensorflow as tf
import numpy as np
import argparse
from model import *
from data.data_loader import *
from skimage import io
from skimage.transform import resize
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score
import time
import os
import pdb
np.random.seed(0)

def load_val_data(image_path, label_path):
	
	inputs = open(image_path, 'r').readlines()
	input_files = [input.strip().split(' ')[1] for input in inputs]
	label_file = open(label_path, 'r').readlines()
	labels = [input.strip().split(' ')[1] for input in label_file]
	test_index = 5864
	inputs_and_labels = zip(input_files[test_index:], labels[test_index:])
	
	return inputs_and_labels
	
def compute_nmi(embeddings, groundtruth, n_cluster):
	"""
	Computes Normalized mutual information scores on the embeddings by using K-means clustering.
	Args: 
		embeddings: [num_samples, embedding_dim]
		groundtruth: [num_samples] -- Groundtruth class
		n_cluster: n_samples (Number of clusters to be formed)
	"""
	kmeans= KMeans(n_clusters=n_cluster, n_jobs=-1, random_state=1, max_iter=1000).fit(embeddings)  # n_cluster = 100, embeddings is 5924 x 512
	kmeans_nmi = normalized_mutual_info_score(groundtruth, kmeans.labels_)  # K-means NMI 
	print "K-means NMI: {}".format(kmeans_nmi)
	
def recall_at_k(embeddings, groundtruth, inputs_and_labels, recall_scales):
    """
    Computes Recall at k
    Args: 
        embedddings : [num_samples, embedding_dim] 
        groundtruth : [num_samples] -- Groundtruth class 
        recall_scales : list of recall factors [1, 2, 4, 8, 10]
    """
    num_samples = embeddings.shape[0]
    pdist_matrix = pairwise_distances(embeddings)
    # Set the diagonal elements to a very high value
    for row in range(num_samples):
        for col in range(num_samples):
            if row==col:
                pdist_matrix[row, col] = 1e10   
    # For each sample, sort the distances to the neighbouring samples
    # Get the sorted topK indices( distances ascending order sorted)
    # Increment if the groundtruth class id is in list of topK indices. 
    path='/shared/kgcoe-research/mil/video_project/cvs/birds/images/'
    root='/shared/kgcoe-research/mil/peri/birds_data/fail_triplet_obj_image/'
    if not os.path.isdir(root):
        os.mkdir(root)
    for k in recall_scales:
        num_correct=0
        for i in range(num_samples):
            this_class_index = groundtruth[i]
            sorted_indices = np.argsort(pdist_matrix[i, :])
            knn_indices = sorted_indices[:k]
            knn_class_indices = groundtruth[knn_indices]
            if this_class_index in knn_class_indices:
                num_correct+=1
            # else:
                # if k==1:
                    # ref= inputs_and_labels[i][0]
                    # ref_im=resize(io.imread(path+ref), [224,224])
                    # test = inputs_and_labels[int(knn_indices)][0]
                    # test_im=resize(io.imread(path+test), [224,224])
                    # if test_im.shape != ref_im.shape: continue
                    # io.imsave(root+str(i)+'.png', np.concatenate([ref_im, test_im], axis=1))
        if k==1: print "Num correct: {}, Num samples: {}".format(num_correct, num_samples)
        recall = float(num_correct)/num_samples
        print "Recall@{}: {}".format(k, recall)
	
def process_mask(mask_np):
	processed_mask = (2.0/255)*mask_np -1.
	return processed_mask

def get_row_col_vectors(H):
	lin_vector =  np.linspace(0, H-1, H)
	tile_lin_vector = np.reshape(np.tile(lin_vector, H), [H, H])
	row_vector = tile_lin_vector.T
	col_vector = tile_lin_vector
	process_row_vec = (2.0/H)*row_vector -1.
	process_col_vec = (2.0/H)*col_vector -1.
	
	return np.expand_dims(process_row_vec, -1), np.expand_dims(process_col_vec, -1)
	
def evaluate(args):

    # Load the testing data
    inputs_and_labels = load_val_data(args.image_path, args.label_path)

    # Decode the tensors from tf record using tf.dataset API
    data = DataLoader(record_path=args.record_path, batch_size=args.batch_size, num_epochs=args.num_epochs, mode=args.mode)
    image, mask, background_image, object_image, label = data._read_mask_data()
    mask_not = tf.tile(tf.cast(tf.logical_not(tf.cast(mask, tf.bool)), tf.float32), [1,1,1,3])
    background_image_after = tf.multiply(image, mask_not)
    object_image_after = tf.multiply(image, mask)

    # Build the model and get the embeddings
    model = DAML(args.base, is_training=False)
    if args.model=='triplet_single':
        anchor_features = model.feature_extractor(object_image) #, pos_emb, neg_emb, syn_emb
        anchor_embedding = model.build_embedding(anchor_features, args.embedding_dim) #scope_name='anchor_embedding'
        anchor_embedding = tf.nn.l2_normalize(anchor_embedding)
    elif args.model=='triplet_mask':
        anchor_embedding = model.build_mask_triplet_model(image, background_image_after)
        anchor_embedding = tf.nn.l2_normalize(anchor_embedding)
    elif args.model=='object_whole':
        anchor_embedding = model.build_object_whole_triplet_model(image, object_image)
        anchor_embedding = tf.nn.l2_normalize(anchor_embedding)
    elif args.model=='object_whole_separate':
        whole_embedding, object_embedding = model.build_object_whole_triplet_model(image, object_image)
        whole_embedding = tf.nn.l2_normalize(whole_embedding)
        object_embedding = tf.nn.l2_normalize(object_embedding)
    else:
        raise ValueError("Invalid Model !!")
        
    saver = tf.train.Saver()
    embeddings=np.zeros([args.num, args.embedding_dim])
    labels=np.zeros([args.num])
    with tf.Session() as sess:
        saver.restore(sess, args.checkpoint)
        for i in range(0, args.num, args.batch_size):
            if i%500==0 and i>0: print "Evaluated: {}/{}".format(i, args.num)
            # Extract the embeddings by executing the graph
            anc_emb_value, label_value = sess.run([whole_embedding, label])
            embeddings[i:i+args.batch_size, :] = np.squeeze(anc_emb_value)
            labels[i:i+args.batch_size] = label_value
 
    compute_nmi(embeddings, labels, 100)
    recall_at_k(embeddings, labels, inputs_and_labels, [1, 2, 4, 8])
	
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', default='inception_v1', help='Base architecture of feature extractor')
    parser.add_argument('--embedding_dim', type=int, default=512, help='Embedding dimension')
    parser.add_argument('--num_epochs', type=int, default=1, help='Embedding dimension')
    parser.add_argument('--num', type=int, default=5900, help='Embedding dimension')
    parser.add_argument('--model', type=str, default='triplet_mask', help='Network to load')
    parser.add_argument('--record_path', type=str, default="/shared/kgcoe-research/mil/peri/birds_data/birds_ob_test_mask.tfrecord", help="Path to birds dataset tfrecord")
    parser.add_argument('--checkpoint', type=str, default="/shared/kgcoe-research/mil/peri/tf_checkpoints/inception_v1.ckpt", help="Path to feature extractor checkpoint")
    parser.add_argument('--image_path', type=str, default="/shared/kgcoe-research/mil/peri/birds_data/CUB_200_2011/images.txt", help="Path to birds dataset")
    parser.add_argument('--label_path', type=str, default="/shared/kgcoe-research/mil/peri/birds_data/CUB_200_2011/image_class_labels.txt", help="Path to birds dataset")
    parser.add_argument('--batch_size', type=int, default=20, help="batch size to test")
    parser.add_argument('--mode', type=str, default='val', help="Mode")
    args = parser.parse_args()
    print '--------------------------------'
    for key, value in vars(args).items():
        print key, ' : ', value
    print '--------------------------------'
    evaluate(args)