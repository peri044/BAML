import tensorflow as tf
import numpy as np
from dnn_library import *
from data.data_loader import DataLoader
import argparse
from model import *
from skimage import io
import time
import os
import datetime
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF logging
import pdb

def online_batching(images, labels):
	batch_size = images.shape[0]
	rand_indices = np.random.permutation(batch_size)
	second_set_images = np.copy(images)
	second_set_labels = np.copy(labels)
	second_set_images = images[rand_indices]
	second_set_labels = labels[rand_indices]
	pos_flag = labels==second_set_labels
	
	return images, second_set_images, pos_flag
	
def permutate(top_image_np, bottom_image_np, label_np, pos_flag_np):
	"""
	Generate negative pairs from the positive pairs by randomly select images from bottom image whose label doesn't match with the top image.
	Args: 
		top_image_np: First image in a sample pair
		bottom_image_np: Second image in a sample pair
		label_np: Labels of the above images (class)
		pos_flag_np: Flag indicating if they are positive images or not
	Returns:
		Positive and negative pairs from the batch along with labels and flags
	"""
	batch_size = top_image_np.shape[0]
	bottom_neg_image = np.copy(bottom_image_np)
	top_neg_image = np.copy(top_image_np)
	neg_labels = np.copy(label_np)
	for i in range(batch_size):
		non_duplicate = True
		while(non_duplicate):
			rand_int = np.random.randint(low=0, high=batch_size, size=1)[0]
			if label_np[i] != label_np[rand_int]:
				top_neg_image[i] = top_image_np[rand_int]
				neg_labels[i] = label_np[rand_int]
				non_duplicate = False

	return top_image_np, bottom_image_np, top_neg_image, label_np, neg_labels, pos_flag_np, np.zeros(batch_size)
				
def get_vars(all_vars, scope_name, index):
	"""
	Helper function used to return specific variables of a subgraph
	Args: 
		all_vars: All trainable variables in the graph
		scope_name: Scope name of the variables to retrieve
		index: Clip the variables in the graph at this index
	Returns:
		Dictionary of pre-trained checkpoint variables: new variables
	"""
	ckpt_vars = [var for var in all_vars if var.op.name.startswith(scope_name)]
	ckpt_var_dict = {}
	for var in ckpt_vars:
		actual_var_name  = var.op.name  #Conv2d_1a_7x7
		# if actual_var_name.find("Conv2d_1a_7x7") !=-1: pdb.set_trace()
		if actual_var_name.find("Conv2d_1a_7x7") ==-1 and actual_var_name.find('Logits') ==-1:
			clip_var_name = actual_var_name[index:]
			ckpt_var_dict[clip_var_name] = var
		
	return ckpt_var_dict
	
def get_training_op(loss, finetune):
	"""
	Computes the training op for the graph which needs to be run in the session
	Args: 
		loss: Loss of the network
	Returns: 
		Saver: Temporary saver to restore pre-trained weights
		train_op: Training op
	"""

	# Gather all the variables in the graph
	all_vars = tf.trainable_variables()
	# Global step for the graph
	global_step = tf.train.get_or_create_global_step(graph=tf.get_default_graph())
	
	INITIAL_LEARNING_RATE=0.0001
	DECAY_STEPS = 32000
	LEARNING_RATE_DECAY_FACTOR = 0.96
	# Decay the learning rate exponentially based on the number of steps.
	lr_fe = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
								  global_step,
								  DECAY_STEPS,
								  LEARNING_RATE_DECAY_FACTOR,
								  staircase=True)
	lr_mc = tf.train.exponential_decay(10*INITIAL_LEARNING_RATE,
								  global_step,
								  DECAY_STEPS,
								  LEARNING_RATE_DECAY_FACTOR,
								  staircase=True)
	
	# Define the optimizers. Here, feature extractor and metric embedding layers have different learning rates during training.
	optimizer_FE = tf.train.MomentumOptimizer(learning_rate=lr_fe, momentum=0.9)
	optimizer_MC = tf.train.MomentumOptimizer(learning_rate=lr_mc, momentum=0.9)

	# Get variables of specific sub networks using scope names
	vars_fe = get_vars(all_vars, scope_name='Feature_extractor', index=18)
	vars_me = get_vars(all_vars, scope_name='MetricEmbedding', index=0)
	vars_gen = get_vars(all_vars, scope_name='Generator', index=0)
	# pdb.set_trace()
	# Temporary saver just to initialize feature extractor pre-trained weights
	if finetune:
		saver = tf.train.Saver(keep_checkpoint_every_n_hours=1)
	else:
		# train from scratch loading pre-trained inception checkpoint
		saver = tf.train.Saver(vars_fe, keep_checkpoint_every_n_hours=1)
	
	# Calculate gradients for respective layers
	grad = tf.gradients(loss, vars_fe.values() + vars_me.values() + vars_gen.values())
	grad_fe = grad[: len(vars_fe.values())]
	grad_mc = grad[len(vars_fe.values()):]
	
	# Apply the gradients, update ops for batchnorm
	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	with tf.control_dependencies(update_ops):
		# Apply the gradients
		train_op_fe = optimizer_FE.apply_gradients(zip(grad_fe, vars_fe.values()), global_step=global_step)
		train_op_mc = optimizer_MC.apply_gradients(zip(grad_mc, vars_me.values()+vars_gen.values()))
		
		# Group individual training ops
		train_op = tf.group(train_op_fe, train_op_mc)
	
	return train_op, saver, global_step
	
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
	
	return process_row_vec, process_col_vec
	
	
def train(args):

    # Decode the tensors from tf record using tf.dataset API
    data = DataLoader(batch_size=args.batch_size, num_epochs=args.num_epochs)
    image, label = data._read_data()

    # Define anchor, negative and positive input placeholders
    anchor_image_placeholder = tf.placeholder(shape=[args.batch_size, 224, 224, 3], dtype=tf.float32, name='anchor_images')
    positive_image_placeholder = tf.placeholder(shape=[args.batch_size, 224, 224, 3], dtype=tf.float32, name='positive_images')
    negative_image_placeholder = tf.placeholder(shape=[args.batch_size, 224, 224, 3], dtype=tf.float32, name='negative_images')
    coord_conv_anchor = tf.placeholder(shape=[args.batch_size, 224, 224, 4], dtype=tf.float32, name='coord_conv_anchor')
    label_placeholder = tf.placeholder(shape=[args.batch_size], dtype=tf.uint8)

    # Build the model and get the embeddings
    model = DAML(args.base, margin=args.margin, is_training=True)
    anchor_features = model.feature_extractor(anchor_image_placeholder)
    anchor_embedding = model.build_embedding(anchor_features)

    # Get loss for DAML
    total_loss = model.triplet_loss(label_placeholder, anchor_embedding)

    # Get the training op for the whole network.
    train_op, saver, global_step = get_training_op(total_loss, args.finetune)

    #Define summaries
    tf.summary.image('Anchor Image', image)
    # tf.summary.image('Mask', mask)
    tf.summary.scalar('Total Loss', total_loss)
    summary_tensor = tf.summary.merge_all()
    now = datetime.datetime.now()
    summary_dir_name = args.exp_path+'/summaries_'+args.model+'_'+now.strftime("%Y-%m-%d_%H_%M")
    checkpoint_dir_name = args.exp_path+'/checkpoints_'+args.model+'_'+now.strftime("%Y-%m-%d_%H_%M")
    if args.enable_gen:
        summary_dir_name = args.exp_path+'/gen_summaries_'+args.model+'_'+now.strftime("%Y-%m-%d_%H_%M")
        checkpoint_dir_name = args.exp_path+'/gen_checkpoints_'+args.model+'_'+now.strftime("%Y-%m-%d_%H_%M")
    summary_filewriter = tf.summary.FileWriter(summary_dir_name, tf.get_default_graph())

    row_vector, col_vector = get_row_col_vectors(224)
    row_vec_batch = np.tile(np.expand_dims(np.expand_dims(row_vector, 0), -1), (args.batch_size, 1, 1 ,1))
    col_vec_batch = np.tile(np.expand_dims(np.expand_dims(col_vector, 0), -1), (args.batch_size, 1, 1 ,1))

    # Finalizes the graph and handles multi-threading via coordinators
    checkpoint_saver = tf.train.Saver(keep_checkpoint_every_n_hours=1.0)
    checkpoint_saver_hook = tf.train.CheckpointSaverHook(saver=checkpoint_saver, checkpoint_dir=checkpoint_dir_name, save_steps=2000)
    with tf.train.MonitoredTrainingSession(hooks=[checkpoint_saver_hook]) as sess:
        count=0
        #Restore the feature_extractor checkpoint
        saver.restore(sess, args.checkpoint)
        
        while not sess.should_stop():
            try:
                start_time = time.time()
                # Get a batch of input pairs which are positive
                image_np, label_np = sess.run([image, label]) #mask,
                # post_mask = process_mask(mask_np)
                # coord_conv_batch = np.concatenate([image_np, post_mask], axis=3) #, row_vec_batch, col_vec_batch

                # Run the training op
                _, global_step_value, total_loss_value, summary_value =  sess.run([train_op, global_step, total_loss, summary_tensor], 
                                                                feed_dict={anchor_image_placeholder: image_np, 
                                                                           label_placeholder: label_np})

                count+=1
                if count%100 == 0:
                    iter_time = time.time() - start_time
                    print 'Iteration: {} Loss: {} Step time: {}'.format(count, total_loss_value, iter_time)
                    summary_filewriter.add_summary(summary_value, count)

            except tf.errors.OutOfRangeError:
                break
                
        print "Training completed"
		
if __name__=="__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--base', default='inception_v1', help='Base architecture of feature extractor')
	parser.add_argument('--model', default='DAMLContrastive', help='Network to load')
	parser.add_argument('--exp_path', type=str, default="/shared/kgcoe-research/mil/peri/birds_data/experiments", help="Path to birds dataset")
	parser.add_argument('--checkpoint', type=str, default="/shared/kgcoe-research/mil/peri/tf_checkpoints/inception_v1.ckpt", help="Path to feature extractor checkpoint")
	parser.add_argument('--batch_size', type=int, default=32, help="batch size to train")
	parser.add_argument('--num_epochs', type=int, default=5, help="Num epochs to train")
	parser.add_argument('--margin', type=int, default=1, help="Margin")
	parser.add_argument('--enable_gen', action='store_true', help="Flag to use generator")
	parser.add_argument('--finetune', action='store_true', help="batch size to train")
	parser.add_argument('--metric_weight', type=int, default=1, help="Number of iterations to train")
	parser.add_argument('--reg_weight', type=int, default=1, help="Number of iterations to train")
	parser.add_argument('--adv_weight', type=int, default=50, help="Number of iterations to train")
	args = parser.parse_args()
	print '--------------------------------'
	for key, value in vars(args).items():
		print key, ' : ', value
	print '--------------------------------'
	train(args)