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
from loss import triplet_semihard_loss, triplet_loss, lifted_struct_loss
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF logging
import pdb
# tf.enable_eager_execution()

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
	
	# Concatenate pos, pos and neg labels for anchor, positive and negative images
	whole_labels = np.concatenate([label_np, label_np, neg_labels], axis=0)
	whole_labels = np.reshape(whole_labels, [whole_labels.shape[0], 1])
	# Build label pairwise matrix
	adjacency = np.equal(whole_labels, np.transpose(whole_labels))
	# Build a positive label mask. This is a hack to make lifted structured loss to work.
	# Set the diagnal values to zero as they are the same samples
	# Set the anchor to negative label flags to zero. 
	# This is to handle synthetic negative embeddings by the generator in DAML.
	# Since the negatives are randomly sampled, a negative for one sample might become the positive for another. 
	# Since this positive is not a true positive but a synthetic negative embedding for another sample, we need to mask this as false.
	mask_positive = np.copy(adjacency)
	mask_positive[:2*batch_size, 2*batch_size:]=False
	mask_positive[2*batch_size:, :2*batch_size] = False

	return top_image_np, bottom_image_np, top_neg_image, label_np, neg_labels, pos_flag_np, np.zeros(batch_size), adjacency, mask_positive
				
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
		actual_var_name  = var.op.name
		if actual_var_name.find('Logits') ==-1:
			clip_var_name = actual_var_name[index:]
			ckpt_var_dict[clip_var_name] = var
		
	return ckpt_var_dict
	
def get_training_op(loss, params):
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
	
	INITIAL_LEARNING_RATE=params.lr
	DECAY_STEPS = params.decay_steps
	LEARNING_RATE_DECAY_FACTOR = params.decay_factor
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

	# Temporary saver just to initialize feature extractor pre-trained weights
	if params.mode=='all':
		saver = tf.train.Saver(keep_checkpoint_every_n_hours=1)
	elif params.mode=='scratch':
		# train from scratch loading pre-trained inception checkpoint
		saver = tf.train.Saver(vars_fe, keep_checkpoint_every_n_hours=1)
	elif params.mode=='only_gen':
		# training generator + other networks
		# Get variables of specific sub networks using scope names
		vars_fe = get_vars(all_vars, scope_name='Feature_extractor', index=0)
		vars_me = get_vars(all_vars, scope_name='MetricEmbedding', index=0)
		vars_gen = get_vars(all_vars, scope_name='Generator', index=0)
		saver = tf.train.Saver(dict(vars_fe.items() + vars_me.items()), keep_checkpoint_every_n_hours=1)
	elif params.mode=='coordconv':
		# training generator + other networks
		# Get variables of specific sub networks using scope names
		for key in vars_fe.keys():
			if key.find('Conv2d_1a_7x7')!=-1:
				del vars_fe[key]
		saver = tf.train.Saver(dict(vars_fe.items()), keep_checkpoint_every_n_hours=1)
		
	# Calculate gradients for respective layers
	grad = tf.gradients(loss, vars_fe.values() + vars_me.values() + vars_gen.values())
	grad_fe = grad[: len(vars_fe.values())]
	grad_mc = grad[len(vars_fe.values()):]

	# Apply the gradients, update ops for batchnorm
	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	with tf.control_dependencies(update_ops):
		train_op_fe = optimizer_FE.apply_gradients(zip(grad_fe, vars_fe.values()), global_step=global_step)
		train_op_mc = optimizer_MC.apply_gradients(zip(grad_mc, vars_me.values()+vars_gen.values()))  # don't need to pass the global step as increment already happened in the previous line
		
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
    data = DataLoader(record_path=args.record_path, batch_size=args.batch_size, num_epochs=args.num_epochs)
 
    image, mask, background_image, object_image, label = data._read_mask_data()
    # Old preprocessing
    mask_not = tf.tile(tf.cast(tf.logical_not(tf.cast(mask, tf.bool)), tf.float32), [1,1,1,3])
    background_image_after = tf.multiply(image, mask_not)
    object_image_after = tf.multiply(image, mask)
    # Define the model
    model = DAML(args.base, margin=args.margin, embedding_dim=args.embedding_dim, is_training=True)

    # Build the model
    if args.model=="triplet":
        print "Training : {}".format(args.model)
        # Get the triplet embeddings
        anchor_embedding, positive_embedding, negative_embedding = model.build_triplet_model(anchor_image_placeholder, positive_image_placeholder, negative_image_placeholder)
        # L2 normalize the embeddings before loss
        anchor_embedding_l2 = tf.nn.l2_normalize(anchor_embedding, name='normalized_anchor')
        positive_embedding_l2 = tf.nn.l2_normalize(positive_embedding, name='normalized_positive')
        negative_embedding_l2 = tf.nn.l2_normalize(negative_embedding, name='normalized_negative')
        # compute the triplet loss
        total_loss, positive_distance, negative_distance = triplet_loss(anchor_embedding_l2, positive_embedding_l2, negative_embedding_l2)
        
        # Define the summaries
        tf.summary.scalar('Total Loss', total_loss)
        tf.summary.scalar('Positive-Anchor distance', positive_distance)
        tf.summary.scalar('Negative-Anchor distance', negative_distance)
        tf.summary.image('Anchor_image', anchor_image_placeholder)
        tf.summary.image('Positive_image', positive_image_placeholder)
        tf.summary.image('Negative_image', negative_image_placeholder)
        
    elif args.model=="mask-triplet":
        anchor_embedding = model.build_mask_triplet_model(image, background_image)
        # compute the triplet loss
        total_loss = model.triplet_loss(label, anchor_embedding)

        # Define the summaries
        tf.summary.scalar('Total Loss', total_loss)
        tf.summary.image('Mask', mask)
        tf.summary.image('Anchor Image', image)
        tf.summary.image('Object image', object_image)
        tf.summary.image('Background Image', background_image)
        
    elif args.model=="object_whole":
        whole_embedding, object_embedding = model.build_object_whole_triplet_model(image, object_image)
        anchor_embedding = whole_embedding + object_embedding
        # compute the triplet loss
        total_loss = model.triplet_loss(label, anchor_embedding)

        # Define the summaries
        tf.summary.scalar('Total Loss', total_loss)
        tf.summary.image('Mask', mask)
        tf.summary.image('Anchor Image', image)
        tf.summary.image('Object image', object_image)
        tf.summary.image('Background Image', background_image)
        
    elif args.model=="object_whole_separate":
        whole_embedding, object_embedding = model.build_object_whole_triplet_model(image, object_image_after)
        
        # compute the triplet loss
        whole_loss = model.triplet_loss(label, whole_embedding)
        object_loss = model.triplet_loss(label, object_embedding)
        total_loss = whole_loss + object_loss
        # Define the summaries
        # pdb.set_trace()
        tf.summary.scalar('Total Loss', total_loss)
        tf.summary.scalar('whole_loss', whole_loss)
        tf.summary.scalar('object_loss', object_loss)
        tf.summary.image('Mask', mask)
        tf.summary.image('Anchor Image', image)
        tf.summary.image('Object image', object_image_after)
        
        
    elif args.model=="triplet_single":
        print "Training : {}".format(args.model)
        # Get the anchor embeddings
        anchor_features = model.feature_extractor(object_image)
        anchor_embedding = model.build_embedding(anchor_features)

        # compute the lifted loss
        total_loss = model.triplet_loss(label, anchor_embedding)

        # Define the summaries
        tf.summary.scalar('Total Loss', total_loss)
        tf.summary.image('Anchor_image', image)
        tf.summary.image('Object image', object_image)
        tf.summary.image('Background_image', background_image)
        tf.summary.image('Mask', mask)
        
    elif args.model=="triplet_mask":
        print "Training : {}".format(args.model)
        # Get the anchor embeddings
        coord_conv_anchor = tf.placeholder(shape=[args.batch_size, 224, 224, 4], dtype=tf.float32, name='anchor_images')
        anchor_features = model.feature_extractor(coord_conv_anchor)
        anchor_embedding = model.build_embedding(anchor_features)

        # compute the lifted loss
        total_loss = model.triplet_loss(label_placeholder, anchor_embedding)

        # Define the summaries
        tf.summary.scalar('Total Loss', total_loss)
        tf.summary.image('Anchor_image', image)
        tf.summary.image('Anchor_image', mask)
        
    elif args.model=="lifted_single":
        print "Training : {}".format(args.model)
        # Get the anchor embeddings
        anchor_features = model.feature_extractor(anchor_image_placeholder)
        anchor_embedding = model.build_embedding(anchor_features)
        
        # compute the lifted loss
        total_loss = model.lifted_loss(label_placeholder, anchor_embedding)
        
        # Define the summaries
        tf.summary.scalar('Total Loss', total_loss)
        tf.summary.image('Anchor_image', anchor_image_placeholder)
        
    elif args.model=="lifted":
        print "Training : {}".format(args.model)
        anchor_embedding, positive_embedding, negative_embedding = model.build_triplet_model(anchor_image_placeholder, positive_image_placeholder, negative_image_placeholder)
        concat_embeddings = tf.concat([anchor_embedding, positive_embedding, negative_embedding], axis=0)
        positive_mask_placeholder = tf.placeholder(shape=[3*args.batch_size, 3*args.batch_size], dtype=tf.bool)
        # compute the lifted loss
        total_loss, _ = lifted_struct_loss(positive_mask_placeholder, concat_embeddings)
        
        # Define the summaries
        tf.summary.scalar('Total Loss', total_loss)
        tf.summary.image('Anchor_image', anchor_image_placeholder)
        tf.summary.image('Positive_image', positive_image_placeholder)
        tf.summary.image('Negative_image', negative_image_placeholder)
        
    elif args.model=="daml-lifted":
        print "Training : {}".format(args.model)
        anchor_embedding, positive_embedding, negative_embedding = model.build_triplet_model(anchor_image_placeholder, positive_image_placeholder, negative_image_placeholder)
        concat_embeddings = tf.concat([anchor_embedding, positive_embedding, negative_embedding], axis=0)
        positive_mask_placeholder = tf.placeholder(shape=[3*args.batch_size, 3*args.batch_size], dtype=tf.bool)
        # compute the lifted loss
        lifted_loss_t, _ = lifted_struct_loss(positive_mask_placeholder, concat_embeddings)
        
        # Get the synthetic embeddings
        synthetic_neg_embedding = model.generator(anchor_embedding, positive_embedding, negative_embedding)
        
        # L2 normalize the embeddings before loss
        anchor_embedding_l2 = tf.nn.l2_normalize(anchor_embedding, name='normalized_anchor')
        positive_embedding_l2 = tf.nn.l2_normalize(positive_embedding, name='normalized_positive')
        negative_embedding_l2 = tf.nn.l2_normalize(negative_embedding, name='normalized_negative')
        synthetic_neg_embedding_l2 = tf.nn.l2_normalize(synthetic_neg_embedding, name='normalized_synthetic')
        J_hard, J_reg, J_adv = model.daml_loss(anchor_embedding_l2, positive_embedding_l2, negative_embedding_l2, synthetic_neg_embedding_l2)
        J_gen = args.hard_weight*J_hard + args.reg_weight*J_reg + args.adv_weight*J_adv
        total_loss = args.metric_weight*lifted_loss_t + J_gen
        # Define the summaries
        tf.summary.scalar('J_m', lifted_loss_t)
        tf.summary.scalar('J_hard', J_hard)
        tf.summary.scalar('J_reg', J_reg)
        tf.summary.scalar('J_adv', J_adv)
        tf.summary.scalar('J_gen', J_gen)
        tf.summary.scalar('Total Loss', total_loss)
        tf.summary.image('Anchor_image', anchor_image_placeholder)
        tf.summary.image('Positive_image', positive_image_placeholder)
        tf.summary.image('Negative_image', negative_image_placeholder)
        
    elif args.model=="daml-triplet":
        # Get the triplet embeddings
        anchor_embedding, positive_embedding, negative_embedding = model.build_triplet_model(anchor_image_placeholder, positive_image_placeholder, negative_image_placeholder)
        positive_mask_placeholder = tf.placeholder(shape=[3*args.batch_size, 3*args.batch_size], dtype=tf.bool)
        # Get the synthetic embeddings
        synthetic_neg_embedding = model.generator(anchor_embedding, positive_embedding, negative_embedding)
        # L2 normalize the embeddings before loss
        anchor_embedding_l2 = tf.nn.l2_normalize(anchor_embedding, name='normalized_anchor')
        positive_embedding_l2 = tf.nn.l2_normalize(positive_embedding, name='normalized_positive')
        negative_embedding_l2 = tf.nn.l2_normalize(negative_embedding, name='normalized_negative')
        synthetic_neg_embedding_l2 = tf.nn.l2_normalize(synthetic_neg_embedding, name='normalized_synthetic')
        # Calculate Triplet loss
        triplet_loss_t, positive_distance, negative_distance = triplet_loss(anchor_embedding_l2, positive_embedding_l2, synthetic_neg_embedding_l2)
        J_hard, J_reg, J_adv = model.daml_loss(anchor_embedding_l2, positive_embedding_l2, negative_embedding_l2, synthetic_neg_embedding_l2)
        J_gen = args.hard_weight*J_hard + args.reg_weight*J_reg + args.adv_weight*J_adv
        total_loss = args.metric_weight*triplet_loss_t + J_gen
        # Define the summaries
        tf.summary.scalar('J_m', triplet_loss_t)
        tf.summary.scalar('J_hard', J_hard)
        tf.summary.scalar('J_reg', J_reg)
        tf.summary.scalar('J_adv', J_adv)
        tf.summary.scalar('J_gen', J_gen)
        tf.summary.scalar('Total Loss', total_loss)
        tf.summary.scalar('Positive-Anchor distance', positive_distance)
        tf.summary.scalar('Negative-Anchor distance', negative_distance)
        tf.summary.image('Anchor_image', anchor_image_placeholder)
        tf.summary.image('Positive_image', positive_image_placeholder)
        tf.summary.image('Negative_image', negative_image_placeholder)


    # Get the training op for the whole network.
    train_op, initial_saver, global_step = get_training_op(total_loss, args)

    #Merge summaries
    summary_tensor = tf.summary.merge_all()

    now = datetime.datetime.now()
    summary_dir_name = args.exp_path+'/s_'+args.model+'_'+args.mode+'_'+now.strftime("%Y-%m-%d_%H_%M")
    checkpoint_dir_name = args.exp_path+'/c_'+args.model+'_'+args.mode+'_'+now.strftime("%Y-%m-%d_%H_%M")
    if args.mode=='only_gen':
        summary_dir_name = args.exp_path+'/gen_summaries_'+args.model+'_'+args.mode+'_'+now.strftime("%Y-%m-%d_%H_%M")
        checkpoint_dir_name = args.exp_path+'/gen_checkpoints_'+args.model+'_'+args.mode+'_'+now.strftime("%Y-%m-%d_%H_%M")
    summary_filewriter = tf.summary.FileWriter(summary_dir_name, tf.get_default_graph())

    # Checkpoint saver to save the variables of the entire graph. Training monitored session handles queue runners internally.
    checkpoint_saver = tf.train.Saver(keep_checkpoint_every_n_hours=1.0)
    checkpoint_saver_hook = tf.train.CheckpointSaverHook(saver=checkpoint_saver, checkpoint_dir=checkpoint_dir_name, save_steps=args.save_steps)
    with tf.train.MonitoredTrainingSession(hooks=[checkpoint_saver_hook]) as sess:
        #Restore the feature_extractor checkpoint
        initial_saver.restore(sess, args.checkpoint)
        print "Restored: {}".format(args.checkpoint)
        while not sess.should_stop():
            try:
                start_time = time.time()
                # Get a batch of input pairs which are positive
                # image_np, mask_np, label_np = sess.run([image, mask, label])										
                # top_image_np, bottom_image_np, label_np, pos_flag_np = sess.run([top_image, bottom_image, label, pos_flag])		

                # Create positive and negative pairing 
                # anchor_image_b, positive_image_b, negative_image_b,  \
                            # pos_labels_b, neg_labels_b, pos_flag_b, neg_flag_b, adjacency, positive_mask = permutate(top_image_np, bottom_image_np, label_np, pos_flag_np)
                # Run the training op
                # _, global_step_value, total_loss_value, summary_value =  sess.run([train_op, global_step, total_loss, summary_tensor], 
                                                                # feed_dict={anchor_image_placeholder: anchor_image_b,
                                                                           # positive_image_placeholder: positive_image_b,
                                                                           # negative_image_placeholder: negative_image_b,
                                                                           # positive_mask_placeholder: adjacency
                                                                           # })
                                                                           
                # Run the training op
                _, global_step_value, total_loss_value, summary_value =  sess.run([train_op, global_step, total_loss, summary_tensor])
                                                                           
                # post_mask = process_mask(mask_np)
                # coord_conv_batch = np.concatenate([image_np, post_mask], axis=3) #, row_vec_batch, col_vec_batch
                # Run the training op
                # _, global_step_value, total_loss_value, summary_value =  sess.run([train_op, global_step, total_loss, summary_tensor], 
                                                                # feed_dict={coord_conv_anchor: coord_conv_batch,
                                                                           # label_placeholder: label_np})
                if (global_step_value+1)%100 == 0:
                    iter_time = time.time() - start_time
                    print 'Iteration: {} Loss: {} Step time: {}'.format(global_step_value+1, total_loss_value, iter_time)
                    summary_filewriter.add_summary(summary_value, global_step_value)
                
            except tf.errors.OutOfRangeError:
                break
                
        print "Training completed"
		
if __name__=="__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--base', default='inception_v1', help='Base architecture of feature extractor')
	parser.add_argument('--model', default='mask-triplet', help='Network to load')
	parser.add_argument('--record_path', type=str, default="/shared/kgcoe-research/mil/peri/birds_data/birds_ob_train_mask.tfrecord", help="Path to Train TF record")
	parser.add_argument('--exp_path', type=str, default="/shared/kgcoe-research/mil/peri/birds_data/experiments", help="Path to birds dataset")
	parser.add_argument('--optimizer', type=str, default="adam", help="Optmizer")
	parser.add_argument('--checkpoint', type=str, default="/shared/kgcoe-research/mil/peri/tf_checkpoints/inception_v1.ckpt", help="Path to feature extractor checkpoint")
	parser.add_argument('--batch_size', type=int, default=32, help="batch size to train")
	parser.add_argument('--decay_steps', type=int, default=15000, help="Decay steps")
	parser.add_argument('--save_steps', type=int, default=2000, help="Save steps")
	parser.add_argument('--decay_factor', type=float, default=0.96, help="Decay factor")
	parser.add_argument('--lr', type=float, default=0.0001, help="Decay factor")
	parser.add_argument('--num_epochs', type=int, default=440, help="Num epochs to train")
	parser.add_argument('--embedding_dim', type=int, default=512, help="Num epochs to train")
	parser.add_argument('--margin', type=float, default=1, help="Margin")
	parser.add_argument('--mode', type=str,  required=True, help="Modes to train scratch|no_daml|only_gen")
	parser.add_argument('--metric_weight', type=int, default=1, help="Number of iterations to train")
	parser.add_argument('--reg_weight', type=int, default=1, help="Number of iterations to train")
	parser.add_argument('--hard_weight', type=int, default=1, help="Number of iterations to train")
	parser.add_argument('--adv_weight', type=int, default=50, help="Number of iterations to train")
	args = parser.parse_args()
	print '--------------------------------'
	for key, value in vars(args).items():
		print key, ' : ', value
	print '--------------------------------'
	train(args)