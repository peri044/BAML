import tensorflow as tf
import numpy as np
from dnn_library import *
import pdb

slim=tf.contrib.slim

class DAML(object):
    """
    Implementation of DAML model. Refer to http://openaccess.thecvf.com/content_cvpr_2018/papers/Duan_Deep_Adversarial_Metric_CVPR_2018_paper.pdf
    """
    def __init__(self, base, margin=1., embedding_dim=512, is_training=True):
        self.scope_name='DAML'
        self.is_training = is_training
        self.base_arch = base
        self.margin=margin
        self.embedding_dim = embedding_dim
        
    def generator(self, anchor_embedding, positive_embedding, negative_embedding, scope_name='synthetic_embedding'):
        """
        Generator that generates synthetic negatives from the negative images. 3-layer fully connected layer network
        """
        with tf.variable_scope('Generator') as scope:
            
            # Fuse all three embeddings. Dim: 1536 (512x3)
            fused_embedding = tf.concat([anchor_embedding, positive_embedding, negative_embedding], axis=1, name='fused_embedding')
            
            with slim.arg_scope([slim.fully_connected],
                                 activation_fn=tf.nn.relu,
                                 weights_initializer=tf.contrib.layers.xavier_initializer(),
                                 weights_regularizer=slim.l2_regularizer(0.0002)):
                
                fused_fc_1 = slim.fully_connected(fused_embedding, 1024, scope = 'fused_fc_1')
                negative_synthetic = slim.fully_connected(fused_fc_1, 512, activation_fn=None, scope = 'negative_synthetic')
                
        return negative_synthetic		
        
    def feature_extractor(self, image, reuse=None):
        """
        Builds the model architecture
        """
                        
        # Define the network and pass the input image
        with tf.variable_scope('Feature_extractor', reuse=reuse) as scope:
            with slim.arg_scope(model[self.base_arch]['scope']):
                logits, end_points = model[self.base_arch]['net'](image, num_classes=model[self.base_arch]['num_classes'], is_training=self.is_training)
            
        # Dropout features of inception v1 (size: 1024)
        feat_anchor = end_points['AvgPool_0a_7x7']  ## Dropout_0b
        if self.is_training:
            feat_anchor = tf.squeeze(end_points['AvgPool_0a_7x7'])
        return feat_anchor
        
    def build_embedding(self, feat_anchor, embedding_dim=512, scope_name="embedding", reuse=tf.AUTO_REUSE):
        """
        Build the embedding network
        """
        with tf.variable_scope('MetricEmbedding', reuse=reuse) as scope:
            with slim.arg_scope([slim.fully_connected],
                                 activation_fn=tf.nn.relu,
                                 weights_initializer=tf.contrib.layers.xavier_initializer(),
                                 weights_regularizer=slim.l2_regularizer(0.0002)):
                anchor_embedding = slim.fully_connected(feat_anchor, embedding_dim, activation_fn=None, scope=scope_name)

        return anchor_embedding		
        
    def build_triplet_model(self, anchor_image, positive_image, negative_image):
        
        # Anchor_image
        anchor_features = self.feature_extractor(anchor_image)
        anchor_embedding = self.build_embedding(anchor_features, self.embedding_dim)
        # Positive_image
        positive_features = self.feature_extractor(positive_image, reuse=True)
        positive_embedding = self.build_embedding(positive_features, self.embedding_dim, reuse=True)
        # Negative_image
        negative_features = self.feature_extractor(negative_image, reuse=True)
        negative_embedding = self.build_embedding(negative_features, self.embedding_dim, reuse=True)
        
        return anchor_embedding, positive_embedding, negative_embedding
        
    def build_mask_triplet_model(self, original_image, background_image):
        
        # original_image
        original_features = self.feature_extractor(original_image)
        original_embedding = self.build_embedding(original_features, self.embedding_dim, scope_name='image_embedding')
        # background_image
        background_features = self.feature_extractor(background_image, reuse=True)
        background_embedding = self.build_embedding(background_features, self.embedding_dim, scope_name='background_embedding')
        # background_embedding = self.build_embedding(background_features, self.embedding_dim, scope_name='image_embedding')

        final_embedding = original_embedding - background_embedding
        
        return final_embedding
    
    def build_object_whole_triplet_model(self, whole_image, object_image):
        
        # whole_image
        whole_features = self.feature_extractor(whole_image)
        whole_embedding = self.build_embedding(whole_features, self.embedding_dim, scope_name='image_embedding')
        # object image
        object_features = self.feature_extractor(object_image, reuse=True)
        object_embedding = self.build_embedding(object_features, self.embedding_dim, scope_name='object_embedding')
        
        return whole_embedding, object_embedding
        
        
    def daml_loss(self, anchor_embedding, positive_embedding, negative_embedding, synthetic_neg_embedding):
        """
        Defines the loss for the model
        """
        with tf.name_scope('DAML_Loss') as scope:
            # Adversarial loss Eqn.(6) in the paper
            J_hard = tf.reduce_sum(tf.squared_difference(synthetic_neg_embedding, anchor_embedding), name='J_hard') 
            J_reg = tf.reduce_sum(tf.squared_difference(synthetic_neg_embedding, negative_embedding), name='J_reg')
            pos_pair_distance = tf.reduce_sum(tf.squared_difference(positive_embedding, anchor_embedding), name='pos_pair_distance')
            neg_pair_distance = tf.reduce_sum(tf.squared_difference(synthetic_neg_embedding, anchor_embedding), name='neg_pair_distance')
            J_adv = tf.maximum(neg_pair_distance - pos_pair_distance - self.margin, 0., name='J_adv')
            # J_adv = tf.square(neg_pair_distance - pos_pair_distance - self.margin,  name='J_adv')
            
            return J_hard, J_reg, J_adv
            
        
    def contrastive_loss(self, labels, anchor_embedding, positive_embedding):
        """
        Defines the loss for the model
        """
        with tf.name_scope('Loss') as scope:
            # L2 normalize the embeddings before using Contrastive loss
            normalized_anchors = tf.nn.l2_normalize(anchor_embedding, axis=1, name='normalized_anchors')
            normalized_embeddings = tf.nn.l2_normalize(positive_embedding, axis=1, name='normalized_embeddings')
            distances = tf.sqrt(tf.reduce_sum(tf.squared_difference(normalized_anchors, normalized_embeddings),1))
            J_m = tf.contrib.losses.metric_learning.contrastive_loss(labels, normalized_anchors, normalized_embeddings, margin=self.margin)
            
        return J_m, 0., 0., 0., distances
        
    def triplet_loss(self, labels, anchor_embedding):
        """
        Computes the triplet loss for the embeddings
        """
        with tf.name_scope('Loss') as scope:
            # L2 normalize the embeddings before using Triplet loss
            # pdb.set_trace()
            normalized_embeddings = tf.nn.l2_normalize(anchor_embedding, axis=1, name='normalized_embeddings')
            J_m = tf.contrib.losses.metric_learning.triplet_semihard_loss(labels, normalized_embeddings, margin=float(self.margin))
            
            return J_m
            
    def lifted_loss(self, labels, anchor_embedding):
        """
        Computes the Lifted Structured loss for the embeddings
        """
        with tf.name_scope('Loss') as scope:
            # No L2 normalization for lifted loss
            J_m = tf.contrib.losses.metric_learning.lifted_struct_loss(labels, anchor_embedding, margin=float(self.margin))
            
            return J_m
	