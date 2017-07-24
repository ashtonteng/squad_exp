import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"


class AttentionLayer():
    """Implements Context-to-Query Attention. Pays attention to different parts of the query when
        reading the passage. Returns, for each word in the passage, a weighted vector of questions."""
    def __init__(self, args, p_inputs, q_inputs, scope):
        """p_inputs: batch_size x p_length x hidden_size"""
        """q_inputs: batch_size x q_length x hidden_size"""
        print("building attention layer", scope)
        batch_size = args.batch_size
        vocab_size = args.vocab_size
        hidden_size = args.AttentionLayer_size
        model = args.model
        num_layers = args.num_layers
        training = args.training

        p_inputs_shape = tf.shape(p_inputs)
        q_inputs_shape = tf.shape(q_inputs)
        p_length = p_inputs_shape[1]
        q_length = q_inputs_shape[1]

        p_inputs_aug = tf.tile(tf.expand_dims(p_inputs, 2), [1, 1, q_length, 1]) #batch_size x p_length x q_length x hidden_size
        q_inputs_aug = tf.tile(tf.expand_dims(q_inputs, 1), [1, p_length, 1, 1]) #batch_size x p_length x q_length x hidden_size
        pq_elementwise = tf.multiply(p_inputs_aug, q_inputs_aug) #batch_size x p_length x q_length x hidden_size
        combo_input = tf.concat([p_inputs_aug, q_inputs_aug, pq_elementwise], axis=3) #batch_size x p_length x q_length x 3*hidden_size

        with tf.variable_scope(scope):
            w_sim = tf.get_variable("w_sim", [3*hidden_size, 1])

        #in order to matmul combo_input with w_sim, we need to first tile w_sim batch_size number of times, and flatten combo_input
        combo_input_flat = tf.reshape(combo_input, [batch_size, -1, 3*hidden_size]) #batch_size x p_length*q_length x 3*hidden_size
        w_sim_tiled = tf.tile(tf.expand_dims(w_sim, 0), [batch_size, 1, 1]) #batch_size x 3*hidden_size x 1
        sim_mtx_flat = tf.matmul(combo_input_flat, w_sim_tiled) #batch_size x p_length*q_length x 1
        sim_mtx = tf.reshape(sim_mtx_flat, [batch_size, p_length, q_length, 1]) #batch_size x p_length x q_length x 1

        #C2Q attention: how relevant are the query words to each context word?
        #a #for each p, find weights to put on q. ##batch_size x p_length x q_length x hidden_size
        att_on_q = tf.nn.softmax(sim_mtx, dim=2)
        #q_inputs_aug = batch_size x p_length x q_length x hidden_size
        weighted_q = tf.multiply(att_on_q, q_inputs_aug)
        linear_combo_q_for_each_p = tf.reduce_sum(weighted_q, axis=2) #batch_size x p_length x hidden_size

        #Q2C Attention: which context words have the closest similarity to one of the query words?
        #for each context word choose which query word it helps contribute to the most
        #then normalize over all context words, to get a distribution of helpfulness of all context words to this query
        att_on_p = tf.nn.softmax(tf.reduce_max(sim_mtx, axis=2), dim=1) #batch_size x p_length x 1
        weighted_p = tf.multiply(att_on_p, p_inputs) #batch_size x p_length x hidden_size

        self.outputs = tf.concat([p_inputs, linear_combo_q_for_each_p, tf.multiply(p_inputs, linear_combo_q_for_each_p), tf.multiply(p_inputs, weighted_p)], axis=2)

        



