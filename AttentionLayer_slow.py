import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"


class AttentionLayer():
    """Implements Context-to-Query Attention. Pays attention to different parts of the query when
        reading the passage. Returns, for each word in the passage, a weighted vector of questions."""
    def __init__(self, p_inputs, q_inputs, p_length, q_length, hidden_size, scope, batch_size=10, training=True):
        """p_inputs: batch_size x p_length x hidden_size"""
        """q_inputs: batch_size x q_length x hidden_size"""

        p_inputs_aug = tf.tile(tf.expand_dims(p_inputs, 2), [1, 1, q_length, 1]) #batch_size x p_length x q_length x hidden_size
        q_inputs_aug = tf.tile(tf.expand_dims(q_inputs, 1), [1, p_length, 1, 1]) #batch_size x p_length x q_length x hidden_size
        pq_elementwise = tf.multiply(p_inputs_aug, q_inputs_aug) #batch_size x p_length x q_length x hidden_size
        combo_input = tf.concat([p_inputs_aug, q_inputs_aug, pq_elementwise], axis=3) #batch_size x p_length x q_length x 3*hidden_size

        with tf.variable_scope(scope):
            w_sim = tf.get_variable("w_sim", [6*hidden_size, 1])

        #in order to matmul combo_input with w_sim, we need to first tile w_sim batch_size number of times, and flatten combo_input
        #print("combo_input", combo_input.get_shape())
        combo_input_flat = tf.reshape(combo_input, [batch_size, -1, 6*hidden_size]) #batch_size x p_length*q_length x 3*hidden_size
        #print("combo_input_flat", combo_input_flat.get_shape())
        w_sim_tiled = tf.tile(tf.expand_dims(w_sim, 0), [batch_size, 1, 1]) #batch_size x 3*hidden_size x 1
        #print("w_sim_tiled", w_sim_tiled.get_shape())
        sim_mtx_flat = tf.squeeze(tf.matmul(combo_input_flat, w_sim_tiled), 2) #batch_size x p_length*q_length
        #print("sim_mtx_flat", sim_mtx_flat.get_shape())
        sim_mtx = tf.reshape(tf.transpose(sim_mtx_flat), [p_length, q_length, batch_size]) #p_length x q_length x batch_size

        #C2Q attention: how relevant are the query words to each context word?
        att_on_q = tf.nn.softmax(sim_mtx, dim=1) #a #for each p, find weights to put on q. #p_length x q_length x batch_size

        weighted_q_for_each_p = [] #U~ p_length x hidden_size x batch_size
        for t in range(p_length):
            weights_q = tf.tile(tf.expand_dims(tf.transpose(att_on_q[t]), axis=2), tf.constant([1, 1, 2*hidden_size])) #batch_size x q_length x hidden_size. Transposed to fit q_inputs size.
            weighted_q = tf.multiply(weights_q, q_inputs) #batch_size x q_length x hidden_size
            linear_combo_q = tf.transpose(tf.reduce_sum(weighted_q, axis=1)) #linear combination of weighted question words. hidden_size x batch_size
            weighted_q_for_each_p.append(linear_combo_q)
        weighted_q_for_each_p = tf.stack(weighted_q_for_each_p, axis=0) #p_length x hidden_size x batch_size
        weighted_q_for_each_p = tf.reshape(weighted_q_for_each_p, [batch_size, p_length, 2*hidden_size])

        print("weighted_q_for_each_p", weighted_q_for_each_p.get_shape())

        #Q2C Attention: which context words have the closest similarity to one of the query words?
        #for each context word choose which query word it helps contribute to the most
        #then normalize over all context words, to get a distribution of helpfulness of all context words to this query
        att_on_p = tf.nn.softmax(tf.reduce_max(sim_mtx, axis=1), dim=0) #p_length x batch_size
        weights_p = tf.tile(tf.expand_dims(att_on_p, axis=0), tf.constant([2*hidden_size, 1, 1])) #hidden_size x p_length x batch_size
        weighted_p = tf.multiply(tf.transpose(weights_p), p_inputs) #batch_size x p_length x hidden_size

        print("weighted_p", weighted_p.get_shape())

        #p_inputs = batch_size x p_length x hidden_size
        #weighted_q_for_each_p = batch_size x p_length x hidden_size
        #weighted_p = batch_size x p_length x hidden_size
        self.outputs = tf.concat([p_inputs, weighted_q_for_each_p, tf.multiply(p_inputs, weighted_q_for_each_p), tf.multiply(p_inputs, weighted_p)], axis=2)
        #self.outputs = batch_size x p_length x 4*hidden_size

        



