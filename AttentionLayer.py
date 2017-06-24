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
            w_sim = tf.get_variable("w_sim", [6*hidden_size, 1, 1, 1]) #WWRRRROOOONNNNGGG
        print("reshaping")
        sim_mtx = tf.matmul(combo_input, w_sim) #p_length x q_length x batch_size
        #sim_mtx = tf.matmul(tf.reshape(combo_input, [-1, 6*hidden_size]), w_sim) #p_length x q_length x batch_size
        print("sim_mtx shape", sim_mtx.get_shape())

        #C2Q attention: how relevant are the query words to each context word?
        att_on_q_temp = []# p_length, q_length, batch_size #a
        for t in range(p_length):
            att_on_q_temp.append(tf.nn.softmax(sim_mtx[t], dim=1)) #softmax for each row, for all batches. q_length x batch_size
        att_on_q = tf.stack(att_on_q_temp)
        print(att_on_q.get_shape(), "yooooo")

        weighted_q_for_each_p = np.empty((p_length, hidden_size, batch_size)) #U~  x batch_size x p_length x hidden_size
        for t in range(p_length):
            weights_q = tf.tile(tf.expand_dims(tf.transpose(att_on_q[t]), axis=2), tf.constant([1, 1, hidden_size])) #batch_size x q_length x hidden_size. Transposed to fit q_inputs size.
            weighted_q = tf.multiply(weights_q, q_inputs) #batch_size x q_length x hidden_size
            linear_combo_q = tf.reduce_sum(weighted_q, axis=1) #linear combination of weighted question words. batch_size x hidden_size
            weighted_q_for_each_p[:, t, :] = linear_combo_q 

        #Q2C Attention: which context words have the closest similarity to one of the query words?
        #for each context word choose which query word it helps contribute to the most
        #then normalize over all context words, to get a distribution of helpfulness of all context words to this query
        att_on_p = tf.nn.softmax(tf.reduce_max(sim_mtx, axis=1), dim=0) #p_length x batch_size
        weights_p = tf.tile(tf.expand_dims(att_on_p, axis=0), tf.constant([hidden_size, 1, 1])) #hidden_size x p_length x batch_size
        weighted_p = tf.multiply(tf.transpose(weights_p), p_inputs) #batch_size x p_length x hidden_size

        #p_inputs = batch_size x p_length x hidden_size
        #weighted_q_for_each_p = batch_size x p_length x hidden_size
        #weighted_p = batch_size x p_length x hidden_size
        self.outputs = tf.concat([p_inputs, weighted_q_for_each_p, tf.multiply(p_inputs, weighted_q_for_each_p), tf.multiply(p_inputs, weighted_p)], axis=2)
        #self.outputs = batch_size x p_length x 4*hidden_size

        



