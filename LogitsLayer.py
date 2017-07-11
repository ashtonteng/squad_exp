import tensorflow as tf
from tensorflow.contrib import rnn
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"

class LogitsLayer():
    def __init__(self, inputs, scope, batch_size=20, input_keep_prob=1.0, 
                 output_keep_prob=1.0, training=True):
        print("building logits layer", scope)
        #inputs = #batch_size x p_length x hidden_size
        input_shape = inputs.get_shape()
        
        # dropout beta testing: double check which one should affect next line
        if training and output_keep_prob < 1.0:
            inputs = tf.nn.dropout(inputs, output_keep_prob)
        
        with tf.variable_scope(scope):
            w_p1 = tf.get_variable("w_p1", [input_shape[2], 1], initializer=tf.random_normal_initializer) #8*rnn_size x 1
            w_p2 = tf.get_variable("w_p2", [input_shape[2], 1], initializer=tf.random_normal_initializer)

        w_p1_tiled = tf.tile(tf.expand_dims(w_p1, 0), [batch_size, 1, 1]) #batch_size x 8*hidden_size x 1
        w_p2_tiled = tf.tile(tf.expand_dims(w_p2, 0), [batch_size, 1, 1])

        self.pred_start_dist = tf.squeeze(tf.nn.softmax(tf.matmul(inputs, w_p1_tiled), 1), -1) #batch_size x p_length
        self.pred_end_dist = tf.squeeze(tf.nn.softmax(tf.matmul(inputs, w_p2_tiled), 1), -1)

        #self.predicted_starts = tf.argmax(p1, axis=1) #batch_size
        #self.predicted_ends = tf.argmax(p2, axis=1)