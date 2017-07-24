import tensorflow as tf
from tensorflow.contrib import rnn
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"

class LogitsLayer():
    def __init__(self, args, inputs, scope):
        print("building logits layer", scope)
        batch_size = args.batch_size
        vocab_size = args.vocab_size
        keep_prob = args.keep_prob
        model = args.model
        num_layers = args.num_layers
        training = args.training

        #inputs = #batch_size x p_length x hidden_size
        input_shape = inputs.get_shape()
        
        if training and args.keep_prob < 1.0:
            inputs = tf.nn.dropout(inputs, args.keep_prob)
        
        with tf.variable_scope(scope): #tf.random_normal_initializer(0.0, 0.1)
            w_p1 = tf.get_variable("w_p1", [input_shape[2], 1], initializer=tf.random_normal_initializer(0.0, 0.1)) #8*hidden_size x 1
            w_p2 = tf.get_variable("w_p2", [input_shape[2], 1], initializer=tf.random_normal_initializer(0.0, 0.1))
            #b_p1 = tf.get_variable("b_p1", [1], initializer=tf.constant_initializer(0.1))
            #b_p2 = tf.get_variable("b_p2", [1], initializer=tf.constant_initializer(0.1))
            tf.summary.histogram("logits_p1", w_p1)
            tf.summary.histogram("logits_p2", w_p2)

        w_p1_tiled = tf.tile(tf.expand_dims(w_p1, 0), [batch_size, 1, 1]) #batch_size x 8*hidden_size x 1
        w_p2_tiled = tf.tile(tf.expand_dims(w_p2, 0), [batch_size, 1, 1])

        pred_start_dist = tf.nn.softmax(tf.matmul(inputs, w_p1_tiled), 1) #batch_size x p_length x 1

        start_dist_weighted_inputs = tf.multiply(pred_start_dist, inputs)

        pred_end_dist = tf.nn.softmax(tf.matmul(start_dist_weighted_inputs, w_p2_tiled), 1) #end dist depends on start dist

        self.pred_start_dist = tf.squeeze(pred_start_dist, -1) #batch_size x p_length
        self.pred_end_dist = tf.squeeze(pred_end_dist, -1)