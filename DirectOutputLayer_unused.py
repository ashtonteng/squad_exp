import tensorflow as tf
from tensorflow.contrib import rnn
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"

class DirectOutputLayer():
    def __init__(self, inputs, scope, num_layers=1, batch_size=20, input_keep_prob=1.0, 
                 output_keep_prob=1.0, training=True):
        print("building pointer layer", scope)
        #inputs = #batch_size x p_length x hidden_size
        input_shape = inputs.get_shape()
        
        # dropout beta testing: double check which one should affect next line
        if training and output_keep_prob < 1.0:
            inputs = tf.nn.dropout(inputs, output_keep_prob)

        self.targets_start = tf.placeholder(tf.int32, [batch_size])
        self.targets_end = tf.placeholder(tf.int32, [batch_size])
        
        with tf.variable_scope(scope):
            w_p1 = tf.get_variable("w_p1", [input_shape[2], 1], initializer=tf.random_normal_initializer) #8*rnn_size x 1
            w_p2 = tf.get_variable("w_p2", [input_shape[2], 1], initializer=tf.random_normal_initializer)

        w_p1_tiled = tf.tile(tf.expand_dims(w_p1, 0), [batch_size, 1, 1]) #batch_size x 8*hidden_size x 1
        w_p2_tiled = tf.tile(tf.expand_dims(w_p2, 0), [batch_size, 1, 1])

        p1 = tf.squeeze(tf.nn.softmax(tf.matmul(inputs, w_p1_tiled), 1), -1) #batch_size x p_length
        p2 = tf.squeeze(tf.nn.softmax(tf.matmul(inputs, w_p2_tiled), 1), -1)
        self.predicted_starts = tf.argmax(p1, axis=1) #batch_size
        self.predicted_ends = tf.argmax(p2, axis=1)

        #INDEX DISTANCE LOSS
        #self.cost = tf.reduce_mean(tf.abs(self.predicted_starts-self.targets_start))+tf.reduce_mean(tf.abs(self.predicted_ends-self.targets_end))

        #NEGATIVE LOG LOSS
        p1_mask = tf.one_hot(self.targets_start, depth=tf.shape(inputs)[1], on_value=1.0, off_value=0.0) #batch_size x p_length
        p2_mask = tf.one_hot(self.targets_end, depth=tf.shape(inputs)[1], on_value=1.0, off_value=0.0)
        self.cost = -tf.reduce_mean(tf.log(tf.reduce_sum(tf.multiply(p1, p1_mask), axis=1)) + tf.log(tf.reduce_sum(tf.multiply(p2, p2_mask), axis=1)))

        #CROSS ENTROPY LOSS
        #transform each of the batch_size*seq_length entries in self.targets into one-hot vector
        #targets_start_onehot = tf.one_hot(self.targets_start, depth=tf.shape(inputs)[1], on_value=1.0, off_value=0.0) #batch_size x p_length
        #targets_end_onehot = tf.one_hot(self.targets_end, depth=tf.shape(inputs)[1], on_value=1.0, off_value=0.0)

        #self.probs = tf.nn.softmax(self.logits) #probability distribution over vocabulary for next token
        #with tf.name_scope('cost'):
        #    self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=p1, labels=targets_start_onehot)
        #                            + tf.nn.softmax_cross_entropy_with_logits(logits=p2, labels=targets_end_onehot))
        self.learning_rate = tf.Variable(0.0, trainable=False)

        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            gvs = optimizer.compute_gradients(self.cost)
            capped_gvs = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gvs]
            self.train_op = optimizer.apply_gradients(capped_gvs)