import tensorflow as tf
from tensorflow.contrib import rnn
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"

class PointerLayer():
    def __init__(self, inputs, scope, num_layers=1, batch_size=10, rnn_size=128, input_keep_prob=1.0, 
                 output_keep_prob=1.0, training=True):

        inputs = tf.reshape(inputs, [batch_size, -1]) #concatenate the seq_length number of cells
        input_shape = inputs.get_shape()
        # dropout beta testing: double check which one should affect next line
        if training and output_keep_prob < 1.0:
            inputs = tf.nn.dropout(inputs, output_keep_prob)

        self.targets_start = tf.placeholder(tf.int32, [batch_size])
        self.targets_end = tf.placeholder(tf.int32, [batch_size])
    
        with tf.variable_scope(scope):
            w_p1 = tf.get_variable("w_p1", [input_shape[1], 1000]) # 1000 is random
            w_p2 = tf.get_variable("w_p2", [input_shape[1], 1000])

        self.p1 = tf.matmul(inputs, w_p1)
        self.p2 = tf.matmul(inputs, w_p2)

        #transform each of the batch_size*seq_length entries in self.targets into one-hot vector
        targets_start_reshaped = tf.reshape(self.targets_start, [-1, 1])
        targets_end_reshaped = tf.reshape(self.targets_end, [-1, 1])
        targets_start_onehot = tf.one_hot(targets_start_reshaped, depth=1000, on_value=1.0, off_value=0.0)
        targets_end_onehot = tf.one_hot(targets_end_reshaped, depth=1000, on_value=1.0, off_value=0.0)
        targets_start_squeezed = tf.squeeze(targets_start_onehot, [1])
        targets_end_squeezed = tf.squeeze(targets_end_onehot, [1])

        #self.probs = tf.nn.softmax(self.logits) #probability distribution over vocabulary for next token
        with tf.name_scope('cost'):
            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.p1, labels=targets_start_squeezed)
                                     + tf.nn.softmax_cross_entropy_with_logits(logits=self.p2, labels=targets_end_squeezed))
        
        self.learning_rate = tf.Variable(0.0, trainable=False)

        with tf.name_scope('optimizer'):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.train_op = self.optimizer.minimize(self.cost)
