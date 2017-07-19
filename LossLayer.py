import tensorflow as tf
from tensorflow.contrib import rnn
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"

class LossLayer():
    def __init__(self, args, pred_start_dist, pred_end_dist, scope="loss"):
        print("building loss layer", scope)

        batch_size = args.batch_size
        vocab_size = args.vocab_size
        output_keep_prob = args.output_keep_prob
        input_keep_prob = args.input_keep_prob
        model = args.model
        num_layers = args.num_layers
        training = args.training

        #pred_start_dist_shape = pred_start_dist.get_shape()
        self.pred_start_dist = pred_start_dist
        self.pred_end_dist = pred_end_dist
        max_seq_length = tf.shape(pred_start_dist)[1]
        self.targets_start = tf.placeholder(tf.int32, [batch_size])
        self.targets_end = tf.placeholder(tf.int32, [batch_size])
        targets_start_mask = tf.one_hot(self.targets_start, depth=max_seq_length, on_value=1.0, off_value=0.0)
        targets_end_mask = tf.one_hot(self.targets_end, depth=max_seq_length, on_value=1.0, off_value=0.0)

        self.cost = -tf.reduce_mean(tf.log(tf.reduce_sum(tf.multiply(targets_start_mask, self.pred_start_dist), axis=1)) + tf.log(tf.reduce_sum(tf.multiply(targets_end_mask, self.pred_end_dist), axis=1)))
        tf.summary.scalar('cost', self.cost)
        #INDEX DISTANCE LOSS
        #self.cost = tf.reduce_mean(tf.abs(self.predicted_starts-self.targets_start))+tf.reduce_mean(tf.abs(self.predicted_ends-self.targets_end))

        #NEGATIVE LOG LOSS
        #p1_mask = tf.one_hot(self.targets_start, depth=tf.shape(inputs)[1], on_value=1.0, off_value=0.0) #batch_size x p_length
        #p2_mask = tf.one_hot(self.targets_end, depth=tf.shape(inputs)[1], on_value=1.0, off_value=0.0)
        #self.cost = -tf.reduce_mean(tf.log(tf.reduce_sum(tf.multiply(p1, p1_mask), axis=1)) + tf.log(tf.reduce_sum(tf.multiply(p2, p2_mask), axis=1)))

        #CROSS ENTROPY LOSS
        #transform each of the batch_size*seq_length entries in self.targets into one-hot vector
        #targets_start_onehot = tf.one_hot(self.targets_start, depth=tf.shape(inputs)[1], on_value=1.0, off_value=0.0) #batch_size x p_length
        #targets_end_onehot = tf.one_hot(self.targets_end, depth=tf.shape(inputs)[1], on_value=1.0, off_value=0.0)

        #self.probs = tf.nn.softmax(self.logits) #probability distribution over vocabulary for next token
        #with tf.name_scope('cost'):
        #    self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=p1, labels=targets_start_onehot)
        #                            + tf.nn.softmax_cross_entropy_with_logits(logits=p2, labels=targets_end_onehot))
        self.learning_rate = tf.Variable(0.0, trainable=False)
        tf.summary.scalar('learning_rate', self.learning_rate)

        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.train_op = optimizer.minimize(self.cost)

            #Clipping gradients to prevent explosion
            #gvs = optimizer.compute_gradients(self.cost)
            #capped_gvs = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gvs]
            #self.train_op = optimizer.apply_gradients(capped_gvs)