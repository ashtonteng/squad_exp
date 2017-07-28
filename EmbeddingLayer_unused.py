import tensorflow as tf
from tensorflow.contrib import rnn
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"

class EmbeddingLayer():
    def __init__(self, args, embed_mtx_shape, trainable, scope, inputs=None):
        print("building BiRNNLayer", scope)
        batch_size = args.batch_size
        vocab_size = args.vocab_size

        with tf.name_scope(scope): 
            self.inputs = tf.placeholder(tf.int32, [batch_size, None]) #allows for variable seq_length
            with tf.variable_scope(scope):
                embed_mtx = tf.get_variable("embed_mtx", shape=embed_mtx_shape, trainable=trainable, initializer=tf.zeros_initializer)
                embed_mtx_placeholder = tf.placeholder(tf.float32, shape=embed_mtx_shape)
                self.embed_init = embed_mtx.assign(embed_mtx_placeholder)
                tf.summary.histogram("embed_mtx", embed_mtx)
                self.outputs = tf.nn.embedding_lookup(embed_mtx, self.inputs) #batch_size x seq_length x rnn_size