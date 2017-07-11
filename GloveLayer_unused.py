import tensorflow as tf
from tensorflow.contrib import rnn
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"

class GloveLayer():
    def __init__(self, vocab_size, rnn_size, scope, batch_size=20, training=True):
        print("building GloveLayer", scope)
        #Get embedding matrix from feed via embedding_init
        embed_mtx = tf.Variable(tf.constant(0.0, shape=[vocab_size, rnn_size]), trainable=True, name="embed_mtx")
        self.embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, rnn_size])
        self.embedding_init = embed_mtx.assign(self.embedding_placeholder) #assign W 

        #GloveLayer will always directly get data from feed_dict.
        self.inputs = tf.placeholder(tf.int32, [batch_size, None]) #allows for variable seq_length
        self.outputs = tf.nn.embedding_lookup(embed_mtx, self.inputs) #batch_size x seq_length x rnn_size