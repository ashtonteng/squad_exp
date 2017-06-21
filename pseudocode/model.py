import tensorflow as tf
from tensorflow.contrib import rnn
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
import numpy as np

class BiRNNLayer():
    """A feed-forward layer that implements a bi-directional RNN of variable cell types and number of layers.
       INPUT: batch_size x seq_length
       OUTPUT: seq_length x batch_size x rnn_size
    """
    def __init__(self, args, training=True):
        self.args = args
        if not training:
            args.batch_size = 1
        if args.model == 'rnn':
            cell_fn = rnn.BasicRNNCell
        elif args.model == 'gru':
            cell_fn = rnn.GRUCell
        elif args.model == 'lstm':
            cell_fn = rnn.BasicLSTMCell
        elif args.model == 'nas':
            cell_fn = rnn.NASCell
        else:
            raise Exception("model type not supported: {}".format(args.model))

        #list of num_layers forward cells. Each cell is an unrollable RNN of variable length.
        self.cells_fw = []
        for _ in range(args.num_layers):
            cell = cell_fn(args.rnn_size) #rnn_size is the dimension of the hidden layer
            if training and (args.output_keep_prob < 1.0 or args.input_keep_prob < 1.0):
                cell = rnn.DropoutWrapper(cell,
                                          input_keep_prob=args.input_keep_prob,
                                          output_keep_prob=args.output_keep_prob)
            self.cells_fw.append(cell) #cells is num_layers of cell stacked together

        #list of backward cells
        self.cells_bw = []
        for _ in range(args.num_layers):
            cell = cell_fn(args.rnn_size)
            if training and (args.output_keep_prob < 1.0 or args.input_keep_prob < 1.0):
                cell = rnn.DropoutWrapper(cell,
                                          input_keep_prob=args.input_keep_prob,
                                          output_keep_prob=args.output_keep_prob)
            self.cells_bw.append(cell)

        #placeholder for input data
        self.input_data = tf.placeholder(
            tf.int32, [args.batch_size, args.seq_length])

        #define initial hideen states of each cell as all the default zero_state
        self.initial_states_fw = tuple([self.cells_fw[i].zero_state(args.batch_size, tf.float32) for i in range(args.num_layers)])
        self.initial_states_bw = tuple([self.cells_bw[i].zero_state(args.batch_size, tf.float32) for i in range(args.num_layers)])

        #We define an embedding. This is a look-up table for every item in the vocabulary, for a rnn_size-dimensional hidden vector.
        #This embedding will be learned over time as a part of back-propagation.
        embedding = tf.get_variable("embedding", [args.vocab_size, args.rnn_size])
        #we look up our examples in the embedding to expand the input to rnn_size dimensions.
        inputs = tf.nn.embedding_lookup(embedding, self.input_data)

        # dropout beta testing: double check which one should affect next line
        if training and args.output_keep_prob:
            inputs = tf.nn.dropout(inputs, args.output_keep_prob)

        #split the input items one by one. If char_level, split everything into letters. If word_level, split into words.
        inputs = tf.split(inputs, args.seq_length, 1) 
        #inputs is a length seq_length list of batch_size x rnn_size tensors
        inputs = [tf.squeeze(input_, [1]) for input_ in inputs] #get rid of the 1-dimension at axis 1, flatten

        #define bidirectional_rnn layer
        #outputs: batch_size x rnn_size and there are seq_length number of outputs. Outputs at every step!
        self.outputs, self.final_state_fw, self.final_state_bw = rnn.stack_bidirectional_rnn(self.cells_fw, self.cells_bw, inputs, self.initial_states_fw, self.initial_states_bw, tf.float32, scope="rnnlm")

        #first concatenate the outputs by combining all seq_length number of ouputs --> seq_length*batch_size x rnn_size
        #At each unit, we concatenate both fw and bw outputs, so its hidden state is now rnn_size*2 dimensions
        #self.output = tf.reshape(tf.concat(outputs, 1), [-1, 2*args.rnn_size])
