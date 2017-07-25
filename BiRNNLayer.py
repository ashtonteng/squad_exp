import tensorflow as tf
from tensorflow.contrib import rnn
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"

class BiRNNLayer():
    def __init__(self, args, scope, inputs=None):
        print("building BiRNNLayer", scope)
        batch_size = args.batch_size
        vocab_size = args.vocab_size
        hidden_size = args.BiRNNLayer_size
        model = args.model
        num_layers = args.num_layers
        training = args.training

        if inputs is None: #this layer is first layer! Get feed_dict data and embed.
            self.inputs = tf.placeholder(tf.int32, [batch_size, None]) #allows for variable seq_length
            with tf.variable_scope(scope):
                self.embedding = tf.get_variable("embedding", [vocab_size, hidden_size])
            inputs = tf.nn.embedding_lookup(self.embedding, self.inputs) #batch_size x seq_length x hidden_size
            tf.summary.histogram("glove_embedding", self.embedding)

        if training and args.keep_prob < 1.0:
            print("im dropping out!")
            inputs = tf.nn.dropout(inputs, args.keep_prob)

        if model == 'rnn':
            cell_fn = rnn.BasicRNNCell
        elif model == 'gru':
            cell_fn = rnn.GRUCell
        elif model == 'lstm':
            cell_fn = rnn.BasicLSTMCell
        elif model == 'nas':
            cell_fn = rnn.NASCell
        else:
            raise Exception("model type not supported: {}".format(model))

        cells_fw = []
        for _ in range(num_layers):
            cell = cell_fn(hidden_size)
            if training and args.keep_prob < 1.0:
                cell = rnn.DropoutWrapper(cell, input_keep_prob=args.keep_prob, output_keep_prob=args.keep_prob)
            cells_fw.append(cell) #cells is num_layers of cell stacked together

        cells_bw = []
        for _ in range(num_layers):
            cell = cell_fn(hidden_size)
            if training and args.keep_prob < 1.0:
                cell = rnn.DropoutWrapper(cell, input_keep_prob=args.keep_prob, output_keep_prob=args.keep_prob)
            cells_bw.append(cell)

        initial_states_fw = [cells_fw[i].zero_state(batch_size, tf.float32) for i in range(num_layers)]
        initial_states_bw = [cells_bw[i].zero_state(batch_size, tf.float32) for i in range(num_layers)]

        #computes the actual lengths of each input to feed into stack_bidirectional_dynamic_rnn
        def compute_lengths(inputs):
            used = tf.sign(tf.reduce_max(tf.abs(inputs), reduction_indices=2))
            lengths = tf.reduce_sum(used, reduction_indices=1)
            lengths = tf.cast(lengths, tf.int32) #lengths must be integers
            return lengths
        seq_lengths = compute_lengths(inputs)

        #outputs: batch_size x seq_length x hidden_size has all the outputs at each timepoint
        #output_states_fw: num_layers tuple of batch_size x hidden_size, representing the last state in the forward rnn for each layer
        #same for output_states_bw
        outputs, output_states_fw, output_states_bw = rnn.stack_bidirectional_dynamic_rnn(cells_fw, cells_bw, inputs, initial_states_fw, initial_states_bw, dtype=tf.float32, sequence_length=seq_lengths, scope=scope)
        self.outputs = outputs

        #output_state_fw = tf.concat([output_states_fw[i].h for i in range(num_layers)], axis=1) #combine output states from all layers, throwing away cell state
        #output_state_bw = tf.concat([output_states_bw[i].h for i in range(num_layers)], axis=1)
        #print(self.outputs.get_shape())
        #print(output_state_fw.get_shape())
        #to predict something in the middle, we need to concatenate the hidden units from both sides 
        #we have both fw and bw outputs, so hidden_size*2
        #output_states_combo = tf.concat([output_state_fw, output_state_bw], axis=1) #batch_size x num_layers*hidden_size*2
