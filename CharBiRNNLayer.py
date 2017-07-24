import tensorflow as tf
from tensorflow.contrib import rnn
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"

class CharBiRNNLayer():
    def __init__(self, args, scope, inputs=None):
        print("building CharBiRNNLayer", scope)
        batch_size = args.batch_size
        char_vocab_size = args.char_vocab_size
        hidden_size = args.BiRNNLayer_size
        model = args.model
        num_layers = args.num_layers
        training = args.training
        max_word_length = args.max_word_length

        self.max_seq_length = tf.placeholder(tf.int32, shape=())
        self.inputs = tf.placeholder(tf.int32, [batch_size, None, max_word_length]) #variable sentence length

        with tf.variable_scope(scope):
            import numpy as np
            #self.embedding = tf.constant((np.random.random((char_vocab_size, hidden_size))), dtype="float32")
            self.embedding = tf.get_variable("char_embedding", [char_vocab_size, hidden_size], dtype="float32", initializer=tf.random_normal_initializer(0.0, 0.1))
            inputs = tf.nn.embedding_lookup(self.embedding, self.inputs) #batch_size x seq_length x word_length x hidden_size
            tf.summary.histogram("char_embedding_hist", self.embedding)

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

        #outputs: batch_size x seq_length x hidden_size has all the outputs at each timepoint
        #output_states_fw: num_layers tuple of batch_size x hidden_size, representing the last state in the forward rnn for each layer
        #same for output_states_bw

        #create TensorArray of length seq_length, containing tensors of size batch_size x 2*hidden_size, to be populated by tf.while_loop
        initial_ta = tf.TensorArray(tf.float32, size=self.max_seq_length)

        def condition(time, output_ta):
            #elements_finished = (time >= seq_lengths) #this operation produces boolean tensor of [batch_size] defining if corresponding sequence has ended
            #finished = tf.reduce_all(elements_finished) #AND operation over all batches. True if all batches finished.
            return tf.less(time, self.max_seq_length)

        def body(time, output_ta):
            time_index = tf.stack([tf.constant(0, dtype=tf.int32), time, tf.constant(0, dtype=tf.int32), tf.constant(0, dtype=tf.int32)], axis=0)
            inputs_slice = tf.squeeze(tf.slice(inputs, time_index, [-1, 1, -1, -1]), 1) #batch_size x word_length x hidden_size
            seq_lengths = compute_lengths(inputs_slice)
            _, output_states_fw, output_states_bw = rnn.stack_bidirectional_dynamic_rnn(cells_fw, cells_bw, inputs_slice, initial_states_fw, initial_states_bw, dtype=tf.float32, sequence_length=seq_lengths, scope=scope)
            outputs_slice = tf.stack([output_states_fw, output_states_bw], -1)
            output_ta = output_ta.write(time, outputs_slice)
            return time + 1, output_ta

        time = tf.constant(0)
        time, output_ta = tf.while_loop(condition, body, [time, initial_ta])
        self.outputs = tf.reshape(output_ta.stack(), [batch_size, self.max_seq_length, 2*hidden_size])