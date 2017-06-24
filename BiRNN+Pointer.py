import tensorflow as tf
from tensorflow.contrib import rnn
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"


class Model():
    def __init__(self, args, training=True):
        self.args = args
        if not training:
            args.batch_size = 1
            #args.seq_length = 1
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

        
        self.cells_fw = []
        for _ in range(args.num_layers):
            cell = cell_fn(args.rnn_size)
            if training and (args.output_keep_prob < 1.0 or args.input_keep_prob < 1.0):
                cell = rnn.DropoutWrapper(cell,
                                          input_keep_prob=args.input_keep_prob,
                                          output_keep_prob=args.output_keep_prob)
            self.cells_fw.append(cell) #cells is num_layers of cell stacked together

        self.cells_bw = []
        for _ in range(args.num_layers):
            cell = cell_fn(args.rnn_size)
            if training and (args.output_keep_prob < 1.0 or args.input_keep_prob < 1.0):
                cell = rnn.DropoutWrapper(cell,
                                          input_keep_prob=args.input_keep_prob,
                                          output_keep_prob=args.output_keep_prob)
            self.cells_bw.append(cell)

        self.input_data = tf.placeholder(tf.int32, [args.batch_size, None]) #allows for variable seq_length
        self.targets_start = tf.placeholder(tf.int32, [args.batch_size])
        self.targets_end = tf.placeholder(tf.int32, [args.batch_size])

        self.initial_states_fw = [self.cells_fw[i].zero_state(args.batch_size, tf.float32) for i in range(args.num_layers)]
        self.initial_states_bw = [self.cells_bw[i].zero_state(args.batch_size, tf.float32) for i in range(args.num_layers)]

        embedding = tf.get_variable("embedding", [args.vocab_size, args.rnn_size])
        inputs = tf.nn.embedding_lookup(embedding, self.input_data) #batch_size x seq_length x rnn_size

        # dropout beta testing: double check which one should affect next line
        if training and args.output_keep_prob:
            inputs = tf.nn.dropout(inputs, args.output_keep_prob)

        #computes the actual lengths of each input to feed into stack_bidirectional_dynamic_rnn
        def compute_lengths(inputs):
            used = tf.sign(tf.reduce_max(tf.abs(inputs), reduction_indices=2))
            lengths = tf.reduce_sum(used, reduction_indices=1)
            lengths = tf.cast(lengths, tf.int32) #lengths must be integers
            return lengths
        seq_lengths = compute_lengths(inputs)
        max_length = tf.reduce_max(seq_lengths)

        #outputs: batch_size x rnn_size and there are seq_length number of outputs
        outputs, output_states_fw, output_states_bw = rnn.stack_bidirectional_dynamic_rnn(self.cells_fw, self.cells_bw, inputs, self.initial_states_fw, self.initial_states_bw, dtype=tf.float32, sequence_length=seq_lengths, scope="rnnlm")

        output_state_fw = tf.concat([output_states_fw[i].h for i in range(args.num_layers)], axis=1) #combine output states from all layers, throwing away cell state
        output_state_bw = tf.concat([output_states_bw[i].h for i in range(args.num_layers)], axis=1)

        #outputs, last_state = legacy_seq2seq.rnn_decoder(inputs, self.initial_state, cell, loop_function=loop if not training else None, scope='rnnlm')
        #output = tf.reshape(tf.concat(outputs, 1), [-1, 2*args.rnn_size])
        #first concatenate the outputs by combining all seq_length number of ouputs-->seq_length*batch_size x rnn_size
        #to predict something in the middle, we need to concatenate the hidden units from both sides 
        #we have both fw and bw outputs, so rnn_size*2
        self.output_state_combo = tf.concat([output_state_fw, output_state_bw], axis=1) #batch_size x num_layers*rnn_size*2
        
        with tf.variable_scope('rnnlm'):
            w_p1 = tf.get_variable("w_p1", [args.num_layers*args.rnn_size*2, 1000]) #*2 for forward and backward weights
            w_p2 = tf.get_variable("w_p2", [args.num_layers*args.rnn_size*2, 1000])
        # seq_length*batch_size x 2*rnn_size x 2*rnn_size x vocab_size + tiled(vocab_size) = 2500 x 66
        self.p1 = tf.matmul(self.output_state_combo, w_p1) #batch_size x seq_length
        self.p2 = tf.matmul(self.output_state_combo, w_p2)

        #transform each of the batch_size*seq_length entries in self.targets into one-hot vector
        targets_start_reshaped = tf.reshape(self.targets_start, [-1, 1])
        targets_end_reshaped = tf.reshape(self.targets_end, [-1, 1])
        targets_start_onehot = tf.one_hot(targets_start_reshaped, depth=1000, on_value = 1.0, off_value=0.0)
        targets_end_onehot = tf.one_hot(targets_end_reshaped, depth=1000, on_value = 1.0, off_value=0.0)
        targets_start_squeezed = tf.squeeze(targets_start_onehot, [1])
        targets_end_squeezed = tf.squeeze(targets_end_onehot, [1])

        #self.probs = tf.nn.softmax(self.logits) #probability distribution over vocabulary for next token
        with tf.name_scope('cost'):
            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.p1, labels=targets_start_squeezed) + tf.nn.softmax_cross_entropy_with_logits(logits=self.p2, labels=targets_end_squeezed))

        self.learning_rate = tf.Variable(0.0, trainable=False)

        with tf.name_scope('optimizer'):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.train_op = self.optimizer.minimize(self.cost)

        # instrument tensorboard
        #tf.summary.histogram('logits', self.logits)
        #tf.summary.histogram('loss', loss)
        #tf.summary.scalar('train_loss', self.cost)

