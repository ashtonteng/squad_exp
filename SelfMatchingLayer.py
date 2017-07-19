import tensorflow as tf
from tensorflow.contrib import rnn
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"

class SelfMatchingLayer():
    def __init__(self, args, inputs, scope):
        print("building self-matching layer", scope)

        batch_size = args.batch_size
        vocab_size = args.vocab_size
        hidden_size = args.SelfMatchingLayer_size
        output_keep_prob = args.output_keep_prob
        input_keep_prob = args.input_keep_prob
        model = args.model
        num_layers = args.num_layers
        training = args.training

        #inputs = #batch_size x seq_length x hidden_size
        max_seq_length = tf.shape(inputs)[1]

        def compute_lengths(inputs):
            used = tf.sign(tf.reduce_max(tf.abs(inputs), reduction_indices=2))
            lengths = tf.reduce_sum(used, reduction_indices=1)
            lengths = tf.cast(lengths, tf.int32) #lengths must be integers
            return lengths
        seq_lengths = compute_lengths(inputs)

        # dropout beta testing: double check which one should affect next line
        #if training and output_keep_prob < 1.0:
        #    inputs = tf.nn.dropout(inputs, output_keep_prob)

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
        
        """
            1) W_vP * v_jP = how important is the jth p word to the t'th p word
            2) W_vP2 * v_tP = how important is the t'th p word just by itself
        """
        with tf.variable_scope(scope): #variables unsed in the pointerRNN
            W_vP = tf.get_variable("W_vP", [hidden_size, hidden_size])#, initializer=tf.random_normal_initializer)
            W_vP2 = tf.get_variable("W_vP2", [hidden_size, hidden_size])#, initializer=tf.random_normal_initializer)
            W_g = tf.get_variable("W_g", [2*hidden_size, 2*hidden_size])#, initializer=tf.random_normal_initializer)
            v = tf.get_variable("v", [hidden_size, 1])

        W_vP_tiled = tf.tile(tf.expand_dims(W_vP, 0), [batch_size, 1, 1]) #batch_size x hidden_size x hidden_size
        W_vP2_tiled = tf.tile(tf.expand_dims(W_vP2, 0), [batch_size, 1, 1]) #batch_size x hidden_size x hidden_size
        v_tiled = tf.tile(tf.expand_dims(v, 0), [batch_size, 1, 1]) #batch_size x hidden_size x 1

        weighted_inputs = tf.matmul(inputs, W_vP_tiled) #batch_size x seq_length x hidden_size
        weighted_inputs2 = tf.matmul(inputs, W_vP2_tiled) #batch_size x seq_length x hidden_size
        #weighted_inputs2_tiled = tf.tile(tf.expand_dims(weighted_inputs2, 1), [1, max_seq_length, 1, 1]) #batch_size x seq_length x seq_length x hidden_size
        #tf.matmul(tf.tanh(tf.add(tf.expand_dims(weighted_inputs, 1), weighted_inputs2_tiled)), v_tiled) #batch_size x seq_length x 

        #create TensorArray of length seq_length, containing tensors of size batch_size x 2*hidden_size, to be populated by tf.while_loop
        initial_ta = tf.TensorArray(tf.float32, size=max_seq_length)

        def condition(time, input_ta):
            #elements_finished = (time >= seq_lengths) #this operation produces boolean tensor of [batch_size] defining if corresponding sequence has ended
            #finished = tf.reduce_all(elements_finished) #AND operation over all batches. True if all batches finished.
            return tf.less(time, max_seq_length)

        def body(time, input_ta):
            time_index = tf.stack([tf.constant(0, dtype=tf.int32), time, tf.constant(0, dtype=tf.int32)], axis=0)
            inputs_slice = tf.slice(inputs, time_index, [-1, 1, -1]) #batch_size x 1 x hidden_size
            weighted_inputs_slice = tf.matmul(inputs_slice, W_vP2_tiled) #batch_size x 1 x hidden_size
            #time_index = tf.stack([tf.constant(0, dtype=tf.int32), time, tf.constant(0, dtype=tf.int32)], axis=0)
            #weighted_inputs2_slice = tf.slice(weighted_inputs2, time_index, [-1, 1, -1]) #batch_size x 1 x hidden_size
            logits = tf.matmul(tf.tanh(tf.add(weighted_inputs, weighted_inputs_slice)), v_tiled) #batch_size x seq_length x hidden_size * batch_size x hidden_size x 1 = #batch_size x seq_length x 1
            attention_over_passage = tf.nn.softmax(logits, dim=1) # batch_size x seq_length x 1
            weighted_passage = tf.reduce_sum(tf.multiply(attention_over_passage, inputs), axis=1) #batch_size x hidden_size
            weighted_passage_with_inputs = tf.concat([tf.squeeze(inputs_slice, axis=1), weighted_passage], axis=1)
            gate = tf.sigmoid(tf.matmul(weighted_passage_with_inputs, W_g)) #batch_size x hidden_size
            output_ta = input_ta.write(time, tf.multiply(gate, weighted_passage_with_inputs))
            return time + 1, output_ta

        time = tf.constant(0)
        time, output_ta = tf.while_loop(condition, body, [time, initial_ta])
        BiRNN_inputs_stacked = tf.reshape(output_ta.stack(), [batch_size, max_seq_length, 2*hidden_size])

        def compute_lengths(inputs):
            used = tf.sign(tf.reduce_max(tf.abs(inputs), reduction_indices=2))
            lengths = tf.reduce_sum(used, reduction_indices=1)
            lengths = tf.cast(lengths, tf.int32) #lengths must be integers
            return lengths
        seq_lengths = compute_lengths(inputs)

        cells_fw = []
        for _ in range(num_layers):
            cell = cell_fn(2*hidden_size)
            if training and (output_keep_prob < 1.0 or input_keep_prob < 1.0):
                cell = rnn.DropoutWrapper(cell, input_keep_prob=input_keep_prob, output_keep_prob=output_keep_prob)
            cells_fw.append(cell) #cells is num_layers of cell stacked together

        cells_bw = []
        for _ in range(num_layers):
            cell = cell_fn(2*hidden_size)
            if training and (output_keep_prob < 1.0 or input_keep_prob < 1.0):
                cell = rnn.DropoutWrapper(cell, input_keep_prob=input_keep_prob, output_keep_prob=output_keep_prob)
            cells_bw.append(cell)

        initial_states_fw = [cells_fw[i].zero_state(batch_size, tf.float32) for i in range(num_layers)]
        initial_states_bw = [cells_bw[i].zero_state(batch_size, tf.float32) for i in range(num_layers)]

        outputs, output_states_fw, output_states_bw = rnn.stack_bidirectional_dynamic_rnn(cells_fw, cells_bw, BiRNN_inputs_stacked, initial_states_fw, initial_states_bw, dtype=tf.float32, sequence_length=seq_lengths, scope=scope)
        self.outputs = outputs