import tensorflow as tf
from tensorflow.contrib import rnn
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"

class MatchingLayer():
    def __init__(self, args, p_inputs, q_inputs, scope):
        print("building matching layer", scope)

        batch_size = args.batch_size
        vocab_size = args.vocab_size
        hidden_size = args.MatchingLayer_size
        model = args.model
        num_layers = args.num_layers
        training = args.training

        #p_inputs = #batch_size x p_length x hidden_size
        #q_inputs = #batch_size x q_length x hidden_size
        max_p_length = tf.shape(p_inputs)[1]
        max_q_length = tf.shape(q_inputs)[1]

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
            1) W_uQ * u_jQ = how important is the jth q word to the t'th p word
            2) W_uP * u_tP = how important is the t'th p word just by itself
            3) W_vP * v_t-1P = how much attention did we pay to the t-1'th word previously
        """
        with tf.variable_scope(scope): #variables unsed in the pointerRNN
            W_uQ = tf.get_variable("W_uQ", [hidden_size, hidden_size])#, initializer=tf.random_normal_initializer)
            W_uP = tf.get_variable("W_uP", [hidden_size, hidden_size])#, initializer=tf.random_normal_initializer)
            W_vP = tf.get_variable("W_vP", [2*hidden_size, hidden_size])#, initializer=tf.random_normal_initializer)
            W_g = tf.get_variable("W_g", [2*hidden_size, 2*hidden_size])#, initializer=tf.random_normal_initializer)
            v = tf.get_variable("v", [hidden_size, 1])
            tf.summary.histogram("W_uQ", W_uQ)
            tf.summary.histogram("W_uP", W_uP)
            tf.summary.histogram("W_vP", W_vP)
            tf.summary.histogram("W_g", W_g)
            tf.summary.histogram("v", v)

        W_uQ_tiled = tf.tile(tf.expand_dims(W_uQ, 0), [batch_size, 1, 1]) #batch_size x hidden_size x hidden_size
        W_uP_tiled = tf.tile(tf.expand_dims(W_uP, 0), [batch_size, 1, 1]) #batch_size x hidden_size x hidden_size
        W_vP_tiled = tf.tile(tf.expand_dims(W_vP, 0), [batch_size, 1, 1]) #batch_size x hidden_size x hidden_size
        v_tiled = tf.tile(tf.expand_dims(v, 0), [batch_size, 1, 1]) #batch_size x hidden_size x 1

        weighted_q_inputs = tf.matmul(q_inputs, W_uQ_tiled) #batch_size x max_q_length x hidden_size
        weighted_p_inputs = tf.matmul(p_inputs, W_uP_tiled)
        #weighted_p_inputs = tf.tile(tf.expand_dims(tf.matmul(p_inputs, W_uP_tiled), 2), [1, 1, max_q_length, 1]) #batch_size x max_p_pength x hidden_sizets, [0, 3, 0, 0], [-1, 1, -1, -1]), 1))
        tf.summary.histogram("weighted_p_inputs", weighted_p_inputs)

        def compute_lengths(inputs):
            used = tf.sign(tf.reduce_max(tf.abs(inputs), reduction_indices=2))
            lengths = tf.reduce_sum(used, reduction_indices=1)
            lengths = tf.cast(lengths, tf.int32) #lengths must be integers
            return lengths
        p_lengths = compute_lengths(p_inputs)

        #TODO: use attention-pooling over the question representation for the initialize hidden vector of pointer network
        def loop_fn(time, cell_output, cell_state, cell_loop_state): #this is what happens between adjacent RNN cell
            emit_output = cell_output #the output that I emit at this step (to be a part of "outputs_ta") is the cell output
            if cell_output is None: #if the cell before me did not have an output, I am at the start of the RNN!
                next_cell_state = cell.zero_state(batch_size, tf.float32) #initialize the state for the first RNN cell
            else:
                next_cell_state = cell_state #just pass on the cell state as it is, no processing in middle. Cell states are changed within the cell.
            next_loop_state = None #dunno what loop_state is, always None
            def get_next_input():
                time_index = tf.stack([tf.constant(0, dtype=tf.int32), time, tf.constant(0, dtype=tf.int32)], axis=0)
                p_inputs_slice = tf.slice(weighted_p_inputs, time_index, [-1, 1, -1]) #batch_size x 1 x hidden_size
                if cell_output is None: #if I am before the first cell, just predict based on inputs alone.
                    logits = tf.matmul(tf.tanh(weighted_q_inputs), v_tiled) #batch_size x max_q_length x hidden_size
                else:
                    logits = tf.matmul( #if I am between cells, I can spit out a predicted based on my previous cell's output
                                     tf.tanh(weighted_q_inputs + #batch_size x max_q_length x hidden_si ze
                                             tf.tile(p_inputs_slice, [1, max_q_length, 1]) + #batch_size x max_q_length x hidden_size
                                             tf.tile(tf.matmul(tf.expand_dims(cell_output, 1), W_vP_tiled), [1, max_q_length, 1])) #batch_size x max_q_length x hidden_size
                                     , v_tiled) #batch_size x max_q_length x 1
                attention_over_question = tf.nn.softmax(logits, dim=1) #a #batch_size x seq_length x 1
                #attention_over_question_tiled = tf.tile(attention_over_question, [1, 1, hidden_size]) #batch_size x seq_length x hidden_size
                weighted_question = tf.reduce_sum(tf.multiply(attention_over_question, q_inputs), axis=1) #c #batch_size x hidden_size
                weighted_question_concat_p_input = tf.concat([tf.squeeze(p_inputs_slice, axis=1), weighted_question], axis=1) #matchLSTM, concatenate c_t and u_t_P along hidden_size #batch_size x 2*hidden_size
                gate = tf.sigmoid(tf.matmul(weighted_question_concat_p_input, W_g)) #batch_size x 2*hidden_size
                return tf.multiply(gate, weighted_question_concat_p_input) #weighted input is next input #c_t
            elements_finished = (time >= p_lengths) #this operation produces boolean tensor of [batch_size] defining if corresponding sequence has ended
            finished = tf.reduce_all(elements_finished) #AND operation over all batches. True if all batches finished.
            #if everything in batch has finished, next_input is empty padding. If not, we compute the input that goes into the next cell.
            next_input = tf.cond(finished, lambda: tf.zeros([batch_size, 2*hidden_size], dtype=tf.float32), get_next_input)
            return elements_finished, next_input, next_cell_state, emit_output, next_loop_state
        
        with tf.variable_scope(scope):
            cell = cell_fn(2*hidden_size) #add dropout wrapper and num_layers support?
            outputs_ta, final_state, _ = tf.nn.raw_rnn(cell, loop_fn, scope=scope) #final_state is the last emit_output #batch_size x hidden_size
        self.outputs = tf.reshape(outputs_ta.stack(), [batch_size, max_p_length, 2*hidden_size])
