import tensorflow as tf
from tensorflow.contrib import rnn
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"

class PointerLayer():
    def __init__(self, args, inputs, scope):
        print("building pointer layer", scope)

        batch_size = args.batch_size
        vocab_size = args.vocab_size
        hidden_size = args.PointerLayer_size
        output_keep_prob = args.output_keep_prob
        input_keep_prob = args.input_keep_prob
        model = args.model
        num_layers = args.num_layers
        training = args.training

        #inputs = #batch_size x p_length x hidden_size
        self.inputs = inputs
        max_seq_length = tf.shape(inputs)[1]
        # dropout beta testing: double check which one should affect next line
        if training and output_keep_prob < 1.0:
            inputs = tf.nn.dropout(inputs, output_keep_prob)

        if not training:
            batch_size = 1
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
        In order to determine how likely the current index j is start/end, we take into account two factors.
            1) Some representation of the jth word in the passage. This is passed in as input to this layer.
            2) How likely the j-1'th word was the start/end. This is produced by a dynamic decoder RNN. The "AnswerRNN".
        """
        with tf.variable_scope(scope): #variables unsed in the pointerRNN
            W_inputs = tf.get_variable("W_inputs", [hidden_size, hidden_size], initializer=tf.random_normal_initializer)
            W_pointerRNN = tf.get_variable("W_pointerRNN", [hidden_size, hidden_size], initializer=tf.random_normal_initializer)
            v = tf.get_variable("v", [hidden_size, 1])
        W_inputs_tiled = tf.tile(tf.expand_dims(W_inputs, 0), [batch_size, 1, 1]) #batch_size x hidden_size x hidden_size
        v_tiled = tf.tile(tf.expand_dims(v, 0), [batch_size, 1, 1]) #batch_size x hidden_size x 1
        weighted_inputs = tf.matmul(inputs, W_inputs_tiled) #batch_size x seq_length x hidden_size

        def compute_lengths(inputs):
            used = tf.sign(tf.reduce_max(tf.abs(inputs), reduction_indices=2))
            lengths = tf.reduce_sum(used, reduction_indices=1)
            lengths = tf.cast(lengths, tf.int32) #lengths must be integers
            return lengths
        seq_lengths = compute_lengths(inputs)
        #TODO: use attention-pooling over the question representation for the initialize hidden vector of pointer network
        def loop_fn(time, cell_output, cell_state, cell_loop_state): #this is what happens between adjacent RNN cells
            emit_output = cell_output #the output that I emit at this step (to be a part of "outputs_ta") is the cell output
            if cell_output is None: #if the cell before me did not have an output, I am at the start of the RNN!
                next_cell_state = cell.zero_state(batch_size, tf.float32) #initialize the state for the first RNN cell
            else:
                next_cell_state = cell_state #just pass on the cell state as it is, no processing in middle. Cell states are changed within the cell.
            next_loop_state = None #dunno what loop_state is, always None
            def get_next_input():
                if cell_output is None: #if I am before the first cell, just predict based on inputs alone.
                    logits = tf.matmul(tf.tanh(weighted_inputs), v_tiled) #batch_size x seq_length x hidden_size
                else:
                    logits = tf.matmul( #if I am between cells, I can spit out a predicted based on my previous cell's output
                                     tf.tanh(weighted_inputs + 
                                             tf.tile(tf.expand_dims(tf.matmul(cell_output, W_pointerRNN), 1), [1, max_seq_length, 1]))
                                     , v_tiled) #batch_size x seq_length x 1
                predicted_probs = tf.nn.softmax(logits, dim=1) #a #batch_size x seq_length x 1
                predicted_probs_tiled = tf.tile(predicted_probs, [1, 1, hidden_size]) #batch_size x seq_length x hidden_size
                weighted_input = tf.reduce_sum(tf.multiply(predicted_probs_tiled, inputs), axis=1) #c #batch_size x hidden_size
                return weighted_input #weighted input is next input #c_t
            elements_finished = (time >= seq_lengths) #this operation produces boolean tensor of [batch_size] defining if corresponding sequence has ended
            finished = tf.reduce_all(elements_finished) #AND operation over all batches. True if all batches finished.
            time += 1
            #if everything in batch has finished, next_input is empty padding. If not, we compute the input that goes into the next cell.
            next_input = tf.cond(finished, lambda: tf.zeros([batch_size, hidden_size], dtype=tf.float32), get_next_input)
            return elements_finished, next_input, next_cell_state, emit_output, next_loop_state
        
        with tf.variable_scope(scope):
            cell = cell_fn(hidden_size) #add dropout wrapper and num_layers support?
            outputs_ta, final_state, _ = tf.nn.raw_rnn(cell, loop_fn, scope=scope) #final_state is the last emit_output #batch_size x hidden_size
            #outputs = outputs_ta.stack()

        logits = tf.matmul(
                            tf.tanh(weighted_inputs + 
                                    tf.tile(tf.expand_dims(tf.matmul(final_state, W_pointerRNN), 1), [1, max_seq_length, 1]))
                            , v_tiled) #batch_size x seq_length x 1
        predicted_probs = tf.squeeze(tf.nn.softmax(logits, dim=1), axis=-1) #batch_size x seq_length
        self.pred_dist = predicted_probs
        #predicted_idx = tf.argmax(self.predicted_probs, axis=2)
        #print("predicted index is", predicted_idx)
