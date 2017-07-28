import tensorflow as tf
from tensorflow.contrib import rnn
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"

class ConvLayer():
    def __init__(self, args, scope, inputs=None):
        print("building ConvLayer", scope)
        batch_size = args.batch_size
        vocab_size = args.vocab_size
        hidden_size = args.ConvLayer_size
        output_keep_prob = args.output_keep_prob
        input_keep_prob = args.input_keep_prob
        model = args.model
        num_layers = 4
        training = args.training
        filter_size = 5
        block_size = 2

        with tf.name_scope(scope):

            if inputs is None: #this layer is first layer! Get feed_dict data and embed.
                self.inputs = tf.placeholder(tf.int32, [batch_size, None]) #allows for variable seq_length
                with tf.variable_scope(scope):
                    self.embedding = tf.get_variable("embedding", [vocab_size, hidden_size])
                inputs = tf.nn.embedding_lookup(self.embedding, self.inputs) #batch_size x seq_length x hidden_size
                tf.summary.histogram("glove_embedding", self.embedding)

            # dropout beta testing: double check which one should affect next line
            if training and output_keep_prob < 1.0:
                inputs = tf.nn.dropout(inputs, output_keep_prob)

            with tf.variable_scope(scope): #variables unsed in the pointerRNN
                inputs = tf.expand_dims(inputs, -1)
                res_inputs = tf.identity(inputs) / 10 #for residual connections, divided by 10 to not overwhelm new inputs after sigmoid

                for i in range(num_layers):
                    #filter_size = filter_size if i < num_layers-1 else 1
                    filter_shape = (filter_size, hidden_size, 1, 1) #num_filters = 1
                    with tf.variable_scope("layer_%d"%i):
                        conv_w = self.conv_op(inputs, filter_shape, "linear")
                        conv_v = self.conv_op(inputs, filter_shape, "gated")
                        inputs = conv_w * tf.sigmoid(conv_v)
                        if i % block_size == 0:
                            inputs += res_inputs
                            res_inputs = inputs
            self.outputs = tf.reshape(inputs, [batch_size, -1, hidden_size])

    def conv_op(self, inputs, filter_shape, name, scope):
        W = tf.get_variable("%s_W"%name, filter_shape, tf.float32, tf.random_normal_initializer(0.0, 0.1))
        b = tf.get_variable("%s_b"%name, filter_shape[-1], tf.float32, tf.constant_initializer(0.1))
        tf.summary.histogram("%s_W"%name, W)
        tf.summary.histogram("%s_b"%name, b)
        return tf.nn.bias_add(tf.nn.conv2d(inputs, W, strides=[1,1,1,1], padding='SAME'), b)
