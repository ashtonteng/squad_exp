from __future__ import print_function
import tensorflow as tf

import argparse
import time
import os
from six.moves import cPickle

from utils import TextLoader
from model import Model


def main():
    parser = argparse.ArgumentParser(
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_dir', type=str, default='data/squad',
                        help='data directory containing input.txt')
    parser.add_argument('--save_dir', type=str, default='save',
                        help='directory to store checkpointed models')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='directory to store tensorboard logs')
    parser.add_argument('--rnn_size', type=int, default=128,
                        help='size of RNN hidden state')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layers in the RNN')
    parser.add_argument('--model', type=str, default='lstm',
                        help='rnn, gru, lstm, or nas')
    parser.add_argument('--batch_size', type=int, default=50,
                        help='minibatch size')
    parser.add_argument('--seq_length', type=int, default=50,
                        help='RNN sequence length')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='number of epochs')
    parser.add_argument('--save_every', type=int, default=1000,
                        help='save frequency')
    parser.add_argument('--grad_clip', type=float, default=5.,
                        help='clip gradients at this value')
    parser.add_argument('--learning_rate', type=float, default=0.002,
                        help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.97,
                        help='decay rate for rmsprop')
    parser.add_argument('--output_keep_prob', type=float, default=1.0,
                        help='probability of keeping weights in the hidden layer')
    parser.add_argument('--input_keep_prob', type=float, default=1.0,
                        help='probability of keeping weights in the input layer')
    parser.add_argument('--init_from', type=str, default=None,
                        help="""continue training from saved model at this path. Path must contain files saved by previous training process:
                            'config.pkl'        : configuration;
                            'chars_vocab.pkl'   : vocabulary definitions;
                            'checkpoint'        : paths to model file(s) (created by tf).
                                                  Note: this file contains absolute paths, be careful when moving files around;
                            'model.ckpt-*'      : file(s) with model definition (created by tf)
                        """)
    args = parser.parse_args()
    train(args)

def train(args):

    """Load Squad Data"""
    train_q = load all questions txt file
    train_p = load all passages txt file

    #Notes
    #Should we split words into equal length chunks, or split by sentences and then pad?

    """Word Embedding Layer"""
    #Questions
    #TODO: figure out the padding issue in batch_cars and batch_words
    word_inputs_q = batch_words(train_q) #the batches are defined by words.
    word_outputs_q = BiRNNLayer(word_inputs_q) #word_seq_length x batch_size x word_rnn_size
    glove_reps_q = get_glove_reps(train_q) #word_seq_length x batch_size x glove_size
    final_word_outputs_q = word_outputs_q + glove_reps_q #word_seq_length x batch_size x (word_rnn_size + glove_size)
    #Passages
    word_inputs_p = batch_words(train_p)
    word_outputs_p = BiRNNLayer(word_inputs_p)
    glove_reps_p = get_glove_reps(train_p)
    final_word_outputs_p = word_outputs_p + glove_reps_p

    """Character Embedding Layer"""
    #Each character sequence has the same characters as the word sequence, padded to align length between sequences.
    #Questions
    char_inputs_q = split_chars(word_inputs_q) #batch_size x char_seq_length #char_seq_length >= num_chars_in_word_seq_length
    char_outputs_q = BiRNNLayer(char_inputs_q) #char_seq_length x batch_size x char_rnn_size
    #group_by_words takes the final_outputs_fw and final_outputs_bw of the characters in that word
    final_char_outputs_q = group_by_words(char_outputs_q) #word_seq_length x batch_size x 2*char_rnn_size 
    #Passages
    char_inputs_p = split_chars(word_inputs_p)
    char_outputs_p = BiRNNLayer(char_inputs_p)
    final_char_outputs_p = group_by_words(char_outputs_p)

    """Combine Character and Word Embeddings"""
    #Let word_rnn_size + glove_size + 2*char_rnn_size = d
    #Questions
    combined_outputs_q = final_char_outptus_q + final_word_outputs_q #word_seq_length x batch_size x d
    #Passages
    combined_outputs_p = final_char_outputs_p + final_word_outputs_p

    """Contextual Embedding Layer"""
    #Questions
    U_q = BiRNNLayer(combined_outputs_q, embed=False) #word_seq_length x batch_size x 2d
    #Passages
    U_p = BiRNNLayer(combined_outputs_p, embed=False)

    """Attention Flow Layer"""
    #Input: contextual vector representations of the question U_q and passage U_p
    #Output: question-aware vector representation of the passage, along with contextual embeddings from previous layer

    #S =  #similarity matrix S_tj is the similarity between the t'th passage word and the j'th question word

    #Passage-to-question attention signifies which query words are most relevant to each context word
    #a_t = #attention weights on query words by t'th context word

    target_starts = one_hot(placeholder([batch_size]))
    target_ends = ont_hot(placeholder([batch_size]))

    with tf.variable_scope('pointer_net'):
        w_p1 = tf.get_variable("w_p1", [?, passage_len])
        w_p2 = tf.get_variable("w_p1", [?, passage_len])

    p1_logits = tf.matmul(output, w_p1) #start_index_distribution #output shape: batch_size x pasage_len
    p2_logits = tf.matmul(output, w_p2)

    self.cost = tf.reduce_mean(tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=p1_logits, labels=target_starts), tf.nn.softmax_cross_entropy_with_logits(logits=p1_logits, labels=target_starts)))





if __name__ == '__main__':
    main()
