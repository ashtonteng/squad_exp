import tensorflow as tf
import argparse
import time
import pickle
import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"

from utils import DataLoader
from BiRNNLayer import BiRNNLayer
from PointerLayer import PointerLayer
from AttentionLayer import AttentionLayer


def main():
    parser = argparse.ArgumentParser(
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_dir', type=str, default='data',
                        help='data directory')
    parser.add_argument('--rnn_size', type=int, default=128,
                        help='size of RNN hidden state')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='number of layers in the RNN')
    parser.add_argument('--model', type=str, default='lstm',
                        help='rnn, gru, lstm, or nas')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='minibatch size')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='number of epochs')
    parser.add_argument('--save_every', type=int, default=100,
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
    args = parser.parse_args()
    train(args)

def train(args):
    data_loader = DataLoader(args.data_dir, args.batch_size)
    data_loader.create_batches()
    args.max_para_length = data_loader.max_para_length
    args.max_ques_length = data_loader.max_ques_length
    args.vocab_size = data_loader.vocab_size

    #Initialize layers
    paragraph_layer = BiRNNLayer(args.vocab_size, batch_size=args.batch_size, scope="paraBiRNN")
    question_layer = BiRNNLayer(args.vocab_size, batch_size=args.batch_size, scope="quesBiRNN")
    attention_layer = AttentionLayer(paragraph_layer.outputs, question_layer.outputs, batch_size=args.batch_size, hidden_size=args.rnn_size, scope="attention")
    output_layer = PointerLayer(attention_layer.outputs, batch_size=args.batch_size, scope="pointer")

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for e in range(args.num_epochs):
            sess.run(tf.assign(output_layer.learning_rate, args.learning_rate * (args.decay_rate ** e)))
            for b in range(data_loader.num_batches):
                start = time.time()
                paragraphs, p_length, questions, q_length, targets_start, targets_end = data_loader.next_batch_variable_seq_length()

                #y1 = [2]*args.batch_size #pointer labels for start index. Not one-hot.
                #y2 = [3]*args.batch_size #pointer labels for end index. Not one-hot
                feed = {paragraph_layer.inputs: paragraphs,
                        paragraph_layer.seq_length: p_length,
                        question_layer.inputs: questions,
                        question_layer.seq_length: q_length,
                        output_layer.targets_start: targets_start, 
                        output_layer.targets_end: targets_end}
                #for i, (c, h) in enumerate(model.initial_states_fw): #for each LSTMStateTuple
                    #feed[c] = state[i].c
                    #feed[h] = state[i].h
                train_loss, p1, p2, a, b, c, _ = sess.run([output_layer.cost, output_layer.p1, output_layer.p2, output_layer.a, output_layer.b, output_layer.c, output_layer.train_op], feed)
                #print(p1[0][2], p2[0][3])
                #print(p1, p2)
                print(targets_start)
                print(len(p1[0]))
                print("-------")
                print(p1)
                print("-------")
                print(p2)
                print("------")
                print(a)
                print("------")
                print(b)
                print('------')
                print(c)

                # instrument for tensorboard
                #summ, train_loss, state_fw, state_bw, _ = sess.run([summaries, model.cost, model.final_state_fw, model.final_state_bw, model.train_op], feed)
                #writer.add_summary(summ, e * data_loader.num_batches + b)

                end = time.time()
                print("{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}"
                      .format(e * data_loader.num_batches + b,
                              args.num_epochs * data_loader.num_batches,
                              e, train_loss, end - start))
                #if (e * data_loader.num_batches + b) % args.save_every == 0\
                #        or (e == args.num_epochs-1 and
                #            b == data_loader.num_batches-1):
                    # save for the last result
                    #checkpoint_path = os.path.join(args.save_dir, 'model.ckpt')
                    #saver.save(sess, checkpoint_path,
                    #           global_step=e * data_loader.num_batches + b)
                    #print("model saved to {}".format(checkpoint_path))


if __name__ == '__main__':
    main()
