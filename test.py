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
from LogitsLayer import LogitsLayer
from LossLayer import LossLayer

def main():
    parser = argparse.ArgumentParser(
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_dir', type=str, default='data',
                        help='data directory')
    parser.add_argument('--save_dir', type=str, default='save',
                        help='directory to store checkpointed models')
    parser.add_argument('--rnn_size', type=int, default=200,
                        help='size of RNN hidden state')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='number of layers in the RNN')
    parser.add_argument('--model', type=str, default='lstm',
                        help='rnn, gru, lstm, or nas')
    args = parser.parse_args()
    train(args)

def test(args):
    data_loader = DataLoader(args.data_dir, embedding_dim=args.rnn_size, batch_size=1)
    data_loader.create_batches()

    with open(os.path.join(args.save_dir, 'config.pkl'), 'rb') as f:
        saved_args = pickle.load(f)
    #with open(os.path.join(args.save_dir, 'chars_vocab.pkl'), 'rb') as f:
    #    chars, vocab = cPickle.load(f)






    args.training = False
    model = Model(saved_args, training=False)
    print("loaded model...")
    with tf.Session() as sess:
        print("started session...")
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(args.save_dir)
        print("got checkpoint...")
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("restored session...")
            data_loader.reset_batch_pointer()
            print("looping through test data...")
            total_loss = 0
            for b in range(data_loader.num_batches):
                start = time.time()
                x, y = data_loader.next_batch()
                feed = {model.input_data: x, model.targets: y}
                test_loss = sess.run(model.cost, feed)
                total_loss += test_loss
            average_loss = total_loss / data_loader.num_batches
            print("average loss is", average_loss)
if __name__ == '__main__':
    main()
