from __future__ import print_function
import tensorflow as tf


import argparse
import time
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
from six.moves import cPickle

from utils import TextLoader
from model import Model


def main():
    parser = argparse.ArgumentParser(
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_dir', type=str, default='data/squad/test_questions',
                        help='data directory containing input.txt')
    parser.add_argument('--save_dir', type=str, default='save',
                        help='directory to store checkpointed models')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='directory to store tensorboard logs')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='minibatch size')
    parser.add_argument('--seq_length', type=int, default=50,
                        help='RNN sequence length')
    args = parser.parse_args()
    test(args)

def test(args):
    data_loader = TextLoader(args.data_dir, args.batch_size, args.seq_length) #batch_size is 1
    with open(os.path.join(args.save_dir, 'config.pkl'), 'rb') as f:
        saved_args = cPickle.load(f)
    #with open(os.path.join(args.save_dir, 'chars_vocab.pkl'), 'rb') as f:
    #    chars, vocab = cPickle.load(f)
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
