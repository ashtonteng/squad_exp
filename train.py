import tensorflow as tf
import argparse
import time
import pickle
import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"

from utils import DataLoader, LoadGloveEmbedding, GetGloveRepresentation
from BiRNNLayer import BiRNNLayer
from PointerLayer import PointerLayer
from AttentionLayer import AttentionLayer

def main():
    parser = argparse.ArgumentParser(
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_dir', type=str, default='data',
                        help='data directory')
    parser.add_argument('--save_dir', type=str, default='save',
                        help='directory to store checkpointed models')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='directory to store tensorboard logs')
    parser.add_argument('--rnn_size', type=int, default=50,
                        help='size of RNN hidden state')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='number of layers in the RNN')
    parser.add_argument('--model', type=str, default='lstm',
                        help='rnn, gru, lstm, or nas')
    parser.add_argument('--batch_size', type=int, default=20,
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
    parser.add_argument('--init_from', type=str, default=None,
                        help="""continue training from saved model at this path. Path must contain files saved by previous training process:
                            'config.pkl'        : configuration;
                            'vocab.pkl'   : vocabulary definitions;
                            'checkpoint'        : paths to model file(s) (created by tf).
                                                  Note: this file contains absolute paths, be careful when moving files around;
                            'model.ckpt-*'      : file(s) with model definition (created by tf)
                        """)
    args = parser.parse_args()
    train(args)

def train(args):
    """Load Train Data"""
    data_loader = DataLoader(args.data_dir, args.batch_size)
    data_loader.create_batches()
    args.vocab_size = data_loader.vocab_size

    """check compatibility if training is continued from previously saved model"""
    if args.init_from is not None:
        # check if all necessary files exist
        assert os.path.isdir(args.init_from)," %s must be a a path" % args.init_from
        assert os.path.isfile(os.path.join(args.init_from,"config.pkl")),"config.pkl file does not exist in path %s"%args.init_from
        #assert os.path.isfile(os.path.join(args.init_from,"vocab.pkl")),"vocab.pkl file does not exist in path %s" % args.init_from
        ckpt = tf.train.get_checkpoint_state(args.init_from)
        assert ckpt, "No checkpoint found"
        assert ckpt.model_checkpoint_path, "No model path found in checkpoint"

        # open old config and check if models are compatible
        with open(os.path.join(args.init_from, 'config.pkl'), 'rb') as f:
            saved_model_args = pickle.load(f)
        need_be_same = ["model", "batch_size", "rnn_size", "num_layers"]
        for checkme in need_be_same:
            assert vars(saved_model_args)[checkme]==vars(args)[checkme],"Command line argument and saved model disagree on '%s' "%checkme

    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)
    with open(os.path.join(args.save_dir, 'config.pkl'), 'wb') as f:
        pickle.dump(args, f)

    """Build Graph of Model"""
    #paragraph_layer = BiRNNLayer(args.vocab_size, batch_size=args.batch_size, rnn_size=args.rnn_size, num_layers=1, scope="paraBiRNN")
    #question_layer = BiRNNLayer(args.vocab_size, batch_size=args.batch_size, rnn_size=args.rnn_size, num_layers=1, scope="quesBiRNN")
    paragraph_char_layer = BiRNNLayer(all_text)
    question_char_layer = BiRNNLayer()
    paragraph_layer = GloveLayer()
    question_layer = GloveLayer()
    attention_layer = AttentionLayer(paragraph_layer.outputs, question_layer.outputs, batch_size=args.batch_size, rnn_size=2*args.rnn_size, scope="attention")
    modelling_layer = BiRNNLayer(args.vocab_size, inputs=attention_layer.outputs, batch_size=args.batch_size, rnn_size=8*args.rnn_size, num_layers=2, scope="modellingBiRNN")
    output_layer = PointerLayer(modelling_layer.outputs, batch_size=args.batch_size, scope="pointer")

    """Run Data through Graph"""
    with tf.Session() as sess:

        # instrument for tensorboard
        summaries = tf.summary.merge_all()
        writer = tf.summary.FileWriter(os.path.join(args.log_dir, time.strftime("%Y-%m-%d-%H-%M-%S")))
        writer.add_graph(sess.graph)

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())
        # restore model
        if args.init_from is not None:
            saver.restore(sess, ckpt.model_checkpoint_path)

        #feed glove embedding into graph
        embedding = LoadGloveEmbedding(glove_dir, glove_dim)
        sess.run(embedding_init, feed_dict={embedding_placeholder: embedding})

        for e in range(args.num_epochs):
            sess.run(tf.assign(output_layer.learning_rate, args.learning_rate * (args.decay_rate ** e)))
            for b in range(data_loader.num_batches):
                start = time.time()
                paragraphs, p_length, questions, q_length, targets_start, targets_end = data_loader.next_batch_variable_seq_length()
                # targets_start = []
                # targets_end = []
                # for paragraph in paragraphs:
                #     try:
                #         targets_start.append(np.where(paragraph==0)[0][0])
                #     except:
                #         targets_start.append(3)
                #     try:
                #         targets_end.append(np.where(paragraph==2)[0][0])
                #     except:
                #         targets_end.append(5)
                feed = {paragraph_layer.inputs: paragraphs,
                        paragraph_layer.seq_length: p_length,
                        question_layer.inputs: questions,
                        question_layer.seq_length: q_length,
                        output_layer.targets_start: targets_start, 
                        output_layer.targets_end: targets_end}
                train_loss, predicted_starts, predicted_ends, _ = sess.run([output_layer.cost, output_layer.predicted_starts, output_layer.predicted_ends, output_layer.train_op], feed)
                
                #printing target and predicted answers for comparison
                for i in range(args.batch_size):
                    target_indices = list(paragraphs[i, targets_start[i]:targets_end[i]+1])
                    target_string = " ".join(list(map(data_loader.integers_words.get, target_indices)))
                    predicted_indices = list(paragraphs[i, predicted_starts[i]:predicted_ends[i]+1])
                    predicted_string = " ".join(list(map(data_loader.integers_words.get, predicted_indices)))
                    try:
                        print(target_string, "|", predicted_string)
                    except:
                        print(target_indices, "|", predicted_indices)

                # instrument for tensorboard
                #summ, train_loss, state_fw, state_bw, _ = sess.run([summaries, model.cost, model.final_state_fw, model.final_state_bw, model.train_op], feed)
                #writer.add_summary(summ, e * data_loader.num_batches + b)

                end = time.time()
                print("{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}"
                      .format(e * data_loader.num_batches + b,
                              args.num_epochs * data_loader.num_batches,
                              e, train_loss, end - start))

                if (e * data_loader.num_batches + b) % args.save_every == 0\
                       or (e == args.num_epochs-1 and
                           b == data_loader.num_batches-1):
                    #save for the last result
                    checkpoint_path = os.path.join(args.save_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path,
                              global_step=e * data_loader.num_batches + b)
                    print("model saved to {}".format(checkpoint_path))


if __name__ == '__main__':
    main()
