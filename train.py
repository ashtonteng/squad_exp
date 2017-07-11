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
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='directory to store tensorboard logs')
    parser.add_argument('--rnn_size', type=int, default=200,
                        help='size of RNN hidden state')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='number of layers in the RNN')
    parser.add_argument('--model', type=str, default='lstm',
                        help='rnn, gru, lstm, or nas')
    parser.add_argument('--batch_size', type=int, default=50,
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
    data_loader = DataLoader(args.data_dir, embedding_dim=args.rnn_size, batch_size=args.batch_size)
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
    embed_mtx = tf.Variable(tf.constant(0.0, shape=data_loader.embed_mtx.shape), trainable=True, name="embed_mtx")
    embed_mtx_placeholder = tf.placeholder(tf.float32, shape=data_loader.embed_mtx.shape)
    embed_init = embed_mtx.assign(embed_mtx_placeholder)
    #GloveLayer will always directly get data from feed_dict.
    para_inputs_integers = tf.placeholder(tf.int32, [args.batch_size, None]) #allows for variable seq_length
    para_inputs_vectors = tf.nn.embedding_lookup(embed_mtx, para_inputs_integers) #batch_size x seq_length x rnn_size
    ques_inputs_integers = tf.placeholder(tf.int32, [args.batch_size, None]) #allows for variable seq_length
    ques_inputs_vectors = tf.nn.embedding_lookup(embed_mtx, ques_inputs_integers) #batch_size x seq_length x rnn_size

    paragraph_layer = BiRNNLayer(args.vocab_size, inputs=para_inputs_vectors, batch_size=args.batch_size, rnn_size=args.rnn_size, num_layers=1, scope="paraBiRNN")
    question_layer = BiRNNLayer(args.vocab_size, inputs=ques_inputs_vectors, batch_size=args.batch_size, rnn_size=args.rnn_size, num_layers=1, scope="quesBiRNN")
    attention_layer = AttentionLayer(paragraph_layer.outputs, question_layer.outputs, batch_size=args.batch_size, rnn_size=2*args.rnn_size, scope="attentionLayer")
    logits_layer = LogitsLayer(attention_layer.outputs, batch_size=args.batch_size, scope="logits")
    loss_layer = LossLayer(logits_layer.pred_start_dist, logits_layer.pred_end_dist, batch_size=args.batch_size, scope="lossLayer")
    #start_pointer_layer = PointerLayer(attention_layer.outputs, batch_size=args.batch_size, hidden_size=8*args.rnn_size, scope="startPointer")
    #end_pointer_layer = PointerLayer(attention_layer.outputs, batch_size=args.batch_size, hidden_size=8*args.rnn_size, scope="endPointer")
    #loss_layer = LossLayer(start_pointer_layer.pred_dist, end_pointer_layer.pred_dist, batch_size=args.batch_size, scope="lossLayer")

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
        sess.run(embed_init, feed_dict={embed_mtx_placeholder: data_loader.embed_mtx})

        for e in range(args.num_epochs):
            sess.run(tf.assign(loss_layer.learning_rate, args.learning_rate * (args.decay_rate ** e)))
            for b in range(data_loader.num_batches):
                start = time.time()
                paragraphs, questions, targets_start, targets_end = data_loader.next_batch()
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
                #targets_start = [2]*args.batch_size
                #targets_end = [3]*args.batch_size
                feed = {para_inputs_integers: paragraphs,
                        ques_inputs_integers: questions,
                        loss_layer.targets_start: targets_start, 
                        loss_layer.targets_end: targets_end}

                train_loss, pred_start_dist, pred_end_dist, _ = sess.run([loss_layer.cost, loss_layer.pred_start_dist, loss_layer.pred_end_dist, loss_layer.train_op], feed)
                #printing target and predicted answers for comparison
                if b % 30 == 0:
                    predicted_starts = np.argmax(pred_start_dist, axis=1)
                    predicted_ends = np.argmax(pred_end_dist, axis=1)
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
