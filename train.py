import tensorflow as tf
import argparse
import time
import pickle
import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"

from utils import DataLoader
from BiRNNLayer import BiRNNLayer
from CharBiRNNLayer import CharBiRNNLayer
from ConvLayer import ConvLayer
from PointerLayer import PointerLayer
from AttentionLayer import AttentionLayer
from LogitsLayer import LogitsLayer
from LossLayer import LossLayer
from MatchingLayer import MatchingLayer
from SelfMatchingLayer import SelfMatchingLayer

def main():
    parser = argparse.ArgumentParser(
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_dir', type=str, default='data',
                        help='data directory')
    parser.add_argument('--save_dir', type=str, default='save',
                        help='directory to store checkpointed models')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='directory to store tensorboard logs')
    parser.add_argument('--glove_size', type=int, default=100,
                        help='size of glove embeddings')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='number of layers in the RNN')
    parser.add_argument('--model', type=str, default='gru',
                        help='rnn, gru, lstm, or nas')
    parser.add_argument('--batch_size', type=int, default=60,
                        help='minibatch size')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='number of epochs')
    parser.add_argument('--save_every', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--grad_clip', type=float, default=5.,
                        help='clip gradients at this value')
    parser.add_argument('--learning_rate', type=float, default=0.03,
                        help='learning rate')
    parser.add_argument('--reg_scaling_factor', type=float, default=1e-6,
                        help='l2 loss parameter')
    parser.add_argument('--decay_rate', type=float, default=0.999,
                        help='decay rate for rmsprop')
    parser.add_argument('--keep_prob', type=float, default=0.8,
                        help='probability of not dropping out weights')
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
    data_loader = DataLoader(data_dir=args.data_dir, embedding_dim=args.glove_size, batch_size=args.batch_size, training=True)
    data_loader.create_batches()
    args.vocab_size = data_loader.word_vocab_size
    args.char_vocab_size = data_loader.char_vocab_size
    args.max_word_length = data_loader.max_word_length
    #some additional args
    args.ConvLayer_size = args.glove_size
    args.CharBiRNN_size = args.glove_size
    args.BiRNNLayer_size = args.glove_size
    args.AttentionLayer_size = 2*args.glove_size #*2 is BiRNN
    args.MatchingLayer_size = args.glove_size*2 
    args.SelfMatchingLayer_size = args.glove_size*4
    args.LogitsLayer_size = args.AttentionLayer_size
    args.PointerLayer_size = args.glove_size*16
    args.training = True

    """check compatibility if training is continued from previously saved model"""
    if args.init_from is not None:
        print("continuing training from a previous session...")
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
        need_be_same = ["model", "batch_size", "num_layers", "glove_size", "word_vocab_size", "char_vocab_size", "BiRNNLayer_size", "AttentionLayer_size", "PointerLayer_size"]
        for checkme in need_be_same:
            assert vars(saved_model_args)[checkme]==vars(args)[checkme],"Command line argument and saved model disagree on '%s' "%checkme

    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)
    with open(os.path.join(args.save_dir, 'config.pkl'), 'wb') as f:
        pickle.dump(args, f)

    """Build Graph of Model"""
    embed_mtx = tf.get_variable("embed_mtx", shape=data_loader.embed_mtx.shape, trainable=True, initializer=tf.zeros_initializer)
    embed_mtx_placeholder = tf.placeholder(tf.float32, shape=data_loader.embed_mtx.shape)
    embed_init = embed_mtx.assign(embed_mtx_placeholder)
    #GloveLayer will always directly get data from feed_dict.
    para_words_inputs_integers = tf.placeholder(tf.int32, [args.batch_size, None]) #allows for variable seq_length
    para_words_inputs_vectors = tf.nn.embedding_lookup(embed_mtx, para_words_inputs_integers) #batch_size x seq_length x rnn_size
    ques_words_inputs_integers = tf.placeholder(tf.int32, [args.batch_size, None]) #allows for variable seq_length
    ques_words_inputs_vectors = tf.nn.embedding_lookup(embed_mtx, ques_words_inputs_integers) #batch_size x seq_length x rnn_size

    #paragraph_layer = ConvLayer(args, inputs=para_inputs_vectors, scope="paraConv")
    #question_layer = ConvLayer(args, inputs=ques_inputs_vectors, scope="quesConv")
    #char_paragraph_layer = CharBiRNNLayer(args, scope="charBiRNN")
    #char_question_layer = CharBiRNNLayer(args, scope="charBiRNN")
    #para_combo = tf.concat([char_paragraph_layer.outputs, para_words_inputs_vectors.outputs], axis=-1)
    #ques_combo = tf.concat([char_question_layer.outputs, ques_words_inputs_vectors.outputs], axis=-1)

    paragraph_layer = BiRNNLayer(args, inputs=para_words_inputs_vectors, scope="paraBiRNN")
    question_layer = BiRNNLayer(args, inputs=ques_words_inputs_vectors, scope="quesBiRNN")


    #attention_layer = AttentionLayer(args, para_combo, ques_combo, scope="attentionLayer")
    #matching_layer = MatchingLayer(args, paragraph_layer.outputs, question_layer.outputs, scope="matchingLayer")
    #self_matching_layer = SelfMatchingLayer(args, matching_layer.outputs, scope="selfmatchingLayer")
    attention_layer = AttentionLayer(args, paragraph_layer.outputs, question_layer.outputs, scope="attentionLayer")
    logits_layer = LogitsLayer(args, attention_layer.outputs, scope="logits")
    loss_layer = LossLayer(args, logits_layer.pred_start_logits, logits_layer.pred_end_logits, scope="lossLayer")
    #start_pointer_layer = PointerLayer(args, self_matching_layer.outputs, scope="startPointer")
    #end_pointer_layer = PointerLayer(args, self_matching_layer.outputs, scope="endPointer")
    #loss_layer = LossLayer(args, start_pointer_layer.pred_dist, end_pointer_layer.pred_dist, scope="lossLayer")
    

    from tensorflow.python import debug as tf_debug
    """Run Data through Graph"""
    with tf.Session() as sess:
        #tensorflow debugger
        #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        #sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

        # instrument for tensorboard
        summaries = tf.summary.merge_all()
        writer = tf.summary.FileWriter(os.path.join(args.log_dir, time.strftime("%Y-%m-%d-%H-%M-%S")))
        writer.add_graph(sess.graph)

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())
        # restore model
        if args.init_from is not None:
            #glove embedding is already partially learned, load from saver
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            #if this is the first time training, feed glove embedding into graph
            sess.run(embed_init, feed_dict={embed_mtx_placeholder: data_loader.embed_mtx})

        for e in range(args.num_epochs):
            sess.run(tf.assign(loss_layer.learning_rate, args.learning_rate * (args.decay_rate ** e)))
            for b in range(data_loader.num_batches):
                start = time.time()
                qaIDs, max_para_length, max_ques_length, paragraphs_words, questions_words, paragraphs_chars, questions_chars, targets_start, targets_end = data_loader.next_batch_variable_seq_length()
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
                feed = {#char_paragraph_layer.max_seq_length: max_para_length,
                        #char_question_layer.max_seq_length: max_ques_length,
                        #char_paragraph_layer.inputs: paragraphs_chars,
                        #char_question_layer.inputs: questions_chars,
                        para_words_inputs_integers: paragraphs_words,
                        ques_words_inputs_integers: questions_words,
                        loss_layer.targets_start: targets_start, 
                        loss_layer.targets_end: targets_end}

                
                summ, train_loss, lossL2, pred_start_logits, pred_end_logits, _ = sess.run([summaries, loss_layer.cost, loss_layer.lossL2, loss_layer.pred_start_logits, loss_layer.pred_end_logits, loss_layer.train_op], feed)
                #print(tf.shape(testtest))
                #write summaries to tensorboard
                writer.add_summary(summ, e * data_loader.num_batches + b)

                #printing target and predicted answers for comparison
                if (e * data_loader.num_batches + b) % args.save_every == 0\
                       or (e == args.num_epochs-1 and
                           b == data_loader.num_batches-1):
                    predicted_starts = np.argmax(pred_start_logits, axis=1)
                    predicted_ends = np.argmax(pred_end_logits, axis=1)
                    for i in range(args.batch_size):
                        target_indices = list(paragraphs_words[i, targets_start[i]:targets_end[i]+1])
                        target_string = " ".join(list(map(data_loader.integers_words.get, target_indices)))
                        predicted_indices = list(paragraphs_words[i, predicted_starts[i]:predicted_ends[i]+1])
                        predicted_string = " ".join(list(map(data_loader.integers_words.get, predicted_indices)))
                        try:
                            print(target_string, "|", predicted_string)
                        except:
                            print(target_indices, "|", predicted_indices)
                
                #print to console
                end = time.time()
                print("{}/{} (epoch {}), cost = {:.3f}, lossL2 = {:.3f}, time/batch = {:.3f}"
                      .format(e * data_loader.num_batches + b,
                              args.num_epochs * data_loader.num_batches,
                              e, train_loss, lossL2, end - start))

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
