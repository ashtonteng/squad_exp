import tensorflow as tf
import argparse
import time
import pickle
import numpy as np
import os
import json
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
    parser.add_argument('--save_dir', type=str, default='save',
                        help='directory to store checkpointed models')
    args = parser.parse_args()
    test(args)

def test(args):
    print("loading saved args...")
    with open(os.path.join(args.save_dir, 'config.pkl'), 'rb') as f:
        args = pickle.load(f)
    args.training = False
    args.batch_size = 1
    args.test_data_file = "train-v1.1.json"

    print("loading data...")
    data_loader = DataLoader(data_dir=args.data_dir, embedding_dim=args.glove_size, batch_size=args.batch_size, training=False, test_data_file=args.test_data_file)
    print("testing on", args.test_data_file)
    data_loader.create_batches()

    """Build Graph of Model"""
    embed_mtx = tf.get_variable("embed_mtx", shape=data_loader.embed_mtx.shape, trainable=True, initializer=tf.zeros_initializer)
    #GloveLayer will always directly get data from feed_dict.
    para_words_inputs_integers = tf.placeholder(tf.int32, [args.batch_size, None]) #allows for variable seq_length
    para_words_inputs_vectors = tf.nn.embedding_lookup(embed_mtx, para_words_inputs_integers) #batch_size x seq_length x rnn_size
    ques_words_inputs_integers = tf.placeholder(tf.int32, [args.batch_size, None]) #allows for variable seq_length
    ques_words_inputs_vectors = tf.nn.embedding_lookup(embed_mtx, ques_words_inputs_integers) #batch_size x seq_length x rnn_size

    paragraph_layer = BiRNNLayer(args, inputs=para_words_inputs_vectors, scope="paraBiRNN")
    question_layer = BiRNNLayer(args, inputs=ques_words_inputs_vectors, scope="quesBiRNN")
    attention_layer = AttentionLayer(args, paragraph_layer.outputs, question_layer.outputs, scope="attentionLayer")
    logits_layer = LogitsLayer(args, attention_layer.outputs, scope="logits")
    loss_layer = LossLayer(args, logits_layer.pred_start_dist, logits_layer.pred_end_dist, scope="lossLayer")
    #start_pointer_layer = PointerLayer(args, attention_layer.outputs, scope="startPointer")
    #end_pointer_layer = PointerLayer(args, attention_layer.outputs, scope="endPointer")
    #loss_layer = LossLayer(args, start_pointer_layer.pred_dist, end_pointer_layer.pred_dist, scope="lossLayer")
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(args.save_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print("restoring session", ckpt.model_checkpoint_path, "...")
            saver.restore(sess, ckpt.model_checkpoint_path)
            total_loss = 0
            predictions = dict()
            for b in range(data_loader.num_batches):
                qaIDs, max_para_length, max_ques_length, paragraphs_words, questions_words, paragraphs_chars, questions_chars, targets_start, targets_end = data_loader.next_batch_variable_seq_length()
                feed = {#char_paragraph_layer.max_seq_length: max_para_length,
                        #char_question_layer.max_seq_length: max_ques_length,
                        #char_paragraph_layer.inputs: paragraphs_chars,
                        #char_question_layer.inputs: questions_chars,
                        para_words_inputs_integers: paragraphs_words,
                        ques_words_inputs_integers: questions_words,
                        loss_layer.targets_start: targets_start, 
                        loss_layer.targets_end: targets_end}

                batch_mean_loss, pred_start_dist, pred_end_dist = sess.run([loss_layer.cost, loss_layer.pred_start_dist, loss_layer.pred_end_dist], feed)
                
                #write predictions to json file
                predicted_starts = np.argmax(pred_start_dist, axis=1)
                predicted_ends = np.argmax(pred_end_dist, axis=1)
                for i in range(args.batch_size):
                    qaID = qaIDs[i]
                    predicted_idx_span = list(paragraphs_words[i, predicted_starts[i]:predicted_ends[i]+1])
                    predicted_string = " ".join(list(map(data_loader.integers_words.get, predicted_idx_span)))
                    predictions[qaID] = predicted_string

                #printing target and predicted answers for comparison
                # if b % 30 == 0:
                #     predicted_starts = np.argmax(pred_start_dist, axis=1)
                #     predicted_ends = np.argmax(pred_end_dist, axis=1)
                #     for i in range(args.batch_size):
                #         target_indices = list(paragraphs[i, targets_start[i]:targets_end[i]+1])
                #         target_string = " ".join(list(map(data_loader.integers_words.get, target_indices)))
                #         predicted_indices = list(paragraphs[i, predicted_starts[i]:predicted_ends[i]+1])
                #         predicted_string = " ".join(list(map(data_loader.integers_words.get, predicted_indices)))
                #         try:
                #             print(target_string, "|", predicted_string)
                #         except:
                #             print(target_indices, "|", predicted_indices)

                if b % 50 == 0:
                    print("{}/{}, batch_mean_loss = {:.3f}".format(b, data_loader.num_batches, batch_mean_loss))
                total_loss += batch_mean_loss
            average_loss = total_loss / data_loader.num_batches
            print("done testing! average loss is", average_loss)
            print("writing predictions to json predictions.json...")
            with open("predictions.json", "w") as f:
                json.dump(predictions, f)
            print("testing complete!")
        else:
            raise Exception("unable to restore ckpt.model_checkpoint_path.")

if __name__ == '__main__':
    main()
