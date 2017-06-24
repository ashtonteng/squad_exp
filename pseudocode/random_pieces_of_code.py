"""Embedding lookup for static variables"""

W = tf.Variable(tf.constant(0.0, shape=[vocab_size, embedding_dim]),
                trainable=False, name="W")

embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, embedding_dim])
embedding_init = W.assign(embedding_placeholder)

# ...
sess = tf.Session()

sess.run(embedding_init, feed_dict={embedding_placeholder: embedding})

"""Vocab lookup: char --> integer"""
import codecs
import numpy as np
from six.moves import cPickle
mapping_input_file = "input.txt"
vocab_file = "vocab.pkl"
with codecs.open(mapping_input_file, "r", encoding="utf-8") as f:
    data = f.read().split(" ")
with open(vocab_file, 'rb') as f:
    chars = cPickle.load(f)
    vocab_size = len(chars)
vocab = dict(zip(chars, range(len(chars)))) #{char: integer}
tensor = np.array(list(map(vocab.get, data)))


"""Get all words in passages and questions for Char-RNN"""
with open("all_passages.txt", "w", encoding="utf-8") as file:
    for article in data:
        for paragraph in article["paragraphs"]:
            file.write(paragraph["context"] + " ")

with open("all_questions.txt", "w", encoding="utf-8") as file:
    for article in data:
        for paragraph in article["paragraphs"]:
            for qa in paragraph["qas"]:
                file.write(qa["question"] + " ")

with open("all_text.txt", "w", encoding="utf-8") as file:
    for article in data:
        for paragraph in article["paragraphs"]:
            file.write(paragraph["context"] + " ")
            for qa in paragraph["qas"]:
                file.write(qa["question"] + " ")

def find_sim(p_batch, q_batch):
    finds the similarity between the context word p and question word q, for an entire batch.
       p_batch: batch_size x hidden_size.
       q_batch: batch_size x hidden_size.
       returns: batch_size x 1
    return tf.matmul(tf.concat([p_batch, q_batch, tf.multiply(p_batch, q_batch)], axis=1), w_sim)

temp1 = []
for t in range(p_length):
    print(t)
    temp2 = []
    for j in range(q_length):
        print(j)
        p_batch = tf.squeeze(tf.slice(p_inputs, [0, t, 0], [batch_size, 1, hidden_size]), axis=1) #takes the t'th passage word data for entire batch.
        q_batch = tf.squeeze(tf.slice(q_inputs, [0, j, 0], [batch_size, 1, hidden_size]), axis=1) #batch_size x hidden_size
        temp2.append(find_sim(p_batch, q_batch)) #batch_size x 1
    temp1.append(tf.stack(temp2)) #each element is batch_size x q_length
sim_mtx = tf.stack(temp1) #tf.constant(np.empty((p_length, q_length, batch_size))) #S
print(sim_mtx.get_shape())