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