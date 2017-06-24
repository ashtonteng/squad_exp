import json
import re
import numpy as np
import os
import pickle
import collections

"""
SQuAD data structure:

data {
    data {
        paragraphs {
            qas {
                answers {
                    answer_start #starting position
                    text #actual answer
                }
                id
                question
            }
            context
        }
        title
    }
    version
}
"""
class DataLoader():
    """Preprocesses data, creates batches, and provides the next batch of data"""
    def __init__(self, data_dir, batch_size, encoding="utf-8", training=True):
        if training:
            self.train_or_test = "train"
        else:
            self.train_or_test  ="test"
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.encoding = encoding
        self.all_txt_file = os.path.join(data_dir, "all_text.txt")
        self.vocab_file = os.path.join(data_dir, "vocab.pkl")
        self.integers_words_file = os.path.join(data_dir, "integers_words.pkl")
        self.words_integers_file = os.path.join(data_dir, "words_integers.pkl")
        #self.train_data_dir = os.path.join(data_dir, "train")
        #self.test_data_dir = os.path.join(data_dir, "test")
        self.para_dict_file = os.path.join(data_dir, "para_dict.pkl")
        self.para_to_qa_dict_file = os.path.join(data_dir, "para_to_qa_dict.pkl")
        self.qa_data_dict_file = os.path.join(data_dir, "qa_data_dict.pkl")
        
        if training:
            if os.path.exists(self.para_dict_file) and os.path.exists(self.para_to_qa_dict_file) and os.path.exists(self.qa_data_dict_file):
                print("loading preprocessed data...") 
                self.para_dict = pickle.load(open(self.para_dict_file, "rb"))
                self.para_to_qa_dict = pickle.load(open(self.para_to_qa_dict_file, "rb"))
                self.qa_data_dict = pickle.load(open(self.qa_data_dict_file, "rb"))
            else:
                print("preprocessing data...")
                train_data = self.load_json_data(os.path.join(data_dir, "train-v1.1.json"))["data"]
                self.para_dict, self.para_to_qa_dict, self.qa_data_dict = self.parse_json_data(train_data)
            if not os.path.exists(self.vocab_file):
                print("generating vocab from training data...")
                self.vocab, self.words_integers, self.integers_words = generate_vocab()
            else:
                print("loading vocab file...")
                self.vocab = pickle.load(open(self.vocab_file, 'rb'))
                self.words_integers = pickle.load(open(self.words_integers_file, 'rb'))
                self.integers_words = pickle.load(open(self.integers_words_file, 'rb'))
            self.vocab_size = len(self.vocab)
        #else: #testing
            #test_data = self.load_json_data(os.path.join(data_dir, "test-v1.1.json"))
            #self.para_dict, self.para_to_qa_dict, self.qa_data_dict = self.parse_json_data(test_data)
            #if not os.path.exists(self.vocab_file):
                #raise RuntimeError("Could not find vocab file. Train first before testing!")
                
    def load_json_data(self, path):
        return json.load(open(path))
    
    def parse_json_data(self, data):
        #each paraID is a string composed of article title + number
        para_dict = dict() #{paraID: [list of words in paragraph]}
        para_to_qa_dict = dict() #{paraID: [list of qaIDs associated with paragraph]}
        qa_data_dict = dict() #{qaID: (paraID, list of words in question, answer_start_index, answer_end_index)}
        for article in data:
            #build para_dict
            for idx, paragraph in enumerate(article["paragraphs"]):
                paraID = article['title'] + "_" + str(idx).zfill(4) #reformat number to 4 digits
                words = [x for x in re.split('(\W)', paragraph["context"].strip().lower()) if x and x != " "]
                para_dict[paraID] = words
                for qa in paragraph["qas"]:
                    qaID = qa["id"]
                    try:
                        para_to_qa_dict[paraID].append(qaID)
                    except:
                        para_to_qa_dict[paraID] = [qaID]
                    q_words = [x for x in re.split('(\W)', qa["question"].strip().lower()) if x and x != " "]
                    start_index = qa["answers"][0]["answer_start"] #there are multiples answers. choose the first one.
                    end_index = start_index + len(qa["answers"][0]["text"]) - 1
                    qa_data_dict[qaID] = (paraID, q_words, start_index, end_index)
        pickle.dump(para_dict, open(self.para_dict_file, "wb"))
        pickle.dump(para_to_qa_dict, open(self.para_to_qa_dict_file, "wb"))
        pickle.dump(qa_data_dict, open(self.qa_data_dict_file, "wb"))
        return para_dict, para_to_qa_dict, qa_data_dict
    
    def map_chars_to_integers(self):
        pass
        
    def generate_vocab(self):
        """Takes all the text in the training data and assigns an integer to each word."""
        with open(self.all_txt_file, "r", encoding=self.encoding) as f:
            data = f.read().lower()
        data = [x for x in re.split('(\W)', data) if x and x != " "]
        counter = collections.Counter(data)
        count_pairs = sorted(counter.items(), key=lambda x: -x[1])
        words, _ = zip(*count_pairs)
        vocab = set(words)
        pickle.dump(vocab, open(self.vocab_file, 'wb'))
        words_integers = dict(zip(words, range(len(words)))) #{word:integer}
        integers_words = dict(zip(range(len(words)), words)) #{integer:word}
        pickle.dump(words_integers, open(self.words_integers_file, 'wb'))
        pickle.dump(integers_words, open(self.integers_words_file, 'wb'))
        return vocab, words_integers, integers_words
        #self.tensor = np.array(list(map(vocab.get, data)))
        #np.save(tensor_file, self.tensor)
        
    def map_words_to_integers(self, word_list, after_pad_length):
        """Uses the words_integers map to transform data to integers"""
        ret = np.zeros(after_pad_length, dtype=int)
        mapped = np.array(list(map(self.words_integers.get, word_list)))
        ret[:len(mapped)] = mapped
        return ret

    def preprocess_chars(self):
        pass
    
    def load_preprocessed(self):
        """Loads word-integer mapping of text"""
        pass
    
    def create_batches(self):
        """Groups qaIDs into groups of batch_size"""
        deque = collections.deque(self.qa_data_dict)
        num_batches = int(len(deque)/self.batch_size)
        all_batches = []
        for _ in range(num_batches):
            this_batch = [deque.pop() for _ in range(self.batch_size)]
            all_batches.append(this_batch)
        last_batch = list(this_batch) #copy second last batch
        last_batch[:len(deque)] = list(deque) #replace some spots with leftovers
        all_batches.append(last_batch)
        self.batch_deque = collections.deque(all_batches)
        self.num_batches = num_batches + 1 #added the incomplete last_batch

        self.max_para_length = len(self.para_dict[max(self.para_dict, key=lambda x: len(self.para_dict[x]))]) #length of longest paragraph
        self.max_ques_length = len(self.qa_data_dict[max(self.qa_data_dict, key=lambda qaID: len(self.qa_data_dict[qaID][1]))][1])

    def next_batch(self):
        #returns the next_batch of data, with fixed-length paragraphs and questions.
        qaIDs = self.batch_deque.pop()
        paragraphs = [] #batch_size x seq_length
        questions = []
        targets_start = [] #batch_size
        targets_end = [] #batch_size
        for qaID in qaIDs:
            paraID, question_words, answer_start, answer_end = self.qa_data_dict[qaID]
            paragraph_words = self.para_dict[paraID]
            paragraphs.append(paragraph_words)
            questions.append(question_words) #TODO, integrate question words!
            targets_start.append(answer_start)
            targets_end.append(answer_end)
        paragraphs_integer_array = np.zeros((self.batch_size, self.max_para_length), dtype=int)
        questions_integer_array = np.zeros((self.batch_size, self.max_ques_length), dtype=int)
        for i in range(self.batch_size):
            paragraphs_integer_array[i] = self.map_words_to_integers(paragraphs[i], self.max_para_length)
            questions_integer_array[i] = self.map_words_to_integers(questions[i], self.max_ques_length)
        return paragraphs_integer_array, questions_integer_array, np.array(targets_start), np.array(targets_end)

    def next_batch_variable_seq_length(self): #currently unused
        #para_dict = {paraID: [list of words in paragraph]}
        #qa_data_dict = {qaID: (paraID, list of words in question, answer_start_index, answer_end_index)}
        """Returns the next batch of data, split into inputs and targets.
            Maps input words to integers, and stores them in a numpy array."""
        qaIDs = self.batch_deque.pop()
        paragraphs = [] #batch_size x seq_length
        questions = []
        targets_start = [] #batch_size
        targets_end = [] #batch_size
        max_para_length = 0 #length of longest paragraph in this batch
        max_ques_length = 0 #length of longest question in this batch
        for qaID in qaIDs:
            paraID, question_words, answer_start, answer_end = self.qa_data_dict[qaID]
            paragraph_words = self.para_dict[paraID]
            max_para_length = max(max_para_length, len(paragraph_words))
            paragraphs.append(paragraph_words)
            max_ques_length = max(max_ques_length, len(question_words))
            questions.append(question_words) #TODO, integrate question words!
            targets_start.append(answer_start)
            targets_end.append(answer_end)
        paragraphs_integer_array = np.zeros((self.batch_size, max_para_length), dtype=int)
        questions_integer_array = np.zeros((self.batch_size, max_ques_length), dtype=int)
        for i in range(self.batch_size):
            paragraphs_integer_array[i] = self.map_words_to_integers(paragraphs[i], max_para_length)
            questions_integer_array[i] = self.map_words_to_integers(questions[i], max_ques_length)
        return paragraphs_integer_array, questions_integer_array, np.array(targets_start), np.array(targets_end)

def LoadGloveEmbedding(glove_dir, glove_dim):
    """Load GloVE embeddings into dictionary"""
    embedding = {}
    f = open(os.path.join(glove_dir, 'glove.6B.'+str(glove_dim)+'d.txt'), encoding="utf-8")
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embedding[word] = coefs
    f.close()
    return embedding

def GetGloveRepresentation(para_dict, embedding, dim):
    para_glove_dict = dict()
    for paraID in para_dict:
        words = para_dict[paraID]
        glove_words = []
        for word in words:
            try:
                glove_word = embedding[word]
            except: #this word is not in glove, replace with zeros
                glove_word = np.zeros((dim,), dtype="float32")
            glove_words.append(glove_word)
        para_glove_dict[paraID] = glove_words
    return para_glove_dict #{paraID: [list of words in paragraph with glove representation]}

