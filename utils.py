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
    def __init__(self, data_dir, embedding_dim, batch_size, training, train_data_file="train-v1.1.json", test_data_file="dev-v1.1.json", encoding="utf-8"):
        self.data_dir = data_dir
        self.squad_dir = squad_dir = os.path.join(data_dir, "squad")
        self.glove_dir = glove_dir = os.path.join(data_dir, "glove")
        if not os.path.isdir(glove_dir):
            os.mkdir(glove_dir)
            raise Exception("GloVE vectors are missing! Please download from website and put in data/glove.")
        self.pickle_dir = pickle_dir = os.path.join(data_dir, "pickle")
        if not os.path.isdir(pickle_dir):
            os.mkdir(pickle_dir)
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.encoding = encoding
        self.all_txt_file = os.path.join(squad_dir, "all_text.txt")
        #self.word_vocab_file = os.path.join(pickle_dir, "word_vocab.pkl")
        self.char_vocab_file = os.path.join(pickle_dir, "char_vocab.pkl")
        self.integers_words_file = os.path.join(pickle_dir, "integers_words.pkl")
        self.words_integers_file = os.path.join(pickle_dir, "words_integers.pkl")
        self.chars_integers_file = os.path.join(pickle_dir, "chars_integers.pkl")
        self.embed_mtx_file = os.path.join(pickle_dir, "embed_mtx_"+str(embedding_dim)+".pkl")

        if os.path.exists(self.integers_words_file) and os.path.exists(self.words_integers_file) and os.path.exists(self.chars_integers_file) and os.path.exists(self.embed_mtx_file):
            print("loading vocab files...")
            #self.word_vocab = pickle.load(open(self.word_vocab_file, 'rb'))
            self.words_integers = pickle.load(open(self.words_integers_file, 'rb'))
            self.chars_integers = pickle.load(open(self.chars_integers_file, 'rb'))
            self.integers_words = pickle.load(open(self.integers_words_file, 'rb'))
            self.embed_mtx = pickle.load(open(self.embed_mtx_file, 'rb'))
        else:
            if not training:
                raise Exception("vocab files are missing, train first before testing!")
            print("generating vocab from training data and glove...")
            self.words_integers, self.integers_words, self.embed_mtx = self.generate_word_vocab_and_embedding_mtx()
            self.chars_integers = self.generate_char_vocab()
        self.word_vocab_size = self.embed_mtx.shape[0]
        self.char_vocab_size = len(self.chars_integers)
        self.oov_integer = self.words_integers["<OOV>"]
        self.char_oov_integer = self.chars_integers["<OOV>"]
        self.max_word_length = max([len(word) for word in self.words_integers.keys()])

        if training:
            data = self.load_json_data(os.path.join(self.squad_dir, train_data_file))["data"]
            self.para_dict_file = os.path.join(pickle_dir, "para_dict_train.pkl")
            self.para_to_qa_dict_file = os.path.join(pickle_dir, "para_to_qa_dict_train.pkl")
            self.qa_data_dict_file = os.path.join(pickle_dir, "qa_data_dict_train.pkl")

        else: #testing
            data = self.load_json_data(os.path.join(self.squad_dir, test_data_file))["data"]
            self.para_dict_file = os.path.join(pickle_dir, "para_dict_dev.pkl")
            self.para_to_qa_dict_file = os.path.join(pickle_dir, "para_to_qa_dict_dev.pkl")
            self.qa_data_dict_file = os.path.join(pickle_dir, "qa_data_dict_dev.pkl")


        if os.path.exists(self.para_dict_file) and os.path.exists(self.para_to_qa_dict_file) and os.path.exists(self.qa_data_dict_file):
            print("loading preprocessed data...") 
            self.para_dict = pickle.load(open(self.para_dict_file, "rb"))
            self.para_to_qa_dict = pickle.load(open(self.para_to_qa_dict_file, "rb"))
            self.qa_data_dict = pickle.load(open(self.qa_data_dict_file, "rb"))
        else:
            print("preprocessing data...")
            self.para_dict, self.para_to_qa_dict, self.qa_data_dict = self.parse_json_data(data)
                
    def load_json_data(self, path):
        return json.load(open(path))
    
    def parse_json_data(self, data):
        #each paraID is a string composed of article title + number
        para_dict = dict() #{paraID: [list of words in paragraph]}
        para_to_qa_dict = dict() #{paraID: [list of qaIDs associated with paragraph]}
        qa_data_dict = dict() #{qaID: (paraID, list of words in question, answer_start_index, answer_end_index)}

        import re
        import difflib
        import inflect
        p = inflect.engine()

        def find_exact_sub_list(sublst, lst):
            """Given a list and a sublist, find the starting and ending indices of the sublist in the list."""
            for index in (i for i, e in enumerate(lst) if e == sublst[0]): #if list_word == sublist[0]
                if lst[index:index+len(sublst)] == sublst:
                    return index, index + len(sublst) - 1
            raise ValueError("no exact match found between sublist and list!")
            
        def find_sub_list(sublst, lst):
            try:
                return find_exact_sub_list(sublst, lst)
            except:
                pass
            if len(sublst) > 1:
                try:
                    return find_sub_list(sublst[1:], lst)
                except:
                    pass
                try:
                    return find_sub_list(sublst[1:], lst)
                except:
                    pass
            elif len(sublst) == 1:
                word = sublst[0]
                if bool(re.search(r'\d', word)): #if word contains a number
                    try:
                        number_word = p.number_to_words(word)
                        return find_sub_list([number_word], lst)
                    except:
                        pass
                try: #find closest word in paragraph and choose that
                    idx = np.argmax([difflib.SequenceMatcher(None, word, x).ratio() for x in lst])
                    return (idx, idx)
                    #similarities = [difflib.SequenceMatcher(None, word, x).ratio() for x in lst]
                    #if max(similarities) > 0.4:
                    #    idx = np.argmax(similarities)
                    #    return (idx, idx)
                except:
                    pass
            raise ValueError("approximate match between sublist and list not found!")

        for article in data:
            #build para_dict
            for idx, paragraph in enumerate(article["paragraphs"]):
                paraID = article['title'] + "_" + str(idx).zfill(4) #reformat number to 4 digits
                para_words = [x for x in re.split('(\W)', paragraph["context"].strip().lower()) if x and x != " "]
                para_dict[paraID] = para_words
                for qa in paragraph["qas"]:
                    qaID = qa["id"]
                    try:
                        para_to_qa_dict[paraID].append(qaID)
                    except:
                        para_to_qa_dict[paraID] = [qaID]
                    q_words = [x for x in re.split('(\W)', qa["question"].strip().lower()) if x and x != " "]
                    a_words = [x for x in re.split('(\W)', qa["answers"][0]["text"].strip().lower()) if x and x != " "] #there are multiples answers. choose the first one.
                    try:
                        start_index, end_index = find_sub_list(a_words, para_words)
                        qa_data_dict[qaID] = (paraID, q_words, start_index, end_index)
                    except ValueError:
                        print("question", qaID, "discarded.")
                   # start_index_char = qa["answers"][0]["answer_start"] 
                   # end_index_char = start_index + len(qa["answers"][0]["text"]) - 1
        pickle.dump(para_dict, open(self.para_dict_file, "wb"))
        pickle.dump(para_to_qa_dict, open(self.para_to_qa_dict_file, "wb"))
        pickle.dump(qa_data_dict, open(self.qa_data_dict_file, "wb"))
        return para_dict, para_to_qa_dict, qa_data_dict

    def get_glove_embeddings(self):
        """Load GloVE embeddings into dictionary"""
        glove_embedding = {}
        f = open(os.path.join(self.glove_dir, 'glove.6B.'+str(self.embedding_dim)+'d.txt'), encoding="utf-8")
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            glove_embedding[word] = coefs
        f.close()
        return glove_embedding

    def generate_char_vocab(self): #char_embedding_mtx is learned during training
        with open(self.all_txt_file, "r", encoding=self.encoding) as f:
            data = f.read()
        counter = collections.Counter(data)
        count_pairs = sorted(counter.items(), key=lambda x: -x[1])
        chars, _ = zip(*count_pairs)
        chars_integers = dict(zip(chars, range(len(chars))))
        chars_integers["<OOV>"] = len(chars_integers)
        with open(self.chars_integers_file, 'wb') as f:
            pickle.dump(chars_integers, f)
        return chars_integers

    def generate_word_vocab_and_embedding_mtx(self):
        """Takes all the text in the training data and glove and assigns an integer to each word."""
        with open(self.all_txt_file, "r", encoding=self.encoding) as f:
            data = f.read().lower()
        words = [x for x in re.split('(\W)', data) if x and x != " "]
        counter = collections.Counter(words)
        count_pairs = sorted(counter.items(), key=lambda x: -x[1])
        words, _ = zip(*count_pairs)
        train_vocab_size = len(words)
        words_integers = dict(zip(words, range(1, train_vocab_size+1))) #{word:integer}
        integers_words = dict(zip(range(1, train_vocab_size+1), words)) #{integer:word}
        pad_integer = 0
        words_integers["<PAD>"] = pad_integer
        integers_words[pad_integer] = "<PAD>"

        embeddings = []
        glove_embeddings = self.get_glove_embeddings()
        max_integer = train_vocab_size #padding integer is 0. OOV integer is max_integer+1
        for integer in range(train_vocab_size+1): #all train_vocab plus <PAD>
            word = integers_words[integer]
            if word in glove_embeddings: #if word in glove, initialize to glove embedding
                embeddings.append(glove_embeddings[word])
            else: #if word not in glove, initialize to random embedding [0.0, 1.0]
                embeddings.append(numpy.random.uniform(low=-1.0, high=1.0, size=self.embedding_dim)) 
        for glove_word in glove_embeddings:
            if glove_word not in words_integers: #also add all glove_words not used in training to embedding matrix
                new_integer = max_integer = max_integer + 1
                words_integers[glove_word] = new_integer
                integers_words[new_integer] = glove_word
                embeddings.append(glove_embeddings[glove_word])
        
        #assign an oov integer with random embedding. These words are not in glove or training data.
        oov_integer = max_integer + 1
        words_integers["<OOV>"] = oov_integer
        integers_words[oov_integer] = "<OOV>"
        embeddings.append(numpy.random.uniform(low=-1.0, high=1.0, size=self.embedding_dim)) 
        
        embed_mtx = np.asarray(embeddings)
        #vocab = set(words_integers.keys())
        #pickle.dump(vocab, open(self.word_vocab_file, 'wb'))
        pickle.dump(words_integers, open(self.words_integers_file, 'wb'))
        pickle.dump(integers_words, open(self.integers_words_file, 'wb'))
        pickle.dump(embed_mtx, open(self.embed_mtx_file, 'wb'))
        return words_integers, integers_words, embed_mtx

    def generate_vocab(self):
        #Deprecated, does not use glove
        """Takes all the text in the training data and assigns an integer to each word."""
        with open(self.all_txt_file, "r", encoding=self.encoding) as f:
            data = f.read().lower()
        data = [x for x in re.split('(\W)', data) if x and x != " "]
        counter = collections.Counter(data)
        count_pairs = sorted(counter.items(), key=lambda x: -x[1])
        words, _ = zip(*count_pairs)
        #vocab = set(words)
        #pickle.dump(vocab, open(self.word_vocab_file, 'wb'))
        words_integers = dict(zip(words, range(len(words)))) #{word:integer}
        integers_words = dict(zip(range(len(words)), words)) #{integer:word}
        pickle.dump(words_integers, open(self.words_integers_file, 'wb'))
        pickle.dump(integers_words, open(self.integers_words_file, 'wb'))
        return vocab, words_integers, integers_words
        
    def map_words_to_integers(self, word_list, after_pad_length):
        """Uses the words_integers map to transform data to integers with padding"""
        #if word is not included in words_integers map, replace with OOV integer.
        ret = np.zeros(after_pad_length, dtype=int)
        mapped = list(map(self.words_integers.get, word_list))
        mapped_replace_None_with_OOV = np.array([self.oov_integer if i is None else i for i in mapped])
        ret[:len(mapped_replace_None_with_OOV)] = mapped_replace_None_with_OOV
        return ret

    def map_chars_to_integers(self, char_list, max_word_length):
        ret = np.zeros(max_word_length, dtype=int)
        mapped = list(map(self.chars_integers.get, char_list))
        mapped_replace_None_with_OOV = np.array([self.oov_integer if i is None else i for i in mapped])
        ret[:len(mapped_replace_None_with_OOV)] = mapped_replace_None_with_OOV
        return ret

    def preprocess_chars(self):
        pass
    
    def load_preprocessed(self):
        """Loads word-integer mapping of text"""
        pass

    # def create_bins(self):
    #     import matplotlib.pyplot as plt
    #     passage_lengths = [len(p) for p in self.para_dict.values()]
    #     self.num_bins = 50
    #     def histedges_equalN(x, nbin):
    #         npt = len(x)
    #         return np.interp(np.linspace(0, npt, nbin + 1), np.arange(npt), np.sort(x))
    #     n, bins, patches = plt.hist(para_lengths, histedges_equalN(passage_lengths, self.num_bins))
    #     bins = bins.astype("int")

    
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
        
    def next_batch(self):
        #para_dict = {paraID: [list of words in paragraph]}
        #qa_data_dict = {qaID: (paraID, list of words in question, answer_start_index, answer_end_index)}
        """Returns the next batch of data, split into inputs and targets.
            Maps input words to integers, and stores them in a numpy array.
            This method pads paragraphs to the longest paragraph in the entire dataset."""
        #self.max_para_length = len(self.para_dict[max(self.para_dict, key=lambda x: len(self.para_dict[x]))]) #length of longest paragraph
        #self.max_ques_length = len(self.qa_data_dict[max(self.qa_data_dict, key=lambda qaID: len(self.qa_data_dict[qaID][1]))][1])
        try:
            qaIDs = self.batch_deque.pop()
        except:
            self.create_batches()
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
        para_words_int_array = np.zeros((self.batch_size, self.max_para_length), dtype=int)
        ques_words_int_array = np.zeros((self.batch_size, self.max_ques_length), dtype=int)
        para_chars_int_array = np.zeros((self.batch_size, self.max_para_length, self.max_word_length), dtype=int)
        ques_chars_int_array = np.zeros((self.batch_size, self.max_ques_length, self.max_word_length), dtype=int)
        for i in range(self.batch_size):
            para_words_int_array[i] = self.map_words_to_integers(paragraphs[i], self.max_para_length)
            ques_words_int_array[i] = self.map_words_to_integers(questions[i], self.max_ques_length)
            for j in range(self.max_para_length):
                para_chars_int_array[i][j] = self.map_chars_to_integers(paragraphs[i][j], self.max_word_length)
            for k in range(self.max_ques_length):
                para_chars_int_array[i][k] = self.map_chars_to_integers(questions[i][k], self.max_word_length)
        return qaIDs, para_words_int_array, ques_words_int_array, para_chars_int_array, ques_chars_int_array, np.array(targets_start), np.array(targets_end)

    def next_batch_variable_seq_length(self):
        #para_dict = {paraID: [list of words in paragraph]}
        #qa_data_dict = {qaID: (paraID, list of words in question, answer_start_index, answer_end_index)}
        """Returns the next batch of data, split into inputs and targets.
            Maps input words to integers, and stores them in a numpy array.
            This method pads paragraphs to the longest paragraph in this batch."""
        try:
            qaIDs = self.batch_deque.pop()
        except:
            self.create_batches()
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
        para_words_int_array = np.zeros((self.batch_size, max_para_length), dtype=int)
        ques_words_int_array = np.zeros((self.batch_size, max_ques_length), dtype=int)
        para_chars_int_array = np.zeros((self.batch_size, max_para_length, self.max_word_length), dtype=int)
        ques_chars_int_array = np.zeros((self.batch_size, max_ques_length, self.max_word_length), dtype=int)
        for i in range(self.batch_size):
            para_words_int_array[i] = self.map_words_to_integers(paragraphs[i], max_para_length)
            ques_words_int_array[i] = self.map_words_to_integers(questions[i], max_ques_length)
            for j in range(len(paragraphs[i])):
                para_chars_int_array[i][j] = self.map_chars_to_integers(paragraphs[i][j], self.max_word_length)
            for k in range(len(questions[i])):
                ques_chars_int_array[i][k] = self.map_chars_to_integers(questions[i][k], self.max_word_length)
        return qaIDs, max_para_length, max_ques_length, para_words_int_array, ques_words_int_array, para_chars_int_array, ques_chars_int_array, np.array(targets_start), np.array(targets_end)

def GetGloveRepresentation(inputs, vocab, embeddings, dim):
    from tensorflow.contrib import learn
    #init vocab processor
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    #fit the vocab from glove
    pretrain = vocab_processor.fit(vocab)
    #transform inputs
    x = np.array(list(vocab_processor.transform(inputs)))


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

