import json
import re
import numpy as np
import os

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

def LoadJsonData(filePath):
    with open(filePath) as dataFile:
        data = json.load(dataFile)
    return data

def ParseJsonData(data):
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
    return para_dict, para_to_qa_dict, qa_data_dict

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

