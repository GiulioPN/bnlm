# This preprocessing is intended for english
# coding: utf-8

import os
import sys
import re
import string
from nltk.corpus import stopwords
import spacy
from nltk.tokenize import sent_tokenize
from collections import Counter



def sentences(list_):
    """Returns sentence tokenized text list"""
    text = ''.join(list_)

    # Sentence tokenize with help of sent_tokenize from nltk  
    sentence = sent_tokenize(text)

    return sentence


def remove(text):
    """Returns text with all the filtering necessary"""
    t = re.sub(r"(\d+\.\d+)","",text)
    #t = re.sub(r"(\d+th?|st?|nd?|rd?)","", t)
    t = re.sub(r"\d{2}.\d{2}.\d{4}","",t)
    t = re.sub(r"\d{2}\/\d{2}\/\d{4}","",t)
    t = re.sub(r"\d{2}(\/|\.)\d{2}(\/|\.)\d{2}","",t)
    t = re.sub(r"($|€|¥|₹|£)","",t)
    t = re.sub(r"(%)","",t)
    t = re.sub(r"\d+","",t)
    t = re.sub(r"\n","",t)
    t = re.sub(r"\xa0", "", t)
    return t

def pun(text):
    """Return punctuations from text"""
    table = str.maketrans("","", string.punctuation)
    t = text.translate(table)
    return t

nlp = spacy.load('en')

def lemmatizer(text):
    """Returns text after lemmatization"""
    sent = []
    doc = nlp(text)
    for word in doc:
        sent.append(word.lemma_)
    return " ".join(sent)


def extras(sentences):
    """Returns text after removing some extra symbols"""
    t = re.sub(r"\"|\—|\'|\’","",sentences)
    word_list = t.split()
    for index, word in enumerate(word_list):
        if len(word) <=1:
            del word_list[index]
    t = ' '.join(word_list)

    return t

# Stop word removal
def stop_word(data_name):
    list_ = []
    stop_words = stopwords.words('english')
    words_list = sentence.split()
    for word in words_list:
        if word not in stop_words:
            list_.append(word)
    return ' '.join(list_)

def less_common_words_by_freq(voc_count, min_freq):
    """Returns less common words form counter voc_count variable"""
    less_common_words = []
    for key in voc_count:
        if voc_count[key]<= min_freq:
            less_common_words.append(key)
    less_common_words.sort()
    return less_common_words

def buildVocabulary(file_name):

    raw_data_path = "../data/"+data_name+".txt"
    print("The raw data '"+raw_data_path+"' will be opened.")

    with open(raw_data_path, 'r') as file:
        data = file.read().replace('\n', '')

    voc_list = list(Counter(data.split()).keys())
    voc_list.sort()

    # build up dictionary (key = index, value = word )
    voc = dict(enumerate(voc_list))

    # remove form voc less common words (freq <= 3)
    voc_counter = Counter(data.split())
    less_common_words = less_common_words_by_freq(voc_counter, 3)

    # Remove less common word form vocabulary and data
    for key_rm in less_common_words:
        del voc_counter[key_rm]

    # build up dictionary (key = index, value = word )
    voc_list2 = list(voc_counter.keys())
    voc_list2.sort()
    voc = dict(enumerate(voc_list2))

    inv_voc = {v: k for k, v in voc.items()}

    #save vocabulary
    with open('voc.txt', 'w') as f:
        for key in inv_voc.keys():
            f.write("%s\n" % key)
    return inv_voc

def sentence2num(inv_voc):
    # build data of sentence
    data_ls = []
    with open('../data/brown-postprocess.txt', 'r') as f:
        for line in f:
            data_ls.append(line.replace('\n', ''))

    # build data of nummber sequence
    data_num_seq_ls = []
    for sentence in data_ls:
        num_seq = []
        for word in sentence.split():
            if word in inv_voc.keys():
                num_seq.append(inv_voc[word])
        data_num_seq_ls.append(num_seq)


    # save sentences
    with open('num_sentences.txt', 'w') as f:
        for num_sentence in data_num_seq_ls:
            for num in num_sentence:
                f.write("%s " % num)
            f.write("\n")

def main(data_name):
    """Main Function
        param data_name a string with the name of raw data    
    """
    raw_data_path = "rawData/"+data_name+".txt"
    print("The raw data '"+raw_data_path+"' will be opened.")

    with open(raw_data_path, 'r') as f:
        contents = f.readlines()
    # Joining the lines to make text block 
    contents = ''.join(contents)
    print("Start preprocessing procedure in 5 steps.")
    # Sentence tokenize the text
    print("\t 1. Text tokenization...")
    sent_tokenized = sentences(contents)
    # Lemmatization 
    print("\t 2. Lemmatization...")
    t1 = [lemmatizer(sent) for sent in sent_tokenized]
    # Removing stop words
    print("\t 3. Removing stop words...")
    t2 = [stop_word(sent) for sent in t1]
    # Removing all the unnecessary things from the text 
    print("\t 4. Removing unnecessary characters ...")
    t3 = [remove(line) for line in t2]
    # Removing punctuations
    print("\t 5. Removing punctuations ...")
    t4 =[pun(line.lower()) for line in t3]

    t5 = [extras(sent) for sent in t4]
    print("Preprocessing done for the file.")
    
    post_processing_data = "../data/"+data_name+"-postprocess.txt"
    with open(post_processing_data, 'w') as f:
        for item in t5:
            f.write("%s\n" % item)

    print("The postprocessing data '"+post_processing_data+"' has been saved.")

if __name__ == "__main__":
    print("Put the raw data in 'rawData' folder.")
    data_name = "sotu_train2"
    main(data_name)