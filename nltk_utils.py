import nltk
#nltk.download('punkt')
#from nltk.stem.porter import PorterSetmmer
import numpy as np
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()


def tokenize(sentence):                     #to break sentence into tokens
    return nltk.word_tokenize(sentence)

def stem(word):                             #lower + stem 
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
    tokenized_sentence = [stem(w) for w in tokenized_sentence]

    bag = np.zeros(len(all_words), dtype = np.float32)

    for idx,w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0
    return bag


