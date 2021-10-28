import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import numpy as np
# nltk.download('punkt')

stemmer = StemmerFactory().create_stemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem_indo(kalimat):
    return stemmer.stem(kalimat.lower())

def bag_of_word(tokenized_sentenced, all_word):
    
    tokenized_sentence = [stem_indo(w) for w in tokenized_sentenced]

    bag = np.zeros(len(all_word), dtype=np.float)
    for index, word in enumerate(all_word):
        if word in tokenized_sentence:
            bag[index] = 1.0

    return bag
