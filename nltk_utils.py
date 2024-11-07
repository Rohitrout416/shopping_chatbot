import nltk
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenize_sentence, all_words):
    pass

words = ['organize', 'organizes', 'organs']
stemmed_words = [stem(word) for word in words]
print(stemmed_words)