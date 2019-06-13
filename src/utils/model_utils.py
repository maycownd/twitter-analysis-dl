from gensim.models import KeyedVectors
import numpy as np


def load_embedding(path='../models/glove/glove_s50.txt'):
    print("Loading Word2Vec Model...")
    return KeyedVectors.load_word2vec_format(path)


def get_vector(word, model):
    if word in model:
        return model[word]
    else:
        return None


def is_in_model(word, model):
    return word in model


def create_embedding_matrix(embedding_index, vocab_size, tokenizer, dimensions=50):
    embedding_matrix = np.zeros((vocab_size, dimensions))
    if embedding_index is None:
        embedding_index = load_embedding()
    for word, i in tokenizer.word_index.items():
        embedding_vector = get_vector(word, embedding_index)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix
