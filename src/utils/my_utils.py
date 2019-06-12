from gensim.models import KeyedVectors


def load_embedding(path):
    print("Loading Word2Vec Model...")
    return KeyedVectors.load(path)