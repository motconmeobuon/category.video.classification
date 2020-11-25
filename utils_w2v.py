import numpy as np
from tqdm import tqdm
from gensim.models import Word2Vec, KeyedVectors

def load_model_w2v(path_model, bin=False):
    return KeyedVectors.load_word2vec_format(path_model, binary=bin)

def word2vec(data, model_w2v, dims=1024):

    def create_word_vector(data, size=dims):
        vector = np.zeros(size, dtype='float32').reshape(1, size)
        count = 0
        for word in data:
            try:
                vector += model_w2v[word].reshape(1, size)
                count += 1
            except KeyError:
                continue

        if count != 0:
            vector /= count
        return vector


    # print(type(data))
    if type(data) == list:
        data_split = []
        for text in data:
            data_split.append(text.split())

        res = []
        for text in tqdm(data_split):
            converted = create_word_vector(text)
            res.append(np.array(converted))
        res = np.concatenate(res)

        return res
    else:
        data_split = data.split()
        converted = create_word_vector(data_split)

        res = np.concatenate(converted)

        return res


def word2vec_tfidf(data, model_w2v, model_tfidf, dims=1024):
    glove_words =  set(model_w2v.vocab)

    dictionary = dict(zip(model_tfidf.get_feature_names(), list(model_tfidf.idf_)))
    tfidf_words = set(model_tfidf.get_feature_names())

    def create_word_vector(data, size=dims):
        vector = np.zeros(size, dtype='float32').reshape(1, size)
        tfidf_weight = 0
        for word in data:
            if (word in glove_words) and (word in tfidf_words):
                vec = model_w2v[word]
                _tfdif = dictionary[word] * (text.count(word)/len(data))
                vector += (vec * _tfdif)
                tfidf_weight += 1

        if (tfidf_weight != 0):
            vector /= tfidf_weight
        return vector


    if type(data) == list:
        data_split = []
        for text in data:
            data_split.append(text.split())

        res = []
        for text in tqdm(data_split):
            converted = create_word_vector(text)
            res.append(np.array(converted))
        res = np.concatenate(res)

        return res
    else:
        data_split = data.split()
        converted = create_word_vector(data_split)

        res = np.concatenate(converted)

        return res