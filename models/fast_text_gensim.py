import os
import numpy as np
import pickle
from scipy.spatial import distance
from tqdm import tqdm

from settings import DATA_PATH
import gensim
import io


class FastTextGensim():
    def __init__(self, file_name):
        self.embeddings = None
        self.model = None
        if "filtered" in file_name:
            self.load_filtered(file_name)
        else:
            self.load_model(file_name)

        self.embeddings = {}

        for i in range(len(self.model.wv.vocab)):
            word = self.model.wv.index2entity[i]
            vec = self.model.wv.vectors[i]
            self.embeddings[word] = vec

    def load_model(self, file_name):
        self.model = gensim.models.fasttext.load_facebook_model(
            f"{DATA_PATH}/models/{file_name}", encoding="utf-8")

    def load_filtered(self, file_name):
        file_path = os.path.join(DATA_PATH, os.path.join("models", file_name))
        if not os.path.isfile(file_path):
            raise ValueError("File {} does not exists.".format(file_path))
        with open(file_path, 'rb') as file:
            self.model = pickle.load(file)

    def contains(self, word):
        return word in self.embeddings.keys()

    def get(self, word):
        """
        Return embedding for a word.
        :param word:
        :return: numpy.ndarray with word embedding
        """
        ret_val = self.embeddings.get(word, None)
        if ret_val is None:
            raise ValueError("Word {} is not avalaible in model.".format(word))
        return ret_val

    def semantic_relatedness(self, word1, word2, dist_type='cosine'):
        v1 = self.get(word1)
        v2 = self.get(word2)
        dist_fun = self.__get_distance_function(dist_type)
        return dist_fun(v1, v2)

    def synonyms(self, word, top=10, dist_type='cosine'):
        dist_fun = self.__get_distance_function(dist_type)
        distances = {}
        for neighbour in tqdm(self.embeddings.keys()):
            distances[neighbour] = dist_fun(self.get(word), self.get(neighbour))
        sorted_dist = sorted(distances.items(), key=lambda kv: kv[1])
        closest = [sd[0] for sd in sorted_dist[:top]]
        return closest

    def __get_distance_function(self, dist_type='cosine'):
        if dist_type == 'euclidean':
            return distance.euclidean
        elif dist_type == 'manhattan':
            return distance.cityblock
        elif dist_type == "cosine":
            return distance.cosine

    def filter_model_with_polimorf(self, vocabulary):
        new_vectors = []
        new_vocab = {}
        new_index2entity = []

        for i in tqdm(range(len(self.model.wv.vocab))):
            word = self.model.wv.index2entity[i]
            vec = self.model.wv.vectors[i]
            vocab = self.model.wv.vocab[word]
            if word in vocabulary:
                vocab.index = len(new_index2entity)
                new_index2entity.append(word)
                new_vocab[word] = vocab
                new_vectors.append(vec)

        self.model.wv.vocab = new_vocab
        self.model.wv.vectors = new_vectors
        self.model.wv.index2entity = new_index2entity
        self.model.wv.index2word = new_index2entity

    def save(self, file_name):
        file_path = os.path.join(DATA_PATH, os.path.join("models", file_name + ".bin"))
        with open(file_path, 'wb') as file:
            pickle.dump(self.model, file)
