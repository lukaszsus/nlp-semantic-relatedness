import os
import numpy as np
import pickle
from scipy.spatial import distance
from tqdm import tqdm

from settings import DATA_PATH


class IpiPanModel():
    """
    DATA_PATH/models directory should contains saved models.
    Models are available to download on http://dsmodels.nlp.ipipan.waw.pl/.

    Naming convention from IPI-PAN docs:
        Nazwa pliku: corpus-type-stype-dim-arch-alg.txt.gz
        corpus      nazwa korpusu - nkjp, wiki lub nkjp+wiki
        type        typ modelu - model oparty na formach (forms) lub lematach (lemmas)
        stype       podtyp modelu - wszystkie części mowy (all) lub tylko wybrane części mowy (restricted)
        dim         rozmiar wektora - 100 lub 300
        arch        architektura sieci neuronowej - CBOW (cbow) lub Skip-Gram (skipg)
        alg         algorytm uczący - Hierarchical Softmax (hs) lub Negative Sampling (ns)

        Niektóre modele ograniczone zostały tylko do tych słów, które wystąpiły co najmniej 30 lub 50 razy w korpusie.
        Jest to zaznaczone po nazwie algorytmu uczącego alg. it100 w nazwie pliku oznacza, że dany model został
        wytrenowany w stu iteracjach.
    """
    def __init__(self, file_name):
        self.embeddings = None
        self.load_model(file_name)

    def load_model(self, file_name):
        file_extension = os.path.splitext(file_name)[1]
        if file_extension == '.txt':
            self.load_from_raw_file(file_name)
        else:
            self.load_from_binary(file_name)

    def load_from_raw_file(self, file_name):
        self.embeddings = dict()
        print(DATA_PATH)
        file_path = os.path.join(DATA_PATH, os.path.join("models", file_name))
        if not os.path.isfile(file_path):
            raise ValueError("File {} does not exists.".format(file_path))
        f = open(file_path, "r")
        for x in f:
            x = x.replace("\r\n", "\n").replace("\n", "")
            x = x.split(" ")
            self.embeddings[x[0]] = np.array(x[1:], dtype=float)

    def load_from_binary(self, file_name):
        file_path = os.path.join(DATA_PATH, os.path.join("models", file_name))
        if not os.path.isfile(file_path):
            raise ValueError("File {} does not exists.".format(file_path))
        with open(file_path, 'rb') as file:
            self.embeddings = pickle.load(file)

    def contains(self, word):
        return word in self.embeddings.keys()

    def get(self, word):
        """
        Return embedding for a word.
        :param word:
        :return: numpy.ndarray with word embedding
        """
        ret_val = self.embeddings.get(word.encode(encoding='UTF-8'), None)
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
        new_word2vec_dictionary = dict()
        print("Number of words in Word2Vec: {}".format(len(self.embeddings.keys())))
        for word, embedding in tqdm(self.embeddings.items()):
            if word in vocabulary:
                if word == 'łódź':
                    print(word)
                new_word2vec_dictionary[word] = embedding
        return new_word2vec_dictionary

    def save(self, file_name):
        file_path = os.path.join(DATA_PATH, os.path.join("models", file_name + ".bin"))
        with open(file_path, 'wb') as file:
            pickle.dump(self.embeddings, file)